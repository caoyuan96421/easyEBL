"""Phantom specimen (dummy-image test) and the HAL's aux0/aux1 analog outputs."""
import threading

import numpy as np
import pytest

from easyebl.core import FieldCalibration, OutputChunk
from easyebl.hal import (
    DAQConfig,
    DAQManager,
    DeviceBusyError,
    HALError,
    ResourceConflictError,
    SimParams,
    SimulatedBackend,
    VoltageRangeError,
)
from easyebl.hal.phantom import PhantomSpecimen, make_test_chart
from easyebl.imaging import RasterConfig


@pytest.fixture(scope="module")
def chart():
    return make_test_chart(512)


def test_chart_is_asymmetric_and_in_range(chart):
    assert chart.shape == (512, 512) and chart.dtype == np.float32
    assert 0.0 <= chart.min() and chart.max() <= 1.0
    for flipped in (chart[::-1], chart[:, ::-1], chart.T):
        assert np.abs(flipped - chart).mean() > 0.02  # mirrored/transposed axes are detectable


def test_intensity_mapping(chart):
    ph = PhantomSpecimen(chart, extent_v=5.0, low_v=0.0, high_v=4.0, blank_v=-0.5)
    # pixel (row 0, col 0) is the top-left corner: x = -5 V, y = +5 V
    np.testing.assert_allclose(ph.intensity_v(np.array([-4.999]), np.array([4.999])), 4.0 * chart[0, 0], rtol=1e-6)
    np.testing.assert_allclose(ph.intensity_v(np.array([4.999]), np.array([-4.999])), 4.0 * chart[-1, -1], rtol=1e-6)
    out = ph.output(np.array([0.0, 0.0, 9.0]), np.array([0.0, 0.0, 0.0]), np.array([False, True, False]))
    assert out[1] == -0.5  # blanked: no beam, no signal
    assert 0.0 <= out[2] <= 4.0  # outside the chart: background
    noisy = PhantomSpecimen(chart, noise_v=0.2, seed=1)
    x = np.zeros(200_000)
    assert np.std(noisy.output(x, x, np.zeros_like(x, bool))) == pytest.approx(0.2, rel=0.02)


def test_expected_image_follows_geometry(chart):
    ph = PhantomSpecimen(chart, extent_v=10.0)
    cal = FieldCalibration(100.0, 10.0)
    full = ph.expected_image(RasterConfig(nx=200, ny=200, width_um=100.0), cal)
    half = ph.expected_image(RasterConfig(nx=100, ny=100, width_um=50.0), cal)
    np.testing.assert_allclose(half, full[50:150, 50:150])  # zoom = centre crop at the same pixel size
    shifted = ph.expected_image(RasterConfig(nx=100, ny=100, width_um=50.0, center_um=(10.0, 0.0)), cal)
    np.testing.assert_allclose(shifted, full[50:150, 70:170])  # beam shift +10 µm = 20 px (0.5 µm/px)


# ------------------------------------------------------------------ config / rates
def test_aux_config_and_rate_limits():
    d = DAQConfig()
    d.validate()
    assert d.aux_outputs == {"aux0": None, "aux1": None}  # 2-AO board: dummies
    c = DAQConfig.usb6363()
    assert c.aux_physical("aux0") == "ao2" and c.aux_physical("aux1") == "ao3"
    assert c.stream_ao_channels == 2 and c.max_scan_rate() == 2e6
    c.phantom_output = "aux0"
    assert c.stream_ao_channels == 3 and c.max_scan_rate() == pytest.approx(100e6 / 65)
    d.phantom_output = "aux0"
    assert d.phantom_physical is None and d.stream_ao_channels == 2  # phantom on a dummy: no effect
    with pytest.raises(ValueError):
        DAQConfig(aux_outputs={"aux2": "ao2"}).validate()
    with pytest.raises(ValueError):
        DAQConfig(aux_outputs={"aux0": "ao0"}).validate()  # collides with X
    with pytest.raises(ValueError):
        DAQConfig(phantom_output="ao2").validate()
    with pytest.raises(KeyError):
        c.aux_physical("ao2")


# ------------------------------------------------------------------ HAL aux outputs
def _mgr(cfg=None):
    return DAQManager(SimulatedBackend(cfg or DAQConfig.usb6363(), SimParams())).open()


def test_aux_outputs_real_and_dummy():
    m = _mgr()
    h = m.get_handle("x")
    h.set_aux_output("aux1", 1.25)
    assert m.backend.aux_values["aux1"] == 1.25 and h.aux_outputs["aux1"] == ("ao3", 1.25)
    with pytest.raises(VoltageRangeError):
        h.set_aux_output("aux1", 6.0)  # beyond the ±5 V AO range of the 6363 preset
    with pytest.raises(HALError):
        h.set_aux_output("aux7", 1.0)
    other = m.get_handle("y")
    with pytest.raises(ResourceConflictError):
        other.set_aux_output("aux1", 0.5)
    h.release("aux1")
    assert m.backend.aux_values["aux1"] == 0.0  # released → 0 V
    other.set_aux_output("aux1", 0.5)
    m.close()
    assert m.backend.aux_values["aux1"] == 0.0  # closing → 0 V
    dm = _mgr(DAQConfig())  # two-AO board: aux outputs exist but are dummies
    dh = dm.get_handle("x")
    dh.set_aux_output("aux0", 2.0)
    assert dh.aux_outputs["aux0"] == (None, 2.0)
    dm.close()


def test_aux_refused_while_streaming_and_on_phantom_output():
    cfg = DAQConfig.usb6363()
    cfg.phantom_output = "aux0"
    m = _mgr(cfg)
    h = m.get_handle("x")
    with pytest.raises(ResourceConflictError):
        h.set_aux_output("aux0", 1.0)  # the phantom owns it
    started, release = threading.Event(), threading.Event()

    def chunks():
        started.set()
        while not release.is_set():
            yield OutputChunk.constant(1000, 0, 0, True)

    t = threading.Thread(target=lambda: m.get_handle("s").run_stream(500e3, chunks()))
    t.start()
    assert started.wait(2)
    with pytest.raises(DeviceBusyError):
        h.set_aux_output("aux1", 1.0)
    release.set()
    t.join(5)
    h.set_aux_output("aux1", 1.0)
    m.close()


def test_rate_validation_counts_the_phantom_channel():
    cfg = DAQConfig.usb6363()
    cfg.phantom_output = "aux0"
    m = _mgr(cfg)
    with pytest.raises(HALError, match="3 AO channels"):
        m.get_handle("x").run_stream(2e6, [OutputChunk.constant(100, 0, 0, True)])
    m.get_handle("x").run_stream(100e6 / 65, [OutputChunk.constant(100, 0, 0, True)])
    m.close()


def test_set_phantom_at_runtime():
    m = _mgr()  # USB-6363 preset: aux0 = ao2, aux1 = ao3, phantom off
    h = m.get_handle("x")
    assert {k: h.phantom[k] for k in ("output", "physical", "noise_v")} == {"output": None, "physical": None, "noise_v": 0.0}
    assert m.config.max_scan_rate() == 2e6
    h.set_phantom("aux0", 0.1)
    assert {k: h.phantom[k] for k in ("output", "physical", "noise_v")} == {"output": "aux0", "physical": "ao2", "noise_v": 0.1}
    assert m.backend.phantom_source() is not None and m.config.max_scan_rate() == pytest.approx(100e6 / 65)
    with pytest.raises(ResourceConflictError):
        h.set_aux_output("aux0", 1.0)  # now owned by the phantom
    other = m.get_handle("y")
    other.set_aux_output("aux1", 0.7)
    with pytest.raises(ResourceConflictError):
        h.set_phantom("aux1")  # claimed by another handle
    started, release = threading.Event(), threading.Event()

    def chunks():
        started.set()
        while not release.is_set():
            yield OutputChunk.constant(1000, 0, 0, True)

    t = threading.Thread(target=lambda: m.get_handle("s").run_stream(100e6 / 65, chunks()))
    t.start()
    assert started.wait(2)
    with pytest.raises(DeviceBusyError):
        h.set_phantom(None)  # routing cannot change mid-stream …
    h.set_phantom("aux0", 0.3)  # … but the noise can
    assert m.backend.phantom.noise_v == 0.3
    release.set()
    t.join(5)
    h.set_phantom(None)
    assert m.backend.phantom_source() is None and m.config.max_scan_rate() == 2e6
    assert m.backend.aux_values["aux0"] == 0.0  # the old phantom output is back at its static value
    with pytest.raises(HALError):
        h.set_phantom("ao2")
    m.close()


def test_ai_terminal_setting():
    from easyebl.hal import AI_TERMINALS

    assert "RSE" in AI_TERMINALS
    with pytest.raises(ValueError):
        DAQConfig(ai_terminal="SE").validate()
    m = _mgr()
    h = m.get_handle("x")
    started, release = threading.Event(), threading.Event()

    def chunks():
        started.set()
        while not release.is_set():
            yield OutputChunk.constant(1000, 0, 0, True)

    assert m.config.ai_terminal == "RSE"  # default: single-ended, no AI0− needed
    t = threading.Thread(target=lambda: m.get_handle("s").run_stream(500e3, chunks()), daemon=True)
    t.start()
    try:
        assert started.wait(2)
        with pytest.raises(DeviceBusyError):
            h.set_ai_terminal("DIFF")  # wiring setting: never under a running stream
    finally:
        release.set()
        t.join(5)
    h.set_ai_terminal("DIFF")
    assert m.config.ai_terminal == "DIFF"
    m.close()
