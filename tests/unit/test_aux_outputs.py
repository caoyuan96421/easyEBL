"""The HAL's aux0/aux1 analog outputs and AI input mode."""
import threading

import numpy as np

import pytest

from easyebl.core import OutputChunk
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


# ------------------------------------------------------------------ config / rates
def test_aux_config_and_rate_limits():
    d = DAQConfig()
    d.validate()
    assert d.aux_outputs == {"aux0": None, "aux1": None}  # 2-AO board: dummies
    c = DAQConfig.usb6363()
    assert c.aux_physical("aux0") == "ao2" and c.aux_physical("aux1") == "ao3"
    assert c.stream_ao_channels == 2 and c.max_scan_rate() == 2e6
    with pytest.raises(ValueError):
        DAQConfig(aux_outputs={"aux2": "ao2"}).validate()
    with pytest.raises(ValueError):
        DAQConfig(aux_outputs={"aux0": "ao0"}).validate()  # collides with X
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
        h.set_aux_output("aux1", 11.0)  # beyond the ±10 V AO range of the 6363 preset
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


def test_aux_refused_while_streaming():
    m = _mgr()
    h = m.get_handle("x")
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


def test_static_ai_read_on_the_simulator():
    """Park the beam anywhere and read the video input – no stream needed."""
    m = DAQManager(SimulatedBackend(DAQConfig.usb6363(), SimParams(noise_sigma=0.01))).open()
    h = m.get_handle("x")
    h.park(0.0, 0.0, blank=True)
    dark = h.read_ai_mean("video")["video"]
    vals = []
    for x in np.linspace(-9, 9, 7):
        h.park(x, 0.0, blank=False)
        vals.append(h.read_ai_mean("video", samples=200)["video"])
    assert np.ptp(vals) > 0.05 and min(vals) >= dark - 0.05  # the specimen shows; blanked = no signal
    data = h.read_ai(None, samples=64, rate=10e3)
    assert data.shape == (len(m.config.ai_channels), 64)
    with pytest.raises(ValueError):
        h.read_ai("nope")
    with pytest.raises(ValueError):
        h.read_ai("video", rate=1e9)
    m.close()


def test_static_ai_read_refused_while_streaming():
    m = _mgr()
    started, release = threading.Event(), threading.Event()

    def chunks():
        started.set()
        while not release.is_set():
            yield OutputChunk.constant(1000, 0, 0, True)

    t = threading.Thread(target=lambda: m.get_handle("s").run_stream(500e3, chunks()))
    t.start()
    assert started.wait(2)
    with pytest.raises(DeviceBusyError):
        m.get_handle("x").read_ai("video")
    release.set()
    t.join(5)
    m.close()
