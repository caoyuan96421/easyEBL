"""Dummy-image test on the simulator: phantom on aux0 looped back to the video input.

The simulator models exactly the hardware wiring AO2 → AI0 (one-sample AI
latency, no SEM physics), so the engine's images must equal the phantom
chart at the pixel centres – for any field of view, beam shift, rotation and
scan speed – and the noise must average down as 1/√(samples per pixel).
"""
import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend
from easyebl.imaging.engine import EngineOptions, ImagingEngine

pytestmark = pytest.mark.system

CAL = FieldCalibration(100.0, 5.0)


@pytest.fixture
def phantom_engine():
    made = []

    def _make(noise_v=0.0, rate=1e6):
        cfg = DAQConfig.usb6363()
        cfg.phantom_output = "aux0"
        cfg.phantom_noise_v = noise_v
        be = SimulatedBackend(cfg, SimParams(noise_sigma=0.0, v_full_scale=5.0))
        mgr = DAQManager(be).open()
        eng = ImagingEngine(mgr, CAL, options=EngineOptions(rate=rate, latency_s=0.05))
        made.append((eng, mgr))
        return eng, be.phantom_source()

    yield _make
    for eng, mgr in made:
        eng.close()
        mgr.close()


def frame(eng, n=2):
    f = None
    for _ in range(n):  # skip the first (it may straddle a settings change)
        f = eng.wait_frame(None if f is None else f.seq, complete=True, timeout=20)
    return f


def test_full_field_matches_chart(phantom_engine):
    eng, ph = phantom_engine()
    eng.update(nx=256, ny=256, width_um=90.0, dwell_s=2e-6)
    eng.run()
    f = frame(eng)
    np.testing.assert_allclose(f.image, ph.expected_image(f.config, CAL), atol=1e-5)


@pytest.mark.parametrize("width", [60.0, 30.0, 10.0])
def test_field_of_view(phantom_engine, width):
    eng, ph = phantom_engine()
    eng.update(nx=200, ny=150, width_um=width, dwell_s=2e-6)
    eng.run()
    f = frame(eng)
    np.testing.assert_allclose(f.image, ph.expected_image(f.config, CAL), atol=1e-5)
    assert f.pixel_size_um[0] == pytest.approx(width / 200)


def test_beam_shift_moves_the_image(phantom_engine):
    eng, ph = phantom_engine()
    eng.update(nx=200, ny=200, width_um=40.0, dwell_s=2e-6)  # 0.2 µm per pixel
    eng.run()
    a = frame(eng)
    eng.shift_beam(4.0, -2.0)  # = 20 px right, 10 px down
    b = frame(eng, 3)
    assert b.config.center_um == (4.0, -2.0)
    np.testing.assert_allclose(b.image, ph.expected_image(b.config, CAL), atol=1e-5)
    # the same specimen detail moved by exactly (-20, -10) px in the image
    np.testing.assert_allclose(b.image[:-10, :-20][10:], a.image[10:, 20:][10:], atol=1e-5)


def test_rotation_and_bidirectional(phantom_engine):
    eng, ph = phantom_engine()
    eng.update(nx=160, ny=120, width_um=50.0, dwell_s=2e-6, rotation_rad=0.3, bidirectional=True)
    eng.run()
    f = frame(eng)
    np.testing.assert_allclose(f.image, ph.expected_image(f.config, CAL), atol=1e-5)


def test_noise_averages_with_dwell(phantom_engine):
    """With white noise on the phantom, pixel noise = σ / √(samples per pixel) at every speed."""
    sigma = 0.2
    eng, ph = phantom_engine(noise_v=sigma)
    eng.run()
    results = []
    for dwell in (1e-6, 4e-6, 16e-6):  # 1, 4, 16 samples per pixel at 1 MS/s
        eng.update(nx=128, ny=96, width_um=60.0, dwell_s=dwell)
        f = frame(eng, 3)
        resid = f.image - ph.expected_image(f.config, CAL)
        spp = f.config.samples_per_pixel
        results.append((spp, float(np.std(resid))))
        assert np.std(resid) == pytest.approx(sigma / np.sqrt(spp), rel=0.1), results
    assert results[0][1] > 3.5 * results[2][1]


def test_phantom_on_dummy_output_does_nothing():
    cfg = DAQConfig()  # two-AO board: aux0 is a dummy
    cfg.phantom_output = "aux0"
    cfg.ai_terminal = "DIFF"  # the faulty setup: DIFF with AI0− unconnected
    be = SimulatedBackend(cfg, SimParams())
    assert be.phantom_source() is None and cfg.stream_ao_channels == 2


def test_engine_switches_phantom_on_and_off():
    cfg = DAQConfig.usb6363()  # phantom off at start
    be = SimulatedBackend(cfg, SimParams(noise_sigma=0.0, v_full_scale=5.0))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, CAL, options=EngineOptions(latency_s=0.05))
    try:
        assert eng.rate == 2e6
        eng.update(nx=128, ny=96, width_um=60.0, dwell_s=4e-6)
        eng.run()
        frame(eng)
        ph = eng.set_phantom("aux0")
        assert ph["physical"] == "ao2" and eng.rate == pytest.approx(100e6 / 65)
        f = frame(eng, 3)  # resumed live by itself, now imaging the chart
        assert eng.state.value == "live" and f.config.rate == pytest.approx(100e6 / 65)
        np.testing.assert_allclose(f.image, be.phantom_source().expected_image(f.config, CAL), atol=1e-5)
        eng.set_phantom("aux0", noise_v=0.1)  # noise only: no restart
        streams = eng.status().streams
        frame(eng, 2)
        assert eng.status().streams == streams
        eng.set_phantom(None)
        assert eng.rate == 2e6 and be.phantom_source() is None
        assert frame(eng, 2) is not None and eng.state.value == "live"
    finally:
        eng.close()
        mgr.close()


def test_floating_ai_minus_is_detected_and_rse_fixes_it():
    """Reproduces the bench fault: phantom loop on AI0+, AI0− unconnected, input in DIFF mode."""
    cfg = DAQConfig.usb6363()
    cfg.phantom_output = "aux0"
    cfg.ai_terminal = "DIFF"  # the faulty setup: DIFF with AI0− unconnected
    be = SimulatedBackend(cfg, SimParams(noise_sigma=0.0, v_full_scale=5.0, ai_minus_floating=True))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, CAL, options=EngineOptions(rate=1e6, latency_s=0.05))
    sub = eng.subscribe()
    try:
        eng.update(nx=128, ny=96, width_um=60.0, dwell_s=2e-6)
        eng.run()
        f = frame(eng, 4)
        assert eng.status().saturation > 0.5  # most pixels pinned at the −10 V rail
        warn = [e for e in sub.drain() if e.kind == "warning" and "saturated" in e.message]
        assert len(warn) == 1 and "AI−" in warn[0].message  # one warning, with the wiring hint
        assert "VIDEO SATURATED" in eng.status().summary()
        eng.set_ai_terminal("RSE")  # AI+ against AI GND: the floating AI− no longer matters
        assert eng.handle.ai_terminal == "RSE"
        f = frame(eng, 3)
        assert eng.state.value == "live"
        np.testing.assert_allclose(f.image, be.phantom_source().expected_image(f.config, CAL), atol=1e-5)
        assert eng.status().saturation == 0.0
    finally:
        eng.close()
        mgr.close()
