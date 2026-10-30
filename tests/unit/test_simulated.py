import numpy as np
import pytest

from easyebl.core import AffineTransform, OutputChunk
from easyebl.hal import DAQManager, DoseMap, FunctionSample, SimParams, SimulatedBackend
from easyebl.hal.simulated import ImageSample, render_cross

RATE = 100e3


def run(be, chunks, ai=("video",)):
    m = DAQManager(be).open()
    r = m.get_handle("t").run_stream(RATE, chunks, ai_channels=list(ai))
    return r


def ramp_sample():
    return FunctionSample(lambda x, y: x / 50.0)  # video ∝ x position


def test_step_response_matches_first_order_lag():
    tau = 50e-6
    be = SimulatedBackend(params=SimParams(lag_tau_s=tau, noise_sigma=0, video_delay_s=0, video_gain_v=1.0,
                                           ai_latency_samples=0), sample=ramp_sample())
    n = 200
    x = np.full(n, 5.0)  # step from 0 V to 5 V (25 µm)
    r = run(be, [OutputChunk(x, np.zeros(n), np.zeros(n, bool))])
    t = (np.arange(n) + 1) / RATE
    expected = 25.0 / 50.0 * (1 - np.exp(-t / tau))
    np.testing.assert_allclose(r.ai[0], expected, atol=1e-9)


def test_chunk_boundaries_do_not_matter():
    p = SimParams(lag_tau_s=30e-6, noise_sigma=0, video_delay_s=40e-6, blank_delay_s=30e-6)
    x = np.sin(np.linspace(0, 20, 600)) * 4
    b = np.sin(np.linspace(0, 33, 600)) > 0
    one = run(SimulatedBackend(params=p, sample=ramp_sample()), [OutputChunk(x, x, b)])
    parts = [OutputChunk(x[s:e], x[s:e], b[s:e]) for s, e in [(0, 7), (7, 250), (250, 251), (251, 600)]]
    be2 = SimulatedBackend(params=p, sample=ramp_sample())
    many = run(be2, parts)
    np.testing.assert_allclose(one.ai, many.ai, atol=1e-12)


def test_video_delay_and_blank_signal():
    p = SimParams(lag_tau_s=0, noise_sigma=0, video_delay_s=20e-6, video_gain_v=1.0)  # 2 + 1 (AI latency)
    be = SimulatedBackend(params=p, sample=FunctionSample(lambda x, y: np.ones_like(x)))
    blank = np.zeros(20, bool)
    blank[10:] = True
    r = run(be, [OutputChunk(np.zeros(20), np.zeros(20), blank)])
    np.testing.assert_allclose(r.ai[0][:3], 0)  # delay line starts empty
    np.testing.assert_allclose(r.ai[0][3:13], 1)
    np.testing.assert_allclose(r.ai[0][13:], 0)  # blanked, delayed by 3


def test_dose_deposition_and_blank_delay():
    p = SimParams(lag_tau_s=0, noise_sigma=0, beam_current_a=1e-9, blank_delay_s=20e-6)  # 2 samples late
    be = SimulatedBackend(params=p)
    n = 100
    blank = np.ones(n, bool)
    blank[10:60] = False  # 50 samples on
    run(be, [OutputChunk(np.zeros(n), np.zeros(n), blank)], ai=())
    assert be.dose.total_charge() == pytest.approx(50 * 1e-9 / RATE)


def test_blank_leakage():
    p = SimParams(lag_tau_s=0, beam_current_a=1e-9, blank_leakage=0.01)
    be = SimulatedBackend(params=p)
    run(be, [OutputChunk(np.zeros(1000), np.zeros(1000), np.ones(1000, bool))], ai=())
    assert be.dose.total_charge() == pytest.approx(0.01 * 1000 * 1e-9 / RATE)


def test_true_distortion_moves_beam():
    p = SimParams(lag_tau_s=0, distortion=AffineTransform.translation(2.0, 0))
    be = SimulatedBackend(params=p)
    np.testing.assert_allclose(be.volts_to_true_um(np.array([[0.0, 0.0]])), [[2.0, 0.0]])
    np.testing.assert_allclose(be.true_um_to_volts(np.array([[2.0, 0.0]])), [[0.0, 0.0]], atol=1e-12)


def test_static_unblanked_beam_deposits_dose():
    be = SimulatedBackend(params=SimParams(beam_current_a=1e-9))
    be.open()
    be.write_static(0, 0, False)
    import time

    time.sleep(0.02)
    be.write_static(0, 0, True)
    assert be.dose.total_charge() > 1e-9 * 0.015


def test_dose_map_geometry():
    d = DoseMap(extent_um=10, pixel_um=1)
    d.deposit(np.array([0.5, -4.5, 100]), np.array([0.5, 4.5, 0]), 1e-15)
    q = d.charge_c
    assert q[4, 5] == pytest.approx(1e-15) and q[0, 0] == pytest.approx(1e-15)
    assert d.total_charge() == pytest.approx(2e-15)  # out-of-map deposit ignored
    xs, ys = d.coords()
    assert xs[5] == pytest.approx(0.5) and ys[4] == pytest.approx(0.5)
    assert d.dose_uC_cm2()[4, 5] == pytest.approx(1e-15 / (1e-4) ** 2 * 1e6)


def test_image_sample_orientation():
    img = np.zeros((10, 10))
    img[0, 9] = 1.0  # top-right pixel
    s = ImageSample(img, (0, 10, 0, 10))
    assert s.intensity(np.array([9.5]), np.array([9.5]))[0] == pytest.approx(1.0)
    assert s.intensity(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(0.0)


def test_render_cross_subpixel_centroid():
    ax = (np.arange(200) + 0.5) * 0.1 - 10
    xx, yy = np.meshgrid(ax, ax[::-1])
    c = render_cross(xx, yy, 0.123, -0.037, 6, 1, 0.1)
    cx = (c * xx).sum() / c.sum()
    cy = (c * yy).sum() / c.sum()
    assert cx == pytest.approx(0.123, abs=2e-3) and cy == pytest.approx(-0.037, abs=2e-3)
    assert c.max() <= 1.0
