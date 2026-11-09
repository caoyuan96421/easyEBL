"""Raster imaging end-to-end on the simulated SEM."""
import threading

import numpy as np
import pytest

from easyebl.core import AffineTransform
from easyebl.hal import SimParams
from easyebl.imaging import RasterConfig, RasterImager

from ..conftest import ground_truth

pytestmark = pytest.mark.system

CFG = RasterConfig(nx=160, ny=160, width_um=40, center_um=(22, 20), dwell_s=5e-6, rate=400e3, preline_s=25e-6)


def xshift_px(a, b):
    """Sub-pixel x-shift of image a relative to b (row-wise cross-correlation)."""
    n = a.shape[1]
    A = np.fft.fft(a - a.mean(), axis=1)
    B = np.fft.fft(b - b.mean(), axis=1)
    c = np.fft.ifft(A * np.conj(B), axis=1).real.sum(0)
    k = int(np.argmax(c))
    y0, y1, y2 = c[(k - 1) % n], c[k], c[(k + 1) % n]
    k = k if k < n // 2 else k - n
    return k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_image_matches_specimen(make_manager, quiet_params):
    m = make_manager(quiet_params)
    img = RasterImager(m.get_handle("img"), __import__("easyebl").core.FieldCalibration(100, 10)).acquire(CFG)
    gt = ground_truth(m.backend, img)
    assert not np.isnan(img.image()).any()
    np.testing.assert_allclose(img.image(), gt, atol=1e-9)  # ideal system: exact
    assert img.rows_completed == CFG.ny and not img.aborted


def test_lag_shift_and_compensation(make_manager, calibration):
    p = SimParams(lag_tau_s=0.0, video_delay_s=10e-6, noise_sigma=0.01)  # pure 4-sample delay
    m = make_manager(p)
    im = RasterImager(m.get_handle("img"), calibration)
    raw = im.acquire(CFG)
    gt = ground_truth(m.backend, raw)
    # video delay of 10 µs at 5 µs/pixel -> image appears shifted by +2 px (features later in the line)
    assert xshift_px(raw.image(), gt) == pytest.approx(2.0, abs=0.15)
    comp = im.acquire(CFG.with_(lag_comp_s=10e-6))
    assert xshift_px(comp.image(), gt) == pytest.approx(0.0, abs=0.15)
    assert corr(comp.image(), gt) > 0.98


def test_bidirectional_rows_agree_after_compensation(make_manager, calibration):
    p = SimParams(lag_tau_s=0.0, video_delay_s=10e-6, noise_sigma=0.01)
    m = make_manager(p)
    im = RasterImager(m.get_handle("img"), calibration)
    bi = im.acquire(CFG.with_(bidirectional=True))
    np.testing.assert_array_equal(bi.line_directions[:4], [1, -1, 1, -1])
    fwd, back = bi.image()[0::2], bi.image()[1::2]
    assert xshift_px(fwd, back) == pytest.approx(4.0, abs=0.3)  # ±2 px
    bi_c = im.acquire(CFG.with_(bidirectional=True, lag_comp_s=10e-6))
    assert xshift_px(bi_c.image()[0::2], bi_c.image()[1::2]) == pytest.approx(0.0, abs=0.2)


def test_frame_averaging_reduces_noise(make_manager, calibration):
    p = SimParams(lag_tau_s=0, video_delay_s=0, noise_sigma=0.3)
    m = make_manager(p)
    im = RasterImager(m.get_handle("img"), calibration)
    small = CFG.with_(nx=64, ny=64)
    one = im.acquire(small)
    four = im.acquire(small.with_(frames=4))
    gt = ground_truth(m.backend, one)
    e1 = np.std(one.image() - gt)
    e4 = np.std(four.image() - gt)
    assert e4 == pytest.approx(e1 / 2, rel=0.2)


def test_rotated_and_distorted_scan(make_manager, calibration):
    p = SimParams(lag_tau_s=0, video_delay_s=0, noise_sigma=0,
                  distortion=AffineTransform.from_params(0.5, -0.3, 0.002, 1.003, 1.003))
    m = make_manager(p)
    im = RasterImager(m.get_handle("img"), calibration.with_correction(p.distortion.inverse()))
    img = im.acquire(CFG.with_(rotation_rad=0.3, nx=96, ny=96))
    gt = ground_truth(m.backend, img)
    np.testing.assert_allclose(img.image(), gt, atol=1e-6)  # correction cancels the true distortion


def test_live_rows_and_abort(make_manager, calibration):
    p = SimParams(realtime=True)
    m = make_manager(p)
    h = m.get_handle("img")
    im = RasterImager(h, calibration)
    rows = []

    def on_row(r, imgs):
        rows.append(r)
        if len(rows) == 5:
            threading.Thread(target=h.abort).start()

    img = im.acquire(CFG.with_(nx=64, ny=256, height_um=40.0), on_row=on_row, lines_per_chunk=1)
    assert img.aborted and 5 <= img.rows_completed < 256
    assert np.isnan(img.image()[-1]).all()
    assert m.beam_state.blank and not m.is_busy


def test_dark_scan_and_sync_markers(make_manager, calibration):
    from easyebl.hal import DAQConfig

    m = make_manager(SimParams(noise_sigma=0), config=DAQConfig(timed_lines={"line": "port0/line1"}), record=True)
    im = RasterImager(m.get_handle("img"), calibration)
    img = im.acquire(CFG.with_(nx=32, ny=8, blank_during_pixels=True, line_sync_marker="line"))
    assert np.allclose(img.image(), 0)  # beam blanked -> no signal
    assert m.backend.dose.total_charge() == 0
    rec = m.backend.recorded
    assert sum(c.markers["line"].sum() for c in rec) == 8 * 32 * img.config.samples_per_pixel


# ---------------------------------------------------------------- live streaming
def test_live_frames_match_single_scans(make_manager, quiet_params):
    """Back-to-back frames in one stream are identical to separate acquisitions."""
    from easyebl.core import FieldCalibration

    cal = FieldCalibration(100, 10)
    m = make_manager(quiet_params)
    imager = RasterImager(m.get_handle("img"), cal)
    cfg = CFG.with_(nx=64, ny=48, lag_comp_s=12e-6, bidirectional=True)
    ref = imager.acquire(cfg).image()
    got = []
    n = imager.live(cfg, lambda k, img: got.append((k, img)), max_frames=4)
    assert n == 4 and [k for k, _ in got] == [0, 1, 2, 3]
    for _, img in got:
        assert img.rows_completed == cfg.ny and not np.isnan(img.image()).any()
        np.testing.assert_allclose(img.image(), ref, atol=1e-9)
    # without lag compensation the ideal system reproduces the specimen exactly
    got0 = []
    imager.live(cfg.with_(lag_comp_s=0.0), lambda k, img: got0.append(img), max_frames=2)
    for img in got0:
        np.testing.assert_allclose(img.image(), ground_truth(m.backend, img), atol=1e-9)


def test_live_with_lag_is_steady(make_manager):
    """With deflection lag, later frames equal each other (steady state)."""
    from easyebl.core import FieldCalibration

    m = make_manager(SimParams(lag_tau_s=8e-6, noise_sigma=0, video_delay_s=0))
    imager = RasterImager(m.get_handle("img"), FieldCalibration(100, 10))
    cfg = CFG.with_(nx=48, ny=32)
    got = []
    imager.live(cfg, lambda k, img: got.append(img.image()), max_frames=3)
    np.testing.assert_allclose(got[1], got[2], atol=1e-9)


def test_live_stop_event_and_abort(make_manager, quiet_params):
    from easyebl.core import FieldCalibration

    m = make_manager(quiet_params)
    h = m.get_handle("img")
    imager = RasterImager(h, FieldCalibration(100, 10))
    stop = threading.Event()
    got = []

    def on_frame(k, img):
        got.append(k)
        if k == 2:
            stop.set()

    n = imager.live(CFG.with_(nx=32, ny=16), on_frame, stop=stop)
    assert n >= 3 and got == list(range(n))
    assert m.beam_state.blank and not m.is_busy
