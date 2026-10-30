"""Automatic marker alignment end-to-end on the simulated SEM."""
import numpy as np
import pytest

from easyebl.alignment import AlignmentError, MarkerAligner, MarkSpec
from easyebl.core import AffineTransform
from easyebl.hal import SimParams, make_demo_sample
from easyebl.imaging import RasterConfig, RasterImager

pytestmark = pytest.mark.system

NOMINAL = np.array([(-35.0, -35.0), (35.0, -35.0), (35.0, 35.0), (-35.0, 35.0)])
SCAN = RasterConfig(nx=128, ny=128, dwell_s=4e-6, rate=400e3, preline_s=30e-6, blank_preline=True)
TRUE = AffineTransform.from_params(tx=1.3, ty=-0.7, rotation_rad=0.004, sx=1.0015, sy=1.0015)


def setup(make_manager, calibration, truth=TRUE, marks=NOMINAL, **sim):
    kw = dict(noise_sigma=0.05, lag_tau_s=6e-6, video_delay_s=2e-6)
    kw.update(sim)
    params = SimParams(**kw)
    m = make_manager(params, sample=make_demo_sample(marks=truth.apply(marks)))
    return RasterImager(m.get_handle("align"), calibration)


def placement_error_nm(result, truth, pts=NOMINAL):
    e = result.transform.apply(pts) - truth.apply(pts)
    return float(np.max(np.hypot(*e.T)) * 1e3)


def test_recovers_similarity_transform(make_manager, calibration):
    im = setup(make_manager, calibration)
    res = MarkerAligner(im, SCAN, window_um=20).align(NOMINAL, "similarity")
    assert all(m.ok for m in res.measurements)
    assert placement_error_nm(res, TRUE) < 40  # pixel size is 156 nm
    assert res.rms_um < 0.04
    # reported lag shift ≈ (effective deflection + video delay) × scan speed, same sign for all marks
    lag = np.array([m.lag_shift_um[0] for m in res.measurements])
    assert np.all(lag > 0.05) and np.ptp(lag) < 0.03


def test_unidirectional_scan_is_biased_by_lag(make_manager, calibration):
    im = setup(make_manager, calibration)
    bi = MarkerAligner(im, SCAN, window_um=20).align(NOMINAL, "translation")
    uni = MarkerAligner(im, SCAN, window_um=20, bidirectional_average=False).align(NOMINAL, "translation")
    err_bi = abs(bi.transform.offset[0] - np.mean(TRUE.apply(NOMINAL)[:, 0] - NOMINAL[:, 0]))
    err_uni = abs(uni.transform.offset[0] - np.mean(TRUE.apply(NOMINAL)[:, 0] - NOMINAL[:, 0]))
    assert err_uni > 0.08 and err_bi < 0.03 and err_uni > 3 * err_bi


def test_affine_with_five_marks(make_manager, calibration):
    marks = np.vstack([NOMINAL, [[0.0, 30.0]]])
    truth = AffineTransform.from_params(-0.8, 0.5, -0.003, 1.002, 0.999, 0.001)
    im = setup(make_manager, calibration, truth=truth, marks=marks)
    res = MarkerAligner(im, SCAN, window_um=20).align(marks, "affine")
    assert placement_error_nm(res, truth, marks) < 50


def test_missing_mark_is_rejected(make_manager, calibration):
    present = NOMINAL[:3]
    im = setup(make_manager, calibration, marks=present)  # mark 4 absent from the specimen
    al = MarkerAligner(im, SCAN, window_um=20, min_score=0.6)
    res = al.align([MarkSpec(tuple(p), f"M{i}") for i, p in enumerate(NOMINAL)], "similarity")
    assert [m.ok for m in res.measurements] == [True, True, True, False]
    assert "score" in res.measurements[3].message or "disagree" in res.measurements[3].message
    assert placement_error_nm(res, TRUE, present) < 50
    with pytest.raises(AlignmentError):
        al.align([MarkSpec(tuple(NOMINAL[3]))], "translation")


def test_locate_in_existing_image(make_manager, calibration):
    im = setup(make_manager, calibration, lag_tau_s=0.0, video_delay_s=0.0)
    al = MarkerAligner(im, SCAN, window_um=20)
    img = im.acquire(SCAN.with_(center_um=tuple(NOMINAL[1]), width_um=20))
    pos, match = al.locate_in_image(img)
    np.testing.assert_allclose(pos, TRUE.apply(NOMINAL[1]), atol=0.04)
    assert match.score > 0.8
