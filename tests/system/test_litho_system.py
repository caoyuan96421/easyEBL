"""Vector-scan lithography end-to-end: compile → stream → simulated dose map."""
import threading

import numpy as np
import pytest
from scipy import ndimage

from easyebl.core import AffineTransform
from easyebl.hal import SimParams
from easyebl.litho import (
    Dot,
    ExposureParameters,
    LithoWriter,
    Pattern,
    Polygon,
    Polyline,
    VectorScanSettings,
    rectangle,
    regular_polygon,
)

pytestmark = pytest.mark.system

I = 0.25e-9  # 300 µC/cm² at 50 nm -> 30 µs = exactly 15 samples at 500 kS/s
PARAMS = ExposureParameters(beam_current_a=I, area_dose_uC_cm2=300, area_step_um=0.05,
                            line_dose_pC_cm=2000, line_step_um=0.02, dot_dose_fC=20)


def setup(make_manager, calibration, **sim):
    kw = dict(beam_current_a=I, lag_tau_s=4e-6, dose_pixel_um=0.1, noise_sigma=0)
    kw.update(sim)
    p = SimParams(**kw)
    m = make_manager(p)
    w = LithoWriter(m.get_handle("litho"), calibration, VectorScanSettings(rate=500e3))
    return m, w


def masks(sim, poly, grow=3):
    xs, ys = sim.dose.coords()
    X, Y = np.meshgrid(xs, ys)
    inside = poly.contains(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    core = ndimage.binary_erosion(inside, iterations=grow)
    outside = ~ndimage.binary_dilation(inside, iterations=grow)
    return core, outside


@pytest.mark.parametrize(
    "poly",
    [
        Polygon([[-5, -5], [5, -5], [5, 2], [0, 5], [-5, 5]]),
        regular_polygon(10, -10, 4, 48),
        Polygon([[-20, 10], [-5, 10], [-5, 12], [-18, 12], [-18, 20], [-20, 20]]),  # concave L
    ],
)
def test_polygon_dose_uniform_and_contained(make_manager, calibration, poly):
    m, w = setup(make_manager, calibration)
    rep = w.write_polygon(poly, PARAMS)
    assert not rep.aborted and rep.samples_written == rep.stats.total_samples
    sim = m.backend
    dose = sim.dose.dose_uC_cm2()
    core, outside = masks(sim, poly)
    assert np.median(dose[core]) == pytest.approx(300, rel=0.02)
    assert np.std(dose[core]) / 300 < 0.05
    assert dose[outside].max() == 0  # blanker hides every jump
    # total charge = exposed points × charge per point (quantised dwell)
    n = rep.stats.exposure_samples
    assert sim.dose.total_charge() == pytest.approx(n / 500e3 * I, rel=1e-9)
    assert sim.dose.total_charge() == pytest.approx(poly.area * 300 * 1e-14, rel=0.03)


def test_dose_factors_scale_dose(make_manager, calibration):
    m, w = setup(make_manager, calibration)
    a = rectangle(-10, -2, -4, 2, dose_factor=1.0)
    b = rectangle(4, -2, 10, 2, dose_factor=1.6)
    w.expose(Pattern([a, b]), PARAMS)
    d = m.backend.dose.dose_uC_cm2()
    ca, _ = masks(m.backend, a)
    cb, _ = masks(m.backend, b)
    assert np.median(d[cb]) / np.median(d[ca]) == pytest.approx(1.6, rel=0.03)


@pytest.mark.parametrize("mode", ["diffuse", "round"])
def test_coarse_dwell_quantisation(make_manager, calibration, mode):
    """1 nA / 50 nm / 300 µC/cm² is 3.75 samples per point: rounding overdoses, diffusion does not."""
    m, w = setup(make_manager, calibration, beam_current_a=1e-9)
    w.settings = VectorScanSettings(rate=500e3, quantization=mode)
    sq = rectangle(-5, -5, 5, 5)
    rep = w.write_polygon(sq, ExposureParameters(beam_current_a=1e-9, area_dose_uC_cm2=300, area_step_um=0.05))
    total = m.backend.dose.total_charge() / (sq.area * 1e-14)  # mean dose, µC/cm²
    if mode == "diffuse":
        assert total == pytest.approx(300, rel=1e-3) and abs(rep.stats.dose_error_total) < 1e-3
        assert rep.stats.dose_error_max == pytest.approx(0.75 / 3.75, rel=1e-6)  # 3 or 4 samples per point
    else:
        assert total == pytest.approx(320, rel=1e-3) and rep.stats.dose_error_total == pytest.approx(1 / 15)
    core, _ = masks(m.backend, sq)
    # local ripple: each 0.1 µm dose pixel holds 4 points, i.e. 15 ± 1 samples (±6.7 %)
    assert np.std(m.backend.dose.dose_uC_cm2()[core]) / total < 0.08


def test_lines_and_dots_charge(make_manager, calibration):
    m, w = setup(make_manager, calibration)
    pat = Pattern([Polyline([[-20, 0], [20, 0]], 1.0), Polyline([[-20, 5], [0, 15]], 0.5),
                   Dot(10, 10, 2.0), Dot(-10, -10)])
    rep = w.expose(pat, PARAMS)
    L1, L2 = 40.0, float(np.hypot(20, 10))
    expect = (L1 + 0.5 * L2) * 2000 * 1e-16 + (2 + 1) * 20e-15
    assert m.backend.dose.total_charge() == pytest.approx(expect, rel=0.01)
    assert rep.stats.n_runs == 4
    # dot charge lands where it should
    dm = m.backend.dose
    col = int((10 + dm.n * dm.pixel_um / 2) / dm.pixel_um)
    row = int((dm.n * dm.pixel_um / 2 - 10) / dm.pixel_um)
    assert dm.charge_c[row, col] == pytest.approx(40e-15, rel=0.01)


def test_alignment_transform_moves_exposure(make_manager, calibration):
    m, w = setup(make_manager, calibration, lag_tau_s=0.0)
    sq = rectangle(-3, -3, 3, 3)
    t = AffineTransform.translation(2.0, -1.0)
    w.expose(Pattern([sq]), PARAMS, transform=t)
    core, outside = masks(m.backend, sq.transformed(t))
    d = m.backend.dose.dose_uC_cm2()
    assert np.median(d[core]) == pytest.approx(300, rel=0.02) and d[outside].max() == 0


def test_true_field_distortion_is_corrected_by_calibration(make_manager, calibration):
    distortion = AffineTransform.from_params(0.7, -0.4, 0.003, 1.004, 1.004)
    m, w = setup(make_manager, calibration, distortion=distortion, lag_tau_s=0.0)
    w.calibration = calibration.with_correction(distortion.inverse())
    sq = rectangle(5, 5, 15, 12)
    w.write_polygon(sq, PARAMS)
    core, outside = masks(m.backend, sq)
    d = m.backend.dose.dose_uC_cm2()
    assert np.median(d[core]) == pytest.approx(300, rel=0.03) and d[outside].max() == 0


def test_blanker_delay_compensation(make_manager, calibration):
    """A late blanker leaves dose trails along every jump unless the XY stream is delayed to match."""
    dots = Pattern([Dot(-20 + 4 * i + 0.05, 0.05) for i in range(11)])  # dose-pixel centres
    m, w = setup(make_manager, calibration, blank_delay_s=6e-6, lag_tau_s=4e-6)
    base = dict(rate=500e3, blank_lead_s=0.0, settle_base_s=60e-6, settle_per_um_s=0.0)
    w.settings = VectorScanSettings(**base)
    w.expose(dots, PARAMS)
    smeared = m.backend.dose.charge_c.copy()
    m.backend.dose.reset()
    w.settings = VectorScanSettings(**base, blanker_delay_s=6e-6)
    w.expose(dots, PARAMS)
    fixed = m.backend.dose.charge_c
    hot = lambda q: np.count_nonzero(q > 1e-3 * q.max())  # noqa: E731
    assert hot(fixed) == 11  # every dot lands in one dose pixel
    assert hot(smeared) >= 11 + 2 * 10  # ≥2 trail pixels on each of the 10 jumps
    assert fixed.sum() == pytest.approx(smeared.sum(), rel=1e-9)  # same total charge, misplaced


def test_abort_mid_exposure(make_manager, calibration):
    m, w = setup(make_manager, calibration, realtime=True)
    big = rectangle(-20, -20, 20, 20)
    seen = []

    def progress(f):
        seen.append(f)
        if len(seen) == 3:
            threading.Thread(target=w.handle.abort).start()

    w.settings = VectorScanSettings(rate=500e3, chunk_samples=20_000)
    rep = w.write_polygon(big, PARAMS, progress=progress)
    assert rep.aborted and rep.samples_written < rep.stats.total_samples
    assert m.beam_state.blank and not m.is_busy


def test_out_of_field_pattern_rejected_before_streaming(make_manager, calibration):
    m, w = setup(make_manager, calibration)
    from easyebl.core import OutOfFieldError

    with pytest.raises(OutOfFieldError):
        w.write_polygon(rectangle(40, 40, 55, 45), PARAMS)
    assert m.backend.dose.total_charge() == 0
