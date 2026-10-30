import numpy as np
import pytest

from easyebl.core import AffineTransform, FieldCalibration, OutOfFieldError
from easyebl.litho import (
    Dot,
    DoseCalculator,
    ExposureParameters,
    ExposureRun,
    Pattern,
    Polygon,
    Polyline,
    VectorScanCompiler,
    VectorScanSettings,
    build_runs,
    line_points,
    order_elements,
    points_in_polygon,
    raster_fill,
    rectangle,
    regular_polygon,
)
from easyebl.litho import dose as D

CAL = FieldCalibration(100.0, 10.0)


# ------------------------------------------------------------------------ dose
class TestDose:
    def test_reference_values(self):
        # 300 µC/cm², 20 nm step, 100 pA -> 12 µs
        assert D.area_dwell_s(300, 0.02, 100e-12) == pytest.approx(12e-6)
        # 1000 pC/cm, 10 nm step, 100 pA -> 10 µs
        assert D.line_dwell_s(1000, 0.01, 100e-12) == pytest.approx(10e-6)
        assert D.dot_dwell_s(10, 100e-12) == pytest.approx(100e-6)

    def test_inverses(self):
        dw = D.area_dwell_s(250, 0.05, 1e-9)
        assert D.area_dose_uC_cm2(dw, 0.05, 1e-9) == pytest.approx(250)
        dw = D.line_dwell_s(700, 0.02, 50e-12)
        assert D.line_dose_pC_cm(dw, 0.02, 50e-12) == pytest.approx(700)
        assert D.dot_dose_fC(D.dot_dwell_s(3, 2e-12), 2e-12) == pytest.approx(3)

    def test_helpers(self):
        assert D.step_from_points(100, 5000) == pytest.approx(0.02)
        assert D.dac_step_um(100, 16) == pytest.approx(100 / 65536)
        n, actual, err = D.quantize_dwell(12e-6, 400e3)
        assert n == 5 and actual == pytest.approx(12.5e-6) and err == pytest.approx(1 / 24)
        assert D.quantize_dwell(1e-9, 400e3)[0] == 1
        assert D.area_write_time_s(100, 300, 100e-12) == pytest.approx(3.0)
        I = D.max_current_for_dwell(300, 0.02, 5e-6)
        assert D.area_dwell_s(300, 0.02, I) == pytest.approx(5e-6)
        for bad in (0, -1, float("nan")):
            with pytest.raises(ValueError):
                D.area_dwell_s(bad, 0.02, 1e-10)
        with pytest.raises(ValueError):
            D.step_from_points(100, 0)

    def test_calculator_warnings(self):
        ok = DoseCalculator(rate_hz=500e3).report()
        assert ok.area_dwell_samples == 6 and not ok.warnings
        bad = DoseCalculator(current_pA=5000, points=2**20, rate_hz=400e3).report()
        text = " ".join(bad.warnings)
        assert "below the DAC resolution" in text and "minimum" in text
        assert set(bad.as_dict()) >= {"area_dwell_s", "warnings"}


# -------------------------------------------------------------------- geometry
class TestGeometry:
    def test_polygon_basics(self):
        p = Polygon([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
        assert len(p.points) == 4 and p.area == pytest.approx(2) and p.bbox == (0, 0, 2, 1)
        np.testing.assert_array_equal(p.contains([[1, 0.5], [3, 0.5]]), [True, False])
        q = p.transformed(AffineTransform.translation(1, 1))
        assert q.bbox == (1, 1, 3, 2) and q.dose_factor == 1.0
        with pytest.raises(ValueError):
            Polygon([[0, 0], [1, 1]])
        with pytest.raises(ValueError):
            Polygon([[0, 0], [1, 0], [0, 1]], dose_factor=-1)

    def test_line_dot_pattern(self):
        l = Polyline([[0, 0], [3, 4], [3, 0]])
        assert l.length == pytest.approx(9)
        with pytest.raises(ValueError):
            Polyline([[0, 0]])
        d = Dot(1, 2, 2.0)
        assert d.transformed(AffineTransform.scaling(2)).x == 2
        pat = Pattern([rectangle(0, 0, 2, 2, dose_factor=1.5), l, d])
        s = pat.summary()
        assert (s["polygons"], s["lines"], s["dots"]) == (1, 1, 1)
        assert s["area_um2"] == pytest.approx(4) and s["dose_factor_range"] == (1.0, 2.0)
        c = pat.centered()
        b = c.bbox
        assert (b[0] + b[2]) / 2 == pytest.approx(0) and (b[1] + b[3]) / 2 == pytest.approx(0)
        assert Pattern().bbox is None
        assert len(Pattern.concat([pat, pat])) == 6

    def test_points_in_polygon_even_odd(self):
        sq = np.array([[0, 0], [4, 0], [4, 4], [0, 4.0]])
        assert points_in_polygon(np.array([[2, 2]]), sq)[0]
        assert regular_polygon(0, 0, 1, 64).area == pytest.approx(np.pi, rel=2e-3)


# ------------------------------------------------------------------------ fill
class TestFill:
    def test_rectangle_count_and_grid(self):
        r = rectangle(0, 0, 1.0, 0.5)
        f = raster_fill(r, 0.1)
        pts = f.all_points()
        assert f.n_points == 50 and len(f.runs) == 5
        np.testing.assert_allclose(np.sort(np.unique(pts[:, 0])), np.arange(10) * 0.1 + 0.05)
        assert np.all(r.contains(pts))

    def test_tiling_has_no_gaps_or_overlaps(self):
        a = raster_fill(rectangle(0, 0, 1.0, 1.0), 0.1).all_points()
        b = raster_fill(rectangle(1.0, 0, 2.0, 1.0), 0.1).all_points()
        both = np.vstack([a, b])
        assert len(np.unique(np.round(both, 9), axis=0)) == len(both) == 200

    def test_area_accuracy_on_arbitrary_shapes(self):
        for poly in (regular_polygon(0.013, -0.02, 3, 50), Polygon([[0, 0], [5, 0.3], [2, 4]])):
            f = raster_fill(poly, 0.02)
            assert f.n_points * 0.02**2 == pytest.approx(poly.area, rel=0.01)
            assert np.all(poly.contains(f.all_points()))

    def test_serpentine_and_direction(self):
        r = rectangle(0, 0, 0.5, 0.3)
        f = raster_fill(r, 0.1, serpentine=True)
        assert f.runs[0][0, 0] < f.runs[0][-1, 0] and f.runs[1][0, 0] > f.runs[1][-1, 0]
        fr = raster_fill(r, 0.1, reverse=True)
        assert all(run[0, 0] > run[-1, 0] for run in fr.runs)
        fy = raster_fill(r, 0.1, direction="y")
        assert len(fy.runs) == 5 and np.allclose(fy.runs[0][:, 0], fy.runs[0][0, 0])
        with pytest.raises(ValueError):
            raster_fill(r, 0)
        with pytest.raises(ValueError):
            raster_fill(r, 0.1, direction="z")

    def test_concave_and_hole(self):
        u = Polygon([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]])
        f = raster_fill(u, 0.1)
        top_row = [r for r in f.runs if r[0, 1] > 2]
        assert len(top_row) == 2 * 10  # two separate spans per row in the upper part
        from easyebl.litho.gui_vector import _ring

        ring = Polygon(_ring(6, 3, 64))
        pts = raster_fill(ring, 0.1).all_points()
        r = np.hypot(*pts.T)
        assert r.min() > 2.9 and r.max() < 6.0
        assert len(pts) * 0.01 == pytest.approx(ring.area, rel=0.02)

    def test_line_points(self):
        lp = line_points(Polyline([[0, 0], [1, 0], [1, 0.5]]), 0.1)
        assert len(lp.points) == 15 and lp.spacing_um.sum() == pytest.approx(1.5)
        np.testing.assert_allclose(lp.points[0], [0.05, 0])
        lp2 = line_points(Polyline([[0, 0], [1, 0]]), 0.3)  # 1/0.3 -> 3 points of 1/3
        np.testing.assert_allclose(lp2.spacing_um, [1 / 3] * 3)
        with pytest.raises(ValueError):
            line_points(Polyline([[0, 0], [0, 0]]), 0.1)


# -------------------------------------------------------------------- compiler
def compile_runs(runs, **kw):
    kw.setdefault("quantization", "round")
    s = VectorScanSettings(rate=100e3, settle_base_s=50e-6, settle_per_um_s=10e-6, settle_max_s=1e-3,
                           blank_lead_s=20e-6, park_um=(0, 0), park_samples=3, chunk_samples=37, **kw)
    return VectorScanCompiler(CAL, s).compile(runs)


class TestCompiler:
    def test_stream_structure(self):
        run1 = ExposureRun([[0, 0], [0.1, 0]], [30e-6, 50e-6])
        run2 = ExposureRun([[10, 0]], 20e-6)
        job = compile_runs([run1, run2])
        w = job.preview_arrays()
        st = job.stats
        assert len(w) == st.total_samples
        assert st.exposure_samples == 3 + 5 + 2 and st.n_jumps == 2 and st.n_runs == 2
        assert np.count_nonzero(~w.blank) == 10
        x = w.x / CAL.volts_per_um
        on = ~w.blank
        np.testing.assert_allclose(x[on], [0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 10], atol=1e-12)
        # jump to run2: lead (2 samples at 0.1) + settle 50 µs + 10 µs/µm * 9.9 µm = 149 µs -> 15 samples
        i_end1 = np.nonzero(on)[0][7]
        blanked = w.blank[i_end1 + 1 : i_end1 + 1 + 17]
        assert blanked.all() and np.all(x[i_end1 + 1 : i_end1 + 3] == pytest.approx(0.1))
        assert np.all(x[i_end1 + 3 : i_end1 + 18] == pytest.approx(10))
        assert w.blank[-3:].all() and x[-1] == 0  # parked

    def test_chunks_respect_size(self):
        job = compile_runs([ExposureRun(np.random.default_rng(0).uniform(-5, 5, (40, 2)), 30e-6)])
        sizes = [len(c) for c in job.chunks()]
        assert all(s == 37 for s in sizes[:-1]) and sum(sizes) == job.stats.total_samples

    def test_quantisation_stats(self):
        job = compile_runs([ExposureRun([[0, 0], [1, 1]], [25e-6, 1e-6])])
        st = job.stats
        assert st.clamped_points == 1 and st.dose_error_max == pytest.approx(9.0)
        assert st.requested_exposure_s == pytest.approx(26e-6)
        z = compile_runs([ExposureRun([[0, 0], [1, 1]], [0.0, 10e-6])])
        assert z.stats.zero_dose_points == 1 and z.stats.exposure_samples == 1

    def test_split_and_merge(self):
        run = ExposureRun([[0, 0], [0.1, 0], [5, 0], [5.1, 0]], 10e-6)
        assert compile_runs([run], max_unblanked_step_um=1.0).stats.n_jumps == 2
        a, b = ExposureRun([[0, 0]], 10e-6), ExposureRun([[0.05, 0]], 10e-6)
        assert compile_runs([a, b]).stats.n_jumps == 2
        assert compile_runs([a, b], merge_runs_within_um=0.1).stats.n_jumps == 1

    def test_blank_between_points(self):
        job = compile_runs([ExposureRun([[0, 0], [1, 0]], 20e-6)], blank_between_points=True, point_settle_s=30e-6)
        w = job.preview_arrays()
        on = np.nonzero(~w.blank)[0]
        assert len(on) == 4 and on[2] - on[1] == 4  # 3 blanked settle samples between points

    def test_blanker_delay_shifts_xy(self):
        runs = [ExposureRun([[0, 0]], 20e-6), ExposureRun([[5, 0]], 20e-6)]
        w0 = compile_runs(runs).preview_arrays()
        w2 = compile_runs(runs, blanker_delay_s=20e-6).preview_arrays()
        assert len(w2) == len(w0) + 2
        np.testing.assert_array_equal(w2.blank[: len(w0)], w0.blank)
        np.testing.assert_allclose(w2.x[2:], w0.x)  # XY delayed by 2 samples

    def test_error_diffusion(self):
        run = ExposureRun(np.zeros((8, 2)), np.full(8, 25e-6))  # 2.5 samples each
        r = compile_runs([run])
        d = compile_runs([run], quantization="diffuse")
        assert r.stats.exposure_samples == 16 and r.stats.dose_error_total == pytest.approx(-0.2)
        assert d.stats.exposure_samples == 20 and d.stats.dose_error_total == pytest.approx(0)
        assert set(d.samples[0]) <= {2, 3}
        with pytest.raises(ValueError):
            VectorScanSettings(quantization="dither")

    def test_out_of_field(self):
        with pytest.raises(OutOfFieldError):
            compile_runs([ExposureRun([[60, 0]], 10e-6)])
        VectorScanCompiler(CAL).compile([ExposureRun([[60, 0]], 10e-6)], check=False)

    def test_run_validation(self):
        with pytest.raises(ValueError):
            ExposureRun([[0, 0], [1, 1]], [1e-6])
        with pytest.raises(ValueError):
            ExposureRun([[0, 0]], -1e-6)
        with pytest.raises(ValueError):
            compile_runs([ExposureRun(np.zeros((3, 2)), 1e-5)]).preview_arrays(max_samples=1)


# ---------------------------------------------------------------------- writer
class TestBuildRuns:
    def test_dwell_per_kind(self):
        p = ExposureParameters(beam_current_a=100e-12, area_dose_uC_cm2=300, area_step_um=0.02,
                               line_dose_pC_cm=1000, line_step_um=0.01, dot_dose_fC=10)
        pat = Pattern([rectangle(0, 0, 0.1, 0.1, dose_factor=2.0), Polyline([[0, 0], [0.1, 0]], 0.5), Dot(0, 0, 3.0),
                       Dot(1, 1, 0.0)])
        runs = build_runs(pat, p)
        area = [r for r in runs if r.kind == "area"]
        assert sum(len(r) for r in area) == 25
        assert area[0].dwell_s[0] == pytest.approx(24e-6)
        line = [r for r in runs if r.kind == "line"][0]
        assert len(line) == 10 and line.dwell_s[0] == pytest.approx(5e-6)
        dots = [r for r in runs if r.kind == "dot"]
        assert len(dots) == 1 and dots[0].dwell_s[0] == pytest.approx(300e-6)  # zero-dose dot skipped

    def test_transform_applied_after_fill(self):
        p = ExposureParameters(area_step_um=0.1)
        t = AffineTransform.translation(0.033, 0)
        a = build_runs(Pattern([rectangle(0, 0, 1, 1)]), p)
        b = build_runs(Pattern([rectangle(0, 0, 1, 1)]), p, transform=t)
        np.testing.assert_allclose(np.vstack([r.points_um for r in b]), np.vstack([r.points_um for r in a]) + [0.033, 0])

    def test_ordering(self):
        els = [Dot(10, 0), Polyline([[0, 0], [1, 0]]), Dot(1.2, 0), rectangle(5, 5, 6, 6)]
        assert [e.kind for e in order_elements(els, "kind")] == ["dot", "dot", "line", "area"]
        near = order_elements(els, "nearest")
        assert near[0] is els[1] and near[1] is els[2]
        with pytest.raises(ValueError):
            order_elements(els, "random")
