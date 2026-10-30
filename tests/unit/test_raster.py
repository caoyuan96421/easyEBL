import numpy as np
import pytest

from easyebl.core import FieldCalibration, OutOfFieldError
from easyebl.imaging import LineAssembler, RasterConfig, RasterPlan

CAL = FieldCalibration(100.0, 10.0)


def plan(**kw):
    base = dict(nx=8, ny=4, width_um=8.0, rate=100e3, dwell_s=30e-6, preline_s=40e-6, flyback_s=50e-6,
                frame_settle_s=100e-6)
    base.update(kw)
    return RasterPlan(RasterConfig(**base), CAL)


def volts_to_um(v):
    return v / CAL.volts_per_um


class TestConfig:
    def test_quantisation(self):
        c = RasterConfig(rate=400e3, dwell_s=5.1e-6)
        assert c.samples_per_pixel == 2 and c.actual_dwell_s == pytest.approx(5e-6)
        assert RasterConfig(rate=400e3, dwell_s=1e-9).samples_per_pixel == 1

    @pytest.mark.parametrize("kw", [dict(nx=0), dict(preline_mode="x"), dict(flyback_shape="x"), dict(x_direction=2),
                                    dict(y_direction="left"), dict(frames=0), dict(settle_samples=5, dwell_s=10e-6, rate=100e3),
                                    dict(width_um=-1)])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            RasterConfig(**kw)

    def test_height_and_pixels(self):
        c = RasterConfig(nx=100, ny=50, width_um=10)
        assert c.height == pytest.approx(5) and c.pixel_size_um == pytest.approx((0.1, 0.1))


class TestPlan:
    def test_sample_counts(self):
        p = plan()
        assert (p.spp, p.n_fb, p.n_pre, p.n_pix) == (3, 5, 4, 24)
        assert p.line_len == 33 and p.frame_len == 10 + 4 * 33
        w = p.full_waveform()
        assert len(w) == p.total

    def test_pixel_positions_and_blanking(self):
        p = plan(blank_preline=True)
        w = p.full_waveform()
        x, y = volts_to_um(w.x), volts_to_um(w.y)
        for line in range(4):
            starts = p.pixel_sample_starts(0, line)
            row = p.line_rows[line]
            for col, s in enumerate(starts):
                seg = slice(s, s + p.spp)
                np.testing.assert_allclose(x[seg], -4 + col + 0.5)
                np.testing.assert_allclose(y[seg], 2 - row - 0.5 * 1)
                assert not w.blank[seg].any()
            ls = p.line_start_sample(0, line)
            assert w.blank[ls : ls + p.n_fb].all()  # flyback blanked
            assert w.blank[ls + p.n_fb : ls + p.n_fb + p.n_pre].all()  # pre-line blanked (configured)
        assert w.blank[: p.n_frame_settle].all()
        assert w.blank[-p.n_tail :].all()

    def test_preline_unblanked_when_configured(self):
        p = plan(blank_preline=False, blank_flyback=False)
        w = p.full_waveform()
        ls = p.line_start_sample(0, 1)
        assert not w.blank[ls : ls + p.n_fb + p.n_pre].any()
        # first line's "flyback" is part of the frame settle and always blanked
        ls0 = p.line_start_sample(0, 0)
        assert w.blank[ls0 : ls0 + p.n_fb].all()

    def test_preline_ramp_vs_hold(self):
        ramp = plan(preline_mode="ramp")
        hold = plan(preline_mode="hold")
        for p, expect in [(ramp, [-5.5, -4.5, -4.5, -4.5]), (hold, [-3.5] * 4)]:
            x = volts_to_um(p.full_waveform().x)
            s = p.line_start_sample(0, 1) + p.n_fb
            # 4 pre-line samples, spp=3: ramp continues the staircase backwards
            # (pixel -2 for one sample, then pixel -1 for a full pixel dwell)
            np.testing.assert_allclose(x[s : s + 4], expect)

    def test_flyback_shapes(self):
        for shape in ("linear", "cosine", "step"):
            p = plan(flyback_shape=shape)
            x = volts_to_um(p.full_waveform().x)
            s = p.line_start_sample(0, 1)
            fb = x[s : s + p.n_fb]
            assert fb[-1] == pytest.approx(x[s + p.n_fb])  # arrives at pre-line start
            if shape == "step":
                assert np.all(fb == fb[-1])
            else:
                assert np.all(np.diff(fb) <= 1e-12)  # monotonic return

    def test_bidirectional_and_direction(self):
        p = plan(bidirectional=True)
        np.testing.assert_array_equal(p.line_dirs, [1, -1, 1, -1])
        s = p.pixel_sample_starts(0, 1)
        assert np.all(np.diff(s) < 0)  # column 0 scanned last on reverse lines
        x = volts_to_um(p.full_waveform().x)
        assert x[s[0]] == pytest.approx(-3.5) and x[s[-1]] == pytest.approx(3.5)
        p2 = plan(x_direction=-1)
        assert np.all(p2.line_dirs == -1)
        p3 = plan(y_direction="up")
        np.testing.assert_array_equal(p3.line_rows, [3, 2, 1, 0])

    def test_rotation(self):
        p = plan(rotation_rad=np.pi / 2, nx=4, ny=4, width_um=4)
        w = p.full_waveform()
        s = p.pixel_sample_starts(0, 0)[0]  # row 0, col 0: local (-1.5, +1.5)
        np.testing.assert_allclose([volts_to_um(w.x[s]), volts_to_um(w.y[s])], [-1.5, -1.5], atol=1e-12)

    def test_markers(self):
        p = plan(line_sync_marker="line", frame_sync_marker="frame", frames=2)
        w = p.full_waveform()
        line = w.markers["line"]
        assert line.sum() == 2 * 4 * p.n_pix
        assert w.markers["frame"].sum() == 2 * p.n_pix
        s = p.pixel_sample_starts(1, 0).min()
        assert w.markers["frame"][s] and line[s]

    def test_out_of_field(self):
        with pytest.raises(OutOfFieldError):
            plan(center_um=(48.0, 0.0))
        with pytest.raises(OutOfFieldError):  # pre-line overscan leaves the field
            RasterPlan(RasterConfig(nx=10, ny=10, width_um=99.0, preline_s=1e-3, rate=100e3, dwell_s=10e-6), CAL)

    def test_negative_lag_rejected(self):
        with pytest.raises(ValueError):
            plan(lag_comp_s=-1e-6)


class TestAssembler:
    def _feed_index_signal(self, p, block_sizes=(17, 1, 400, 10_000)):
        """AI value == sample index → pixel value == mean index of the samples used."""
        asm = LineAssembler(p, ["v"])
        sig = np.arange(p.total, dtype=float)[None, :]
        i, k = 0, 0
        while i < p.total:
            n = block_sizes[k % len(block_sizes)]
            asm.feed(sig[:, i : i + n])
            i += n
            k += 1
        return asm

    def test_pixel_means_and_frames(self):
        p = plan(frames=2)
        asm = self._feed_index_signal(p)
        img = asm.images()["v"]
        for line in range(4):
            row = p.line_rows[line]
            exp = np.mean([p.pixel_sample_starts(f, line) + 1 for f in range(2)], axis=0)  # mean of s, s+1, s+2
            np.testing.assert_allclose(img[row], exp)
        assert asm.rows_done == 8 and np.all(asm.counts == 2)

    def test_settle_and_fractional_lag(self):
        p = plan(settle_samples=1, lag_comp_s=25e-6)  # 2.5 samples
        img = self._feed_index_signal(p).images()["v"]
        s = p.pixel_sample_starts(0, 0)
        # samples s+1, s+2 shifted by +2.5 → mean index s + 1.5 + 2.5
        np.testing.assert_allclose(img[p.line_rows[0]], s + 4.0)

    def test_incomplete_data_gives_nan_rows(self):
        p = plan()
        asm = LineAssembler(p, ["v"])
        asm.feed(np.zeros((1, p.line_start_sample(0, 2))))
        img = asm.images()["v"]
        assert asm.rows_done == 1 and np.isnan(img[1:]).all()
