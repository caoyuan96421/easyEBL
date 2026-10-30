import numpy as np
import pytest
from scipy import ndimage

from easyebl.alignment import (
    cross_template,
    fit_transform,
    match_template,
    ncc_map,
    preprocess,
    square_template,
    template_from_image,
)
from easyebl.core import AffineTransform


def scene(cx, cy, n=96, px=0.25, size=8.0, width=1.0, noise=0.0, seed=0):
    """Cross rendered at sub-pixel position (cx, cy) in pixel coords (col, row)."""
    from easyebl.hal.simulated import render_cross

    ax = np.arange(n)
    cols, rows = np.meshgrid(ax, ax)
    img = render_cross(cols * px, rows * px, cx * px, cy * px, size, width, px)
    img = ndimage.gaussian_filter(img, 0.7)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img


class TestTemplates:
    def test_cross_shape_and_symmetry(self):
        t = cross_template(8, 1, 0.25)
        assert t.shape[0] == t.shape[1] and t.shape[0] % 2 == 1
        np.testing.assert_allclose(t, t[::-1, ::-1])
        assert t[t.shape[0] // 2, t.shape[1] // 2] > 0.9
        d = cross_template(8, 1, 0.25, bright=False)
        assert d[t.shape[0] // 2, t.shape[1] // 2] < 0.1

    def test_square_and_cutout(self):
        t = square_template(4, 0.5)
        assert t.shape[0] % 2 == 1
        img = np.arange(100.0).reshape(10, 10)
        np.testing.assert_array_equal(template_from_image(img, (5, 4), 1), img[3:6, 4:7])
        with pytest.raises(ValueError):
            template_from_image(img, (0, 0), 2)
        with pytest.raises(ValueError):
            cross_template(8, 1, 0)


class TestNCC:
    def test_perfect_match_and_range(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=(40, 50))
        t = img[10:20, 5:18].copy()
        m = ncc_map(img, t)
        assert m.shape == (31, 38)
        r, c = np.unravel_index(np.argmax(m), m.shape)
        assert (r, c) == (10, 5) and m[r, c] == pytest.approx(1.0)
        assert m.min() >= -1 and m.max() <= 1
        # invariant to gain/offset
        m2 = ncc_map(3 * img + 7, t)
        np.testing.assert_allclose(m, m2, atol=1e-6)

    def test_errors(self):
        with pytest.raises(ValueError):
            ncc_map(np.zeros((5, 5)), np.ones((6, 6)))
        with pytest.raises(ValueError):
            ncc_map(np.zeros((10, 10)), np.ones((3, 3)))

    @pytest.mark.parametrize("cx,cy", [(48.0, 48.0), (40.3, 55.71), (51.5, 44.25)])
    def test_subpixel_accuracy(self, cx, cy):
        img = scene(cx, cy)
        t = cross_template(8, 1, 0.25)
        m = match_template(img, t)
        assert m.col == pytest.approx(cx, abs=0.12) and m.row == pytest.approx(cy, abs=0.12)
        assert m.score > 0.9 and m.distinctness > 0.2

    def test_noise_robustness(self):
        img = scene(45.4, 50.2, noise=0.3, seed=3)
        m = match_template(img, cross_template(8, 1, 0.25))
        assert m.col == pytest.approx(45.4, abs=0.3) and m.row == pytest.approx(50.2, abs=0.3)

    def test_search_region(self):
        img = scene(30, 30) + scene(70, 70)
        t = cross_template(8, 1, 0.25)
        m = match_template(img, t, search_center=(70, 70), search_radius_px=10)
        assert m.col == pytest.approx(70, abs=0.2)
        with pytest.raises(ValueError):
            match_template(img, t, search_center=(-500, -500), search_radius_px=1)

    def test_preprocess(self):
        a = np.ones((10, 10))
        a[2, 3] = np.nan
        p = preprocess(a)
        assert np.isfinite(p).all()
        ramp = np.add.outer(np.arange(50.0), np.arange(50.0))
        assert abs(preprocess(ramp, background_sigma_px=3)[15:-15, 15:-15]).max() < 1e-6


class TestFitting:
    def setup_method(self):
        self.src = np.array([[-35, -35], [35, -35], [35, 35], [-35, 35], [0, 10.0]])

    @pytest.mark.parametrize(
        "model,true",
        [
            ("translation", AffineTransform.translation(1.2, -0.4)),
            ("rigid", AffineTransform.from_params(0.5, 0.2, 0.004)),
            ("similarity", AffineTransform.from_params(0.5, 0.2, -0.004, 1.002, 1.002)),
            ("affine", AffineTransform.from_params(0.5, 0.2, 0.004, 1.002, 0.998, 0.001)),
        ],
    )
    def test_recovers_exact(self, model, true):
        f = fit_transform(self.src, true.apply(self.src), model)
        np.testing.assert_allclose(f.transform.matrix, true.matrix, atol=1e-10)
        assert f.rms < 1e-9 and f.max_error < 1e-9

    def test_lower_order_model_leaves_residuals(self):
        true = AffineTransform.from_params(0, 0, 0.01)
        f = fit_transform(self.src, true.apply(self.src), "translation")
        assert f.rms > 0.1

    def test_min_points_and_degenerate(self):
        with pytest.raises(ValueError):
            fit_transform(self.src[:1], self.src[:1], "rigid")
        with pytest.raises(ValueError):
            fit_transform(self.src[:2], self.src[:2], "affine")
        line = np.array([[0, 0], [1, 1], [2, 2.0]])
        with pytest.raises(ValueError):
            fit_transform(line, line, "affine")
        with pytest.raises(ValueError):
            fit_transform(self.src, self.src, "projective")
        with pytest.raises(ValueError):
            fit_transform(self.src, self.src[:3])

    def test_weights(self):
        dst = self.src.copy()
        dst[4] += [10, 0]  # outlier
        w = np.array([1, 1, 1, 1, 0.0])
        f = fit_transform(self.src, dst, "translation", weights=w)
        np.testing.assert_allclose(f.transform.offset, [0, 0], atol=1e-12)
        with pytest.raises(ValueError):
            fit_transform(self.src, dst, "translation", weights=-w)
