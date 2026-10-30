"""Template generation and normalised cross-correlation (NCC) matching."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage, signal

__all__ = [
    "cross_template",
    "square_template",
    "template_from_image",
    "preprocess",
    "ncc_map",
    "match_template",
    "MatchResult",
]


# ------------------------------------------------------------------- templates
def _grid(half_um: float, pixel_um: float):
    n = int(math.ceil(half_um / pixel_um))
    ax = (np.arange(-n, n + 1)) * pixel_um
    return np.meshgrid(ax, ax[::-1])


def cross_template(
    size_um: float,
    width_um: float,
    pixel_um: float,
    margin_um: float | None = None,
    bright: bool = True,
    blur_px: float = 0.7,
) -> np.ndarray:
    """Cross ('+') mark template sampled at *pixel_um* (odd square array, mark centred)."""
    if pixel_um <= 0:
        raise ValueError("pixel_um must be positive")
    margin = size_um * 0.25 if margin_um is None else margin_um
    xx, yy = _grid(size_um / 2 + margin, pixel_um)
    dx, dy = np.abs(xx), np.abs(yy)
    m = ((dx <= size_um / 2) & (dy <= width_um / 2)) | ((dy <= size_um / 2) & (dx <= width_um / 2))
    t = m.astype(float)
    if blur_px > 0:
        t = ndimage.gaussian_filter(t, blur_px)
    return t if bright else 1.0 - t


def square_template(
    size_um: float, pixel_um: float, margin_um: float | None = None, bright: bool = True, blur_px: float = 0.7
) -> np.ndarray:
    margin = size_um * 0.4 if margin_um is None else margin_um
    xx, yy = _grid(size_um / 2 + margin, pixel_um)
    t = ((np.abs(xx) <= size_um / 2) & (np.abs(yy) <= size_um / 2)).astype(float)
    if blur_px > 0:
        t = ndimage.gaussian_filter(t, blur_px)
    return t if bright else 1.0 - t


def template_from_image(image: np.ndarray, center_colrow, half_size_px: int) -> np.ndarray:
    """Cut a (2h+1)² template from an image around (col, row)."""
    c, r = (int(round(v)) for v in center_colrow)
    h = int(half_size_px)
    if r - h < 0 or c - h < 0 or r + h + 1 > image.shape[0] or c + h + 1 > image.shape[1]:
        raise ValueError("template window exceeds the image")
    return np.array(image[r - h : r + h + 1, c - h : c + h + 1], dtype=float)


# ------------------------------------------------------------------ matching
def preprocess(image: np.ndarray, background_sigma_px: float | None = None, smooth_px: float = 0.0):
    """Fill NaNs, optionally remove slow background and smooth."""
    img = np.array(image, dtype=float)
    if np.any(~np.isfinite(img)):
        fill = np.nanmean(img) if np.any(np.isfinite(img)) else 0.0
        img[~np.isfinite(img)] = fill
    if background_sigma_px:
        img = img - ndimage.gaussian_filter(img, background_sigma_px)
    if smooth_px:
        img = ndimage.gaussian_filter(img, smooth_px)
    return img


def _box_sum(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sum over every h×w window ('valid' size) using an integral image."""
    s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation, 'valid' mode: shape (H-h+1, W-w+1), values in [-1, 1].

    Entry (r, c) scores the template placed with its top-left corner at (r, c).
    """
    img = np.asarray(image, float)
    t = np.asarray(template, float)
    h, w = t.shape
    if img.shape[0] < h or img.shape[1] < w:
        raise ValueError("template larger than image")
    t0 = t - t.mean()
    tnorm = math.sqrt(float((t0 * t0).sum()))
    if tnorm == 0:
        raise ValueError("template has zero variance")
    num = signal.fftconvolve(img, t0[::-1, ::-1], mode="valid")
    n = h * w
    s1 = _box_sum(img, h, w)
    s2 = _box_sum(img * img, h, w)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = np.sqrt(var) * tnorm
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-12 * max(1.0, float(den.max())), num / den, 0.0)
    return np.clip(out, -1.0, 1.0)


def _parabolic(fm1: float, f0: float, fp1: float) -> float:
    d = fm1 - 2 * f0 + fp1
    if d >= 0:  # not a maximum
        return 0.0
    return float(np.clip(0.5 * (fm1 - fp1) / d, -0.5, 0.5))


@dataclass
class MatchResult:
    col: float  # image column of the template centre (sub-pixel)
    row: float  # image row of the template centre (sub-pixel)
    score: float  # NCC peak value
    score_map: np.ndarray | None = None
    second_score: float = float("nan")  # best score outside the peak neighbourhood

    @property
    def colrow(self) -> np.ndarray:
        return np.array([self.col, self.row])

    @property
    def distinctness(self) -> float:
        """Peak minus runner-up – low values flag ambiguous matches."""
        return self.score - self.second_score


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    search_center=None,
    search_radius_px: float | None = None,
    keep_map: bool = True,
    exclusion_px: int | None = None,
) -> MatchResult:
    """Locate *template* in *image* with sub-pixel precision.

    ``search_center`` (col, row) + ``search_radius_px`` restrict the search to
    template-centre positions within that circle.
    """
    t = np.asarray(template, float)
    h, w = t.shape
    m = ncc_map(image, t)
    oy, ox = (h - 1) / 2.0, (w - 1) / 2.0  # template centre offset
    work = m.copy()
    if search_center is not None and search_radius_px is not None:
        rr, cc = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
        dc = cc + ox - search_center[0]
        dr = rr + oy - search_center[1]
        work[(dc * dc + dr * dr) > search_radius_px**2] = -np.inf
    if not np.any(np.isfinite(work)):
        raise ValueError("search region does not overlap valid correlation area")
    r, c = np.unravel_index(int(np.argmax(work)), work.shape)
    peak = float(m[r, c])
    sub_r = _parabolic(m[r - 1, c], m[r, c], m[r + 1, c]) if 0 < r < m.shape[0] - 1 else 0.0
    sub_c = _parabolic(m[r, c - 1], m[r, c], m[r, c + 1]) if 0 < c < m.shape[1] - 1 else 0.0
    ex = exclusion_px if exclusion_px is not None else max(2, min(h, w) // 4)
    w2 = work.copy()
    w2[max(0, r - ex) : r + ex + 1, max(0, c - ex) : c + ex + 1] = -np.inf
    second = float(np.max(w2)) if np.any(np.isfinite(w2)) else float("nan")
    return MatchResult(
        col=c + sub_c + ox,
        row=r + sub_r + oy,
        score=peak,
        score_map=m if keep_map else None,
        second_score=second,
    )
