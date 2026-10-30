"""Least-squares fitting of 2-D transforms between point sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.transforms import AffineTransform

__all__ = ["fit_transform", "FitResult", "MODELS", "MIN_POINTS"]

MODELS = ("translation", "rigid", "similarity", "affine")
MIN_POINTS = {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3}


@dataclass
class FitResult:
    transform: AffineTransform
    model: str
    residuals: np.ndarray  # (N, 2) dst - T(src)

    @property
    def rms(self) -> float:
        if len(self.residuals) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.sum(self.residuals**2, axis=1))))

    @property
    def max_error(self) -> float:
        if len(self.residuals) == 0:
            return 0.0
        return float(np.max(np.hypot(self.residuals[:, 0], self.residuals[:, 1])))


def _weighted_centroid(p, w):
    return (p * w[:, None]).sum(0) / w.sum()


def fit_transform(src, dst, model: str = "similarity", weights=None) -> FitResult:
    """Find T minimising Σ w‖dst − T(src)‖².

    * ``translation`` – shift only (≥1 point)
    * ``rigid``       – rotation + shift (≥2)
    * ``similarity``  – rotation + uniform scale + shift (≥2)
    * ``affine``      – full 6-parameter affine (≥3, non-collinear)
    """
    src = np.atleast_2d(np.asarray(src, float))
    dst = np.atleast_2d(np.asarray(dst, float))
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src and dst must both be (N, 2)")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    n = len(src)
    if n < MIN_POINTS[model]:
        raise ValueError(f"model '{model}' needs at least {MIN_POINTS[model]} point(s), got {n}")
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")

    if model == "translation":
        t = _weighted_centroid(dst - src, w)
        T = AffineTransform.translation(*t)
    elif model in ("rigid", "similarity"):
        mu_s, mu_d = _weighted_centroid(src, w), _weighted_centroid(dst, w)
        s0, d0 = src - mu_s, dst - mu_d
        cov = (d0 * w[:, None]).T @ s0 / w.sum()
        u, sv, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(u @ vt)) or 1.0
        D = np.diag([1.0, d])
        R = u @ D @ vt
        if model == "similarity":
            var_s = (w * (s0**2).sum(1)).sum() / w.sum()
            if var_s <= 0:
                raise ValueError("degenerate point set (all source points coincide)")
            scale = float(np.trace(np.diag(sv) @ D) / var_s)
        else:
            scale = 1.0
        A = scale * R
        t = mu_d - A @ mu_s
        T = AffineTransform.from_linear(A, t)
    else:  # affine
        X = np.column_stack([src, np.ones(n)])
        sw = np.sqrt(w)[:, None]
        if np.linalg.matrix_rank(X * sw) < 3:
            raise ValueError("degenerate point set for affine fit (collinear points)")
        sol, *_ = np.linalg.lstsq(X * sw, dst * sw, rcond=None)
        A = sol[:2].T
        t = sol[2]
        T = AffineTransform.from_linear(A, t)
    res = dst - T.apply(src)
    return FitResult(T, model, res)
