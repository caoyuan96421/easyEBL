"""Analysis of dummy-image (phantom loopback) scans: fidelity, registration, noise.

Used by ``scripts/hw_phantom_test.py`` and the hardware tests.  All functions
compare a measured image with :meth:`PhantomSpecimen.expected_image`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fit", "fit_to_expected", "subpixel_shift", "pair_noise"]


@dataclass
class Fit:
    gain: float  # measured = gain · expected + offset (AI gain, wiring)
    offset_v: float
    r2: float  # fraction of the expected image's variance reproduced
    resid_rms_v: float  # what is left: noise + any misregistration/bandwidth error

    def __str__(self) -> str:
        return (f"gain {self.gain:.4f}, offset {self.offset_v * 1e3:+.1f} mV, R² {self.r2:.5f}, "
                f"residual {self.resid_rms_v * 1e3:.2f} mV rms")


def fit_to_expected(measured: np.ndarray, expected: np.ndarray) -> Fit:
    m = np.asarray(measured, float).ravel()
    e = np.asarray(expected, float).ravel()
    ok = np.isfinite(m) & np.isfinite(e)
    m, e = m[ok], e[ok]
    A = np.column_stack([e, np.ones_like(e)])
    (g, o), *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - (g * e + o)
    var = np.var(m)
    return Fit(float(g), float(o), float(1 - np.var(resid) / var) if var > 0 else 0.0, float(np.std(resid)))


def subpixel_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dx, dy) in pixels by which image *a* is displaced relative to *b* (phase correlation)."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    F /= np.maximum(np.abs(F), 1e-12)
    c = np.fft.ifft2(F).real
    ny, nx = c.shape
    k = np.unravel_index(int(np.argmax(c)), c.shape)

    def refine(axis_vals, i, n):
        y0, y1, y2 = axis_vals[(i - 1) % n], axis_vals[i], axis_vals[(i + 1) % n]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        p = i + d
        return p - n if p > n / 2 else p

    dy = refine(c[:, k[1]], k[0], ny)
    dx = refine(c[k[0], :], k[1], nx)
    return float(dx), float(dy)


def pair_noise(a: np.ndarray, b: np.ndarray) -> float:
    """Per-pixel noise (V rms) from two frames of the same scene: std(a − b) / √2."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.nanstd(d) / np.sqrt(2))
