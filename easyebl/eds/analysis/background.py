"""Spectrum background (bremsstrahlung) estimation: the SNIP algorithm.

SNIP (Ryan et al. 1988, Morháč 2009) clips peaks iteratively with a window
that ends at about one peak width; working on the log-log-sqrt transform makes
it robust for the large dynamic range of EDS spectra.
"""
from __future__ import annotations

import numpy as np

from ..spectrum import EnergyCalibration

__all__ = ["snip", "snip_background", "smooth"]


def smooth(y: np.ndarray, width: int = 3) -> np.ndarray:
    """Box-car smoothing (odd *width*), edges handled by reflection."""
    w = max(1, int(width) | 1)
    if w == 1:
        return np.asarray(y, dtype=float)
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(np.asarray(y, dtype=float), pad, mode="reflect")
    return np.convolve(yp, k, mode="valid")


def snip(y: np.ndarray, width_channels, iterations: int | None = None, decreasing: bool = True) -> np.ndarray:
    """SNIP background of *y*.  *width_channels*: scalar or per-channel window half width
    (≈ 1–1.5 × FWHM of the peaks in channels)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    w = np.broadcast_to(np.asarray(width_channels, dtype=float), (n,)).copy()
    w = np.clip(np.round(w), 1, n // 2 - 1).astype(int)
    v = np.log(np.log(np.sqrt(np.clip(y, 0, None) + 1.0) + 1.0) + 1.0)
    m = int(w.max()) if iterations is None else int(iterations)
    idx = np.arange(n)
    order = range(m, 0, -1) if decreasing else range(1, m + 1)
    for p in order:
        pp = np.minimum(w, p)
        lo = np.clip(idx - pp, 0, n - 1)
        hi = np.clip(idx + pp, 0, n - 1)
        avg = 0.5 * (v[lo] + v[hi])
        v = np.minimum(v, avg)
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0


def snip_background(counts: np.ndarray, calibration: EnergyCalibration, fwhm_kev=None, factor: float = 1.6,
                    smooth_width: int = 3, e_min_kev: float = 0.15) -> np.ndarray:
    """SNIP background for an EDS spectrum, window = *factor* × FWHM(E) (default 130 eV at Mn Kα)."""
    from ...xray.generation import Detector

    n = len(counts)
    e = calibration.energies(n)
    if fwhm_kev is None:
        fwhm = Detector().fwhm_kev(np.clip(e, 0.05, None))
    elif callable(fwhm_kev):
        fwhm = fwhm_kev(np.clip(e, 0.05, None))
    else:
        fwhm = np.full(n, float(fwhm_kev))
    w = factor * fwhm / calibration.channel_width(np.arange(n))
    ys = smooth(counts, smooth_width)
    # start above the noise threshold: below it the counts fall to zero, which SNIP would
    # drag into the low-energy background.  Mirror the spectrum about the start channel
    # instead (a local minimum there is left alone by the clipping).
    k0 = int(np.clip(np.ceil(calibration.channel(e_min_kev)), 0, n - 2))
    nz = np.nonzero(ys[k0:] > 0)[0]
    k0 = k0 + (int(nz[0]) if nz.size else 0)
    pad = int(min(n - k0 - 1, 2 * np.ceil(w.max())))
    seg = ys[k0:]
    ext = np.concatenate([seg[1:pad + 1][::-1], seg])
    wext = np.concatenate([w[k0 + 1:k0 + pad + 1][::-1], w[k0:]])
    bg_ext = snip(ext, wext)
    bg = np.zeros(n)
    bg[k0:] = np.clip(bg_ext[pad:], 0, None)
    return bg
