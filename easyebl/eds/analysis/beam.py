"""Beam energy from the spectrum itself: the Duane–Hunt limit of the continuum.

Bremsstrahlung reaches up to the landing energy of the electrons, E0 (charging
lowers it below the nominal high voltage).  Near the end the continuum follows
Kramers' law times the detector efficiency,

    N(E) = k · (E0 − E)/E · ε(E)          (E < E0, zero above),

so a maximum-likelihood fit of that shape to the top of the spectrum gives E0 –
without a nominal beam energy.  Characteristic peaks inside the window are
clipped iteratively.

The estimate needs the end of the continuum inside the MCA range and a few
hundred counts near it; otherwise :func:`duane_hunt` returns ``None`` and the
caller falls back to a nominal value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...xray import generation as gen
from ..spectrum import Spectrum

__all__ = ["DuaneHunt", "duane_hunt", "estimate_beam_kv"]


@dataclass
class DuaneHunt:
    kv: float  # E0 (keV)
    sigma_kv: float  # 1σ statistical uncertainty
    counts: int  # counts in the fitted window
    window_kev: tuple  # (lo, hi) fitted


def _rebin(e: np.ndarray, c: np.ndarray, width_kev: float):
    de = float(np.median(np.diff(e)))
    k = max(1, int(round(width_kev / de)))
    n = (len(c) // k) * k
    return e[:n].reshape(-1, k).mean(axis=1), c[:n].reshape(-1, k).sum(axis=1).astype(float)


def _rough_end(eb: np.ndarray, cb: np.ndarray, min_counts: int) -> float | None:
    """Highest energy with clearly more counts above it than the flat level beyond the continuum."""
    tail = np.cumsum(cb[::-1])[::-1]
    level = max(0.3 * min_counts, 2e-4 * float(cb.sum()))
    above = np.nonzero(tail > level)[0]
    return float(eb[above[-1]]) if above.size else None


def _deviance(y: np.ndarray, m: np.ndarray) -> float:
    m = np.maximum(m, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / m), 0.0)
    return float(2.0 * np.sum(m - y + t))


def duane_hunt(spectrum: Spectrum, detector: gen.Detector | None = None, min_counts: int = 300,
               e_min_kev: float = 2.0) -> DuaneHunt | None:
    """Fit the high-energy end of the continuum; None if it is not measurable."""
    c = np.asarray(spectrum.counts, dtype=float)
    if c.size < 32 or c.sum() < 10 * min_counts:
        return None
    det = detector or gen.Detector()
    e_top = float(spectrum.energies[-1])
    eb, cb = _rebin(spectrum.energies, c, 0.1)
    ok = eb > e_min_kev
    eb, cb = eb[ok], cb[ok]
    if eb.size < 10:
        return None
    e0 = _rough_end(eb, cb, min_counts)
    if e0 is None:
        return None
    eff = np.asarray(gen.efficiency(eb, det), dtype=float)
    floor = 0.02  # counts per bin beyond the end (pile-up, noise)
    res = None
    for _ in range(4):  # the window follows E0
        lo, hi = max(e_min_kev, 0.6 * e0), min(e0 + 3.0, float(eb[-1]))
        m = (eb >= lo) & (eb <= hi)
        if m.sum() < 8:
            return None
        x, y, ef = eb[m], cb[m], eff[m]
        grid = np.arange(max(lo + 0.5, e0 - 3.0), min(hi, e0 + 3.0), 0.01)
        if grid.size < 3:
            return None
        keep = np.ones(x.size, bool)
        for _ in range(4):
            devs = np.empty(grid.size)
            for i, g in enumerate(grid):
                s = np.clip(g - x, 0.0, None) / x * ef
                k = y[keep].sum() / max(s[keep].sum(), 1e-12)  # Poisson ML scale
                devs[i] = _deviance(y[keep], k * s[keep] + floor)
            j = int(np.argmin(devs))
            s = np.clip(grid[j] - x, 0.0, None) / x * ef
            k = y[keep].sum() / max(s[keep].sum(), 1e-12)
            model = k * s + floor
            new = (y - model) < 4.0 * np.sqrt(np.maximum(model, 1.0)) + 2.0  # clip characteristic peaks
            if np.array_equal(new, keep):
                break
            keep = new
        # 1σ from the deviance curve (Δ = 1)
        inside = grid[devs <= devs[j] + 1.0]
        sig = max(0.5 * float(inside.max() - inside.min()), 0.01) if inside.size else 0.5
        at_edge = j in (0, grid.size - 1)
        res = DuaneHunt(float(grid[j]), sig, int(y[keep].sum()), (float(lo), float(hi)))
        if abs(grid[j] - e0) < 0.05:
            break
        e0 = float(grid[j])
    if res is None or res.counts < min_counts or at_edge:
        return None
    if not (e_min_kev + 1.0 < res.kv < e_top - 0.3):  # the end must lie inside the MCA range
        return None
    if res.sigma_kv > max(0.5, 0.05 * res.kv):
        return None
    return res


def estimate_beam_kv(spectrum: Spectrum, **kw) -> float | None:
    """E0 (keV) from the Duane–Hunt limit, rounded to 0.1 kV; None if not measurable."""
    r = duane_hunt(spectrum, **kw)
    return None if r is None else round(r.kv, 1)
