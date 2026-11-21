"""Synthetic EDS spectra for tests, from the X-ray generation model (no DAQ, no pulse processor).

The expected spectrum of a homogeneous material is built from
:mod:`easyebl.xray.generation` (characteristic lines + continuum), passed through
an ideal SDD (Si escape peaks, noise ⊕ Fano resolution, LLD) and Poisson-sampled.
"""
import functools

import numpy as np

from easyebl.eds import EnergyCalibration, Spectrum, SpectrumMap
from easyebl.xray import generation as gen

CHANNELS = 2048
KEV_PER_CHANNEL = 0.0102
OFFSET_KEV = -0.015
FWHM_MN_EV = 132.0
FANO = 0.115
LLD_KEV = 0.12
LIVE_FRACTION = 0.9  # nominal live/real ratio of the synthetic acquisitions


def true_calibration() -> EnergyCalibration:
    return EnergyCalibration(OFFSET_KEV + KEV_PER_CHANNEL / 2, KEV_PER_CHANNEL)


def _sigma_kev(e: np.ndarray) -> np.ndarray:
    eps = 3.85e-3
    k = (2.3548 ** 2) * FANO * eps
    noise2 = max((FWHM_MN_EV / 1e3) ** 2 - k * 5.895, 1e-6)
    return np.sqrt(noise2 + k * np.maximum(e, 0.0)) / 2.3548


@functools.lru_cache(maxsize=16)
def _rate_shape(comp_items: tuple, beam_kv: float) -> np.ndarray:
    """Expected counts per channel per detected photon (sums to ≤ 1) of composition *comp_items*."""
    comp = dict(comp_items)
    cond = gen.Conditions(beam_kv=beam_kv)
    de = 0.01
    grid = np.arange(0.1 + de / 2, beam_kv, de)
    lines = gen.characteristic(comp, cond)
    e = np.concatenate([[ln.energy for ln, _ in lines], grid])
    w = np.concatenate([[i for _, i in lines], gen.continuum(grid, comp, cond) * de])
    w = w / w.sum()
    # Si escape: a fraction of each photon is detected 1.74 keV lower
    pe = gen.escape_fraction(e)
    e = np.concatenate([e, e - 1.74])
    w = np.concatenate([w * (1 - pe), w * pe])
    keep = (e > 0) & (w > 0)
    e, w = e[keep], w[keep]
    # Gaussian resolution onto the channel grid
    edges = OFFSET_KEV + KEV_PER_CHANNEL * np.arange(CHANNELS + 1)
    from scipy.special import erf

    sig = _sigma_kev(e)
    z = (edges[None, :] - e[:, None]) / (np.sqrt(2.0) * sig[:, None])
    cdf = 0.5 * (1.0 + erf(z))
    spec = (np.diff(cdf, axis=1) * w[:, None]).sum(axis=0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    spec[centres < LLD_KEV] = 0.0
    return spec


def spectrum_of(comp: dict, seconds: float = 10.0, cps: float = 20e3, beam_kv: float = 20.0,
                seed: int = 3) -> Spectrum:
    """Spectrum of a homogeneous material *comp* (mass fractions) at *cps* detected rate."""
    rng = np.random.default_rng(seed)
    live = seconds * LIVE_FRACTION
    lam = _rate_shape(tuple(sorted(comp.items())), float(beam_kv)) * cps * live
    counts = rng.poisson(lam).astype(np.int64)
    slow = int(counts.sum())
    fast = int(round(slow / LIVE_FRACTION))
    return Spectrum(counts, true_calibration(), seconds, live, fast, slow, beam_kv)


def synthetic_map(labels: np.ndarray, comps: list, dwell_s: float = 0.05, beam_kv: float = 20.0, seed: int = 0,
                  channels: int = CHANNELS, cps: float = 20e3) -> SpectrumMap:
    """Spectrum image of a phase label map (Poisson counts from each phase's mean spectrum)."""
    rng = np.random.default_rng(seed)
    ny, nx = labels.shape
    cal = true_calibration()
    rates = [_rate_shape(tuple(sorted(c.items())), float(beam_kv))[:channels] * cps for c in comps]
    sm = SpectrumMap.empty(nx, ny, channels, cal, (0.0, 0.0, 10.0, 10.0), dwell_s, beam_kv=beam_kv)
    for j in range(ny):
        for i in range(nx):
            sm.cube[j, i] = rng.poisson(rates[labels[j, i]] * dwell_s * LIVE_FRACTION)
    sm.real_time[:] = dwell_s
    sm.live_time[:] = dwell_s * LIVE_FRACTION
    sm.fast_count[:] = sm.cube.sum(axis=2) / LIVE_FRACTION
    sm.done[:] = True
    sm.se = labels.astype(float)
    return sm
