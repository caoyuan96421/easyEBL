"""Element maps from a spectrum image: window maps, fitted (deconvolved) maps, quant maps, phases.

* :func:`window_maps` – counts in ±1 FWHM around each element's main line,
  minus the background estimated from the sum spectrum (fast, live);
* :func:`fitted_maps` – per-pixel linear least squares with the same family
  references as the spectrum fit (separates overlaps, removes background);
* :func:`quant_maps` – mass % per pixel: fitted intensities divided by the
  matrix factors of the sum-spectrum quantification, normalised per pixel;
* :func:`phase_map` – k-means clustering of the (normalised) maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import xray
from ...xray import generation as gen
from ..spectrum import SpectrumMap
from .background import snip_background
from .fitting import family_reference, fit_spectrum
from .quant import choose_family, intensity_factors, quantify

__all__ = ["ElementMap", "window_maps", "fitted_maps", "quant_maps", "phase_map", "bin_map", "PhaseResult"]


@dataclass
class ElementMap:
    label: str  # "Fe Kα" or "Fe" (quant)
    data: np.ndarray  # (ny, nx)
    unit: str  # counts | wt%
    energy_kev: float = 0.0


def bin_map(sm: SpectrumMap, factor: int) -> SpectrumMap:
    """Sum *factor*×*factor* pixel blocks (more counts per pixel, lower resolution)."""
    if factor <= 1:
        return sm
    ny, nx = sm.shape
    by, bx = ny // factor, nx // factor
    if by == 0 or bx == 0:
        raise ValueError("binning larger than the map")

    def b(a, fn=np.sum):
        a = a[:by * factor, :bx * factor]
        return fn(fn(a.reshape(by, factor, bx, factor, *a.shape[2:]), axis=3), axis=1)

    cube = b(sm.cube.astype(np.int64))
    cx, cy, w, h = sm.region_um
    w2, h2 = w * bx * factor / nx, h * by * factor / ny
    region = (cx - w / 2 + w2 / 2, cy + h / 2 - h2 / 2, w2, h2)
    se = None if sm.se is None else b(np.nan_to_num(sm.se), np.mean)
    done = None if sm.done is None else b(sm.done.astype(int)) == factor * factor
    return SpectrumMap(cube, sm.calibration, region, sm.dwell_s * factor * factor, b(sm.real_time), b(sm.live_time),
                       b(sm.fast_count), se, done, sm.beam_kv, sm.takeoff_deg, sm.timestamp,
                       dict(sm.metadata, binned=factor))


def _main_family(el, kv):
    return choose_family(el, kv) or (xray.families(el, kv) or [None])[0]


def window_maps(sm: SpectrumMap, elements, detector: gen.Detector | None = None) -> list[ElementMap]:
    det = detector or gen.Detector()
    s = sm.sum_spectrum()
    bg_sum = snip_background(s.counts, sm.calibration, fwhm_kev=det.fwhm_kev)
    tot_sum = max(float(s.counts.sum()), 1.0)
    pix_tot = sm.cube.sum(axis=2, dtype=np.int64).astype(float)
    out = []
    for el in elements:
        el = xray.symbol(el)
        fam = _main_family(el, sm.beam_kv)
        if fam is None:
            continue
        ln = xray.main_line(el, fam)
        fw = float(det.fwhm_kev(ln.energy))
        img = sm.window_image(ln.energy - fw, ln.energy + fw)
        k0 = max(0, int(np.ceil(sm.calibration.channel(ln.energy - fw) - 0.5)))
        k1 = min(sm.n_channels, int(np.floor(sm.calibration.channel(ln.energy + fw) + 0.5)))
        bg_frac = float(bg_sum[k0:k1].sum()) / tot_sum
        out.append(ElementMap(f"{el} {fam}α", img - bg_frac * pix_tot, "counts", ln.energy))
    return out


def fitted_maps(sm: SpectrumMap, elements, detector: gen.Detector | None = None, e_range=(0.2, None)):
    """(maps, fit of the sum spectrum): net counts per family and pixel by linear least squares."""
    det = detector or gen.Detector()
    s = sm.sum_spectrum()
    fit = fit_spectrum(s, elements, e_range=e_range, detector=det)
    n = sm.n_channels
    e = sm.energies
    lo, hi = fit.e_range
    sel = (e >= lo) & (e <= hi)
    bg_shape = fit.background / max(fit.background[sel].sum(), 1e-9)
    refs = fit.refs
    A = np.column_stack([r.shape for r in refs] + [bg_shape])[sel]
    # weights from the mean pixel model (stable for low-count pixels)
    npx = max(int(sm.done.sum()) if sm.done is not None else int(np.prod(sm.shape)), 1)
    wvec = 1.0 / np.maximum(fit.model[sel] / npx, 1e-3)
    Aw = A * wvec[:, None]
    M = np.linalg.pinv(A.T @ Aw) @ Aw.T  # (k+1, nsel)
    cube = sm.cube.reshape(-1, n)[:, sel].astype(float)
    coef = cube @ M.T  # (npix, k+1)
    ny, nx = sm.shape
    maps = []
    for i, r in enumerate(refs):
        maps.append(ElementMap(f"{r.element} {r.family}", coef[:, i].reshape(ny, nx), "counts", r.main_energy))
    return maps, fit


def quant_maps(sm: SpectrumMap, elements, detector: gen.Detector | None = None, oxygen_by_stoichiometry=False,
               smooth_px: int = 0) -> tuple[list[ElementMap], object]:
    """Mass % maps.  Matrix factors come from the sum-spectrum quantification (exact for one
    phase, approximate for mixtures – good enough to see where each element is and roughly how much)."""
    det = detector or gen.Detector()
    maps, fit = fitted_maps(sm, elements, det)
    q = quantify(fit, sm.beam_kv, sm.takeoff_deg, det, oxygen_by_stoichiometry=oxygen_by_stoichiometry)
    comp = {r.element: r.wt / 100.0 for r in q.rows if r.wt > 0}
    cond = gen.Conditions(beam_kv=sm.beam_kv, takeoff_deg=sm.takeoff_deg, detector=det)
    fams = {r.element: r.family for r in q.rows if not r.by_stoichiometry}
    g = intensity_factors(fams, comp, cond) if comp else {}
    by_label = {m.label: m for m in maps}
    conc = {}
    for el, fam in fams.items():
        m = by_label.get(f"{el} {fam}")
        if m is None or g.get(el, 0) <= 0:
            continue
        d = np.clip(m.data, 0, None)
        if smooth_px > 0:
            from .background import smooth

            d = np.apply_along_axis(smooth, 0, np.apply_along_axis(smooth, 1, d, 2 * smooth_px + 1), 2 * smooth_px + 1)
        conc[el] = d / g[el]
    if oxygen_by_stoichiometry and conc:
        from .quant import VALENCE

        a_o = xray.atomic_mass("O")
        conc["O"] = sum(c / xray.atomic_mass(el) * VALENCE.get(el, 0) / 2 * a_o for el, c in conc.items())
    tot = sum(conc.values()) if conc else None
    out = []
    for el, c in conc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(ElementMap(el, np.where(tot > 0, 100.0 * c / tot, np.nan), "wt%"))
    return out, q


@dataclass
class PhaseResult:
    labels: np.ndarray  # (ny, nx) int, -1 = not assigned
    centroids: np.ndarray  # (k, n_elements) in the units of the input maps
    elements: list
    fractions: np.ndarray  # area fraction per phase


def phase_map(maps: list[ElementMap], k: int = 4, iterations: int = 50, seed: int = 0,
              min_total: float | None = None) -> PhaseResult:
    """k-means clustering of element maps into *k* phases (pixels as composition vectors)."""
    X = np.stack([np.nan_to_num(m.data) for m in maps], axis=-1).astype(float)
    ny, nx, d = X.shape
    X = X.reshape(-1, d)
    valid = np.isfinite(X).all(axis=1)
    if min_total is not None:
        valid &= X.sum(axis=1) >= min_total
    Xv = X[valid]
    scale = Xv.std(axis=0) + 1e-12
    Z = Xv / scale
    rng = np.random.default_rng(seed)
    k = max(1, min(int(k), len(Z)))
    # k-means++ initialisation
    cent = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d2 = np.min([((Z - c) ** 2).sum(axis=1) for c in cent], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(Z[rng.choice(len(Z), p=p)])
    C = np.array(cent)
    lab = np.zeros(len(Z), int)
    for _ in range(iterations):
        dist = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = Z[lab == j].mean(axis=0)
    labels = np.full(ny * nx, -1)
    labels[valid] = lab
    fr = np.array([(lab == j).mean() for j in range(k)])
    order = np.argsort(-fr)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    labels[valid] = remap[lab]
    return PhaseResult(labels.reshape(ny, nx), (C * scale)[order], [m.label for m in maps], fr[order])
