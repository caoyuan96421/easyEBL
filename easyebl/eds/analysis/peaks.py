"""Automatic element identification (AutoID).

1. find significant peaks (net of the SNIP background);
2. explain the strongest unexplained peak: candidate element families with a
   line within the detector resolution are scored by how well the *other*
   lines that element must show (other lines of the family, other families)
   are actually present – a candidate that predicts strong peaks where there
   are none is rejected; common elements get a small prior;
3. the winner explains every peak near its lines; Si escape peaks and sum
   peaks of strong lines are explained too; repeat;
4. a final fit drops elements whose fitted area is not significant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import xray
from ...xray import generation as gen
from ..spectrum import Spectrum
from .background import snip_background
from .calibration import find_peaks
from .fitting import SI_KA, fit_spectrum

__all__ = ["Identification", "identify_elements", "COMMON_ELEMENTS", "EXCLUDED_ELEMENTS"]

COMMON_ELEMENTS = frozenset(
    "C N O F Na Mg Al Si P S Cl K Ca Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Zr Nb Mo Ag In Sn Sb Ba La Ce Hf Ta W "
    "Pt Au Pb Bi".split())
#: never proposed automatically (noble gases, radioactive, H–Be: no usable lines on an SDD)
EXCLUDED_ELEMENTS = frozenset("H He Li Be Ne Ar Kr Xe Rn Tc Pm Po At Fr Ra Ac Pa Np Pu".split())


@dataclass
class Identification:
    element: str
    families: list
    score: float
    peaks_kev: list
    fitted_area: float = 0.0
    significance: float = 0.0


def _window_net(net: np.ndarray, cal, e: float, fwhm: float) -> float:
    k0 = int(max(0, np.floor(cal.channel(e - fwhm))))
    k1 = int(min(net.size, np.ceil(cal.channel(e + fwhm)) + 1))
    return float(net[k0:k1].sum()) if k1 > k0 else 0.0


def residual_peaks(counts: np.ndarray, model: np.ndarray, cal, det: gen.Detector, z_min: float = 6.0,
                   e_min_kev: float = 0.25, e_max_kev: float | None = None):
    """Peaks the model misses: local maxima of the excess (counts − model) summed over ±FWHM/2,
    in units of its Poisson σ.  Returns [(energy, Peak)]."""
    from .calibration import Peak

    y = np.asarray(counts, float)
    m = np.maximum(np.asarray(model, float), 1.0)
    n = y.size
    e = cal.energies(n)
    fw_ch = det.fwhm_kev(np.clip(e, 0.05, None)) / cal.channel_width(np.arange(n))
    h = np.maximum(1, np.round(fw_ch / 2).astype(int))
    d = y - m
    cd = np.concatenate([[0.0], np.cumsum(d)])
    cm = np.concatenate([[0.0], np.cumsum(m)])
    k = np.arange(n)
    lo, hi = np.clip(k - h, 0, n), np.clip(k + h + 1, 0, n)
    exc = cd[hi] - cd[lo]
    z = exc / np.sqrt(cm[hi] - cm[lo])
    e_max = e_max_kev if e_max_kev is not None else e[-1]
    ok = (e >= e_min_kev) & (e <= e_max)
    out = []
    cand = np.nonzero(ok[1:-1] & (z[1:-1] > z_min) & (z[1:-1] >= z[:-2]) & (z[1:-1] >= z[2:]))[0] + 1
    for c in cand[np.argsort(-z[cand])]:
        if any(abs(c - q.channel) < 2 * fw_ch[c] for _, q in out):
            continue
        w = np.clip(d[lo[c]:hi[c]], 0, None)
        cen = float((w * np.arange(lo[c], hi[c])).sum() / w.sum()) if w.sum() > 0 else float(c)
        out.append((float(cal.energy(cen)), Peak(cen, fw_ch[c] / 2.3548, float(d[c]), float(exc[c]), float(z[c]))))
    return out


def identify_elements(spectrum: Spectrum, min_significance: float = 5.0, detector: gen.Detector | None = None,
                      allowed=None, exclude=(), always=(), max_elements: int = 15,
                      e_min_kev: float = 0.25, passes: int = 3, min_fit_significance: float = 8.0
                      ) -> list[Identification]:
    """Elements present in *spectrum* (strongest first)."""
    det = detector or gen.Detector()
    cal = spectrum.calibration
    y = np.asarray(spectrum.counts, float)
    kv = spectrum.beam_kv
    bg = snip_background(y, cal, fwhm_kev=det.fwhm_kev)
    excl = set(EXCLUDED_ELEMENTS) | {xray.symbol(e) for e in exclude}
    allow = None if allowed is None else {xray.symbol(e) for e in allowed}
    cond = gen.Conditions(beam_kv=kv, takeoff_deg=spectrum.takeoff_deg, detector=det)
    always_s = {xray.symbol(e) for e in always}
    found: dict[str, Identification] = {}
    for s in always_s:
        found[s] = Identification(s, xray.families(s, kv, e_min=e_min_kev), 0.0, [])
    tol = lambda e: 0.6 * float(det.fwhm_kev(e)) + 0.015  # noqa: E731

    def predicted(el: str):
        comp = {el: 1.0}
        out = []
        for f in xray.families(el, kv, e_min=e_min_kev):
            for ln in xray.lines(el, f, kv, min_weight=0.02):
                g = gen.generation_factors(ln, comp, cond)
                if g > 0:
                    out.append((ln, g))
        return out

    def score(el: str, e_peak: float, area: float, net: np.ndarray) -> float:
        """Consistency of 'this peak belongs to el': its other strong lines must be there."""
        pred = predicted(el)
        ref = sum(g for ln, g in pred if abs(ln.energy - e_peak) < tol(e_peak))
        if ref <= 0:
            return -np.inf
        groups: dict = {}
        for ln, g in pred:  # lines closer than half the resolution form one peak
            key = round(ln.energy / (0.5 * float(det.fwhm_kev(ln.energy))))
            gsum, en = groups.get(key, (0.0, ln.energy))
            groups[key] = (gsum + g, en)
        sc = 0.0
        for g, en in groups.values():
            if abs(en - e_peak) < tol(e_peak) or en < e_min_kev or en > kv:
                continue
            expected = area * g / ref
            fwe = float(det.fwhm_kev(en))
            obs = _window_net(net, cal, en, fwe)
            b = _window_net(bg, cal, en, fwe)
            sd = np.sqrt(max(b, 1.0) + max(expected, 0.0))
            if expected > 4 * sd:  # a line that must be visible
                ratio = obs / expected
                sc += 1.0 if 0.4 < ratio < 2.5 else (-3.0 if ratio < 0.15 else -1.0)
        return sc + (0.6 if el in COMMON_ELEMENTS else 0.0)

    def explain(pk, net, strong, residual=False):
        """One identification pass over peaks *pk* [(E, Peak)]."""
        explained = [False] * len(pk)
        for _ in range(4 * max_elements):
            todo = [i for i, x in enumerate(explained) if not x]
            if not todo or len(found) >= max_elements:
                return
            i = todo[0]
            explained[i] = True
            e_peak, p = pk[i]
            t = tol(e_peak)
            # artefacts of strong peaks: Si escape (E − 1.74 keV) and sum peaks (E₁ + E₂)
            if any(abs(e2 - SI_KA - e_peak) < t and a2 > 15 * p.area for e2, a2 in strong):
                continue
            if any(abs(e1 + e2 - e_peak) < t and p.area < 0.05 * min(a1, a2)
                   for k1, (e1, a1) in enumerate(strong) for e2, a2 in strong[k1:]):
                continue
            # the flank of a much stronger peak (a slightly wrong width leaves a residual there)
            if any(0.3 * t < abs(e2 - e_peak) < 2.5 * t and a2 > 20 * p.area for e2, a2 in strong):
                continue
            # an element already found that has a line here (not in the residual: a residual peak
            # is precisely what the elements found so far do not explain)
            if not residual and any(abs(ln.energy - e_peak) < t for el in found for ln, _ in predicted(el)):
                continue
            best = None
            seen = set()
            for ln in xray.lines_near(e_peak, tol=t + 0.005, beam_kv=kv, min_weight=0.2):
                if ln.element in excl or (allow is not None and ln.element not in allow) or ln.energy < e_min_kev:
                    continue
                if ln.element in seen or ln.element in found:
                    continue
                seen.add(ln.element)
                sc = score(ln.element, e_peak, p.area, net) - 2.0 * abs(ln.energy - e_peak) / t + 0.5 * ln.weight
                if best is None or sc > best[0]:
                    best = (sc, ln)
            if best is None or best[0] < -2.5:
                continue  # unidentified
            el = best[1].element
            found[el] = Identification(el, xray.families(el, kv, e_min=e_min_kev), float(best[0]), [e_peak])
            for j, (e2, _p2) in enumerate(pk):
                if not explained[j] and any(abs(ln.energy - e2) < tol(e2) for ln, _ in predicted(el)):
                    explained[j] = True
                    found[el].peaks_kev.append(e2)

    peaks = find_peaks(spectrum, min_significance=min_significance, e_min_kev=e_min_kev)
    strong = [(float(cal.energy(p.channel)), p.area) for p in peaks[:8]]
    explain([(float(cal.energy(p.channel)), p) for p in peaks], y - bg, strong)
    fit = None
    for _ in range(max(0, passes - 1)):
        if not found:
            break
        fit = fit_spectrum(spectrum, list(found), e_range=(e_min_kev, None), detector=det, background=bg)
        rp = residual_peaks(y, fit.model, cal, det, z_min=max(6.0, min_significance + 1.0) * max(1.0, np.sqrt(fit.chi2_reduced)),
                            e_min_kev=e_min_kev, e_max_kev=kv)
        if not rp:
            break
        n0 = len(found)
        explain(rp, y - fit.model, strong, residual=True)
        if len(found) == n0:
            break
    if not found:
        return []
    fit = fit_spectrum(spectrum, list(found), e_range=(e_min_kev, None), detector=det, background=bg)
    from .quant import choose_family

    out = []
    for el, ident in found.items():
        fam_a = {r.family: (float(a), float(sg)) for r, a, sg in zip(fit.refs, fit.areas, fit.sigmas)
                 if r.element == el}
        ident.fitted_area = sum(a for a, _ in fam_a.values())
        main = choose_family(el, kv, available=list(fam_a)) if fam_a else None
        if main is None:
            ident.significance = 0.0
        else:
            a, sg = fam_a[main]
            ident.significance = a / sg if sg > 0 else 0.0
            # another family far stronger than the main one predicts (matrix absorption only
            # weakens soft lines) means the "element" is a residual of some other peak
            g = {f: gen.family_intensity(el, f, {el: 1.0}, cond) for f in fam_a}
            for f, (a2, s2) in fam_a.items():
                if f == main or g.get(f, 0) <= 0 or g.get(main, 0) <= 0:
                    continue
                exp2 = a * g[f] / g[main]
                if a2 > 3 * exp2 + 5 * s2:
                    ident.significance = 0.0
        if el in always_s or ident.significance >= min_fit_significance:
            out.append(ident)
    return sorted(out, key=lambda d: -d.fitted_area)
