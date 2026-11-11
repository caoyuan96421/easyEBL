"""Spectrum fitting: element-family reference shapes + background, linear least squares.

Each element family (Fe K, Fe L, Au M …) is modelled as the sum of its lines,
each a gaussian with the detector resolution FWHM(E), with relative heights
from the X-ray generation model for the beam voltage (detector efficiency
included), plus Si escape peaks.  The background is the SNIP estimate.  The
spectrum is fitted by weighted, non-negative linear least squares, which
deconvolves overlaps (Ti Kβ / V Kα, S K / Mo L / Pb M …) as well as the
detector resolution allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import math

import numpy as np

from ... import xray
from ...xray import generation as gen
from ..spectrum import EnergyCalibration, Spectrum
from .background import snip_background

__all__ = ["FamilyRef", "family_reference", "FitResult", "fit_spectrum", "escape_fraction", "pileup_shape"]

SI_KA, SI_K_EDGE, escape_fraction = gen.SI_KA, gen.SI_K_EDGE, gen.escape_fraction


@dataclass
class FamilyRef:
    element: str
    family: str
    lines: list  # [(XrayLine, relative height)]
    shape: np.ndarray  # response (counts per unit area), sums to ≈ 1 inside the spectrum
    main_energy: float

    @property
    def label(self) -> str:
        return f"{self.element} {self.family}"


@functools.lru_cache(maxsize=4096)
def _family_lines(el: str, family: str, beam_kv: float, takeoff_deg: float, det: gen.Detector):
    """((line, relative intensity), …) of a family in the pure element (cached)."""
    cond = gen.Conditions(beam_kv=beam_kv, takeoff_deg=takeoff_deg, detector=det)
    ls = xray.lines(el, family, beam_kv=beam_kv, min_weight=1e-3)
    inten = [(ln, gen.generation_factors(ln, {el: 1.0}, cond)) for ln in ls]
    inten = [(ln, i) for ln, i in inten if i > 0]
    tot = sum(i for _, i in inten)
    return tuple((ln, i / tot) for ln, i in inten) if tot > 0 else ()


def family_reference(element, family: str, calibration: EnergyCalibration, n_channels: int,
                     beam_kv: float = 20.0, detector: gen.Detector | None = None, takeoff_deg: float = 35.0,
                     escape: bool = True, width_scale: float = 1.0, shift_kev: float = 0.0) -> FamilyRef | None:
    """Response of one family: gaussians (FWHM(E) × *width_scale*) at the line energies
    (+ *shift_kev*, a calibration correction) with escape peaks; unit area for the lines."""
    det = detector or gen.Detector()
    el = xray.symbol(element)
    inten = _family_lines(el, family, float(beam_kv), float(takeoff_deg), det)
    if not inten:
        return None
    k = np.arange(n_channels)
    e = calibration.energy(k)
    de = calibration.channel_width(k)
    shape = np.zeros(n_channels)
    for ln, rel in inten:
        comps = [(ln.energy, 1.0)]
        if escape and ln.energy > SI_K_EDGE:
            comps.append((ln.energy - SI_KA, float(escape_fraction(ln.energy))))
        for en, frac in comps:
            if frac <= 0:
                continue
            sig = float(det.fwhm_kev(max(en, 0.05))) * width_scale / 2.3548
            en = en + shift_kev
            k0 = max(0, int(np.floor(calibration.channel(en - 5 * sig))))
            k1 = min(n_channels, int(np.ceil(calibration.channel(en + 5 * sig))) + 1)
            if k1 <= k0:
                continue
            ee = e[k0:k1]
            shape[k0:k1] += rel * frac * de[k0:k1] / (sig * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((ee - en) / sig) ** 2)
    main = max(inten, key=lambda t: t[1])[0].energy
    return FamilyRef(el, family, list(inten), shape, main)


@dataclass
class FitResult:
    refs: list  # FamilyRef
    areas: np.ndarray  # net counts per family (all lines, without escape)
    sigmas: np.ndarray
    background: np.ndarray
    model: np.ndarray
    chi2_reduced: float
    e_range: tuple
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """{"Fe K": (area, sigma), …}"""
        return {r.label: (float(a), float(s)) for r, a, s in zip(self.refs, self.areas, self.sigmas)}

    def area(self, element, family: str | None = None) -> float:
        el = xray.symbol(element)
        return float(sum(a for r, a in zip(self.refs, self.areas)
                         if r.element == el and (family is None or r.family == family)))

    def residuals(self, counts) -> np.ndarray:
        c = np.asarray(counts, float)
        return (c - self.model) / np.sqrt(np.maximum(self.model, 1.0))


def _families_for(elements, beam_kv: float, e_min: float) -> list[tuple[str, str]]:
    out = []
    for el in elements:
        for fam in xray.families(el, beam_kv=beam_kv, e_min=e_min):
            out.append((xray.symbol(el), fam))
    return out


def pileup_shape(net: np.ndarray, calibration: EnergyCalibration) -> np.ndarray:
    """Shape of the sum (pile-up) spectrum: the net spectrum convolved with itself, on the
    channel axis (E₁ + E₂ = channel k₁ + k₂ + offset/gain), normalised to unit area."""
    x = np.clip(np.asarray(net, float), 0, None)
    n = x.size
    if x.sum() <= 0:
        return np.zeros(n)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(x, m) ** 2, m)[: 2 * n]
    shift = calibration.offset_kev / calibration.gain_kev  # E(k1)+E(k2) = E(k1+k2) + offset
    k = np.arange(n)
    out = np.interp(k - shift, np.arange(c.size), np.clip(c, 0, None), left=0.0, right=0.0)
    s = out.sum()
    return out / s if s > 0 else out


def fit_spectrum(spectrum: Spectrum, elements, e_range=(0.2, None), detector: gen.Detector | None = None,
                 background: np.ndarray | None = None, families: list | None = None,
                 pileup: bool = True, refine: bool = True, _ws: float = 1.0, _shift: float = 0.0) -> FitResult:
    """Fit the net line areas of *elements* (symbols) in *spectrum*.

    *families* (optional) restricts the fit to given (element, family) pairs;
    by default every family excited by the beam with lines above ``e_range[0]``.
    *refine*: also fit a global peak-width scale and energy shift (absorbs a
    slightly wrong resolution or calibration; reported in ``extras``).
    """
    y = np.asarray(spectrum.counts, dtype=float)
    n = y.size
    cal = spectrum.calibration
    e = cal.energies(n)
    det = detector or gen.Detector()
    e_lo = float(e_range[0])
    e_hi = float(e_range[1]) if e_range[1] is not None else min(spectrum.beam_kv, float(e[-1]))
    bg = snip_background(y, cal, fwhm_kev=det.fwhm_kev) if background is None else np.asarray(background, float)
    background = bg
    if refine:
        best = None
        for ws in (0.94, 0.97, 1.0, 1.03, 1.06, 1.1):
            r = fit_spectrum(spectrum, elements, e_range, detector, background, families, pileup, refine=False,
                             _ws=ws, _shift=0.0)
            if best is None or r.chi2_reduced < best[0].chi2_reduced:
                best = (r, ws)
        ws0 = best[1]
        for ws in (ws0 - 0.01, ws0 + 0.01):
            r = fit_spectrum(spectrum, elements, e_range, detector, background, families, pileup, refine=False,
                             _ws=ws, _shift=0.0)
            if r.chi2_reduced < best[0].chi2_reduced:
                best = (r, ws)
        ws0 = best[1]
        sh0 = 0.0
        for sh in (-0.012, -0.008, -0.004, 0.004, 0.008, 0.012):
            r = fit_spectrum(spectrum, elements, e_range, detector, background, families, pileup, refine=False,
                             _ws=ws0, _shift=sh)
            if r.chi2_reduced < best[0].chi2_reduced:
                best, sh0 = (r, ws0), sh
        res = best[0]
        res.extras.update(width_scale=ws0, shift_ev=sh0 * 1e3)
        return res
    fams = families or _families_for(elements, spectrum.beam_kv, e_lo)
    refs = []
    for el, fam in fams:
        r = family_reference(el, fam, cal, n, spectrum.beam_kv, det, spectrum.takeoff_deg, width_scale=_ws,
                             shift_kev=_shift)
        if r is not None and e_lo <= r.main_energy <= e_hi and r.shape[(e >= e_lo) & (e <= e_hi)].sum() > 0.2:
            refs.append(r)
    sel = (e >= e_lo) & (e <= e_hi)
    shapes = [r.shape for r in refs]
    pile = None
    if pileup and refs:
        pile = pileup_shape(np.where(e >= e_lo, y - bg, 0.0), cal)
        if pile[sel].sum() > 0.05:
            shapes.append(pile)
        else:
            pile = None
    n_lines = len(shapes)
    # background: the SNIP estimate times a smooth, piecewise-linear correction (knots ≥ 0.35 keV
    # apart – far wider than a peak), fitted with the lines.  SNIP alone sits a little low in the
    # continuum and cannot follow the low-energy hump shaped by the window absorption.
    knots = [k for k in (e_lo, 0.6, 1.0, 2.0, 3.5, 6.0, 10.0, 15.0) if e_lo <= k < e_hi] + [e_hi]
    knots = sorted(set(knots))
    for j, kc in enumerate(knots):
        hat = np.zeros(n)
        if j > 0:
            a = knots[j - 1]
            m = (e >= a) & (e <= kc)
            hat[m] = (e[m] - a) / (kc - a)
        if j + 1 < len(knots):
            b = knots[j + 1]
            m = (e >= kc) & (e <= b)
            hat[m] = (b - e[m]) / (b - kc)
        if (hat[sel] * bg[sel]).sum() > 0:
            shapes.append(hat * bg)
    nc = len(shapes)
    yy = y[sel]
    w = 1.0 / np.maximum(yy, 1.0)
    areas = np.zeros(nc)
    cov = np.zeros((nc, nc))
    for _pass in range(2):  # 2nd pass: Pearson weights from the first model (less bias than 1/y)
        active = list(range(nc))
        while active:
            A = np.column_stack([shapes[i][sel] for i in active])
            Aw = A * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(Aw, yy * np.sqrt(w), rcond=None)
            if (coef >= 0).all():
                areas[:] = 0
                areas[active] = coef
                try:
                    c = np.linalg.inv(Aw.T @ Aw)
                except np.linalg.LinAlgError:
                    c = np.linalg.pinv(Aw.T @ Aw)
                cov[:] = 0
                cov[np.ix_(active, active)] = c
                break
            active.pop(int(np.argmin(coef)))  # drop the most negative component, refit
        mod = sum(a * sh[sel] for a, sh in zip(areas, shapes))
        w = 1.0 / np.maximum(mod, 1.0)
    model = np.zeros(n)
    for a, sh in zip(areas, shapes):
        model += a * sh
    bg_fit = np.zeros(n)
    for a, sh in zip(areas[n_lines:], shapes[n_lines:]):
        bg_fit += a * sh
    bg_fit[~sel] = bg[~sel]
    model[~sel] = (bg + sum(a * sh for a, sh in zip(areas[:n_lines], shapes[:n_lines])))[~sel]
    dof = max(1, int(sel.sum()) - len(active))
    chi2 = float((((y - model)[sel]) ** 2 / np.maximum(model[sel], 1.0)).sum() / dof)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None)) * max(1.0, math.sqrt(chi2))
    nr = len(refs)
    extras = {"beam_kv": spectrum.beam_kv}
    if pile is not None:
        extras["pileup_counts"] = float(areas[nr])
    bg = bg_fit
    return FitResult(refs, areas[:nr], sig[:nr], bg, model, chi2, (e_lo, e_hi), extras)
