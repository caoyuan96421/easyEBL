"""Energy calibration of the MCA: peak search, centroids and the channel → keV fit.

Workflow (what the calibration dialog does):

1. acquire a spectrum of a known standard (e.g. Cu: Cu L ≈ 0.93 and Cu Kα 8.04
   keV; or Al + Cu; Mn Kα 5.895 keV from a ⁵⁵Fe source);
2. :func:`find_peaks` lists the peaks (channel centroid, width, counts);
3. each chosen peak is assigned a line energy (by hand, or
   :func:`auto_assign` matches the expected lines to the peaks using the
   current calibration as a guide, falling back to the two strongest peaks);
4. :func:`fit_calibration` fits ``E = offset + gain·ch (+ quad·ch²)`` by
   weighted least squares and reports residuals; :func:`fit_resolution`
   estimates the detector FWHM at Mn Kα and the Fano factor from the widths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import xray
from ..spectrum import EnergyCalibration, Spectrum
from .background import smooth, snip

__all__ = ["Peak", "find_peaks", "refine_peak", "fit_calibration", "CalibrationFit", "auto_assign",
           "fit_resolution", "STANDARDS", "standard_lines"]

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

#: Calibration standards: name → line labels (element + line, e.g. "Cu Ka1") with their energies.
STANDARDS = {
    "Cu (L + Kα)": ["Cu La1", "Cu Ka1"],
    "Al + Cu": ["Al Ka1", "Cu La1", "Cu Ka1"],
    "Mn / ⁵⁵Fe source (Kα, Kβ)": ["Mn Ka1", "Mn Kb1"],
    "Si + Cu": ["Si Ka1", "Cu Ka1"],
    "Ti (L + Kα + Kβ)": ["Ti La1", "Ti Ka1", "Ti Kb1"],
    "Ni (L + Kα)": ["Ni La1", "Ni Ka1"],
    "Au (M + Lα)": ["Au Ma", "Au La1"],
    "Phantom specimen (Al, Si, Cu)": ["Cu La1", "Al Ka1", "Si Ka1", "Cu Ka1"],
}


def line_energy(label: str, blend_kev: float = 0.05) -> float:
    """Energy (keV) of the peak a line label ("Cu Ka1") produces in an SDD spectrum: the
    weighted mean of the lines of the same family within ±*blend_kev* (Kα1+Kα2, Lα+Lβ1 …)."""
    el, name = label.split()
    ls = [ln for ln in xray.lines(el) if ln.name == name]
    if not ls:
        raise ValueError(f"unknown line '{label}'")
    ln = ls[0]
    near = [l2 for l2 in xray.lines(el, ln.family) if abs(l2.energy - ln.energy) <= blend_kev]
    w = np.array([l2.weight for l2 in near])
    return float(np.dot(w, [l2.energy for l2 in near]) / w.sum()) if w.sum() > 0 else ln.energy


def standard_lines(name: str) -> list[tuple[str, float]]:
    return [(lab, line_energy(lab)) for lab in STANDARDS[name]]


@dataclass
class Peak:
    channel: float  # centroid (fractional channel)
    sigma_ch: float  # gaussian σ in channels
    height: float  # net counts at the maximum
    area: float  # net counts
    significance: float  # area / sqrt(background + area)
    channel_err: float = 0.0  # 1σ uncertainty of the centroid

    @property
    def fwhm_ch(self) -> float:
        return self.sigma_ch * FWHM_FACTOR


def refine_peak(counts: np.ndarray, guess_ch: float, sigma_ch: float, background: np.ndarray | None = None,
                iterations: int = 6) -> Peak | None:
    """Gaussian centroid of the peak near *guess_ch* (iterative ±2σ moments on net counts)."""
    y = np.asarray(counts, dtype=float)
    bg = np.zeros_like(y) if background is None else np.asarray(background, dtype=float)
    net = y - bg
    c, s = float(guess_ch), max(float(sigma_ch), 0.7)
    n = y.size
    for _ in range(iterations):
        lo, hi = int(max(0, np.floor(c - 2.0 * s))), int(min(n, np.ceil(c + 2.0 * s) + 1))
        if hi - lo < 3:
            return None
        k = np.arange(lo, hi)
        w = np.clip(net[lo:hi], 0, None)
        tot = w.sum()
        if tot <= 0:
            return None
        c_new = float((w * k).sum() / tot)
        # the ±2σ window truncates the gaussian: its second moment is 0.774 σ²
        var = float((w * (k - c_new) ** 2).sum() / tot) / 0.774
        s_new = float(np.clip(np.sqrt(max(var, 0.25)), 0.5, 50.0))
        done = abs(c_new - c) < 1e-3 and abs(s_new - s) < 1e-3
        c, s = c_new, s_new
        if done:
            break
    lo, hi = int(max(0, np.floor(c - 3 * s))), int(min(n, np.ceil(c + 3 * s) + 1))
    area = float(np.clip(net[lo:hi], 0, None).sum())
    b = float(bg[lo:hi].sum())
    if area <= 0:
        return None
    height = float(net[int(np.clip(round(c), 0, n - 1))])
    sig = area / np.sqrt(b + area)
    err = s / np.sqrt(area) * np.sqrt(1.0 + b / area)
    return Peak(c, s, height, area, sig, err)


def find_peaks(spectrum: Spectrum, min_significance: float = 5.0, fwhm_kev: float | None = None,
               e_min_kev: float = 0.15, max_peaks: int = 40) -> list[Peak]:
    """Peaks of a spectrum: local maxima of the smoothed net counts, refined, strongest first."""
    y = np.asarray(spectrum.counts, dtype=float)
    cal = spectrum.calibration
    n = y.size
    e = cal.energies(n)
    from ...xray.generation import Detector

    det = Detector()
    fwhm_e = det.fwhm_kev(np.clip(e, 0.05, None)) if fwhm_kev is None else np.full(n, float(fwhm_kev))
    fw_ch = fwhm_e / cal.channel_width(np.arange(n))
    from .background import snip_background

    bg = snip_background(y, cal, fwhm_kev=(det.fwhm_kev if fwhm_kev is None else float(fwhm_kev)))
    net_s = smooth(y - bg, int(max(3, np.median(fw_ch) // 2)))
    peaks: list[Peak] = []
    cand = np.nonzero((net_s[1:-1] > net_s[:-2]) & (net_s[1:-1] >= net_s[2:]) & (net_s[1:-1] > 0))[0] + 1
    cand = cand[e[cand] >= e_min_kev]
    for k in cand[np.argsort(-net_s[cand])]:
        p = refine_peak(y, float(k), fw_ch[k] / FWHM_FACTOR, bg)
        if p is None or p.significance < min_significance:
            continue
        if any(abs(p.channel - q.channel) < 1.2 * max(p.fwhm_ch, q.fwhm_ch) for q in peaks):
            continue  # shoulder of a stronger peak
        peaks.append(p)
        if len(peaks) >= max_peaks:
            break
    return sorted(peaks, key=lambda p: -p.area)


@dataclass
class CalibrationFit:
    calibration: EnergyCalibration
    channels: np.ndarray
    energies: np.ndarray
    residuals_ev: np.ndarray
    labels: list

    @property
    def rms_ev(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_ev ** 2))) if self.residuals_ev.size else 0.0


def fit_calibration(channels, energies, order: int = 1, weights=None, labels=None, name: str = "") -> CalibrationFit:
    """Least-squares fit E(ch) (order 1: offset + gain; 2: + quad).  One point fixes only the gain
    (offset kept 0)."""
    ch = np.asarray(channels, dtype=float)
    en = np.asarray(energies, dtype=float)
    if ch.size != en.size or ch.size == 0:
        raise ValueError("need matching channels and energies")
    if ch.size == 1:
        if ch[0] <= 0:
            raise ValueError("a single calibration point must be at a positive channel")
        cal = EnergyCalibration(0.0, float(en[0] / ch[0]), 0.0, name)
    else:
        order = min(int(order), ch.size - 1, 2)
        w = np.ones_like(ch) if weights is None else np.asarray(weights, dtype=float)
        A = np.column_stack([ch ** k for k in range(order + 1)])
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], en * sw, rcond=None)
        coef = list(coef) + [0.0] * (3 - len(coef))
        cal = EnergyCalibration(float(coef[0]), float(coef[1]), float(coef[2]), name)
    res = (en - cal.energy(ch)) * 1e3
    return CalibrationFit(cal, ch, en, np.atleast_1d(res), list(labels or [""] * ch.size))


def auto_assign(spectrum: Spectrum, lines: list[tuple[str, float]], tolerance_kev: float = 0.25,
                peaks: list[Peak] | None = None) -> list[tuple[str, float, Peak]]:
    """Match expected (label, energy) lines to found peaks.

    First by the spectrum's current calibration (nearest peak within
    *tolerance_kev*); if that matches fewer than two lines, by rank – the
    expected lines and the strongest peaks, both sorted by energy (works for
    a badly mis-calibrated MCA as long as the standard's lines dominate).
    """
    peaks = find_peaks(spectrum) if peaks is None else peaks
    if not peaks:
        return []
    cal = spectrum.calibration
    used = set()
    out = []
    for lab, en in sorted(lines, key=lambda t: t[1]):
        best = None
        for i, p in enumerate(peaks):
            if i in used:
                continue
            d = abs(cal.energy(p.channel) - en)
            if d <= tolerance_kev and (best is None or d < best[0] or
                                       (abs(d - best[0]) < 0.02 and p.area > peaks[best[1]].area)):
                best = (d, i)
        if best is not None:
            used.add(best[1])
            out.append((lab, en, peaks[best[1]]))
    if len(out) >= min(2, len(lines)):
        return out
    return _pattern_assign(peaks, lines, spectrum.calibration)


def _pattern_assign(peaks: list, lines: list, cal: EnergyCalibration) -> list:
    """Assignment for a badly calibrated MCA: try every (line, peak) pair as an anchor of a
    gain (offset ≈ 0, then refined with a second pair) and keep the hypothesis under which the
    most expected lines – weighted by peak significance – land on found peaks."""
    if len(peaks) == 0 or len(lines) == 0:
        return []
    exp = sorted(lines, key=lambda t: t[1])
    best = (0.0, [])
    cand = peaks[:12]
    for a_lab, a_e in exp:
        for pa in cand:
            for b_lab, b_e in exp:
                if b_e <= a_e:
                    continue
                for pb in cand:
                    if pb.channel <= pa.channel:
                        continue
                    g = (b_e - a_e) / (pb.channel - pa.channel)
                    off = a_e - g * pa.channel
                    if not (0.2 * cal.gain_kev < g < 5 * cal.gain_kev) or abs(off) > 0.3:
                        continue
                    used, pairs, score = set(), [], 0.0
                    for lab, e in exp:
                        k = (e - off) / g
                        tol = max(2.5, 0.06 / g)  # channels: ±60 eV or 2.5 ch
                        near = [(abs(p.channel - k), n) for n, p in enumerate(cand)
                                if n not in used and abs(p.channel - k) < tol]
                        if near:
                            n = min(near)[1]
                            used.add(n)
                            pairs.append((lab, e, cand[n]))
                            score += np.log1p(cand[n].significance)
                    if len(pairs) >= 2 and score > best[0]:
                        best = (score, pairs)
    return best[1]


def fit_resolution(fit_points: list[tuple[float, float]]) -> tuple[float, float]:
    """(FWHM at Mn Kα in eV, Fano factor) from [(energy keV, FWHM keV)] of single, clean peaks.

    FWHM² = noise² + 2.355²·F·ε·E (ε = 3.85 eV per e-h pair in Si); with one
    point the Fano factor is fixed at 0.115.
    """
    e = np.array([p[0] for p in fit_points], float)
    f2 = np.array([p[1] for p in fit_points], float) ** 2
    eps = 3.85e-3
    k = FWHM_FACTOR ** 2 * eps
    if e.size >= 2 and np.ptp(e) > 1.0:
        A = np.column_stack([np.ones_like(e), e])
        (a, b), *_ = np.linalg.lstsq(A, f2, rcond=None)
        fano = float(np.clip(b / k, 0.05, 0.3))
        noise2 = max(float(a), 1e-6)
    else:
        fano = 0.115
        noise2 = max(float(np.mean(f2 - k * fano * e)), 1e-6)
    return float(np.sqrt(noise2 + k * fano * 5.895) * 1e3), fano
