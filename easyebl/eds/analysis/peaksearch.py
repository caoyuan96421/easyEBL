"""Peak search: local maxima of the background-subtracted spectrum, refined to gaussian centroids."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spectrum import Spectrum
from .background import smooth

__all__ = ["Peak", "find_peaks", "refine_peak", "FWHM_FACTOR"]

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


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
