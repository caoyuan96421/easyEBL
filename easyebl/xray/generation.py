"""A compact EDS X-ray generation model (the "ZA" part of ZAF, plus the detector).

Used twice, deliberately with the same physics:

* by the **EDS phantom**, to turn a phase composition into the photon spectrum
  that the simulated SDD sees;
* by **standardless quantification**, to turn measured net line intensities
  back into mass fractions.

Model (per electron, into the detector's solid angle, arbitrary but fixed units)::

    I(line) = K_CHAR · (C/A) · n_s·ω_s·b · R/S · Q(U0) · f(χ) · ε(E)
    Q(U0)   = U0·ln U0 − U0 + 1            (∫ σ_Bethe/(dE/ds) dE with σ ∝ ln U/(U·E_c²), dE/ds ∝ S/E)
    S       = Σ C_j Z_j/A_j · ln(1.166 E_m / J_j),   E_m = (E0 + E_c)/2   (stopping power)
    R       = 1 − η̄ · 0.6 (1 − 1/U0)      (backscatter loss; η from Reuter's fit)
    f(χ)    = Philibert absorption, χ = µ/ρ(E)·cosec(take-off)
    ε(E)    = detector efficiency (window, dead layer, Si thickness)

    continuum dN/dE = K_CONT · Z̄ · (E0 − E)/E · f(χ_E) · ε(E)   (Kramers)

Secondary fluorescence and the Coster–Kronig transitions are ignored, so
absolute results are **semi-quantitative** (typically ±5–20 % relative for
major elements, worse for light elements and strong overlaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from . import XrayLine, atomic_mass, lines, mac, mac_line, normalize_composition, z_of

__all__ = ["Detector", "Conditions", "characteristic", "continuum", "efficiency", "family_intensity",
           "generation_factors", "escape_fraction", "K_CHAR", "K_CONT", "SI_KA", "SI_K_EDGE"]

SI_KA = 1.740  # keV, Si Kα (escape peaks appear this much below the parent line)
SI_K_EDGE = 1.839

K_CHAR = 0.0356  # pure Cu Kα at 20 kV ≈ 1e-4 photons / electron / sr
K_CONT = 3.0e-7  # Kramers constant in the same units (gives P/B ≈ 50 for Cu Kα at 20 kV)

# window materials (mass fractions, density g/cm³)
_POLYMER = ({6: 0.75, 1: 0.05, 8: 0.15, 7: 0.05}, 1.4)


@dataclass(frozen=True)
class Detector:
    """SDD efficiency model.  ``window``: "AP3.3" (polymer + Al on a Si grid), "Be8" or "none"."""

    window: str = "AP3.3"
    si_thickness_um: float = 450.0
    dead_layer_nm: float = 50.0
    fwhm_mn_ev: float = 130.0  # resolution at Mn Kα (5.895 keV)
    fano: float = 0.115

    def efficiency(self, energy) -> np.ndarray:
        return efficiency(energy, self)

    def fwhm_kev(self, energy) -> np.ndarray:
        """Energy resolution (FWHM, keV) at *energy*: electronic noise ⊕ Fano statistics."""
        e = np.asarray(energy, dtype=float)
        eps = 3.85e-3  # keV per electron-hole pair in Si
        k = (2.3548 ** 2) * self.fano * eps
        noise2 = max((self.fwhm_mn_ev / 1e3) ** 2 - k * 5.895, 1e-6)
        return np.sqrt(noise2 + k * np.clip(e, 0, None))


@dataclass(frozen=True)
class Conditions:
    beam_kv: float = 20.0
    takeoff_deg: float = 35.0
    detector: Detector = field(default_factory=Detector)


def _transmission(e, comp: dict, density: float, thickness_cm: float):
    mu = sum(w * mac(e, z) for z, w in comp.items())
    return np.exp(-mu * density * thickness_cm)


def efficiency(energy, det: Detector | None = None) -> np.ndarray:
    """Fraction of photons of *energy* (keV) absorbed in the active Si."""
    det = det or Detector()
    e = np.atleast_1d(np.asarray(energy, dtype=float))
    t = np.ones_like(e)
    if det.window == "AP3.3":
        t = t * _transmission(e, *_POLYMER, 300e-7) * _transmission(e, {13: 1.0}, 2.70, 40e-7)
        t = t * 0.77  # 23 % of the area is covered by the Si support grid
    elif det.window == "Be8":
        t = t * _transmission(e, {4: 1.0}, 1.85, 8e-4)
    elif det.window != "none":
        raise ValueError(f"unknown window '{det.window}'")
    t = t * _transmission(e, {14: 1.0}, 2.33, det.dead_layer_nm * 1e-7)
    absorbed = 1.0 - _transmission(e, {14: 1.0}, 2.33, det.si_thickness_um * 1e-4)
    out = t * absorbed
    out[e <= 0] = 0.0
    return out if np.ndim(energy) else float(out[0])


# ------------------------------------------------------------------ physics pieces
def _eta(z: float) -> float:
    """Backscatter coefficient (Reuter 1972 fit)."""
    return -0.0254 + 0.016 * z - 1.86e-4 * z * z + 8.3e-7 * z ** 3


def _j_kev(z: int) -> float:
    """Mean ionisation potential (Berger & Seltzer), keV."""
    return (9.76 * z + 58.8 * z ** -0.19) * 1e-3


def _philibert(chi: np.ndarray, e0: float, ec: float, a_mean: float, z_mean: float) -> np.ndarray:
    sigma = 4.5e5 / np.maximum(e0 ** 1.65 - np.asarray(ec, dtype=float) ** 1.65, 1e-3)
    h = 1.2 * a_mean / (z_mean * z_mean)
    x = chi / sigma
    return 1.0 / ((1.0 + x) * (1.0 + h / (1.0 + h) * x))


def _means(c: dict) -> tuple[float, float]:
    z_mean = sum(w * z for z, w in c.items())
    a_mean = sum(w * atomic_mass(z) for z, w in c.items())
    return z_mean, a_mean


def generation_factors(line: XrayLine, comp: dict, cond: Conditions) -> float:
    """Measured intensity per unit mass fraction of *line*'s element in matrix *comp*.

    I(line) = C_element · generation_factors(line, comp, cond).
    Returns 0 if the beam cannot excite the line.
    """
    c = normalize_composition(comp)
    e0 = float(cond.beam_kv)
    ec = line.edge
    if ec <= 0 or e0 <= ec * 1.0001:
        return 0.0
    u0 = e0 / ec
    q = u0 * math.log(u0) - u0 + 1.0
    em = 0.5 * (e0 + ec)
    s = sum(w * z / atomic_mass(z) * math.log(max(1.166 * em / _j_kev(z), 1.0001)) for z, w in c.items())
    eta = sum(w * _eta(z) for z, w in c.items())
    r = 1.0 - eta * 0.6 * (1.0 - 1.0 / u0)
    cosec = 1.0 / math.sin(math.radians(cond.takeoff_deg))
    chi = sum(w * mac_line(line, z) for z, w in c.items()) * cosec
    z_mean, a_mean = _means(c)
    f = float(_philibert(np.asarray(chi), e0, ec, a_mean, z_mean))
    eps = float(efficiency(line.energy, cond.detector))
    return K_CHAR / atomic_mass(line.z) * line.strength * r / s * q * f * eps


def characteristic(comp: dict, cond: Conditions | None = None, min_weight: float = 1e-3) -> list[tuple]:
    """[(XrayLine, intensity)] of every excited line of a material (mass fractions)."""
    cond = cond or Conditions()
    c = normalize_composition(comp)
    out = []
    for z, w in c.items():
        for ln in lines(z, beam_kv=cond.beam_kv, min_weight=min_weight):
            g = generation_factors(ln, c, cond)
            if g > 0:
                out.append((ln, w * g))
    return out


def family_intensity(element, family: str, comp: dict, cond: Conditions | None = None) -> float:
    """Total measured intensity of one family (sum of its lines) in matrix *comp*."""
    cond = cond or Conditions()
    c = normalize_composition(comp)
    z = z_of(element)
    w = c.get(z, 0.0)
    return w * sum(generation_factors(ln, c, cond) for ln in lines(z, family, cond.beam_kv, min_weight=1e-3))


def continuum(energy, comp: dict, cond: Conditions | None = None) -> np.ndarray:
    """Bremsstrahlung intensity density (per keV) at *energy* (keV array)."""
    cond = cond or Conditions()
    c = normalize_composition(comp)
    e = np.asarray(energy, dtype=float)
    e0 = float(cond.beam_kv)
    z_mean, a_mean = _means(c)
    out = np.zeros_like(e)
    ok = (e > 0.05) & (e < e0)
    if not ok.any():
        return out
    ee = e[ok]
    cosec = 1.0 / math.sin(math.radians(cond.takeoff_deg))
    chi = sum(w * mac(ee, z) for z, w in c.items()) * cosec
    f = _philibert(chi, e0, ee, a_mean, z_mean)
    out[ok] = K_CONT * z_mean * (e0 - ee) / ee * f * efficiency(ee, cond.detector)
    return out


def escape_fraction(energy_kev) -> np.ndarray:
    """Fraction of photons of *energy* that lose a Si Kα photon (escape peak), Reed & Ware (1972)."""
    e = np.asarray(energy_kev, dtype=float)
    out = np.zeros_like(e)
    m = e > SI_K_EDGE
    if not m.any():
        return out if np.ndim(e) else float(out)
    w_k = 0.047  # Si K fluorescence yield
    r = 10.8  # Si K jump ratio
    mu_k = mac(SI_KA, 14)
    mu_e = mac(e[m], 14)
    out[m] = 0.5 * w_k * (1 - 1 / r) * (1 - mu_k / mu_e * np.log(1 + mu_e / mu_k))
    return out if np.ndim(e) else float(out)
