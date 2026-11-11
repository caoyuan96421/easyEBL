"""Standardless quantification (ZA, iterative) from fitted net line intensities.

For each element one family is used (K preferred above ~1 keV when the
overvoltage is ≥ 1.5, else L, else M).  With the generation model of
:mod:`easyebl.xray.generation`::

    I_i = C_i · G_i(matrix)          (G: ionisation, backscatter, stopping, absorption, detector)

the mass fractions are found by iteration: C ← (I/G(C)) normalised to 100 %.
Oxygen can be set by stoichiometry (oxides with standard valences).  The
result is semi-quantitative: typically ±5–20 % relative for major elements –
no secondary fluorescence, no standards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import xray
from ...xray import generation as gen
from .fitting import FitResult

__all__ = ["QuantRow", "QuantResult", "quantify", "choose_family", "VALENCE", "intensity_factors"]

#: Standard cation valences for "oxygen by stoichiometry" (oxide formulas).
VALENCE = {
    "Na": 1, "K": 1, "Li": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Cu": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2,
    "Ni": 2, "Co": 2, "Mn": 2, "Fe": 3, "Pb": 2, "Cd": 2, "Al": 3, "B": 3, "Cr": 3, "Ga": 3, "In": 3, "La": 3,
    "Ce": 4, "Y": 3, "Sc": 3, "Bi": 3, "Si": 4, "Ti": 4, "Zr": 4, "Hf": 4, "Sn": 4, "Ge": 4, "Th": 4, "V": 5,
    "Nb": 5, "Ta": 5, "P": 5, "As": 5, "Sb": 5, "Mo": 6, "W": 6, "S": 6, "U": 6, "C": 4, "N": 5,
}


@dataclass
class QuantRow:
    element: str
    family: str
    intensity: float
    intensity_sigma: float
    wt: float  # mass %
    wt_sigma: float
    at: float  # atom %
    oxide: str = ""
    oxide_wt: float = float("nan")
    by_stoichiometry: bool = False

    @property
    def label(self) -> str:
        return f"{self.element} {self.family}"


@dataclass
class QuantResult:
    rows: list
    iterations: int
    raw_total: float  # Σ C before normalisation (arbitrary units – no beam-current standard)
    beam_kv: float
    method: str = "standardless ZA (easyEBL generation model)"
    notes: list = field(default_factory=list)

    def wt(self) -> dict:
        return {r.element: r.wt for r in self.rows}

    def at(self) -> dict:
        return {r.element: r.at for r in self.rows}

    def table(self) -> str:
        lines = [f"{'Element':<8}{'Line':<6}{'Wt%':>8}{'σ':>7}{'At%':>8}" +
                 ("    Oxide   Wt%" if any(r.oxide for r in self.rows) else "")]
        for r in self.rows:
            s = f"{r.element:<8}{r.family:<6}{r.wt:8.2f}{r.wt_sigma:7.2f}{r.at:8.2f}"
            if r.oxide:
                s += f"    {r.oxide:<7}{r.oxide_wt:6.2f}"
            if r.by_stoichiometry:
                s += "  (stoich.)"
            lines.append(s)
        lines.append(f"{'Total':<14}{sum(r.wt for r in self.rows):8.2f}")
        return "\n".join(lines)


def choose_family(element, beam_kv: float, available=None) -> str | None:
    """Family to quantify *element* with: K when its Kα is ≥ 1 keV… and U0 ≥ 1.5, else L, else M."""
    el = xray.symbol(element)
    fams = [f for f in xray.families(el, beam_kv) if available is None or f in available]
    if not fams:
        return None

    def ok(f):
        ln = xray.main_line(el, f)
        return ln is not None and beam_kv / ln.edge >= 1.5

    for pref in ("K", "L", "M"):
        if pref in fams and ok(pref):
            ln = xray.main_line(el, pref)
            if pref != "K" or ln.energy >= 0.9 or xray.z_of(el) <= 11:
                return pref
    return fams[0]


def intensity_factors(elements_fam: dict, comp: dict, cond: gen.Conditions) -> dict:
    """G_i: intensity per unit mass fraction of each (element → family) in matrix *comp*."""
    out = {}
    for el, fam in elements_fam.items():
        out[el] = sum(gen.generation_factors(ln, comp, cond) for ln in xray.lines(el, fam, cond.beam_kv, min_weight=1e-3))
    return out


def _oxide_formula(el: str, v: int) -> tuple[str, float, float]:
    """(formula, cations, oxygens) per formula unit for valence *v*: M2O_v or MO_{v/2}."""
    if v % 2 == 0:
        n_m, n_o = 1, v // 2
    else:
        n_m, n_o = 2, v
    f = f"{el}{n_m if n_m > 1 else ''}O{n_o if n_o > 1 else ''}"
    return f, n_m, n_o


def quantify(fit: FitResult | dict, beam_kv: float | None = None, takeoff_deg: float = 35.0,
             detector: gen.Detector | None = None, oxygen_by_stoichiometry: bool = False,
             families: dict | None = None, exclude=("O",), max_iter: int = 50, tol: float = 1e-5) -> QuantResult:
    """Mass/atom % from a :class:`FitResult` (or a dict {"Fe K": (area, sigma), …}).

    *families*: {element: family} overrides the automatic choice.
    *oxygen_by_stoichiometry*: O is computed from the cations (and its measured
    peak, if fitted, is ignored); otherwise O is quantified from its K line
    unless listed in *exclude* (default: O measured lines are used only when
    not excluded – the AP3.3 window makes O K quantitative only roughly).
    """
    if isinstance(fit, FitResult):
        data = fit.as_dict()
        kv = beam_kv or (fit.extras.get("beam_kv") if fit.extras else None) or 20.0
    else:
        data = dict(fit)
        kv = beam_kv or 20.0
    det = detector or gen.Detector()
    cond = gen.Conditions(beam_kv=kv, takeoff_deg=takeoff_deg, detector=det)
    # intensities per element: the chosen family
    by_el: dict = {}
    for label, (a, s) in data.items():
        el, fam = label.split()
        by_el.setdefault(el, {})[fam] = (float(a), float(s))
    notes = []
    chosen = {}
    for el, fams in by_el.items():
        if oxygen_by_stoichiometry and el == "O":
            continue
        if not oxygen_by_stoichiometry and el in exclude and el == "O":
            notes.append("O excluded (enable 'oxygen by stoichiometry' for oxides)")
            continue
        fam = (families or {}).get(el) or choose_family(el, kv, available=list(fams))
        if fam is None or fam not in fams:
            continue
        chosen[el] = fam
    intens = {el: by_el[el][f][0] for el, f in chosen.items()}
    sig = {el: by_el[el][f][1] for el, f in chosen.items()}
    pos = {el: max(i, 0.0) for el, i in intens.items()}
    if not pos or sum(pos.values()) <= 0:
        return QuantResult([], 0, 0.0, kv, notes=notes + ["no positive intensities"])
    fams = dict(chosen)
    valence = {el: VALENCE.get(el) for el in pos}
    if oxygen_by_stoichiometry:
        missing = [el for el, v in valence.items() if v is None]
        if missing:
            notes.append(f"no standard valence for {', '.join(missing)} – counted as metal")
    a_o = xray.atomic_mass("O")

    def with_oxygen(c: dict) -> dict:
        if not oxygen_by_stoichiometry:
            return dict(c)
        out = dict(c)
        o = 0.0
        for el, w in c.items():
            v = valence.get(el)
            if v:
                o += w / xray.atomic_mass(el) * v / 2 * a_o
        out["O"] = o
        return out

    # start: pure-element factors
    g_pure = {el: intensity_factors({el: fams[el]}, {el: 1.0}, cond)[el] for el in pos}
    c = {el: (pos[el] / g_pure[el] if g_pure[el] > 0 else 0.0) for el in pos}
    raw = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        full = with_oxygen(c)
        tot = sum(full.values())
        comp = {el: w / tot for el, w in full.items() if w > 0}
        g = intensity_factors({el: fams[el] for el in pos}, comp, cond)
        new = {el: (pos[el] / g[el] if g[el] > 0 else 0.0) for el in pos}
        raw = sum(with_oxygen(new).values())
        new_n = {el: w / raw for el, w in new.items()}
        old_n = {el: w / sum(with_oxygen(c).values()) for el, w in c.items()}
        c = new
        if max(abs(new_n[el] - old_n[el]) for el in pos) < tol:
            break
    full = with_oxygen(c)
    tot = sum(full.values())
    wt = {el: 100.0 * w / tot for el, w in full.items()}
    mol = {el: w / xray.atomic_mass(el) for el, w in wt.items()}
    mt = sum(mol.values())
    rows = []
    for el in sorted(wt, key=lambda e: -wt[e]):
        if el == "O" and oxygen_by_stoichiometry:
            rows.append(QuantRow("O", "K", float("nan"), float("nan"), wt[el], float("nan"), 100 * mol[el] / mt,
                                 by_stoichiometry=True))
            continue
        rel = sig[el] / intens[el] if intens[el] > 0 else np.inf
        r = QuantRow(el, fams[el], intens[el], sig[el], wt[el], wt[el] * rel, 100 * mol[el] / mt)
        if oxygen_by_stoichiometry and valence.get(el):
            f, n_m, n_o = _oxide_formula(el, valence[el])
            m_ox = n_m * xray.atomic_mass(el) + n_o * a_o
            r.oxide, r.oxide_wt = f, wt[el] * m_ox / (n_m * xray.atomic_mass(el))
        rows.append(r)
    return QuantResult(rows, it, raw, kv, notes=notes)
