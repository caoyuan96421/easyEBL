"""Dose ↔ dwell-time arithmetic.

Units used throughout (EBL conventions):

==============  ===========  =====================================
quantity        unit         relation
==============  ===========  =====================================
area dose       µC/cm²       dwell = D · step² / I
line dose       pC/cm        dwell = D_l · step / I
dot dose        fC           dwell = Q / I
beam current    A (pA in UI)
step / field    µm
==============  ===========  =====================================
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

__all__ = [
    "area_dwell_s",
    "line_dwell_s",
    "dot_dwell_s",
    "area_dose_uC_cm2",
    "line_dose_pC_cm",
    "dot_dose_fC",
    "step_from_points",
    "dac_step_um",
    "quantize_dwell",
    "area_write_time_s",
    "max_current_for_dwell",
    "DoseCalculator",
    "DoseReport",
]

UC_PER_CM2_TO_C_PER_UM2 = 1e-6 / 1e8  # 1 µC/cm² = 1e-14 C/µm²
PC_PER_CM_TO_C_PER_UM = 1e-12 / 1e4  # 1 pC/cm = 1e-16 C/µm
FC = 1e-15


def _pos(name: str, v: float) -> None:
    if not (v > 0) or not math.isfinite(v):
        raise ValueError(f"{name} must be a positive finite number (got {v!r})")


def area_dwell_s(dose_uC_cm2: float, step_um: float, current_a: float) -> float:
    """Dwell per point for an area exposure on a square grid of pitch *step_um*."""
    _pos("dose", dose_uC_cm2), _pos("step", step_um), _pos("current", current_a)
    return dose_uC_cm2 * UC_PER_CM2_TO_C_PER_UM2 * step_um**2 / current_a


def line_dwell_s(dose_pC_cm: float, step_um: float, current_a: float) -> float:
    _pos("line dose", dose_pC_cm), _pos("step", step_um), _pos("current", current_a)
    return dose_pC_cm * PC_PER_CM_TO_C_PER_UM * step_um / current_a


def dot_dwell_s(dose_fC: float, current_a: float) -> float:
    _pos("dot dose", dose_fC), _pos("current", current_a)
    return dose_fC * FC / current_a


def area_dose_uC_cm2(dwell_s: float, step_um: float, current_a: float) -> float:
    return dwell_s * current_a / (step_um**2 * UC_PER_CM2_TO_C_PER_UM2)


def line_dose_pC_cm(dwell_s: float, step_um: float, current_a: float) -> float:
    return dwell_s * current_a / (step_um * PC_PER_CM_TO_C_PER_UM)


def dot_dose_fC(dwell_s: float, current_a: float) -> float:
    return dwell_s * current_a / FC


def step_from_points(field_um: float, points: int) -> float:
    """Grid pitch when *points* addressable positions span *field_um*."""
    _pos("field", field_um)
    if points < 1:
        raise ValueError("points must be ≥ 1")
    return field_um / points


def dac_step_um(field_um: float, bits: int = 16, used_fraction: float = 1.0) -> float:
    """Finest beam step if ±full-scale (= field) spans 2**bits codes."""
    return field_um / (2**bits * used_fraction)


def quantize_dwell(dwell_s: float, rate_hz: float, min_samples: int = 1) -> tuple[int, float, float]:
    """Round a dwell to whole clock samples.

    Returns ``(n_samples, actual_dwell_s, relative_error)``.
    """
    _pos("rate", rate_hz)
    n = max(min_samples, int(round(dwell_s * rate_hz)))
    actual = n / rate_hz
    return n, actual, (actual - dwell_s) / dwell_s if dwell_s > 0 else 0.0


def area_write_time_s(area_um2: float, dose_uC_cm2: float, current_a: float) -> float:
    """Pure exposure time (no settling/overhead) for *area_um2* at *dose*."""
    return area_um2 * dose_uC_cm2 * UC_PER_CM2_TO_C_PER_UM2 / current_a


def max_current_for_dwell(dose_uC_cm2: float, step_um: float, min_dwell_s: float) -> float:
    """Largest beam current that keeps the dwell ≥ *min_dwell_s*."""
    return dose_uC_cm2 * UC_PER_CM2_TO_C_PER_UM2 * step_um**2 / min_dwell_s


@dataclass
class DoseReport:
    field_um: float
    points: int
    step_um: float
    dac_step_um: float
    current_a: float
    rate_hz: float
    area_dose_uC_cm2: float
    area_dwell_s: float
    area_dwell_samples: int
    area_dwell_error: float
    line_dose_pC_cm: float
    line_dwell_s: float
    line_dwell_samples: int
    dot_dose_fC: float
    dot_dwell_s: float
    dot_dwell_samples: int
    time_per_100um2_s: float  # 10 µm × 10 µm square (kept for compatibility)
    time_per_um2_s: float
    time_full_field_s: float  # field_um × field_um, every point exposed
    time_full_field_quantised_s: float  # same, with the quantised dwell actually streamed
    warnings: list

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DoseCalculator:
    """Convenience bundle used by the GUI and the writer."""

    field_um: float = 100.0
    points: int = 5000  # addressable points across the field (sets the step)
    current_pA: float = 100.0
    rate_hz: float = 500e3
    area_dose_uC_cm2: float = 300.0
    line_dose_pC_cm: float = 1000.0
    dot_dose_fC: float = 10.0
    min_dwell_s: float = 2.5e-6
    max_dose_error: float = 0.02
    dac_bits: int = 16

    @property
    def step_um(self) -> float:
        return step_from_points(self.field_um, self.points)

    @property
    def current_a(self) -> float:
        return self.current_pA * 1e-12

    def report(self) -> DoseReport:
        step = self.step_um
        I = self.current_a
        warn = []
        a = area_dwell_s(self.area_dose_uC_cm2, step, I)
        l = line_dwell_s(self.line_dose_pC_cm, step, I)
        d = dot_dwell_s(self.dot_dose_fC, I)
        na, aa, ea = quantize_dwell(a, self.rate_hz)
        nl, _, el = quantize_dwell(l, self.rate_hz)
        nd, _, ed = quantize_dwell(d, self.rate_hz)
        dac = dac_step_um(self.field_um, self.dac_bits)
        if step < dac:
            warn.append(f"step {step * 1e3:.2f} nm is below the DAC resolution {dac * 1e3:.2f} nm")
        for name, dw, err in (("area", a, ea), ("line", l, el), ("dot", d, ed)):
            if dw < self.min_dwell_s:
                warn.append(
                    f"{name} dwell {dw * 1e6:.3g} µs < minimum {self.min_dwell_s * 1e6:.3g} µs "
                    f"(lower the current or increase the step)"
                )
            if abs(err) > self.max_dose_error:
                warn.append(f"{name} dwell quantisation error {err * 100:+.1f}% (raise the clock rate)")
        return DoseReport(
            field_um=self.field_um,
            points=self.points,
            step_um=step,
            dac_step_um=dac,
            current_a=I,
            rate_hz=self.rate_hz,
            area_dose_uC_cm2=self.area_dose_uC_cm2,
            area_dwell_s=a,
            area_dwell_samples=na,
            area_dwell_error=ea,
            line_dose_pC_cm=self.line_dose_pC_cm,
            line_dwell_s=l,
            line_dwell_samples=nl,
            dot_dose_fC=self.dot_dose_fC,
            dot_dwell_s=d,
            dot_dwell_samples=nd,
            time_per_100um2_s=area_write_time_s(100.0, self.area_dose_uC_cm2, I),
            time_per_um2_s=area_write_time_s(1.0, self.area_dose_uC_cm2, I),
            time_full_field_s=area_write_time_s(self.field_um**2, self.area_dose_uC_cm2, I),
            time_full_field_quantised_s=self.points**2 * na / self.rate_hz,
            warnings=warn,
        )
