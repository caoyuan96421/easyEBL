"""Chip heights for the lithography (no Qt): the working distance in focus, at the calibration Z.

The stage Z is not accurate enough to bring samples into focus, and the WD (from the focus) is
nominal.  So the stage Z stays at the calibration sample's ``Z_cal`` on every chip and the focus
does the rest:

* on the calibration sample the user focuses at ``Z_cal``: that focus is **WD1** (the stage
  position *Calibration sample*, :mod:`easyebl.litho.settings`);
* on a chip the user focuses with the stage still at ``Z_cal``: that focus is **WD2** (one value
  for the chip) or **WD2x** (≥ 3 points, chip coordinates, a least-squares plane gives the WD of
  each field from its centre).

A chip height is only good at the Z it was recorded at (:attr:`ChipHeight.z_um`): when that is
not the calibration sample's Z the schedule does not run; a WD2 more than 1 mm from WD1 is warned
about.  Modes:

* **constant** – one WD2 for the whole chip;
* **map** – WD2x at ≥ 3 points;
* **none** – the focus is left as it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

__all__ = ["fit_plane", "ChipHeight", "HEIGHT_MODES", "Z_MATCH_UM", "WD_WARN_MM", "z_problem"]

HEIGHT_MODES = ("none", "constant", "map")
#: a chip height recorded farther than this from the calibration sample's Z is not usable (µm)
Z_MATCH_UM = 1.0
#: a chip's WD2 farther than this from WD1 is warned about (mm)
WD_WARN_MM = 1.0


def z_problem(z_um: float | None, z_cal: float | None, what: str = "the height") -> str | None:
    """Why something recorded at stage Z *z_um* does not go with the calibration Z *z_cal* (None: it does)."""
    if z_cal is None:
        return "the calibration sample's Z is not recorded (Stage positions → Record WD)"
    if z_um is None:
        return f"{what} has no stage Z"
    if abs(float(z_um) - float(z_cal)) > Z_MATCH_UM:
        return (f"{what} was recorded at Z {float(z_um):.1f} µm, not at the calibration sample's Z "
                f"{float(z_cal):.1f} µm: record it again with the stage at Z_cal")
    return None


def fit_plane(points) -> tuple[np.ndarray, float]:
    """Least-squares plane ``v = a + b·x + c·y`` through ≥ 3 points (x, y, v): ``([a, b, c], rms)``.

    Raises ValueError for fewer than 3 points or points on one line."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 3:
        raise ValueError(f"a height map needs at least 3 points (has {len(p)})")
    xy = p[:, :2] - p[:, :2].mean(0)
    span = float(np.max(np.hypot(xy[:, 0], xy[:, 1]))) or 1.0
    a = np.column_stack([np.ones(len(p)), xy / span])
    if np.linalg.matrix_rank(a, tol=1e-6) < 3:
        raise ValueError("the height map points lie on one line")
    coef, *_ = np.linalg.lstsq(a, p[:, 2], rcond=None)
    res = p[:, 2] - a @ coef
    b, c = coef[1] / span, coef[2] / span
    a0 = coef[0] - b * p[:, 0].mean() - c * p[:, 1].mean()
    return np.array([a0, b, c]), float(np.sqrt(np.mean(res * res)))


@dataclass
class ChipHeight:
    """The height of a chip as the WD in focus: *mode* none | constant (*wd_mm*, WD2) | map
    (*points*: [(x, y, WD mm)], CCS µm – WD2x), all recorded with the stage at *z_um*."""

    mode: str = "none"
    wd_mm: float | None = None
    points: list = field(default_factory=list)
    z_um: float | None = None

    def __post_init__(self):
        if self.mode not in HEIGHT_MODES:
            raise ValueError(f"height mode must be one of {', '.join(HEIGHT_MODES)}")
        self.points = [tuple(float(v) for v in p) for p in self.points]
        if any(len(p) != 3 for p in self.points):
            raise ValueError("a height map point is (x, y, WD)")
        self.wd_mm = None if self.wd_mm is None else float(self.wd_mm)
        self.z_um = None if self.z_um is None else float(self.z_um)

    def plane(self) -> tuple[np.ndarray, float]:
        return fit_plane(self.points)

    @property
    def ready(self) -> bool:
        """The chip has a usable height (mode none: nothing to use)."""
        return self.mode != "none" and self.problem() is None

    def problem(self, z_cal: float | None = None, check_z: bool = False) -> str | None:
        """Why the height cannot be used (None: usable, or mode none).  *check_z*: also that it was
        recorded at the calibration Z *z_cal*."""
        if self.mode == "none":
            return None
        if self.mode == "constant" and self.wd_mm is None:
            return "WD2 not recorded"
        if self.mode == "map":
            try:
                self.plane()
            except ValueError as e:
                return str(e)
        if check_z:
            return z_problem(self.z_um, z_cal, "the chip height")
        return None

    def wds(self) -> list:
        """The recorded WDs (mm)."""
        if self.mode == "constant":
            return [] if self.wd_mm is None else [self.wd_mm]
        if self.mode == "map":
            return [p[2] for p in self.points]
        return []

    def wd_at(self, ccs) -> float | None:
        """The WD (mm) in focus at the CCS point *ccs* (None: mode none, or not usable)."""
        if self.mode == "constant":
            return self.wd_mm
        if self.mode == "map":
            try:
                (a, b, c), _ = self.plane()
            except ValueError:
                return None
            return float(a + b * float(ccs[0]) + c * float(ccs[1]))
        return None

    def tilt_mrad(self) -> float | None:
        """The tilt of the WD plane (mrad: 1 mm of WD over 1 mm in CCS is 1000 mrad)."""
        if self.mode != "map":
            return None
        try:
            (_, b, c), _ = self.plane()
        except ValueError:
            return None
        return float(math.hypot(b, c) * 1e3 * 1e3)  # mm per µm → µm per µm, in mrad

    def describe(self) -> str:
        at = "" if self.z_um is None else f" at Z {self.z_um:.0f} µm"
        if self.mode == "constant":
            return "WD2 not recorded" if self.wd_mm is None else f"WD2 {self.wd_mm:.3f} mm{at}"
        if self.mode == "map":
            try:
                (a, b, c), rms = self.plane()
            except ValueError as e:
                return f"map: {e}"
            return (f"map {len(self.points)} pts, tilt {self.tilt_mrad():.2f} mrad, rms {rms * 1e3:.1f} µm"
                    + at)
        return "focus kept"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "wd_mm": self.wd_mm, "points": [list(p) for p in self.points],
                "z_um": self.z_um}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ChipHeight":
        d = d or {}
        if "value_um" in d and "wd_mm" not in d:  # an older file: H = WD − Z, not convertible
            return cls()
        return cls(str(d.get("mode", "none")), d.get("wd_mm"), d.get("points") or [], d.get("z_um"))
