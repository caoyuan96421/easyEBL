"""Sample heights for the lithography (no Qt): *H = WD − Z*.

After the beam calibration the column is not touched again – the focus (the working distance,
WD) stays at the schedule's value – so every sample is brought into focus with the stage's Z
alone.  A surface in focus at working distance ``WD`` (from the focus, mm) with the stage at
``Z`` (µm, growing away from the lens) has the height ::

    H = 1000·WD − Z            (µm)

which does not depend on the column: at another WD the same surface is in focus at
``Z = 1000·WD − H``.  Every named stage position has an H, and so has every chip:

* **constant** – one H for the whole chip (focused by hand on one feature);
* **map** – H measured at ≥ 3 points (chip coordinates, CCS), a least-squares plane through them
  gives the H of each field from its centre;
* **none** – the stage's Z is left where it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

__all__ = ["height_um", "z_for", "fit_plane", "ChipHeight", "HEIGHT_MODES"]

HEIGHT_MODES = ("none", "constant", "map")


def height_um(wd_mm: float, z_um: float) -> float:
    """H (µm) of a surface in focus at working distance *wd_mm* with the stage at *z_um*."""
    return 1000.0 * float(wd_mm) - float(z_um)


def z_for(wd_mm: float, h_um: float) -> float:
    """The stage Z (µm) that brings a surface of height *h_um* into focus at *wd_mm*."""
    return 1000.0 * float(wd_mm) - float(h_um)


def fit_plane(points) -> tuple[np.ndarray, float]:
    """Least-squares plane ``H = a + b·x + c·y`` through ≥ 3 points (x, y, H): ``([a, b, c], rms)``.

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
    """The height of a chip: *mode* none | constant (*value_um*) | map (*points*: [(x, y, H)], CCS µm)."""

    mode: str = "none"
    value_um: float | None = None
    points: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in HEIGHT_MODES:
            raise ValueError(f"height mode must be one of {', '.join(HEIGHT_MODES)}")
        self.points = [tuple(float(v) for v in p) for p in self.points]
        if any(len(p) != 3 for p in self.points):
            raise ValueError("a height map point is (x, y, H)")
        self.value_um = None if self.value_um is None else float(self.value_um)

    def plane(self) -> tuple[np.ndarray, float]:
        return fit_plane(self.points)

    @property
    def ready(self) -> bool:
        """The chip has a usable height (mode none: nothing to use)."""
        if self.mode == "constant":
            return self.value_um is not None
        if self.mode == "map":
            try:
                self.plane()
            except ValueError:
                return False
            return True
        return False

    def problem(self) -> str | None:
        """Why the height cannot be used (None: usable or mode none)."""
        if self.mode == "constant" and self.value_um is None:
            return "constant height not recorded"
        if self.mode == "map":
            try:
                self.plane()
            except ValueError as e:
                return str(e)
        return None

    def at(self, ccs) -> float | None:
        """H (µm) at the CCS point *ccs* (None: mode none, or not usable)."""
        if self.mode == "constant":
            return self.value_um
        if self.mode == "map":
            try:
                (a, b, c), _ = self.plane()
            except ValueError:
                return None
            return float(a + b * float(ccs[0]) + c * float(ccs[1]))
        return None

    def tilt_mrad(self) -> float | None:
        if self.mode != "map":
            return None
        try:
            (_, b, c), _ = self.plane()
        except ValueError:
            return None
        return float(math.hypot(b, c) * 1e3)

    def describe(self) -> str:
        if self.mode == "constant":
            return "H not recorded" if self.value_um is None else f"H {self.value_um:.1f} µm"
        if self.mode == "map":
            try:
                (a, b, c), rms = self.plane()
            except ValueError as e:
                return f"map: {e}"
            return f"map {len(self.points)} pts, tilt {math.hypot(b, c) * 1e3:.2f} mrad, rms {rms:.1f} µm"
        return "Z kept"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "value_um": self.value_um, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ChipHeight":
        d = d or {}
        return cls(str(d.get("mode", "none")), d.get("value_um"), d.get("points") or [])
