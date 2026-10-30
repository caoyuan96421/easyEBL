"""Pattern primitives: filled polygons, single-pass lines, dots.

Coordinates are µm in design space.  ``dose_factor`` multiplies the base dose
of the element's kind (area / line / dot).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

import numpy as np

from ..core.transforms import AffineTransform

__all__ = ["Polygon", "Polyline", "Dot", "Pattern", "points_in_polygon", "regular_polygon", "rectangle"]


def _as_points(p) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")
    return a


@dataclass
class Polygon:
    """Closed polygon (even–odd fill; keyhole polygons from GDS are fine)."""

    points: np.ndarray
    dose_factor: float = 1.0
    layer: int | None = None
    datatype: int | None = None
    name: str = ""

    def __post_init__(self):
        p = _as_points(self.points)
        if len(p) > 1 and np.allclose(p[0], p[-1]):
            p = p[:-1]
        if len(p) < 3:
            raise ValueError("a polygon needs at least 3 distinct vertices")
        self.points = p
        if not self.dose_factor >= 0:
            raise ValueError("dose_factor must be ≥ 0")

    kind = "area"

    @property
    def signed_area(self) -> float:
        x, y = self.points[:, 0], self.points[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    @property
    def bbox(self) -> tuple:
        mn, mx = self.points.min(0), self.points.max(0)
        return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    def contains(self, xy) -> np.ndarray:
        return points_in_polygon(np.atleast_2d(xy), self.points)

    def transformed(self, t: AffineTransform) -> "Polygon":
        return replace(self, points=t.apply(self.points))

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points, np.roll(self.points, -1, axis=0)


@dataclass
class Polyline:
    """Open path exposed as a single-pass line (no width)."""

    points: np.ndarray
    dose_factor: float = 1.0
    layer: int | None = None
    datatype: int | None = None
    name: str = ""

    kind = "line"

    def __post_init__(self):
        p = _as_points(self.points)
        if len(p) < 2:
            raise ValueError("a polyline needs at least 2 points")
        self.points = p
        if not self.dose_factor >= 0:
            raise ValueError("dose_factor must be ≥ 0")

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.points, axis=0).T)))

    @property
    def bbox(self) -> tuple:
        mn, mx = self.points.min(0), self.points.max(0)
        return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))

    def transformed(self, t: AffineTransform) -> "Polyline":
        return replace(self, points=t.apply(self.points))


@dataclass
class Dot:
    x: float
    y: float
    dose_factor: float = 1.0
    layer: int | None = None
    datatype: int | None = None
    name: str = ""

    kind = "dot"

    @property
    def points(self) -> np.ndarray:
        return np.array([[self.x, self.y]], float)

    @property
    def bbox(self) -> tuple:
        return (self.x, self.y, self.x, self.y)

    def transformed(self, t: AffineTransform) -> "Dot":
        x, y = t.apply(np.array([self.x, self.y]))
        return replace(self, x=float(x), y=float(y))


Element = Polygon | Polyline | Dot


@dataclass
class Pattern:
    """An ordered collection of exposure elements."""

    elements: list = field(default_factory=list)
    name: str = ""
    source: str = ""

    def __iter__(self) -> Iterator:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

    def add(self, *els) -> "Pattern":
        self.elements.extend(els)
        return self

    def by_kind(self, kind: str) -> list:
        return [e for e in self.elements if e.kind == kind]

    @property
    def bbox(self) -> tuple | None:
        if not self.elements:
            return None
        b = np.array([e.bbox for e in self.elements])
        return (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))

    def transformed(self, t: AffineTransform) -> "Pattern":
        return Pattern([e.transformed(t) for e in self.elements], self.name, self.source)

    def centered(self) -> "Pattern":
        """Translate so the bounding-box centre is at the origin."""
        b = self.bbox
        if b is None:
            return self
        return self.transformed(AffineTransform.translation(-(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2))

    def dose_factors(self) -> np.ndarray:
        return np.array([e.dose_factor for e in self.elements])

    def total_area(self) -> float:
        return float(sum(e.area for e in self.by_kind("area")))

    def summary(self) -> dict:
        df = self.dose_factors()
        return {
            "polygons": len(self.by_kind("area")),
            "lines": len(self.by_kind("line")),
            "dots": len(self.by_kind("dot")),
            "area_um2": self.total_area(),
            "bbox": self.bbox,
            "dose_factor_range": (float(df.min()), float(df.max())) if len(df) else None,
        }

    @classmethod
    def concat(cls, patterns: Iterable["Pattern"]) -> "Pattern":
        els = []
        for p in patterns:
            els.extend(p.elements)
        return cls(els)


# ------------------------------------------------------------------ helpers
def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even–odd point-in-polygon test, vectorised over points."""
    xy = np.atleast_2d(np.asarray(xy, float))
    px, py = xy[:, 0][:, None], xy[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    cross = cond & (px < xint)
    return (np.count_nonzero(cross, axis=1) % 2) == 1


def rectangle(x0: float, y0: float, x1: float, y1: float, **kw) -> Polygon:
    return Polygon(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), **kw)


def regular_polygon(cx: float, cy: float, r: float, n: int = 64, **kw) -> Polygon:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]), **kw)
