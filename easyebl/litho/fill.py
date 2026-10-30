"""Turning pattern elements into ordered exposure points.

* :func:`raster_fill` – grid-aligned raster (scan-line) fill of a polygon.
  Exposure points are pixel centres of a global grid (``origin + k·step``),
  so neighbouring polygons tile without gaps or double exposure, and a point
  is exposed iff its centre lies inside the polygon (even–odd rule).  Every
  contiguous span of a scan line becomes one *run* (exposed without blanking).
* :func:`line_points` – single-pass lines, points at the centres of equal
  sub-segments (each point represents ``spacing`` of length).
* :func:`dot_points` – dots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import Dot, Polygon, Polyline

__all__ = ["FillResult", "raster_fill", "line_points", "LinePoints", "dot_points"]


@dataclass
class FillResult:
    runs: list = field(default_factory=list)  # list of (M, 2) arrays, exposure order
    step_um: float = 0.0

    @property
    def n_points(self) -> int:
        return int(sum(len(r) for r in self.runs))

    def all_points(self) -> np.ndarray:
        return np.vstack(self.runs) if self.runs else np.zeros((0, 2))


def _row_spans(px: np.ndarray, py: np.ndarray, y: float) -> np.ndarray:
    """x-intervals [xa, xb) of the even–odd interior on the horizontal line *y*."""
    x0, y0 = px, py
    x1, y1 = np.roll(px, -1), np.roll(py, -1)
    hit = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
    if not np.any(hit):
        return np.zeros((0, 2))
    xs = x0[hit] + (y - y0[hit]) * (x1[hit] - x0[hit]) / (y1[hit] - y0[hit])
    xs.sort()
    return xs.reshape(-1, 2) if len(xs) % 2 == 0 else xs[: len(xs) - 1].reshape(-1, 2)


def raster_fill(
    polygon: Polygon,
    step_um: float,
    direction: str = "x",
    serpentine: bool = False,
    origin_um: tuple = (0.0, 0.0),
    reverse: bool = False,
) -> FillResult:
    """Grid-aligned scan-line fill.

    Parameters
    ----------
    direction:
        ``"x"`` – lines along x, stepping in y (default); ``"y"`` – lines along y.
    serpentine:
        Alternate the line direction (faster, but sensitive to deflection
        lag: alternate lines are offset by ±v·τ).
    origin_um:
        Grid origin; points are at ``origin + (k + ½)·step``.
    reverse:
        Run every line in the negative direction (unidirectional).
    """
    if not step_um > 0:
        raise ValueError("step_um must be positive")
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    pts = polygon.points
    gx, gy = origin_um
    if direction == "y":  # work in swapped coordinates
        pts = pts[:, ::-1]
        gx, gy = gy, gx
    px, py = pts[:, 0], pts[:, 1]
    s = step_um
    j0 = math.ceil((py.min() - gy) / s - 0.5)
    j1 = math.floor((py.max() - gy) / s - 0.5)
    runs = []
    row_index = 0
    for j in range(j0, j1 + 1):
        y = gy + (j + 0.5) * s
        spans = _row_spans(px, py, y)
        row_runs = []
        for xa, xb in spans:
            i0 = math.ceil((xa - gx) / s - 0.5)
            i1 = math.ceil((xb - gx) / s - 0.5) - 1  # x < xb
            if i1 < i0:
                continue
            xs = gx + (np.arange(i0, i1 + 1) + 0.5) * s
            row_runs.append(np.column_stack([xs, np.full(len(xs), y)]))
        if not row_runs:
            continue
        flip = reverse ^ (serpentine and row_index % 2 == 1)
        if flip:
            row_runs = [r[::-1] for r in row_runs[::-1]]
        runs.extend(row_runs)
        row_index += 1
    if direction == "y":
        runs = [r[:, ::-1].copy() for r in runs]
    return FillResult(runs=runs, step_um=s)


@dataclass
class LinePoints:
    points: np.ndarray  # (M, 2)
    spacing_um: np.ndarray  # (M,) length represented by each point


def line_points(polyline: Polyline, step_um: float) -> LinePoints:
    """Points along a polyline, each representing an equal share of its segment."""
    if not step_um > 0:
        raise ValueError("step_um must be positive")
    out, sp = [], []
    p = polyline.points
    for a, b in zip(p[:-1], p[1:]):
        L = float(math.hypot(*(b - a)))
        if L == 0:
            continue
        n = max(1, int(round(L / step_um)))
        t = (np.arange(n) + 0.5) / n
        out.append(a + (b - a) * t[:, None])
        sp.append(np.full(n, L / n))
    if not out:
        raise ValueError("polyline has zero length")
    return LinePoints(np.vstack(out), np.concatenate(sp))


def dot_points(dots: list) -> np.ndarray:
    return np.array([[d.x, d.y] for d in dots], float).reshape(-1, 2)
