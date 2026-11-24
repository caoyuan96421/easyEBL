"""The stage view: a zoomable, pannable plot in SCS (stage µm) with each field's WCS overlaid.

* every Expose field: its field of view (dashed square at the stage target), its pattern where
  it will be written (through the field's transform), a cross at its WCS origin; a corrected
  target is joined to the nominal centre by a thin line;
* the selected field: highlighted, with its WCS drawn as a Cartesian grid and x / y axes
  (shifted to the field centre; rotated / stretched once a Global Align is known);
* Global Align markers (magenta ×), Local XY markers (cyan ×);
* named stage positions, the stage (white ○) and its current field of view (green).
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from ..coords import FieldTransform, StageAxes
from ..geometry import Dot, Polygon
from ..schedule import ExposeField

__all__ = ["StageView", "outline_arrays", "nice_step"]

C_PATTERN = (90, 190, 255)
C_WARN = (255, 90, 90)
C_SELECTED = (255, 220, 40)
C_FOV = (80, 255, 120)
C_PLAN_FOV = (150, 150, 170, 110)
C_GLOBAL = (255, 80, 200)
C_LOCAL = (60, 230, 230)
C_POS = (255, 170, 60)
C_WCS = (255, 220, 40, 90)


def outline_arrays(pieces, max_points: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """x, y of point arrays, NaN-separated (one plot item for many shapes)."""
    parts, n = [], 0
    for p in pieces:
        parts.append(p)
        parts.append(np.array([[np.nan, np.nan]]))
        n += len(p) + 1
        if n > max_points:
            break
    if not parts:
        return np.zeros(0), np.zeros(0)
    a = np.vstack(parts)
    return a[:, 0], a[:, 1]


def _shapes(pattern) -> list:
    out = []
    for e in pattern:
        if isinstance(e, Dot):
            out.append(np.array([[e.x, e.y], [e.x, e.y]]))
        elif isinstance(e, Polygon):
            out.append(np.vstack([e.points, e.points[:1]]))
        else:
            out.append(e.points)
    return out


def nice_step(span: float) -> float:
    if span <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(span))
    for m in (1, 2, 5, 10):
        if span <= m * e:
            return m * e
    return 10 * e


def _square(centre, half) -> np.ndarray:
    cx, cy = centre
    return np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half],
                     [cx - half, cy - half]])


class StageView(pg.PlotWidget):
    """See module docstring.  ``stepClicked(index)`` when a field is clicked; ``contextRequested(x, y,
    screen_pos)`` on a right click (SCS µm of the click, global screen position for a menu)."""

    stepClicked = QtCore.Signal(int)
    contextRequested = QtCore.Signal(float, float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "stage X (SCS)", "µm")
        self.setLabel("left", "stage Y (SCS)", "µm")
        self.setBackground((18, 18, 22))
        self.getPlotItem().getViewBox().setDefaultPadding(0.05)
        self.getPlotItem().getViewBox().setMenuEnabled(False)  # our own context menu (Move…)
        self.getPlotItem().ctrlMenu = None
        self.show_wcs_grid = True

        def item(color, width=1.0, style=None):
            pen = pg.mkPen(color, width=width, style=style) if style is not None else pg.mkPen(color, width=width)
            it = pg.PlotDataItem(pen=pen, connect="finite")
            self.addItem(it)
            return it

        self._plan_fov = item(C_PLAN_FOV, 1, QtCore.Qt.DashLine)
        self._links = item((150, 150, 170, 140), 1)
        self._wcs_grid = item(C_WCS, 1)
        self._patterns = item(C_PATTERN, 1)
        self._warn = item(C_WARN, 1.2)
        self._selected = item(C_SELECTED, 2.2)
        self._wcs_axes = item(C_SELECTED, 2)
        self._fov = item(C_FOV, 2)
        self._origins = pg.ScatterPlotItem(symbol="+", size=9, pen=pg.mkPen(C_PLAN_FOV[:3], width=1), brush=None)
        self._global = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen(C_GLOBAL, width=2), brush=None)
        self._local = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen(C_LOCAL, width=2), brush=None)
        self._positions = pg.ScatterPlotItem(symbol="s", size=10, pen=pg.mkPen(C_POS), brush=pg.mkBrush(C_POS))
        self._stage = pg.ScatterPlotItem(symbol="o", size=16, pen=pg.mkPen((255, 255, 255), width=2), brush=None)
        for it in (self._origins, self._global, self._local, self._positions, self._stage):
            self.addItem(it)
        self._labels: list = []
        self._axis_labels = [pg.TextItem("x", color=C_SELECTED, anchor=(0, 0.5)),
                             pg.TextItem("y", color=C_SELECTED, anchor=(0.5, 1))]
        self._fov_label = pg.TextItem("FoV", color=C_FOV, anchor=(0, 1))
        for t in (*self._axis_labels, self._fov_label):
            self.addItem(t)
        self._boxes: list = []  # (index, x0, y0, x1, y1) for click hit-testing
        self.scene().sigMouseClicked.connect(self._clicked)
        self._fitted = False

    # ---------------------------------------------------------------- schedule
    def set_schedule(self, schedule, library, plans, axes: StageAxes, selected: int | None = None) -> None:
        m = axes.matrix
        normal, warned, sel, fovs, links, origins, glob, loc = [], [], [], [], [], [], [], []
        boxes = []
        sel_frame = None
        for i, op in enumerate(schedule.operations):
            p = plans[i] if plans is not None and i < len(plans) else None
            if not isinstance(op, ExposeField) or p is None or p.target is None:
                continue
            tgt = np.asarray(p.target, float)
            tr = p.transform or FieldTransform()
            half = p.field_um / 2.0
            fovs.append(_square(tgt, half))
            boxes.append((i, tgt[0] - half, tgt[1] - half, tgt[0] + half, tgt[1] + half))
            org = tgt + m @ tr.shift
            origins.append(org)
            if np.hypot(*(tgt - np.asarray(op.centre))) > 1e-6:
                links.append(np.array([op.centre, tgt]))
            for wcs, dst in [(mk, glob) for mk in op.active_markers] + ([(op.local_marker, loc)] if op.local_xy else []):
                dst.append(tgt + m @ tr.to_bcs(wcs))  # markers are WCS: drawn with their field
            try:
                pat = library.get(op.pattern)
            except Exception:  # noqa: BLE001 - a warning in the list
                pat = []
            shapes = [tgt + (tr.to_bcs(s) @ m.T) for s in _shapes(pat)]
            (sel if i == selected else warned if p.warnings else normal).extend(shapes)
            if i == selected:
                sel_frame = (tgt, tr, half)
        self._patterns.setData(*outline_arrays(normal))
        self._warn.setData(*outline_arrays(warned))
        self._selected.setData(*outline_arrays(sel))
        self._plan_fov.setData(*outline_arrays(fovs))
        self._links.setData(*outline_arrays(links))
        for it, pts in ((self._origins, origins), (self._global, glob), (self._local, loc)):
            if pts:
                it.setData(pos=np.asarray(pts, float).reshape(-1, 2))
            else:
                it.clear()
        self._draw_wcs(sel_frame, m)
        self._boxes = boxes
        if not self._fitted and boxes:
            self._fitted = True
            self.zoom_all()

    def _draw_wcs(self, frame, m) -> None:
        """The selected field's WCS: grid and axes, as written (through its transform)."""
        if frame is None or not self.show_wcs_grid:
            self._wcs_grid.setData([], [])
            self._wcs_axes.setData([], [])
            for t in self._axis_labels:
                t.setVisible(False)
            return
        tgt, tr, half = frame
        step = nice_step(half / 4.0)
        n = int(half // step)
        ticks = np.arange(-n, n + 1) * step
        lines = []
        for v in ticks:
            lines.append(np.array([[v, -n * step], [v, n * step]]))
            lines.append(np.array([[-n * step, v], [n * step, v]]))

        def to_scs(w):
            return tgt + tr.to_bcs(w) @ m.T

        self._wcs_grid.setData(*outline_arrays([to_scs(l) for l in lines]))
        ax_len = n * step if n else half
        o, x, y = to_scs(np.array([[0.0, 0.0], [ax_len, 0.0], [0.0, ax_len]]))
        self._wcs_axes.setData(*outline_arrays([np.array([o, x]), np.array([o, y])]))
        self._axis_labels[0].setText(f"x WCS ({step:g} µm grid)")
        self._axis_labels[0].setPos(*x)
        self._axis_labels[1].setText("y WCS")
        self._axis_labels[1].setPos(*y)
        for t in self._axis_labels:
            t.setVisible(True)

    # ---------------------------------------------------------------- live state
    def set_fov(self, stage, field_um: float) -> None:
        if stage is None:
            self._fov.setData([], [])
            self._fov_label.setVisible(False)
            return
        sq = _square(stage, field_um / 2.0)
        self._fov.setData(sq[:, 0], sq[:, 1])
        self._fov_label.setVisible(True)
        self._fov_label.setPos(sq[3, 0], sq[3, 1])

    def set_stage_positions(self, positions) -> None:
        for t in self._labels:
            self.removeItem(t)
        self._labels = []
        pts = []
        for p in positions:
            pts.append((p.x, p.y))
            t = pg.TextItem(p.name, color=C_POS, anchor=(0, 1))
            t.setPos(p.x, p.y)
            self.addItem(t)
            self._labels.append(t)
        if pts:
            self._positions.setData(pos=np.asarray(pts, float))
        else:
            self._positions.clear()

    def set_stage(self, pos) -> None:
        if pos is None:
            self._stage.clear()
        else:
            self._stage.setData(pos=[np.asarray(pos, float)])

    def zoom_all(self) -> None:
        items = [i for i in (self._plan_fov, self._patterns, self._warn, self._selected, self._fov, self._positions,
                             self._global, self._local, self._stage)
                 if i.boundingRect().isValid() and not i.boundingRect().isNull()]
        self.getPlotItem().getViewBox().autoRange(items=items or None)

    # ---------------------------------------------------------------- clicks
    def _clicked(self, ev) -> None:
        vb = self.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        x, y = p.x(), p.y()
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.contextRequested.emit(x, y, ev.screenPos().toPoint())
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        hits = [(i, (x - (x0 + x1) / 2) ** 2 + (y - (y0 + y1) / 2) ** 2) for i, x0, y0, x1, y1 in self._boxes
                if x0 <= x <= x1 and y0 <= y <= y1]
        if hits:
            self.stepClicked.emit(min(hits, key=lambda h: h[1])[0])
