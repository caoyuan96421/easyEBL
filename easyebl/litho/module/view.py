"""The stage view: a zoomable, pannable plot in SCS (stage µm) with the chips and their fields.

* every field of every Exposure: its field of view (dashed square at the stage target, through
  the stage → beam rotation), its pattern where the chip map puts it, a cross at its centre;
* the selected step's fields: highlighted;
* chips: their CCS origin and x / y axes (through the chip map: nominal, or measured by a Global
  Align); Global Align markers (magenta ×), Local markers (cyan ×);
* named stage positions, the stage (white ○) and its current field of view (green).
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from ..coords import StageAxes
from ..geometry import Dot, Polygon
from ..schedule import Exposure, exposure_fields

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


def _beam_square(centre, half, binv=None) -> np.ndarray:
    """The deflection field (a square in BCS) on the stage: *centre* + B⁻¹·square."""
    sq = _square((0.0, 0.0), half)
    if binv is not None:
        sq = sq @ np.asarray(binv, float).T
    return sq + np.asarray(centre, float)


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

        def item(color, width=1.0, style=None):
            pen = pg.mkPen(color, width=width, style=style) if style is not None else pg.mkPen(color, width=width)
            it = pg.PlotDataItem(pen=pen, connect="finite")
            self.addItem(it)
            return it

        self._plan_fov = item(C_PLAN_FOV, 1, QtCore.Qt.DashLine)
        self._links = item(C_WCS, 1.5)  # the chips' CCS axes
        self._patterns = item(C_PATTERN, 1)
        self._warn = item(C_WARN, 1.2)
        self._selected = item(C_SELECTED, 2.2)
        self._fov = item(C_FOV, 2)
        self._origins = pg.ScatterPlotItem(symbol="+", size=9, pen=pg.mkPen(C_PLAN_FOV[:3], width=1), brush=None)
        self._global = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen(C_GLOBAL, width=2), brush=None)
        self._local = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen(C_LOCAL, width=2), brush=None)
        self._positions = pg.ScatterPlotItem(symbol="s", size=10, pen=pg.mkPen(C_POS), brush=pg.mkBrush(C_POS))
        self._stage = pg.ScatterPlotItem(symbol="o", size=16, pen=pg.mkPen((255, 255, 255), width=2), brush=None)
        self._chips = pg.ScatterPlotItem(symbol="o", size=10, pen=pg.mkPen(C_SELECTED, width=2), brush=None)
        self._chip_labels: list = []
        for it in (self._origins, self._global, self._local, self._positions, self._stage, self._chips):
            self.addItem(it)
        self._labels: list = []
        self._fov_label = pg.TextItem("FoV", color=C_FOV, anchor=(0, 1))
        self.addItem(self._fov_label)
        self._boxes: list = []  # (index, x0, y0, x1, y1) for click hit-testing
        self.scene().sigMouseClicked.connect(self._clicked)
        self._fitted = False
        self.flipped = (False, False)

    def set_orientation(self, axes: StageAxes, as_sem: bool = True) -> None:
        """*as_sem*: flip the mirrored stage axes so the plot looks like the SEM image (CCS x right,
        y up); the axis numbers stay stage coordinates (they then run right to left / top down)."""
        fx, fy = (bool(as_sem and axes.mirror_x), bool(as_sem and axes.mirror_y))
        if (fx, fy) == self.flipped:
            return
        vb = self.getPlotItem().getViewBox()
        vb.invertX(fx)
        vb.invertY(fy)
        self.flipped = (fx, fy)
        self.setLabel("bottom", "stage X (SCS)" + ("  ← increasing" if fx else ""), "µm")
        self.setLabel("left", "stage Y (SCS)" + ("  ↓ increasing" if fy else ""), "µm")

    # ---------------------------------------------------------------- schedule
    def set_schedule(self, schedule, library, plans, axes: StageAxes, selected: int | None = None,
                     chips: dict | None = None) -> None:
        """*chips*: {chip name: measured ChipTransform} (else nominal)."""
        normal, warned, sel, fovs, origins, glob, loc, axes_lines = [], [], [], [], [], [], [], []
        boxes = []
        chip_pts = []
        for c in schedule.chips:
            g = (chips or {}).get(c.name) or c.nominal(axes)
            o, x, y = g.to_scs(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
            span = 50.0
            axes_lines.append(np.array([o, o + (x - o) * span]))
            axes_lines.append(np.array([o, o + (y - o) * span]))
            chip_pts.append((c.name, o, g.measured))
        for i, op in enumerate(schedule.operations):
            p = plans[i] if plans is not None and i < len(plans) else None
            if not isinstance(op, Exposure) or p is None or not p.fields or p.chip is None:
                continue
            g = p.chip
            half = p.field_um / 2.0
            binv = np.linalg.inv(p.stage_to_beam) if p.stage_to_beam is not None else axes.matrix
            for m in (op.global_markers if op.global_align else []):
                glob.append(g.to_scs(np.asarray(m, float)))
            for m in (op.local_markers if op.local_align else []):
                loc.append(g.to_scs(np.asarray(m, float)))
            try:
                defs = exposure_fields(op, library)
            except Exception:  # noqa: BLE001 - a warning in the list
                defs = []
            for fp in p.fields:
                tgt = np.asarray(fp.target, float)
                fovs.append(_beam_square(tgt, half, binv))
                boxes.append((i, tgt[0] - half, tgt[1] - half, tgt[0] + half, tgt[1] + half))
                origins.append(tgt)
                if op.alignment_only or fp.index >= len(defs):
                    continue
                try:
                    pat = library.get(defs[fp.index].spec(op))
                except Exception:  # noqa: BLE001
                    pat = []
                c = np.asarray(fp.centre_ccs, float)
                shapes = [g.to_scs(s_ + c) for s_ in _shapes(pat)]
                (sel if i == selected else warned if fp.warnings else normal).extend(shapes)
        self._patterns.setData(*outline_arrays(normal))
        self._warn.setData(*outline_arrays(warned))
        self._selected.setData(*outline_arrays(sel))
        self._plan_fov.setData(*outline_arrays(fovs))
        self._links.setData(*outline_arrays(axes_lines))
        for it, pts in ((self._origins, origins), (self._global, glob), (self._local, loc)):
            if pts:
                it.setData(pos=np.asarray(pts, float).reshape(-1, 2))
            else:
                it.clear()
        self._draw_chips(chip_pts)
        self._boxes = boxes
        if not self._fitted and boxes:
            self._fitted = True
            self.zoom_all()

    def _draw_chips(self, chips) -> None:
        for t in self._chip_labels:
            self.removeItem(t)
        self._chip_labels = []
        pts = []
        for name, o, measured in chips:
            pts.append(o)
            t = pg.TextItem(f"{name} (CCS{' aligned' if measured else ''})", color=C_SELECTED, anchor=(0, 0))
            t.setPos(*o)
            self.addItem(t)
            self._chip_labels.append(t)
        if pts:
            self._chips.setData(pos=np.asarray(pts, float).reshape(-1, 2))
        else:
            self._chips.clear()

    # ---------------------------------------------------------------- live state
    def set_fov(self, stage, field_um: float, binv=None) -> None:
        if stage is None:
            self._fov.setData([], [])
            self._fov_label.setVisible(False)
            return
        sq = _beam_square(stage, field_um / 2.0, binv)
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
                             self._global, self._local, self._stage, self._chips)
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
