"""The stage view: a zoomable, pannable plot in SCS (stage µm) with the chips and their fields.

* every field of every Exposure: its field of view (dashed square at the stage target, through
  the stage → beam rotation), its pattern where the chip map puts it, a cross at its centre;
  the whole GDS – every layer, also outside the fields – is drawn gray under it (the layers the
  Exposure does not write stay gray);
* the selected step's fields: highlighted;
* chips: their CCS origin and x / y axes (through the chip map: nominal, or measured by a Global
  Align); Global Align markers (magenta ×), Local markers (cyan ×);
* marker picking (:meth:`StageView.start_pick`): the corners and centres of every polygon of the
  Exposure's GDS snap, the nearest one under the cursor is ringed;
* named stage positions, the stage (white ○) and its current field of view (green: the full
  deflection field; red: the SEM imaging scan when it is reduced or shifted).
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..coords import StageAxes
from ..geometry import Dot, Polygon
from ...core.transforms import AffineTransform
from ..schedule import Exposure, PatternSpec, exposure_fields

__all__ = ["StageView", "outline_arrays", "nice_step", "snap_targets", "SNAP_PX"]

#: a pick snaps to a target nearer than this (screen pixels)
SNAP_PX = 12.0

C_PATTERN = (90, 190, 255)
C_WARN = (255, 90, 90)
C_SELECTED = (255, 220, 40)
C_FOV = (80, 255, 120)
C_PLAN_FOV = (150, 150, 170, 110)
C_GLOBAL = (255, 80, 200)
C_LOCAL = (60, 230, 230)
C_POS = (255, 170, 60)
C_WCS = (255, 220, 40, 90)
C_OTHER = (120, 120, 128)
C_SCAN = (255, 50, 50)


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


def _centroid(ring: np.ndarray) -> np.ndarray:
    """The area centroid of a closed polygon ring (the vertex mean when it has no area)."""
    p = np.asarray(ring, float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(0)
    return np.array([((p[:, 0] + q[:, 0]) * cross).sum(), ((p[:, 1] + q[:, 1]) * cross).sum()]) / (6.0 * a)


def snap_targets(op, library) -> tuple[np.ndarray, list]:
    """Where a marker pick snaps, for the Exposure *op*: every corner and the centre of every polygon of
    its GDS (all layers, also outside its fields), dots and path points – in CCS.  Returns (points
    (N, 2), kinds: "corner" / "centre" / "point")."""
    pts, kinds = [], []
    for e in _whole_gds(op, library) if op.path else []:
        if isinstance(e, Dot):
            pts.append((e.x, e.y))
            kinds.append("point")
        elif isinstance(e, Polygon):
            ring = np.asarray(e.points, float)
            pts.extend(map(tuple, ring))
            kinds.extend(["corner"] * len(ring))
            pts.append(tuple(_centroid(ring)))
            kinds.append("centre")
        else:
            ring = np.asarray(e.points, float)
            pts.extend(map(tuple, ring))
            kinds.extend(["point"] * len(ring))
    if not pts:
        return np.zeros((0, 2)), []
    a = np.asarray(pts, float)
    _, keep = np.unique(np.round(a, 6), axis=0, return_index=True)  # shared corners once
    keep = np.sort(keep)
    return a[keep], [kinds[i] for i in keep]


def _whole_gds(op, library):
    """The top cell of the Exposure's GDS, flattened, every layer, in CCS (GDS + offset); [] when it
    cannot be read."""
    try:
        top = op.top
        if not top:
            tops = library.top_cells(op.path)
            if len(tops) != 1:
                return []
            top = tops[0]
        return library.get(PatternSpec("gds", op.path, top, None, op.dose,
                                       AffineTransform.translation(*map(float, op.offset))))
    except Exception:  # noqa: BLE001 - shown as a warning elsewhere
        return []


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
    screen_pos)`` on a right click (SCS µm of the click, global screen position for a menu); in pick
    mode ``markerPicked(x, y)`` (CCS µm) and ``pickCancelled()``."""

    stepClicked = QtCore.Signal(int)
    contextRequested = QtCore.Signal(float, float, object)
    markerPicked = QtCore.Signal(float, float)
    pickCancelled = QtCore.Signal()

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
        self._other = item(C_OTHER, 1)  # the GDS layers that are not exposed (under the exposed ones)
        self._patterns = item(C_PATTERN, 1)
        self._warn = item(C_WARN, 1.2)
        self._selected = item(C_SELECTED, 2.2)
        self._fov = item(C_FOV, 2)
        self._scan = item(C_SCAN, 2)  # the SEM imaging scan when it is a part of the field
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
        self._scan_label = pg.TextItem("imaging", color=C_SCAN, anchor=(0, 1))
        self.addItem(self._scan_label)
        self._scan_label.setVisible(False)
        self._boxes: list = []  # (index, x0, y0, x1, y1) for click hit-testing
        self._snap = pg.ScatterPlotItem(symbol="o", size=16, pen=pg.mkPen((255, 255, 0), width=2), brush=None)
        self._snap.setZValue(20)
        self.addItem(self._snap)
        self._pick = None  # pick mode: {"scs": (N, 2), "ccs": (N, 2), "kinds": [...], "to_ccs": fn}
        self.pick_banner = QtWidgets.QLabel(self)
        self.pick_banner.setStyleSheet("background:rgba(60,60,0,220);color:#ffd;padding:4px;font-weight:bold")
        self.pick_banner.hide()
        self.scene().sigMouseClicked.connect(self._clicked)
        self.scene().sigMouseMoved.connect(self._moved)
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
        other = []  # every layer of the GDS, gray: the exposed ones are drawn over them
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
            if op.path:  # the whole GDS, every layer, also what is in no field (e.g. in the top cell itself)
                other.extend(g.to_scs(s_) for s_ in _shapes(_whole_gds(op, library)))
        self._other.setData(*outline_arrays(other))
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

    def set_scan(self, stage, area, field_um: float, binv=None) -> bool:
        """The SEM imaging scan (*area*: centre XCS, width, height, rotation; see
        ``LithoHardware.scan_area``) on the stage, red – only when it is a part of the field (reduced
        or shifted); True when drawn."""
        part = False
        if stage is not None and area is not None:
            (cx, cy), w, h, rot = area
            tol = 1e-3 * max(field_um, 1e-9)
            part = w < field_um - tol or h < field_um - tol or abs(cx) > tol or abs(cy) > tol
        if not part:
            self._scan.setData([], [])
            self._scan_label.setVisible(False)
            return False
        c, s_ = math.cos(rot), math.sin(rot)
        rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])
        xcs = rect @ np.array([[c, -s_], [s_, c]]).T + (cx, cy)  # the scan in XCS ≈ BCS
        m = np.asarray(binv, float) if binv is not None else np.eye(2)
        pts = xcs @ m.T + np.asarray(stage, float)
        self._scan.setData(pts[:, 0], pts[:, 1])
        self._scan_label.setVisible(True)
        self._scan_label.setPos(pts[3, 0], pts[3, 1])
        return True

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
    # ---------------------------------------------------------------- marker picking
    @property
    def picking(self) -> bool:
        return self._pick is not None

    def start_pick(self, scs, ccs, kinds, to_ccs, text: str) -> None:
        """Pick mode: a click near one of the targets *scs* (SCS, N × 2; their CCS *ccs*) emits
        ``markerPicked`` with its CCS; Shift-click the clicked point itself (``to_ccs``: SCS → CCS);
        Esc / right click ``pickCancelled``.  *text* is shown in a banner."""
        self._pick = {"scs": np.asarray(scs, float).reshape(-1, 2), "ccs": np.asarray(ccs, float).reshape(-1, 2),
                      "kinds": list(kinds), "to_ccs": to_ccs}
        self.set_pick_text(text)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setFocus()

    def set_pick_text(self, text: str) -> None:
        self.pick_banner.setText(text + "  ·  Shift-click: the point itself · Esc: done / cancel")
        self.pick_banner.adjustSize()
        self.pick_banner.move(8, 8)
        self.pick_banner.show()
        self.pick_banner.raise_()

    def stop_pick(self) -> None:
        self._pick = None
        self._snap.clear()
        self.pick_banner.hide()
        self.unsetCursor()

    def nearest_target(self, x: float, y: float, radius_px: float = SNAP_PX):
        """The index of the pick target nearest (x, y) SCS within *radius_px* screen pixels (None: none)."""
        if self._pick is None or not len(self._pick["scs"]):
            return None
        px, py = self.getPlotItem().getViewBox().viewPixelSize()
        d = (self._pick["scs"] - (x, y)) / (max(px, 1e-12), max(py, 1e-12))
        r = np.hypot(d[:, 0], d[:, 1])
        k = int(np.argmin(r))
        return k if r[k] <= radius_px else None

    def _moved(self, pos) -> None:
        if self._pick is None:
            return
        vb = self.getPlotItem().getViewBox()
        p = vb.mapSceneToView(pos)
        k = self.nearest_target(p.x(), p.y())
        if k is None:
            self._snap.clear()
        else:
            self._snap.setData(pos=[self._pick["scs"][k]])

    def pick_at(self, x: float, y: float, free: bool = False) -> bool:
        """A pick-mode click at SCS (x, y): the nearest target (or, *free*, the point itself) → ``markerPicked``."""
        if self._pick is None:
            return False
        if free:
            c = np.asarray(self._pick["to_ccs"]((x, y)), float)
        else:
            k = self.nearest_target(x, y)
            if k is None:
                return False
            c = self._pick["ccs"][k]
        self.markerPicked.emit(float(c[0]), float(c[1]))
        return True

    def keyPressEvent(self, ev):  # noqa: N802
        if self._pick is not None and ev.key() == QtCore.Qt.Key_Escape:
            self.pickCancelled.emit()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _clicked(self, ev) -> None:
        vb = self.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        x, y = p.x(), p.y()
        if self._pick is not None:
            ev.accept()
            if ev.button() == QtCore.Qt.RightButton:
                self.pickCancelled.emit()
            elif ev.button() == QtCore.Qt.LeftButton:
                self.pick_at(x, y, free=bool(ev.modifiers() & QtCore.Qt.ShiftModifier))
            return
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
