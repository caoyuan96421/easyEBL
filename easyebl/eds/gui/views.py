"""Plot widgets of the EDS window: spectrum view, electron image with sites, map tiles, periodic table."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ... import xray
from ..spectrum import ElectronImage, Spectrum

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

__all__ = ["SpectrumView", "ElectronImageView", "MapTile", "PeriodicTable", "SITE_COLORS"]

SITE_COLORS = ["#ffd54f", "#4fc3f7", "#81c784", "#f06292", "#ba68c8", "#ff8a65", "#a1887f", "#e0e0e0"]
FAMILY_COLORS = {"K": "#ff6b6b", "L": "#4dabf7", "M": "#69db7c"}


class SpectrumView(pg.PlotWidget):
    """Spectrum display (counts vs keV) with element line markers, fit overlay and an energy window."""

    cursorMoved = QtCore.Signal(float, float)  # energy keV, counts
    windowChanged = QtCore.Signal(float, float)  # energy window (keV) dragged

    def __init__(self, parent=None):
        super().__init__(parent, background="#1e1f22")
        self.setLabel("bottom", "Energy", units="keV")
        self.setLabel("left", "Counts")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        self.addLegend(offset=(-10, 10), labelTextColor="#ddd")
        self._curves: dict = {}
        self._model = self.plot([], [], pen=pg.mkPen("#ff922b", width=1.2), name="fit")
        self._bg = self.plot([], [], pen=pg.mkPen("#868e96", width=1, style=QtCore.Qt.DashLine), name="background")
        self._markers: list = []
        self._labels: list = []
        self.log_y = False
        self.beam_kv = 20.0
        self._region = pg.LinearRegionItem(values=(6.2, 6.6), brush=pg.mkBrush(255, 255, 255, 25),
                                           pen=pg.mkPen("#aaa"))
        self._region.setVisible(False)
        self._region.sigRegionChangeFinished.connect(lambda r: self.windowChanged.emit(*r.getRegion()))
        self.addItem(self._region)
        self._cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#666", width=1))
        self.addItem(self._cursor, ignoreBounds=True)
        self.scene().sigMouseMoved.connect(self._moved)
        self.setXRange(0, 10, padding=0)
        self._primary: Spectrum | None = None

    # ----------------------------------------------------------------- data
    def set_spectrum(self, key: str, sp: Spectrum | None, color: str = "#ffd54f", name: str | None = None,
                     primary: bool = True) -> None:
        if sp is None:
            c = self._curves.pop(key, None)
            if c is not None:
                self.removeItem(c)
            return
        e = sp.energies
        y = sp.counts.astype(float)
        if key not in self._curves:
            self._curves[key] = self.plot([], [], pen=pg.mkPen(color, width=1.3), stepMode="center",
                                          fillLevel=None, name=name or key)
        de = sp.calibration.gain_kev
        edges = np.concatenate([e - de / 2, [e[-1] + de / 2]])
        self._curves[key].setData(edges, np.maximum(y, 0.5) if self.log_y else y)
        if primary:
            self._primary = sp
            self.beam_kv = sp.beam_kv

    def clear_spectra(self) -> None:
        for c in self._curves.values():
            self.removeItem(c)
        self._curves.clear()
        self._primary = None
        self.set_fit(None, None)

    def set_fit(self, model, background, energies=None) -> None:
        if model is None:
            self._model.setData([], [])
            self._bg.setData([], [])
            return
        e = energies if energies is not None else (self._primary.energies if self._primary else np.arange(len(model)))
        lift = (lambda a: np.maximum(a, 0.5)) if self.log_y else (lambda a: a)
        self._model.setData(e, lift(np.asarray(model, float)))
        self._bg.setData(e, lift(np.asarray(background, float)))

    def set_log(self, on: bool) -> None:
        self.log_y = bool(on)
        self.setLogMode(y=self.log_y)
        sp = self._primary
        for k, c in list(self._curves.items()):
            x, y = c.getData()
            if y is not None and len(y):
                c.setData(x, np.maximum(y, 0.5) if on else y)
        if sp is not None:
            self.autoscale()

    def autoscale(self) -> None:
        sp = self._primary
        if sp is None:
            return
        e = sp.energies
        m = (e > 0.15) & (e < min(sp.beam_kv, e[-1]))
        top = float(sp.counts[m].max()) if m.any() else 1.0
        self.setXRange(0.0, min(sp.beam_kv, float(e[-1])), padding=0.01)
        if self.log_y:
            self.setYRange(0, np.log10(max(top, 10)) * 1.08, padding=0)
        else:
            self.setYRange(0, max(top, 5) * 1.1, padding=0)
        self._place_labels()

    # ------------------------------------------------------------- markers
    def set_markers(self, elements, beam_kv: float | None = None) -> None:
        """KLM line markers of *elements* (heights ∝ relative weight), labelled at the main line."""
        for it in self._markers + self._labels:
            self.removeItem(it)
        self._markers, self._labels = [], []
        kv = beam_kv or self.beam_kv
        for el in elements:
            for fam in xray.families(el, kv):
                col = FAMILY_COLORS.get(fam, "#ccc")
                for ln in xray.lines(el, fam, kv, min_weight=0.05):
                    item = pg.PlotCurveItem([ln.energy, ln.energy], [0, 1], pen=pg.mkPen(col, width=1.2))
                    item._weight = ln.weight
                    self.addItem(item)
                    self._markers.append(item)
                main = xray.main_line(el, fam)
                if main is not None and main.energy < kv:
                    t = pg.TextItem(f"{el} {fam}", color=col, anchor=(0.5, 1.0))
                    t._energy = main.energy
                    self.addItem(t)
                    self._labels.append(t)
        self._place_labels()

    def _place_labels(self) -> None:
        sp = self._primary
        (x0, x1), (y0, y1) = self.viewRange()
        top = y1 if not self.log_y else y1
        for it in self._markers:
            h = it._weight
            if sp is not None:
                e = it.xData[0]
                k = int(np.clip(round(sp.calibration.channel(e)), 0, sp.n_channels - 1))
                local = float(sp.counts[max(0, k - 3):k + 4].max())
                yv = max(local, 1.0)
                if self.log_y:
                    yv = np.log10(max(yv, 1.0))
                it.setData([e, e], [y0, y0 + (yv - y0) * (0.25 + 0.75 * h) if yv > y0 else y0 + (top - y0) * 0.1 * h])
        for t in self._labels:
            e = t._energy
            yv = top * 0.9
            if sp is not None:
                k = int(np.clip(round(sp.calibration.channel(e)), 0, sp.n_channels - 1))
                local = float(sp.counts[max(0, k - 3):k + 4].max())
                yv = np.log10(max(local, 1.0)) if self.log_y else local
                yv = min(max(yv, y0 + 0.05 * (top - y0)) + 0.04 * (top - y0), top * 0.98)
            t.setPos(e, yv)

    # ------------------------------------------------------------- window
    def show_window(self, on: bool, lo: float | None = None, hi: float | None = None) -> None:
        if lo is not None and hi is not None:
            self._region.setRegion((lo, hi))
        self._region.setVisible(on)

    def window(self) -> tuple:
        return tuple(self._region.getRegion())

    def _moved(self, pos):
        vb = self.getPlotItem().vb
        if not self.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        self._cursor.setPos(p.x())
        c = 0.0
        if self._primary is not None:
            k = int(round(self._primary.calibration.channel(p.x())))
            if 0 <= k < self._primary.n_channels:
                c = float(self._primary.counts[k])
        self.cursorMoved.emit(float(p.x()), c)


class ElectronImageView(pg.GraphicsLayoutWidget):
    """Electron image (µm coordinates) with acquisition sites and drawing tools.

    Tools: ``point``, ``rect``, ``ellipse``, ``line`` (click/drag on the image);
    ``siteRequested(kind, params)`` is emitted with µm parameters:
    point (x, y) · rect/ellipse (cx, cy, w, h) · line (x0, y0, x1, y1).
    """

    siteRequested = QtCore.Signal(str, tuple)
    pointerMoved = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#111")
        self.vb = self.addViewBox(lockAspect=True, enableMenu=False)
        self.vb.invertY(False)
        self.img = pg.ImageItem()
        self.overlay = pg.ImageItem()
        self.overlay.setOpacity(0.55)
        self.vb.addItem(self.img)
        self.vb.addItem(self.overlay)
        self.tool = "point"
        self._items: dict = {}
        self._drag = None
        self._drag_item = None
        self.region_um = (0.0, 0.0, 100.0, 100.0)
        self.scene().sigMouseMoved.connect(self._moved)
        self._levels = None

    def set_image(self, im: ElectronImage | np.ndarray, region_um=None, levels=None) -> None:
        if isinstance(im, ElectronImage):
            data, region_um = im.data, im.region_um
        else:
            data = np.asarray(im, float)
        region_um = tuple(region_um or self.region_um)
        cx, cy, w, h = region_um
        d = np.nan_to_num(np.asarray(data, float), nan=float(np.nanmin(data)) if np.isfinite(data).any() else 0.0)
        if levels is None:
            lo, hi = np.percentile(d, [0.5, 99.5]) if d.size else (0, 1)
            levels = (lo, hi if hi > lo else lo + 1)
        self.img.setImage(d[::-1], levels=levels, autoLevels=False)
        self.img.setRect(QtCore.QRectF(cx - w / 2, cy - h / 2, w, h))
        if region_um != self.region_um or self._levels is None:
            self.vb.setRange(QtCore.QRectF(cx - w / 2, cy - h / 2, w, h), padding=0.01)
        self.region_um = region_um
        self._levels = levels

    def set_overlay(self, data: np.ndarray | None, region_um=None, color=(255, 80, 80)) -> None:
        if data is None:
            self.overlay.clear()
            return
        cx, cy, w, h = region_um or self.region_um
        d = np.nan_to_num(np.asarray(data, float))
        hi = np.percentile(d, 99.5) if d.size else 1.0
        a = np.clip(d / (hi if hi > 0 else 1.0), 0, 1)
        rgba = np.zeros(d.shape + (4,), np.ubyte)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
        rgba[..., 3] = (255 * a).astype(np.ubyte)
        self.overlay.setImage(rgba[::-1])
        self.overlay.setRect(QtCore.QRectF(cx - w / 2, cy - h / 2, w, h))

    # ---------------------------------------------------------------- sites
    def add_site(self, key, kind: str, params: tuple, color: str, label: str = "") -> None:
        self.remove_site(key)
        pen = pg.mkPen(color, width=2)
        pen.setCosmetic(True)
        if kind == "point":
            x, y = params
            it = pg.ScatterPlotItem([x], [y], symbol="+", size=16, pen=pen, brush=None)
        elif kind in ("rect", "map"):
            cx, cy, w, h = params
            it = QtWidgets.QGraphicsRectItem(cx - w / 2, cy - h / 2, w, h)
            if kind == "map":
                pen.setStyle(QtCore.Qt.DashLine)
            it.setPen(pen)
        elif kind == "ellipse":
            cx, cy, w, h = params
            it = QtWidgets.QGraphicsEllipseItem(cx - w / 2, cy - h / 2, w, h)
            it.setPen(pen)
        elif kind == "line":
            x0, y0, x1, y1 = params
            it = pg.PlotCurveItem([x0, x1], [y0, y1], pen=pen)
        else:
            return
        self.vb.addItem(it)
        txt = None
        if label:
            txt = pg.TextItem(label, color=color, anchor=(0, 1))
            px, py = (params[0], params[1]) if kind != "line" else (params[2], params[3])
            if kind in ("rect", "ellipse", "map"):
                px, py = params[0] - params[2] / 2, params[1] + params[3] / 2
            txt.setPos(px, py)
            self.vb.addItem(txt)
        self._items[key] = (it, txt)

    def remove_site(self, key) -> None:
        old = self._items.pop(key, None)
        if old:
            for it in old:
                if it is not None:
                    self.vb.removeItem(it)

    def clear_sites(self) -> None:
        for k in list(self._items):
            self.remove_site(k)

    # ---------------------------------------------------------------- mouse
    def _to_um(self, scene_pos) -> tuple:
        p = self.vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _moved(self, pos):
        x, y = self._to_um(pos)
        self.pointerMoved.emit(x, y)
        if self._drag is not None and self._drag_item is not None:
            x0, y0 = self._drag
            if self.tool == "line":
                self._drag_item.setData([x0, x], [y0, y])
            else:
                r = QtCore.QRectF(QtCore.QPointF(min(x0, x), min(y0, y)), QtCore.QPointF(max(x0, x), max(y0, y)))
                self._drag_item.setRect(r)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.tool in ("point", "rect", "ellipse", "line", "map"):
            x, y = self._to_um(self.mapToScene(ev.position().toPoint()))
            if self.tool == "point":
                self.siteRequested.emit("point", (x, y))
                ev.accept()
                return
            self._drag = (x, y)
            pen = pg.mkPen("#fff", width=1, style=QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            if self.tool == "line":
                self._drag_item = pg.PlotCurveItem([x, x], [y, y], pen=pen)
            elif self.tool == "ellipse":
                self._drag_item = QtWidgets.QGraphicsEllipseItem(x, y, 0, 0)
                self._drag_item.setPen(pen)
            else:
                self._drag_item = QtWidgets.QGraphicsRectItem(x, y, 0, 0)
                self._drag_item.setPen(pen)
            self.vb.addItem(self._drag_item)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and ev.button() == QtCore.Qt.LeftButton:
            x0, y0 = self._drag
            x1, y1 = self._to_um(self.mapToScene(ev.position().toPoint()))
            self.vb.removeItem(self._drag_item)
            self._drag = self._drag_item = None
            if self.tool == "line":
                if np.hypot(x1 - x0, y1 - y0) > 0:
                    self.siteRequested.emit("line", (x0, y0, x1, y1))
            else:
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w > 0 and h > 0:
                    self.siteRequested.emit(self.tool, ((x0 + x1) / 2, (y0 + y1) / 2, w, h))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class MapTile(QtWidgets.QFrame):
    """One element map with its label and a colour scale."""

    clicked = QtCore.Signal(int, int)  # row, col

    def __init__(self, label: str, data: np.ndarray, unit: str, color=(255, 255, 255), parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel(f"<b>{label}</b> <span style='color:#999'>({unit})</span>")
        lay.addWidget(self.title)
        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground("#111")
        self.vb = self.gw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb.setMouseEnabled(False, False)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        lay.addWidget(self.gw)
        self.data = np.asarray(data, float)
        self.color = color
        d = np.nan_to_num(self.data)
        hi = np.percentile(d, 99.5) if d.size else 1.0
        lo = min(0.0, float(np.percentile(d, 0.5)))
        lut = np.zeros((256, 3), np.ubyte)
        t = np.linspace(0, 1, 256)
        for i in range(3):
            lut[:, i] = (t * color[i]).astype(np.ubyte)
        self.img.setImage(d[::-1], levels=(lo, hi if hi > lo else lo + 1), lut=lut)
        self.vb.autoRange(padding=0)
        self.setMinimumSize(150, 150)
        self.gw.scene().sigMouseClicked.connect(self._click)

    def _click(self, ev):
        p = self.vb.mapSceneToView(ev.scenePos())
        ny, nx = self.data.shape
        col, row_flip = int(np.floor(p.x())), int(np.floor(p.y()))
        row = ny - 1 - row_flip
        if 0 <= row < ny and 0 <= col < nx:
            self.clicked.emit(row, col)


class PeriodicTable(QtWidgets.QDialog):
    """Pick elements (toggle buttons in periodic-table layout)."""

    _LAYOUT = [
        "H . . . . . . . . . . . . . . . . He",
        "Li Be . . . . . . . . . . B C N O F Ne",
        "Na Mg . . . . . . . . . . Al Si P S Cl Ar",
        "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
        "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
        "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
        ". . Ac . . . . . . . . . . . . . . .",
        ". . Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu . .",
        ". . Th Pa U Np Pu . . . . . . . . . . .",
    ]

    def __init__(self, selected=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elements")
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(2)
        self.buttons = {}
        for r, row in enumerate(self._LAYOUT):
            for c, sym in enumerate(row.split()):
                if sym == ".":
                    continue
                b = QtWidgets.QToolButton()
                b.setText(sym)
                b.setCheckable(True)
                b.setFixedSize(34, 28)
                b.setChecked(sym in selected)
                if sym in ("H", "He", "Li", "Ne", "Ar", "Kr", "Xe", "Rn"):
                    b.setEnabled(False)
                b.setStyleSheet("QToolButton:checked{background:#2a82da;color:white;font-weight:bold}")
                grid.addWidget(b, r + (1 if r >= 7 else 0), c)
                self.buttons[sym] = b
        grid.setRowMinimumHeight(7, 8)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        clear = bb.addButton("Clear", QtWidgets.QDialogButtonBox.ResetRole)
        clear.clicked.connect(lambda: [b.setChecked(False) for b in self.buttons.values()])
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(bb)

    def selected(self) -> list:
        return [s for s, b in self.buttons.items() if b.isChecked()]
