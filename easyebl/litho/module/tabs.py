"""Side tabs of the litho window: Stage Positions, Chips and Exposure Schedule."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..height import height_um
from ..schedule import (KEEP_AS_IS, MAGNIFICATIONS, ORDERS, BeamCalibrationStep, ColumnSetting, Exposure,
                        FieldCorrection, Operation, dose_readout)

__all__ = ["StagePositionsTab", "PositionsHub", "ChipsTab", "HeightDialog", "ScheduleTab", "OpEditor"]

STATE_COLORS = {"pending": None, "running": "#fd4", "done": "#5c5", "skipped": "#999", "failed": "#f55",
                "aborted": "#f95"}
STATE_MARK = {"pending": "", "running": "▶ ", "done": "✓ ", "skipped": "– ", "failed": "✗ ", "aborted": "■ "}


def _spin(value=0.0, lo=-1e9, hi=1e9, decimals=3, suffix=" µm", step=1.0) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setSingleStep(step)
    s.setValue(float(value))
    s.setKeyboardTracking(False)
    return s


def _pair(x, y, setter, decimals=3) -> QtWidgets.QWidget:
    sx, sy = _spin(x, decimals=decimals), _spin(y, decimals=decimals)
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(sx)
    h.addWidget(sy)
    upd = lambda *_: setter((float(sx.value()), float(sy.value())))  # noqa: E731
    sx.valueChanged.connect(upd)
    sy.valueChanged.connect(upd)
    w.spins = (sx, sy)
    return w


def _table(cols) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.horizontalHeader().setStretchLastSection(True)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    t.verticalHeader().setVisible(False)
    return t


def _selected_row(table) -> int:
    rows = table.selectionModel().selectedRows()
    return rows[0].row() if rows else -1


# ====================================================================== stage positions
class PositionsHub(QtCore.QObject):
    """Tells every Stage Positions tab that the positions of a settings object changed (the
    Lithography and the Microscope windows show the same list)."""

    changed = QtCore.Signal(object)  # the LithoSettings

    _instance = None

    @classmethod
    def instance(cls) -> "PositionsHub":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def _here(stage_position, stage_z, wd):
    """(x, y, h) of the stage now (in focus: h = WD − Z), or an error text."""
    pos, z, w = stage_position(), stage_z(), wd()
    if pos is None or z is None:
        return "the stage position is not known (microscope not connected?)"
    if w is None:
        return "the working distance is not known (microscope not connected?)"
    return float(pos[0]), float(pos[1]), height_um(w, z)


class StagePositionsTab(QtWidgets.QWidget):
    """Named stage positions of fixed objects (calibration samples, Faraday cup …): X, Y and the
    height H = WD − Z (µm), recorded with the object in focus; a move to one sets Z so that it is in
    focus at the column's present WD.  Saved at once; every tab on the same settings follows."""

    changed = QtCore.Signal()
    gotoRequested = QtCore.Signal(float, float, object)  # SCS µm, Z µm (None: keep Z)

    def __init__(self, settings, stage_position=None, stage_z=None, wd=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.stage_position = stage_position or (lambda: None)  # () -> (x, y) | None
        self.stage_z = stage_z or (lambda: None)  # () -> z µm | None
        self.wd = wd or (lambda: None)  # () -> the column's WD (mm) | None
        self.table = _table(["Name", "X (µm)", "Y (µm)", "H = WD − Z (µm)", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        b_cur = QtWidgets.QPushButton("Add current stage")
        b_cur.setToolTip("a new position at the stage's X, Y; its height from the stage Z and the WD now – focus "
                         "on the object first")
        b_cur.clicked.connect(self.add_current)
        b_add = QtWidgets.QPushButton("Add…")
        b_add.clicked.connect(self.add_manual)
        self.b_replace = QtWidgets.QPushButton("Replace with current")
        self.b_replace.setToolTip("the selected position becomes the stage's X, Y and height now (in focus)")
        self.b_replace.clicked.connect(self.replace_selected)
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_del.clicked.connect(self.delete_selected)
        self.b_go = QtWidgets.QPushButton("Go to (in focus)")
        self.b_go.setToolTip("move the stage to the selected X, Y and to the Z that has it in focus at the WD now")
        self.b_go.clicked.connect(lambda: self._goto(True))
        self.b_go_xy = QtWidgets.QPushButton("Go to XY")
        self.b_go_xy.setToolTip("move the stage to the selected X, Y; Z stays")
        self.b_go_xy.clicked.connect(lambda: self._goto(False))
        g = QtWidgets.QGridLayout()
        g.addWidget(b_cur, 0, 0)
        g.addWidget(b_add, 0, 1)
        g.addWidget(self.b_replace, 0, 2)
        g.addWidget(self.b_go, 1, 0)
        g.addWidget(self.b_go_xy, 1, 1)
        g.addWidget(self.b_del, 1, 2)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addLayout(g)
        lay.addWidget(self.msg)
        PositionsHub.instance().changed.connect(self._hub_changed)
        self.refresh()

    def _hub_changed(self, settings) -> None:
        if settings is self.settings:
            sel = _selected_row(self.table)
            self.refresh()
            if 0 <= sel < self.table.rowCount():
                self.table.selectRow(sel)
            self.changed.emit()

    def refresh(self) -> None:
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.settings.stage_positions))
            for r, p in enumerate(self.settings.stage_positions):
                vals = (p.name, f"{p.x:.2f}", f"{p.y:.2f}", "" if p.h is None else f"{p.h:.1f}", p.note)
                for c, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if c == 3 and p.h is None:
                        it.setBackground(QtGui.QColor("#622"))
                        it.setToolTip("no height: Replace with current (with the object in focus)")
                    self.table.setItem(r, c, it)
        finally:
            self.table.blockSignals(False)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        p = self._selected()
        for b in (self.b_replace, self.b_del, self.b_go_xy):
            b.setEnabled(p is not None)
        self.b_go.setEnabled(p is not None and p.h is not None)

    def _selected(self):
        r = _selected_row(self.table)
        return self.settings.stage_positions[r] if 0 <= r < len(self.settings.stage_positions) else None

    def _save(self, text: str = "") -> None:
        try:
            self.settings.save()
            self.msg.setText(text)
            self.msg.setStyleSheet("")
        except OSError as e:
            self.msg.setText(f"not saved: {e}")
            self.msg.setStyleSheet("color:#f66")
        PositionsHub.instance().changed.emit(self.settings)  # this tab and the others refresh

    def _error(self, text: str) -> None:
        self.msg.setText(text)
        self.msg.setStyleSheet("color:#f66")

    def _edited(self, item) -> None:
        r, c = item.row(), item.column()
        p = self.settings.stage_positions[r]
        txt = item.text().strip()
        try:
            if c == 0:
                if not txt:
                    raise ValueError("a stage position needs a name")
                if txt != p.name and self.settings.find(txt) >= 0:
                    raise ValueError(f"a stage position named {txt!r} exists already")
                p.name = txt
            elif c in (1, 2):
                v = float(txt)
                if c == 1:
                    p.x = v
                else:
                    p.y = v
            elif c == 3:
                p.h = float(txt) if txt else None
            else:
                p.note = txt
        except ValueError as e:
            self.refresh()
            return self._error(str(e))
        self._save("saved")

    def add(self, name: str, x: float, y: float, h: float | None = None) -> None:
        try:
            self.settings.add_position(name, x, y, h)
        except ValueError as e:
            return self._error(str(e))
        self._save(f"added {name!r}")
        self.table.selectRow(len(self.settings.stage_positions) - 1)

    def add_current(self, name: str | None = None) -> None:
        here = _here(self.stage_position, self.stage_z, self.wd)
        if isinstance(here, str):
            return self._error(here)
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Stage position", "Name:",
                                                      text=self.settings.unique_name())
            if not ok:
                return
        self.add(name, *here)

    def replace_selected(self, confirm: bool = True) -> None:
        p = self._selected()
        if p is None:
            return
        here = _here(self.stage_position, self.stage_z, self.wd)
        if isinstance(here, str):
            return self._error(here)
        x, y, h = here
        if confirm and QtWidgets.QMessageBox.question(
                self, "Stage position", f"Replace {p.name!r} with the stage now: ({x:.2f}, {y:.2f}) µm, "
                                        f"H {h:.1f} µm?\n\n(The object should be in focus.)") \
                != QtWidgets.QMessageBox.Yes:
            return
        r = _selected_row(self.table)
        p.x, p.y, p.h = x, y, h
        self._save(f"{p.name!r}: ({x:.2f}, {y:.2f}) µm, H {h:.1f} µm")
        self.table.selectRow(r)

    def add_manual(self) -> None:
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Stage position")
        f = QtWidgets.QFormLayout(d)
        name = QtWidgets.QLineEdit(self.settings.unique_name())
        x, y = _spin(0, 0, 1e6, 2), _spin(0, 0, 1e6, 2)
        h = _spin(0, -1e6, 1e6, 1)
        h.setToolTip("WD − Z in µm with the object in focus (WD in mm × 1000)")
        f.addRow("name", name)
        f.addRow("X", x)
        f.addRow("Y", y)
        f.addRow("H = WD − Z", h)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.add(name.text(), x.value(), y.value(), h.value())

    def delete_selected(self) -> None:
        r = _selected_row(self.table)
        if r < 0:
            return
        name = self.settings.stage_positions[r].name
        del self.settings.stage_positions[r]
        self._save(f"deleted {name!r}")

    def _goto(self, in_focus: bool) -> None:
        p = self._selected()
        if p is None:
            return
        z = None
        if in_focus:
            if p.h is None:
                return self._error(f"{p.name!r} has no height: Go to XY, focus, then Replace with current")
            z = p.z_at(self.wd())
            if z is None:
                return self._error("the working distance is not known: cannot compute the Z")
        self.gotoRequested.emit(p.x, p.y, z)


# ====================================================================== chips
class ChipsTab(QtWidgets.QWidget):
    """The schedule's chips: name, the nominal stage position of the CCS origin, the alignment, the
    height (H = WD − Z: none, constant or a height map)."""

    changed = QtCore.Signal()
    resetRequested = QtCore.Signal(str)  # forget the chip's Global Align
    moveRequested = QtCore.Signal(str)  # Move the chip… dialog
    heightRequested = QtCore.Signal(str)  # Height… dialog
    selectionChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.schedule = None
        self.stage_position = lambda: None
        self.alignment = lambda name: ""  # () -> status text of a chip's CCS → SCS map
        self.table = _table(["Name", "Origin X (µm)", "Origin Y (µm)", "Alignment", "Height", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(lambda: self.selectionChanged.emit(self.selected()))
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        info = QtWidgets.QLabel("A chip has its own coordinates (CCS, the GDS top level of its Exposures + their "
                                "offsets).  The origin is where CCS (0, 0) nominally is on the stage; a Global Align "
                                "measures the real map, used by every later Exposure on the chip.  Its height "
                                "(H = WD − Z) brings it into focus with the stage Z alone, at the schedule's WD.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        b_add = QtWidgets.QPushButton("Add chip")
        b_add.setToolTip("a new chip with its origin at the current stage position (0, 0 if not known)")
        b_add.clicked.connect(self.add_chip)
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_del.clicked.connect(self.delete_selected)
        self.b_here = QtWidgets.QPushButton("Origin = stage")
        self.b_here.setToolTip("put the selected chip's origin at the current stage position")
        self.b_here.clicked.connect(self.origin_to_stage)
        self.b_move = QtWidgets.QPushButton("Move…")
        self.b_move.clicked.connect(lambda: self.selected() and self.moveRequested.emit(self.selected()))
        self.b_reset = QtWidgets.QPushButton("Reset alignment")
        self.b_reset.setToolTip("forget the selected chip's Global Align (back to the nominal origin)")
        self.b_reset.clicked.connect(lambda: self.selected() and self.resetRequested.emit(self.selected()))
        self.b_height = QtWidgets.QPushButton("Height…")
        self.b_height.setToolTip("the chip's height: kept Z, one height (focus on a feature), or a height map "
                                 "(≥ 3 points)")
        self.b_height.clicked.connect(lambda: self.selected() and self.heightRequested.emit(self.selected()))
        g = QtWidgets.QGridLayout()
        g.addWidget(b_add, 0, 0)
        g.addWidget(self.b_del, 0, 1)
        g.addWidget(self.b_here, 0, 2)
        g.addWidget(self.b_move, 1, 0)
        g.addWidget(self.b_reset, 1, 1)
        g.addWidget(self.b_height, 1, 2)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.table, 1)
        lay.addLayout(g)
        lay.addWidget(self.msg)

    def set_schedule(self, schedule) -> None:
        self.schedule = schedule
        self.refresh()

    def refresh(self, select: str | None = None) -> None:
        if self.schedule is None:
            return
        cur = select if select is not None else self.selected()
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.schedule.chips))
            for r, c in enumerate(self.schedule.chips):
                vals = (c.name, f"{c.origin[0]:.3f}", f"{c.origin[1]:.3f}", self.alignment(c.name),
                        c.height.describe(), c.note)
                for k, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if k in (3, 4):
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    if k == 4 and (c.height.mode == "none" or c.height.problem()):
                        it.setForeground(QtGui.QBrush(QtGui.QColor("#fb4")))
                    self.table.setItem(r, k, it)
        finally:
            self.table.blockSignals(False)
        names = [c.name for c in self.schedule.chips]
        if cur in names:
            self.table.selectRow(names.index(cur))
        elif names:
            self.table.selectRow(0)

    def selected(self) -> str:
        r = _selected_row(self.table)
        return self.schedule.chips[r].name if self.schedule is not None and 0 <= r < len(self.schedule.chips) else ""

    def select(self, name: str) -> None:
        names = [c.name for c in self.schedule.chips] if self.schedule is not None else []
        if name in names:
            self.table.selectRow(names.index(name))

    def _error(self, text: str) -> None:
        self.msg.setText(text)
        self.msg.setStyleSheet("color:#f66")

    def _ok(self, text: str) -> None:
        self.msg.setText(text)
        self.msg.setStyleSheet("")
        self.changed.emit()

    def _edited(self, item) -> None:
        r, k = item.row(), item.column()
        c = self.schedule.chips[r]
        txt = item.text().strip()
        try:
            if k == 0:
                self.schedule.rename_chip(c.name, txt)
            elif k in (1, 2):
                v = float(txt)
                c.origin = (v, c.origin[1]) if k == 1 else (c.origin[0], v)
            elif k == 5:
                c.note = txt
        except ValueError as e:
            self.refresh()
            return self._error(str(e))
        self.refresh(select=c.name)
        self._ok("")

    def add_chip(self, origin=None) -> str:
        pos = origin if origin is not None else (self.stage_position() or (0.0, 0.0))
        c = self.schedule.add_chip(origin=pos)
        self.refresh(select=c.name)
        self._ok(f"added {c.name!r} at ({pos[0]:.2f}, {pos[1]:.2f}) µm")
        return c.name

    def delete_selected(self) -> None:
        name = self.selected()
        if not name:
            return
        used = [e for e in self.schedule.exposures if e.chip == name]
        if used:
            return self._error(f"{name!r} is used by {len(used)} Exposure(s)")
        self.schedule.chips = [c for c in self.schedule.chips if c.name != name]
        self.refresh()
        self._ok(f"deleted {name!r}")

    def set_origin(self, name: str, x: float, y: float) -> None:
        c = self.schedule.chip(name)
        if c is None:
            return self._error("no chip selected (Chips tab)")
        c.origin = (float(x), float(y))
        self.refresh(select=name)
        self._ok(f"{name}: origin at ({x:.2f}, {y:.2f}) µm")

    def origin_to_stage(self) -> None:
        pos = self.stage_position()
        if pos is None:
            return self._error("the stage position is not known")
        self.set_origin(self.selected(), *pos)


class HeightDialog(QtWidgets.QDialog):
    """A chip's height (H = WD − Z, µm): Z kept, one height, or a height map of ≥ 3 points (CCS) with
    a least-squares plane.  *Record here* takes the stage now – focus on the chip first – and
    converts its X, Y to CCS through the chip's present map (*to_ccs*); *Go to* moves to a point in
    focus at the WD now (*goto(x, y, z)*, the point through *to_scs*).  :meth:`value` is the edited
    :class:`~easyebl.litho.height.ChipHeight`; *fields_ccs*: the chip's field centres (for the Z range)."""

    def __init__(self, chip, stage_position, stage_z, wd, to_ccs, to_scs, goto=None, fields_ccs=(),
                 schedule_wd=None, parent=None):
        super().__init__(parent)
        from ..height import ChipHeight

        self.setWindowTitle(f"Height of {chip.name}")
        self.h = ChipHeight.from_dict(chip.height.to_dict())
        self.stage_position, self.stage_z, self.wd = stage_position, stage_z, wd
        self.to_ccs, self.to_scs, self.goto = to_ccs, to_scs, goto
        self.fields_ccs = list(fields_ccs)
        self.schedule_wd = schedule_wd
        intro = QtWidgets.QLabel(
            "H = WD − Z (µm) of the chip surface, recorded with it in focus.  After the beam calibration the "
            "column is not touched: the stage's Z brings the chip into focus, Z = 1000·WD − H.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa")
        self.mode = QtWidgets.QComboBox()
        for k, t in (("none", "none: keep the stage's Z"), ("constant", "constant: one height"),
                     ("map", "height map: a plane through ≥ 3 points")):
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.h.mode)))
        self.mode.currentIndexChanged.connect(self._mode)
        # constant
        self.g_const = QtWidgets.QGroupBox("Constant")
        fc = QtWidgets.QFormLayout(self.g_const)
        self.h_value = _spin(self.h.value_um or 0.0, -1e6, 1e6, 1, " µm", 1)
        self.h_value.valueChanged.connect(lambda v: setattr(self.h, "value_um", float(v)))
        b_rec = QtWidgets.QPushButton("Record here")
        b_rec.setToolTip("the height of the stage now (focus on a feature of the chip first)")
        b_rec.clicked.connect(self.record_constant)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.h_value, 1)
        row.addWidget(b_rec)
        fc.addRow("H", row)
        # map
        self.g_map = QtWidgets.QGroupBox("Height map (chip coordinates)")
        vm = QtWidgets.QVBoxLayout(self.g_map)
        self.table = _table(["X (µm)", "Y (µm)", "H (µm)"])
        self.table.setMinimumHeight(130)
        self.table.itemChanged.connect(self._points_edited)
        vm.addWidget(self.table)
        hb = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Record here")
        b_add.setToolTip("a point at the stage now (focus on the chip there first)")
        b_add.clicked.connect(self.record_point)
        b_go = QtWidgets.QPushButton("Go to")
        b_go.setToolTip("move to the selected point, in focus at the WD now")
        b_go.clicked.connect(self._goto_point)
        b_go.setEnabled(goto is not None)
        b_del = QtWidgets.QPushButton("Delete")
        b_del.clicked.connect(self._delete_point)
        for b in (b_add, b_go, b_del):
            hb.addWidget(b)
        vm.addLayout(hb)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet("color:#f66")
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(intro)
        f = QtWidgets.QFormLayout()
        f.addRow("mode", self.mode)
        lay.addLayout(f)
        lay.addWidget(self.g_const)
        lay.addWidget(self.g_map)
        lay.addWidget(self.info)
        lay.addWidget(self.msg)
        lay.addWidget(bb)
        self._fill()
        self._mode()
        self.resize(520, 520)

    def _mode(self, *_):
        self.h.mode = self.mode.currentData()
        if self.h.mode == "constant" and self.h.value_um is None:
            self.h.value_um = float(self.h_value.value())
        self.g_const.setVisible(self.h.mode == "constant")
        self.g_map.setVisible(self.h.mode == "map")
        self._info()

    def _fill(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.h.points))
        for r, pt in enumerate(self.h.points):
            for c, v in enumerate(pt):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v:.2f}" if c < 2 else f"{v:.1f}"))
        self.table.blockSignals(False)
        self._info()

    def _info(self):
        txt = self.h.describe()
        if self.h.mode != "none" and self.fields_ccs and self.schedule_wd is not None:
            hs = [self.h.at(c) for c in self.fields_ccs]
            hs = [v for v in hs if v is not None]
            if hs:
                zs = [1000.0 * self.schedule_wd - v for v in hs]
                txt += (f" · Z over the {len(zs)} field(s) at WD {self.schedule_wd:g} mm: {min(zs):.1f} … "
                        f"{max(zs):.1f} µm")
        self.info.setText(txt)

    def _here(self):
        here = _here(self.stage_position, self.stage_z, self.wd)
        if isinstance(here, str):
            self.msg.setText(here)
            return None
        self.msg.clear()
        return here

    def record_constant(self):
        here = self._here()
        if here is not None:
            self.h_value.setValue(here[2])

    def record_point(self):
        here = self._here()
        if here is None:
            return
        c = self.to_ccs((here[0], here[1]))
        self.h.points.append((float(c[0]), float(c[1]), float(here[2])))
        self._fill()
        self.table.selectRow(len(self.h.points) - 1)

    def _points_edited(self, *_):
        out = []
        try:
            for r in range(self.table.rowCount()):
                out.append(tuple(float(self.table.item(r, c).text()) for c in range(3)))
        except (ValueError, AttributeError):
            self.msg.setText("a point is X, Y, H in µm")
            return
        self.msg.clear()
        self.h.points = out
        self._info()

    def _delete_point(self):
        r = _selected_row(self.table)
        if 0 <= r < len(self.h.points):
            del self.h.points[r]
            self._fill()

    def _goto_point(self):
        r = _selected_row(self.table)
        if not 0 <= r < len(self.h.points) or self.goto is None:
            return
        x, y, h = self.h.points[r]
        wd = self.wd()
        if wd is None:
            self.msg.setText("the working distance is not known: cannot compute the Z")
            return
        scs = self.to_scs((x, y))
        self.goto(float(scs[0]), float(scs[1]), 1000.0 * wd - h)

    def value(self):
        return self.h


def mag_combo(value, setter=None) -> QtWidgets.QComboBox:
    """The SEM's permitted magnifications; a value off the list is shown as such until replaced."""
    c = QtWidgets.QComboBox()
    for m in MAGNIFICATIONS:
        c.addItem(f"×{m:,}", m)
    i = c.findData(value)
    if i < 0:
        c.addItem(f"×{value:g} (not permitted)", value)
        i = c.count() - 1
    c.setCurrentIndex(i)
    c.setMaxVisibleItems(20)
    if setter is not None:
        c.currentIndexChanged.connect(lambda _: setter(c.currentData()))
    return c


def _layers_text(layers) -> str:
    if layers is None:
        return ""
    return ", ".join(f"{x[0]}/{x[1]}" if isinstance(x, (tuple, list)) else str(x) for x in layers)


def _parse_layers(text: str):
    text = text.strip()
    if not text:
        return None
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            a, b = part.split("/", 1)
            out.append((int(a), int(b)))
        else:
            out.append(int(part))
    return out


# ====================================================================== op editor
class OpEditor(QtWidgets.QWidget):
    """Form for one operation; edits apply to the operation at once (``changed``).

    The window sets the providers: ``presets()`` (beam preset names | None), ``positions()``
    ({name: StagePosition}), ``chips()`` (chip names), ``top_cells(path)``, ``current_for(op)``
    (nominal current at that step | None), ``fields_info(op)`` (text about the Exposure's fields
    and markers), ``field_centres(op)`` (CCS centres of its fields) and ``current_wd()`` (the
    column's WD from the focus, mm | None)."""

    changed = QtCore.Signal()
    pickGdsRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.op: Operation | None = None
        self.schedule = None  # the schedule of the op (its WD, which steps are fixed)
        self.presets = lambda: None
        self.positions = lambda: {}
        self.chips = lambda: []
        self.top_cells = lambda path: []
        self.current_for = lambda op: None
        self.fields_info = lambda op: ""
        self.field_centres = lambda op: []
        self.current_wd = lambda: None
        self.readout = None
        self.info = None
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(4, 4, 4, 4)
        self.title = QtWidgets.QLabel("(no step selected)")
        self.title.setStyleSheet("font-weight:bold")
        self.lay.addWidget(self.title)
        self.body = QtWidgets.QWidget()
        self.lay.addWidget(self.body)
        self.msg = QtWidgets.QLabel()
        self.msg.setStyleSheet("color:#f88")
        self.msg.setWordWrap(True)
        self.lay.addWidget(self.msg)

    # ---------------------------------------------------------------- helpers
    def _set(self, name, value, rebuild: bool = False) -> None:
        if self.op is not None and getattr(self.op, name) != value:
            setattr(self.op, name, value)
            self.changed.emit()
            if rebuild:
                self._rebuild()

    def _rebuild(self) -> None:
        QtCore.QTimer.singleShot(0, lambda op=self.op: self.set_op(op) if op is self.op else None)

    def _spin_for(self, form, label, name, **kw):
        s = _spin(getattr(self.op, name), **kw)
        s.valueChanged.connect(lambda v, n=name: self._set(n, float(v)))
        form.addRow(label, s)
        return s

    def refresh_info(self) -> None:
        if self.info is not None and isinstance(self.op, Exposure):
            self.info.setText(self.fields_info(self.op))

    def set_op(self, op: Operation | None) -> None:
        self.op = None
        self.readout = None
        self.info = None
        self.msg.clear()
        self.body.setParent(None)
        self.body.deleteLater()
        self.body = QtWidgets.QWidget()
        self.lay.insertWidget(1, self.body)
        if op is None:
            self.title.setText("(no step selected)")
            return
        self.op = op
        idx = next((k for k, o in enumerate(self.schedule.operations) if o is op), -1) \
            if self.schedule is not None else -1
        fixed = self.schedule is not None and self.schedule.is_fixed(idx)
        self.title.setText(op.TITLE + ("  (fixed step)" if fixed else ""))
        v = QtWidgets.QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QFormLayout()
        v.addLayout(top)
        lab = QtWidgets.QLineEdit(op.label)
        lab.setPlaceholderText("(optional)")
        lab.editingFinished.connect(lambda: self._set("label", lab.text()))
        top.addRow("label", lab)
        if not fixed:
            en = QtWidgets.QCheckBox("enabled")
            en.setChecked(op.enabled)
            en.toggled.connect(lambda on: self._set("enabled", bool(on)))
            top.addRow(en)
        if isinstance(op, ColumnSetting):
            self._column(top, first=idx == 0)
        elif isinstance(op, BeamCalibrationStep):
            self._beamcal(top)
        elif isinstance(op, FieldCorrection):
            self._fieldcal(top)
        elif isinstance(op, Exposure):
            self._exposure(v)

    # ---------------------------------------------------------------- column
    def _cfg(self, name, value) -> None:
        """Set a field of the step's calibration config."""
        if self.op is not None and getattr(self.op.config, name) != value:
            setattr(self.op.config, name, value)
            self.changed.emit()

    def _cfg_spin(self, form, label, name, scale=1.0, **kw):
        s = _spin(getattr(self.op.config, name) * scale, **kw)
        s.valueChanged.connect(lambda v, n=name: self._cfg(n, float(v) / scale))
        form.addRow(label, s)
        return s

    def _int_spin(self, form, label, name, lo, hi, step=1, suffix=""):
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setValue(int(getattr(self.op.config, name)))
        s.setKeyboardTracking(False)
        s.valueChanged.connect(lambda v, n=name: self._cfg(n, int(v)))
        form.addRow(label, s)
        return s

    def _position_combo(self, form, label) -> QtWidgets.QComboBox:
        """The named stage positions, with their height (one without: marked)."""
        op = self.op
        pos = QtWidgets.QComboBox()
        pos.addItem("(choose a stage position)", "")
        for n, p in self.positions().items():
            h = getattr(p, "h", None)
            pos.addItem(f"{n}  ({p.x:.1f}, {p.y:.1f}, " + (f"H {h:.1f})" if h is not None else "NO HEIGHT)"), n)
        if op.position and pos.findData(op.position) < 0:
            pos.addItem(f"{op.position} (unknown)", op.position)
        pos.setCurrentIndex(max(0, pos.findData(op.position)))
        pos.setToolTip("a named stage position (Stage Positions tab): its X, Y and height H – the stage Z is set "
                       "to have it in focus at the schedule's WD")
        pos.currentIndexChanged.connect(lambda _: self._cfg("position", pos.currentData() or ""))
        form.addRow(label, pos)
        return pos

    def _beamcal(self, f) -> None:
        op = self.op
        info = QtWidgets.QLabel(
            "One image of a calibration sample with a square grid over the whole field of view, at ≤ 0.1 µm "
            "pixels (5000 × 5000 for a 500 µm field).  The fit gives the beam's X / Y scale, shear and pincushion "
            "distortion; valid for the magnification and WD it was measured at.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f.addRow(info)
        self._position_combo(f, "grid sample at")
        self._int_spin(f, "image", "pixels", 200, 20000, 500, " px across")
        self._cfg_spin(f, "grid pitch", "pitch_um", lo=0.05, hi=100, decimals=3, suffix=" µm", step=0.1)
        self._int_spin(f, "frames averaged", "frames", 1, 64)
        self._cfg_spin(f, "refuse a fit worse than", "max_rms_um", 1e3, lo=1, hi=5000, decimals=0, suffix=" nm",
                       step=10)
        self._cfg_spin(f, "low-pass", "filter_periods", lo=0.5, hi=20, decimals=1, suffix=" periods", step=0.5)
        mf = QtWidgets.QCheckBox("focus and stigmate by hand first, at")
        mf.setChecked(bool(op.config.manual_focus))
        mf.setToolTip("before the image: zoom in at the grid and wait for you to focus and stigmate (the focus may "
                      "change the WD: the WD then is the schedule's for the rest of the run, and the grid "
                      "position's height is recorded again)")
        mm = mag_combo(op.config.focus_magnification, lambda m: self._cfg("focus_magnification", int(m)))
        mm.setEnabled(mf.isChecked())
        mf.toggled.connect(lambda on: (self._cfg("manual_focus", bool(on)), mm.setEnabled(on)))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(mf)
        row.addWidget(mm, 1)
        f.addRow(row)
        ru = QtWidgets.QCheckBox("use a stored calibration of the same magnification and WD (else measure)")
        ru.setChecked(op.reuse)
        ru.toggled.connect(lambda on: self._set("reuse", bool(on)))
        f.addRow(ru)

    def _fieldcal(self, f) -> None:
        op = self.op
        info = QtWidgets.QLabel(
            "At a feature with structure (a Faraday cup, a calibration sample) images at known stage shifts give "
            "the stage → beam rotation, through the beam calibration of the condition; valid for the "
            "magnification, beam preset and WD it was measured at.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f.addRow(info)
        self._position_combo(f, "at")
        cfg = op.config
        pat = QtWidgets.QComboBox()
        pat.addItem("3 × 3 grid (9 images)", "grid")
        pat.addItem("cross (5 images)", "cross")
        pat.setCurrentIndex(max(0, pat.findData(cfg.pattern)))
        pat.currentIndexChanged.connect(lambda _: self._cfg("pattern", pat.currentData()))
        f.addRow("stage shifts", pat)
        self._cfg_spin(f, "shift", "step_fraction", 100.0, lo=1, hi=45, decimals=1, suffix=" % of the field", step=5)
        self._cfg_spin(f, "image width", "image_fraction", 100.0, lo=10, hi=100, decimals=0, suffix=" % of the field",
                       step=5)
        px = QtWidgets.QComboBox()
        for n in (128, 256, 512, 1024):
            px.addItem(f"{n} × {n}", n)
        px.setCurrentIndex(max(0, px.findData(cfg.image_px)))
        px.currentIndexChanged.connect(lambda _: self._cfg("image_px", int(px.currentData())))
        f.addRow("image", px)
        self._int_spin(f, "frames averaged", "frames", 1, 64)
        self._cfg_spin(f, "settle after a move", "settle_s", lo=0, hi=60, decimals=2, suffix=" s", step=0.1)
        self._cfg_spin(f, "refuse a fit worse than", "max_residual_px", lo=0.1, hi=50, decimals=1, suffix=" px",
                       step=0.5)
        dn = self._cfg_spin(f, "de-noise", "denoise_px", lo=0, hi=20, decimals=1, suffix=" px", step=0.5)
        dn.setSpecialValueText("off")
        dn.setToolTip("Gaussian low-pass (σ) before matching: about the size of the feature's detail; more for "
                      "noisier images")
        bg = self._cfg_spin(f, "flatten (background)", "background_fraction", 100.0, lo=0, hi=50, decimals=1,
                            suffix=" % of the image", step=2.5)
        bg.setSpecialValueText("off")
        bg.setToolTip("a blur this wide (σ) is subtracted: brightness gradients and charging do not match")
        wh = self._cfg_spin(f, "whitening", "whitening", lo=0, hi=1, decimals=2, suffix="", step=0.1)
        wh.setToolTip("1: pure phase correlation (sharp, but noise counts as much as the feature); lower is more "
                      "robust on noisy, low-contrast images")
        ru = QtWidgets.QCheckBox("keep the field calibration in force when it applies (same magnification, beam "
                                 "preset, WD and beam calibration; else measure)")
        ru.setChecked(op.reuse)
        ru.toggled.connect(lambda on: self._set("reuse", bool(on)))
        f.addRow(ru)
        notes = QtWidgets.QLineEdit(op.notes)
        notes.editingFinished.connect(lambda: self._set("notes", notes.text()))
        f.addRow("notes", notes)

    def set_wd_to_current(self, spin=None) -> bool:
        """The schedule's WD := the column's WD now (from the focus, to 0.001 mm)."""
        try:
            wd = self.current_wd()
        except Exception:  # noqa: BLE001
            wd = None
        if wd is None:
            self.msg.setText("the working distance cannot be read (microscope not connected?)")
            return False
        wd = round(float(wd), 3)
        if not 4.0 <= wd <= 60.0:
            self.msg.setText(f"the column's WD {wd:g} mm is outside 4–60 mm")
            return False
        self.msg.clear()
        if spin is not None:
            spin.setValue(wd)  # → _set_wd
        else:
            self._set_wd(wd)
        return True

    def _set_wd(self, v: float) -> None:
        if self.schedule is not None and self.schedule.wd_mm != float(v):
            self.schedule.wd_mm = float(v)
            self.changed.emit()

    def _column(self, f, first: bool = False) -> None:
        op = self.op
        pr = QtWidgets.QComboBox()
        pr.addItem("Keep as is", KEEP_AS_IS)
        for n in self.presets() or []:
            pr.addItem(n, n)
        if op.preset and pr.findData(op.preset) < 0:
            pr.addItem(f"{op.preset} (unknown)", op.preset)
        pr.setCurrentIndex(max(0, pr.findData(op.preset)))
        pr.setToolTip("a beam preset of the microscope module (Microscope → Beam presets…)")
        pr.currentIndexChanged.connect(lambda _: self._set("preset", pr.currentData()))
        f.addRow("beam", pr)
        f.addRow("magnification", mag_combo(op.magnification, lambda m: self._set("magnification", m)))
        cs = _spin(op.current_pa or 0.0, 0, 1e6, 2, " pA", 10)
        cs.setSpecialValueText("unknown")
        cs.setToolTip("nominal beam current: only for the dose readout of the fields")
        cs.valueChanged.connect(lambda val: self._set("current_pa", float(val) or None))
        f.addRow("nominal current", cs)
        if first and self.schedule is not None:
            wd = _spin(self.schedule.wd_mm, 4, 60, 3, " mm", 0.1)
            wd.setToolTip("the working distance of the whole schedule: this step sets the focus to it; nothing "
                          "changes the focus after that – the stage Z brings each sample into focus (Z = 1000·WD − H)")
            wd.valueChanged.connect(self._set_wd)
            f.addRow("working distance", wd)
            b = QtWidgets.QPushButton("Set to current WD")
            b.setToolTip("the column's working distance now (from the focus)")
            b.clicked.connect(lambda: self.set_wd_to_current(wd))
            self.b_wd_current = b
            f.addRow("", b)

    # ---------------------------------------------------------------- exposure
    def _exposure(self, v) -> None:
        op = self.op

        # --- chip and GDS
        g = QtWidgets.QGroupBox("Pattern (GDS top level = CCS − offset; one field per top-level cell)")
        f = QtWidgets.QFormLayout(g)
        chip = QtWidgets.QComboBox()
        for n in self.chips():
            chip.addItem(n, n)
        if op.chip and chip.findData(op.chip) < 0:
            chip.addItem(f"{op.chip} (unknown)", op.chip)
        chip.setCurrentIndex(max(0, chip.findData(op.chip)))
        chip.currentIndexChanged.connect(lambda _: self._set("chip", chip.currentData() or ""))
        f.addRow("chip", chip)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton(os.path.basename(op.path) if op.path else "Choose GDS file…")
        b.setToolTip(op.path or "no file: the step aligns only")
        b.clicked.connect(self.pickGdsRequested.emit)
        row.addWidget(b, 1)
        if op.path:
            clr = QtWidgets.QToolButton(text="×", toolTip="no GDS file (alignment only)")
            clr.clicked.connect(lambda: self._set("path", "", rebuild=True))
            row.addWidget(clr)
        f.addRow("file", row)
        if op.path:
            top = QtWidgets.QComboBox()
            top.addItem("(the only top-level cell)", "")
            try:
                names = self.top_cells(op.path)
            except Exception as e:  # noqa: BLE001
                names = []
                self.msg.setText(f"{type(e).__name__}: {e}")
            for n in names:
                top.addItem(n, n)
            if op.top and top.findData(op.top) < 0:
                top.addItem(op.top, op.top)
            top.setCurrentIndex(max(0, top.findData(op.top)))
            top.currentIndexChanged.connect(lambda _: self._set("top", top.currentData() or ""))
            f.addRow("top cell", top)
            lay_ed = QtWidgets.QLineEdit(_layers_text(op.layers))
            lay_ed.setPlaceholderText("all layers (e.g. 1/0, 2)")
            lay_ed.editingFinished.connect(lambda: self._set_layers(lay_ed.text()))
            f.addRow("layers", lay_ed)
        f.addRow("offset (GDS origin in CCS)", _pair(*op.offset, lambda p: self._set("offset", p)))
        co = QtWidgets.QCheckBox("field centre = the cell's origin (else its bounding-box centre)")
        co.setChecked(op.cell_origin)
        co.toggled.connect(lambda on: self._set("cell_origin", bool(on)))
        f.addRow(co)
        order = QtWidgets.QComboBox()
        for k, t in (("file", "as in the file"), ("raster", "rows, left → right"), ("serpentine", "serpentine rows")):
            order.addItem(t, k)
        order.setCurrentIndex(max(0, order.findData(op.order)))
        order.currentIndexChanged.connect(lambda _: self._set("order", order.currentData()))
        f.addRow("field order", order)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#9aa")
        f.addRow(self.info)
        v.addWidget(g)

        # --- dose
        if op.path:
            g = QtWidgets.QGroupBox("Dose")
            f = QtWidgets.QFormLayout(g)
            self.readout = QtWidgets.QLabel()
            self.readout.setWordWrap(True)
            self.readout.setStyleSheet("color:#9aa")
            d = self._spin_for(f, "dwell / point", "dwell_us", lo=0.001, hi=1e6, decimals=3, suffix=" µs", step=0.1)
            st = self._spin_for(f, "point step", "step_nm", lo=0.1, hi=1e5, decimals=2, suffix=" nm", step=1)
            d.valueChanged.connect(self._readout)
            st.valueChanged.connect(self._readout)
            f.addRow(self.readout)
            self._readout()
            v.addWidget(g)

        # --- global align
        g = QtWidgets.QGroupBox("Global Align (CCS markers, all in one field: written first)")
        g.setCheckable(True)
        g.setChecked(op.global_align)
        g.toggled.connect(self._toggle_global)
        f = QtWidgets.QFormLayout(g)
        if op.global_align:
            n = QtWidgets.QSpinBox()
            n.setRange(2, 4)
            n.setValue(max(2, min(4, len(op.global_markers))))
            n.valueChanged.connect(self._marker_count)
            f.addRow("markers", n)
            for k, mk in enumerate(op.global_markers):
                f.addRow(f"marker {k + 1}", _pair(*mk, lambda p, k=k: self._set_marker(k, p)))
            f.addRow(QtWidgets.QLabel("2 markers: rotation, scale, shift · 3–4: affine (scale and shear too)"))
        v.addWidget(g)

        # --- local markers
        g = QtWidgets.QGroupBox("Local markers (CCS, at most one per field: a shift for that field)")
        g.setCheckable(True)
        g.setChecked(op.local_align)
        g.toggled.connect(lambda on: self._set("local_align", bool(on), rebuild=True))
        lv = QtWidgets.QVBoxLayout(g)
        if op.local_align:
            t = _table(["X (µm)", "Y (µm)"])
            t.setRowCount(len(op.local_markers))
            for r, m in enumerate(op.local_markers):
                t.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{m[0]:g}"))
                t.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{m[1]:g}"))
            t.setMinimumHeight(110)
            t.itemChanged.connect(lambda it, t=t: self._local_edited(t))
            self.local_table = t
            lv.addWidget(t)
            h = QtWidgets.QHBoxLayout()
            b_add = QtWidgets.QPushButton("Add")
            b_add.clicked.connect(lambda: self._local_add((0.0, 0.0)))
            b_del = QtWidgets.QPushButton("Remove")
            b_del.clicked.connect(lambda: self._local_remove(_selected_row(t)))
            b_gen = QtWidgets.QPushButton("One per field…")
            b_gen.setToolTip("a marker at the same place relative to every field centre (replaces the list)")
            b_gen.clicked.connect(self._local_per_field)
            for w in (b_add, b_del, b_gen):
                h.addWidget(w)
            lv.addLayout(h)
        v.addWidget(g)
        if op.global_align or op.local_align:
            g = QtWidgets.QFormLayout()
            self._spin_for(g, "marker view", "view_um", lo=0.01, hi=1e5)
            v.addLayout(g)
        self.refresh_info()

    def _readout(self, *_):
        op = self.op
        if isinstance(op, Exposure) and self.readout is not None:
            self.readout.setText(dose_readout(op.dwell_us, op.step_nm, self.current_for(op)))

    def _set_layers(self, text: str) -> None:
        try:
            self._set("layers", _parse_layers(text))
            self.msg.clear()
        except ValueError:
            self.msg.setText("layers: e.g. 1/0, 2 (layer/datatype, or a layer number)")

    def _toggle_global(self, on: bool) -> None:
        op = self.op
        if op is None:
            return
        if on and len(op.global_markers) < 2:
            op.global_markers = (list(op.global_markers) + [(-10.0, -10.0), (10.0, 10.0)])[:2]
        self._set("global_align", bool(on), rebuild=True)

    def _marker_count(self, n: int) -> None:
        op = self.op
        mk = list(op.global_markers)
        defaults = [(-10.0, -10.0), (10.0, 10.0), (-10.0, 10.0), (10.0, -10.0)]
        while len(mk) < n:
            mk.append(defaults[len(mk)])
        op.global_markers = mk[:n]
        self.changed.emit()
        self._rebuild()

    def _set_marker(self, k: int, p) -> None:
        mk = list(self.op.global_markers)
        mk[k] = p
        self._set("global_markers", mk)

    def _local_edited(self, t) -> None:
        out = []
        try:
            for r in range(t.rowCount()):
                out.append((float(t.item(r, 0).text()), float(t.item(r, 1).text())))
        except (ValueError, AttributeError):
            self.msg.setText("a local marker is (X, Y) in µm")
            return
        self.msg.clear()
        self._set("local_markers", out)
        self.refresh_info()

    def _local_add(self, p) -> None:
        self._set("local_markers", list(self.op.local_markers) + [tuple(p)], rebuild=True)

    def _local_remove(self, r: int) -> None:
        mk = list(self.op.local_markers)
        if 0 <= r < len(mk):
            del mk[r]
            self._set("local_markers", mk, rebuild=True)

    def set_local_per_field(self, dx: float, dy: float) -> int:
        cs = self.field_centres(self.op)
        self._set("local_markers", [(c[0] + dx, c[1] + dy) for c in cs], rebuild=True)
        return len(cs)

    def _local_per_field(self) -> None:
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("A local marker per field")
        f = QtWidgets.QFormLayout(d)
        p = _pair(-40.0, 40.0, lambda _: None)
        f.addRow("relative to each field centre", p)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.set_local_per_field(p.spins[0].value(), p.spins[1].value())


# ====================================================================== schedule
class ScheduleTab(QtWidgets.QWidget):
    """Steps list, step editor, schedule settings, run controls."""

    changed = QtCore.Signal()
    selectionChanged = QtCore.Signal(int)
    runRequested = QtCore.Signal(int)  # from this step
    pauseRequested = QtCore.Signal(bool)
    abortRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.schedule = None
        self.plans = None
        self.states: list = []
        self.new_exposure = lambda: Exposure()  # what Add → Exposure adds (the window sets the chip)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._select)
        add = QtWidgets.QToolButton()
        add.setText("Add ▾")
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(add)
        m.addAction(Exposure.TITLE, lambda: self.add_op(self.new_exposure()))
        m.addAction(ColumnSetting.TITLE, lambda: self.add_op(ColumnSetting(magnification=self._last_mag())))
        m.addAction(FieldCorrection.TITLE, lambda: self.add_op(self._new_fieldcal()))
        m.addAction(BeamCalibrationStep.TITLE, lambda: self.add_op(self._new_beamcal()))
        add.setMenu(m)
        self.add_menu = m
        self.b_add = add
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(add)
        self.b_dup = QtWidgets.QToolButton(text="Dup", toolTip="duplicate the selected step")
        self.b_del = QtWidgets.QToolButton(text="Del", toolTip="delete the selected step (not the three fixed "
                                                                    "first steps)")
        self.b_up = QtWidgets.QToolButton(text="▲", toolTip="move up")
        self.b_down = QtWidgets.QToolButton(text="▼", toolTip="move down")
        self.b_dup.clicked.connect(self.duplicate)
        self.b_del.clicked.connect(self.delete)
        self.b_up.clicked.connect(lambda: self.move(-1))
        self.b_down.clicked.connect(lambda: self.move(+1))
        for b in (self.b_dup, self.b_del, self.b_up, self.b_down):
            bar.addWidget(b)
        bar.addStretch(1)
        self.count = QtWidgets.QLabel()
        bar.addWidget(self.count)

        self.editor = OpEditor()
        self.editor.changed.connect(self._op_edited)
        self.warn = QtWidgets.QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#f88")

        self.s_policy = QtWidgets.QComboBox()
        for k, t in (("ask", "ask (skip or abort)"), ("skip", "skip the field"), ("abort", "abort the schedule")):
            self.s_policy.addItem(t, k)
        self.s_fill = QtWidgets.QComboBox()
        self.s_fill.addItems(["x", "y"])
        sg = QtWidgets.QGroupBox("Schedule settings")
        sf = QtWidgets.QFormLayout(sg)
        self.s_maxz = _spin(2000.0, 1, 50000, 0, " µm", 100)
        self.s_maxz.setToolTip("a stage Z move larger than this (to bring a sample into focus) fails the step: "
                               "a wrong height or WD must not drive the sample into the lens")
        sf.addRow("pattern outside the FoV", self.s_policy)
        sf.addRow("area fill direction", self.s_fill)
        sf.addRow("largest Z move", self.s_maxz)
        self.s_policy.currentIndexChanged.connect(self._settings_edited)
        self.s_fill.currentIndexChanged.connect(self._settings_edited)
        self.s_maxz.valueChanged.connect(self._settings_edited)

        self.b_run = QtWidgets.QPushButton("Run")
        self.b_run.setToolTip("run the schedule from the first step")
        self.b_run.clicked.connect(lambda: self.runRequested.emit(0))
        self.b_from = QtWidgets.QPushButton("Run from selected")
        self.b_from.setToolTip("from the selected step, skipping the three fixed steps – only when the column is "
                               "at the schedule's WD and the beam and field calibrations are in force")
        self.b_from.clicked.connect(lambda: self.runRequested.emit(max(0, self.list.currentRow())))
        self.b_pause = QtWidgets.QPushButton("Pause")
        self.b_pause.setCheckable(True)
        self.b_pause.toggled.connect(self.pauseRequested.emit)
        self.b_abort = QtWidgets.QPushButton("Abort")
        self.b_abort.setStyleSheet("background:#a33;color:white")
        self.b_abort.clicked.connect(self.abortRequested.emit)
        rb = QtWidgets.QHBoxLayout()
        for b in (self.b_run, self.b_from, self.b_pause, self.b_abort):
            rb.addWidget(b)

        sp = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addLayout(bar)
        tl.addWidget(self.list, 1)
        sp.addWidget(top)
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.editor)
        bl.addWidget(self.warn)
        bl.addWidget(sg)
        bl.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidget(bottom)
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sp.addWidget(sc)
        sp.setSizes([220, 620])
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(sp, 1)
        lay.addLayout(rb)
        self.set_running(False)

    # ---------------------------------------------------------------- model
    def set_schedule(self, schedule) -> None:
        self.schedule = schedule
        self.editor.schedule = schedule
        self.states = ["pending"] * len(schedule)
        st = schedule.settings
        self.s_maxz.blockSignals(True)
        self.s_maxz.setValue(st.max_z_change_um)
        self.s_maxz.blockSignals(False)
        for w, val in ((self.s_policy, self.s_policy.findData(st.on_out_of_field)),
                       (self.s_fill, self.s_fill.findText(st.fill_direction))):
            w.blockSignals(True)
            w.setCurrentIndex(max(0, val))
            w.blockSignals(False)
        self.refresh(select=0)

    def set_plans(self, plans) -> None:
        self.plans = plans
        self._update_items()
        self._show_warnings()
        self.editor._readout()
        self.editor.refresh_info()

    def set_state(self, index: int, state: str, message: str = "") -> None:
        if 0 <= index < len(self.states):
            self.states[index] = state
            self._update_item(index, message)

    def reset_states(self) -> None:
        self.states = ["pending"] * len(self.schedule or [])
        self._update_items()

    def refresh(self, select: int | None = None) -> None:
        cur = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        if len(self.states) != len(self.schedule):
            self.states = (self.states + ["pending"] * len(self.schedule))[:len(self.schedule)]
        for _ in self.schedule.operations:
            self.list.addItem(QtWidgets.QListWidgetItem())
        self._update_items()
        self.list.blockSignals(False)
        cur = max(0, min(cur, len(self.schedule) - 1))
        self.list.setCurrentRow(cur)
        self._select(cur)
        self.count.setText(f"{len(self.schedule)} steps")

    def _update_items(self) -> None:
        for i in range(self.list.count()):
            self._update_item(i)

    def _update_item(self, i: int, message: str = "") -> None:
        it = self.list.item(i)
        if it is None or i >= len(self.schedule):
            return
        op = self.schedule[i]
        st = self.states[i] if i < len(self.states) else "pending"
        p = self.plans[i] if self.plans is not None and i < len(self.plans) else None
        warns = p.warnings if p is not None else []
        nf = f" ({len(p.fields)} fields)" if p is not None and isinstance(op, Exposure) and p.fields else ""
        title = op.title() + (f", WD {self.schedule.wd_mm:g} mm" if i == 0 else "")
        it.setText(f"{i + 1}. {STATE_MARK[st]}{title}{nf}" + ("  ⚠" if warns else "")
                   + (f"  – {message}" if message and st == "running" else ""))
        tip = "\n".join(warns)
        if message:
            tip = (tip + "\n" if tip else "") + message
        it.setToolTip(tip)
        col = STATE_COLORS.get(st)
        if not op.enabled:
            col = "#777"
        elif col is None and warns:
            col = "#f88"
        it.setForeground(QtGui.QBrush(QtGui.QColor(col)) if col else QtGui.QBrush())
        f = it.font()
        f.setStrikeOut(not op.enabled)
        f.setBold(isinstance(op, ColumnSetting) or self.schedule.is_fixed(i))
        it.setFont(f)

    def _show_warnings(self) -> None:
        i = self.list.currentRow()
        w = self.plans[i].warnings if self.plans is not None and 0 <= i < len(self.plans) else []
        self.warn.setText("\n".join("⚠ " + x for x in w))

    def _select(self, i: int) -> None:
        ok = self.schedule is not None and 0 <= i < len(self.schedule)
        self.editor.set_op(self.schedule[i] if ok else None)
        self._show_warnings()
        self._sync_buttons()
        self.selectionChanged.emit(i)

    def _sync_buttons(self) -> None:
        i = self.list.currentRow()
        s = self.schedule
        n = len(s) if s is not None else 0
        idle = self.b_run.isEnabled()
        self.b_del.setEnabled(idle and s is not None and s.can_delete(i))
        self.b_dup.setEnabled(idle and s is not None and not s.is_fixed(i) and 0 <= i < n)
        self.b_up.setEnabled(idle and s is not None and s.can_move(i, i - 1))
        self.b_down.setEnabled(idle and s is not None and s.can_move(i, i + 1))

    def selected(self) -> int:
        return self.list.currentRow()

    def select(self, i: int) -> None:
        self.list.setCurrentRow(i)

    def _op_edited(self) -> None:
        self._update_item(self.list.currentRow())
        self.changed.emit()

    def _settings_edited(self, *_):
        if self.schedule is None:
            return
        self.schedule.settings.on_out_of_field = self.s_policy.currentData()
        self.schedule.settings.fill_direction = self.s_fill.currentText()
        self.schedule.settings.max_z_change_um = float(self.s_maxz.value())
        self.changed.emit()

    def _last_mag(self) -> float:
        i = self.list.currentRow()
        for op in reversed(self.schedule.operations[:max(1, i + 1)]):
            if isinstance(op, ColumnSetting):
                return op.magnification
        return self.schedule[0].magnification

    def _new_fieldcal(self) -> FieldCorrection:
        """A Field calibration set up like the fixed one."""
        from ..fieldcal import FieldCalConfig

        return FieldCorrection(config=FieldCalConfig.from_dict(self.schedule[2].config.to_dict()))

    def _new_beamcal(self) -> BeamCalibrationStep:
        from ..beamcal import BeamCalConfig

        return BeamCalibrationStep(config=BeamCalConfig.from_dict(self.schedule[1].config.to_dict()))

    # ---------------------------------------------------------------- edits
    def add_op(self, op: Operation) -> None:
        from ..schedule import FIXED_STEPS

        i = self.list.currentRow()
        at = len(self.schedule) if i < 0 else max(i + 1, len(FIXED_STEPS))
        self.schedule.operations.insert(at, op)
        self.states.insert(at, "pending")
        self.refresh(select=at)
        self.changed.emit()

    def duplicate(self) -> None:
        from ..schedule import op_from_dict

        i = self.list.currentRow()
        if i > 0:
            self.add_op(op_from_dict(self.schedule[i].to_dict()))

    def delete(self) -> None:
        i = self.list.currentRow()
        if not self.schedule.can_delete(i):
            return
        del self.schedule.operations[i]
        del self.states[i]
        self.refresh(select=min(i, len(self.schedule) - 1))
        self.changed.emit()

    def move(self, d: int) -> None:
        i = self.list.currentRow()
        j = i + d
        if not self.schedule.can_move(i, j):
            return
        ops = self.schedule.operations
        ops[i], ops[j] = ops[j], ops[i]
        self.states[i], self.states[j] = self.states[j], self.states[i]
        self.refresh(select=j)
        self.changed.emit()

    def set_running(self, on: bool) -> None:
        self.b_run.setEnabled(not on)
        self.b_from.setEnabled(not on)
        self.b_pause.setEnabled(on)
        self.b_abort.setEnabled(on)
        if not on:
            self.b_pause.blockSignals(True)
            self.b_pause.setChecked(False)
            self.b_pause.blockSignals(False)
        self.editor.setEnabled(not on)
        self.b_add.setEnabled(not on)
        self._sync_buttons()
