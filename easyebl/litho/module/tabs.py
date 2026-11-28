"""Side tabs of the litho window: Stage Positions, Chips and Exposure Schedule."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..schedule import (KEEP_AS_IS, MAGNIFICATIONS, ORDERS, BeamCalibrationStep, ColumnSetting, Exposure, Operation,
                        dose_readout)

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


def _here(stage_position, stage_z):
    """(x, y, z) of the stage now, or an error text."""
    pos, z = stage_position(), stage_z()
    if pos is None or z is None:
        return "the stage position is not known (microscope not connected?)"
    return float(pos[0]), float(pos[1]), float(z)


class StagePositionsTab(QtWidgets.QWidget):
    """Named stage positions: X, Y and ΔZ (µm, relative to the calibration sample's Z; positive
    away from the gun).  The *Calibration sample* (its Z_cal and WD1: *Record WD*) and the *Faraday
    cup* are always there (replaced, not deleted).  *Go to* moves to X, Y, Z_cal + ΔZ and sets the
    focus to WD1.  Saved at once; every tab on the same settings follows."""

    changed = QtCore.Signal()
    gotoRequested = QtCore.Signal(float, float, object, object)  # SCS µm, Z µm (None: keep Z), WD mm (None: keep)

    def __init__(self, settings, stage_position=None, stage_z=None, wd=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.stage_position = stage_position or (lambda: None)  # () -> (x, y) | None
        self.stage_z = stage_z or (lambda: None)  # () -> z µm | None
        self.wd = wd or (lambda: None)  # () -> the column's WD (mm) | None
        self.table = _table(["Name", "X (µm)", "Y (µm)", "ΔZ (µm)", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self.l_cal = QtWidgets.QLabel()
        self.l_cal.setWordWrap(True)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        b_cur = QtWidgets.QPushButton("Add current stage")
        b_cur.setToolTip("a new position at the stage's X, Y; ΔZ from the stage Z against the calibration sample's")
        b_cur.clicked.connect(self.add_current)
        b_add = QtWidgets.QPushButton("Add…")
        b_add.clicked.connect(self.add_manual)
        self.b_replace = QtWidgets.QPushButton("Replace with current")
        self.b_replace.setToolTip("the selected position becomes the stage's X, Y, Z now (the calibration sample: "
                                  "its Z becomes Z_cal, every other ΔZ follows so their Z stays)")
        self.b_replace.clicked.connect(self.replace_selected)
        self.b_record = QtWidgets.QPushButton("Record WD")
        self.b_record.setToolTip("WD1 and Z_cal of the selected field size: the WD in focus on the calibration sample "
                                 "now and the stage Z – focus and stigmate on it at a high magnification first; no "
                                 "other ΔZ changes")
        self.b_record.clicked.connect(self.record_wd)
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_del.clicked.connect(self.delete_selected)
        self.b_go = QtWidgets.QPushButton("Go to XYZ")
        self.b_go.setToolTip("move the stage to the selected X, Y and Z_cal + ΔZ; the focus to WD1")
        self.b_go.clicked.connect(lambda: self._goto(True))
        self.b_go_xy = QtWidgets.QPushButton("Go to XY")
        self.b_go_xy.setToolTip("move the stage to the selected X, Y; Z and the focus stay")
        self.b_go_xy.clicked.connect(lambda: self._goto(False))
        # the list (and its notes) on the left, the buttons in a column on the right
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.table, 1)
        left.addWidget(self.l_cal)
        left.addWidget(self.msg)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(3)
        for b in (self.b_go, self.b_go_xy, None, b_cur, b_add, self.b_replace, self.b_record, None, self.b_del):
            if b is None:
                col.addSpacing(6)
            else:
                col.addWidget(b)
        col.addStretch(1)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(left, 1)
        lay.addLayout(col)
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
                dz = "0 (reference)" if p.role == "calibration" else ("" if p.dz is None else f"{p.dz:+.1f}")
                vals = (p.name, f"{p.x:.2f}", f"{p.y:.2f}", dz, p.note)
                for c, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if p.fixed and c == 0:
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                        it.setToolTip("always kept (Replace with current, not deleted)")
                    if c == 3 and p.role == "calibration":
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    elif c == 3 and p.dz is None:
                        it.setBackground(QtGui.QColor("#622"))
                        it.setToolTip("no ΔZ: Replace with current")
                    self.table.setItem(r, c, it)
        finally:
            self.table.blockSignals(False)
        cal = self.settings.calibration
        fs = self.settings.field_size or "no field size selected"
        if cal.z_um is None or cal.wd_mm is None:
            self.l_cal.setText(f"{cal.name} ({fs}): " + ("Z_cal not recorded" if cal.z_um is None else
                                                         f"Z_cal {cal.z_um:.1f} µm")
                               + ", WD1 not recorded – focus on it, then Record WD")
            self.l_cal.setStyleSheet("color:#fb4")
        else:
            self.l_cal.setText(f"{cal.name} ({fs}): Z_cal {cal.z_um:.1f} µm, WD1 {cal.wd_mm:.4f} mm")
            self.l_cal.setStyleSheet("")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        p = self._selected()
        for b in (self.b_replace, self.b_go_xy):
            b.setEnabled(p is not None)
        self.b_del.setEnabled(p is not None and not p.fixed)
        self.b_go.setEnabled(p is not None and self.settings.z_of(p) is not None)

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
                if p.role != "calibration":
                    p.dz = float(txt) if txt else None
            else:
                p.note = txt
        except ValueError as e:
            self.refresh()
            return self._error(str(e))
        self._save("saved")

    def add(self, name: str, x: float, y: float, dz: float | None = None) -> None:
        try:
            self.settings.add_position(name, x, y, dz)
        except ValueError as e:
            return self._error(str(e))
        self._save(f"added {name!r}")
        self.table.selectRow(len(self.settings.stage_positions) - 1)

    def add_current(self, name: str | None = None) -> None:
        here = _here(self.stage_position, self.stage_z)
        if isinstance(here, str):
            return self._error(here)
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Stage position", "Name:",
                                                      text=self.settings.unique_name())
            if not ok:
                return
        x, y, z = here
        self.add(name, x, y, self.settings.dz_here(z))

    def replace_selected(self, confirm: bool = True) -> None:
        p = self._selected()
        if p is None:
            return
        here = _here(self.stage_position, self.stage_z)
        if isinstance(here, str):
            return self._error(here)
        x, y, z = here
        cal = p.role == "calibration"
        if cal:
            what = (f"Z_cal {z:.1f} µm; every other ΔZ changes so their Z stays (record the WD again: Record WD)"
                    if p.z_um is None or abs(z - p.z_um) > 1e-6 else f"Z_cal {z:.1f} µm (unchanged)")
        else:
            dz = self.settings.dz_here(z)
            what = "ΔZ not known (the calibration sample's Z is not recorded)" if dz is None else f"ΔZ {dz:+.1f} µm"
        if confirm and QtWidgets.QMessageBox.question(
                self, "Stage position", f"Replace {p.name!r} with the stage now: ({x:.2f}, {y:.2f}) µm, {what}?") \
                != QtWidgets.QMessageBox.Yes:
            return
        r = _selected_row(self.table)
        self.settings.replace_position(p.name, x, y, z)
        self._save(f"{p.name!r}: ({x:.2f}, {y:.2f}) µm, {what}")
        self.table.selectRow(r)

    def record_wd(self, confirm: bool = True) -> bool:
        """WD1 and Z_cal from the column and the stage now (no other ΔZ changes)."""
        cal = self.settings.calibration
        wd, z = self.wd(), self.stage_z()
        if wd is None:
            self._error("the working distance is not known (microscope not connected?)")
            return False
        if z is None:
            self._error("the stage Z is not known (microscope not connected?)")
            return False
        if confirm and QtWidgets.QMessageBox.question(
                self, "Record WD", f"WD1 = {wd:.4f} mm at Z {z:.1f} µm for {cal.name!r}?\n\n(Focused and stigmated "
                                   "on it at a high magnification?  The chip heights must be recorded at this Z.)") \
                != QtWidgets.QMessageBox.Yes:
            return False
        r = _selected_row(self.table)
        self.settings.record_wd(wd, z)
        self._save(f"{cal.name!r}: WD1 {wd:.4f} mm at Z_cal {z:.1f} µm")
        if r >= 0:
            self.table.selectRow(r)
        return True

    def add_manual(self) -> None:
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Stage position")
        f = QtWidgets.QFormLayout(d)
        name = QtWidgets.QLineEdit(self.settings.unique_name())
        x, y = _spin(0, 0, 1e6, 2), _spin(0, 0, 1e6, 2)
        dz = _spin(0, -1e6, 1e6, 1)
        dz.setToolTip("its Z relative to the calibration sample's (µm, positive away from the gun)")
        f.addRow("name", name)
        f.addRow("X", x)
        f.addRow("Y", y)
        f.addRow("ΔZ", dz)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.add(name.text(), x.value(), y.value(), dz.value())

    def delete_selected(self) -> None:
        p = self._selected()
        if p is None:
            return
        try:
            self.settings.delete_position(p.name)
        except ValueError as e:
            return self._error(str(e))
        self._save(f"deleted {p.name!r}")

    def _goto(self, xyz: bool) -> None:
        p = self._selected()
        if p is None:
            return
        z = wd = None
        if xyz:
            z = self.settings.z_of(p)
            if z is None:
                return self._error(f"{p.name!r}: Z not known (" + ("Record WD on the calibration sample"
                                   if self.settings.calibration.z_um is None else "no ΔZ: Replace with current") + ")")
            wd = self.settings.calibration.wd_mm
        self.gotoRequested.emit(p.x, p.y, z, wd)


# ====================================================================== chips
class ChipsTab(QtWidgets.QWidget):
    """The schedule's chips: name, the nominal stage position of the CCS origin, the alignment, the
    height (the WD in focus at Z_cal: none, constant WD2 or a map of WD2x)."""

    changed = QtCore.Signal()
    resetRequested = QtCore.Signal(str)  # forget the chip's Global Align
    moveRequested = QtCore.Signal(str)  # Move pattern… dialog
    heightRequested = QtCore.Signal(str)  # Height… dialog
    selectionChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.schedule = None
        self.stage_position = lambda: None
        self.alignment = lambda name: ""  # () -> status text of a chip's CCS → SCS map
        self.z_cal = lambda: None  # () -> the calibration sample's Z (µm) | None
        self.table = _table(["Name", "Origin X (µm)", "Origin Y (µm)", "Alignment", "Height", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(lambda: self.selectionChanged.emit(self.selected()))
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        info = QtWidgets.QLabel("A chip has its own coordinates (CCS, the GDS top level of its Exposures + their "
                                "offsets).  The origin is where CCS (0, 0) nominally is on the stage; a Global Align "
                                "measures the real map, used by every later Exposure on the chip.  Its height is "
                                "the WD in focus on it with the stage at the calibration sample's Z (WD2): the "
                                "focus follows it during the run, Z stays.")
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
        self.b_height.setToolTip("the chip's height: the focus kept, one WD2 (focus on a feature at Z_cal), or a "
                                 "WD map (≥ 3 points)")
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
                    if k == 4 and (c.height.mode == "none" or c.height.problem(self.z_cal(), check_z=True)):
                        it.setForeground(QtGui.QBrush(QtGui.QColor("#fb4")))
                        it.setToolTip(c.height.problem(self.z_cal(), check_z=True) or "the focus is kept")
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
    """A chip's height as the WD in focus on it, with the stage at the calibration sample's Z
    (*z_cal*): the focus kept, one WD2, or a map of WD2x at ≥ 3 points (CCS) with a least-squares
    plane.  *Record here* takes the WD and the stage now – focus on the chip first, the stage Z at
    Z_cal (refused at another Z) – and converts its X, Y to CCS through the chip's present map
    (*to_ccs*); *Go to* moves to a point (*goto(x, y, z, wd)*, the point through *to_scs*, Z_cal,
    its WD).  :meth:`value` is the edited :class:`~easyebl.litho.height.ChipHeight`; *fields_ccs*:
    the chip's field centres (for the WD range); *nominal_wd*: the field size's (the value to start
    with); *wd1*: the calibration sample's (a WD2 more than 1 mm from it is warned about)."""

    def __init__(self, chip, stage_position, stage_z, wd, to_ccs, to_scs, goto=None, fields_ccs=(),
                 z_cal=None, nominal_wd=None, wd1=None, parent=None):
        super().__init__(parent)
        from ..height import ChipHeight

        self.setWindowTitle(f"Height of {chip.name}")
        self.h = ChipHeight.from_dict(chip.height.to_dict())
        self.stage_position, self.stage_z, self.wd = stage_position, stage_z, wd
        self.to_ccs, self.to_scs, self.goto = to_ccs, to_scs, goto
        self.fields_ccs = list(fields_ccs)
        self.z_cal, self.nominal_wd, self.wd1 = z_cal, nominal_wd, wd1
        intro = QtWidgets.QLabel(
            "The WD in focus on the chip (WD2), with the stage Z left at the calibration sample's Z_cal"
            + (f" ({z_cal:.1f} µm)" if z_cal is not None else " (not recorded: Stage positions → Record WD)")
            + ".  During the run Z stays at Z_cal and the focus is set to the chip's WD at each field.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa")
        self.mode = QtWidgets.QComboBox()
        for k, t in (("none", "none: keep the focus"), ("constant", "constant: one WD2"),
                     ("map", "map: a plane through WD2 at ≥ 3 points")):
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.h.mode)))
        self.mode.currentIndexChanged.connect(self._mode)
        # constant
        self.g_const = QtWidgets.QGroupBox("Constant")
        fc = QtWidgets.QFormLayout(self.g_const)
        self.wd_value = _spin(self.h.wd_mm or wd1 or nominal_wd or 15.0, 4, 60, 4, " mm", 0.01)
        self.wd_value.valueChanged.connect(self._typed)
        b_rec = QtWidgets.QPushButton("Record here")
        b_rec.setToolTip("the WD in focus now (focus on a feature of the chip first, the stage Z at Z_cal)")
        b_rec.clicked.connect(self.record_constant)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.wd_value, 1)
        row.addWidget(b_rec)
        fc.addRow("WD2", row)
        # map
        self.g_map = QtWidgets.QGroupBox("WD map (chip coordinates)")
        vm = QtWidgets.QVBoxLayout(self.g_map)
        self.table = _table(["X (µm)", "Y (µm)", "WD (mm)"])
        self.table.setMinimumHeight(130)
        self.table.itemChanged.connect(self._points_edited)
        vm.addWidget(self.table)
        hb = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Record here")
        b_add.setToolTip("a point at the stage now, the WD in focus (focus on the chip there first, Z at Z_cal)")
        b_add.clicked.connect(self.record_point)
        b_go = QtWidgets.QPushButton("Go to")
        b_go.setToolTip("move to the selected point (Z_cal), the focus to its WD")
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

    def _typed(self, v):
        self.h.wd_mm = float(v)
        if self.h.z_um is None:  # typed: taken as focused at Z_cal
            self.h.z_um = self.z_cal
        self._info()

    def _mode(self, *_):
        self.h.mode = self.mode.currentData()
        if self.h.mode == "constant" and self.h.wd_mm is None:
            self._typed(self.wd_value.value())
        self.g_const.setVisible(self.h.mode == "constant")
        self.g_map.setVisible(self.h.mode == "map")
        self._info()

    def _fill(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.h.points))
        for r, pt in enumerate(self.h.points):
            for c, v in enumerate(pt):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{v:.2f}" if c < 2 else f"{v:.4f}"))
        self.table.blockSignals(False)
        self._info()

    def _info(self):
        from ..schedule import wd_far_problem

        txt = self.h.describe()
        warn = []
        if self.h.mode != "none":
            if self.fields_ccs:
                wds = [w for w in (self.h.wd_at(c) for c in self.fields_ccs) if w is not None]
                if wds:
                    txt += f" · WD over the {len(wds)} field(s): {min(wds):.4f} … {max(wds):.4f} mm"
            p = self.h.problem(self.z_cal, check_z=True)
            if p:
                warn.append(p)
            w = wd_far_problem(self.h, self.wd1)
            if w:
                warn.append(w)
        self.info.setText(txt + ("".join(f"\n⚠ {w}" for w in warn)))
        self.info.setStyleSheet("color:#fb4" if warn else "")

    def _here(self):
        """(x, y, z, wd) now – the stage at Z_cal – or None (the reason in the message)."""
        from ..height import z_problem

        pos, z, wd = self.stage_position(), self.stage_z(), self.wd()
        if pos is None or z is None:
            self.msg.setText("the stage position is not known (microscope not connected?)")
            return None
        if wd is None:
            self.msg.setText("the working distance is not known (microscope not connected?)")
            return None
        p = z_problem(z, self.z_cal, "the stage")
        if p:
            self.msg.setText(p.replace("was recorded", "is") if self.z_cal is not None else p)
            return None
        self.msg.clear()
        return float(pos[0]), float(pos[1]), float(z), float(wd)

    def record_constant(self):
        here = self._here()
        if here is not None:
            self.h.z_um = here[2]
            self.wd_value.setValue(round(here[3], 4))
            self._typed(round(here[3], 4))

    def record_point(self):
        here = self._here()
        if here is None:
            return
        c = self.to_ccs((here[0], here[1]))
        self.h.z_um = here[2]
        self.h.points.append((float(c[0]), float(c[1]), round(here[3], 4)))
        self._fill()
        self.table.selectRow(len(self.h.points) - 1)

    def _points_edited(self, *_):
        out = []
        try:
            for r in range(self.table.rowCount()):
                out.append(tuple(float(self.table.item(r, c).text()) for c in range(3)))
        except (ValueError, AttributeError):
            self.msg.setText("a point is X, Y in µm and the WD in mm")
            return
        self.msg.clear()
        self.h.points = out
        if self.h.z_um is None:
            self.h.z_um = self.z_cal
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
        x, y, wd = self.h.points[r]
        scs = self.to_scs((x, y))
        self.goto(float(scs[0]), float(scs[1]), self.z_cal, wd)

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
    and markers), ``field_centres(op)`` (CCS centres of its fields), ``field_sizes()``
    ([FieldSize]) and ``stored_beamcal()`` (the beam calibration stored for the schedule's field
    size | None)."""

    changed = QtCore.Signal()
    pickGdsRequested = QtCore.Signal()
    #: pick markers on the plot: ("global", k) one Global marker, ("global_all", -1) all in turn, ("local", -1)
    pickMarkersRequested = QtCore.Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.op: Operation | None = None
        self.schedule = None  # the schedule of the op (which steps are fixed)
        self.presets = lambda: None
        self.positions = lambda: {}
        self.chips = lambda: []
        self.top_cells = lambda path: []
        self.current_for = lambda op: None
        self.fields_info = lambda op: ""
        self.field_centres = lambda op: []
        self.field_sizes = lambda: []
        self.stored_beamcal = lambda: None
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

    def _beamcal(self, f) -> None:
        op = self.op
        info = QtWidgets.QLabel(
            "On the calibration sample (Stage positions; the stage at Z_cal, the focus at WD1): one image of its "
            "square grid over the whole field of view, at ≤ 0.1 µm pixels (5000 × 5000 for a 500 µm field).  The "
            "fit gives the beam's X / Y scale, shear and pincushion distortion; valid for the field size, at any WD.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f.addRow(info)
        self._int_spin(f, "image", "pixels", 200, 20000, 500, " px across")
        self._cfg_spin(f, "grid pitch", "pitch_um", lo=0.05, hi=100, decimals=3, suffix=" µm", step=0.1)
        self._int_spin(f, "frames averaged", "frames", 1, 64)
        self._cfg_spin(f, "refuse a fit worse than", "max_rms_um", 1e3, lo=1, hi=5000, decimals=0, suffix=" nm",
                       step=10)
        self._cfg_spin(f, "low-pass", "filter_periods", lo=0.5, hi=20, decimals=1, suffix=" periods", step=0.5)
        mf = QtWidgets.QCheckBox("confirm the focus by hand first, at")
        mf.setChecked(bool(op.config.manual_focus))
        mf.setToolTip("before the image: zoom in on the grid (the focus at WD1 – it should be very close) and wait for "
                      "you to focus and stigmate; the WD then is WD1 from there on")
        mm = mag_combo(op.config.focus_magnification, lambda m: self._cfg("focus_magnification", int(m)))
        mm.setEnabled(mf.isChecked())
        mf.toggled.connect(lambda on: (self._cfg("manual_focus", bool(on)), mm.setEnabled(on)))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(mf)
        row.addWidget(mm, 1)
        f.addRow(row)
        ru = QtWidgets.QCheckBox("use a stored calibration of the field size (else measure)")
        ru.setChecked(op.reuse)
        ru.setToolTip("beam calibrations are stored per field size (litho.json, every accepted one); with this on "
                      "the step uses the stored one of the schedule's field size and measures nothing – it "
                      "measures when none is stored")
        ru.toggled.connect(lambda on: (self._set("reuse", bool(on)), self._stored_text()))
        f.addRow(ru)
        self.stored = QtWidgets.QLabel()
        self.stored.setWordWrap(True)
        f.addRow(self.stored)
        self._stored_text()

    def _stored_text(self) -> None:
        """What is stored for the schedule's field size (and what the step will do)."""
        lab = getattr(self, "stored", None)
        if lab is None or not isinstance(self.op, BeamCalibrationStep):
            return
        fs = self.schedule.field_size if self.schedule is not None else ""
        bc = self.stored_beamcal()
        if not fs:
            text, col = "no field size chosen (the first Column setting)", "#fb4"
        elif bc is None:
            text, col = f"none stored for field size {fs}: the step measures", "#9aa"
        else:
            text = (f"stored for field size {fs} ({bc.created.replace('T', ' ')}): {bc.describe()}"
                    + (" – used, nothing measured" if self.op.reuse else " – not used: the step measures a new one"))
            col = "#5d5" if self.op.reuse else "#9aa"
        lab.setText(text)
        lab.setStyleSheet(f"color:{col}")

    def _column(self, f, first: bool = False) -> None:
        op = self.op
        if first:
            fs = QtWidgets.QComboBox()
            for s in self.field_sizes() or []:
                fs.addItem(s.describe(), s.name)
            if op.field_size and fs.findData(op.field_size) < 0:
                fs.addItem(f"{op.field_size} (unknown)", op.field_size)
            if not op.field_size:
                fs.insertItem(0, "(choose a field size)", "")
            fs.setCurrentIndex(max(0, fs.findData(op.field_size)))
            fs.setToolTip("the schedule's field size (Calibration → Field sizes…): sets the magnification; the focus "
                          "goes to WD1.  The Microscope window must be at the same field size to run")
            fs.currentIndexChanged.connect(lambda _: self._set("field_size", fs.currentData() or ""))
            f.addRow("field size", fs)
            self.field_size_combo = fs
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
        cs = _spin(op.current_pa or 0.0, 0, 1e6, 2, " pA", 10)
        cs.setSpecialValueText("unknown")
        cs.setToolTip("nominal beam current: only for the dose readout of the fields")
        cs.valueChanged.connect(lambda val: self._set("current_pa", float(val) or None))
        f.addRow("nominal current", cs)
        if not first:
            note = QtWidgets.QLabel("The field size is the schedule's (the first Column setting).")
            note.setStyleSheet("color:#9aa")
            f.addRow(note)

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
            f.addRow("dose factor from", self._dose_source(op))
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
                row = QtWidgets.QHBoxLayout()
                row.addWidget(_pair(*mk, lambda p, k=k: self._set_marker(k, p)), 1)
                b = QtWidgets.QPushButton("Pick")
                b.setToolTip("click a polygon corner or centre of the GDS on the plot (any layer)")
                b.clicked.connect(lambda _=False, k=k: self.pickMarkersRequested.emit("global", k))
                row.addWidget(b)
                f.addRow(f"marker {k + 1}", row)
            b = QtWidgets.QPushButton("Pick all on the plot…")
            b.setToolTip("marker 1, 2, … in turn: click a polygon corner or centre of the GDS (any layer)")
            b.clicked.connect(lambda: self.pickMarkersRequested.emit("global_all", -1))
            self.b_pick_global = b
            f.addRow(b)
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
            b_pick = QtWidgets.QPushButton("Pick on the plot…")
            b_pick.setToolTip("each click on a polygon corner or centre of the GDS (any layer) adds a marker; Esc: done")
            b_pick.clicked.connect(lambda: self.pickMarkersRequested.emit("local", -1))
            self.b_pick_local = b_pick
            for w in (b_add, b_del, b_gen, b_pick):
                h.addWidget(w)
            lv.addLayout(h)
        v.addWidget(g)
        g = QtWidgets.QFormLayout()
        if op.global_align or op.local_align:
            self._spin_for(g, "marker view", "view_um", lo=0.01, hi=1e5)
        if op.global_align:
            t = self._spin_for(g, "alignment tolerance", "align_tolerance_um", lo=0.001, hi=100, step=0.1)
            t.setToolTip("the Global Align markers are measured twice (1 → 2 → 1 → 2): the two passes must agree "
                         "within this, else repeat / continue / abort")
        v.addLayout(g)
        self.refresh_info()

    def _dose_source(self, op) -> QtWidgets.QComboBox:
        """Which GDS property gives each shape's dose factor (× the dwell); the properties holding a
        number in the file are offered."""
        from ..gds import GDSDoseMapping, dose_properties

        box = QtWidgets.QComboBox()
        box.setToolTip("the dose factor of each shape (× the dwell per point): a property of the GDS "
                       "polygons, or none (all 1)")
        try:
            props = dose_properties(op.path)
        except Exception as e:  # noqa: BLE001
            props = {}
            self.msg.setText(f"GDS properties: {type(e).__name__}: {e}")
        box.addItem("none (all 1)", ("none", None))
        box.addItem("any property with a number", ("property", None))
        for attr, (n, lo, hi) in props.items():
            rng = f"{lo:g}" if lo == hi else f"{lo:g} … {hi:g}"
            box.addItem(f"property {attr} ({n} shapes, {rng})", ("property", attr))
        m = op.dose
        cur = (m.mode, m.property_attr if m.mode == "property" else None)
        if m.mode == "auto":
            box.addItem("auto (any property, else the datatype table)", ("auto", m.property_attr))
            cur = ("auto", m.property_attr)
        elif m.mode in ("datatype", "layer"):
            box.addItem(f"{m.mode} table (from an older schedule)", (m.mode, None))
            cur = (m.mode, None)
        i = next((k for k in range(box.count()) if tuple(box.itemData(k)) == cur), -1)
        if i < 0 and m.mode == "property":
            box.addItem(f"property {m.property_attr} (not in the file)", cur)
            i = box.count() - 1
        box.setCurrentIndex(max(0, i))

        def chosen(k):
            mode, attr = box.itemData(k)
            self._set("dose", GDSDoseMapping(mode=mode, property_attr=attr, table=dict(m.table), default=m.default,
                                             value_scale=m.value_scale))
        box.currentIndexChanged.connect(chosen)
        self.dose_box = box
        return box

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
        m.addAction(ColumnSetting.TITLE, lambda: self.add_op(ColumnSetting()))
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
        if isinstance(op, BeamCalibrationStep) and op.reuse and p is not None:
            nf = (f" (stored {p.beamcal.created.replace('T', ' ')})" if p.beamcal is not None
                  else " (none stored: measures)")
        title = op.title()
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
