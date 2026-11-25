"""Side tabs of the litho window: Stage Positions, Chips and Exposure Schedule."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..schedule import (KEEP_AS_IS, MAGNIFICATIONS, ORDERS, ColumnSetting, Exposure, FieldCorrection, Operation,
                        dose_readout)

__all__ = ["StagePositionsTab", "ChipsTab", "ScheduleTab", "OpEditor"]

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
class StagePositionsTab(QtWidgets.QWidget):
    """Named stage positions (X, Y, Z) of fixed objects (calibration samples, Faraday cup …), saved at once."""

    changed = QtCore.Signal()
    gotoRequested = QtCore.Signal(float, float, object)  # SCS µm, Z (None: keep Z)
    chipOriginRequested = QtCore.Signal(float, float)  # put a chip's CCS origin here

    def __init__(self, settings, stage_position=None, stage_z=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.stage_position = stage_position or (lambda: None)  # () -> (x, y) | None
        self.stage_z = stage_z or (lambda: None)  # () -> z | None
        self.table = _table(["Name", "X (µm)", "Y (µm)", "Z (µm)", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        b_cur = QtWidgets.QPushButton("Add current stage")
        b_cur.setToolTip("a new position at the stage's current X, Y and Z")
        b_cur.clicked.connect(self.add_current)
        b_add = QtWidgets.QPushButton("Add…")
        b_add.clicked.connect(self.add_manual)
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_del.clicked.connect(self.delete_selected)
        self.b_go = QtWidgets.QPushButton("Go to XYZ")
        self.b_go.setToolTip("move the stage to the selected position, Z included (where stored)")
        self.b_go.clicked.connect(lambda: self._goto(True))
        self.b_go_xy = QtWidgets.QPushButton("Go to XY")
        self.b_go_xy.setToolTip("move the stage to the selected X, Y; Z stays")
        self.b_go_xy.clicked.connect(lambda: self._goto(False))
        self.b_chip = QtWidgets.QPushButton("Chip origin here")
        self.b_chip.setToolTip("put the CCS origin of the selected chip (Chips tab) at the selected position")
        self.b_chip.clicked.connect(self._chip_origin)
        g = QtWidgets.QGridLayout()
        g.addWidget(b_cur, 0, 0)
        g.addWidget(b_add, 0, 1)
        g.addWidget(self.b_del, 0, 2)
        g.addWidget(self.b_go, 1, 0)
        g.addWidget(self.b_go_xy, 1, 1)
        g.addWidget(self.b_chip, 1, 2)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addLayout(g)
        lay.addWidget(self.msg)
        self.refresh()

    def refresh(self) -> None:
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.settings.stage_positions))
            for r, p in enumerate(self.settings.stage_positions):
                vals = (p.name, f"{p.x:.2f}", f"{p.y:.2f}", "" if p.z is None else f"{p.z:.2f}", p.note)
                for c, v in enumerate(vals):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        finally:
            self.table.blockSignals(False)

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
        self.changed.emit()

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
                p.z = float(txt) if txt else None
            else:
                p.note = txt
        except ValueError as e:
            self.refresh()
            return self._error(str(e))
        self._save("saved")

    def add(self, name: str, x: float, y: float, z: float | None = None) -> None:
        try:
            self.settings.add_position(name, x, y, z)
        except ValueError as e:
            return self._error(str(e))
        self.refresh()
        self.table.selectRow(len(self.settings.stage_positions) - 1)
        self._save(f"added {name!r}")

    def add_current(self) -> None:
        pos = self.stage_position()
        if pos is None:
            return self._error("the stage position is not known (microscope not connected?)")
        name, ok = QtWidgets.QInputDialog.getText(self, "Stage position", "Name:",
                                                  text=self.settings.unique_name())
        if ok:
            self.add(name, pos[0], pos[1], self.stage_z())

    def add_manual(self) -> None:
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Stage position")
        f = QtWidgets.QFormLayout(d)
        name = QtWidgets.QLineEdit(self.settings.unique_name())
        x, y = _spin(0, 0, 1e6, 2), _spin(0, 0, 1e6, 2)
        z = _spin(0, 0, 1e6, 2)
        zc = QtWidgets.QCheckBox("store Z")
        zc.toggled.connect(z.setEnabled)
        z.setEnabled(False)
        f.addRow("name", name)
        f.addRow("X", x)
        f.addRow("Y", y)
        f.addRow(zc, z)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.add(name.text(), x.value(), y.value(), z.value() if zc.isChecked() else None)

    def delete_selected(self) -> None:
        r = _selected_row(self.table)
        if r < 0:
            return
        name = self.settings.stage_positions[r].name
        del self.settings.stage_positions[r]
        self.refresh()
        self._save(f"deleted {name!r}")

    def _goto(self, with_z: bool) -> None:
        p = self._selected()
        if p is None:
            return
        if with_z and p.z is None:
            self._error(f"{p.name!r} has no Z stored: moving X / Y only")
        self.gotoRequested.emit(p.x, p.y, p.z if with_z else None)

    def _chip_origin(self) -> None:
        p = self._selected()
        if p is not None:
            self.chipOriginRequested.emit(p.x, p.y)


# ====================================================================== chips
class ChipsTab(QtWidgets.QWidget):
    """The schedule's chips: name, the nominal stage position of the CCS origin, the alignment."""

    changed = QtCore.Signal()
    resetRequested = QtCore.Signal(str)  # forget the chip's Global Align
    moveRequested = QtCore.Signal(str)  # Move the chip… dialog
    selectionChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.schedule = None
        self.stage_position = lambda: None
        self.alignment = lambda name: ""  # () -> status text of a chip's CCS → SCS map
        self.table = _table(["Name", "Origin X (µm)", "Origin Y (µm)", "Alignment", "Note"])
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(lambda: self.selectionChanged.emit(self.selected()))
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        info = QtWidgets.QLabel("A chip has its own coordinates (CCS, the GDS top level of its Exposures + their "
                                "offsets).  The origin is where CCS (0, 0) nominally is on the stage; a Global Align "
                                "measures the real map, used by every later Exposure on the chip.")
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
        g = QtWidgets.QGridLayout()
        g.addWidget(b_add, 0, 0)
        g.addWidget(self.b_del, 0, 1)
        g.addWidget(self.b_here, 0, 2)
        g.addWidget(self.b_move, 1, 0)
        g.addWidget(self.b_reset, 1, 1)
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
                vals = (c.name, f"{c.origin[0]:.3f}", f"{c.origin[1]:.3f}", self.alignment(c.name), c.note)
                for k, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if k == 3:
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
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
            elif k == 4:
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
    (stage position names), ``chips()`` (chip names), ``top_cells(path)``, ``current_for(op)``
    (nominal current at that step | None), ``fields_info(op)`` (text about the Exposure's fields
    and markers) and ``field_centres(op)`` (CCS centres of its fields)."""

    changed = QtCore.Signal()
    pickGdsRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.op: Operation | None = None
        self.presets = lambda: None
        self.positions = lambda: []
        self.chips = lambda: []
        self.top_cells = lambda path: []
        self.current_for = lambda op: None
        self.fields_info = lambda op: ""
        self.field_centres = lambda op: []
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
        self.title.setText(op.TITLE)
        self.op = op
        v = QtWidgets.QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QFormLayout()
        v.addLayout(top)
        lab = QtWidgets.QLineEdit(op.label)
        lab.setPlaceholderText("(optional)")
        lab.editingFinished.connect(lambda: self._set("label", lab.text()))
        top.addRow("label", lab)
        if isinstance(op, ColumnSetting):
            self._column(top)
            return
        en = QtWidgets.QCheckBox("enabled")
        en.setChecked(op.enabled)
        en.toggled.connect(lambda on: self._set("enabled", bool(on)))
        top.addRow(en)
        if isinstance(op, FieldCorrection):
            info = QtWidgets.QLabel("Field calibration: images at known stage shifts give the stage → beam rotation, "
                                    "valid for the present magnification, beam preset and WD (Field → Field "
                                    "calibration settings… for how).  The images go through the beam calibration of "
                                    "the condition.")
            info.setWordWrap(True)
            top.addRow(info)
            pos = QtWidgets.QComboBox()
            pos.addItem("the configured position", "")
            for n in self.positions():
                pos.addItem(n, n)
            if op.position and pos.findData(op.position) < 0:
                pos.addItem(f"{op.position} (unknown)", op.position)
            pos.setCurrentIndex(max(0, pos.findData(op.position)))
            pos.currentIndexChanged.connect(lambda _: self._set("position", pos.currentData()))
            top.addRow("at", pos)
            notes = QtWidgets.QLineEdit(op.notes)
            notes.editingFinished.connect(lambda: self._set("notes", notes.text()))
            top.addRow("notes", notes)
        elif isinstance(op, Exposure):
            self._exposure(v)

    # ---------------------------------------------------------------- column
    def _column(self, f) -> None:
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
        m.addAction(FieldCorrection.TITLE, lambda: self.add_op(FieldCorrection()))
        add.setMenu(m)
        self.add_menu = m
        self.b_add = add
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(add)
        self.b_dup = QtWidgets.QToolButton(text="Dup", toolTip="duplicate the selected step")
        self.b_del = QtWidgets.QToolButton(text="Del", toolTip="delete the selected step (not the first Column setting)")
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
        sf.addRow("pattern outside the FoV", self.s_policy)
        sf.addRow("area fill direction", self.s_fill)
        self.s_policy.currentIndexChanged.connect(self._settings_edited)
        self.s_fill.currentIndexChanged.connect(self._settings_edited)

        self.b_run = QtWidgets.QPushButton("Run")
        self.b_run.setToolTip("run the schedule from the first step")
        self.b_run.clicked.connect(lambda: self.runRequested.emit(0))
        self.b_from = QtWidgets.QPushButton("Run from selected")
        self.b_from.setToolTip("the first Column setting, then from the selected step")
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
        self.states = ["pending"] * len(schedule)
        st = schedule.settings
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
        it.setText(f"{i + 1}. {STATE_MARK[st]}{op.title()}{nf}" + ("  ⚠" if warns else "")
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
        f.setBold(isinstance(op, ColumnSetting))
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
        self.b_dup.setEnabled(idle and i > 0)
        self.b_up.setEnabled(idle and s is not None and s.can_move(i, i - 1))
        self.b_down.setEnabled(idle and 0 < i < n - 1)

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
        self.changed.emit()

    def _last_mag(self) -> float:
        i = self.list.currentRow()
        for op in reversed(self.schedule.operations[:max(1, i + 1)]):
            if isinstance(op, ColumnSetting):
                return op.magnification
        return self.schedule[0].magnification

    # ---------------------------------------------------------------- edits
    def add_op(self, op: Operation) -> None:
        i = self.list.currentRow()
        at = len(self.schedule) if i < 0 else i + 1
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
