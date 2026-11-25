"""Side tabs of the litho window: Stage Positions and Exposure Schedule."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..schedule import (KEEP_AS_IS, MAGNIFICATIONS, ColumnSetting, ExposeField, FieldCorrection, Operation,
                        PatternSpec, dose_readout)

__all__ = ["StagePositionsTab", "ScheduleTab", "OpEditor"]

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


# ====================================================================== stage positions
class StagePositionsTab(QtWidgets.QWidget):
    """Named SCS positions of fixed objects (calibration sample, Faraday cup …), saved at once."""

    changed = QtCore.Signal()
    gotoRequested = QtCore.Signal(float, float)  # SCS µm
    newFieldRequested = QtCore.Signal(float, float)  # an Expose field centred at this position (SCS)

    def __init__(self, settings, stage_position=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.stage_position = stage_position or (lambda: None)  # () -> (x, y) | None
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "X (µm)", "Y (µm)", "Note"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._edited)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        b_cur = QtWidgets.QPushButton("Add current stage")
        b_cur.setToolTip("a new position at the stage's current X / Y")
        b_cur.clicked.connect(self.add_current)
        b_add = QtWidgets.QPushButton("Add…")
        b_add.clicked.connect(self.add_manual)
        self.b_del = QtWidgets.QPushButton("Delete")
        self.b_del.clicked.connect(self.delete_selected)
        self.b_go = QtWidgets.QPushButton("Go to")
        self.b_go.setToolTip("move the stage to the selected position")
        self.b_go.clicked.connect(self._goto)
        self.b_step = QtWidgets.QPushButton("New field here")
        self.b_step.setToolTip("add an Expose field centred at the selected position to the schedule")
        self.b_step.clicked.connect(self._new_field)
        g = QtWidgets.QGridLayout()
        g.addWidget(b_cur, 0, 0)
        g.addWidget(b_add, 0, 1)
        g.addWidget(self.b_del, 0, 2)
        g.addWidget(self.b_go, 1, 0)
        g.addWidget(self.b_step, 1, 1, 1, 2)
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
                for c, v in enumerate((p.name, f"{p.x:.2f}", f"{p.y:.2f}", p.note)):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        finally:
            self.table.blockSignals(False)

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _selected_name(self) -> str:
        r = self._selected_row()
        return self.settings.stage_positions[r].name if 0 <= r < len(self.settings.stage_positions) else ""

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
            else:
                p.note = txt
        except ValueError as e:
            self.refresh()
            return self._error(str(e))
        self._save("saved")

    def add(self, name: str, x: float, y: float) -> None:
        try:
            self.settings.add_position(name, x, y)
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
            self.add(name, *pos)

    def add_manual(self) -> None:
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Stage position")
        f = QtWidgets.QFormLayout(d)
        name = QtWidgets.QLineEdit(self.settings.unique_name())
        x, y = _spin(0, 0, 1e6, 2), _spin(0, 0, 1e6, 2)
        f.addRow("name", name)
        f.addRow("X", x)
        f.addRow("Y", y)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.add(name.text(), x.value(), y.value())

    def delete_selected(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        name = self.settings.stage_positions[r].name
        del self.settings.stage_positions[r]
        self.refresh()
        self._save(f"deleted {name!r}")

    def _new_field(self) -> None:
        r = self._selected_row()
        if r >= 0:
            p = self.settings.stage_positions[r]
            self.newFieldRequested.emit(p.x, p.y)

    def _goto(self) -> None:
        r = self._selected_row()
        if r >= 0:
            p = self.settings.stage_positions[r]
            self.gotoRequested.emit(p.x, p.y)


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


# ====================================================================== op editor
class OpEditor(QtWidgets.QWidget):
    """Form for one operation; edits apply to the operation at once (``changed``).

    The window sets the providers: ``presets()`` (beam preset names | None), ``cells(path)``
    (cell names of a GDS file), ``current_for(op)`` (nominal current at that step | None)."""

    changed = QtCore.Signal()
    pickGdsRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.op: Operation | None = None
        self.presets = lambda: None
        self.positions = lambda: []  # () -> stage position names
        self.cells = lambda path: []
        self.current_for = lambda op: None
        self.readout = None
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

    @staticmethod
    def _pair(x, y, setter) -> QtWidgets.QWidget:
        sx, sy = _spin(x, decimals=3), _spin(y, decimals=3)
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

    def set_op(self, op: Operation | None) -> None:
        self.op = None
        self.readout = None
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
            info = QtWidgets.QLabel("Field calibration: images at known stage shifts give the stage → beam map, valid "
                                    "for the present magnification and beam preset (Field → Field calibration "
                                    "settings… for how).")
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
        elif isinstance(op, ExposeField):
            self._expose(v)

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

    # ---------------------------------------------------------------- expose
    def _expose(self, v) -> None:
        op = self.op

        # --- field centre
        g = QtWidgets.QGroupBox("Field centre (stage)")
        f = QtWidgets.QFormLayout(g)
        centre = self._pair(*op.centre, lambda p: self._set("centre", p))
        f.addRow("X, Y", centre)
        cc = QtWidgets.QCheckBox("correct by the global alignment")
        cc.setToolTip("move to the centre recomputed from the field of the last Global Align: its measured origin "
                      "+ the spacing between the centres, transformed by the measured alignment")
        cc.setChecked(op.correct_centre)
        cc.toggled.connect(lambda on: self._set("correct_centre", bool(on)))
        f.addRow(cc)
        v.addWidget(g)

        # --- pattern
        g = QtWidgets.QGroupBox("Pattern (WCS: (0, 0) = field centre)")
        f = QtWidgets.QFormLayout(g)
        kind = QtWidgets.QComboBox()
        for k, t in (("gds", "GDS cell"), ("dummy", "dummy test pattern"), ("none", "none (alignment only)")):
            kind.addItem(t, k)
        kind.setCurrentIndex(kind.findData(op.pattern.kind))
        kind.currentIndexChanged.connect(lambda _: self._set_pattern_kind(kind.currentData()))
        self.kind_box = kind
        f.addRow("pattern", kind)
        if op.pattern.kind == "gds":
            b = QtWidgets.QPushButton(os.path.basename(op.pattern.path) if op.pattern.path else "Choose GDS file…")
            b.setToolTip(op.pattern.path or "")
            b.clicked.connect(self.pickGdsRequested.emit)
            f.addRow("file", b)
            cell = QtWidgets.QComboBox()
            try:
                names = self.cells(op.pattern.path) if op.pattern.path else []
            except Exception as e:  # noqa: BLE001
                names = []
                self.msg.setText(f"{type(e).__name__}: {e}")
            cell.addItems(names)
            if op.pattern.cell and cell.findText(op.pattern.cell) < 0:
                cell.addItem(op.pattern.cell)
            cell.setCurrentText(op.pattern.cell)
            cell.currentTextChanged.connect(self._set_cell)
            f.addRow("cell", cell)
            if not op.pattern.placement.is_identity():
                o = op.pattern.placement.offset
                f.addRow(QtWidgets.QLabel(f"recentred by ({o[0]:+.3f}, {o[1]:+.3f}) µm (imported field)"))
            else:
                f.addRow(QtWidgets.QLabel("the cell is taken as centred on its origin"))
        elif op.pattern.kind == "dummy":
            sz = _spin(op.pattern.size_um, 0.01, 1e5)
            sz.valueChanged.connect(lambda val: self._set_dummy(float(val)))
            f.addRow("size", sz)
        v.addWidget(g)

        # --- dose
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
        g = QtWidgets.QGroupBox("Global Align (markers in WCS: µm from the field centre)")
        g.setCheckable(True)
        g.setChecked(op.global_align)
        g.toggled.connect(self._toggle_global)
        f = QtWidgets.QFormLayout(g)
        if op.global_align:
            n = QtWidgets.QSpinBox()
            n.setRange(2, 4)
            n.setValue(max(2, min(4, len(op.markers))))
            n.valueChanged.connect(self._marker_count)
            f.addRow("markers", n)
            for k, mk in enumerate(op.markers):
                f.addRow(f"marker {k + 1}", self._pair(*mk, lambda p, k=k: self._set_marker(k, p)))
            f.addRow(QtWidgets.QLabel("2 markers: rotation, scale, shift · 3–4: also stretch and shear"))
        v.addWidget(g)

        # --- local XY
        g = QtWidgets.QGroupBox("Local XY (one marker in WCS, shift only)")
        g.setCheckable(True)
        g.setChecked(op.local_xy)
        g.toggled.connect(lambda on: self._set("local_xy", bool(on), rebuild=True))
        f = QtWidgets.QFormLayout(g)
        if op.local_xy:
            f.addRow("marker", self._pair(*op.local_marker, lambda p: self._set("local_marker", p)))
        v.addWidget(g)
        if op.global_align or op.local_xy:
            g = QtWidgets.QFormLayout()
            self._spin_for(g, "marker view", "view_um", lo=0.01, hi=1e5)
            v.addLayout(g)

    def _readout(self, *_):
        op = self.op
        if isinstance(op, ExposeField) and self.readout is not None:
            self.readout.setText(dose_readout(op.dwell_us, op.step_nm, self.current_for(op)))

    def _set_pattern_kind(self, kind: str) -> None:
        op = self.op
        if op is None or op.pattern.kind == kind:
            return
        old = op.pattern
        op.pattern = PatternSpec(kind, old.path, old.cell, old.layers, old.dose, size_um=old.size_um) \
            if kind == "gds" else PatternSpec(kind, size_um=old.size_um)
        self.changed.emit()
        self._rebuild()
        if kind == "gds" and not old.path:
            QtCore.QTimer.singleShot(0, self.pickGdsRequested.emit)

    def _set_cell(self, cell: str) -> None:
        op = self.op
        if op is not None and cell and cell != op.pattern.cell:
            op.pattern = PatternSpec("gds", op.pattern.path, cell, op.pattern.layers, op.pattern.dose)
            self.changed.emit()

    def _set_dummy(self, size: float) -> None:
        if self.op is not None:
            self.op.pattern = PatternSpec("dummy", size_um=size)
            self.changed.emit()

    def _toggle_global(self, on: bool) -> None:
        op = self.op
        if op is None:
            return
        if on and len(op.markers) < 2:
            op.markers = (list(op.markers) + [(-10.0, -10.0), (10.0, 10.0)])[:2]
        self._set("global_align", bool(on), rebuild=True)

    def _marker_count(self, n: int) -> None:
        op = self.op
        mk = list(op.markers)
        defaults = [(-10.0, -10.0), (10.0, 10.0), (-10.0, 10.0), (10.0, -10.0)]
        while len(mk) < n:
            mk.append(defaults[len(mk)])
        op.markers = mk[:n]
        self.changed.emit()
        self._rebuild()

    def _set_marker(self, k: int, p) -> None:
        mk = list(self.op.markers)
        mk[k] = p
        self._set("markers", mk)

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
        self.new_field_centre = lambda: (0.0, 0.0)  # where a new Expose field goes (the stage, if known)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._select)
        add = QtWidgets.QToolButton()
        add.setText("Add ▾")
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(add)
        m.addAction(ExposeField.TITLE, lambda: self.add_op(ExposeField(centre=self.new_field_centre())))
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
        sp.setSizes([280, 560])
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
        warns = self.plans[i].warnings if self.plans is not None and i < len(self.plans) else []
        it.setText(f"{i + 1}. {STATE_MARK[st]}{op.title()}" + ("  ⚠" if warns else ""))
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
