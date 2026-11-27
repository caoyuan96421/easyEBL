"""The Lithography window: stage view (SCS with the chips and their fields), Stage Positions, Chips
and Exposure Schedule tabs, calibration and alignment status.

Opened from the easyEBL menu (Open → Lithography).  The window works through a
:class:`~easyebl.litho.runner.LithoHardware`: :class:`~.hardware.BackendHardware` on the
easyEBL backend, or :class:`~easyebl.litho.runner.SimulatedHardware` (tests, demos).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..coords import stage_rotation
from ..fieldcal import Condition
from ..height import Z_MATCH_UM
from ..runner import LithoHardware, ScheduleRunner
from ..schedule import (FIXED_STEPS, ColumnSetting, Exposure, PatternLibrary, Schedule, ScheduleError,
                        assign_markers, calibration_problem, exposure_fields, field_size_of, find_beamcal, plan)
from ..settings import LithoSettings
from .beamcal_window import BeamCalWindow
from .dialogs import FieldCalSetupDialog, FieldSizesDialog, MoveFieldsDialog, MoveStageDialog, StageAxesDialog
from .fieldcal_window import FieldCalWindow
from .hardware import GuiBridge
from .tabs import ChipsTab, HeightDialog, PositionsHub, ScheduleTab, StagePositionsTab
from .view import StageView, snap_targets

__all__ = ["LithoWindow"]

SCHEDULE_FILTER = "easyEBL schedule (*.ebls.json *.json);;All files (*)"
GDS_FILTER = "GDSII (*.gds *.gds2 *.gdsii);;All files (*)"


class _RunSignals(QtCore.QObject):
    event = QtCore.Signal(str, object)
    progress = QtCore.Signal(float)


class LithoWindow(QtWidgets.QMainWindow):
    def __init__(self, hardware=None, settings: LithoSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings or LithoSettings.shared()  # shared with the Microscope window
        self.bridge = GuiBridge(self)
        self.sig = _RunSignals(self)
        self.sig.event.connect(self._on_event)
        if hardware is not None and not isinstance(hardware, LithoHardware) and callable(hardware):
            hardware = hardware(self)  # a factory: (window) -> LithoHardware (uses window.bridge)
        self.hw: LithoHardware = hardware or LithoHardware()
        if hasattr(self.hw, "status"):
            self.hw.status = lambda text: self.sig.event.emit("status", {"text": text})
        if hasattr(self.hw, "progress"):
            self.hw.progress = self.sig.progress.emit
        self.axes = self.hw.stage_axes()  # Hardware → Configure… → Microscope
        self.library = PatternLibrary()
        self.schedule = Schedule()
        self.chips: dict = {}  # chip name -> measured ChipTransform (this window session only)
        self.fieldcal_window: FieldCalWindow | None = None
        self.beamcal_window: BeamCalWindow | None = None
        self._cal_future = None  # a result waiting for Accept / Discard
        self._beam_future = None
        self._cal_in_schedule = False
        self.last_map = None  # FieldMap of the field written last
        self.runner: ScheduleRunner | None = None
        self._thread: threading.Thread | None = None
        self.modified = False
        self.ask_on_close = True
        self.plans = None

        # --- central view
        self.view = StageView()
        self.view.stepClicked.connect(lambda i: self.tab_schedule.select(i))
        self.view.contextRequested.connect(self._context_menu)
        self.view.markerPicked.connect(self._marker_picked)
        self.view.pickCancelled.connect(self.stop_pick)
        self._picking = None  # marker picking on the plot: {op, kind, k, n}
        self.setCentralWidget(self.view)

        # --- side tabs
        self.tab_positions = StagePositionsTab(self.settings, self.hw.stage_position, self.hw.stage_z, self._wd)
        self.tab_positions.changed.connect(self._positions_changed)
        self.tab_positions.gotoRequested.connect(self.goto_stage)
        self.tab_chips = ChipsTab()
        self.tab_chips.stage_position = self.hw.stage_position
        self.tab_chips.alignment = self._chip_status
        self.tab_chips.z_cal = lambda: self.settings.calibration.z_um
        self.tab_chips.changed.connect(self._chips_edited)
        self.tab_chips.resetRequested.connect(self.reset_chip)
        self.tab_chips.moveRequested.connect(lambda name: self.move_chip_dialog(name))
        self.tab_chips.heightRequested.connect(lambda name: self.height_dialog(name))
        self.tab_schedule = ScheduleTab()
        self.tab_schedule.new_exposure = self._new_exposure
        ed = self.tab_schedule.editor
        ed.presets = self.hw.preset_names
        ed.positions = self.settings.positions
        ed.chips = lambda: [c.name for c in self.schedule.chips]
        ed.top_cells = self.library.top_cells
        ed.current_for = self._current_for
        ed.fields_info = self._fields_info
        ed.field_centres = self._field_centres
        ed.field_sizes = lambda: self.settings.field_sizes
        ed.stored_beamcal = self._stored_beamcal
        ed.pickGdsRequested.connect(self.pattern_for_selected)
        ed.pickMarkersRequested.connect(lambda kind, k: self.pick_markers(kind, k))
        self.tab_schedule.changed.connect(self._schedule_edited)
        self.tab_schedule.selectionChanged.connect(lambda i: self._replan_soon())
        self.tab_schedule.runRequested.connect(self.run)
        self.tab_schedule.pauseRequested.connect(self.pause)
        self.tab_schedule.abortRequested.connect(self.abort)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_positions, "Stage Positions")
        self.tabs.addTab(self.tab_chips, "Chips")
        self.tabs.addTab(self.tab_schedule, "Exposure Schedule")
        self.tabs.setCurrentWidget(self.tab_schedule)
        dock = QtWidgets.QDockWidget("Lithography")
        dock.setObjectName("litho-side")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(460)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.side_dock = dock

        # --- log
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumWidth(180)
        self.sig.progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        ld = QtWidgets.QDockWidget("Messages")
        ld.setObjectName("litho-log")
        ld.setWidget(self.log_box)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, ld)
        ld.setMaximumHeight(140)
        self.log_dock = ld

        self._make_menus()
        self._make_status()
        self._replan_timer = QtCore.QTimer(self)
        self._replan_timer.setSingleShot(True)
        self._replan_timer.setInterval(120)
        self._replan_timer.timeout.connect(self.replan)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)
        self.set_schedule(self.schedule)
        self._set_running_ui(False)
        self._orient()
        self._positions_changed()
        self._tick()
        self.resize(1500, 950)

    # ================================================================ UI
    def _make_menus(self):
        mb = self.menuBar()
        tab = self.tab_schedule

        def act(menu, text, fn=None, shortcut=None):
            a = menu.addAction(text)
            if fn is not None:
                a.triggered.connect(lambda _=False, fn=fn: fn())
            if shortcut is not None:
                a.setShortcut(shortcut)
            return a

        m = mb.addMenu("&Schedule")
        self.a_new = act(m, "New", self.new_schedule, QtGui.QKeySequence.New)
        self.a_open = act(m, "Open…", self.open_dialog, QtGui.QKeySequence.Open)
        self.a_save = act(m, "Save", self.save, QtGui.QKeySequence.Save)
        self.a_save_as = act(m, "Save as…", self.save_as, QtGui.QKeySequence.SaveAs)
        m.addSeparator()
        act(m, "Validate", self.validate, "F7")
        self.a_run = act(m, "Run", lambda: self.run(0), "F5")
        self.a_run_from = act(m, "Run from the selected step", lambda: self.run(max(0, tab.selected())), "Shift+F5")
        self.a_pause = act(m, "Pause after this step")
        self.a_pause.setCheckable(True)
        self.a_pause.toggled.connect(tab.b_pause.setChecked)
        tab.b_pause.toggled.connect(self.a_pause.setChecked)
        self.a_abort = act(m, "Abort", self.abort)
        m.addSeparator()
        self.a_reset = act(m, "Reset the alignment of all chips", self.reset_all_chips)

        m = mb.addMenu("S&tep")
        self.a_add_exp = act(m, "Add Exposure", lambda: tab.add_op(self._new_exposure()), "Ctrl+E")
        self.a_add_gds = act(m, "Add Exposure of a GDS file…", self.add_exposure_dialog, "Ctrl+I")
        self.a_add_column = act(m, "Add Column setting", lambda: tab.add_op(ColumnSetting()))
        self.a_add_beam = act(m, "Add Beam calibration", lambda: tab.add_op(tab._new_beamcal()))
        m.addSeparator()
        self.a_dup = act(m, "Duplicate", tab.duplicate, "Ctrl+D")
        self.a_del = act(m, "Delete", tab.delete)
        self.a_up = act(m, "Move up", lambda: tab.move(-1), "Ctrl+Up")
        self.a_down = act(m, "Move down", lambda: tab.move(+1), "Ctrl+Down")
        m.addSeparator()
        self.a_move = act(m, "Move the chip…", lambda: self.move_chip_dialog(), "Ctrl+M")
        self.field_actions = [self.a_add_exp, self.a_add_gds, self.a_add_column, self.a_add_beam,
                              self.a_dup,
                              self.a_del, self.a_up, self.a_down, self.a_move]

        m = mb.addMenu("&Calibration")
        self.a_run_cal = act(m, "Run the fixed steps only (column setting, beam calibration)",
                             self.run_calibration_steps, "Ctrl+K")
        self.a_fieldcal = act(m, "Field calibration…", lambda: self.field_calibration_dialog())
        self.a_sizes = act(m, "Field sizes…", lambda: self.field_sizes_dialog())
        m.addSeparator()
        self.a_forget_beam = act(m, "Forget the beam calibration of this field size", self.forget_beamcal)
        self.a_forget_cal = act(m, "Forget the field calibration of this field size", self.forget_calibration)

        m = mb.addMenu("&View")
        act(m, "Zoom to all", self.view.zoom_all, "Ctrl+0")
        act(m, "Zoom to the selected step", self.zoom_selected, "Ctrl+9")
        m.addSeparator()
        self.a_as_sem = m.addAction("Show as the SEM image (flip mirrored stage axes)")
        self.a_as_sem.setCheckable(True)
        self.a_as_sem.setChecked(True)
        self.a_as_sem.toggled.connect(lambda on: self._orient())
        m.addAction(self.side_dock.toggleViewAction())
        m.addAction(self.log_dock.toggleViewAction())

        m = mb.addMenu("St&age")
        act(m, "Move stage…", lambda: self.move_stage_dialog())
        act(m, "Go to the selected step's first field", self.goto_selected_field)
        act(m, "Add the stage position to Stage Positions…", self.tab_positions.add_current)
        m.addSeparator()
        act(m, "Stage settings…", self.edit_axes)

    def _make_status(self):
        sb = self.statusBar()
        self.l_global = QtWidgets.QLabel()
        self.l_fs = QtWidgets.QLabel()
        self.l_wd = QtWidgets.QLabel()
        self.l_beam = QtWidgets.QLabel()
        self.l_cal = QtWidgets.QLabel()
        self.l_field_tr = QtWidgets.QLabel()
        self.l_field = QtWidgets.QLabel()
        self.l_stage = QtWidgets.QLabel()
        self.l_run = QtWidgets.QLabel()
        self.l_run.setStyleSheet("color:#fd4")
        for w in (self.l_global, self.l_fs, self.l_wd, self.l_beam, self.l_cal, self.l_field_tr):
            sb.addWidget(w)
        for w in (self.l_run, self.l_field, self.l_stage, self.progress):
            sb.addPermanentWidget(w)
        for w in (self.l_global, self.l_fs, self.l_wd, self.l_beam, self.l_cal, self.l_field_tr, self.l_field,
                  self.l_stage):
            w.setContentsMargins(6, 0, 10, 0)

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(time.strftime("%H:%M:%S  ") + text)

    def _field(self) -> float:
        try:
            return float(self.hw.field_um())
        except Exception:  # noqa: BLE001
            return 100.0

    def _field_at_1x(self) -> float:
        try:
            return float(self.hw.field_at_1x_um())
        except Exception:  # noqa: BLE001
            return 100_000.0

    def _wd(self):
        try:
            return self.hw.working_distance_mm()
        except Exception:  # noqa: BLE001
            return None

    def _selected_chip(self) -> str:
        i = self.tab_schedule.selected()
        if 0 <= i < len(self.schedule) and isinstance(self.schedule[i], Exposure) and self.schedule[i].chip:
            return self.schedule[i].chip
        return self.tab_chips.selected() or (self.schedule.chips[0].name if self.schedule.chips else "")

    def _chip_status(self, name: str) -> str:
        g = self.chips.get(name)
        return "nominal (origin)" if g is None else g.describe(self.axes)

    def _update_status(self):
        name = self._selected_chip()
        g = self.chips.get(name)
        if not name:
            self.l_global.setText("Chip: none")
            self.l_global.setStyleSheet("color:#9aa")
        elif g is None:
            self.l_global.setText(f"Chip {name}: nominal (no Global Align)")
            self.l_global.setStyleSheet("color:#9aa")
        else:
            self.l_global.setText(f"Chip {name}: {g.describe(self.axes)}")
            self.l_global.setStyleSheet("color:#5d5")
        sel, sch = self.settings.field_size, self.schedule.field_size
        self.l_fs.setText(f"Field size {sel or 'none'} (Microscope)" + ("" if sel == sch else f", schedule {sch or '?'}"))
        self.l_fs.setStyleSheet("color:#5d5" if sel and sel == sch else "color:#fb4")
        wd = self._wd()
        cal_p = self.settings.calibration
        wd1 = f"WD1 {cal_p.wd_mm:g}" if cal_p.wd_mm is not None else "WD1 not recorded"
        self.l_wd.setText(("WD ?" if wd is None else f"WD {wd:.3f} mm") + f" ({wd1})")
        self.l_wd.setStyleSheet("color:#9aa" if cal_p.wd_mm is None else "")
        cond = self.condition()
        bc = self.beamcal()
        if bc is None:
            self.l_beam.setText(f"Beam calibration: none for {cond.describe()}")
            self.l_beam.setStyleSheet("color:#fb4" if self.settings.beamcals else "color:#9aa")
        else:
            self.l_beam.setText(f"Beam calibration: {bc.describe()}")
            self.l_beam.setStyleSheet("color:#5d5")
        cal = self.calibration
        if cal is None:
            self.l_cal.setText(f"Field calibration: none for {sel or 'no field size'}")
            self.l_cal.setStyleSheet("color:#9aa")
        elif cal.valid_for(cond):
            self.l_cal.setText(f"Field calibration: {cal.describe(self.axes)}")
            self.l_cal.setStyleSheet("color:#5d5")
        else:
            self.l_cal.setText(f"Field calibration: for {cal.condition.describe()}, not {cond.describe()}")
            self.l_cal.setStyleSheet("color:#fb4")
        self.l_field_tr.setText("Last field: " + ("–" if self.last_map is None else self.last_map.describe()))
        self.l_field.setText(f"field {self._field():.4g} µm")

    def _tick(self):
        try:
            pos = self.hw.stage_position()
        except Exception:  # noqa: BLE001
            pos = None
        self.l_stage.setText("stage ?" if pos is None else f"stage ({pos[0]:.1f}, {pos[1]:.1f}) µm")
        self.view.set_stage(pos)
        binv = np.linalg.inv(self.stage_to_beam())
        field = self._field()
        self.view.set_fov(pos, field, binv)
        try:
            area = self.hw.scan_area()
        except Exception:  # noqa: BLE001
            area = None
        self.view.set_scan(pos, area, field, binv)
        self._update_status()

    def condition(self) -> Condition:
        """The column condition now (magnification, the beam preset this program recalled last, WD,
        the field size selected in the Microscope window)."""
        try:
            mag = self.hw.magnification()
        except Exception:  # noqa: BLE001
            mag = None
        return Condition(mag, getattr(self.hw, "last_preset", None), self._wd(), self.settings.field_size or None)

    def _stored_beamcal(self):
        """The beam calibration stored for the schedule's field size (None: none, or no field size)."""
        fs = field_size_of(self.schedule.field_size, self.settings.field_sizes)
        if fs is None:
            return None
        return find_beamcal(self.settings.beamcals, Condition(float(fs.magnification), None, None, fs.name))

    def beamcal(self, condition: Condition | None = None):
        return find_beamcal(self.settings.beamcals, condition or self.condition())

    @property
    def calibration(self):
        """The field calibration of the field size selected in the Microscope window (None: none)."""
        return self.settings.fieldcal()

    def stage_to_beam(self) -> np.ndarray:
        cond = self.condition()
        cal = self.calibration
        if cal is not None and cal.valid_for(cond):
            return stage_rotation(cal.stage_to_beam, self.axes)
        return self.axes.matrix

    def _title(self):
        name = os.path.basename(self.schedule.path) if self.schedule.path else (self.schedule.name or "untitled")
        self.setWindowTitle(f"easyEBL – Lithography – {name}{' *' if self.modified else ''}")

    # ================================================================ model
    def _current_for(self, op) -> float | None:
        cur = None
        for o in self.schedule.operations:
            if isinstance(o, ColumnSetting) and o.enabled and o.current_pa:
                cur = o.current_pa
            if o is op:
                break
        return cur

    def _plan_of(self, op):
        if self.plans is None:
            return None
        i = next((k for k, o in enumerate(self.schedule.operations) if o is op), -1)
        return self.plans[i] if 0 <= i < len(self.plans) else None

    def _fields_info(self, op) -> str:
        p = self._plan_of(op)
        if p is None:
            return ""
        if not p.fields:
            return "no fields"
        try:
            flds = exposure_fields(op, self.library)
            asg = assign_markers(flds, op, p.field_um)
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        parts = [f"{len(flds)} field(s) of {p.field_um:.4g} µm"]
        if op.global_align:
            parts.append("Global Align in " + (flds[asg.global_field].name if asg.global_field is not None
                                                else "NO field (not all markers in one field)"))
        if op.local_align:
            parts.append(f"{len(asg.local)} field(s) with a Local marker")
        return " · ".join(parts)

    def _field_centres(self, op) -> list:
        try:
            return [f.centre_ccs for f in exposure_fields(op, self.library)]
        except Exception:  # noqa: BLE001
            return []

    def _new_exposure(self) -> Exposure:
        name = self._selected_chip()
        if not name:
            name = self.tab_chips.add_chip()
        return Exposure(chip=name)

    def _schedule_edited(self):
        self.modified = True
        self._title()
        self._replan_soon()

    def _chips_edited(self):
        self.modified = True
        self._title()
        self.tab_schedule.refresh()
        self._replan_soon()

    def _replan_soon(self):
        self._replan_timer.start()

    def replan(self):
        self.plans = self._plan()
        self.tab_schedule.set_plans(self.plans)
        self.tab_schedule.editor._stored_text()  # a beam calibration stored since
        sel = self.tab_schedule.selected()
        self.view.set_schedule(self.schedule, self.library, self.plans, self.axes, sel if sel >= 0 else None,
                               self.chips)
        self.tab_chips.refresh()
        self._update_status()
        return self.plans

    def _plan(self, start: int = 0) -> list:
        try:
            presets = self.hw.preset_names()
        except Exception:  # noqa: BLE001
            presets = None
        return plan(self.schedule, self._field_at_1x(), self.axes, presets, self.library, self.chips,
                    calibration=self.settings.fieldcal(self.schedule.field_size),
                    preset=getattr(self.hw, "last_preset", None), positions=self.settings.positions(),
                    beamcals=self.settings.beamcals, start=start, field_sizes=self.settings.field_sizes,
                    selected=self.settings.field_size)

    def warnings(self) -> list:
        return [(p.index, w) for p in (self.plans or self.replan()) for w in p.warnings]

    def validate(self, start: int = 0):
        self.replan()
        ws = self.warnings()
        if start >= len(FIXED_STEPS):
            ws = [(p.index, w) for p in self._plan(start) if p.index >= start for w in p.warnings]
        if not ws:
            self.log(f"validate: {len(self.schedule)} steps, no warnings")
        for i, w in ws:
            self.log(f"validate: step {i + 1}: {w}")
        return ws

    def _positions_changed(self):
        self.view.set_stage_positions(self.settings.stage_positions)
        self.tab_chips.refresh()
        self._replan_soon()

    def set_schedule(self, schedule: Schedule, modified: bool = False):
        self.schedule = schedule
        self.chips = {k: v for k, v in self.chips.items() if schedule.chip(k) is not None}
        self.modified = modified
        self.view._fitted = False
        self.tab_chips.set_schedule(schedule)
        self.tab_schedule.set_schedule(schedule)
        self._title()
        self.replan()

    # ================================================================ files
    def _confirm_discard(self) -> bool:
        if not self.modified or not self.ask_on_close:
            return True
        r = QtWidgets.QMessageBox.question(self, "Lithography", "Discard the unsaved changes of the schedule?")
        return r == QtWidgets.QMessageBox.Yes

    def _dir(self) -> str:
        return self.settings.last_dir or os.getcwd()

    def _remember_dir(self, path: str):
        self.settings.last_dir = os.path.dirname(os.path.abspath(path))
        self._save_settings()

    def new_schedule(self):
        if self._running() or not self._confirm_discard():
            return
        self.set_schedule(Schedule())

    def open_dialog(self):
        if self._running() or not self._confirm_discard():
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open schedule", self._dir(), SCHEDULE_FILTER)
        if p:
            self.open_schedule(p)

    def open_schedule(self, path: str) -> bool:
        try:
            s = Schedule.load(path)
        except (OSError, ScheduleError) as e:
            self.log(f"ERROR opening {path}: {e}")
            if self.ask_on_close:
                QtWidgets.QMessageBox.warning(self, "Lithography", f"Cannot open {path}:\n{e}")
            return False
        self.library.clear()
        self.chips = {}
        self.set_schedule(s)
        self._remember_dir(path)
        self.log(f"opened {path} ({len(s)} steps, {len(s.chips)} chip(s))")
        return True

    def save(self) -> bool:
        if not self.schedule.path:
            return self.save_as()
        return self.save_to(self.schedule.path)

    def save_as(self) -> bool:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save schedule", os.path.join(self._dir(), (self.schedule.name or "schedule") + ".ebls.json"),
            SCHEDULE_FILTER)
        return self.save_to(p) if p else False

    def save_to(self, path: str) -> bool:
        try:
            self.schedule.save(path)
        except (OSError, ScheduleError) as e:
            self.log(f"ERROR saving {path}: {e}")
            return False
        self.modified = False
        self._title()
        self._remember_dir(path)
        self.log(f"saved {path}")
        return True

    def add_exposure_dialog(self):
        if self._running():
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Exposure of a GDS file", self._dir(), GDS_FILTER)
        if p:
            self.add_exposure_from_gds(p)

    def add_exposure_from_gds(self, path: str) -> Exposure | None:
        try:
            self.library.top_cells(path)
        except Exception as e:  # noqa: BLE001
            self.log(f"ERROR reading {path}: {e}")
            return None
        op = self._new_exposure()
        op.path = os.path.abspath(path)
        op.label = os.path.basename(path)
        self.tab_schedule.add_op(op)
        self._remember_dir(path)
        self.replan()
        p = self._plan_of(op)
        self.log(f"Exposure of {os.path.basename(path)} on {op.chip}: {len(p.fields) if p else 0} field(s)")
        return op

    def pattern_for_selected(self):
        i = self.tab_schedule.selected()
        if i < 0 or not isinstance(self.schedule[i], Exposure):
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pattern GDS", self._dir(), GDS_FILTER)
        if p:
            self.set_pattern_file(i, p)

    def set_pattern_file(self, i: int, path: str) -> None:
        try:
            self.library.top_cells(path)
        except Exception as e:  # noqa: BLE001
            self.log(f"ERROR reading {path}: {e}")
            return
        op = self.schedule[i]
        op.path = os.path.abspath(path)
        op.top = ""
        self._remember_dir(path)
        self.tab_schedule.refresh(select=i)
        self._schedule_edited()

    # ================================================================ chips / view
    def _chip_fields_scs(self, name: str) -> list:
        """(exposure, field centre SCS (nominal), half field) of every field on chip *name*."""
        plans = self.plans or self.replan()
        out = []
        for i, op in enumerate(self.schedule.operations):
            if isinstance(op, Exposure) and op.chip == name and i < len(plans):
                for fp in plans[i].fields:
                    out.append((op, fp.target, plans[i].field_um / 2.0))
        return out

    def anchors(self, name: str) -> dict:
        """Reference points for Move the chip (stage µm, through the chip's present map; None where none)."""
        chip = self.schedule.chip(name)
        if chip is None:
            return {}
        g = self.chips.get(name) or chip.nominal(self.axes)
        o = tuple(float(v) for v in g.to_scs(np.zeros(2)))
        flds = self._chip_fields_scs(name)
        first = tuple(flds[0][1]) if flds else None
        box = None
        if flds:
            xs = [f[1][0] for f in flds]
            ys = [f[1][1] for f in flds]
            h = max(f[2] for f in flds)
            box = ((min(xs) - h + max(xs) + h) / 2.0, (min(ys) - h + max(ys) + h) / 2.0)
        e = next((op for op in self.schedule.exposures if op.chip == name and op.global_align and op.global_markers),
                 None)
        mark = tuple(float(v) for v in g.to_scs(np.asarray(e.global_markers[0], float))) if e else None
        return {"origin": ("the chip origin (CCS 0, 0)", o),
                "first": ("the first field centre", first),
                "bbox": ("the centre of the bounding box of the chip's fields", box),
                "marker": ("Global Align marker 1", mark)}

    def move_chip_dialog(self, name: str | None = None, click=None) -> bool:
        if self._running():
            return False
        name = name or self._selected_chip()
        if not name or self.schedule.chip(name) is None:
            self.log("no chip to move (Chips tab)")
            return False
        d = MoveFieldsDialog(self.anchors(name), self.hw.stage_position(), click, self)
        d.setWindowTitle(f"Move chip {name}")
        self._move_dialog = d
        if d.exec() != QtWidgets.QDialog.Accepted or d.value() is None:
            return False
        self.move_chip(name, *d.value())
        return True

    # ---------------------------------------------------------------- move stage
    def _on_chip(self, chip, ccs) -> tuple:
        """(z, wd) on *chip* at CCS *ccs*: Z_cal and its WD2 there (None: not known)."""
        wd = chip.height.wd_at(ccs)
        return (self.settings.calibration.z_um if wd is not None else None), wd

    def stage_targets(self, click=None) -> tuple:
        """Targets for Move stage…: (click, {chip: markers}, positions), each ``(label, (x, y), z, wd)``:
        on a chip Z_cal and its WD2 there, a stage position Z_cal + ΔZ and WD1 (None: not known, kept)."""
        click_t = None
        if click is not None:
            cx, cy = float(click[0]), float(click[1])
            label, z, wd = "the clicked position", None, None
            for chip in self.schedule.chips:  # on a chip: in a field of its Exposures
                fields = self._chip_fields_scs(chip.name)
                if any(abs(cx - c[0]) <= half and abs(cy - c[1]) <= half for _, c, half in fields):
                    g = self.chips.get(chip.name) or chip.nominal(self.axes)
                    z, wd = self._on_chip(chip, g.to_ccs(np.array([cx, cy])))
                    label = f"the clicked position (chip {chip.name})"
                    break
            click_t = (label, (cx, cy), z, wd)
        markers = {}
        for chip in self.schedule.chips:
            g = self.chips.get(chip.name) or chip.nominal(self.axes)
            out, seen = [], set()
            for i, op in enumerate(self.schedule.operations):
                if not isinstance(op, Exposure) or op.chip != chip.name:
                    continue
                for kind, pts, on in (("Global", op.global_markers, op.global_align),
                                      ("Local", op.local_markers, op.local_align)):
                    for j, m in enumerate(pts if on else []):
                        key = (round(float(m[0]), 4), round(float(m[1]), 4))
                        if key in seen:
                            continue
                        seen.add(key)
                        scs = g.to_scs(np.asarray(m, float))
                        out.append((f"step {i + 1} {kind} {j + 1} (CCS {m[0]:.1f}, {m[1]:.1f})",
                                    (float(scs[0]), float(scs[1])), *self._on_chip(chip, m)))
            markers[chip.name] = out
        wd1 = self.settings.calibration.wd_mm
        positions = [(p.name, (p.x, p.y), z, wd1 if z is not None else None)
                     for p in self.settings.stage_positions for z in (self.settings.z_of(p),)]
        return click_t, markers, positions

    def move_stage_dialog(self, click=None, exec_: bool = True):
        """Move stage…: to the clicked position, a marker of a chip or a stage position (in focus)."""
        if self._running():
            return None
        c, markers, positions = self.stage_targets(click)
        d = MoveStageDialog(c, markers, positions, chip=self._selected_chip(), parent=self)
        self._move_stage_dialog = d
        if not exec_:
            return d
        if d.exec() == QtWidgets.QDialog.Accepted and d.value() is not None:
            self.goto_stage(*d.value())
        return d

    def move_chip(self, name: str, dx: float, dy: float) -> None:
        self.schedule.shift_chip(name, dx, dy)
        g = self.chips.get(name)
        if g is not None:  # a measured map moves along
            g.offset = g.offset + np.array([dx, dy], float)
        self.log(f"moved chip {name} by ({dx:+.3f}, {dy:+.3f}) µm")
        self.tab_chips.refresh(select=name)
        self.modified = True
        self._title()
        self.replan()

    def height_dialog(self, name: str | None = None, exec_: bool = True):
        """The chip's height (Chips → Height…): the focus kept, one WD2, or a map of WD2x recorded at Z_cal."""
        if self._running():
            return None
        name = name or self._selected_chip()
        chip = self.schedule.chip(name)
        if chip is None:
            self.log("no chip (Chips tab)")
            return None
        g = self.chips.get(name) or chip.nominal(self.axes)
        centres = [f for op in self.schedule.exposures if op.chip == name for f in self._field_centres(op)]
        d = HeightDialog(chip, self.hw.stage_position, self.hw.stage_z, self._wd,
                         to_ccs=lambda scs: g.to_ccs(np.asarray(scs, float)),
                         to_scs=lambda ccs: g.to_scs(np.asarray(ccs, float)),
                         goto=self.goto_stage, fields_ccs=centres, z_cal=self.settings.calibration.z_um,
                         nominal_wd=getattr(field_size_of(self.schedule.field_size, self.settings.field_sizes),
                                            "wd_mm", None), wd1=self.settings.calibration.wd_mm, parent=self)
        self._height_dialog = d
        if not exec_:
            return d
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.set_chip_height(name, d.value())
        return d

    def set_chip_height(self, name: str, height) -> None:
        chip = self.schedule.chip(name)
        chip.height = height
        self.log(f"chip {name}: height {height.describe()}")
        self.tab_chips.refresh(select=name)
        self.modified = True
        self._title()
        self._replan_soon()

    # ---------------------------------------------------------------- marker picking
    def pick_markers(self, kind: str, index: int = -1) -> bool:
        """Pick markers of the selected Exposure on the plot: *kind* "global" (marker *index*),
        "global_all" (1 … n in turn) or "local" (each click adds one; Esc: done).  A click snaps to the
        nearest corner or centre of any polygon of its GDS (every layer); Shift-click takes the point."""
        if self._running():
            return False
        i = self.tab_schedule.selected()
        op = self.schedule[i] if 0 <= i < len(self.schedule) else None
        if not isinstance(op, Exposure) or self.schedule.chip(op.chip) is None:
            self.log("pick markers: select an Exposure on a chip first")
            return False
        chip = self.schedule.chip(op.chip)
        g = self.chips.get(chip.name) or chip.nominal(self.axes)
        ccs, kinds = snap_targets(op, self.library)
        scs = g.to_scs(ccs) if len(ccs) else np.zeros((0, 2))
        n = len(op.global_markers)
        self._picking = {"op": op, "kind": kind, "k": 0 if kind == "global_all" else int(index), "n": n, "added": 0}
        self.view.start_pick(scs, ccs, kinds, lambda p: g.to_ccs(np.asarray(p, float)), self._pick_text())
        if not len(ccs):
            self.log("pick markers: the Exposure has no GDS polygons to snap to – Shift-click a point")
        return True

    def _pick_text(self) -> str:
        p = self._picking
        if p["kind"] == "local":
            return f"Local markers: click a polygon corner or centre to add one ({p['added']} added)"
        return f"Global marker {p['k'] + 1} of {p['n']}: click a polygon corner or centre"

    def _marker_picked(self, x: float, y: float) -> None:
        p = self._picking
        if p is None:
            return
        op = p["op"]
        pt = (round(float(x), 6), round(float(y), 6))
        if p["kind"] == "local":
            op.local_markers = list(op.local_markers) + [pt]
            p["added"] += 1
            self.log(f"Local marker {len(op.local_markers)} at CCS ({pt[0]:.3f}, {pt[1]:.3f}) µm")
        else:
            mk = list(op.global_markers)
            mk[p["k"]] = pt
            op.global_markers = mk
            self.log(f"Global marker {p['k'] + 1} at CCS ({pt[0]:.3f}, {pt[1]:.3f}) µm")
            p["k"] += 1
        ed = self.tab_schedule.editor
        if ed.op is op:
            ed.set_op(op)  # the new values in the form
        self.tab_schedule.refresh(select=self.tab_schedule.selected())
        self._schedule_edited()
        if p["kind"] == "global" or (p["kind"] == "global_all" and p["k"] >= p["n"]):
            self.stop_pick()
        else:
            self.view.set_pick_text(self._pick_text())

    def stop_pick(self) -> None:
        if self._picking is not None and self._picking["kind"] == "local":
            self.log(f"Local markers: {self._picking['added']} added")
        self._picking = None
        self.view.stop_pick()

    def _context_menu(self, x: float, y: float, screen_pos):
        m = QtWidgets.QMenu(self)
        a = m.addAction("Move stage…")
        a.triggered.connect(lambda: self.move_stage_dialog(click=(x, y)))
        a.setEnabled(not self._running())
        a = m.addAction("Move the chip…")
        a.triggered.connect(lambda: self.move_chip_dialog(click=(x, y)))
        a.setEnabled(not self._running() and bool(self.schedule.chips))
        m.addSeparator()
        m.addAction("Zoom to all").triggered.connect(self.view.zoom_all)
        self._last_menu = m
        m.popup(screen_pos)

    def zoom_selected(self):
        i = self.tab_schedule.selected()
        plans = self.plans or self.replan()
        if 0 <= i < len(plans) and plans[i].fields:
            t = np.array([f.target for f in plans[i].fields], float)
            h = plans[i].field_um * 0.6
            self.view.getPlotItem().getViewBox().setRange(xRange=(t[:, 0].min() - h, t[:, 0].max() + h),
                                                          yRange=(t[:, 1].min() - h, t[:, 1].max() + h))

    def _orient(self):
        self.view.set_orientation(self.axes, self.a_as_sem.isChecked())

    def goto_selected_field(self):
        i = self.tab_schedule.selected()
        plans = self.plans or self.replan()
        if 0 <= i < len(plans) and plans[i].target is not None:
            wd = plans[i].fields[0].wd
            self.goto_stage(*plans[i].target, self.settings.calibration.z_um if wd is not None else None, wd)

    def _set_running_ui(self, on: bool):
        if on and self._picking is not None:
            self.stop_pick()
        self.tab_schedule.set_running(on)
        self.tab_chips.setEnabled(not on)
        for a in [self.a_new, self.a_open, self.a_reset, self.a_run, self.a_run_from, self.a_run_cal,
                  self.a_forget_cal, self.a_forget_beam, self.a_fieldcal, self.a_sizes] + self.field_actions:
            a.setEnabled(not on)
        self.a_pause.setEnabled(on)
        self.a_abort.setEnabled(on)

    # ================================================================ alignment
    def edit_axes(self):
        d = StageAxesDialog(self.axes, self, self.settings.stage_tolerance_um)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.settings.stage_tolerance_um = d.tolerance()
            self._save_settings()

    def _save_settings(self):
        try:
            self.settings.save()
        except OSError as e:
            self.log(f"litho settings not saved: {e}")

    def reset_chip(self, name: str):
        if self._running():
            return
        if self.chips.pop(name, None) is not None:
            self.log(f"chip {name}: alignment reset (nominal origin)")
        self._replan_soon()

    def reset_all_chips(self):
        if self._running():
            return
        self.chips = {}
        self.last_map = None
        self.log("alignment of all chips reset (nominal origins)")
        self._replan_soon()

    # ================================================================ field calibration
    def forget_calibration(self):
        if self._running():
            return
        name = self.settings.field_size
        if self.settings.fieldcals.pop(name, None) is not None:
            self._save_settings()
            self.log(f"field calibration of field size {name} forgotten (nominal stage ↔ beam rotation)")
        self._update_status()
        self._replan_soon()

    def field_sizes_dialog(self, exec_: bool = True):
        """Calibration → Field sizes…: edit the list (shared with the Microscope window)."""
        if self._running():
            return None
        d = FieldSizesDialog(self.settings.field_sizes, self._field_at_1x(), self)
        self._sizes_dialog = d
        if not exec_:
            return d
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.set_field_sizes(d.value())
        return d

    def set_field_sizes(self, sizes) -> None:
        self.settings.set_field_sizes(sizes)
        self._save_settings()
        self.log("field sizes: " + "; ".join(s.describe() for s in sizes))
        PositionsHub.instance().changed.emit(self.settings)  # the Microscope window's selector follows
        self.tab_schedule.refresh(select=self.tab_schedule.selected())
        self._replan_soon()

    def field_calibration_dialog(self, exec_: bool = True):
        """Calibration → Field calibration…: set it up, then measure (the field size selected now)."""
        if self._running():
            return None
        cur = self.calibration
        d = FieldCalSetupDialog(self.settings.fieldcal_config, self.settings.positions(), self.settings.field_size,
                                cur.describe(self.axes) if cur is not None else None, self)
        self._fieldcal_dialog = d
        if not exec_:
            return d
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.settings.fieldcal_config = d.value()
            self._save_settings()
            self.run_field_calibration()
        return d

    def run_field_calibration(self) -> bool:
        """Measure the field calibration of the field size selected now (on the runner thread)."""
        if self._running():
            return False
        name = self.settings.field_size
        if not name:
            self.log("field calibration: no field size selected (Microscope window)")
            return False
        s = Schedule([ColumnSetting(field_size=name)])
        self.bridge.cancelled.clear()
        self._cal_in_schedule = False
        self.runner = ScheduleRunner(s, self.hw, self.axes, self.library,
                                     on_event=lambda kind, data: self.sig.event.emit(kind, data),
                                     positions=self.settings.positions(),
                                     stage_tolerance_um=self.settings.stage_tolerance_um,
                                     review=self._review_calibration, beamcals=self.settings.beamcals,
                                     field_sizes=self.settings.field_sizes)
        self._set_running_ui(True)
        self.l_run.setText("field calibration")
        self.log(f"field calibration of field size {name}")
        self._thread = threading.Thread(target=self.runner.run_field_calibration,
                                        args=(self.settings.fieldcal_config,), name="litho-fieldcal", daemon=True)
        self._thread.start()
        return True

    # ================================================================ beam calibration
    def forget_beamcal(self):
        if self._running():
            return
        cond = self.condition()
        if cond.field_size is None:
            fs = field_size_of(self.schedule.field_size, self.settings.field_sizes)
            cond = Condition(fs.magnification if fs else cond.magnification, None, None, fs.name if fs else None)
        if self.settings.forget_beamcal(cond):
            self._save_settings()
            self.log(f"beam calibration for {cond.describe()} forgotten")
        self._update_status()
        self._replan_soon()

    def _make_runner(self) -> ScheduleRunner:
        return ScheduleRunner(self.schedule, self.hw, self.axes, self.library, self.chips,
                              on_event=lambda kind, data: self.sig.event.emit(kind, data),
                              calibration=self.settings.fieldcal(self.schedule.field_size),
                              positions=self.settings.positions(),
                              stage_tolerance_um=self.settings.stage_tolerance_um,
                              review=self._review_calibration, beamcals=self.settings.beamcals,
                              review_beam=self._review_beamcal, field_sizes=self.settings.field_sizes,
                              selected=self.settings.field_size)

    # ---------------------------------------------------------------- calibration windows
    def _calwin(self) -> FieldCalWindow:
        if self.fieldcal_window is None:
            w = FieldCalWindow(self)
            w.decided.connect(self._cal_decided)
            w.abortRequested.connect(self.abort)
            self.fieldcal_window = w
        return self.fieldcal_window

    def _beamwin(self) -> BeamCalWindow:
        if self.beamcal_window is None:
            w = BeamCalWindow(self)
            w.decided.connect(self._beam_decided)
            w.abortRequested.connect(self.abort)
            self.beamcal_window = w
        return self.beamcal_window

    def _review_calibration(self, res) -> bool:
        """(runner thread) Show *res* in the calibration window and wait for Accept / Discard."""
        def show(fut):
            w = self._calwin()
            w.show_result(res, self.axes, ask=True)  # a refused one too: Use anyway overrides
            self._cal_future = fut

        return bool(self.bridge.call(show, on_cancel=self._cal_cancelled))

    def _review_beamcal(self, res) -> bool:
        def show(fut):
            w = self._beamwin()
            w.show_result(res, ask=True)  # a refused one too: Use anyway overrides
            self._beam_future = fut

        return bool(self.bridge.call(show, on_cancel=self._beam_cancelled))

    def _cal_decided(self, ok: bool):
        fut, self._cal_future = self._cal_future, None
        if fut is not None and not fut.done():
            fut.set_result(bool(ok))

    def _beam_decided(self, ok: bool):
        fut, self._beam_future = self._beam_future, None
        if fut is not None and not fut.done():
            fut.set_result(bool(ok))

    def _cal_cancelled(self):
        self._cal_future = None
        if self.fieldcal_window is not None:
            self.fieldcal_window.cancel()

    def _beam_cancelled(self):
        self._beam_future = None
        if self.beamcal_window is not None:
            self.beamcal_window.cancel()

    # ================================================================ stage
    def goto_stage(self, x: float, y: float, z=None, wd=None):
        """Move the stage to (x, y) and Z *z* (None: Z stays), then the focus to *wd* (None: stays)."""
        if self._running():
            return
        where = (f"({x:.2f}, {y:.2f}" + (f", Z {z:.2f}" if z is not None else "") + ") µm"
                 + (f", WD {wd:.4f} mm" if wd is not None else ""))
        self.log(f"stage → {where}")

        def job():
            try:
                if z is None:
                    self.hw.move_stage(x, y)
                else:
                    self.hw.move_stage(x, y, z)
                if wd is not None:
                    self.hw.set_working_distance(float(wd))
                self.sig.event.emit("log", {"text": f"stage at {where}"})
            except Exception as e:  # noqa: BLE001
                self.sig.event.emit("log", {"text": f"ERROR stage move: {type(e).__name__}: {e}"})

        threading.Thread(target=job, name="litho-goto", daemon=True).start()

    # ================================================================ run
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_problems(self, start: int = 0) -> list:
        """Why the schedule cannot run (from step *start*): the field size (the Microscope window's
        must be the schedule's), the calibration sample (Z_cal, WD1), the chip heights (recorded at
        Z_cal); after the fixed steps also the column (the field size's magnification, the stage at
        Z_cal, a beam calibration).  Empty: it can."""
        out = []
        fs = field_size_of(self.schedule.field_size, self.settings.field_sizes)
        sel = self.settings.field_size
        if fs is None:
            out.append(f"no field size named {self.schedule.field_size!r}" if self.schedule.field_size
                       else "the schedule has no field size (its first Column setting)")
        elif sel != fs.name:
            out.append(f"the Microscope window is at field size {sel or 'none'}, the schedule at {fs.name}: select "
                       f"{fs.name} there")
        cal = self.settings.calibration
        w = calibration_problem(cal)
        if w:
            out.append(w)
        used = {op.chip for i, op in enumerate(self.schedule.operations)
                if isinstance(op, Exposure) and op.enabled and i >= start}
        for chip in self.schedule.chips:
            if chip.name in used:
                w = chip.height.problem(cal.z_um, check_z=True)
                if w:
                    out.append(f"chip {chip.name}: {w}")
        if start >= len(FIXED_STEPS) and fs is not None:
            try:
                mag = self.hw.magnification()
            except Exception:  # noqa: BLE001
                mag = None
            if mag is None or int(round(mag)) != fs.magnification:
                out.append(f"the column is not at the field size's ×{fs.magnification:g}: run the schedule from the "
                           "start")
            z = self.hw.stage_z()
            if cal.z_um is not None and (z is None or abs(z - cal.z_um) > max(self.settings.stage_tolerance_um,
                                                                               Z_MATCH_UM)):
                out.append("the stage is not at the calibration sample's Z: run the schedule from the start")
            cond = Condition(fs.magnification, getattr(self.hw, "last_preset", None), None, fs.name)
            if self.beamcal(cond) is None:
                out.append(f"no beam calibration for {cond.describe()}: run the schedule (or its fixed steps) first")
        return out

    def run_from_problem(self, start: int) -> str | None:
        """The first of :meth:`run_problems` (None: it can run)."""
        ps = self.run_problems(start)
        return ps[0] if ps else None

    def run_calibration_steps(self) -> bool:
        """Only the fixed steps: the column (field size, WD1) and the beam calibration."""
        return self.run(0, stop=len(FIXED_STEPS))

    def run(self, start: int = 0, confirm: bool | None = None, stop: int | None = None) -> bool:
        if self._running():
            return False
        if start < len(FIXED_STEPS):
            start = 0
        why = self.run_problems(start)
        if why:
            for w in why:
                self.log(f"cannot run from step {start + 1}: {w}")
            if self.ask_on_close if confirm is None else confirm:
                QtWidgets.QMessageBox.warning(self, "Run schedule", f"Cannot run from step {start + 1}:\n\n"
                                              + "\n".join(why))
            return False
        ws = [(i, w) for i, w in self.validate(start) if i >= start and (stop is None or i < stop)]
        if ws and (self.ask_on_close if confirm is None else confirm):
            text = "\n".join(f"step {i + 1}: {w}" for i, w in ws[:15]) + ("\n…" if len(ws) > 15 else "")
            if QtWidgets.QMessageBox.question(self, "Run schedule", f"{len(ws)} warning(s):\n\n{text}\n\nRun anyway?") \
                    != QtWidgets.QMessageBox.Yes:
                return False
        self.bridge.cancelled.clear()
        self._cal_in_schedule = True
        self.tab_schedule.reset_states()
        self.runner = self._make_runner()
        self._set_running_ui(True)
        self.progress.setValue(0)
        self.l_run.setText("running")
        self.log(f"run from step {start + 1}" + (f" to {stop}" if stop is not None else ""))
        self._thread = threading.Thread(target=self.runner.run, args=(start, stop), name="litho-run", daemon=True)
        self._thread.start()
        return True

    def pause(self, on: bool):
        if self.runner is not None:
            self.runner.pause(on)
            self.l_run.setText("paused (after this step)" if on else "running")

    def abort(self):
        if self.runner is not None:
            self.log("abort requested")
            self.bridge.cancelled.set()  # releases a wait for the user (e.g. Accept / Discard)
            self.runner.abort()

    def wait(self, timeout: float = 10.0) -> bool:
        """Wait for the run thread (tests), processing GUI events."""
        t0 = time.perf_counter()
        app = QtWidgets.QApplication.instance()
        while self._running() and time.perf_counter() - t0 < timeout:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        return not self._running()

    def _on_event(self, kind: str, data):
        if kind == "log":
            self.log(data["text"])
        elif kind == "status":
            self.l_run.setText(data["text"])
        elif kind == "step":
            self.tab_schedule.set_state(data["index"], data["state"], data.get("message", ""))
            if data["state"] == "running":
                self.tab_schedule.select(data["index"])
        elif kind == "field":
            msg = f"field {data['k'] + 1}/{data['n']} {data['name']}: {data['state']}"
            self.tab_schedule.set_state(data["index"], "running", msg)
            self.l_run.setText(msg)
            if data.get("map") is not None:
                self.last_map = data["map"]
        elif kind == "alignment":
            self.chips[data["chip"]] = data["transform"]
            self._update_status()
            self._replan_soon()
        elif kind == "calibration_start":
            w = self._calwin()
            w.start(data["n"], data["position"], data["condition"], data["field_um"], self.axes,
                    in_schedule=self._cal_in_schedule)
            w.show()
            w.raise_()
        elif kind == "calibration_image":
            if self.fieldcal_window is not None:
                self.fieldcal_window.add_image(data["k"], data["n"], data["shift"], data["image"], data["reference"])
        elif kind == "calibration":
            res = data["result"]
            name = res.condition.field_size or self.settings.field_size
            if name:
                self.settings.fieldcals[name] = res
                self._save_settings()
            self.log(f"field calibration of field size {name}: {res.describe(self.axes)} (saved)")
            self._update_status()
            self._replan_soon()
        elif kind == "beamcal_start":
            w = self._beamwin()
            w.start(data["position"], data["condition"], data["field_um"], data["pixels"])
            w.show()
            w.raise_()
        elif kind == "beamcal_image":
            if self.beamcal_window is not None:
                self.beamcal_window.set_image(data["image"], data["pixel_um"])
        elif kind == "beamcal":
            self.settings.set_beamcal(data["result"])
            self._save_settings()
            self.log(f"beam calibration: {data['result'].describe()} (saved for {data['result'].condition.describe()})")
            self._update_status()
            self._replan_soon()
        elif kind == "focus_request":
            self.l_run.setText(f"focus and stigmate by hand at ×{data['magnification']:,}")
        elif kind == "wd1":
            # focused again by hand on the calibration sample: WD1 from now on (Z_cal unchanged)
            self.settings.record_wd(float(data["wd_mm"]), data.get("z_um"))
            self._save_settings()
            PositionsHub.instance().changed.emit(self.settings)  # every Stage Positions tab
            self._update_status()
        elif kind == "exposed":
            self.progress.setValue(1000)
        elif kind == "finished":
            for w in (self.fieldcal_window, self.beamcal_window):
                if w is None:
                    continue
                if w.isVisible() and w.result is None and data["state"] != "done":
                    w.failed(data.get("message", data["state"]))
                else:
                    w.cancel()
            self._set_running_ui(False)
            self.l_run.setText(data["state"])
            self.runner = None
            self._update_status()
            self._replan_soon()

    # ================================================================ close
    def closeEvent(self, ev):  # noqa: N802
        if self._running():
            if self.ask_on_close and QtWidgets.QMessageBox.question(
                    self, "Lithography", "A schedule is running. Abort it and close?") != QtWidgets.QMessageBox.Yes:
                ev.ignore()
                return
            self.abort()
            self._thread.join(10)
        elif not self._confirm_discard():
            ev.ignore()
            return
        self._timer.stop()
        close = getattr(self.hw, "close", None)
        if close is not None:
            close()
        super().closeEvent(ev)
