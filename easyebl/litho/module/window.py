"""The Lithography window: stage view (SCS with each field's WCS), Stage Positions and Exposure
Schedule tabs, alignment status.

Opened from the easyEBL menu (Open → Lithography).  The window works through a
:class:`~easyebl.litho.runner.LithoHardware`: :class:`~.hardware.BackendHardware` on the
easyEBL backend, or :class:`~easyebl.litho.runner.SimulatedHardware` (tests, demos).
"""
from __future__ import annotations

import os
import threading
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..coords import FieldTransform
from ..runner import LithoHardware, ScheduleRunner
from ..schedule import (ColumnSetting, ExposeField, FieldCorrection, PatternLibrary, PatternSpec, Schedule,
                        ScheduleError, plan)
from ..settings import LithoSettings
from .dialogs import ImportDialog, MoveFieldsDialog, StageAxesDialog
from .hardware import GuiBridge
from .tabs import ScheduleTab, StagePositionsTab
from .view import StageView

__all__ = ["LithoWindow"]

SCHEDULE_FILTER = "easyEBL schedule (*.ebls.json *.json);;All files (*)"
GDS_FILTER = "GDSII (*.gds *.gds2 *.gdsii);;All files (*)"


class _RunSignals(QtCore.QObject):
    event = QtCore.Signal(str, object)
    progress = QtCore.Signal(float)


class LithoWindow(QtWidgets.QMainWindow):
    def __init__(self, hardware=None, settings: LithoSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings or LithoSettings.load()
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
        self.library = PatternLibrary()
        self.schedule = Schedule()
        self.glob = None  # the last measured Global Align (this window session only)
        self.transform = FieldTransform()  # of the field exposed last
        self.runner: ScheduleRunner | None = None
        self._thread: threading.Thread | None = None
        self.modified = False
        self.ask_on_close = True
        self.plans = None

        # --- central view
        self.view = StageView()
        self.view.stepClicked.connect(lambda i: self.tab_schedule.select(i))
        self.view.contextRequested.connect(self._context_menu)
        self.setCentralWidget(self.view)

        # --- side tabs
        self.tab_positions = StagePositionsTab(self.settings, self.hw.stage_position)
        self.tab_positions.changed.connect(self._positions_changed)
        self.tab_positions.gotoRequested.connect(self.goto_stage)
        self.tab_positions.newFieldRequested.connect(
            lambda x, y: self.tab_schedule.add_op(ExposeField(centre=(x, y))))
        self.tab_schedule = ScheduleTab()
        self.tab_schedule.new_field_centre = lambda: self.hw.stage_position() or (0.0, 0.0)
        ed = self.tab_schedule.editor
        ed.presets = self.hw.preset_names
        ed.cells = self.library.cells
        ed.current_for = self._current_for
        ed.pickGdsRequested.connect(self.pattern_for_selected)
        self.tab_schedule.changed.connect(self._schedule_edited)
        self.tab_schedule.selectionChanged.connect(lambda i: self._replan_soon())
        self.tab_schedule.runRequested.connect(self.run)
        self.tab_schedule.pauseRequested.connect(self.pause)
        self.tab_schedule.abortRequested.connect(self.abort)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_positions, "Stage Positions")
        self.tabs.addTab(self.tab_schedule, "Exposure Schedule")
        self.tabs.setCurrentWidget(self.tab_schedule)
        dock = QtWidgets.QDockWidget("Lithography")
        dock.setObjectName("litho-side")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(440)
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
        self.tab_schedule.set_schedule(self.schedule)
        self._set_running_ui(False)
        self._positions_changed()
        self._title()
        self.replan()
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
        self.a_reset = act(m, "Reset global alignment", self.reset_global)

        m = mb.addMenu("F&ield")
        self.a_add_field = act(m, "Add Expose field", lambda: tab.add_op(ExposeField(centre=tab.new_field_centre())),
                               "Ctrl+E")
        self.a_add_column = act(m, "Add Column setting",
                                lambda: tab.add_op(ColumnSetting(magnification=tab._last_mag())))
        self.a_add_corr = act(m, "Add Field correction", lambda: tab.add_op(FieldCorrection()))
        m.addSeparator()
        self.a_dup = act(m, "Duplicate", tab.duplicate, "Ctrl+D")
        self.a_del = act(m, "Delete", tab.delete)
        self.a_up = act(m, "Move up", lambda: tab.move(-1), "Ctrl+Up")
        self.a_down = act(m, "Move down", lambda: tab.move(+1), "Ctrl+Down")
        m.addSeparator()
        self.a_move = act(m, "Move all fields…", lambda: self.move_fields_dialog(), "Ctrl+M")
        self.field_actions = [self.a_add_field, self.a_add_column, self.a_add_corr, self.a_dup, self.a_del, self.a_up,
                              self.a_down, self.a_move]

        m = mb.addMenu("&Import")
        self.a_import = act(m, "Fractured GDS…", self.import_dialog, "Ctrl+I")
        self.a_import_cell = act(m, "GDS cell as a new field…", self.import_cell_dialog)

        m = mb.addMenu("&View")
        act(m, "Zoom to all", self.view.zoom_all, "Ctrl+0")
        act(m, "Zoom to the selected field", self.zoom_selected, "Ctrl+9")
        m.addSeparator()
        self.a_grid = m.addAction("WCS grid of the selected field")
        self.a_grid.setCheckable(True)
        self.a_grid.setChecked(True)
        self.a_grid.toggled.connect(self._toggle_grid)
        m.addAction(self.side_dock.toggleViewAction())
        m.addAction(self.log_dock.toggleViewAction())

        m = mb.addMenu("S&tage")
        act(m, "Go to the selected field centre", self.goto_selected_field)
        act(m, "Add the stage position to Stage Positions…", self.tab_positions.add_current)
        m.addSeparator()
        act(m, "Stage axes…", self.edit_axes)

    def _make_status(self):
        sb = self.statusBar()
        self.l_global = QtWidgets.QLabel()
        self.l_field_tr = QtWidgets.QLabel()
        self.l_field = QtWidgets.QLabel()
        self.l_stage = QtWidgets.QLabel()
        self.l_run = QtWidgets.QLabel()
        self.l_run.setStyleSheet("color:#fd4")
        sb.addWidget(self.l_global)
        sb.addWidget(self.l_field_tr)
        for w in (self.l_run, self.l_field, self.l_stage, self.progress):
            sb.addPermanentWidget(w)
        for w in (self.l_global, self.l_field_tr, self.l_field, self.l_stage):
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

    def _update_status(self):
        if self.glob is None:
            self.l_global.setText("Global alignment: none (identity)")
            self.l_global.setStyleSheet("color:#9aa")
        else:
            self.l_global.setText(f"Global alignment: {self.glob.describe()}")
            self.l_global.setStyleSheet("color:#5d5")
        self.l_field_tr.setText("Last field: " + ("–" if self.transform.is_identity else self.transform.describe()))
        self.l_field.setText(f"field {self._field():.4g} µm")

    def _tick(self):
        try:
            pos = self.hw.stage_position()
        except Exception:  # noqa: BLE001
            pos = None
        self.l_stage.setText("stage ?" if pos is None else f"stage ({pos[0]:.1f}, {pos[1]:.1f}) µm")
        self.view.set_stage(pos)
        self.view.set_fov(pos, self._field())
        self._update_status()

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

    def _schedule_edited(self):
        self.modified = True
        self._title()
        self._replan_soon()

    def _replan_soon(self):
        self._replan_timer.start()

    def replan(self):
        try:
            presets = self.hw.preset_names()
        except Exception:  # noqa: BLE001
            presets = None
        self.plans = plan(self.schedule, self._field_at_1x(), self.settings.axes, presets, self.library, self.glob)
        self.tab_schedule.set_plans(self.plans)
        sel = self.tab_schedule.selected()
        self.view.set_schedule(self.schedule, self.library, self.plans, self.settings.axes, sel if sel >= 0 else None)
        return self.plans

    def warnings(self) -> list:
        return [(p.index, w) for p in (self.plans or self.replan()) for w in p.warnings]

    def validate(self):
        self.replan()
        ws = self.warnings()
        if not ws:
            self.log(f"validate: {len(self.schedule)} steps, no warnings")
        for i, w in ws:
            self.log(f"validate: step {i + 1}: {w}")
        return ws

    def _positions_changed(self):
        self.view.set_stage_positions(self.settings.stage_positions)

    def set_schedule(self, schedule: Schedule, modified: bool = False):
        self.schedule = schedule
        self.modified = modified
        self.view._fitted = False
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
        try:
            self.settings.save()
        except OSError:
            pass

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
        self.set_schedule(s)
        self._remember_dir(path)
        self.log(f"opened {path} ({len(s)} steps)")
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

    def import_dialog(self):
        if self._running():
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import fractured GDS", self._dir(), GDS_FILTER)
        if not p:
            return
        try:
            d = ImportDialog(p, self.library, self.hw.preset_names(), self.settings.axes, self.hw.stage_position,
                             self.schedule[0].magnification, self)
        except Exception as e:  # noqa: BLE001
            self.log(f"ERROR reading {p}: {e}")
            return
        if d.exec() != QtWidgets.QDialog.Accepted:
            return
        self.import_gds(p, d.options(), replace=d.replace.isChecked())

    def import_gds(self, path: str, options=None, replace: bool = True) -> Schedule:
        from ..fracture import import_fractured_gds

        s = import_fractured_gds(path, options, self.library)
        self._remember_dir(path)
        if replace:
            self.set_schedule(s, modified=True)
        else:
            self.schedule.operations.extend(s.operations[1:])  # the fields; the Column setting is the present one
            self.set_schedule(self.schedule, modified=True)
        self.log(f"imported {len(s) - 1} field(s) from {os.path.basename(path)}")
        return s

    def pattern_for_selected(self):
        i = self.tab_schedule.selected()
        if i < 0 or not isinstance(self.schedule[i], ExposeField):
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pattern GDS", self._dir(), GDS_FILTER)
        if p:
            self.set_pattern_file(i, p)

    def set_pattern_file(self, i: int, path: str, cell: str | None = None) -> None:
        try:
            cells = self.library.cells(path)
        except Exception as e:  # noqa: BLE001
            self.log(f"ERROR reading {path}: {e}")
            return
        op = self.schedule[i]
        op.pattern = PatternSpec("gds", os.path.abspath(path), cell or (cells[0] if cells else ""))
        self._remember_dir(path)
        self.tab_schedule.refresh(select=i)
        self._schedule_edited()

    def import_cell_dialog(self):
        if self._running():
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "GDS cell as a new field", self._dir(), GDS_FILTER)
        if not p:
            return
        tab = self.tab_schedule
        tab.add_op(ExposeField(centre=tab.new_field_centre(), label=os.path.basename(p)))
        self.set_pattern_file(tab.selected(), p)

    # ================================================================ move / view
    def anchors(self) -> dict:
        """Reference points for Move all fields (stage µm; None where there is none)."""
        fields = self.schedule.fields
        first = fields[0].centre if fields else None
        box = None
        if fields:
            plans = self.plans or self.replan()
            squares = []
            for i, op in enumerate(self.schedule.operations):
                if isinstance(op, ExposeField):
                    h = plans[i].field_um / 2.0 if i < len(plans) else 0.0
                    squares.append((op.centre[0] - h, op.centre[1] - h, op.centre[0] + h, op.centre[1] + h))
            b = list(zip(*squares))
            box = ((min(b[0]) + max(b[2])) / 2.0, (min(b[1]) + max(b[3])) / 2.0)
        g = next((o for o in fields if o.global_align and o.markers), None)
        mark = tuple(float(v) for v in g.nominal_scs([g.markers[0]], self.settings.axes)[0]) if g else None
        return {"first": ("the first field centre", first),
                "bbox": ("the centre of the bounding box of all fields", box),
                "marker": ("marker 1 of the first field with Global Align", mark)}

    def move_fields_dialog(self, click=None) -> bool:
        if self._running():
            return False
        if not self.schedule.fields:
            self.log("no fields to move")
            return False
        d = MoveFieldsDialog(self.anchors(), self.hw.stage_position(), click, self)
        self._move_dialog = d
        if d.exec() != QtWidgets.QDialog.Accepted or d.value() is None:
            return False
        self.move_fields(*d.value())
        return True

    def move_fields(self, dx: float, dy: float) -> None:
        n = self.schedule.shift_fields(dx, dy)
        self.log(f"moved {n} field(s) by ({dx:+.3f}, {dy:+.3f}) µm")
        self.tab_schedule.refresh()
        self._schedule_edited()

    def _context_menu(self, x: float, y: float, screen_pos):
        m = QtWidgets.QMenu(self)
        a = m.addAction("Move…")
        a.triggered.connect(lambda: self.move_fields_dialog(click=(x, y)))
        a.setEnabled(not self._running() and bool(self.schedule.fields))
        m.addSeparator()
        m.addAction("Zoom to all").triggered.connect(self.view.zoom_all)
        self._last_menu = m
        m.popup(screen_pos)

    def zoom_selected(self):
        i = self.tab_schedule.selected()
        plans = self.plans or self.replan()
        if 0 <= i < len(plans) and plans[i].target is not None:
            x, y = plans[i].target
            h = plans[i].field_um * 0.6
            self.view.getPlotItem().getViewBox().setRange(xRange=(x - h, x + h), yRange=(y - h, y + h))

    def _toggle_grid(self, on: bool):
        self.view.show_wcs_grid = bool(on)
        self.replan()

    def goto_selected_field(self):
        i = self.tab_schedule.selected()
        plans = self.plans or self.replan()
        if 0 <= i < len(plans) and plans[i].target is not None:
            self.goto_stage(*plans[i].target)

    def _set_running_ui(self, on: bool):
        self.tab_schedule.set_running(on)
        for a in [self.a_new, self.a_open, self.a_import, self.a_import_cell, self.a_reset, self.a_run,
                  self.a_run_from] + self.field_actions:
            a.setEnabled(not on)
        self.a_pause.setEnabled(on)
        self.a_abort.setEnabled(on)

    # ================================================================ alignment
    def edit_axes(self):
        d = StageAxesDialog(self.settings.axes, self)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.settings.axes = d.value()
            try:
                self.settings.save()
            except OSError as e:
                self.log(f"stage axes not saved: {e}")
            self._replan_soon()

    def reset_global(self):
        if self._running():
            return
        self.glob = None
        self.transform = FieldTransform()
        self.log("global alignment reset (identity)")
        self._update_status()
        self._replan_soon()

    # ================================================================ stage
    def goto_stage(self, x: float, y: float):
        if self._running():
            return
        self.log(f"stage → ({x:.2f}, {y:.2f}) µm")

        def job():
            try:
                self.hw.move_stage(x, y)
                self.sig.event.emit("log", {"text": f"stage at ({x:.2f}, {y:.2f}) µm"})
            except Exception as e:  # noqa: BLE001
                self.sig.event.emit("log", {"text": f"ERROR stage move: {type(e).__name__}: {e}"})

        threading.Thread(target=job, name="litho-goto", daemon=True).start()

    # ================================================================ run
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, start: int = 0, confirm: bool | None = None) -> bool:
        if self._running():
            return False
        ws = [(i, w) for i, w in self.validate() if i == 0 or i >= start]
        if ws and (self.ask_on_close if confirm is None else confirm):
            text = "\n".join(f"step {i + 1}: {w}" for i, w in ws[:15]) + ("\n…" if len(ws) > 15 else "")
            if QtWidgets.QMessageBox.question(self, "Run schedule", f"{len(ws)} warning(s):\n\n{text}\n\nRun anyway?") \
                    != QtWidgets.QMessageBox.Yes:
                return False
        self.bridge.cancelled.clear()
        self.tab_schedule.reset_states()
        self.runner = ScheduleRunner(self.schedule, self.hw, self.settings.axes, self.library, self.glob,
                                     on_event=lambda kind, data: self.sig.event.emit(kind, data))
        self._set_running_ui(True)
        self.progress.setValue(0)
        self.l_run.setText("running")
        self.log(f"run from step {start + 1}")
        self._thread = threading.Thread(target=self.runner.run, args=(start,), name="litho-run", daemon=True)
        self._thread.start()
        return True

    def pause(self, on: bool):
        if self.runner is not None:
            self.runner.pause(on)
            self.l_run.setText("paused (after this step)" if on else "running")

    def abort(self):
        if self.runner is not None:
            self.log("abort requested")
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
        elif kind == "alignment":
            self.glob = data["glob"]
            self.transform = data["transform"]
            self._update_status()
            self._replan_soon()
        elif kind == "exposed":
            self.progress.setValue(1000)
        elif kind == "finished":
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
