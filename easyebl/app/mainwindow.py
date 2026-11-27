"""The easyEBL main window: every subsystem in one window (sized for a 1920 × 1080 screen).

* **Centre** – tabs *SEM*, *EDS*, *Lithography*: the image / spectrum / stage plot of each.
* **Left** – the microscope: column (HV, field size, magnification, focus, stigmator, beam, display)
  and stage (STOP, position, moves), and the beam monitor (the scan coils' X / Y as an
  oscilloscope in XY mode).
* **Right** – the settings of the subsystem of the active tab (SEM scan / image / field of view,
  EDS acquire / map / composition, lithography chips and exposure schedule).
* **Bottom** – messages, stage positions, calibration and alignment data, a small SEM view (for
  picking markers without leaving the Lithography tab), the EDS Data Manager and live spectrum,
  the microscope log.
* **Menus and toolbar** – the active subsystem's appear next to the program's own; its keyboard
  shortcuts work only while its tab is shown.  **Esc stops everything**: the stage (every axis),
  a running exposure schedule, EDS acquisitions and the SEM scan.

The subsystem windows are built as before (:class:`~easyebl.imaging.sem.window.SEMWindow`,
:class:`~easyebl.eds.gui.window.EDSWindow`, :class:`~easyebl.litho.module.LithoWindow`,
:class:`~easyebl.microscope.gui.MicroscopeWindow`) and their panels moved in here: they keep their
logic, and the separate windows stay available (*Windows* menu, the classic menu window).
"""
from __future__ import annotations

import os
import time

from PySide6 import QtCore, QtGui, QtWidgets

from .backend import COMPONENTS

__all__ = ["MainWindow", "ModuleHost", "SEMMiniView", "CalibrationView", "AlignmentView", "UI_POINT_SIZE"]

#: the one font size of the program (points) – small enough for everything to fit a 1080p screen
UI_POINT_SIZE = 8.0

TABS = (("sem", "SEM"), ("eds", "EDS"), ("litho", "Lithography"))
_STATE_COLORS = {"ok": "#4c4", "sim": "#6af", "waiting": "#fb4", "warn": "#fb4", "error": "#f55", "off": "#777"}


def _placeholder(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setAlignment(QtCore.Qt.AlignCenter)
    lab.setWordWrap(True)
    lab.setStyleSheet("color:#9aa")
    return lab


def _unscroll(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """The widget inside a QScrollArea (taken out of it), else *w*."""
    if isinstance(w, QtWidgets.QScrollArea) and w.widget() is not None:
        inner = w.takeWidget()
        w.deleteLater()
        return inner
    return w


def _dock_named(win: QtWidgets.QMainWindow, name: str) -> QtWidgets.QDockWidget | None:
    for d in win.findChildren(QtWidgets.QDockWidget):
        if d.objectName() == name:
            return d
    return None


def _take_dock(win: QtWidgets.QMainWindow, name: str) -> QtWidgets.QWidget | None:
    """Remove dock *name* from *win*; returns its contents (out of a scroll area)."""
    d = _dock_named(win, name)
    if d is None:
        return None
    w = d.widget()
    win.removeDockWidget(d)
    d.toggleViewAction().setVisible(False)
    if w is not None:
        w.setParent(None)
        w = _unscroll(w)
    d.hide()
    return w


def _all_actions(menu: QtWidgets.QMenu) -> list:
    out = []
    for a in menu.actions():
        out.append(a)
        if a.menu() is not None:
            out.extend(_all_actions(a.menu()))
    return out


class ModuleHost:
    """One subsystem in the main window: its window (embedded, the tab), what was moved out of it,
    its menus, toolbars and shortcuts (enabled while its tab is shown)."""

    def __init__(self, key: str, window: QtWidgets.QMainWindow | None):
        self.key = key
        self.window = window
        self.right: QtWidgets.QWidget | None = None
        self.log: QtWidgets.QWidget | None = None
        self.menus: list = []  # QMenu
        self.toolbars: list = []  # QToolBar
        self.shortcuts: list = []  # window-wide QShortcut
        self.action_keys: dict = {}  # QAction -> [QKeySequence]
        self.moved: list = []  # widgets moved into the main window (deleted with the host)

    def take_menus(self) -> None:
        mb = self.window.menuBar()
        for a in list(mb.actions()):
            if a.menu() is None:
                continue
            for sub in _all_actions(a.menu()):
                if sub.text().replace("&", "") == "Close":  # the tab is closed with the program
                    sub.setVisible(False)
            self.menus.append(a.menu())
        mb.hide()

    def take_toolbars(self) -> None:
        for tb in self.window.findChildren(QtWidgets.QToolBar):
            if tb.parent() is self.window:
                self.window.removeToolBar(tb)
                self.toolbars.append(tb)

    def collect_shortcuts(self) -> None:
        """Window-wide shortcuts and menu / toolbar actions with keys; Esc ones are switched off
        for good (Esc is the main window's: stop everything)."""
        esc = QtGui.QKeySequence("Esc")
        for s in self.window.findChildren(QtGui.QShortcut):
            if s.context() not in (QtCore.Qt.WindowShortcut, QtCore.Qt.ApplicationShortcut):
                continue
            if s.key() == esc:
                s.setEnabled(False)
                continue
            self.shortcuts.append(s)
        acts = []
        for m in self.menus:
            acts.extend(_all_actions(m))
        for tb in self.toolbars:
            acts.extend(tb.actions())
        for a in acts:
            keys = [k for k in a.shortcuts() if k != esc]
            if a.shortcuts():
                a.setShortcuts(keys)
            if keys:
                self.action_keys[a] = keys

    def set_active(self, on: bool) -> None:
        for m in self.menus:
            m.menuAction().setVisible(on)
        for tb in self.toolbars:
            tb.setVisible(on)
        for s in self.shortcuts:
            s.setEnabled(on)
        for a, keys in self.action_keys.items():
            a.setShortcuts(keys if on else [])


# ==================================================================== bottom-bar pages
class SEMMiniView(QtWidgets.QWidget):
    """A small live SEM image of the SEM tab's window – also for picking: a marker pick asked by
    the lithography shows here and in the SEM tab (click / Confirm / Cancel in either); the
    wheel zooms the scan, a click outside a pick centres the beam there."""

    REFRESH_MS = 100

    def __init__(self, sem_window, parent=None):
        super().__init__(parent)
        from ..imaging.sem.view import SEMImageView

        self.sem = sem_window
        self.view = SEMImageView()
        self.view.setMinimumSize(160, 120)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        bar = QtWidgets.QHBoxLayout()
        self.l_pick = QtWidgets.QLabel("SEM imaging – the SEM tab has every control")
        self.l_pick.setWordWrap(True)
        self.b_run = QtWidgets.QPushButton("Run / freeze")
        self.b_run.clicked.connect(lambda: sem_window.scan._run_freeze())
        self.b_full = QtWidgets.QPushButton("Full field")
        self.b_full.clicked.connect(lambda: sem_window.fov._do(sem_window.engine.full_field))
        self.b_confirm = QtWidgets.QPushButton("Confirm (Enter)")
        self.b_confirm.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold}")
        self.b_confirm.clicked.connect(sem_window.confirm_pick)
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.clicked.connect(sem_window.cancel_pick)
        bar.addWidget(self.l_pick, 1)
        for b in (self.b_run, self.b_full, self.b_confirm, self.b_cancel):
            bar.addWidget(b)
        v.addLayout(bar)
        v.addWidget(self.view, 1)
        self.view.pointPicked.connect(self._picked)
        self.view.centerRequested.connect(lambda x, y: sem_window._center_at(x, y))
        self.view.shiftDragged.connect(lambda dx, dy: sem_window._drag_shift(dx, dy))
        self.view.zoomSteps.connect(sem_window.fov.zoom)
        self._seq = 0
        self._pick_key = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.REFRESH_MS)
        self.refresh()

    def _picked(self, x: float, y: float) -> None:
        if getattr(self.sem, "_pick_cb", None) is not None:
            self.sem._picked(x, y)

    def refresh(self) -> None:
        sv = self.sem.view
        key = (sv._pick_prompt, tuple(sv._pick_targets), sv._pick_centre_cross) if sv.pick_mode else None
        if key != self._pick_key:
            self._pick_key = key
            if key is None:
                self.view.stop_pick()
            else:
                self.view.start_pick(key[0], key[1], centre_cross=key[2])
        picking = key is not None
        self.b_confirm.setVisible(picking and getattr(self.sem, "_pick_centre", False))
        self.b_cancel.setVisible(picking)
        if picking:
            self.l_pick.setText(self.sem.l_pick.text() or key[0])
        else:
            self.l_pick.setText("SEM imaging – the SEM tab has every control")
        if not self.isVisible():
            return
        f = self.sem.engine.latest()
        if f is not None and f.seq != self._seq:
            self._seq = f.seq
            self.view.set_frame(f, self.sem.levels())

    def keyPressEvent(self, ev):  # noqa: N802
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and self.sem.view.pick_mode:
            self.sem.confirm_pick()
            ev.accept()
            return
        super().keyPressEvent(ev)


def _table(headers) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.horizontalHeader().setStretchLastSection(True)
    t.verticalHeader().setDefaultSectionSize(18)
    return t


def _fill(t: QtWidgets.QTableWidget, rows) -> None:
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, (text, color) in enumerate(row):
            it = t.item(r, c)
            if it is None:
                it = QtWidgets.QTableWidgetItem()
                t.setItem(r, c, it)
            it.setText(str(text))
            it.setToolTip(str(text))
            it.setForeground(QtGui.QColor(color or "#ddd"))
    t.resizeColumnsToContents()


class CalibrationView(QtWidgets.QWidget):
    """What is stored per field size (lithography): magnification, nominal WD, WD1 and Z_cal of the
    calibration sample, the beam calibration and the field calibration."""

    HEADERS = ["Field size", "Mag.", "Nominal WD", "WD1", "Z_cal", "Beam calibration", "Field calibration"]

    def __init__(self, litho, parent=None):
        super().__init__(parent)
        self.litho = litho
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = _table(self.HEADERS)
        v.addWidget(self.table)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def rows(self) -> list:
        from ..litho.fieldcal import Condition
        from ..litho.schedule import find_beamcal

        s = self.litho.settings
        out = []
        for fs in s.field_sizes:
            cur = fs.name == s.field_size
            name_col = "#5d5" if cur else None
            wd1, zc = s.wd1(fs.name), s.z_cal(fs.name)
            bc = find_beamcal(s.beamcals, Condition(float(fs.magnification), None, None, fs.name))
            fc = s.fieldcal(fs.name)
            if fc is not None:
                import math

                import numpy as np

                b = np.asarray(fc.stage_to_beam, float)
                rot = math.degrees(math.atan2(b[1, 0] - b[0, 1], b[0, 0] + b[1, 1]))
                fct = f"rotation {rot:+.3f}° · rms {fc.residual_um:.3f} µm · " + \
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(fc.timestamp))
            else:
                fct = "none"
            out.append([
                (fs.name + ("  (selected)" if cur else ""), name_col),
                (f"×{fs.magnification:g}", None),
                (f"{fs.wd_mm:g} mm", None),
                (f"{wd1:.3f} mm" if wd1 is not None else "not recorded", None if wd1 is not None else "#fb4"),
                (f"{zc:.1f} µm" if zc is not None else "not recorded", None if zc is not None else "#fb4"),
                (f"{bc.created.replace('T', ' ')}: {bc.describe()}" if bc is not None else "none", None),
                (fct, None),
            ])
        return out

    def refresh(self) -> None:
        if not self.isVisible() and self.table.rowCount():
            return
        try:
            _fill(self.table, self.rows())
        except Exception:  # noqa: BLE001 - a display only
            pass


class AlignmentView(QtWidgets.QWidget):
    """The chips of the lithography schedule: alignment (Global Align result) and height."""

    HEADERS = ["Chip", "Origin (stage µm)", "Alignment", "Height"]

    def __init__(self, litho, parent=None):
        super().__init__(parent)
        self.litho = litho
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = _table(self.HEADERS)
        v.addWidget(self.table)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def rows(self) -> list:
        lw = self.litho
        out = []
        for c in lw.schedule.chips:
            aligned = c.name in lw.chips
            try:
                h = c.height.describe() if c.height is not None else ""
            except Exception:  # noqa: BLE001
                h = ""
            out.append([(c.name, None), (f"({c.origin[0]:.1f}, {c.origin[1]:.1f})", None),
                        (lw._chip_status(c.name), "#5d5" if aligned else "#9aa"), (h or "not measured", None)])
        return out

    def refresh(self) -> None:
        if not self.isVisible() and self.table.rowCount():
            return
        try:
            _fill(self.table, self.rows())
        except Exception:  # noqa: BLE001
            pass


# ==================================================================== the window
class MainWindow(QtWidgets.QMainWindow):
    """See module docstring.  *app* is the :class:`~easyebl.app.gui.EasyEBLApp`; :meth:`build`
    creates the subsystems from its backend, :meth:`teardown` closes them (before a reconnect)."""

    STATE_VERSION = 1

    def __init__(self, app, settings_path: str | None = None):
        super().__init__()
        from .gui import make_icon

        self.app = app
        self.setWindowTitle("easyEBL")
        self.setWindowIcon(make_icon())
        self.setObjectName("easyebl_main")
        self.setDockNestingEnabled(True)
        self.settings_path = settings_path
        self.hosts: dict[str, ModuleHost] = {}
        self.mic = None  # MicroscopeWindow (its panels are in the left dock)
        self.sem_mini: SEMMiniView | None = None
        self._hint_shown = False
        self._restored = False

        # --- centre: one tab per subsystem
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self._pages = {}
        for key, title in TABS:
            page = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            self._pages[key] = page
            self.tabs.addTab(page, title)
        self.tabs.currentChanged.connect(lambda _i: self._activate())

        # --- docks
        self.left_dock = self._dock("Microscope", "main_left", QtCore.Qt.LeftDockWidgetArea)
        self.right_dock = self._dock("Settings", "main_right", QtCore.Qt.RightDockWidgetArea)
        self.bottom_dock = self._dock("Data", "main_bottom", QtCore.Qt.BottomDockWidgetArea)
        self.left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(self.left)
        lv.setContentsMargins(2, 2, 2, 2)
        lv.setSpacing(2)
        self.left_tabs = QtWidgets.QTabWidget()
        lv.addWidget(self.left_tabs, 1)
        self.monitor_box = QtWidgets.QGroupBox("Beam monitor")
        QtWidgets.QVBoxLayout(self.monitor_box).setContentsMargins(2, 2, 2, 2)
        lv.addWidget(self.monitor_box)
        self.left_dock.setWidget(self.left)
        self.right_stack = QtWidgets.QStackedWidget()
        self.right_dock.setWidget(self.right_stack)
        self.bottom = QtWidgets.QTabWidget()
        self.bottom_dock.setWidget(self.bottom)
        self.messages = QtWidgets.QStackedWidget()  # the active subsystem's messages
        self.log = QtWidgets.QPlainTextEdit()  # the program's own
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.messages.addWidget(self.log)
        self.bottom.addTab(self.messages, "Messages")
        self.monitor = None

        # --- toolbar, menus, status
        self.toolbar = self.addToolBar("easyEBL")
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setMovable(False)
        self.b_stop = QtWidgets.QPushButton("STOP ALL (Esc)")
        self.b_stop.setStyleSheet("QPushButton{background:#c62828;color:white;font-weight:bold;padding:2px 10px}"
                                  "QPushButton:pressed{background:#8e0000}")
        self.b_stop.setToolTip("stop everything: every stage axis, a running exposure schedule, EDS "
                               "acquisitions and the SEM scan (Esc, anywhere in this window)")
        self.b_stop.clicked.connect(self.stop_everything)
        self.toolbar.addWidget(self.b_stop)
        self.toolbar.addSeparator()
        self.esc = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.esc.setContext(QtCore.Qt.WindowShortcut)
        self.esc.activated.connect(self.stop_everything)
        self._build_menus()
        self._build_status()

    # ------------------------------------------------------------------ construction
    def _dock(self, title: str, name: str, area) -> QtWidgets.QDockWidget:
        d = QtWidgets.QDockWidget(title)
        d.setObjectName(name)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
                      | QtWidgets.QDockWidget.DockWidgetClosable)
        self.addDockWidget(area, d)
        return d

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&easyEBL")
        m.addAction("Hide to tray", self.close)
        m.addAction("Classic menu window", self.app.show_classic_menu)
        m.addSeparator()
        m.addAction("Shut down easyEBL…", self.app.shutdown)
        m = mb.addMenu("&Hardware")
        m.addAction("Configure…", self.app.configure)
        m.addAction("Reconnect all", self.app.reconnect)
        self.m_panels = mb.addMenu("&Panels")
        for d in (self.left_dock, self.right_dock, self.bottom_dock):
            self.m_panels.addAction(d.toggleViewAction())
        self.m_panels.addSeparator()
        self.m_panels.addAction("Reset the layout", self.reset_layout)
        self.m_windows = mb.addMenu("&Windows")
        self.m_windows.setToolTipsVisible(True)
        self._window_actions = []
        from .gui import SUBSYSTEMS

        tools = None
        for sub in SUBSYSTEMS:
            if sub.key in ("eds", "litho"):
                continue  # once only: they are the tabs
            if sub.tool and tools is None:
                self.m_windows.addSeparator()
                tools = self.m_windows.addMenu("Tools")
            title = sub.title + (" (separate window)" if not sub.tool else "")
            a = (tools or self.m_windows).addAction(title, lambda k=sub.key: self.app.open_window(k))
            a.setData(sub.key)
            self._window_actions.append(a)

    def _build_status(self):
        sb = self.statusBar()
        self.status_labels: dict = {}
        for key, name in COMPONENTS.items():
            lab = QtWidgets.QLabel(f"● {name}")
            lab.setContentsMargins(4, 0, 8, 0)
            sb.addPermanentWidget(lab)
            self.status_labels[key] = lab

    def update_status(self, status: dict, config) -> None:
        for key, lab in self.status_labels.items():
            s = status.get(key)
            if s is None:
                continue
            lab.setText(f"<span style='color:{_STATE_COLORS.get(s.state, '#ccc')}'>●</span> {COMPONENTS[key]}")
            lab.setToolTip(s.text)
        for a in self._window_actions:
            ok, why = self.app.available(a.data(), status)
            a.setEnabled(ok)
            a.setToolTip(why)

    def notify(self, msg: str) -> None:
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    # ------------------------------------------------------------------ subsystems
    def build(self) -> None:
        """Create the subsystems the backend can run (a tab says why one cannot)."""
        app = self.app
        st = app.backend.status()
        for key, title in TABS:
            ok, why = app.available(key, st)
            win = None
            if ok:
                try:
                    win = app._create(key)
                except Exception as e:  # noqa: BLE001 - shown in the tab
                    why = f"{type(e).__name__}: {e}"
                    self.notify(f"{title}: cannot open – {why}")
            self._install(key, title, win, why)
        self._install_microscope(st)
        self._install_monitor()
        if self.hosts.get("sem") and self.hosts["sem"].window is not None:
            self.sem_mini = SEMMiniView(self.hosts["sem"].window)
            self.bottom.addTab(self.sem_mini, "SEM view")
        self._order_bottom()
        self._activate()
        self._uniform_font()

    def _install(self, key: str, title: str, win, why: str) -> None:
        page = self._pages[key]
        host = ModuleHost(key, win)
        self.hosts[key] = host
        if win is None:
            lab = _placeholder(f"{title} is not available.\n\n{why}")
            page.layout().addWidget(lab)
            host.moved.append(lab)
            host.right = _placeholder(f"{title}: not available")
            self.right_stack.addWidget(host.right)
            host.moved.append(host.right)
            return
        win.setWindowFlags(QtCore.Qt.Widget)
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        host.take_menus()
        host.take_toolbars()
        getattr(self, f"_take_{key}")(host)
        host.collect_shortcuts()
        before = self.m_panels.menuAction()
        for m in host.menus:
            self.menuBar().insertAction(before, m.menuAction())
        for tb in host.toolbars:
            self.addToolBar(tb)
        if host.right is None:
            host.right = _placeholder("")
        self.right_stack.addWidget(host.right)
        host.moved.append(host.right)
        if host.log is not None:
            self.messages.addWidget(host.log)
            host.moved.append(host.log)
        page.layout().addWidget(win)
        win.show()

    def _bottom_page(self, host: ModuleHost, widget, title: str) -> None:
        if widget is None:
            return
        widget.setParent(None)
        self.bottom.addTab(widget, title)
        host.moved.append(widget)

    def _take_sem(self, host: ModuleHost) -> None:
        w = host.window
        host.right = _take_dock(w, "controls")
        host.log = _take_dock(w, "messages")
        for tb in host.toolbars:  # the window's own full screen: not as a tab
            for a in tb.actions():
                if a.text() == "Full screen":
                    a.setVisible(False)

    def _take_eds(self, host: ModuleHost) -> None:
        w = host.window
        side = _take_dock(w, "eds_controls")  # the tabs + the live MCA panel
        host.log = _take_dock(w, "eds_messages")
        live = side.layout().itemAt(1).widget() if side is not None and side.layout().count() > 1 else None
        host.right = side
        self._bottom_page(host, w.data_tabs, "EDS data")
        self._bottom_page(host, live, "EDS live spectrum")

    def _take_litho(self, host: ModuleHost) -> None:
        w = host.window
        tabs = _take_dock(w, "litho-side")
        host.log = _take_dock(w, "litho-log")
        i = tabs.indexOf(w.tab_positions)
        if i >= 0:
            tabs.removeTab(i)
        host.right = tabs
        self._bottom_page(host, w.tab_positions, "Stage positions")
        cal = CalibrationView(w)
        self._bottom_page(host, cal, "Calibration")
        al = AlignmentView(w)
        self._bottom_page(host, al, "Alignment")
        self.calibration_view, self.alignment_view = cal, al

    def _install_microscope(self, st) -> None:
        b = self.app.backend
        ok, why = self.app.available("microscope", st)
        m = b.microscope
        if not ok or m is None or not m.connected:
            lab = _placeholder(f"Microscope not available.\n\n{why or 'not connected'}")
            self.left_tabs.addTab(lab, "Microscope")
            return
        from ..microscope.gui import MicroscopeWindow

        try:
            mic = MicroscopeWindow(m.link, m.sem, m.stage, m.settings, owns_link=False)
        except Exception as e:  # noqa: BLE001
            self.left_tabs.addTab(_placeholder(f"Microscope: {type(e).__name__}: {e}"), "Microscope")
            return
        self.mic = mic
        mic.setParent(self)
        mic.setWindowFlags(QtCore.Qt.Widget)
        mic.hide()  # never shown: its panels are here
        split = mic.centralWidget()
        panels = [_unscroll(split.widget(i)) for i in range(split.count())]
        for p in panels:
            p.setParent(None)
        for p, title in zip(panels, ("Column", "Stage")):
            self.left_tabs.addTab(p, title)
        for tb in mic.findChildren(QtWidgets.QToolBar):
            if tb.parent() is mic:
                mic.removeToolBar(tb)
                tb.setParent(None)
                tb.setIconSize(QtCore.QSize(12, 12))
                self.left.layout().insertWidget(0, tb)
                tb.show()
                self._mic_toolbar = tb
        log = _take_dock(mic, "mic_messages")
        if log is not None:
            self.bottom.addTab(log, "Microscope log")
            self._mic_log = log
        for s in mic.findChildren(QtGui.QShortcut):  # Esc: the main window's (stop everything)
            if s.key() == QtGui.QKeySequence("Esc"):
                s.setEnabled(False)
        self._mic_status = [mic.l_bus, mic.l_poll]
        for lab in self._mic_status:
            self.statusBar().addWidget(lab)
            lab.show()
        self.left_tabs.setCurrentIndex(0)

    def _install_monitor(self) -> None:
        from ..gui.scanmonitor import ScanMonitorWidget

        app = self.app
        mc, d = app.config.monitor, app.config.daq
        self.monitor = ScanMonitorWidget(app.scan_tap, mc.resolution, mc.retention_s, mc.fade,
                                         on_settings=app.set_monitor_settings,
                                         orientation=(d.invert_x, d.invert_y, d.swap_xy))
        self.monitor.setFixedHeight(250)
        self.monitor_box.layout().addWidget(self.monitor)

    def _order_bottom(self) -> None:
        order = ["Messages", "Stage positions", "Calibration", "Alignment", "SEM view", "EDS data",
                 "EDS live spectrum", "Microscope log"]
        bar = self.bottom.tabBar()
        for target, title in enumerate(order):
            for i in range(self.bottom.count()):
                if self.bottom.tabText(i) == title:
                    bar.moveTab(i, min(target, self.bottom.count() - 1))
                    break
        self.bottom.setCurrentIndex(0)

    def teardown(self) -> bool:
        """Close the subsystems (False: one refused, e.g. unsaved data kept)."""
        for key in ("litho", "eds", "sem"):
            h = self.hosts.get(key)
            if h is not None and h.window is not None and not h.window.close():
                return False
        if self.sem_mini is not None:
            self.sem_mini.timer.stop()
            self.sem_mini.deleteLater()
            self.sem_mini = None
        for h in self.hosts.values():
            for tb in h.toolbars:
                self.removeToolBar(tb)
                tb.deleteLater()
            for m in h.menus:
                self.menuBar().removeAction(m.menuAction())
            for w in h.moved:
                w.deleteLater()
            if h.window is not None:
                h.window.deleteLater()
        self.hosts.clear()
        if self.mic is not None:
            self.mic.close()
            self.mic.deleteLater()
            self.mic = None
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor.deleteLater()
            self.monitor = None
        for tabs in (self.left_tabs, self.bottom):
            while tabs.count() > (1 if tabs is self.bottom else 0):
                w = tabs.widget(tabs.count() - 1)
                tabs.removeTab(tabs.count() - 1)
                w.deleteLater()
        for lab in getattr(self, "_mic_status", []):
            self.statusBar().removeWidget(lab)
            lab.deleteLater()
        self._mic_status = []
        tb = getattr(self, "_mic_toolbar", None)
        if tb is not None:
            tb.deleteLater()
            self._mic_toolbar = None
        return True

    def window_of(self, key: str):
        h = self.hosts.get(key)
        return None if h is None else h.window

    @property
    def sem(self):
        return self.window_of("sem")

    @property
    def eds(self):
        return self.window_of("eds")

    @property
    def litho(self):
        return self.window_of("litho")

    def show_tab(self, key: str):
        """Bring the tab of *key* forward (the window shown)."""
        for i, (k, _t) in enumerate(TABS):
            if k == key:
                self.tabs.setCurrentIndex(i)
        self.bring_up()
        return self.window_of(key)

    def show_microscope(self):
        self.left_dock.show()
        self.left_dock.raise_()
        self.bring_up()
        return self.mic

    def sem_for_litho(self):
        """The SEM window the lithography picks markers in: the SEM tab's – shown in the small SEM
        view of the bottom bar while the Lithography tab is in front."""
        sem = self.sem
        if sem is None:
            return None
        if self.sem_mini is not None and self.tabs.currentWidget() is not self._pages["sem"]:
            self.bottom_dock.show()
            self.bottom.setCurrentWidget(self.sem_mini)
        return sem

    def bring_up(self) -> None:
        self.show()
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def active_key(self) -> str:
        return TABS[max(0, self.tabs.currentIndex())][0]

    def _activate(self) -> None:
        key = self.active_key()
        for k, h in self.hosts.items():
            h.set_active(k == key)
        h = self.hosts.get(key)
        if h is not None and h.right is not None:
            self.right_stack.setCurrentWidget(h.right)
            self.right_dock.setWindowTitle(f"{dict(TABS)[key]} settings")
        if h is not None and h.log is not None:
            self.messages.setCurrentWidget(h.log)
        else:
            self.messages.setCurrentWidget(self.log)

    # ------------------------------------------------------------------ emergency stop
    def stop_everything(self) -> list:
        """Esc: stop the stage (every axis), a running schedule, EDS acquisitions, the SEM scan;
        cancel marker picks.  Returns what was told to stop."""
        done = []

        def attempt(what, fn):
            try:
                fn()
                done.append(what)
            except Exception as e:  # noqa: BLE001 - the others still stop
                self.notify(f"STOP {what}: {type(e).__name__}: {e}")

        if self.mic is not None:
            attempt("stage", self.mic.stop_all)
        else:
            st = getattr(self.app.backend.microscope, "stage", None) if self.app.backend.microscope else None
            if st is not None:
                attempt("stage", st.stop)
        lw = self.litho
        if lw is not None:
            attempt("lithography", lambda: (lw.stop_pick(), lw.abort()))
        ew = self.eds
        if ew is not None:
            attempt("EDS", ew.stop)
        sw = self.sem
        if sw is not None:
            attempt("SEM scan", lambda: (sw.cancel_pick(), sw.engine.stop()))
        elif self.app.backend.engine is not None:
            attempt("SEM scan", self.app.backend.engine.stop)
        self.notify("STOP (Esc): " + (", ".join(done) or "nothing to stop"))
        self.statusBar().showMessage("STOP: " + ", ".join(done), 5000)
        return done

    # ------------------------------------------------------------------ fonts / layout
    def _uniform_font(self) -> None:
        """One font size everywhere: explicit sizes of moved widgets go back to the program's."""
        pt = QtWidgets.QApplication.instance().font().pointSizeF()
        for w in self.findChildren(QtWidgets.QWidget):
            if w.testAttribute(QtCore.Qt.WA_SetFont):
                f = QtGui.QFont(w.font())
                if f.pointSizeF() != pt:
                    f.setPointSizeF(pt)
                    w.setFont(f)

    def _qsettings(self) -> QtCore.QSettings | None:
        if not self.settings_path:
            return None
        return QtCore.QSettings(self.settings_path, QtCore.QSettings.IniFormat)

    def save_layout(self) -> None:
        s = self._qsettings()
        if s is None:
            return
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState(self.STATE_VERSION))
        s.setValue("tab", self.tabs.currentIndex())
        s.sync()

    def restore_layout(self) -> bool:
        s = self._qsettings()
        if s is None or not os.path.exists(self.settings_path):
            self.default_layout()
            return False
        ok = False
        g = s.value("geometry")
        if g is not None:
            ok = self.restoreGeometry(g)
        st = s.value("state")
        if st is not None:
            ok = self.restoreState(st, self.STATE_VERSION) and ok
        try:
            self.tabs.setCurrentIndex(int(s.value("tab", 0)))
        except (TypeError, ValueError):
            pass
        if not ok:
            self.default_layout()
        return ok

    def default_layout(self) -> None:
        """Left 340 px, right 420 px, bottom 230 px on the available screen (maximised)."""
        for d in (self.left_dock, self.right_dock, self.bottom_dock):
            d.setFloating(False)
            d.show()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.right_dock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.bottom_dock)
        # the bottom bar under the centre only: the corners stay the side bars
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.resizeDocks([self.left_dock, self.right_dock], [340, 420], QtCore.Qt.Horizontal)
        self.resizeDocks([self.bottom_dock], [230], QtCore.Qt.Vertical)

    def reset_layout(self) -> None:
        self.default_layout()

    # ------------------------------------------------------------------ closing
    def closeEvent(self, ev):  # noqa: N802
        """Closing hides the window: the backend keeps running in the tray."""
        self.save_layout()
        if self.app.closing:
            return super().closeEvent(ev)
        ev.ignore()
        if not self.app.has_tray:
            self.showMinimized()
            return
        self.hide()
        if not self._hint_shown:
            self._hint_shown = True
            self.app.tray_message("easyEBL is still running. Click the tray icon to bring the window back, "
                                  "right-click it to shut down.")
