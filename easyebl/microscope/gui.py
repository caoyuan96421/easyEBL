"""Microscope window: SEM column settings and stage control over the RS-232 HAL.

Left: the SEM column (high voltage, magnification, focus / WD, stigmator,
condenser, SEM display).  Right: the stage (positions, STOP ALL, moves, jogs).
Every control queues a request on the shared link and returns at once; the
window shows readbacks as they arrive (it never waits for the port).

Opened from the easyEBL program (``easyebl``: Open → Microscope), which opens the
serial port once and shares it with SEM imaging (column panel, metadata) and
EDS (beam voltage).
"""
from __future__ import annotations

import concurrent.futures
import time

from PySide6 import QtCore, QtGui, QtWidgets

from .control import NotAvailable
from .link import MicroscopeLink
from .s3000 import commands as C
from .sem import MAGNIFICATIONS, PRESET_KEYS, SCAN_MODES, SEMControl
from .settings import MicroscopeSettings
from .stage import AXES, StageControl

__all__ = ["MicroscopeWindow", "BeamPresetDialog"]

STATE_COLORS = {"connected": "#4c4", "uncertain": "#fb4", "disconnected": "#f55", "closed": "#888"}
_BIG = "font-size:20px;font-weight:bold"
_NOTE = "color:#9aa"


def _fmt_mag(m) -> str:
    return "×—" if not m else (f"×{m / 1000:g}k" if m >= 1000 else f"×{m:g}")


def _spin(lo, hi, value=0, step=1, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(int(lo), int(hi))
    s.setSingleStep(int(step))
    s.setValue(int(value))
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)  # one request per Enter / arrow click, not per keystroke
    return s


def _dspin(lo, hi, dec, value=0.0, step=1.0, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class MicroscopeWindow(QtWidgets.QMainWindow):
    TICK_MS = 200

    def __init__(self, link: MicroscopeLink, sem: SEMControl, stage: StageControl,
                 settings: MicroscopeSettings | None = None, owns_link: bool = True, confirm: bool | None = None):
        super().__init__()
        self.link, self.sem, self.stage = link, sem, stage
        self.settings = settings or MicroscopeSettings()
        self.owns_link = owns_link
        # physical consequences (HV, moves, automatic routines) are confirmed on the real instrument
        self.confirm = (not link.simulated) if confirm is None else bool(confirm)
        self._sub = link.events.subscribe(maxlen=2000)
        self._syncing = False
        self.setWindowTitle(f"easyEBL Microscope – {sem.name}")

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._scroll(self._sem_panel()))
        split.addWidget(self._scroll(self._stage_panel()))
        split.setSizes([520, 620])
        self.setCentralWidget(split)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.raw = QtWidgets.QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setMaximumBlockCount(3000)
        self.raw.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.log, "Messages")
        tabs.addTab(self.raw, "Port log (TX/RX)")
        tabs.currentChanged.connect(lambda i: self._show_raw() if i == 1 else None)
        self._raw_tabs = tabs
        dock = QtWidgets.QDockWidget("Messages")
        dock.setObjectName("mic_messages")
        dock.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.setMaximumHeight(170)

        self._make_toolbar()
        self._make_status()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.TICK_MS)
        self.notify(f"{sem.name}: SEM settings go first; readbacks are polled in the background by "
                    f"priority, MAG / HV most often (none less often than every {link.poller.MAX_PERIOD_S:g} s)")
        self._sync_all()

    @staticmethod
    def _scroll(w):
        sc = QtWidgets.QScrollArea()
        sc.setWidget(w)
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return sc

    # ================================================================ toolbar / status
    def _make_toolbar(self):
        tb = self.addToolBar("Microscope")
        tb.setObjectName("mic_toolbar")
        tb.setMovable(False)
        self.l_link = QtWidgets.QLabel()
        tb.addWidget(self.l_link)
        tb.addSeparator()
        self.a_resync = tb.addAction("Resync…", self.resync)
        self.a_resync.setToolTip("after a timeout the link refuses commands until you have checked the SEM")
        tb.addAction("Read SEM", self.refresh_sem)
        tb.addAction("Read stage", self.refresh_stage)
        tb.addSeparator()
        tb.addAction("Beam presets…", self.open_presets)

    def _make_status(self):
        sb = self.statusBar()
        self.l_bus = QtWidgets.QLabel()
        self.l_poll = QtWidgets.QLabel()
        for w in (self.l_bus, self.l_poll):
            sb.addPermanentWidget(w)
            w.setContentsMargins(6, 0, 10, 0)

    # ================================================================ SEM panel
    def _sem_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        sem = self.sem

        # --- high voltage
        g = QtWidgets.QGroupBox("Accelerating voltage")
        f = QtWidgets.QGridLayout(g)
        self.l_hv = QtWidgets.QLabel("—")
        self.l_hv.setStyleSheet(_BIG)
        f.addWidget(self.l_hv, 0, 0, 1, 2)
        self.l_emi = QtWidgets.QLabel("emission —")
        self.l_emi.setStyleSheet(_NOTE)
        self.l_emi.setToolTip("R EMI (raw – the protocol does not give its units)")
        f.addWidget(self.l_emi, 0, 2, 1, 2)
        self.hv_set = _dspin(0.3, 30.0, 2, 20.0, 0.1, " kV")
        self.hv_set.setToolTip("setpoint: 0.30–9.99 kV in 0.01 kV, 10.0–30.0 kV in 0.1 kV steps")
        self.b_hv_set = QtWidgets.QPushButton("Set")
        self.b_hv_set.clicked.connect(self.apply_hv)
        self.b_hv_on = QtWidgets.QPushButton("HV ON")
        self.b_hv_on.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold}")
        self.b_hv_on.clicked.connect(lambda: self._confirmed("Switch the high voltage ON?", sem.hv_on, "HV ON"))
        self.b_hv_off = QtWidgets.QPushButton("HV OFF")
        self.b_hv_off.setStyleSheet("QPushButton{background:#8a2c2c;color:white;font-weight:bold}")
        self.b_hv_off.clicked.connect(lambda: self._confirmed("Switch the high voltage OFF?", sem.hv_off, "HV OFF"))
        f.addWidget(QtWidgets.QLabel("Setpoint"), 1, 0)
        f.addWidget(self.hv_set, 1, 1)
        f.addWidget(self.b_hv_set, 1, 2)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.b_hv_on)
        r.addWidget(self.b_hv_off)
        f.addLayout(r, 2, 0, 1, 4)
        lay.addWidget(g)

        # --- magnification
        g = QtWidgets.QGroupBox("Magnification")
        f = QtWidgets.QGridLayout(g)
        self.l_mag = QtWidgets.QLabel("×—")
        self.l_mag.setStyleSheet(_BIG)
        f.addWidget(self.l_mag, 0, 0, 1, 2)
        self.mag_combo = QtWidgets.QComboBox()
        for m in MAGNIFICATIONS:
            self.mag_combo.addItem(_fmt_mag(m), m)
        self.mag_combo.setMaxVisibleItems(20)
        self.mag_combo.activated.connect(lambda i: self._call(sem.set_magnification, self.mag_combo.itemData(i)))
        b_dn = QtWidgets.QPushButton("−")
        b_up = QtWidgets.QPushButton("+")
        for b in (b_dn, b_up):
            b.setFixedWidth(36)
        b_dn.clicked.connect(lambda: self._call(sem.step_magnification, -1))
        b_up.clicked.connect(lambda: self._call(sem.step_magnification, +1))
        f.addWidget(b_dn, 1, 0)
        f.addWidget(self.mag_combo, 1, 1)
        f.addWidget(b_up, 1, 2)
        lay.addWidget(g)

        # --- focus / WD
        g = QtWidgets.QGroupBox("Focus && working distance")
        f = QtWidgets.QGridLayout(g)
        self.wd = _spin(3, 60, 15, 1, " mm")
        self.wd.setToolTip("WD n (3–60 mm): the SEM sets the coarse focus for it and centres the fine focus")
        self.l_wd = QtWidgets.QLabel("—")
        self.l_wd.setToolTip("working distance computed from the coarse and fine focus (the SEM reports whole mm)")
        self.wd.valueChanged.connect(lambda v: self._user(sem.set_working_distance, v))
        self.focus_c = _spin(0, 16383, 8192, sem.coarse_focus_step)
        self.focus_c.valueChanged.connect(lambda v: self._user(sem.set_focus, "C", v))
        self.focus_f = _spin(0, 4095, 2048, sem.focus_step)
        self.focus_f.valueChanged.connect(lambda v: self._user(sem.set_focus, "F", v))
        f.addWidget(QtWidgets.QLabel("Working distance"), 0, 0)
        f.addWidget(self.wd, 0, 1)
        f.addWidget(self.l_wd, 0, 2)
        f.addWidget(QtWidgets.QLabel("Focus coarse"), 1, 0)
        f.addWidget(self.focus_c, 1, 1)
        f.addWidget(QtWidgets.QLabel("Focus fine"), 2, 0)
        f.addWidget(self.focus_f, 2, 1)
        r = QtWidgets.QHBoxLayout()
        for text, coarse in (("Auto focus (coarse)", True), ("Auto focus (fine)", False)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, c=coarse: self._confirmed(
                "Run the SEM's automatic focus? It takes the port for up to a few minutes.",
                lambda: sem.auto_focus(coarse=c), "auto focus"))
            r.addWidget(b)
        f.addLayout(r, 3, 0, 1, 2)
        lay.addWidget(g)

        # --- stigmator
        g = QtWidgets.QGroupBox("Stigmator")
        f = QtWidgets.QGridLayout(g)
        self.stig_x = _spin(0, 4095, 2048, sem.stig_step)
        self.stig_x.valueChanged.connect(lambda v: self._user(sem.set_stigmator, "X", v))
        self.stig_y = _spin(0, 4095, 2048, sem.stig_step)
        self.stig_y.valueChanged.connect(lambda v: self._user(sem.set_stigmator, "Y", v))
        f.addWidget(QtWidgets.QLabel("X"), 0, 0)
        f.addWidget(self.stig_x, 0, 1)
        f.addWidget(QtWidgets.QLabel("Y"), 1, 0)
        f.addWidget(self.stig_y, 1, 1)
        b = QtWidgets.QPushButton("Auto stigma")
        b.clicked.connect(lambda: self._confirmed("Run the SEM's automatic stigmation?", sem.auto_stigma,
                                                  "auto stigma"))
        f.addWidget(b, 2, 0, 1, 2)
        lay.addWidget(g)

        # --- beam current (condenser)
        g = QtWidgets.QGroupBox("Beam current (condenser)")
        f = QtWidgets.QGridLayout(g)
        self.beam_c = _spin(0, 99, 50)
        self.beam_c.valueChanged.connect(lambda v: self._user(sem.set_beam, "C", v))
        self.beam_f = _spin(0, 99, 50)
        self.beam_f.valueChanged.connect(lambda v: self._user(sem.set_beam, "F", v))
        f.addWidget(QtWidgets.QLabel("Coarse"), 0, 0)
        f.addWidget(self.beam_c, 0, 1)
        f.addWidget(QtWidgets.QLabel("Fine"), 1, 0)
        f.addWidget(self.beam_f, 1, 1)
        note = QtWidgets.QLabel("Raw 0–99 settings (the remote protocol reports no calibrated probe current; "
                                "watch the EDS input count rate to judge the current).")
        note.setWordWrap(True)
        note.setStyleSheet(_NOTE)
        f.addWidget(note, 2, 0, 1, 3)
        b = QtWidgets.QPushButton("Presets…")
        b.setToolTip("save and recall named settings: beam coarse / fine, stigmator, contrast / brightness")
        b.clicked.connect(self.open_presets)
        f.addWidget(b, 0, 2, 2, 1)
        lay.addWidget(g)

        # --- SEM display
        g = QtWidgets.QGroupBox("SEM display && detector")
        f = QtWidgets.QGridLayout(g)
        self.cont = _spin(0, 4095, 2048, 16)
        self.cont.valueChanged.connect(lambda v: self._user(sem.set_contrast, v))
        self.brt = _spin(0, 4095, 2048, 16)
        self.brt.valueChanged.connect(lambda v: self._user(sem.set_brightness, v))
        b_abc = QtWidgets.QPushButton("ABC")
        b_abc.setToolTip("automatic brightness / contrast of the SEM's own display")
        b_abc.clicked.connect(lambda: self._call(sem.auto_brightness_contrast))
        self.scan_combo = QtWidgets.QComboBox()
        for k, v in SCAN_MODES.items():
            self.scan_combo.addItem(v, k)
        self.scan_combo.activated.connect(lambda i: self._call(sem.set_scan, self.scan_combo.itemData(i)))
        self.det_combo = QtWidgets.QComboBox()
        self.det_combo.addItems(list(C.SIGNAL_TOKENS))
        self.det_combo.activated.connect(lambda i: self._call(sem.set_detector, self.det_combo.itemText(i)))
        self.b_run = QtWidgets.QPushButton("Freeze")
        self.b_run.setCheckable(True)
        self.b_run.setToolTip("RUN OFF / RUN ON – freezes the SEM's own display (does not blank the beam)")
        self.b_run.toggled.connect(lambda on: self._user(sem.set_running, not on))
        f.addWidget(QtWidgets.QLabel("Contrast"), 0, 0)
        f.addWidget(self.cont, 0, 1)
        f.addWidget(QtWidgets.QLabel("Brightness"), 1, 0)
        f.addWidget(self.brt, 1, 1)
        f.addWidget(b_abc, 0, 2, 2, 1)
        f.addWidget(QtWidgets.QLabel("Scan"), 2, 0)
        f.addWidget(self.scan_combo, 2, 1)
        f.addWidget(self.b_run, 2, 2)
        f.addWidget(QtWidgets.QLabel("Detector"), 3, 0)
        f.addWidget(self.det_combo, 3, 1)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Background read")
        f = QtWidgets.QFormLayout(g)
        self.sem_poll = QtWidgets.QCheckBox("Read SEM settings in the background")
        self.sem_poll.setChecked(sem.polling)
        self.sem_poll.setToolTip("Reads the SEM settings one after another without pause (MAG / HV most "
                                 "often), so changes on the SEM panel reach imaging and EDS")
        self.sem_poll.toggled.connect(self._sem_poll_changed)
        f.addRow(self.sem_poll)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    # ================================================================ stage panel
    def _stage_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        st = self.stage
        self.b_stop = QtWidgets.QPushButton("STOP ALL AXES")
        self.b_stop.setMinimumHeight(48)
        self.b_stop.setStyleSheet("QPushButton{background:#c62828;color:white;font-size:18px;font-weight:bold}"
                                  "QPushButton:pressed{background:#8e0000}")
        self.b_stop.setToolTip("S0 to every axis at once – also while a SEM command holds the port (Esc)")
        self.b_stop.clicked.connect(self.stop_all)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.stop_all)
        lay.addWidget(self.b_stop)

        # --- who drives the stage: easyEBL (#STAGE ON) or the SEM PC (#STAGE OFF)
        self.b_enable = QtWidgets.QPushButton()
        self.b_enable.setCheckable(True)
        self.b_enable.setMinimumHeight(44)
        self.b_enable.setToolTip("Enabled: easyEBL drives the stage (#STAGE ON).\n"
                                 "Disabled: the SEM PC has it (#STAGE OFF) – its trackball works and it\n"
                                 "decides about motor power; positions are still read.")
        self.b_enable.toggled.connect(self._motion_enabled)
        lay.addWidget(self.b_enable)

        # --- DRO
        g = QtWidgets.QGroupBox("Position  (double-click an axis to move it)")
        grid = QtWidgets.QGridLayout(g)
        self.dro, self.dro_state = {}, {}
        for i, a in enumerate(AXES):
            big = a != "T"
            name = QtWidgets.QLabel(a)
            name.setStyleSheet(f"font-size:{30 if big else 16}px;font-weight:bold;color:#9cf")
            val = DROLabel("—")
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            val.setFont(QtGui.QFont("Consolas", 30 if big else 15, QtGui.QFont.Bold))
            val.setStyleSheet("background:#111;color:#7f7;border:1px solid #333;padding:2px 8px")
            val.setMinimumWidth(300 if big else 200)
            val.setToolTip(f"double-click: move {a}")
            val.double_clicked.connect(lambda a=a: self.open_move_dialog(a))
            state = QtWidgets.QLabel("—")
            state.setStyleSheet("color:#bbb")
            grid.addWidget(name, i, 0)
            grid.addWidget(val, i, 1)
            grid.addWidget(QtWidgets.QLabel(st.unit(a)), i, 2)
            grid.addWidget(state, i, 3)
            self.dro[a], self.dro_state[a] = val, state
        grid.setColumnStretch(1, 1)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Read now")
        b.clicked.connect(self.refresh_stage)
        r.addWidget(b)
        self.cb_poll = QtWidgets.QCheckBox("Auto-read")
        self.cb_poll.setChecked(st.polling)
        self.cb_poll.setToolTip("Read the stage positions in the background, with the SEM readbacks "
                                "(X/Y and Z/T after MAG / HV and the column settings, R least often)")
        self.cb_poll.toggled.connect(self._stage_poll_changed)
        r.addWidget(self.cb_poll)
        r.addStretch(1)
        self.l_read = QtWidgets.QLabel("")
        self.l_read.setStyleSheet(_NOTE)
        r.addWidget(self.l_read)
        grid.addLayout(r, len(AXES), 0, 1, 4)
        lay.addWidget(g)

        # --- jog
        self.g_jog = QtWidgets.QGroupBox("Jog")
        f = QtWidgets.QGridLayout(self.g_jog)
        self.jog_um = _dspin(0.1, 10000, 1, self.settings.jog_step_um, 10, " µm")
        self.jog_deg = _dspin(0.01, 90, 2, self.settings.jog_step_deg, 0.5, " °")
        self.jog_um.valueChanged.connect(lambda v_: setattr(self.settings, "jog_step_um", v_))
        self.jog_deg.valueChanged.connect(lambda v_: setattr(self.settings, "jog_step_deg", v_))
        self.backlash = _dspin(0, 1000, 1, self.settings.stage_backlash_um, 10, " µm")
        self.backlash.setToolTip("X / Y always end a move travelling in + by at least this much "
                                 "(backlash clearing: a move ending in − overshoots and comes back); 0: off")
        self.backlash.valueChanged.connect(self._backlash_changed)

        def jb(text, axis, sign, tip):
            b = QtWidgets.QToolButton(text=text)
            b.setFixedSize(48, 40)
            b.setStyleSheet("font-size:16px")
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False: self.jog(axis, sign))
            return b

        f.addWidget(jb("▲", "Y", +1, "Y +"), 0, 1)
        f.addWidget(jb("◀", "X", -1, "X −"), 1, 0)
        f.addWidget(jb("▶", "X", +1, "X +"), 1, 2)
        f.addWidget(jb("▼", "Y", -1, "Y −"), 2, 1)
        f.addWidget(jb("Z+", "Z", +1, "Z + (µm step)"), 0, 4)
        f.addWidget(jb("Z−", "Z", -1, "Z − (µm step)"), 2, 4)
        f.addWidget(jb("R⟲", "R", -1, "rotate − (degree step)"), 0, 5)
        f.addWidget(jb("R⟳", "R", +1, "rotate + (degree step)"), 0, 6)
        f.addWidget(jb("T−", "T", -1, "tilt − (degree step)"), 2, 5)
        f.addWidget(jb("T+", "T", +1, "tilt + (degree step)"), 2, 6)
        f.addWidget(QtWidgets.QLabel("Step"), 3, 0)
        f.addWidget(self.jog_um, 3, 1, 1, 3)
        f.addWidget(self.jog_deg, 3, 4, 1, 3)
        f.addWidget(QtWidgets.QLabel("X/Y approach"), 4, 0)
        f.addWidget(self.backlash, 4, 1, 1, 3)
        lay.addWidget(self.g_jog)
        note = QtWidgets.QLabel("Moves wait in line behind SEM settings; STOP never waits.  The stage has no "
                                "homing or collision model – watch the chamber camera when moving Z, R or T.")
        note.setWordWrap(True)
        note.setStyleSheet(_NOTE)
        lay.addWidget(note)
        lay.addStretch(1)
        self._motion_enabled(False)
        return w

    # ================================================================ helpers
    def notify(self, msg: str):
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def _call(self, fn, *a, what: str = ""):
        """Queue a HAL request; failures are reported when the reply comes back."""
        try:
            fut = fn(*a)
        except NotAvailable as e:
            self.notify(f"{e} – not available")
            return None
        except Exception as e:  # noqa: BLE001 - validation errors: shown at once
            self.notify(f"not sent: {e}")
            return None
        if isinstance(fut, concurrent.futures.Future):
            fut.add_done_callback(lambda f, w=what or getattr(fn, "__name__", "request"): self._report(f, w))
        return fut

    def _report(self, fut, what):
        if fut.cancelled():
            return
        e = fut.exception()
        if e is not None:
            QtCore.QMetaObject.invokeMethod(self, "_report_error", QtCore.Qt.QueuedConnection,
                                            QtCore.Q_ARG(str, f"{what}: {type(e).__name__}: {e}"))

    @QtCore.Slot(str)
    def _report_error(self, msg: str):
        self.notify(msg)

    def _user(self, fn, *a):
        """A control changed by the user (not by a readback)."""
        if not self._syncing:
            self._call(fn, *a)

    def _confirmed(self, question: str, fn, what: str):
        if self.confirm and QtWidgets.QMessageBox.question(self, "Microscope", question) != QtWidgets.QMessageBox.Yes:
            return None
        return self._call(fn, what=what)

    def apply_hv(self):
        v = round(self.hv_set.value(), 2 if self.hv_set.value() < 10 else 1)
        self._call(self.sem.set_hv, f"{v:.2f}" if v < 10 else f"{v:.1f}", what="HV setpoint")

    def _backlash_changed(self, v):
        self.settings.stage_backlash_um = float(v)
        self.stage.backlash_um = float(v)
        try:
            self.settings.save()
        except OSError:
            pass

    def _stage_poll_changed(self, v):
        self.stage.set_polling(v)
        self.settings.stage_polling = bool(v)

    def _sem_poll_changed(self, v):
        self.sem.set_polling(v)
        self.settings.sem_polling = bool(v)

    def refresh_sem(self):
        self._call(self.sem.refresh, what="SEM read")

    def open_presets(self):
        """The beam preset dialog (one instance, non-modal)."""
        d = getattr(self, "presets", None)
        if d is None:
            d = self.presets = BeamPresetDialog(self.sem, self.settings, self)
        d.show()
        d.raise_()
        d.activateWindow()
        return d

    def refresh_stage(self):
        self._call(self.stage.refresh, what="stage read")

    def resync(self):
        if QtWidgets.QMessageBox.question(
                self, "Resync", "A reply timed out – the last command may or may not have been executed.\n"
                "Check the SEM and the stage, then resume communication?") != QtWidgets.QMessageBox.Yes:
            return
        try:
            late = self.link.resync()
            self.notify("link resynchronised" + (f"; late replies: {late}" if late else ""))
        except Exception as e:  # noqa: BLE001
            self.notify(f"resync failed: {e}")

    # ================================================================ stage actions
    def stop_all(self):
        if self.stage.hand_control:
            self.notify("STOP not sent: the SEM PC has the stage (use its controls)")
            return
        self.stage.stop()
        self.notify("STOP sent to all axes")

    @property
    def motion_enabled(self) -> bool:
        return self.b_enable.isChecked()

    def _motion_enabled(self, on: bool):
        """Enabled: easyEBL drives the stage (#STAGE ON).  Disabled: stop, then #STAGE OFF –
        the SEM PC (trackball) has the stage."""
        self.b_enable.setText("STAGE: easyEBL control  –  click to hand back to the SEM PC" if on else
                              "STAGE: SEM PC (trackball)  –  click to enable easyEBL control")
        self.b_enable.setStyleSheet("QPushButton{font-size:15px;font-weight:bold;"
                                    + ("background:#2e7d32;color:white}" if on else "background:#455a64;color:#ddd}"))
        self.g_jog.setEnabled(on)
        try:  # a failure is reported through the link's error events (log)
            self.stage.engage() if on else self.stage.release()
        except Exception as e:  # noqa: BLE001 - link closed
            self.notify(f"stage: {e}")

    def open_move_dialog(self, axis: str):
        """Direct move of one axis (double-click on its readout)."""
        if not self.motion_enabled:
            self.notify("stage motion is not enabled (the SEM PC has the stage)")
            return None
        d = MoveDialog(axis, self.stage.unit(axis), self.stage.positions.get(axis), self)
        d.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        d.accepted.connect(lambda: self.move({axis: d.value()}, relative=d.relative(), confirm=False))
        d.open()
        return d

    def jog(self, axis: str, sign: int):
        step = self.jog_um.value() if self.stage.unit(axis) == "µm" else self.jog_deg.value()
        self.move({axis: sign * step}, relative=True, confirm=False)

    def move(self, targets: dict, relative: bool, confirm: bool | None = None):
        if not self.motion_enabled:
            self.notify("stage motion is not enabled (the SEM PC has the stage)")
            return None
        desc = ", ".join(f"{a} {'by ' if relative else 'to '}{v:+g} {self.stage.unit(a)}" if relative else
                         f"{a} to {v:g} {self.stage.unit(a)}" for a, v in targets.items())
        if (self.confirm if confirm is None else confirm) and QtWidgets.QMessageBox.question(
                self, "Move stage", f"Move {desc}?") != QtWidgets.QMessageBox.Yes:
            return None
        fut = self._call(self.stage.move, targets, relative, what=f"move {desc}")
        if fut is not None:
            self.notify(f"move queued: {desc}")
        return fut

    # ================================================================ updates
    def _show_raw(self):
        entries = self.link.log_entries()[-1500:]
        self.raw.setPlainText("\n".join(e.format() for e in entries))
        self.raw.verticalScrollBar().setValue(self.raw.verticalScrollBar().maximum())

    def _tick(self):
        for ev in self._sub.drain():
            if ev.message and ev.kind in ("state", "error", "info", "stage", "stage_event"):
                self.notify(ev.message)
        self._sync_all()
        if self._raw_tabs.currentIndex() == 1 and self.link.completed != getattr(self, "_raw_n", -1):
            self._raw_n = self.link.completed
            self._show_raw()

    def _sync_all(self):
        s = self.sem.state
        self._syncing = True
        try:
            hv_on, kv = s.get("hv_on"), s.get("hv_kv")
            self.l_hv.setText(f"{kv:g} kV  ON" if hv_on and kv else ("HV OFF" if hv_on is False else "—"))
            self.l_hv.setStyleSheet(_BIG + (";color:#7d7" if hv_on else ";color:#bbb"))
            emi = s.get("emission")
            self.l_emi.setText(f"emission {emi} (raw)" if emi is not None else "emission —")
            sp = s.get("hv_setpoint_kv")
            if sp is not None and not self.hv_set.hasFocus():
                self.hv_set.setValue(sp)
            self.l_mag.setText(_fmt_mag(s.get("mag")))
            if s.get("mag") in MAGNIFICATIONS and not self.mag_combo.view().isVisible():
                self.mag_combo.setCurrentIndex(MAGNIFICATIONS.index(s["mag"]))
            wd = s.get("wd_mm")
            self.l_wd.setText(f"{wd:.3f} mm" if isinstance(wd, float) else "—")
            for box, key in ((self.wd, "wd_readback_mm"), (self.focus_c, "focus_coarse"), (self.focus_f, "focus_fine"),
                             (self.stig_x, "stig_x"), (self.stig_y, "stig_y"), (self.beam_c, "beam_coarse"),
                             (self.beam_f, "beam_fine"), (self.cont, "contrast"), (self.brt, "brightness")):
                v = s.get(key)
                if isinstance(v, int) and box.value() != v and not box.hasFocus():
                    box.setValue(v)
            if s.get("running") is not None:
                self.b_run.setChecked(not s["running"])
                self.b_run.setText("Frozen – Run" if not s["running"] else "Freeze")
            if s.get("scan") in SCAN_MODES:
                self.scan_combo.setCurrentIndex(list(SCAN_MODES).index(s["scan"]))
            det = s.get("detector")
            if det and self.det_combo.findText(det) >= 0:
                self.det_combo.setCurrentIndex(self.det_combo.findText(det))
        finally:
            self._syncing = False
        st = self.stage
        for a in AXES:
            p = st.positions.get(a)
            self.dro[a].setText("—" if p is None else (f"{p:,.1f}" if st.unit(a) == "µm" else f"{p:.3f}"))
            state = st.motion_state(a)
            self.dro_state[a].setText(state)
            self.dro_state[a].setStyleSheet("color:" + {"moving": "#fb4", "completed": "#4c4", "halted": "#f90",
                                                        "limit": "#f55", "fault": "#f55",
                                                        "uncertain": "#f55"}.get(state, "#bbb"))
        t = max((v for v in st.t_read.values() if v), default=None)
        self.l_read.setText("read " + time.strftime("%H:%M:%S", time.localtime(t)) if t else "not read yet")
        state = self.link.state
        self.l_link.setText(f"<b style='color:{STATE_COLORS.get(state, '#ccc')}'>{state.upper()}</b> "
                            f"{self.link.port}" + (f" – {self.link.error}" if self.link.error else ""))
        self.a_resync.setEnabled(state == "uncertain")
        busy = self.link.busy_with
        self.l_bus.setText(f"port: {busy}" + (f" (+{self.link.pending} queued)" if self.link.pending else "")
                           if busy else "port idle")
        self.l_poll.setText(self._poll_text())

    def _poll_text(self) -> str:
        st = self.stage
        if st.moving:
            return "stage moving"
        p = st.poll_period_s() if st.polling else None
        tasks = self.link.poller.tasks()
        self.l_poll.setToolTip("background reads – level: measured read time / period:\n" + "\n".join(
            f"{t.name}  {t.level}: {t.cost:.2f} s / " + (f"{t.period:.1f} s" if t.period else "–")
            for t in sorted(tasks, key=lambda t: t.level)))
        return ("stage auto-read off" if not st.polling else
                f"stage X/Y read every {p:.0f} s" if p else "stage auto-read on")

    # ================================================================ lifecycle
    def closeEvent(self, ev):  # noqa: N802
        self.timer.stop()
        self._sub.close()
        try:
            self.settings.save()
        except OSError:
            pass
        if self.owns_link:
            self.link.close()
        super().closeEvent(ev)


# ==================================================================== stage widgets
class DROLabel(QtWidgets.QLabel):
    """A digital-readout value; double-click opens the move dialog."""

    double_clicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, ev):  # noqa: N802
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(ev)


class MoveDialog(QtWidgets.QDialog):
    """Move one axis: to a position (pre-filled with the current one) or by a distance."""

    def __init__(self, axis: str, unit: str, current: float | None, parent=None):
        super().__init__(parent)
        self.axis, self.current = axis, current
        self.setWindowTitle(f"Move {axis}")
        lay = QtWidgets.QFormLayout(self)
        cur = QtWidgets.QLabel("—" if current is None else
                               (f"{current:,.1f} {unit}" if unit == "µm" else f"{current:.3f} {unit}"))
        cur.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Bold))
        lay.addRow("Current", cur)
        self.rb_abs = QtWidgets.QRadioButton("to")
        self.rb_rel = QtWidgets.QRadioButton("by")
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.rb_abs)
        r.addWidget(self.rb_rel)
        r.addStretch(1)
        lay.addRow("Move", r)
        self.val = _dspin(-1e6, 1e6, 1 if unit == "µm" else 3, 0.0, 10.0 if unit == "µm" else 1.0, f" {unit}")
        self.val.setFont(QtGui.QFont("Consolas", 14))
        lay.addRow(axis, self.val)
        self.rb_abs.toggled.connect(self._mode)
        if current is None:
            self.rb_rel.setChecked(True)
            self.rb_abs.setEnabled(False)  # position not read yet: relative only
        else:
            self.rb_abs.setChecked(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Move")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.val.setFocus()
        self.val.selectAll()

    def _mode(self, absolute: bool):
        self.val.setValue(self.current if absolute and self.current is not None else 0.0)
        self.val.selectAll()

    def relative(self) -> bool:
        return self.rb_rel.isChecked()

    def value(self) -> float:
        return self.val.value()


# ==================================================================== beam presets
PRESET_COLUMNS = (("beam_coarse", "Beam C"), ("beam_fine", "Beam F"), ("stig_x", "Stig X"), ("stig_y", "Stig Y"),
                  ("contrast", "Contrast"), ("brightness", "Brightness"))
assert [k for k, _ in PRESET_COLUMNS] == list(PRESET_KEYS)


class BeamPresetDialog(QtWidgets.QDialog):
    """Named presets of beam 1 (coarse / fine), stigmator X / Y and contrast / brightness.

    *Save current…* reads the six settings fresh from the SEM and stores them
    under a name; *Recall* sends the selected preset (double-click does the
    same).  Names and values can be edited in the table.  Presets are kept in
    the microscope settings file and saved on every change.
    """

    _read_done = QtCore.Signal(object, object, object)  # (name, values | None, error | None) – from the port thread
    _sent = QtCore.Signal(str, object)  # (name, error | None)

    def __init__(self, sem: SEMControl, settings: MicroscopeSettings, parent=None):
        super().__init__(parent)
        self.sem, self.settings = sem, settings
        self.setWindowTitle("Beam presets")
        self.resize(720, 360)
        self._filling = False
        lay = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 1 + len(PRESET_COLUMNS))
        self.table.setHorizontalHeaderLabels(["Name"] + [t for _, t in PRESET_COLUMNS])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditKeyPressed
                                   | QtWidgets.QAbstractItemView.SelectedClicked)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in range(1, self.table.columnCount()):
            hh.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._edited)
        self.table.itemDoubleClicked.connect(lambda _it: self.recall())
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        lay.addWidget(self.table, 1)

        self.l_current = QtWidgets.QLabel()
        self.l_current.setStyleSheet(_NOTE)
        lay.addWidget(self.l_current)

        row = QtWidgets.QHBoxLayout()
        self.b_save = QtWidgets.QPushButton("Save current…")
        self.b_save.setToolTip("read the six settings from the SEM and store them under a new name")
        self.b_save.clicked.connect(lambda: self.save_current())
        self.b_update = QtWidgets.QPushButton("Update from SEM")
        self.b_update.setToolTip("overwrite the selected preset with the SEM's current settings")
        self.b_update.clicked.connect(self.update_selected)
        self.b_recall = QtWidgets.QPushButton("Recall")
        self.b_recall.setToolTip("send the selected preset to the SEM (double-clicking a row does the same)")
        self.b_recall.setStyleSheet("font-weight:bold")
        self.b_recall.clicked.connect(lambda: self.recall())
        self.b_up = QtWidgets.QPushButton("▲")
        self.b_up.setToolTip("move up")
        self.b_up.clicked.connect(lambda: self.move_selected(-1))
        self.b_down = QtWidgets.QPushButton("▼")
        self.b_down.setToolTip("move down")
        self.b_down.clicked.connect(lambda: self.move_selected(+1))
        self.b_delete = QtWidgets.QPushButton("Delete")
        self.b_delete.clicked.connect(self.delete_selected)
        for b in (self.b_save, self.b_update, self.b_recall):
            row.addWidget(b)
        row.addStretch(1)
        for b in (self.b_up, self.b_down, self.b_delete):
            row.addWidget(b)
        b = QtWidgets.QPushButton("Close")
        b.clicked.connect(self.close)
        row.addWidget(b)
        lay.addLayout(row)
        self.l_msg = QtWidgets.QLabel(" ")
        lay.addWidget(self.l_msg)

        self._read_done.connect(self._on_read)
        self._sent.connect(self._on_sent)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._show_current)
        self.timer.start(500)
        self._fill()
        self._show_current()

    # ---------------------------------------------------------------- model
    @property
    def presets(self) -> list:
        return self.settings.beam_presets

    def _save(self):
        try:
            self.settings.save()
        except OSError as e:
            self._msg(f"presets not saved: {e}", error=True)

    def _msg(self, text: str, error: bool = False):
        self.l_msg.setText(text)
        self.l_msg.setStyleSheet("color:#f77" if error else "color:#9c9")
        parent = self.parent()
        if parent is not None and hasattr(parent, "notify"):
            parent.notify("beam presets: " + text)

    def _index(self, name: str) -> int:
        return next((i for i, p in enumerate(self.presets) if p.get("name") == name), -1)

    def selected(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _fill(self, select: int | None = None):
        self._filling = True
        try:
            self.table.setRowCount(len(self.presets))
            for r, p in enumerate(self.presets):
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(p.get("name", ""))))
                for c, (k, _t) in enumerate(PRESET_COLUMNS, start=1):
                    it = QtWidgets.QTableWidgetItem(str(p.get(k, "")))
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    self.table.setItem(r, c, it)
        finally:
            self._filling = False
        if select is not None and 0 <= select < len(self.presets):
            self.table.selectRow(select)
        self._sync_buttons()

    def _sync_buttons(self):
        r = self.selected()
        for b in (self.b_update, self.b_recall, self.b_delete):
            b.setEnabled(r >= 0)
        self.b_up.setEnabled(r > 0)
        self.b_down.setEnabled(0 <= r < len(self.presets) - 1)

    def _show_current(self):
        s = self.sem.state
        self.l_current.setText("SEM now:   " + "   ".join(
            f"{t} {s.get(k) if s.get(k) is not None else '—'}" for k, t in PRESET_COLUMNS))

    # ---------------------------------------------------------------- editing in the table
    def _edited(self, item):
        if self._filling:
            return
        r, c = item.row(), item.column()
        p = self.presets[r]
        text = item.text().strip()
        if c == 0:
            if not text:
                self._msg("a preset needs a name", error=True)
            elif self._index(text) not in (-1, r):
                self._msg(f"a preset named {text!r} exists already", error=True)
            else:
                p["name"] = text
                self._save()
        else:
            key, title = PRESET_COLUMNS[c - 1]
            lo, hi = self.sem.preset_limits(key)
            try:
                v = int(text)
                if not lo <= v <= hi:
                    raise ValueError
            except ValueError:
                self._msg(f"{title} must be a whole number {lo}…{hi}", error=True)
            else:
                p[key] = v
                self._save()
        # re-show the stored values (reverts a refused edit); deferred: the editor is still closing
        QtCore.QTimer.singleShot(0, lambda: self._fill(select=r))

    # ---------------------------------------------------------------- actions
    def _new_name(self) -> str:
        n = len(self.presets) + 1
        while self._index(f"Preset {n}") >= 0:
            n += 1
        return f"Preset {n}"

    def save_current(self, name: str | None = None):
        """Read the settings from the SEM and store them as *name* (asked when not given)."""
        if name is None:
            name, ok = QtWidgets.QInputDialog.getText(self, "Save beam preset", "Name:", text=self._new_name())
            if not ok:
                return None
        name = name.strip()
        if not name:
            return self._msg("a preset needs a name", error=True)
        if self._index(name) >= 0 and QtWidgets.QMessageBox.question(
                self, "Beam presets", f"Overwrite the preset {name!r}?") != QtWidgets.QMessageBox.Yes:
            return None
        return self._read(name)

    def update_selected(self):
        r = self.selected()
        if r < 0:
            return None
        name = self.presets[r]["name"]
        if QtWidgets.QMessageBox.question(self, "Beam presets", f"Overwrite {name!r} with the SEM's current "
                                          "settings?") != QtWidgets.QMessageBox.Yes:
            return None
        return self._read(name)

    def _read(self, name: str):
        try:
            fut = self.sem.read_preset()
        except Exception as e:  # noqa: BLE001 - link closed / refused
            return self._msg(f"cannot read the SEM: {e}", error=True)
        self._msg(f"reading the SEM settings for {name!r}…")
        self.b_save.setEnabled(False)
        fut.add_done_callback(lambda f: self._read_done.emit(
            name, None if f.cancelled() or f.exception() else f.result(),
            "cancelled" if f.cancelled() else f.exception()))
        return fut

    def _on_read(self, name, values, error):
        self.b_save.setEnabled(True)
        if values is None:
            return self._msg(f"{name!r} not saved – reading the SEM failed: {error}", error=True)
        entry = {"name": name, **values}
        i = self._index(name)
        if i >= 0:
            self.presets[i] = entry
        else:
            self.presets.append(entry)
            i = len(self.presets) - 1
        self._save()
        self._fill(select=i)
        self._msg(f"saved {name!r}")
        return None

    def recall(self, name: str | None = None):
        """Send the selected preset (or the one called *name*) to the SEM."""
        i = self.selected() if name is None else self._index(name)
        if i < 0:
            return None
        p = self.presets[i]
        try:
            fut = self.sem.apply_preset(p)
        except Exception as e:  # noqa: BLE001 - incomplete / out-of-range preset, link refused
            self._msg(f"{p.get('name')!r} not sent: {e}", error=True)
            return None
        self._msg(f"sending {p['name']!r}…")
        fut.add_done_callback(lambda f, n=p["name"]: self._sent.emit(
            n, "cancelled" if f.cancelled() else f.exception()))
        return fut

    def _on_sent(self, name, error):
        if error is None:
            self._msg(f"recalled {name!r}")
        else:
            self._msg(f"recalling {name!r} failed: {error}", error=True)

    def move_selected(self, step: int):
        r = self.selected()
        j = r + step
        if r < 0 or not 0 <= j < len(self.presets):
            return
        self.presets[r], self.presets[j] = self.presets[j], self.presets[r]
        self._save()
        self._fill(select=j)

    def delete_selected(self):
        r = self.selected()
        if r < 0:
            return
        name = self.presets[r]["name"]
        if QtWidgets.QMessageBox.question(self, "Beam presets", f"Delete the preset {name!r}?") \
                != QtWidgets.QMessageBox.Yes:
            return
        del self.presets[r]
        self._save()
        self._fill(select=min(r, len(self.presets) - 1))
        self._msg(f"deleted {name!r}")
