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

_DRO_STYLE = "background:#111;color:%s;border:1px solid #333;padding:0px 4px"
#: a stage axis's motion state -> its DRO's colour (normal: green)
_STATE_COLOR = {"moving": "#fb4", "halted": "#f90", "limit": "#f55", "fault": "#f55", "uncertain": "#f55"}

__all__ = ["MicroscopeWindow", "BeamPresetDialog"]

STATE_COLORS = {"connected": "#4c4", "uncertain": "#fb4", "disconnected": "#f55", "closed": "#888"}
_BIG = "font-weight:bold"  # one font size everywhere
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
    def _hv_button(self, hv_on) -> None:
        """The HV button offers the other state: OFF (red) while on, ON (green) while off or unknown."""
        if getattr(self, "_hv_shown", "x") == hv_on:
            return
        self._hv_shown = hv_on
        on = bool(hv_on)
        self.b_hv.setText("HV OFF" if on else "HV ON")
        self.b_hv.setStyleSheet("QPushButton{background:%s;color:white;font-weight:bold}"
                                % ("#8a2c2c" if on else "#2e7d32"))

    #: how long HV ON / OFF may take before the window gives up waiting (s); re-read every HV_POLL_S
    HV_TIMEOUT_S = 120.0
    HV_POLL_S = 1.0

    def toggle_hv(self):
        """HV ON / OFF (asks first); a modal window until the SEM reports it on / off."""
        sem = self.sem
        if sem.state.get("hv_on"):
            target, title = False, "HV OFF"
            fut = self._confirmed("Switch the high voltage OFF?", sem.hv_off, title)
            text = "Switching the high voltage OFF – waiting until the SEM reports it off."
        else:
            kv = self._hv_value()
            target, title = True, "HV ON"
            fut = self._confirmed(f"Switch the high voltage ON at {kv:g} kV?", lambda: sem.hv_on(kv), title)
            text = f"Switching the high voltage ON at {kv:g} kV – waiting until the SEM reports it on."
        if fut is None:
            return None
        self._wait_hv(fut, target, title, text)
        return fut

    def _wait_hv(self, fut, target: bool, title: str, text: str) -> None:
        """The blocking window of HV ON / OFF: closes when the command has been answered and HV reads
        *target* (HV is read again every HV_POLL_S while it does not), or on an error / time-out."""
        from ..gui.busy import BusyDialog

        dlg = BusyDialog(title, text, self)
        self.hv_busy = dlg
        t0 = time.monotonic()
        st = {"read": None, "t_read": t0}
        timer = QtCore.QTimer(dlg)

        def end(ok: bool, msg: str = "") -> None:
            timer.stop()
            if msg:
                self.notify(msg)
            dlg.finish(ok)
            if self.hv_busy is dlg:
                self.hv_busy = None
            self._sync_all()

        def check() -> None:
            if not fut.done():
                pass
            elif fut.exception() is not None:
                e = fut.exception()
                return end(False, f"{title}: {type(e).__name__}: {e}")
            elif self.sem.state.get("hv_on") is target:
                kv = self.sem.state.get("hv_kv")
                return end(True, f"high voltage ON at {kv:g} kV" if target and kv else
                           f"high voltage {'ON' if target else 'OFF'}")
            elif ((st["read"] is None or st["read"].done())
                  and time.monotonic() - st["t_read"] >= self.HV_POLL_S):  # answered, not there yet: read HV
                st["t_read"] = time.monotonic()
                try:
                    st["read"] = self.sem.refresh(("HV", "EMI"))
                except Exception:  # noqa: BLE001
                    pass
            el = time.monotonic() - t0
            dlg.set_progress(f"{el:.0f} s")
            if el > self.HV_TIMEOUT_S:
                end(False, f"{title}: the SEM did not report HV {'on' if target else 'off'} "
                           f"within {self.HV_TIMEOUT_S:.0f} s")

        timer.timeout.connect(check)
        timer.start(250)
        dlg.show_busy()
        check()

    def _sem_panel(self):
        """The column, compact (it fits the left side bar of the main window at 1080p): every
        control in a grid, explanations in the tooltips."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        sem = self.sem

        def grid(title):
            g = QtWidgets.QGroupBox(title)
            f = QtWidgets.QGridLayout(g)
            f.setContentsMargins(6, 4, 6, 4)
            f.setHorizontalSpacing(6)
            f.setVerticalSpacing(3)
            lay.addWidget(g)
            return g, f

        # --- high voltage
        g, f = grid("Accelerating voltage")
        self.l_hv = QtWidgets.QLabel("—")
        self.l_hv.setStyleSheet(_BIG)
        self.l_emi = QtWidgets.QLabel("emission —")
        self.l_emi.setStyleSheet(_NOTE)
        self.l_emi.setToolTip("emission current (R EMI, in µA)")
        self.hv_set = _dspin(0.3, 30.0, 2, sem.DEFAULT_HV_KV, 0.1, " kV")
        self.hv_set.setToolTip("0.30–9.99 kV in 0.01 kV, 10.0–30.0 kV in 0.1 kV steps.  HV on: Enter (or leaving "
                               "the box) sets it on the SEM.  HV off: not sent – HV ON switches on at this value "
                               "(30 kV at start unless HV is already on)")
        self.hv_set.editingFinished.connect(self.apply_hv)
        self.b_hv = QtWidgets.QPushButton("HV ON")  # one button: switches to the other state (asks first)
        self.b_hv.setToolTip("switch the high voltage on / off (asks first)")
        self.b_hv.clicked.connect(self.toggle_hv)
        self.hv_busy = None  # the modal window while HV switches (easyebl.gui.busy.BusyDialog)
        self._hv_button(None)
        f.addWidget(self.l_hv, 0, 0)
        f.addWidget(self.l_emi, 0, 1, 1, 3)
        f.addWidget(self.hv_set, 1, 0, 1, 2)
        f.addWidget(self.b_hv, 1, 2, 1, 2)

        # --- exposure field size (the Lithography's) and magnification
        g, f = grid("Field size && magnification")
        f.addWidget(QtWidgets.QLabel("Field size"), 0, 0)
        f.addWidget(self._field_size_combo(), 0, 1, 1, 4)
        self.l_mag = QtWidgets.QLabel("×—")
        self.l_mag.setStyleSheet(_BIG)
        self.mag_combo = QtWidgets.QComboBox()
        for m in MAGNIFICATIONS:
            self.mag_combo.addItem(_fmt_mag(m), m)
        self.mag_combo.setMaxVisibleItems(20)
        self.mag_combo.activated.connect(lambda i: self._call(sem.set_magnification, self.mag_combo.itemData(i)))
        b_dn = QtWidgets.QPushButton("−")
        b_up = QtWidgets.QPushButton("+")
        for b in (b_dn, b_up):
            b.setFixedWidth(28)
        b_dn.clicked.connect(lambda: self._call(sem.step_magnification, -1))
        b_up.clicked.connect(lambda: self._call(sem.step_magnification, +1))
        f.addWidget(QtWidgets.QLabel("Mag."), 1, 0)
        f.addWidget(self.l_mag, 1, 1)
        f.addWidget(b_dn, 1, 2)
        f.addWidget(self.mag_combo, 1, 3)
        f.addWidget(b_up, 1, 4)
        f.setColumnStretch(3, 1)

        # --- focus / WD and stigmator
        g, f = grid("Focus && stigmator")
        self.wd = _dspin(3.0, 60.0, 3, 15.0, 0.1, " mm")
        self.wd.setToolTip("working distance computed from the coarse and fine focus; a new value sets the "
                           "focus (FOCUS C and F from the focus table)")
        self.wd.valueChanged.connect(lambda v: self._user(sem.set_working_distance, float(v)))
        self.focus_c = _spin(0, 16383, 8192, sem.coarse_focus_step)
        self.focus_c.valueChanged.connect(lambda v: self._user(sem.set_focus, "C", v))
        self.focus_f = _spin(0, 4095, 2048, sem.focus_step)
        self.focus_f.valueChanged.connect(lambda v: self._user(sem.set_focus, "F", v))
        self.stig_x = _spin(0, 4095, 2048, sem.stig_step)
        self.stig_x.valueChanged.connect(lambda v: self._user(sem.set_stigmator, "X", v))
        self.stig_y = _spin(0, 4095, 2048, sem.stig_step)
        self.stig_y.valueChanged.connect(lambda v: self._user(sem.set_stigmator, "Y", v))
        for r, (a, wa, b, wb) in enumerate((("WD", self.wd, "Stig X", self.stig_x),
                                            ("Coarse", self.focus_c, "Stig Y", self.stig_y),
                                            ("Fine", self.focus_f, None, None))):
            f.addWidget(QtWidgets.QLabel(a), r, 0)
            f.addWidget(wa, r, 1)
            if b is not None:
                f.addWidget(QtWidgets.QLabel(b), r, 2)
                f.addWidget(wb, r, 3)
        f.setColumnStretch(1, 1)
        f.setColumnStretch(3, 1)

        # --- beam current (condenser)
        g, f = grid("Beam current (condenser)")
        g.setToolTip("Raw 0–99 settings (the remote protocol reports no calibrated probe current; "
                     "watch the EDS input count rate to judge the current).")
        self.beam_c = _spin(0, 99, 50)
        self.beam_c.valueChanged.connect(lambda v: self._user(sem.set_beam, "C", v))
        self.beam_f = _spin(0, 99, 50)
        self.beam_f.valueChanged.connect(lambda v: self._user(sem.set_beam, "F", v))
        b = QtWidgets.QPushButton("Presets…")
        b.setToolTip("save and recall named settings: beam coarse / fine, stigmator, contrast / brightness")
        b.clicked.connect(self.open_presets)
        self.b_aga = QtWidgets.QPushButton("Auto gun alignment")
        self.b_aga.setToolTip("AGA ON: the SEM's automatic gun alignment (holds the port until it ends)")
        #: what the button runs instead (the main window: the SEM tab's, which hands the scan to the SEM)
        self.aga_handler = None
        self.b_aga.clicked.connect(lambda: self.aga_handler() if self.aga_handler is not None else self._confirmed(
            "Run the SEM's automatic gun alignment? It takes the port until the SEM has finished.",
            lambda: sem.alignment("gun"), "auto gun alignment"))
        f.addWidget(QtWidgets.QLabel("Coarse"), 0, 0)
        f.addWidget(self.beam_c, 0, 1)
        f.addWidget(QtWidgets.QLabel("Fine"), 0, 2)
        f.addWidget(self.beam_f, 0, 3)
        f.addWidget(b, 1, 0, 1, 2)
        f.addWidget(self.b_aga, 1, 2, 1, 2)
        f.setColumnStretch(1, 1)
        f.setColumnStretch(3, 1)

        # --- SEM display
        g, f = grid("SEM display && detector")
        self.cont = _spin(0, 4095, 2048, 16)
        self.cont.valueChanged.connect(lambda v: self._user(sem.set_contrast, v))
        self.brt = _spin(0, 4095, 2048, 16)
        self.brt.valueChanged.connect(lambda v: self._user(sem.set_brightness, v))
        self.scan_combo = QtWidgets.QComboBox()
        self.scan_combo.addItem("unknown", None)  # until the first read
        for k, v in SCAN_MODES.items():
            self.scan_combo.addItem(v, k)
        self.scan_combo.setToolTip("the SEM's own scan mode (its display), read back from the SEM")
        self.scan_combo.activated.connect(
            lambda i: self.scan_combo.itemData(i) is not None and self._call(sem.set_scan, self.scan_combo.itemData(i)))
        self.det_combo = QtWidgets.QComboBox()
        self.det_combo.addItems(list(C.SIGNAL_TOKENS))
        self.det_combo.activated.connect(lambda i: self._call(sem.set_detector, self.det_combo.itemText(i)))
        self.b_run = QtWidgets.QPushButton("Freeze")
        self.b_run.setCheckable(True)
        self.b_run.setToolTip("RUN OFF / RUN ON – freezes the SEM's own display (does not blank the beam)")
        self.b_run.toggled.connect(lambda on: self._user(sem.set_running, not on))
        self.sem_poll = QtWidgets.QCheckBox("Read in background")
        self.sem_poll.setChecked(sem.polling)
        self.sem_poll.setToolTip("Reads the SEM settings one after another without pause (MAG / HV most "
                                 "often), so changes on the SEM panel reach imaging and EDS")
        self.sem_poll.toggled.connect(self._sem_poll_changed)
        f.addWidget(QtWidgets.QLabel("Contrast"), 0, 0)
        f.addWidget(self.cont, 0, 1)
        f.addWidget(QtWidgets.QLabel("Bright."), 0, 2)
        f.addWidget(self.brt, 0, 3)
        f.addWidget(QtWidgets.QLabel("Scan"), 1, 0)
        f.addWidget(self.scan_combo, 1, 1)
        f.addWidget(QtWidgets.QLabel("Detector"), 1, 2)
        f.addWidget(self.det_combo, 1, 3)
        f.addWidget(self.b_run, 2, 0, 1, 2)
        f.addWidget(self.sem_poll, 2, 2, 1, 2)
        f.setColumnStretch(1, 1)
        f.setColumnStretch(3, 1)
        lay.addStretch(1)
        return w

    # ================================================================ stage panel
    def _stage_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        st = self.stage
        lay.setSpacing(4)
        self.b_stop = QtWidgets.QPushButton("STOP ALL AXES")
        self.b_stop.setMinimumHeight(32)
        self.b_stop.setStyleSheet("QPushButton{background:#c62828;color:white;font-weight:bold}"
                                  "QPushButton:pressed{background:#8e0000}")
        self.b_stop.setToolTip("S0 to every axis at once – also while a SEM command holds the port (Esc)")
        self.b_stop.clicked.connect(self.stop_all)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.stop_all)
        lay.addWidget(self.b_stop)

        # --- DRO
        g = QtWidgets.QGroupBox("Position")
        g.setToolTip("double-click an axis to move it")
        grid = QtWidgets.QGridLayout(g)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setVerticalSpacing(2)
        self.dro, self.dro_state = {}, {}  # dro_state: the axis's motion state (the value's colour)
        rows = (("X", "Y", "Z"), ("R", "T"))  # several axes per row: compact
        cell = {a: (i, j) for i, row in enumerate(rows) for j, a in enumerate(row)}
        for a in AXES:
            i, j = cell[a]
            name = QtWidgets.QLabel(a)
            name.setStyleSheet("font-weight:bold;color:#9cf")
            val = DROLabel("—")
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            fv = QtGui.QFont(val.font())  # the size of every other text
            fv.setFamily("Consolas")
            fv.setBold(True)
            val.setFont(fv)
            val.setStyleSheet(_DRO_STYLE % "#7f7")
            val.setMinimumWidth(70)
            val.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            val.setToolTip(f"double-click: move {a}")
            val.double_clicked.connect(lambda a=a: self.open_move_dialog(a))
            unit = QtWidgets.QLabel(st.unit(a))
            unit.setMinimumWidth(18)
            grid.addWidget(name, i, 3 * j)
            grid.addWidget(val, i, 3 * j + 1)
            grid.addWidget(unit, i, 3 * j + 2)
            self.dro[a], self.dro_state[a] = val, "—"
        for j in range(3):
            grid.setColumnStretch(3 * j + 1, 1)
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
        grid.addLayout(r, len(rows), 0, 1, 9)
        lay.addWidget(g)

        # --- jog
        self.g_jog = QtWidgets.QGroupBox("Jog")
        f = QtWidgets.QGridLayout(self.g_jog)
        f.setContentsMargins(6, 4, 6, 4)
        f.setSpacing(3)
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
            b.setFixedSize(32, 26)
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
        f.setColumnMinimumWidth(3, 10)
        steps = QtWidgets.QFormLayout()
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setVerticalSpacing(2)
        steps.addRow("Step", self.jog_um)
        steps.addRow("Step °", self.jog_deg)
        steps.addRow("X/Y approach", self.backlash)
        f.addLayout(steps, 0, 7, 3, 1)
        f.setColumnStretch(8, 1)
        self.g_jog.setToolTip("During a move the port carries nothing else (SEM settings wait until it ends) "
                              "and a second move is refused; STOP never waits.  The stage has no homing or "
                              "collision model – watch the chamber camera when moving Z, R or T.")
        lay.addWidget(self.g_jog)
        lay.addWidget(self._positions_box(), 1)
        self._show_move_state()
        return w

    def _field_size_box(self):
        """The exposure field size (the Lithography's list, shared settings): selecting one sets its
        magnification, the stage Z to 1000 × its nominal WD (µm) and the focus to the WD1 recorded
        for it on the calibration sample (the nominal WD when none is).  A schedule runs only at
        the field size selected here."""
        g = QtWidgets.QGroupBox("Exposure field size")
        f = QtWidgets.QHBoxLayout(g)
        f.addWidget(self._field_size_combo(), 1)
        return g

    def _field_size_combo(self):
        """The field size selector (see :meth:`_field_size_box`)."""
        from ..litho.module.tabs import PositionsHub
        from ..litho.settings import LithoSettings

        self.litho_settings = LithoSettings.shared()
        self.field_size = QtWidgets.QComboBox()
        self.field_size.setToolTip("sets the magnification, the stage Z to the nominal WD and the focus to the WD1 "
                                   "recorded for it (else the nominal WD); tune either by hand after.  The Lithography runs a schedule only at the field size "
                                   "selected here (Lithography → Calibration → Field sizes… edits the list)")
        self.field_size.activated.connect(lambda i: self.select_field_size(self.field_size.itemData(i)))
        self._fill_field_sizes()
        PositionsHub.instance().changed.connect(lambda st: st is self.litho_settings and self._fill_field_sizes())
        return self.field_size

    def _fill_field_sizes(self):
        st = self.litho_settings
        self.field_size.blockSignals(True)
        self.field_size.clear()
        self.field_size.addItem("(none)", "")
        for fs in st.field_sizes:
            self.field_size.addItem(fs.describe(), fs.name)
        self.field_size.setCurrentIndex(max(0, self.field_size.findData(st.field_size)))
        self.field_size.blockSignals(False)

    def select_field_size(self, name: str, move_z: bool = True) -> bool:
        """Select field size *name* ("" none): its magnification, the stage Z to 1000 × its nominal
        WD µm (*move_z*; the move asks first when the window confirms moves) and the focus to its
        WD1 (recorded on the calibration sample; else the nominal WD)."""
        from ..litho.module.tabs import PositionsHub

        st = self.litho_settings
        fs = st.find_field_size(name) if name else None
        if name and fs is None:
            self.notify(f"no field size named {name!r}")
            return False
        st.field_size = fs.name if fs is not None else ""
        try:
            st.save()
        except OSError as e:
            self.notify(f"litho settings not saved: {e}")
        self._fill_field_sizes()
        PositionsHub.instance().changed.emit(st)
        if fs is None:
            self.notify("field size: none")
            return True
        wd1 = st.wd1(fs.name)
        wd = fs.wd_mm if wd1 is None else wd1
        self.notify(f"field size {fs.describe()}; the focus to " + (f"WD1 {wd1:g} mm" if wd1 is not None else
                                                                     f"the nominal {wd:g} mm (no WD1 recorded)"))
        self._call(self.sem.set_magnification, fs.magnification, what=f"magnification ×{fs.magnification}")
        self._call(self.sem.set_working_distance, wd, what=f"working distance {wd:g} mm")
        if move_z:
            self.move({"Z": 1000.0 * fs.wd_mm}, relative=False)
        return True

    def _positions_box(self):
        """The Lithography window's Stage Positions (the same list, shared settings): X, Y and ΔZ
        (against the calibration sample, whose Z_cal and WD1 Record WD takes); Go to XYZ sets Z_cal +
        ΔZ and the focus to WD1."""
        from ..litho.module.tabs import StagePositionsTab
        from ..litho.settings import LithoSettings

        g = QtWidgets.QGroupBox("Stage positions")
        v = QtWidgets.QVBoxLayout(g)
        v.setContentsMargins(4, 4, 4, 4)

        def pos():
            x, y = self.stage.positions.get("X"), self.stage.positions.get("Y")
            return None if x is None or y is None else (float(x), float(y))

        def z():
            val = self.stage.positions.get("Z")
            return None if val is None else float(val)

        self.positions_tab = StagePositionsTab(LithoSettings.shared(), pos, z, self.sem.working_distance_mm)
        self.positions_tab.table.setMinimumHeight(110)
        self.positions_tab.gotoRequested.connect(self.goto_position)
        v.addWidget(self.positions_tab)
        return g

    def goto_position(self, x: float, y: float, z=None, wd=None):
        """Move to (x, y) and Z *z* (None: Z stays); the focus to *wd* mm (None: stays)."""
        targets = {"X": float(x), "Y": float(y)}
        if z is not None:
            targets["Z"] = float(z)
        fut = self.move(targets, relative=False)
        if fut is not None and wd is not None:
            self._call(self.sem.set_working_distance, float(wd), what=f"working distance {float(wd):g} mm")
        return fut

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

    def _hv_value(self) -> float:
        """The box's voltage in the S-3000's steps (0.01 kV below 10 kV, 0.1 kV above)."""
        v = self.hv_set.value()
        return round(v, 2 if v < 10 else 1)

    def apply_hv(self):
        """The voltage box (Enter / leaving it): while HV is on, sent to the SEM (in its steps);
        while off, only kept for the next HV ON."""
        v = self._hv_value()
        if abs(self.hv_set.value() - v) > 1e-9:
            self.hv_set.setValue(v)
        s = self.sem.state
        if not s.get("hv_on") or (s.get("hv_kv") is not None and abs(s["hv_kv"] - v) < 1e-6):
            return None
        return self._call(self.sem.set_hv, f"{v:.2f}" if v < 10 else f"{v:.1f}", what="HV")

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
        self.stage.stop()
        self.notify("STOP sent to all axes")

    @property
    def motion_enabled(self) -> bool:
        """False while a move is in progress (a second one would be refused)."""
        return not self.stage.busy

    def _show_move_state(self):
        busy = self.stage.busy
        self.g_jog.setEnabled(not busy)

    def open_move_dialog(self, axis: str):
        """Direct move of one axis (double-click on its readout)."""
        if not self.motion_enabled:
            self.notify("a stage move is in progress – wait for it to end (or STOP)")
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
            self.notify("a stage move is in progress – wait for it to end (or STOP)")
            return None
        desc = ", ".join(f"{a} {'by ' if relative else 'to '}{v:+g} {self.stage.unit(a)}" if relative else
                         f"{a} to {v:g} {self.stage.unit(a)}" for a, v in targets.items())
        if (self.confirm if confirm is None else confirm) and QtWidgets.QMessageBox.question(
                self, "Move stage", f"Move {desc}?") != QtWidgets.QMessageBox.Yes:
            return None
        fut = self._call(self.stage.move, targets, relative, what=f"move {desc}")
        if fut is not None:
            self.notify(f"move: {desc}")
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
            self._hv_button(hv_on)
            emi = s.get("emission")
            self.l_emi.setText(f"emission {emi} µA" if emi is not None else "emission —")
            sp = s.get("hv_kv")  # on: the SEM's voltage; off: the box keeps the value HV ON will use
            if hv_on and sp is not None and not self.hv_set.hasFocus():
                self.hv_set.setValue(sp)
            self.l_mag.setText(_fmt_mag(s.get("mag")))
            if s.get("mag") in MAGNIFICATIONS and not self.mag_combo.view().isVisible():
                self.mag_combo.setCurrentIndex(MAGNIFICATIONS.index(s["mag"]))
            wd = s.get("wd_mm")
            if isinstance(wd, float) and abs(self.wd.value() - wd) > 5e-4 and not self.wd.hasFocus():
                self.wd.blockSignals(True)
                self.wd.setValue(wd)
                self.wd.blockSignals(False)
            for box, key in ((self.focus_c, "focus_coarse"), (self.focus_f, "focus_fine"),
                             (self.stig_x, "stig_x"), (self.stig_y, "stig_y"), (self.beam_c, "beam_coarse"),
                             (self.beam_f, "beam_fine"), (self.cont, "contrast"), (self.brt, "brightness")):
                v = s.get(key)
                if isinstance(v, int) and box.value() != v and not box.hasFocus():
                    box.setValue(v)
            if s.get("running") is not None:
                self.b_run.setChecked(not s["running"])
                self.b_run.setText("Frozen – Run" if not s["running"] else "Freeze")
            if s.get("scan") in SCAN_MODES:
                self.scan_combo.setCurrentIndex(self.scan_combo.findData(s["scan"]))
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
            if state != self.dro_state[a]:  # the motion state: the value's colour, in its tooltip
                self.dro_state[a] = state
                self.dro[a].setStyleSheet(_DRO_STYLE % _STATE_COLOR.get(state, "#7f7"))
                self.dro[a].setToolTip(f"{a}: {state} – double-click: move {a}")
        self._show_move_state()
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
        if st.busy or st.moving:
            return "stage moving – background reads paused"
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
