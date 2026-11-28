"""The SEM main window."""
from __future__ import annotations

import math
import os
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..engine import SATURATION_WARN, EngineState, ImagingEngine, speed_by_name, speed_of
from ..export import auto_levels, save_frame
from .advanced import AdvancedDialog, SaveSettings
from .display import ContrastBrightness, format_length
from ...microscope.control import MicroscopeControl, StubMicroscope
from .panels import CapturePanel, ColumnPanel, FieldOfViewPanel, ImagePanel, ScanPanel
from ...gui.icons import icon, tool_bar
from .view import SEMImageView, render_annotated

__all__ = ["SEMWindow", "apply_dark_theme"]

STATE_COLORS = {
    EngineState.IDLE: "#888",
    EngineState.STARTING: "#fb0",
    EngineState.LIVE: "#4c4",
    EngineState.FREEZING: "#f90",
    EngineState.SINGLE: "#4af",
    EngineState.CAPTURE: "#4af",
    EngineState.FROZEN: "#6af",
    EngineState.ERROR: "#f55",
    EngineState.DISCONNECTED: "#f55",
    EngineState.PAUSED: "#fb4",
    EngineState.CLOSED: "#888",
}


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    base, alt, text = QtGui.QColor(37, 39, 43), QtGui.QColor(48, 50, 55), QtGui.QColor(225, 225, 225)
    for role, col in [
        (QtGui.QPalette.Window, alt),
        (QtGui.QPalette.WindowText, text),
        (QtGui.QPalette.Base, base),
        (QtGui.QPalette.AlternateBase, alt),
        (QtGui.QPalette.Text, text),
        (QtGui.QPalette.Button, QtGui.QColor(58, 61, 67)),
        (QtGui.QPalette.ButtonText, text),
        (QtGui.QPalette.ToolTipBase, base),
        (QtGui.QPalette.ToolTipText, text),
        (QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218)),
        (QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255)),
    ]:
        pal.setColor(role, col)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(120, 120, 120))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(120, 120, 120))
    app.setPalette(pal)
    app.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #555;border-radius:4px;margin-top:8px}"
                      "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px}")


class SEMWindow(QtWidgets.QMainWindow):
    """Commercial-SEM-style front end for :class:`ImagingEngine`.

    The window never talks to the DAQ directly: commands go through the
    engine's thread-safe API, images are polled from its frame slot by a
    display timer (fixed refresh, so a busy GUI can never back up the scan).
    """

    #: an automatic function of the SEM ended (on any thread): (kind, message, ok)
    autoFinished = QtCore.Signal(str, str, bool)

    REFRESH_MS = 16  # display poll (≈60 Hz); frames are shown as soon as they are published

    def __init__(
        self,
        engine: ImagingEngine,
        microscope: MicroscopeControl | None = None,
        owns_engine: bool = True,
        simulated: bool = False,
        microscope_bundle=None,
    ):
        super().__init__()
        self.engine = engine
        self.owns_engine = owns_engine
        self.simulated = simulated
        # a MicroscopeBundle (easyebl.microscope.bundle) gives the stage, the microscope window and the
        # magnification ↔ field coupling; a bare MicroscopeControl only the column
        self.bundle = microscope_bundle
        #: stage axes against the image (Hardware → Configure… → Microscope): moving the stage to +X
        #: moves the image of the sample to +x when mirrored (S-3000: X yes, Y no) – for Center with stage
        self.stage_mirror = (True, False)
        self.mic = microscope or (microscope_bundle.sem if microscope_bundle is not None else StubMicroscope())
        # the global field calibration (field = K / magnification): the bundle's, else one of our own
        self.field_scale = getattr(microscope_bundle, "field_scale", None)
        self._own_scale = self.field_scale is None
        if self._own_scale:
            from ...microscope.coupling import FieldScale

            self.field_scale = FieldScale(self.mic, [engine])
        self.save_settings = SaveSettings(folder=os.path.join(os.getcwd(), "captures"))
        self.cb = ContrastBrightness(ai_range_v=self._ai_range())
        self._last_seq = 0
        self._capture_future = None
        self._adv: AdvancedDialog | None = None
        self._acb_pending = True  # auto contrast on the first image
        self._abc = None  # the SEM's ABC in progress: {"resume": bool, "esc": bool}
        self.busy = None  # the modal window of a running automatic function (easyebl.gui.busy.BusyDialog)
        self._disp_times: list = []
        self._sub = engine.subscribe(maxlen=200)
        # Record: a video of the displayed image (recorder.py); the save path / discard questions
        # can be replaced (tests): ask_record_path(default) -> path | None, ask_discard(text) -> bool
        self.recorder = None
        self.record_fps = 10.0
        self._rec_key = self._rec_img = None
        self.ask_record_path = None
        self.ask_discard = None

        self.view = SEMImageView()
        self.pick_bar = self._make_pick_bar()
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.pick_bar)
        cl.addWidget(self.view, 1)
        self.setCentralWidget(central)
        self.scan = ScanPanel(engine, self.notify)
        self.capture = CapturePanel()
        self.image = ImagePanel(engine, self.cb, self.notify)
        self.fov = FieldOfViewPanel(engine, self.notify)
        self.column = ColumnPanel(self.mic, self.notify)
        # auto focus / stigmation: the backend's (shared with lithography), else one of our own
        self.autotune = getattr(microscope_bundle, "autotune", None)
        self._own_autotune = False
        if self.autotune is None and all(hasattr(self.mic, a) for a in ("set_focus", "set_stigmator", "state")):
            from ..autotune import AutoTuner

            self.autotune = AutoTuner(engine, self.mic)
            self._own_autotune = True
        self.column.set_autotune(self.autotune)
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(4, 4, 4, 4)
        for w in (self.scan, self.capture, self.image, self.fov, self.column):
            sl.addWidget(w)
        self.b_adv = QtWidgets.QPushButton("Advanced…")
        self.b_adv.clicked.connect(self.open_advanced)
        sl.addWidget(self.b_adv)
        sl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        side.setMinimumWidth(side.sizeHint().width())
        scroll.setMinimumWidth(side.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 6)
        dock = QtWidgets.QDockWidget("Controls")
        dock.setObjectName("controls")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        ldock = QtWidgets.QDockWidget("Messages")
        ldock.setObjectName("messages")
        ldock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, ldock)
        ldock.setMaximumHeight(110)

        self._make_toolbar()
        self._make_status()
        self._make_shortcuts()

        # wiring
        self.image.levelsChanged.connect(self._redraw)
        self.image.b_acb.clicked.connect(self.auto_contrast)
        self.capture.captureRequested.connect(self.start_capture)
        self.capture.saveRequested.connect(self.save_dialog)
        self.view.centerRequested.connect(self._center_at)
        self.view.shiftDragged.connect(self._drag_shift)
        self.view.zoomSteps.connect(self.fov.zoom)  # the swept range – never the column magnification
        self.view.contextRequested.connect(self._context_menu)
        self.view.pointerMoved.connect(self._pointer)
        self.view.measureChanged.connect(self._measure_changed)
        self.autoFinished.connect(self._auto_finished)
        self.column.tune_handler = lambda what: (self.auto_stigmate() if what == "stig" else
                                                 self.focus_and_stigmate() if what == "both" else self.auto_focus(what))
        self.column.tuneEvent.connect(self._tune_busy)

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.REFRESH_MS)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(250)
        cap = engine.handle.capabilities()
        self.setWindowTitle(f"easyEBL SEM – {cap.name}")
        self.notify(f"Connected to {cap.name}. Sample clock {engine.rate / 1e6:g} MS/s.")
        self._update_status()

    # ------------------------------------------------------------------- UI
    def _make_toolbar(self):
        """SEM tools (icon, text under it): scan, capture, the field, our auto functions; the overlays,
        Advanced… and Full screen."""
        tb = tool_bar("sem_tools", self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        self.tools = tb

        def act(name, text, fn, tip, checkable=False):
            a = tb.addAction(icon(name), text)
            a.setToolTip(tip)
            a.setCheckable(checkable)
            if checkable:
                a.toggled.connect(fn)
            else:
                a.triggered.connect(lambda _=False: fn())
            return a

        self.a_run = act("run", "Run", lambda: self.scan._run_freeze(),
                         "Run / Stop the live scan (F5); Stop freezes at the end of the frame, again: at once")
        self.a_single = act("single", "Single", lambda: self.scan._do(self.engine.single),
                            "scan one frame, then freeze (F6)")
        self.a_capture = act("capture", "Capture", self.capture.request, "photo scan with the Capture settings (F9)")
        self.a_record = act("record", "Record", self.toggle_record,
                            "record a video of the image as displayed (with the data bar), in real time; "
                            "stop to save it as MP4 (H.264) or uncompressed AVI.  The arrow: frame rate")
        self.record_fps_menu = m = QtWidgets.QMenu(self)
        grp = QtGui.QActionGroup(m)
        for fps in (5, 10, 25):
            a = m.addAction(f"{fps} frames/s")
            a.setCheckable(True)
            a.setChecked(fps == self.record_fps)
            grp.addAction(a)
            a.triggered.connect(lambda _=False, v=float(fps): setattr(self, "record_fps", v))
        b = tb.widgetForAction(self.a_record)
        if isinstance(b, QtWidgets.QToolButton):
            b.setMenu(m)
            b.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.a_full = act("full_fov", "Full FoV", lambda: self.fov._do(self.engine.full_field),
                          "the whole field of view, no shift or rotation (Ctrl+0)")
        tb.addSeparator()
        self.a_af = act("focus", "Autofocus", lambda: self.auto_focus("auto"),
                        "our auto focus: test images of this scan field (coarse focus only when needed)")
        b = tb.widgetForAction(self.a_af)
        if isinstance(b, QtWidgets.QToolButton):
            m = QtWidgets.QMenu(b)
            m.addAction("Fine focus only (FOCUS F)", lambda: self.auto_focus("fine"))
            m.addAction("Coarse focus only (FOCUS C)", lambda: self.auto_focus("coarse"))
            m.addAction("Focus + stigmation", self.focus_and_stigmate)
            b.setMenu(m)
            b.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.a_as = act("stig", "Auto stigmation", self.auto_stigmate, "our auto stigmation: stigmator X, then Y")
        self.a_aga = act("gun", "Auto gun align", self.auto_gun_alignment,
                         "the SEM's automatic gun alignment (AGA): the scan goes back to the SEM while it runs, "
                         "then the live scan resumes")
        self.a_acb = act("acb", "Auto C/B", self.auto_contrast_brightness,
                         "the SEM's own auto brightness / contrast (ABC): the scan goes back to the SEM "
                         "while it runs, then the live scan resumes")
        tb.addSeparator()
        self.a_cross = act("crosshair", "Crosshair",
                           lambda on: (setattr(self.view, "show_crosshair", on), self.view.update()),
                           "a cross at the image centre", True)
        self.a_line = act("scan_line", "Scan line", lambda on: setattr(self.view, "show_scan_line", on),
                          "the line being scanned", True)
        self.a_line.setChecked(True)
        self.a_inset = act("inset", "Field inset",
                           lambda on: (setattr(self.view, "show_fov_inset", on), self.view.update()),
                           "lower-left corner: the scanned region within the full field of view", True)
        self.a_inset.setChecked(True)
        tb.addSeparator()
        self.a_adv = act("advanced", "Advanced…", self.open_advanced, "scan timing, calibration, inputs, saving")
        self.a_fullscreen = act("full_screen", "Full screen",
                                lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen(),
                                "this window full screen")
        self.a_fullscreen.setObjectName("full_screen")
        self.tune_actions = (self.a_af, self.a_as)

    def _busy(self, title: str, text: str, **kw):
        """The modal window of an automatic function (the program waits until it closes)."""
        from ...gui.busy import BusyDialog

        if self.busy is not None and not self.busy.is_done:
            self.busy.finish(False)
        self.busy = BusyDialog(title, text, self, **kw).show_busy()
        return self.busy

    def _end_busy(self, ok: bool = True) -> None:
        if self.busy is not None:
            self.busy.finish(ok)
            self.busy = None

    def auto_gun_alignment(self):
        """The SEM's automatic gun alignment (``AGA ON``), like Auto C/B: our scan stops (the SEM gets
        its scan back), AGA runs until the SEM answers, then a live scan resumes.  Returns the
        thread (None: not started)."""
        return self._sem_routine(
            "aga", "Auto gun alignment", lambda: self.mic.alignment("gun"),
            "The SEM's automatic gun alignment is running on the SEM's own scan – easyEBL waits until the SEM "
            "has finished.", "Esc: the scan stays stopped afterwards")

    def _auto_finished(self, kind: str, msg: str, ok: bool) -> None:
        """GUI thread: an automatic function of the SEM ended."""
        self._end_busy(ok)
        self.notify(msg)
        if kind in ("abc", "aga"):  # the SEM's own routine ended: our scan again
            st = self._abc or {}
            self._abc = None
            if st.get("resume"):
                try:
                    self.engine.run()
                except Exception as e:  # noqa: BLE001
                    self.notify(f"{st.get('title', kind)}: the live scan did not restart: {e}")
            elif kind == "abc" and ok and not st.get("esc"):
                try:
                    self.engine.single()  # an image for Auto levels
                except Exception as e:  # noqa: BLE001
                    self.notify(f"Auto C/B: no image for Auto levels: {e}")
            if kind == "abc" and ok and not st.get("esc"):
                self._acb_pending = True  # Auto levels (the Image group) on the next image

    def _tune_busy(self, ev: dict) -> None:
        """Our auto focus / stigmation: progress in the modal window; it closes at the end."""
        b = self.busy
        if b is None or getattr(b, "kind", "") != "tune":
            return
        if ev.get("type") == "progress":
            b.set_progress(ev.get("text", ""))
        elif ev.get("type") == "done":
            r = ev.get("result")
            self._end_busy(bool(getattr(r, "ok", True)))

    def _sync_tools(self, st) -> None:
        live = st in (EngineState.LIVE, EngineState.STARTING, EngineState.FREEZING, EngineState.SINGLE,
                      EngineState.CAPTURE)
        a = self.a_run
        if live != getattr(self, "_tools_live", None):
            self._tools_live = live
            a.setText("Stop" if live else "Run")
            a.setIcon(icon("stop" if live else "run"))
        t = self.autotune
        ok = t is not None and t.available and not t.busy
        for x in self.tune_actions:
            x.setEnabled(ok)
        self.a_aga.setEnabled(self.abc_available)
        self.a_acb.setEnabled(self.abc_available)

    def _make_status(self):
        sb = self.statusBar()
        self.l_state = QtWidgets.QLabel()
        self.l_ext = QtWidgets.QLabel()
        self.l_ext.setToolTip("the SEM's scan source: switched to the DAQ while scanning, back when stopped")
        self.l_fps = QtWidgets.QLabel()
        self.l_buf = QtWidgets.QLabel()
        self.l_pix = QtWidgets.QLabel()
        self.l_timing = QtWidgets.QLabel()
        self.l_timing.setStyleSheet("color:#9aa")
        self.l_timing.setToolTip("set by the scan speed preset")
        self.l_ptr = QtWidgets.QLabel()
        self.l_meas = QtWidgets.QLabel()
        self.l_meas.setStyleSheet("color:#fd0")
        sb.addWidget(self.l_meas)
        for w in (self.l_state, self.l_ext, self.l_fps, self.l_buf, self.l_pix, self.l_timing):
            sb.addPermanentWidget(w)
            w.setContentsMargins(6, 0, 10, 0)
        sb.addPermanentWidget(self.l_ptr, 1)
        self.l_ptr.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    def _make_shortcuts(self):
        def sc(key, fn):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.setContext(QtCore.Qt.WindowShortcut)
            s.activated.connect(fn)

        sc("F5", lambda: self.scan._run_freeze())
        sc("F6", lambda: self.scan._do(self.engine.single))
        sc("F3", lambda: (self.scan.step_speed(-1)))
        sc("F4", lambda: (self.scan.step_speed(+1)))
        sc("F9", self.capture.request)
        sc("Ctrl+S", self.save_dialog)
        sc("Ctrl+A", self.auto_contrast)
        sc("Esc", lambda: self.cancel_pick() if self.view.pick_mode else self.engine.stop())
        sc("Return", lambda: self.confirm_pick() if self.view.pick_mode else None)
        sc("Enter", lambda: self.confirm_pick() if self.view.pick_mode else None)
        for key, dx, dy in (("Shift+Left", -1, 0), ("Shift+Right", 1, 0), ("Shift+Up", 0, 1), ("Shift+Down", 0, -1)):
            sc(key, lambda dx=dx, dy=dy: self.fov.step(dx, dy))
        sc("Ctrl++", lambda: self.fov.zoom(+1))
        sc("Ctrl+=", lambda: self.fov.zoom(+1))
        sc("Ctrl+-", lambda: self.fov.zoom(-1))
        sc("Ctrl+0", lambda: self.fov._do(self.engine.full_field))

    # ---------------------------------------------------------------- helpers
    def _ai_range(self) -> float:
        return self.engine.options.ai_range_v or self.engine.daq_config.ai_range_v

    def notify(self, msg: str) -> None:
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    # ---------------------------------------------------------------- display
    def _tick(self):
        f = self.engine.latest()
        if f is not None and f.seq != self._last_seq:
            self._last_seq = f.seq
            if self._acb_pending and f.rows_done > max(2, f.config.ny // 8):
                self._acb_pending = False
                self.auto_contrast(f)
            elif self.image.cb_auto.isChecked() and f.complete:
                self.auto_contrast(f, quiet=True)
            self.view.set_frame(f, self.levels())
            if f.complete:
                now = time.perf_counter()
                self._disp_times.append(now)
                del self._disp_times[:-30]
        for ev in self._sub.drain():
            if ev.kind == "state":
                self.scan.sync_state(EngineState(ev.data.get("state", "idle")))
            elif ev.kind == "settings":  # also changes made by the console / scripts
                self.scan.sync()
                self.fov.sync()
                self._sync_view_info()
                if ev.data.get("changes", {}).get("averaging") is not None:
                    self.image.avg.blockSignals(True)
                    self.image.avg.setValue(self.engine.options.averaging)
                    self.image.avg.blockSignals(False)
            if ev.kind in ("error", "warning") or (ev.kind == "state" and ev.data.get("state") == "error"):
                self.notify(f"{ev.kind.upper()}: {ev.message}")
            elif ev.kind == "info" and "DAQ connection" in ev.message:
                self.notify(ev.message)
        if self._capture_future is not None:
            self._poll_capture()

    def _redraw(self):
        f = self.view.frame()
        if f is not None:
            self.view.set_frame(f, self.levels())

    def levels(self):
        return self.cb.levels()

    # ------------------------------------------------------------- the SEM's ABC and AGA
    @property
    def abc_available(self) -> bool:
        """A routine of the SEM's own (ABC, AGA) can start: the microscope connected, none running."""
        return bool(getattr(self.mic, "connected", False)) and self._abc is None

    @property
    def abc_running(self) -> bool:
        """A routine of the SEM's own (ABC or AGA) is running."""
        return self._abc is not None

    def auto_contrast_brightness(self):
        """The SEM's own auto brightness / contrast (``ABC ON``), then Auto levels (the Image group).
        Returns the thread (None: not started); see :meth:`_sem_routine`."""
        return self._sem_routine(
            "abc", "Auto C/B", lambda: self.mic.auto_brightness_contrast(),
            "The SEM's auto brightness / contrast (ABC) is running on the SEM's own scan – easyEBL waits until "
            "the SEM has finished, then Auto levels.", "Esc: the scan stays stopped after ABC (no Auto levels)")

    def _sem_routine(self, kind: str, title: str, request, text: str, esc_text: str):
        """A routine of the SEM's own needs the SEM's scan: our scan stops (the engine gives the scan
        back to the SEM), *request* is sent – the S-3000 answers it when the routine has finished,
        which ends the request (at most 240 s) – and a live scan resumes (unless Esc meanwhile).  A
        modal window meanwhile; the SEM's routine cannot be interrupted."""
        import threading

        if not self.abc_available:
            self.notify(f"{title} needs the microscope's remote control" if self._abc is None
                        else f"{title}: the SEM's {self._abc.get('title', 'routine')} is still running")
            return None
        st = self.engine.state
        self._abc = {"kind": kind, "title": title, "esc": False,
                     "resume": st in (EngineState.LIVE, EngineState.STARTING, EngineState.FREEZING)}
        self.notify(f"{title}: the scan goes back to the SEM until it has finished")
        self._busy(title, text, esc=self.cancel_abc_resume, esc_text=esc_text)

        def job():
            msg, ok = f"{title}: done", True
            try:
                self.engine.stop()
                if hasattr(self.engine, "wait_idle"):
                    self.engine.wait_idle(10.0)
                got = request().result(240.0)
                if isinstance(got, dict) and got:
                    msg += " – " + ", ".join(f"{k} {v}" for k, v in got.items())
            except Exception as e:  # noqa: BLE001
                msg, ok = f"{title} failed: {e}", False
            self.autoFinished.emit(kind, msg, ok)

        t = threading.Thread(target=job, name=f"easyebl-sem-{kind}", daemon=True)
        self._abc_thread = t
        t.start()
        return t

    def cancel_abc_resume(self) -> None:
        """Esc during ABC / AGA: the SEM finishes its routine, but our live scan does not restart."""
        if self._abc is not None:
            self._abc["resume"] = False
            self._abc["esc"] = True

    def auto_contrast(self, frame=None, quiet: bool = False):
        """Display stretch to the image's levels (not used by the GUI: the display is direct)."""
        f = frame if frame is not None and not isinstance(frame, bool) else self.engine.latest()
        if f is None:
            return
        lo, hi = auto_levels(f.image)
        pad = 0.05 * (hi - lo)
        self.cb.ai_range_v = self._ai_range()
        self.cb.set_levels(lo - pad, hi + pad)
        self.image.sync()
        self._redraw()
        if not quiet:
            self.notify(f"auto contrast/brightness: {lo:.3g} … {hi:.3g} V")

    def _update_status(self):
        self._show_recording()
        s = self.engine.status()
        col = STATE_COLORS.get(s.state, "#ccc")
        self.l_state.setText(f"<b style='color:{col}'>{s.state.value.upper()}</b>")
        self.scan.sync_state(s.state)
        self._sync_tools(s.state)
        disp = 0.0
        ts = self._disp_times
        if len(ts) > 2 and time.perf_counter() - ts[-1] < 2.0:
            disp = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)
        self.l_fps.setText(f"{s.fps_hw:.2f} fps (display {disp:.1f}) · frame {s.frame_time_s:.3g} s"
                           if s.fps_hw else f"frame {s.frame_time_s:.3g} s")
        health = "" if s.margin_s is None else f", margin {s.margin_s * 1e3:.0f} ms"
        warn = f", <span style='color:#f90'>underruns {s.underruns}</span>" if s.underruns else ""
        stall = " <b style='color:#f55'>STALLED</b>" if s.stalled else ""
        if not s.connected:
            stall += " <b style='color:#f55'>DAQ DISCONNECTED – reconnecting…</b>"
        if s.saturation > SATURATION_WARN:
            stall += f" <b style='color:#f55'>VIDEO SATURATED {s.saturation:.0%}</b>"
        if self.engine.handle.has_external_control:
            ext = self.engine.handle.manager.external_control
            self.l_ext.setText("<b style='color:#4c4'>EXT SCAN</b>" if ext else
                               ("<span style='color:#999'>SEM scan</span>" if ext is False else "scan ?"))
        self.l_buf.setText(f"{s.rate / 1e6:g} MS/s · buffer {s.latency_s * 1e3:.0f} ms{health}{warn}{stall}")
        c = self.engine.config
        self.l_pix.setText(f"{c.nx}×{c.ny} · size {format_length(c.width_um)} · "
                           f"pixel {format_length(c.pixel_size_um[0])}")
        sp = speed_of(c)
        self.l_timing.setText(f"{sp.name if sp else 'custom'}: dwell {c.actual_dwell_s * 1e6:g} µs "
                              f"({c.samples_per_pixel} samples) · pre-line {c.preline_s * 1e6:g} µs · "
                              f"flyback {c.flyback_s * 1e6:g} µs · frame settle {c.frame_settle_s * 1e6:g} µs · "
                              f"pixel settle {c.settle_samples}")
        self.column.sync()
        self.fov.sync()
        self._sync_microscope_info()
        self._sync_view_info()

    def _sync_microscope_info(self):
        """Data bar: accelerating voltage and working distance from the microscope (when known)."""
        kv, wd = self.mic.accelerating_voltage_kv(), self.mic.working_distance_mm()
        info = self.view.info
        new_kv = f"{kv:g} kV" if kv else "-- kV"
        new_wd = f"WD {wd:.2f} mm" if wd else "WD --"
        if info.get("kv") != new_kv or info.get("wd") != new_wd:
            info["kv"], info["wd"] = new_kv, new_wd
            self.view.update()

    def _sync_view_info(self):
        """Image magnification (column × full field / scan width) and the field inset."""
        c, cal = self.engine.config, self.engine.calibration
        info = self.view.info
        new = {"column_mag": self.field_scale.magnification, "field_um": cal.field_size_um,
               "scan": ((float(c.center_um[0]), float(c.center_um[1])), c.width_um, c.height, c.rotation_rad)}
        if any(info.get(k) != v for k, v in new.items()):
            info.update(new)
            self.view.update()

    def _measure_changed(self, line):
        if line is None:
            self.l_meas.setText("")
            return
        (x0, y0), (x1, y1) = line
        dx, dy = x1 - x0, y1 - y0
        self.l_meas.setText(f"line {format_length(math.hypot(dx, dy))}  (Δx {format_length(abs(dx))}, "
                            f"Δy {format_length(abs(dy))}, {math.degrees(math.atan2(dy, dx)):+.1f}°)")

    def measured_line_um(self) -> float | None:
        """Length (µm) of the measuring line, for calibrating the field; None without one.  Raises
        ValueError when the displayed image was not taken with the present field."""
        length = self.view.measure_length_um()
        if length is None:
            return None
        f = self.view.frame()
        now = self.engine.calibration.field_size_um
        if f is not None and abs(f.calibration.field_size_um - now) > 1e-6 * now:
            raise ValueError("the image was taken at another field (magnification or calibration changed); "
                             "take a new image and redraw the line")
        return length

    def _calibrated(self, r: float):
        self.view.scale_measure(r)  # every field µm scaled by r: the line stays on its feature
        self.fov.sync()

    def _pointer(self, xy):
        self.l_ptr.setText("" if xy is None else f"x {xy[0]:+.3f} µm, y {xy[1]:+.3f} µm")

    # ---------------------------------------------------------------- pick (lithography alignment)
    def _make_pick_bar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setStyleSheet("QFrame{background:#5a1846} QLabel{color:white}")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        self.l_pick = QtWidgets.QLabel()
        self.l_pick.setWordWrap(True)
        self.b_pick_confirm = QtWidgets.QPushButton("Confirm (Enter)")
        self.b_pick_confirm.setToolTip("the scan centre (the green centre cross) is the marker position")
        self.b_pick_confirm.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold;"
                                          "padding:4px 12px}")
        self.b_pick_confirm.clicked.connect(self.confirm_pick)
        self.b_pick_reset = QtWidgets.QPushButton("Reset")
        self.b_pick_reset.setToolTip("back to the predicted position")
        self.b_pick_reset.clicked.connect(self.reset_pick)
        self.b_pick_cancel = QtWidgets.QPushButton("Cancel (Esc)")
        self.b_pick_cancel.clicked.connect(self.cancel_pick)
        h.addWidget(self.l_pick, 1)
        for b in (self.b_pick_reset, self.b_pick_cancel, self.b_pick_confirm):
            h.addWidget(b)
        bar.hide()
        return bar

    def request_pick(self, prompt: str, callback, targets=(), centre: bool = True) -> None:
        """Ask the user for a point: ``callback((x_um, y_um))`` with the field position, or
        ``callback(None)`` when cancelled (Esc, another request, window closed).  *targets* are drawn
        as magenta crosses (e.g. the predicted mark position).

        *centre* (default): every click moves the scan centre there (beam shift, clamped to the
        field) so the mark comes under the green centre cross, as often as wanted; *Confirm*
        (Enter) answers the scan centre, *Reset* goes back to the first target.  Otherwise the
        first click answers."""
        self.cancel_pick()
        self._pick_cb = callback
        self._pick_centre = bool(centre)
        self._pick_home = tuple(map(float, targets[0])) if len(targets) else tuple(self.engine.config.center_um)
        self._pick_clicks = 0
        self._pick_clamped = False
        self._pick_prompt = prompt
        self.view.pointPicked.connect(self._picked)
        hint = ("click the mark to centre it, again to refine · Enter confirms · Esc cancels" if centre
                else "click the mark · Esc cancels")
        self.view.start_pick(prompt + f"   ({hint})", targets, centre_cross=centre)
        if centre:
            self._update_pick_bar()
            self.pick_bar.show()
        self.notify("pick: " + prompt)

    def _update_pick_bar(self) -> None:
        c = self.engine.config.center_um
        dx, dy = c[0] - self._pick_home[0], c[1] - self._pick_home[1]
        txt = (f"<b>{self._pick_prompt}</b><br>scan centre ({c[0]:+.3f}, {c[1]:+.3f}) µm · "
               f"({dx:+.3f}, {dy:+.3f}) µm from the prediction · {self._pick_clicks} click(s)")
        if self._pick_clamped:
            txt += " · <span style='color:#fc6'>centre limited by the field edge</span>"
        self.l_pick.setText(txt)

    def _picked(self, x: float, y: float) -> None:
        if not getattr(self, "_pick_centre", False):
            cb = self._end_pick()
            if cb is not None:
                self.notify(f"picked ({x:+.3f}, {y:+.3f}) µm")
                cb((x, y))
            return
        try:
            c = self.engine.set_beam_shift(x, y).center_um
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
            return
        self._pick_clicks += 1
        self._pick_clamped = abs(c[0] - x) > 1e-6 or abs(c[1] - y) > 1e-6
        self.fov.sync()
        self._update_pick_bar()

    def confirm_pick(self) -> None:
        """(centring pick) Answer the scan centre."""
        if getattr(self, "_pick_cb", None) is None:
            return
        c = tuple(float(v) for v in self.engine.config.center_um)
        cb = self._end_pick()
        self.notify(f"marker at ({c[0]:+.3f}, {c[1]:+.3f}) µm")
        cb(c)

    def reset_pick(self) -> None:
        if getattr(self, "_pick_cb", None) is None:
            return
        try:
            self.engine.set_beam_shift(*self._pick_home)
            self.fov.sync()
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
        self._pick_clamped = False
        self._update_pick_bar()

    def _end_pick(self):
        cb = getattr(self, "_pick_cb", None)
        self._pick_cb = None
        try:
            self.view.pointPicked.disconnect(self._picked)
        except (RuntimeError, TypeError):
            pass
        self.view.stop_pick()
        self.pick_bar.hide()
        return cb

    def cancel_pick(self) -> None:
        cb = self._end_pick() if getattr(self, "_pick_cb", None) is not None else None
        if cb is not None:
            cb(None)

    # ---------------------------------------------------------------- context menu
    def _stage(self):
        b = self.bundle
        st = getattr(b, "stage", None) if b is not None else None
        return st if st is not None and getattr(b, "connected", False) else None

    def _context_menu(self, xy, global_pos) -> None:
        m = QtWidgets.QMenu(self)
        a = m.addAction("Center with beam")
        a.setEnabled(xy is not None)
        a.triggered.connect(lambda: self._center_at(*xy))
        a = m.addAction("Center with stage")
        a.setEnabled(xy is not None and self._stage() is not None)
        a.setToolTip("move the stage so the clicked point comes to the image centre")
        a.triggered.connect(lambda: self.center_with_stage(*xy))
        m.addSeparator()
        a = m.addAction("Clear measurement")
        a.setEnabled(self.view.measure_line() is not None)
        a.triggered.connect(self.view.clear_measure)
        m.addAction("Reset to Full FoV").triggered.connect(self.reset_full_fov)
        self._last_menu = m
        m.popup(global_pos)

    def stage_move_for(self, x: float, y: float) -> tuple:
        """The relative stage move (µm) that brings the field point (x, y) to the image centre: the
        sample moves by −M·d in the field (M: the stage mirror, stage and scan axes parallel)."""
        cx, cy = self.engine.config.center_um
        mx = -1.0 if self.stage_mirror[0] else 1.0
        my = -1.0 if self.stage_mirror[1] else 1.0
        return (mx * (x - cx), my * (y - cy))

    def center_with_stage(self, x: float, y: float) -> None:
        st = self._stage()
        if st is None:
            self.notify("no stage (microscope not connected)")
            return
        dx, dy = self.stage_move_for(x, y)
        from ...microscope.movewait import run_blocking_move

        try:
            run_blocking_move(st, {"X": dx, "Y": dy}, relative=True, parent=self)
            self.notify(f"stage moved by ({dx:+.2f}, {dy:+.2f}) µm")
        except Exception as e:  # noqa: BLE001
            self.notify(f"stage: {type(e).__name__}: {e}")

    def reset_full_fov(self) -> None:
        """The whole field of view (beam scan) and no digital zoom."""
        self.view.reset_digital_zoom()
        self.fov._do(self.engine.full_field)

    # ---------------------------------------------------------------- actions
    def _center_at(self, x, y):
        try:
            self.engine.set_beam_shift(x, y)
            self.fov.sync()
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def _drag_shift(self, dx, dy):
        try:
            self.engine.shift_beam(dx, dy)
            self.fov.sync()
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def start_capture(self, req: dict):
        if self._capture_future is not None and not self._capture_future.done():
            self.notify("a capture is already running")
            return
        try:
            frames = req.pop("frames", 1)
            self._capture_future = self.engine.capture(frames=frames, **req)
            self._capture_t0 = time.perf_counter()
            cfg = self.engine.config.with_(nx=req["nx"], ny=req["ny"], height_um=None,
                                           **speed_by_name(req["speed"]).fields())
            self._capture_eta = self.engine.frame_time_s(cfg) * frames
            self.capture.progress.setVisible(True)
            self.capture.b_capture.setEnabled(False)
            self.notify(f"capture {req['nx']}×{req['ny']}, {frames} frame(s), ≈ {self._capture_eta:.1f} s")
        except Exception as e:  # noqa: BLE001
            self.notify(f"capture not started: {e}")

    def _poll_capture(self):
        fut = self._capture_future
        el = time.perf_counter() - self._capture_t0
        self.capture.progress.setValue(int(min(1.0, el / max(self._capture_eta, 1e-3)) * 1000))
        if not fut.done():
            return
        self._capture_future = None
        self.capture.progress.setVisible(False)
        self.capture.b_capture.setEnabled(True)
        try:
            frame = fut.result()
        except Exception as e:  # noqa: BLE001
            self.notify(f"capture failed: {e}")
            return
        self.view.set_frame(frame, self.levels())
        self.notify(f"capture done ({frame.config.nx}×{frame.config.ny}, {frame.frames_averaged} frame(s))")
        if self.save_settings.auto_save:
            stem = os.path.join(self.save_settings.folder, time.strftime("sem_%Y%m%d_%H%M%S"))
            self.save_frame(frame, stem)

    def save_frame(self, frame, stem: str):
        s = self.save_settings
        try:
            meta = self.bundle.metadata() if self.bundle is not None else self.mic.snapshot()
            files = save_frame(frame, stem, levels=self.levels(), ai_range_v=self._ai_range(), formats=s.formats,
                               microscope=meta)
            if s.data_bar:
                img = render_annotated(frame, self.levels(), self.view.info)
                p = f"{stem}_databar.png"
                img.save(p)
                files.append(p)
            self.notify("saved " + ", ".join(os.path.basename(str(f)) for f in files) + f" in {os.path.dirname(stem)}")
        except Exception as e:  # noqa: BLE001
            self.notify(f"save failed: {e}")

    def save_dialog(self):
        f = self.view.frame() or self.engine.latest()
        if f is None:
            self.notify("nothing to save yet")
            return
        os.makedirs(self.save_settings.folder, exist_ok=True)
        default = os.path.join(self.save_settings.folder, time.strftime("sem_%Y%m%d_%H%M%S.tif"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", default, "Images (*.tif *.png)")
        if path:
            self.save_frame(f, os.path.splitext(path)[0])

    # ---------------------------------------------------------------- record (video)
    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def toggle_record(self):
        return self.stop_record() if self.recording else self.start_record()

    def start_record(self) -> bool:
        """Start recording the image as displayed (:mod:`.recorder`), at :attr:`record_fps`."""
        from .recorder import SEMRecorder

        if self.recording:
            return False
        self.recorder = SEMRecorder(self._record_image, self.record_fps, parent=self)
        self.recorder.start()
        self.record_fps_menu.setEnabled(False)
        self.notify(f"recording ({self.record_fps:g} frames/s) – Record again to stop and save")
        self._show_recording()
        return True

    def _record_image(self):
        """The image as displayed, with the data bar (the same image object while nothing changed)."""
        f = self.view.frame()
        if f is None:
            return None
        lv = self.levels()
        key = (id(f), f.seq, getattr(f, "rows_done", None), None if lv is None else tuple(lv),
               self.view.info.get("column_mag"))
        if key != self._rec_key:
            self._rec_key, self._rec_img = key, render_annotated(f, lv, self.view.info)
        return self._rec_img

    def _show_recording(self) -> None:
        a = getattr(self, "a_record", None)
        if a is None:
            return
        rec = self.recorder
        if rec is None:
            if a.text() != "Record":
                a.setText("Record")
                a.setIcon(icon("record"))
            return
        t = int(rec.elapsed_s)
        a.setText(f"REC {t // 60}:{t % 60:02d}")
        a.setIcon(icon("record_stop"))

    def stop_record(self, ask: bool = True) -> str | None:
        """Stop recording and save the video (asks where; MP4 or AVI by the file type); the saved
        path, or None (nothing recorded, discarded, cancelled).  *ask* False: discard."""
        from .recorder import FORMATS, write_avi, write_mp4

        rec, self.recorder = self.recorder, None
        self.record_fps_menu.setEnabled(True)
        self._rec_key = self._rec_img = None
        self._show_recording()
        if rec is None:
            return None
        fps = rec.fps
        store = rec.stop()
        rec.deleteLater()
        if store is None or store.count == 0:
            self.notify("recording stopped: no image was shown, nothing recorded")
            if store is not None:
                store.discard()
            return None
        n, dur = store.count, int(store.count / fps)
        what = f"{n} frames, {dur // 60}:{dur % 60:02d} at {fps:g} frames/s, {store.width}×{store.height}"
        try:
            path = None
            while ask and not path:
                os.makedirs(self.save_settings.folder, exist_ok=True)
                default = os.path.join(self.save_settings.folder, time.strftime("sem_%Y%m%d_%H%M%S.mp4"))
                path = self._ask_record_path(default, what)
                if not path and self._ask_discard(f"Discard the recording ({what})?"):
                    break
            if not path:
                self.notify(f"recording discarded ({what})")
                return None
            ext = os.path.splitext(path)[1].lower()
            if ext not in FORMATS.values():
                path, ext = path + ".mp4", ".mp4"
            prog = QtWidgets.QProgressDialog(f"Saving {os.path.basename(path)}…", "Cancel", 0, n, self)
            prog.setWindowTitle("Save the recording")
            prog.setWindowModality(QtCore.Qt.WindowModal)
            prog.setMinimumDuration(300)

            def progress(done, total):
                prog.setValue(done)
                QtWidgets.QApplication.processEvents()
                return not prog.wasCanceled()

            try:
                ok = (write_avi if ext == ".avi" else write_mp4)(path, store, fps, progress)
            except Exception as e:  # noqa: BLE001
                self.notify(f"saving the recording failed: {e}")
                QtWidgets.QMessageBox.warning(self, "Record", f"Saving the recording failed:\n{e}")
                return None
            finally:
                prog.close()
            if not ok:
                self.notify("saving the recording cancelled")
                return None
            self.notify(f"saved recording {os.path.basename(path)} ({what}) in {os.path.dirname(path)}")
            return path
        finally:
            store.discard()

    def _ask_record_path(self, default: str, what: str) -> str | None:
        from .recorder import FORMATS

        if self.ask_record_path is not None:
            return self.ask_record_path(default)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Save the recording ({what})", default,
                                                        ";;".join(FORMATS))
        return path or None

    def _ask_discard(self, text: str) -> bool:
        if self.ask_discard is not None:
            return self.ask_discard(text)
        return QtWidgets.QMessageBox.question(self, "Record", text) == QtWidgets.QMessageBox.Yes

    def open_advanced(self):
        if self._adv is None:
            self._adv = AdvancedDialog(self.engine, self.save_settings, self, field_scale=self.field_scale,
                                       measured=self.measured_line_um, autotune=self.autotune)
            self._adv.applied.connect(self._after_advanced)
            self._adv.calibrated.connect(self._calibrated)
        self._adv.load()
        self._adv.show()
        self._adv.raise_()

    def _after_advanced(self):
        self.cb.ai_range_v = self._ai_range()
        self.scan.sync()
        self.fov.sync()
        self.image.sync()

    # ------------------------------------------------------------------ close
    # ------------------------------------------------------------- auto focus / stigmation
    def _tune(self, what: str, title: str):
        fut = self.column.start_tune(what)
        if fut is not None:
            b = self._busy(title, f"{title}: test images of this scan field – easyEBL waits until it has "
                                  f"finished.  Cancel (Esc) sets the start values again.",
                           cancel=self.column.cancel_tune)
            b.kind = "tune"
            if fut.done():  # (ended before the window was there)
                self._end_busy(fut.exception() is None and bool(getattr(fut.result(), "ok", True)))
        return fut

    def auto_focus(self, mode: str = "auto"):
        """Our auto focus (``"auto"``, ``"fine"``, ``"coarse"``) in a modal window with progress and
        Cancel; returns a future of :class:`~easyebl.imaging.autotune.AutoTuneResult` (None if not
        started)."""
        return self._tune(mode, "Auto focus" + ("" if mode == "auto" else f" ({mode})"))

    def auto_stigmate(self):
        return self._tune("stig", "Auto stigmation")

    def focus_and_stigmate(self):
        return self._tune("both", "Auto focus + stigmation")

    def closeEvent(self, ev):  # noqa: N802
        self.cancel_pick()
        if self.recording:  # offer to save what was recorded
            self.stop_record()
        tuner = self.autotune
        self.column.set_autotune(None)
        if tuner is not None and self._own_autotune:
            tuner.close()
        if self._own_scale:
            self.field_scale.close()
        self.timer.stop()
        self._status_timer.stop()
        try:
            self.engine.stop()  # the engine gives the scan back to the SEM when it stops
            self.engine.wait_idle(5.0)
        finally:
            self._sub.close()
            if self.owns_engine:
                self.engine.close()
        super().closeEvent(ev)
