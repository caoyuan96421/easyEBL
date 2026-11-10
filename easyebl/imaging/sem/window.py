"""The SEM main window."""
from __future__ import annotations

import os
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..engine import SATURATION_WARN, EngineState, ImagingEngine
from ..export import auto_levels, save_frame
from .advanced import AdvancedDialog, SaveSettings
from .display import ContrastBrightness, format_length
from .optics import MicroscopeControl, StubMicroscope
from .panels import BeamShiftPanel, CapturePanel, ColumnPanel, ImagePanel, ScanPanel
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

    REFRESH_MS = 16  # display poll (≈60 Hz); frames are shown as soon as they are published

    def __init__(
        self,
        engine: ImagingEngine,
        microscope: MicroscopeControl | None = None,
        owns_engine: bool = True,
        simulated: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.owns_engine = owns_engine
        self.simulated = simulated
        self.mic = microscope or StubMicroscope()
        self.save_settings = SaveSettings(folder=os.path.join(os.getcwd(), "captures"))
        self.cb = ContrastBrightness(ai_range_v=self._ai_range())
        self._last_seq = 0
        self._capture_future = None
        self._acb_pending = True  # auto contrast on the first image
        self._adv: AdvancedDialog | None = None
        self._relay = None
        self._disp_times: list = []
        self._sub = engine.subscribe(maxlen=200)

        self.view = SEMImageView()
        dc = engine.daq_config
        self._sync_detector_label()
        self.setCentralWidget(self.view)
        self.scan = ScanPanel(engine, self.notify)
        self.capture = CapturePanel()
        self.image = ImagePanel(engine, self.cb, self.notify)
        self.shift = BeamShiftPanel(engine, self.notify)
        self.column = ColumnPanel(self.mic, self.notify)
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(4, 4, 4, 4)
        for w in (self.scan, self.capture, self.image, self.shift, self.column):
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
        self.view.wheelSteps.connect(lambda n: self.column._call(lambda: self.mic.step_magnification(n)))
        self.view.pointerMoved.connect(self._pointer)

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
        if dc.phantom_physical:
            self.notify(f"Dummy-image test: phantom chart on {dc.phantom_output} ({dc.phantom_physical}) – "
                        f"wire it to the video input ({dc.ai_channels.get('video', 'ai0')}).")
        self._update_status()

    # ------------------------------------------------------------------- UI
    def _make_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("main")
        tb.setMovable(False)
        self.cb_ext = QtWidgets.QCheckBox("External scan")
        self.cb_ext.setToolTip("On: the SEM takes its scan from this program (HAL external control). "
                               "Off: the SEM's own scan and controls work as usual.")
        self.cb_ext.setVisible(self.engine.handle.has_external_control)
        self.cb_ext.toggled.connect(self.set_external_scan)
        tb.addWidget(self.cb_ext)
        tb.addSeparator()
        a = tb.addAction("Crosshair")
        a.setCheckable(True)
        a.toggled.connect(lambda on: (setattr(self.view, "show_crosshair", on), self.view.update()))
        a = tb.addAction("Scan line")
        a.setCheckable(True)
        a.setChecked(True)
        a.toggled.connect(lambda on: setattr(self.view, "show_scan_line", on))
        tb.addSeparator()
        tb.addAction("Advanced…", self.open_advanced)
        tb.addAction("Full screen", lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())

    def _make_status(self):
        sb = self.statusBar()
        self.l_state = QtWidgets.QLabel()
        self.l_fps = QtWidgets.QLabel()
        self.l_buf = QtWidgets.QLabel()
        self.l_pix = QtWidgets.QLabel()
        self.l_ptr = QtWidgets.QLabel()
        for w in (self.l_state, self.l_fps, self.l_buf, self.l_pix):
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
        sc("Esc", lambda: self.engine.stop())
        for key, dx, dy in (("Shift+Left", -1, 0), ("Shift+Right", 1, 0), ("Shift+Up", 0, 1), ("Shift+Down", 0, -1)):
            sc(key, lambda dx=dx, dy=dy: self.shift.step(dx, dy))

    # ---------------------------------------------------------------- helpers
    def _ai_range(self) -> float:
        return self.engine.options.ai_range_v or self.engine.daq_config.ai_range_v

    def notify(self, msg: str) -> None:
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def levels(self):
        return self.cb.levels()

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
                if "phantom" in ev.data:
                    self._sync_detector_label()
                    self.notify(ev.message)
                self.scan.sync()
                self.shift.sync()
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

    def auto_contrast(self, frame=None, quiet: bool = False):
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
        s = self.engine.status()
        col = STATE_COLORS.get(s.state, "#ccc")
        self.l_state.setText(f"<b style='color:{col}'>{s.state.value.upper()}</b>")
        self.scan.sync_state(s.state)
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
        if self._relay is not None or self.cb_ext.isChecked():
            ext = self.engine.handle.manager.external_control
            if bool(ext) != self.cb_ext.isChecked() and ext is not None:
                self.cb_ext.blockSignals(True)
                self.cb_ext.setChecked(bool(ext))
                self.cb_ext.blockSignals(False)
        self.l_buf.setText(f"{s.rate / 1e6:g} MS/s · buffer {s.latency_s * 1e3:.0f} ms{health}{warn}{stall}")
        c = self.engine.config
        self.l_pix.setText(f"{c.nx}×{c.ny} · field {format_length(c.width_um)} · "
                           f"pixel {format_length(c.pixel_size_um[0])}")
        self.column.sync()

    def _pointer(self, xy):
        self.l_ptr.setText("" if xy is None else f"x {xy[0]:+.3f} µm, y {xy[1]:+.3f} µm")

    # ---------------------------------------------------------------- actions
    def _center_at(self, x, y):
        try:
            self.engine.set_beam_shift(x, y)
            self.shift.sync()
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def _drag_shift(self, dx, dy):
        try:
            self.engine.shift_beam(dx, dy)
            self.shift.sync()
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
            cfg = self.engine.config.with_(nx=req["nx"], ny=req["ny"], height_um=None, dwell_s=req["dwell_s"])
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
            files = save_frame(frame, stem, levels=self.levels(), ai_range_v=self._ai_range(), formats=s.formats)
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

    def open_advanced(self):
        if self._adv is None:
            self._adv = AdvancedDialog(self.engine, self.save_settings, self)
            self._adv.applied.connect(self._after_advanced)
        self._adv.load()
        self._adv.show()
        self._adv.raise_()

    def _sync_detector_label(self):
        ph = self.engine.handle.phantom
        if ph["physical"]:
            self.view.info["detector"] = f"PHANTOM ({ph['output']} = {ph['physical']})"
        else:
            self.view.info.pop("detector", None)
        self.view.update()

    def _after_advanced(self):
        self._sync_detector_label()
        self.cb.ai_range_v = self._ai_range()
        self.scan.sync()
        self.shift.sync()
        self.image.sync()

    # ------------------------------------------------------- external scan relay
    def set_external_scan(self, on: bool):
        """Switch the SEM between its internal scan and this program (HAL external control)."""
        mgr = self.engine.handle.manager
        was = self.engine.is_scanning
        try:
            if was:  # the switch is refused while a stream runs
                self.engine.stop()
                self.engine.wait_idle(5.0)
            if self._relay is None:
                self._relay = mgr.get_handle("external_scan")
            self._relay.set_external_control(on)
            if not on:
                self._relay.release()
            self.notify("external scan ON (SEM scans from the DAQ)" if on else "external scan OFF (SEM internal scan)")
            if was and on:
                self.engine.run()
        except Exception as e:  # noqa: BLE001
            self.notify(f"scan control error: {e}")
        self.cb_ext.blockSignals(True)
        self.cb_ext.setChecked(bool(mgr.external_control))
        self.cb_ext.blockSignals(False)

    # ------------------------------------------------------------------ close
    def closeEvent(self, ev):  # noqa: N802
        self.timer.stop()
        self._status_timer.stop()
        try:
            self.engine.stop()
            self.engine.wait_idle(5.0)
            if self._relay is not None and self._relay.external_control:
                self.set_external_scan(False)
        finally:
            self._sub.close()
            if self.owns_engine:
                self.engine.close()
            if self._relay is not None:
                self._relay.close()
        super().closeEvent(ev)
