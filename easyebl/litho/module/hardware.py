"""The litho runner's hardware on the easyEBL backend, and the GUI-thread bridge.

:class:`BackendHardware` drives the shared backend:

* stage – :class:`~easyebl.microscope.stage.StageControl` (X, Y, Z; every move behind the modal wait dialog);
* magnification – the SEM (the field follows through the global field calibration);
* beam condition – the microscope module's beam presets (never HV, never the beam itself);
* focus by hand – a prompt that does not block the other windows, with the SEM window scanning
  live (before a beam calibration);
* marks – the SEM imaging window: the scan zooms onto the predicted mark (beam shift + scan
  width, the column is not touched); each click centres the scan there, *Confirm* takes the scan
  centre (centring pick mode);
* exposure – :class:`~easyebl.litho.writer.LithoWriter` on its own DAQ handle (the imaging
  engine is stopped first, so it does not scan over the resist).

Interactive steps run on the GUI thread through :class:`GuiBridge`; the runner thread waits.
"""
from __future__ import annotations

import concurrent.futures
import threading

from PySide6 import QtCore, QtWidgets

from ..runner import LithoHardware

__all__ = ["GuiBridge", "BackendHardware", "FocusPrompt"]


class FocusPrompt(QtWidgets.QDialog):
    """Not modal (the SEM and Microscope windows stay usable), on top: *Continue* / *Abort*.
    ``answered(bool)`` once; closing it is Abort."""

    answered = QtCore.Signal(bool)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lithography: focus and stigmate")
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self._done = False
        lab = QtWidgets.QLabel(text)
        lab.setWordWrap(True)
        self.text = lab
        self.b_continue = QtWidgets.QPushButton("Continue")
        self.b_continue.setDefault(True)
        self.b_abort = QtWidgets.QPushButton("Abort")
        self.b_continue.clicked.connect(lambda: self.finish(True))
        self.b_abort.clicked.connect(lambda: self.finish(False))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.b_abort)
        row.addWidget(self.b_continue)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(lab)
        lay.addLayout(row)
        self.resize(440, 160)

    def finish(self, ok: bool) -> None:
        if not self._done:
            self._done = True
            self.answered.emit(bool(ok))
        self.hide()
        self.deleteLater()

    def reject(self):  # Esc / closing
        self.finish(False)

    def closeEvent(self, ev):  # noqa: N802
        self.finish(False)
        ev.accept()


class GuiBridge(QtCore.QObject):
    """Run ``fn(future)`` on the GUI thread (it resolves the future, now or later) and wait for it."""

    _call = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._call.connect(self._run, QtCore.Qt.QueuedConnection)
        self.cancelled = threading.Event()

    def _run(self, item):
        fn, fut = item
        try:
            fn(fut)
        except Exception as e:  # noqa: BLE001 - handed to the waiting thread
            if not fut.done():
                fut.set_exception(e)

    def call(self, fn, timeout: float | None = None, on_cancel=None):
        """Blocking (not on the GUI thread).  Returns the future's result, or None when
        :attr:`cancelled` is set while waiting (then ``on_cancel()`` runs on the GUI thread)."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        if QtCore.QThread.currentThread() is self.thread():
            self._run((fn, fut))
            return fut.result(timeout)
        self._call.emit((fn, fut))
        waited = 0.0
        while True:
            try:
                return fut.result(0.1)
            except concurrent.futures.TimeoutError:
                waited += 0.1
                if self.cancelled.is_set():
                    if on_cancel is not None:
                        self._call.emit((lambda f: (on_cancel(), f.set_result(None)), concurrent.futures.Future()))
                    return None
                if timeout is not None and waited >= timeout:
                    raise


class BackendHardware(LithoHardware):
    def __init__(self, engine, manager, bundle, bridge: GuiBridge, open_sem=None, parent_widget=None,
                 owner: str = "litho", progress=None, axes=None):
        from ..coords import S3000_AXES

        self.axes = axes or S3000_AXES  # Hardware → Configure… → Microscope
        self.engine = engine
        self.manager = manager
        self.bundle = bundle
        self.bridge = bridge
        self.open_sem = open_sem  # () -> SEMWindow
        self.parent = parent_widget
        self.progress = progress
        self.handle = manager.get_handle(owner) if manager is not None else None
        self.status = lambda text: None  # set by the window (what the run waits for)
        self.move_dialog = None  # the "Stage moving" dialog while a programmed move runs
        self.move_dialogs = 0  # how many were shown
        self.focus_prompt = None  # the "focus and stigmate" prompt while it waits

    # ---------------------------------------------------------------- column
    def _autotune(self):
        t = getattr(self.bundle, "autotune", None) if self.bundle is not None else None
        if t is None:
            raise RuntimeError("auto focus / stigmation needs the microscope and the imaging engine")
        return t

    def auto_focus(self, mode: str = "auto"):
        self.status(f"auto focus ({mode})")
        return self._autotune().run_focus(mode)

    def auto_stigmate(self):
        self.status("auto stigmation")
        return self._autotune().run_stigmate()

    # ---------------------------------------------------------------- stage
    @property
    def stage(self):
        return self.bundle.stage if self.bundle is not None else None

    def stage_axes(self):
        return self.axes

    def stage_z(self):
        st = self.stage
        z = st.positions.get("Z") if st is not None else None
        return None if z is None else float(z)

    def working_distance_mm(self):
        sem = getattr(self.bundle, "sem", None) if self.bundle is not None else None
        try:
            wd = sem.working_distance_mm() if sem is not None else None
        except Exception:  # noqa: BLE001
            wd = None
        return None if wd is None else float(wd)

    def set_working_distance(self, wd_mm):
        """The focus to *wd_mm* through FOCUS C / F (read back: :meth:`working_distance_mm` follows)."""
        sem = getattr(self.bundle, "sem", None) if self.bundle is not None else None
        if sem is None or not self.bundle.connected:
            raise RuntimeError("no microscope: the working distance cannot be set")
        self.status(f"working distance {float(wd_mm):g} mm")
        sem.set_working_distance(float(wd_mm)).result(30)

    def stage_position(self):
        st = self.stage
        if st is None:
            return None
        x, y = st.positions.get("X"), st.positions.get("Y")
        return None if x is None or y is None else (float(x), float(y))

    def read_stage(self):
        st = self.stage
        if st is None:
            raise RuntimeError("no stage (microscope not connected)")
        st.refresh(("X", "Y")).result(30)
        pos = self.stage_position()
        if pos is None:
            raise RuntimeError("the stage position is not known")
        return pos

    def move_stage(self, x, y, z=None):
        """One programmed move (one-way approach in X / Y; Z too when given), behind the modal "Stage moving" dialog of
        the microscope module: it blocks the user's interaction until the move ends and has STOP.
        A move already in progress (e.g. the user's) is waited for first.  Abort stops the stage."""
        from ...microscope import StageBusy

        st = self.stage
        if st is None:
            raise RuntimeError("no stage (microscope not connected)")
        self._wait_not_busy(st)
        where = f"({x:.1f}, {y:.1f}" + (f", Z {z:.1f}" if z is not None else "") + ") µm"
        self.status(f"stage moving to {where}")
        targets = {"X": float(x), "Y": float(y)}
        if z is not None:
            targets["Z"] = float(z)
        try:
            fut = st.move(targets)
        except StageBusy as e:
            raise RuntimeError(f"stage busy: {e}") from e
        self.bridge.call(lambda done: self._show_move(st, fut, f"lithography: to {where}", done), on_cancel=st.stop)
        fut.result(st.move_timeout_s + 30)  # the final positions, or the move's error

    def _wait_not_busy(self, st, timeout: float | None = None):
        if not st.busy:
            return
        self.status("waiting for another stage move to end")
        t = 0.0
        limit = st.move_timeout_s if timeout is None else timeout
        while st.busy:
            if self.bridge.cancelled.is_set():
                raise RuntimeError("aborted while another stage move was in progress")
            if t >= limit:
                raise RuntimeError("another stage move did not end")
            threading.Event().wait(0.1)
            t += 0.1

    def _show_move(self, st, fut, description, done):
        """(GUI thread) The modal wait dialog until *fut* is done; then resolves *done*."""
        from ...microscope.movewait import StageMoveDialog

        if not fut.done():
            d = StageMoveDialog(st, fut, description, self.parent)
            self.move_dialog = d
            self.move_dialogs += 1
            d.exec()
            self.move_dialog = None
            d.deleteLater()
        done.set_result(True)

    # ---------------------------------------------------------------- column
    def magnification(self):
        if self.bundle is None:
            return None
        return self.bundle.sem.magnification() or getattr(self.bundle.field_scale, "magnification", None)

    def capture_image(self, width_um, px, frames=1):
        """A centred photo scan of *width_um* (scan rotation 0) with the imaging engine – narrowed to the
        widest that fits the ±10 V range when the scan waveform (line start, flyback) would not; the
        returned pixel size is the one scanned."""
        n = int(px)
        geo = dict(nx=n, ny=n, center_um=(0.0, 0.0), rotation_rad=0.0)
        w = float(width_um)
        fit = float(self.engine.fit_width(w, **geo))
        if fit < w * (1 - 1e-6):
            self.status(f"imaging {fit:.4g} µm: {w:.4g} µm does not fit the deflection range")
            w = fit
        else:
            self.status(f"imaging {w:.4g} µm at {n} px")
        f = self.engine.capture(frames=int(frames), width_um=w, **geo)
        c = self.engine.config
        per_frame = self.engine.frame_time_s() * n * n / max(1, c.nx * c.ny)  # this capture's pixels
        frame = f.result(max(60.0, 4 * frames * per_frame + 30.0))
        self.engine.stop()
        return frame.image, tuple(frame.config.pixel_size_um)

    def field_um(self):
        return float(self.engine.calibration.field_size_um)

    def field_at_1x_um(self):
        fs = getattr(self.bundle, "field_scale", None)
        if fs is not None:
            return float(fs.k_um)
        mag = self.bundle.sem.magnification() if self.bundle is not None else None
        return self.field_um() * float(mag or 1000.0)

    def set_magnification(self, mag):
        if self.bundle is None or not self.bundle.connected:
            raise RuntimeError("no microscope: the magnification cannot be set")
        m = float(mag)
        self.bundle.sem.set_magnification(int(m) if m.is_integer() else m).result(30)  # the driver takes ints
        fs = self.bundle.field_scale
        if fs is not None:
            fs.apply()
        return self.field_um()

    def preset_names(self):
        if self.bundle is None:
            return None
        return [p.get("name", "") for p in self.bundle.settings.beam_presets]

    def apply_preset(self, name):
        from ...microscope.sem import PRESET_KEYS

        if self.bundle is None or not self.bundle.connected:
            raise RuntimeError("no microscope: the beam preset cannot be recalled")
        p = next((p for p in self.bundle.settings.beam_presets if p.get("name") == name), None)
        if p is None:
            raise KeyError(f"no beam preset named {name!r} (Microscope → Beam presets…)")
        self.status(f"recalling beam preset {name!r}")
        self.bundle.sem.apply_preset({k: p[k] for k in PRESET_KEYS}).result(60)
        self.last_preset = name

    # ---------------------------------------------------------------- interactive
    def measure_marks(self, requests):
        if self.open_sem is None:
            raise RuntimeError("the SEM imaging window is not available for mark picking")
        out = []
        try:
            for r in requests:
                self.status(f"{r.label}: click the mark in the SEM window")

                def show(fut, r=r):
                    sem = self.open_sem()
                    if sem is None:
                        raise RuntimeError("cannot open the SEM imaging window")
                    self._sem = sem
                    self.engine.set_field_of_view(r.view_um, center_um=r.predicted_xcs)
                    self.engine.run()
                    sem.request_pick(f"{r.label} (CCS {r.ccs[0]:.2f}, {r.ccs[1]:.2f} µm) – "
                                     "predicted at the magenta cross, centre it on the green one",
                                     lambda xy: fut.done() or fut.set_result(xy), targets=[r.predicted_xcs])

                def cancel():
                    sem = getattr(self, "_sem", None)
                    if sem is not None:
                        sem.cancel_pick()

                xy = self.bridge.call(show, on_cancel=cancel)
                if xy is None:
                    return None
                out.append((float(xy[0]), float(xy[1])))
            return out
        finally:
            self.bridge._call.emit((lambda f: self._after_marks(), concurrent.futures.Future()))

    def _after_marks(self):
        try:  # stop scanning over the resist; the next scan starts from the full field
            self.engine.stop()
            self.engine.full_field()
        except Exception:  # noqa: BLE001
            pass

    def ask(self, text):
        def show(fut):
            r = QtWidgets.QMessageBox.question(self.parent, "Lithography", text)
            fut.set_result(r == QtWidgets.QMessageBox.Yes)

        return bool(self.bridge.call(show))

    def prepare_run(self):
        sem = getattr(self.bundle, "sem", None) if self.bundle is not None else None
        if sem is None or not self.bundle.connected:
            raise RuntimeError("the microscope is not connected")
        self.status("SEM: RUN mode, Slow scan 1")
        # only what is not so already: the SEM refuses RUN ON when running (E2, "unexecutable")
        self._ensure(sem, "RUN", "running", True, lambda: sem.set_running(True))
        self._ensure(sem, "SCAN", "scan", "1", lambda: sem.set_scan("1"))

    @staticmethod
    def _ensure(sem, item: str, key: str, want, send) -> None:
        """Read *item*; send only when ``sem.state[key]`` is not *want*.  A refusal is an error only
        when the SEM is still not so afterwards."""
        sem.refresh((item,)).result(30)
        if sem.state.get(key) == want:
            return
        try:
            send().result(30)
        except Exception:
            sem.refresh((item,)).result(30)
            if sem.state.get(key) != want:
                raise

    def ask_focus(self, text):
        """The SEM window scans live over the full field; the prompt waits (not modal) – focus and
        stigmate on the SEM panel or in the Microscope / SEM windows.  The scan stops afterwards."""
        self.status("focus and stigmate by hand, then Continue")

        def show(fut):
            if self.open_sem is not None:
                try:
                    sem = self.open_sem()
                    if sem is not None:
                        self.engine.full_field()
                        self.engine.run()
                except Exception:  # noqa: BLE001 - the prompt still shows (the SEM's own display works too)
                    pass
            d = FocusPrompt(text, self.parent)
            self.focus_prompt = d

            def answered(ok):
                self.focus_prompt = None
                try:
                    self.engine.stop()
                except Exception:  # noqa: BLE001
                    pass
                if not fut.done():
                    fut.set_result(bool(ok))

            d.answered.connect(answered)
            d.show()
            d.raise_()

        def cancel():
            d = getattr(self, "focus_prompt", None)
            if d is not None:
                d.finish(False)

        return bool(self.bridge.call(show, on_cancel=cancel))

    # ---------------------------------------------------------------- exposure
    def expose(self, pattern, params, transform, settings):
        from ..writer import LithoWriter

        self.engine.stop()
        self.engine.wait_idle(5.0)
        w = LithoWriter(self.handle, self.engine.calibration, settings)
        job = w.compile(pattern, params, transform)  # before the lease: compiling does not hold the DAQ
        # The DAQ reserved for the exposure (the imaging engine cannot scan over the resist meanwhile)
        # and the SEM on external scan for it (the HAL blanks the beam before switching; the previous
        # scan source comes back when the lease ends, also on failure).  The writer refuses to stream
        # if the SEM is not on external scan.
        with self.handle.exclusive("lithography exposure", timeout=30.0, external_control=True):
            self.status(f"exposing {len(pattern)} element(s)")
            return w.expose_job(job, progress=self.progress)

    def abort(self):
        self.bridge.cancelled.set()
        if self.handle is not None:
            self.handle.abort()

    def close(self):
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:  # noqa: BLE001
                pass
