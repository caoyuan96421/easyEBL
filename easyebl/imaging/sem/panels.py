"""Control panels of the SEM window (scan, capture, image, field of view, column)."""
from __future__ import annotations

import math
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..engine import RESOLUTIONS, SCAN_SPEEDS, EngineState, ImagingEngine, speed_by_name, speed_of
from ...microscope.control import MicroscopeControl, NotAvailable
from .display import ContrastBrightness

__all__ = ["ScanPanel", "ImagePanel", "FieldOfViewPanel", "ColumnPanel", "CapturePanel", "ZOOM_STEP"]

#: one wheel notch / zoom button: ×2 every three steps
ZOOM_STEP = 2 ** (1 / 3)

Notify = Callable[[str], None]

RUN_STYLE = "QPushButton{background:#2e7d32;color:white;font-weight:bold;border-radius:4px}" \
            "QPushButton:hover{background:#388e3c}"
FREEZE_STYLE = "QPushButton{background:#1565c0;color:white;font-weight:bold;border-radius:4px}" \
               "QPushButton:hover{background:#1976d2}"
FREEZING_STYLE = "QPushButton{background:#ef6c00;color:white;font-weight:bold;border-radius:4px}"


def _group(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QGridLayout(g)
    lay.setContentsMargins(8, 10, 8, 8)
    lay.setHorizontalSpacing(6)
    lay.setVerticalSpacing(5)
    return g, lay


def _speed_index(cfg) -> int:
    sp = speed_of(cfg)
    return -1 if sp is None else SCAN_SPEEDS.index(sp)


def _dspin(lo, hi, dec, suffix, step):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSuffix(suffix)
    s.setSingleStep(step)
    s.setKeyboardTracking(False)
    return s


# ======================================================================= custom size
#: the largest custom scan (pixels per side)
MAX_CUSTOM_PX = 16384


class CustomSizeDialog(QtWidgets.QDialog):
    """A scan size of its own (e.g. an ultra-high-resolution capture): width × height in pixels,
    with the frame time at a scan speed and the memory the image takes."""

    def __init__(self, nx: int, ny: int, frame_time=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom size")
        self.frame_time = frame_time  # (nx, ny) -> s, or None
        self.w = QtWidgets.QSpinBox()
        self.h = QtWidgets.QSpinBox()
        for b, v in ((self.w, nx), (self.h, ny)):
            b.setRange(16, MAX_CUSTOM_PX)
            b.setSingleStep(256)
            b.setValue(int(v))
            b.setSuffix(" px")
            b.setKeyboardTracking(False)
        self.square = QtWidgets.QCheckBox("square")
        self.square.setChecked(nx == ny)
        self.w.valueChanged.connect(self._w_changed)
        self.h.valueChanged.connect(self._update)
        self.square.toggled.connect(self._w_changed)
        presets = QtWidgets.QHBoxLayout()
        for n in (4096, 5000, 8000, 10000):
            b = QtWidgets.QPushButton(f"{n}²")
            b.clicked.connect(lambda _=False, n=n: self.set_size(n, n))
            presets.addWidget(b)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#9aa")
        f = QtWidgets.QFormLayout(self)
        f.addRow("width", self.w)
        f.addRow("height", self.h)
        f.addRow("", self.square)
        f.addRow("presets", presets)
        f.addRow(self.info)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._update()

    def set_size(self, nx: int, ny: int) -> None:
        self.square.setChecked(nx == ny)
        self.w.setValue(int(nx))
        self.h.setValue(int(ny))
        self._update()

    def _w_changed(self, *_):
        if self.square.isChecked() and self.h.value() != self.w.value():
            self.h.setValue(self.w.value())
        self._update()

    def value(self) -> tuple:
        return (int(self.w.value()), int(self.h.value()))

    def _update(self, *_):
        nx, ny = self.value()
        mb = nx * ny * 4 / 1e6
        txt = f"{nx * ny / 1e6:.3g} Mpixel · image {mb:.0f} MB"
        if self.frame_time is not None:
            try:
                t = float(self.frame_time(nx, ny))
                txt += f" · frame ≈ {t:.3g} s" if t < 120 else f" · frame ≈ {t / 60:.3g} min"
            except Exception:  # noqa: BLE001
                pass
        self.info.setText(txt)


def _custom_item(combo: QtWidgets.QComboBox, nx: int, ny: int) -> int:
    """Select (adding it if needed) the custom size *nx* × *ny* in a size combo (before 'Custom…')."""
    key = f"{nx}x{ny}"
    i = combo.findData(key)
    if i < 0:
        i = combo.count() - 1  # before the 'Custom…' entry
        combo.insertItem(i, f"{nx} × {ny} (custom)", key)
    combo.setCurrentIndex(i)
    return i


# ======================================================================= scan
class ScanPanel(QtWidgets.QGroupBox):
    """Run/Freeze, single scan, scan speed and resolution."""

    def __init__(self, engine: ImagingEngine, notify: Notify, parent=None):
        super().__init__("Scan", parent)
        self.engine = engine
        self.notify = notify
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)

        self.b_run = QtWidgets.QPushButton("Run")
        self.b_run.setMinimumHeight(44)
        self.b_run.setToolTip("Run / Freeze (F5).  While freezing, press again to stop immediately.")
        self.b_run.clicked.connect(self._run_freeze)
        self.b_single = QtWidgets.QPushButton("Single")
        self.b_single.setMinimumHeight(44)
        self.b_single.setToolTip("Scan one frame, then freeze (F6)")
        self.b_single.clicked.connect(lambda: self._do(engine.single))
        lay.addWidget(self.b_run, 0, 0, 1, 2)
        lay.addWidget(self.b_single, 0, 2)

        lay.addWidget(QtWidgets.QLabel("Speed"), 1, 0)
        self.speed = QtWidgets.QComboBox()
        for s in SCAN_SPEEDS:
            self.speed.addItem(s.name, s.name)
        self.speed.addItem("Custom", None)  # a dwell set from the console
        self.speed.activated.connect(self._speed_chosen)
        b_faster = QtWidgets.QToolButton(text="◀")
        b_faster.setToolTip("Faster scan (F3)")
        b_slower = QtWidgets.QToolButton(text="▶")
        b_slower.setToolTip("Slower scan (F4)")
        b_faster.clicked.connect(lambda: self.step_speed(-1))
        b_slower.clicked.connect(lambda: self.step_speed(+1))
        sp_row = QtWidgets.QHBoxLayout()
        sp_row.addWidget(b_faster)
        sp_row.addWidget(self.speed, 1)
        sp_row.addWidget(b_slower)
        lay.addLayout(sp_row, 1, 1, 1, 2)

        lay.addWidget(QtWidgets.QLabel("Pixels"), 2, 0)
        self.res = QtWidgets.QComboBox()
        for w, h in RESOLUTIONS:
            self.res.addItem(f"{w} × {h}", f"{w}x{h}")
        self.res.addItem("Custom…", None)
        self.res.setToolTip("scan pixels; Custom… for any size up to 16384 × 16384 (ultra-high resolution)")
        self.res.activated.connect(self._res_chosen)
        lay.addWidget(self.res, 2, 1, 1, 2)

        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("color:#9aa")
        lay.addWidget(self.info, 3, 0, 1, 3)
        self.sync()

    def _do(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def _run_freeze(self):
        self._do(self.engine.toggle_run_freeze)

    def _speed_chosen(self, i):
        d = self.speed.itemData(i)
        if d is not None:
            self._do(self.engine.set_speed, d)
        self.sync()

    def step_speed(self, step: int):
        i = _speed_index(self.engine.config)
        if i < 0:
            i = 0
        i = max(0, min(len(SCAN_SPEEDS) - 1, i + step))
        self._do(self.engine.set_speed, SCAN_SPEEDS[i])
        self.sync()

    def _res_chosen(self, i):
        d = self.res.itemData(i)
        if d is None:  # Custom…: any size
            c = self.engine.config
            dlg = CustomSizeDialog(c.nx, c.ny, self._frame_time, self)
            self._custom_dialog = dlg
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                self.sync()
                return
            nx, ny = dlg.value()
            self._do(self.engine.set_resolution, nx, ny)
        else:
            self._do(self.engine.set_resolution, *(int(v) for v in d.split("x")))
        self.sync()

    def _frame_time(self, nx: int, ny: int) -> float:
        c = self.engine.config
        return ny * (nx * c.actual_dwell_s + c.preline_s + c.flyback_s) + c.frame_settle_s

    def sync(self):
        cfg = self.engine.config
        i = _speed_index(cfg)
        self.speed.setCurrentIndex(i if i >= 0 else self.speed.count() - 1)
        j = self.res.findData(f"{cfg.nx}x{cfg.ny}")
        if j >= 0:
            self.res.setCurrentIndex(j)
        else:
            _custom_item(self.res, cfg.nx, cfg.ny)
        try:
            ft = self.engine.frame_time_s()
        except Exception:  # noqa: BLE001
            ft = float("nan")
        self.info.setText(f"frame {ft:.3g} s")

    def sync_state(self, st: EngineState):
        if st == EngineState.FREEZING:
            self.b_run.setText("Freezing…")
            self.b_run.setStyleSheet(FREEZING_STYLE)
        elif st in (EngineState.LIVE, EngineState.STARTING):
            self.b_run.setText("Freeze")
            self.b_run.setStyleSheet(FREEZE_STYLE)
        elif st in (EngineState.SINGLE, EngineState.CAPTURE, EngineState.DISCONNECTED, EngineState.PAUSED):
            self.b_run.setText("Stop")
            self.b_run.setStyleSheet(FREEZING_STYLE)
        else:
            self.b_run.setText("Run")
            self.b_run.setStyleSheet(RUN_STYLE)


# ===================================================================== capture
class CapturePanel(QtWidgets.QGroupBox):
    captureRequested = QtCore.Signal(dict)
    saveRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Capture", parent)
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.addWidget(QtWidgets.QLabel("Size"), 0, 0)
        self.res = QtWidgets.QComboBox()
        for w, h in RESOLUTIONS:
            self.res.addItem(f"{w} × {h}", f"{w}x{h}")
        self.res.addItem("Custom…", None)
        self.res.setToolTip("capture pixels; Custom… for any size up to 16384 × 16384 (ultra-high resolution)")
        self.res.setCurrentIndex(2)  # 1280 × 960
        self._last_res = 2
        self.res.activated.connect(self._res_chosen)
        lay.addWidget(self.res, 0, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Speed"), 1, 0)
        self.speed = QtWidgets.QComboBox()
        for s in SCAN_SPEEDS:
            self.speed.addItem(s.label(), s.name)
        self.speed.setCurrentIndex(5)  # Slow 3
        lay.addWidget(self.speed, 1, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Integrate"), 2, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, 256)
        self.frames.setSuffix(" frame(s)")
        lay.addWidget(self.frames, 2, 1, 1, 2)
        self.b_capture = QtWidgets.QPushButton("Capture")
        self.b_capture.setMinimumHeight(34)
        self.b_capture.setToolTip("Photo scan with these settings, then freeze and save (F9)")
        self.b_capture.clicked.connect(self.request)
        self.b_save = QtWidgets.QPushButton("Save image…")
        self.b_save.setToolTip("Save the displayed image (Ctrl+S)")
        self.b_save.clicked.connect(self.saveRequested.emit)
        lay.addWidget(self.b_capture, 3, 0, 1, 2)
        lay.addWidget(self.b_save, 3, 2)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.progress.setVisible(False)
        lay.addWidget(self.progress, 4, 0, 1, 3)

    def _res_chosen(self, i):
        if self.res.itemData(i) is not None:
            self._last_res = i
            return
        w, h = (int(v) for v in self.res.itemData(self._last_res).split("x"))
        dlg = CustomSizeDialog(w, h, self._frame_time, self)
        self._custom_dialog = dlg
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            self.res.setCurrentIndex(self._last_res)
            return
        self._last_res = _custom_item(self.res, *dlg.value())

    def _frame_time(self, nx: int, ny: int) -> float:
        sp = speed_by_name(self.speed.currentData())
        f = sp.fields()
        return ny * (nx * f["dwell_s"] + f.get("preline_s", 0.0) + f.get("flyback_s", 0.0))

    def request(self):
        w, h = (int(v) for v in self.res.currentData().split("x"))
        self.captureRequested.emit({"nx": w, "ny": h, "speed": self.speed.currentData(),
                                    "frames": self.frames.value()})


# ======================================================================= image
class ImagePanel(QtWidgets.QGroupBox):
    """Contrast, brightness, auto contrast/brightness, frame averaging."""

    levelsChanged = QtCore.Signal()

    def __init__(self, engine: ImagingEngine, cb: ContrastBrightness, notify: Notify, parent=None):
        super().__init__("Image", parent)
        self.engine = engine
        self.cb = cb
        self.notify = notify
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        self.s_con = self._slider()
        self.s_bri = self._slider()
        self.l_con = QtWidgets.QLabel()
        self.l_bri = QtWidgets.QLabel()
        for w in (self.l_con, self.l_bri):
            w.setMinimumWidth(38)
            w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lay.addWidget(QtWidgets.QLabel("Contrast"), 0, 0)
        lay.addWidget(self.s_con, 0, 1)
        lay.addWidget(self.l_con, 0, 2)
        lay.addWidget(QtWidgets.QLabel("Brightness"), 1, 0)
        lay.addWidget(self.s_bri, 1, 1)
        lay.addWidget(self.l_bri, 1, 2)
        self.b_acb = QtWidgets.QPushButton("Auto C/B")
        self.b_acb.setToolTip("Auto contrast & brightness from the current image (Ctrl+A)")
        self.cb_auto = QtWidgets.QCheckBox("Continuous")
        self.cb_auto.setToolTip("Re-adjust contrast/brightness on every frame")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.b_acb, 1)
        row.addWidget(self.cb_auto)
        lay.addLayout(row, 2, 0, 1, 3)
        lay.addWidget(QtWidgets.QLabel("Averaging"), 3, 0)
        self.avg = QtWidgets.QSpinBox()
        self.avg.setRange(1, 256)
        self.avg.setSuffix(" frame(s)")
        self.avg.setToolTip("Running average over N frames (noise reduction for live view)")
        self.avg.setValue(engine.options.averaging)
        self.avg.valueChanged.connect(self._avg)
        lay.addWidget(self.avg, 3, 1, 1, 2)
        self.s_con.valueChanged.connect(self._sliders)
        self.s_bri.valueChanged.connect(self._sliders)
        self.sync()

    @staticmethod
    def _slider():
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setRange(0, 1000)
        s.setSingleStep(5)
        s.setPageStep(50)
        return s

    def _avg(self, n):
        try:
            self.engine.set_options(averaging=int(n))
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def _sliders(self):
        self.cb.contrast = self.s_con.value() / 1000
        self.cb.brightness = self.s_bri.value() / 1000
        self._labels()
        self.levelsChanged.emit()

    def _labels(self):
        self.l_con.setText(f"{self.cb.contrast * 100:.0f}")
        self.l_bri.setText(f"{self.cb.brightness * 100:.0f}")

    def sync(self):
        for s, v in ((self.s_con, self.cb.contrast), (self.s_bri, self.cb.brightness)):
            s.blockSignals(True)
            s.setValue(int(round(v * 1000)))
            s.blockSignals(False)
        self._labels()


# ============================================================== field of view
class FieldOfViewPanel(QtWidgets.QGroupBox):
    """Scan size (the swept range: zoom without touching the column), scan position
    (beam shift) and scan rotation."""

    def __init__(self, engine: ImagingEngine, notify: Notify, parent=None):
        super().__init__("Field of view", parent)
        self.engine = engine
        self.notify = notify
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setHorizontalSpacing(6)

        # --- size
        lay.addWidget(QtWidgets.QLabel("Size"), 0, 0)
        self.size = _dspin(1e-4, 1e7, 3, " µm", 1.0)
        self.size.setToolTip("Scan width: the swept range.  The wheel on the image zooms (the column "
                             "magnification is not changed); at most the full field of view.")
        self.size.valueChanged.connect(lambda v: self._do(self.engine.set_field_of_view, v))
        lay.addWidget(self.size, 0, 1, 1, 2)
        row = QtWidgets.QHBoxLayout()
        for txt, tip, fn in (("−", "zoom out", lambda: self.zoom(-1)), ("+", "zoom in", lambda: self.zoom(+1)),
                             ("Full", "the whole field of view, centred", lambda: self._do(self.engine.full_field))):
            b = QtWidgets.QToolButton(text=txt)
            b.setToolTip(tip)
            b.setMinimumWidth(28)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        lay.addLayout(row, 0, 3)
        self.l_zoom = QtWidgets.QLabel()
        self.l_zoom.setStyleSheet("color:#9aa")
        lay.addWidget(self.l_zoom, 1, 1, 1, 3)

        # --- position (beam shift)
        lay.addWidget(QtWidgets.QLabel("Position"), 2, 0)
        self.px = _dspin(-1e7, 1e7, 3, " µm", 0.1)
        self.py = _dspin(-1e7, 1e7, 3, " µm", 0.1)
        self.px.setPrefix("X ")
        self.py.setPrefix("Y ")
        self.px.valueChanged.connect(lambda v: self._do(self.engine.set_beam_shift, v, self.engine.config.center_um[1]))
        self.py.valueChanged.connect(lambda v: self._do(self.engine.set_beam_shift, self.engine.config.center_um[0], v))
        lay.addWidget(self.px, 2, 1, 1, 2)
        lay.addWidget(self.py, 3, 1, 1, 2)
        pad = QtWidgets.QGridLayout()
        pad.setSpacing(1)
        for (r, c, txt, dx, dy) in [(0, 1, "▲", 0, 1), (1, 0, "◀", -1, 0), (1, 2, "▶", 1, 0), (2, 1, "▼", 0, -1)]:
            b = QtWidgets.QToolButton(text=txt)
            b.setFixedSize(22, 20)
            b.setAutoRepeat(True)
            b.setAutoRepeatInterval(120)
            b.clicked.connect(lambda _=False, dx=dx, dy=dy: self.step(dx, dy))
            pad.addWidget(b, r, c)
        b0 = QtWidgets.QToolButton(text="●")
        b0.setFixedSize(22, 20)
        b0.setToolTip("centre the scan (beam shift 0)")
        b0.clicked.connect(lambda: self._do(self.engine.reset_beam_shift))
        pad.addWidget(b0, 1, 1)
        lay.addLayout(pad, 2, 3, 3, 1)
        lay.addWidget(QtWidgets.QLabel("Step"), 4, 0)
        self.stepc = QtWidgets.QComboBox()
        for pct in (1, 2, 5, 10, 25):
            self.stepc.addItem(f"{pct} % of size", pct / 100)
        self.stepc.setCurrentIndex(2)
        lay.addWidget(self.stepc, 4, 1, 1, 2)

        # --- rotation
        lay.addWidget(QtWidgets.QLabel("Rotation"), 5, 0)
        self.rot = _dspin(-180, 180, 2, " °", 1.0)
        self.rot.setWrapping(True)
        self.rot.valueChanged.connect(lambda v: self._do(self.engine.set_rotation_deg, v))
        lay.addWidget(self.rot, 5, 1, 1, 2)
        b = QtWidgets.QToolButton(text="0°")
        b.setToolTip("no scan rotation")
        b.clicked.connect(lambda: self._do(self.engine.set_rotation_deg, 0.0))
        lay.addWidget(b, 5, 3)

        hint = QtWidgets.QLabel("Image: wheel zooms, drag pans, double-click centres a point.")
        hint.setStyleSheet("color:#9aa")
        hint.setWordWrap(True)
        lay.addWidget(hint, 6, 0, 1, 4)
        self.sync()

    def _do(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
        self.sync()

    def zoom(self, steps: int):
        self._do(self.engine.zoom, ZOOM_STEP ** steps)

    def step(self, dx, dy):
        s = self.stepc.currentData() * self.engine.config.width_um
        self._do(self.engine.shift_beam, dx * s, dy * s)

    def sync(self):
        cfg, cal = self.engine.config, self.engine.calibration
        field = cal.field_size_um
        dec = max(0, min(4, 3 - int(math.floor(math.log10(max(field, 1e-6))))))
        for sp, v in ((self.size, cfg.width_um), (self.px, cfg.center_um[0]), (self.py, cfg.center_um[1]),
                      (self.rot, math.degrees(cfg.rotation_rad))):
            if sp.hasFocus():
                continue
            sp.blockSignals(True)
            if sp is not self.rot:
                sp.setDecimals(dec)
                sp.setSingleStep(10 ** -dec * 10)
            sp.setValue(v)
            sp.blockSignals(False)
        self.size.setMaximum(self.engine.max_width_um())
        self.l_zoom.setText(f"zoom ×{self.engine.zoom_factor:.3g} of the full field ({field:.4g} µm)")


# ================================================================== column
class ColumnPanel(QtWidgets.QGroupBox):
    """Magnification, focus/WD, stigmation, alignment – the SEM column over its remote control.

    Requests are queued on the microscope link (they return at once); errors come
    back through :attr:`failed` on the GUI thread."""

    failed = QtCore.Signal(str)
    tuneEvent = QtCore.Signal(object)  # AutoTuner events, handed to the GUI thread

    def __init__(self, microscope: MicroscopeControl, notify: Notify, parent=None):
        super().__init__("Column", parent)
        self.mic = microscope
        self.notify = notify
        self.autotune = None  # easyebl.imaging.autotune.AutoTuner (set_autotune)
        self.failed.connect(notify)
        self.tuneEvent.connect(self._tune_event)
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)

        def btn(text, fn, tip=""):
            b = QtWidgets.QToolButton(text=text)
            b.setMinimumWidth(30)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False: self._call(fn))
            return b

        lay.addWidget(QtWidgets.QLabel("Magnification"), 0, 0)
        self.l_mag = QtWidgets.QLabel("×—")
        lay.addWidget(self.l_mag, 0, 1)
        lay.addWidget(btn("−", lambda: self.mic.step_magnification(-1), "Lower magnification"), 0, 2)
        lay.addWidget(btn("+", lambda: self.mic.step_magnification(+1), "Higher magnification"), 0, 3)

        lay.addWidget(QtWidgets.QLabel("Focus"), 1, 0)
        self.l_wd = QtWidgets.QLabel("WD —")
        lay.addWidget(self.l_wd, 1, 1)
        lay.addWidget(btn("−", lambda: self.mic.focus(-1), "Focus (fine) −"), 1, 2)
        lay.addWidget(btn("+", lambda: self.mic.focus(+1), "Focus (fine) +"), 1, 3)

        lay.addWidget(QtWidgets.QLabel("Stigma X"), 2, 0)
        lay.addWidget(btn("−", lambda: self.mic.stigmate(-1, 0)), 2, 2)
        lay.addWidget(btn("+", lambda: self.mic.stigmate(+1, 0)), 2, 3)
        lay.addWidget(QtWidgets.QLabel("Stigma Y"), 3, 0)
        lay.addWidget(btn("−", lambda: self.mic.stigmate(0, -1)), 3, 2)
        lay.addWidget(btn("+", lambda: self.mic.stigmate(0, +1)), 3, 3)

        # our own auto focus / stigmation (easyebl.imaging.autotune): test images of this scan field
        row = QtWidgets.QHBoxLayout()
        self.b_af = QtWidgets.QToolButton(text="Auto focus")
        self.b_af.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.b_af.setToolTip("search the objective focus with test images of the present scan field; the step "
                             "follows the magnification (coarse focus first only when needed).  Falls back to "
                             "the start values when no clear optimum is found")
        self.b_af.clicked.connect(lambda: self.start_tune("auto"))
        m = QtWidgets.QMenu(self.b_af)
        m.addAction("Fine focus only (FOCUS F)", lambda: self.start_tune("fine"))
        m.addAction("Coarse focus only (FOCUS C)", lambda: self.start_tune("coarse"))
        self.b_af.setMenu(m)
        self.b_as = QtWidgets.QToolButton(text="Auto stigma")
        self.b_as.setToolTip("full-range search of stigmator X, then Y, with test images; falls back to the "
                             "start values when no clear optimum is found")
        self.b_as.clicked.connect(lambda: self.start_tune("stig"))
        self.b_fs = QtWidgets.QToolButton(text="Focus + stig")
        self.b_fs.setToolTip("auto focus, auto stigma, then fine focus again")
        self.b_fs.clicked.connect(lambda: self.start_tune("both"))
        self.b_tune_cancel = QtWidgets.QToolButton(text="Cancel")
        self.b_tune_cancel.setToolTip("stop the search and set the start values again")
        self.b_tune_cancel.clicked.connect(self.cancel_tune)
        self.b_tune_cancel.setVisible(False)
        b_al = QtWidgets.QToolButton(text="Alignment")
        b_al.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(b_al)
        for kind in ("Gun", "Aperture", "Stigma X", "Stigma Y"):
            menu.addAction(f"{kind} alignment", lambda k=kind: self._call(lambda: self.mic.alignment(k)))
        b_al.setMenu(menu)
        for b in (self.b_af, self.b_as, self.b_fs, self.b_tune_cancel, b_al):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row, 4, 0, 1, 4)
        self.l_tune = QtWidgets.QLabel("")
        self.l_tune.setStyleSheet("color:#9aa")
        self.l_tune.setWordWrap(True)
        self.l_tune.setVisible(False)
        lay.addWidget(self.l_tune, 6, 0, 1, 4)
        self.set_autotune(None)
        self.l_note = QtWidgets.QLabel(f"Microscope: {microscope.name}" + (
            "" if microscope.connected else " – no remote control: set the column on the SEM"))
        self.l_note.setStyleSheet("color:#9aa")
        self.l_note.setWordWrap(True)
        lay.addWidget(self.l_note, 5, 0, 1, 4)

    # ------------------------------------------------------------ auto focus / stigmation
    def set_autotune(self, tuner) -> None:
        if self.autotune is not None:
            self.autotune.remove_listener(self.tuneEvent.emit)
        self.autotune = tuner
        if tuner is not None:
            tuner.add_listener(self.tuneEvent.emit)
        self._tune_buttons(busy=bool(tuner is not None and tuner.busy))

    def _tune_buttons(self, busy: bool) -> None:
        ok = self.autotune is not None and self.autotune.available
        for b in (self.b_af, self.b_as, self.b_fs):
            b.setEnabled(ok and not busy)
            if not ok:
                b.setToolTip("needs the microscope's remote control and the imaging engine")
        self.b_tune_cancel.setVisible(busy)

    def start_tune(self, what: str):
        """*what*: "auto" / "fine" / "coarse" focus, "stig", or "both".  Returns the future (or None)."""
        t = self.autotune
        if t is None:
            self.notify("auto focus / stigmation needs the microscope (remote control) and the imaging engine")
            return None
        try:
            if what == "stig":
                fut = t.auto_stigmate()
            elif what == "both":
                fut = t.focus_and_stigmate()
            else:
                fut = t.auto_focus(what)
        except Exception as e:  # noqa: BLE001
            self.notify(f"auto focus / stigmation: {e}")
            return None
        self._tune_buttons(busy=True)
        return fut

    def cancel_tune(self) -> None:
        if self.autotune is not None:
            self.autotune.cancel()

    def _tune_event(self, ev: dict) -> None:
        typ = ev.get("type")
        if typ == "start":
            self._tune_buttons(busy=True)
            self.l_tune.setVisible(True)
            self.l_tune.setText(ev.get("text", ""))
        elif typ == "progress":
            self.l_tune.setText(ev.get("text", ""))
        elif typ == "done":
            self._tune_buttons(busy=False)
            self.l_tune.setText(ev.get("text", ""))
            self.notify(ev.get("text", ""))

    def _call(self, fn):
        try:
            fut = fn()
        except NotAvailable as e:
            self.notify(f"{str(e).capitalize()} is not available – use the SEM console.")
            return
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
            return
        if hasattr(fut, "add_done_callback"):
            fut.add_done_callback(lambda f: None if f.cancelled() or f.exception() is None
                                  else self.failed.emit(f"microscope: {f.exception()}"))

    def sync(self, frame=None):
        mag = self.mic.magnification()
        self.l_mag.setText(f"×{mag:g}" if mag else "×—")
        wd = self.mic.working_distance_mm()
        self.l_wd.setText(f"WD {wd:.2f} mm" if wd else "WD —")
