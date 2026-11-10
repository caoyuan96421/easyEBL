"""Control panels of the SEM window (scan, image, beam shift, column stubs)."""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..engine import RESOLUTIONS, SCAN_SPEEDS, EngineState, ImagingEngine
from .display import ContrastBrightness
from .optics import MicroscopeControl, NotAvailable

__all__ = ["ScanPanel", "ImagePanel", "BeamShiftPanel", "ColumnPanel", "CapturePanel"]

Notify = Callable[[str], None]

RUN_STYLE = "QPushButton{background:#2e7d32;color:white;font-weight:bold;font-size:15px;border-radius:4px}" \
            "QPushButton:hover{background:#388e3c}"
FREEZE_STYLE = "QPushButton{background:#1565c0;color:white;font-weight:bold;font-size:15px;border-radius:4px}" \
               "QPushButton:hover{background:#1976d2}"
FREEZING_STYLE = "QPushButton{background:#ef6c00;color:white;font-weight:bold;font-size:15px;border-radius:4px}"


def _group(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QGridLayout(g)
    lay.setContentsMargins(8, 10, 8, 8)
    lay.setHorizontalSpacing(6)
    lay.setVerticalSpacing(5)
    return g, lay


def _speed_index(dwell_s: float) -> int:
    best = min(range(len(SCAN_SPEEDS)), key=lambda i: abs(SCAN_SPEEDS[i].dwell_s - dwell_s))
    return best if abs(SCAN_SPEEDS[best].dwell_s - dwell_s) < 1e-12 else -1


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
            self.speed.addItem(s.name, s.dwell_s)
        self.speed.addItem("Custom", None)
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

        lay.addWidget(QtWidgets.QLabel("Scan size"), 2, 0)
        self.res = QtWidgets.QComboBox()
        for w, h in RESOLUTIONS:
            self.res.addItem(f"{w} × {h}", f"{w}x{h}")
        self.res.addItem("Custom", None)
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
        i = _speed_index(self.engine.config.dwell_s)
        if i < 0:
            i = 0
        i = max(0, min(len(SCAN_SPEEDS) - 1, i + step))
        self._do(self.engine.set_speed, SCAN_SPEEDS[i])
        self.sync()

    def _res_chosen(self, i):
        d = self.res.itemData(i)
        if d is not None:
            self._do(self.engine.set_resolution, *(int(v) for v in d.split("x")))
        self.sync()

    def sync(self):
        cfg = self.engine.config
        i = _speed_index(cfg.dwell_s)
        self.speed.setCurrentIndex(i if i >= 0 else self.speed.count() - 1)
        j = self.res.findData(f"{cfg.nx}x{cfg.ny}")
        self.res.setCurrentIndex(j if j >= 0 else self.res.count() - 1)
        try:
            ft = self.engine.frame_time_s()
        except Exception:  # noqa: BLE001
            ft = float("nan")
        self.info.setText(
            f"dwell {cfg.actual_dwell_s * 1e6:g} µs · {cfg.samples_per_pixel} samples/px · frame {ft:.3g} s"
        )

    def sync_state(self, st: EngineState):
        if st == EngineState.FREEZING:
            self.b_run.setText("Freezing…")
            self.b_run.setStyleSheet(FREEZING_STYLE)
        elif st in (EngineState.LIVE, EngineState.STARTING):
            self.b_run.setText("Freeze")
            self.b_run.setStyleSheet(FREEZE_STYLE)
        elif st in (EngineState.SINGLE, EngineState.CAPTURE, EngineState.DISCONNECTED):
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
        self.res.setCurrentIndex(2)  # 1280 × 960
        lay.addWidget(self.res, 0, 1, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Speed"), 1, 0)
        self.speed = QtWidgets.QComboBox()
        for s in SCAN_SPEEDS:
            self.speed.addItem(s.label(), s.dwell_s)
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

    def request(self):
        w, h = (int(v) for v in self.res.currentData().split("x"))
        self.captureRequested.emit({"nx": w, "ny": h, "dwell_s": self.speed.currentData(),
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


# ================================================================ beam shift
class BeamShiftPanel(QtWidgets.QGroupBox):
    def __init__(self, engine: ImagingEngine, notify: Notify, parent=None):
        super().__init__("Beam shift", parent)
        self.engine = engine
        self.notify = notify
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        pad = QtWidgets.QGridLayout()
        pad.setSpacing(2)
        for (r, c, txt, dx, dy) in [(0, 1, "▲", 0, 1), (1, 0, "◀", -1, 0), (1, 2, "▶", 1, 0), (2, 1, "▼", 0, -1)]:
            b = QtWidgets.QToolButton(text=txt)
            b.setFixedSize(30, 26)
            b.setAutoRepeat(True)
            b.setAutoRepeatInterval(120)
            b.clicked.connect(lambda _=False, dx=dx, dy=dy: self.step(dx, dy))
            pad.addWidget(b, r, c)
        b0 = QtWidgets.QToolButton(text="●")
        b0.setFixedSize(30, 26)
        b0.setToolTip("Reset beam shift")
        b0.clicked.connect(self.reset)
        pad.addWidget(b0, 1, 1)
        lay.addLayout(pad, 0, 0, 3, 1)
        lay.addWidget(QtWidgets.QLabel("Step"), 0, 1)
        self.stepc = QtWidgets.QComboBox()
        for pct in (1, 2, 5, 10, 25):
            self.stepc.addItem(f"{pct} % of field", pct / 100)
        self.stepc.setCurrentIndex(2)
        lay.addWidget(self.stepc, 0, 2)
        self.pos = QtWidgets.QLabel()
        lay.addWidget(self.pos, 1, 1, 1, 2)
        hint = QtWidgets.QLabel("Drag the image or double-click a point to centre it.")
        hint.setStyleSheet("color:#9aa")
        hint.setWordWrap(True)
        lay.addWidget(hint, 2, 1, 1, 2)
        self.sync()

    def step(self, dx, dy):
        cfg = self.engine.config
        s = self.stepc.currentData() * cfg.width_um
        try:
            self.engine.shift_beam(dx * s, dy * s)
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
        self.sync()

    def reset(self):
        try:
            self.engine.reset_beam_shift()
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")
        self.sync()

    def sync(self):
        cx, cy = self.engine.config.center_um
        self.pos.setText(f"X {cx:+.3f} µm   Y {cy:+.3f} µm")


# ================================================================== column
class ColumnPanel(QtWidgets.QGroupBox):
    """Magnification, focus/WD, stigmation, alignment – column control (stubs for now)."""

    def __init__(self, microscope: MicroscopeControl, notify: Notify, parent=None):
        super().__init__("Column", parent)
        self.mic = microscope
        self.notify = notify
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

        row = QtWidgets.QHBoxLayout()
        b_af = QtWidgets.QPushButton("Auto focus")
        b_af.clicked.connect(lambda: self._call(self.mic.auto_focus))
        b_as = QtWidgets.QPushButton("Auto stigma")
        b_as.clicked.connect(lambda: self._call(self.mic.auto_stigma))
        b_al = QtWidgets.QPushButton("Alignment ▾")
        menu = QtWidgets.QMenu(b_al)
        for kind in ("Beam", "Aperture", "Stigma X", "Stigma Y"):
            menu.addAction(f"{kind} alignment", lambda k=kind: self._call(lambda: self.mic.alignment(k)))
        b_al.setMenu(menu)
        for b in (b_af, b_as, b_al):
            row.addWidget(b)
        lay.addLayout(row, 4, 0, 1, 4)
        self.l_note = QtWidgets.QLabel(f"Microscope: {microscope.name} (stub – set on the SEM)")
        self.l_note.setStyleSheet("color:#9aa")
        self.l_note.setWordWrap(True)
        lay.addWidget(self.l_note, 5, 0, 1, 4)

    def _call(self, fn):
        try:
            fn()
        except NotAvailable as e:
            self.notify(f"{str(e).capitalize()} is not available yet – use the SEM console.")
        except Exception as e:  # noqa: BLE001
            self.notify(f"error: {e}")

    def sync(self, frame=None):
        mag = self.mic.magnification()
        self.l_mag.setText(f"×{mag:g}" if mag else "×—")
        wd = self.mic.working_distance_mm()
        self.l_wd.setText(f"WD {wd:g} mm" if wd else "WD —")
