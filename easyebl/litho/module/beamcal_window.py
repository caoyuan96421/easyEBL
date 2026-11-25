"""The beam calibration window: progress while it images the grid, then the fit.

Opened by the Lithography window when a beam calibration runs (Field → Run beam calibration now).
Shows the grid image, the fitted X / Y scale, shear and pincushion (with the grid's rotation and
shift, which are not used), the fit quality and a map of the distortion over the field; Accept
keeps the calibration for its condition (magnification, WD), Discard drops it.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

__all__ = ["BeamCalWindow"]

_MAX_SHOWN = 1024  # the image is shown binned down to this many pixels across


def _binned(img: np.ndarray, n: int = _MAX_SHOWN) -> np.ndarray:
    a = np.asarray(img, np.float32)
    f = int(np.ceil(max(a.shape) / n))
    if f <= 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


class BeamCalWindow(QtWidgets.QDialog):
    """``decided(bool)`` when the user accepts (True) or discards (False) a result;
    ``abortRequested`` from the Abort button."""

    decided = QtCore.Signal(bool)
    abortRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Beam calibration")
        self.setModal(False)
        self.result = None
        self.awaiting = False

        self.header = QtWidgets.QLabel("–")
        self.header.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        self.step = QtWidgets.QLabel("")

        self.image = pg.GraphicsLayoutWidget()
        self.image.setBackground((18, 18, 22))
        self._vb = self.image.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._img = pg.ImageItem(axisOrder="row-major")
        self._vb.addItem(self._img)
        self.image.setMinimumSize(320, 320)

        self.numbers = QtWidgets.QLabel()
        self.numbers.setTextFormat(QtCore.Qt.RichText)
        self.numbers.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.numbers.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        self.plot = pg.PlotWidget()
        self.plot.setBackground((18, 18, 22))
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "x (XCS)", "µm")
        self.plot.setLabel("left", "y (XCS)", "µm")
        self._field = pg.PlotDataItem(pen=pg.mkPen((150, 150, 170), width=1, style=QtCore.Qt.DashLine))
        self._vec = pg.PlotDataItem(pen=pg.mkPen((255, 120, 80), width=1.5), connect="pairs")
        self._dots = pg.ScatterPlotItem(symbol="o", size=3, pen=None, brush=pg.mkBrush(200, 200, 220))
        for it in (self._field, self._vec, self._dots):
            self.plot.addItem(it)
        self.plot_title = QtWidgets.QLabel()
        self.plot_title.setStyleSheet("color:#9aa")
        self.plot.setMinimumSize(300, 300)

        self.results = QtWidgets.QWidget()
        rl = QtWidgets.QGridLayout(self.results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.numbers, 0, 0, 2, 1)
        rl.addWidget(self.plot_title, 0, 1)
        rl.addWidget(self.plot, 1, 1)
        rl.setColumnStretch(1, 1)
        self.results.hide()

        self.b_abort = QtWidgets.QPushButton("Abort")
        self.b_abort.clicked.connect(self.abortRequested)
        self.b_accept = QtWidgets.QPushButton("Accept")
        self.b_accept.setDefault(True)
        self.b_accept.clicked.connect(lambda: self._decide(True))
        self.b_discard = QtWidgets.QPushButton("Discard")
        self.b_discard.clicked.connect(lambda: self._decide(False))
        self.b_close = QtWidgets.QPushButton("Close")
        self.b_close.clicked.connect(self.close)
        bb = QtWidgets.QHBoxLayout()
        bb.addStretch(1)
        for b in (self.b_abort, self.b_discard, self.b_accept, self.b_close):
            bb.addWidget(b)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.image, 1)
        top.addWidget(self.results, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.header)
        lay.addWidget(self.bar)
        lay.addWidget(self.step)
        lay.addLayout(top, 1)
        lay.addLayout(bb)
        self._buttons("running")
        self.resize(1100, 640)

    # ---------------------------------------------------------------- state
    def _buttons(self, mode: str) -> None:
        """running | ask (Accept / Discard) | done (Close)."""
        self.b_abort.setVisible(mode == "running")
        self.b_accept.setVisible(mode == "ask")
        self.b_discard.setVisible(mode == "ask")
        self.b_close.setVisible(mode != "ask")

    def start(self, position, condition, field_um: float, pixels: int) -> None:
        self.result = None
        self.awaiting = False
        z = position[2] if len(position) > 2 and position[2] is not None else None
        self.header.setText(f"<b>Beam calibration</b> at stage ({position[0]:.1f}, {position[1]:.1f}"
                            + (f", Z {z:.1f}" if z is not None else "") + f") µm · {condition.describe()} · field "
                            f"{field_um:.4g} µm · {pixels} × {pixels} px ({field_um / max(1, pixels) * 1e3:.0f} nm)")
        self.bar.setRange(0, 0)
        self.bar.show()
        self.step.setText("moving to the grid sample, then imaging the whole field…")
        self.step.setStyleSheet("")
        self._img.clear()
        self.results.hide()
        self._buttons("running")

    def set_image(self, image, pixel_um) -> None:
        self._img.setImage(_binned(image), autoLevels=True)
        self._vb.autoRange(padding=0)
        self.step.setText("fitting the grid…")

    def failed(self, message: str) -> None:
        self.awaiting = False
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.step.setText(message)
        self.step.setStyleSheet("color:#f77")
        self._buttons("done")

    def show_result(self, res, ask: bool = True) -> None:
        """Show *res*; *ask*: wait for Accept / Discard (a refused result is only shown)."""
        self.result = res
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        refused = getattr(res, "problem", None) is not None
        self.awaiting = bool(ask and not refused)
        if refused:
            self.step.setText(f"REFUSED: {res.problem}")
            self.step.setStyleSheet("color:#f77")
        else:
            self.step.setText("measured: Accept to keep it for this condition, Discard to drop it"
                              if self.awaiting else "measured")
            self.step.setStyleSheet("color:#5d5")
        self._fill(res)
        self.results.show()
        self._buttons("ask" if self.awaiting else "done")
        self.show()
        self.raise_()

    # ---------------------------------------------------------------- content
    def _fill(self, res) -> None:
        def row(name, value, note=""):
            return (f"<tr><td style='padding-right:10px'>{name}</td><td align='right'><b>{value}</b></td>"
                    f"<td style='color:#9aa;padding-left:8px'>{note}</td></tr>")

        edge = res.edge_distortion_um() if hasattr(res, "edge_distortion_um") else 0.0
        rows = [
            row("X scale", f"{res.sx:.6f}", f"{(res.sx - 1) * 1e6:+.0f} ppm"),
            row("Y scale", f"{res.sy:.6f}", f"{(res.sy - 1) * 1e6:+.0f} ppm"),
            row("shear", f"{res.shear * 1e3:+.3f} mrad"),
            row("pincushion k", f"{res.k:+.3e} µm⁻²", "barrel" if res.k < 0 else "pincushion" if res.k > 0 else ""),
            row("distortion at the corner", f"{edge * 1e3:+.0f} nm"),
            row("fit rms", f"{res.rms_um * 1e3:.1f} nm"),
            row("worst residual", f"{res.max_um * 1e3:.1f} nm"),
            row("image used", f"{res.used * 100:.0f} %", "edges are left out"),
            row("<i>grid rotation</i>", f"{res.rotation_mrad:+.2f} mrad", "not used"),
            row("<i>grid shift</i>", f"({res.shift_um[0]:+.3f}, {res.shift_um[1]:+.3f}) µm", "not used"),
            row("grid pitch", f"{res.pitch_um:g} µm"),
        ]
        cond = res.condition.describe() if res.condition is not None else "?"
        self.numbers.setText("<table>" + "".join(rows) + "</table>"
                             + f"<p style='color:#9aa'>{cond} · field {res.field_um:.4g} µm · "
                               f"{res.pixels[0]} × {res.pixels[1]} px</p>")
        # distortion map: where the beam lands (BCS) − where it was sent (XCS), on a 9 × 9 grid
        h = (res.field_um or 1.0) / 2.0
        u = np.linspace(-h, h, 9)
        gx, gy = np.meshgrid(u, u)
        x = np.column_stack([gx.ravel(), gy.ravel()])
        d = np.atleast_2d(res.to_bcs(x)) - x
        mx = float(np.max(np.hypot(d[:, 0], d[:, 1]))) or 1.0
        scale = 0.1 * h / mx  # the largest vector is a tenth of the half field
        seg = np.empty((2 * len(x), 2))
        seg[0::2], seg[1::2] = x, x + scale * d
        self._vec.setData(seg[:, 0], seg[:, 1])
        self._dots.setData(pos=x)
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
        self._field.setData(sq[:, 0], sq[:, 1])
        self.plot_title.setText(f"beam position error (scale, shear, pincushion): largest {mx * 1e3:.0f} nm, "
                                f"drawn × {scale:.3g}")
        self.plot.autoRange()

    # ---------------------------------------------------------------- actions
    def _decide(self, ok: bool) -> None:
        if not self.awaiting:
            return
        self.awaiting = False
        self.step.setText("accepted: in use for this condition" if ok else "discarded")
        self.step.setStyleSheet("color:#5d5" if ok else "color:#fb4")
        self._buttons("done")
        self.decided.emit(bool(ok))

    def cancel(self) -> None:
        if self.awaiting:
            self.awaiting = False
            self.step.setText("aborted")
            self._buttons("done")

    def closeEvent(self, ev):  # noqa: N802
        if self.awaiting:
            self._decide(False)
        super().closeEvent(ev)
