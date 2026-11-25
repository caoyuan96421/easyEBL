"""The field calibration window: live progress while it measures, then the diagnostics.

Opened by the Lithography window whenever a field calibration runs (Field → Run field
calibration now, or a Field correction step).  While it measures: the step, the stage shift,
the reference and the latest image.  Afterwards: rotation, scale, stretch / shear, the mirror
check against the stage axes, drift, the fit residual, a table of every image and a plot of
the predicted against the measured image shifts.  A new calibration is used only after
*Accept*; *Discard* (or closing the window) drops it, and stops a schedule run.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..coords import StageAxes
from ..fieldcal import FieldCalResult

__all__ = ["FieldCalWindow"]


def _mirror_text(axes: StageAxes) -> str:
    return f"mirror X {'on' if axes.mirror_x else 'off'}, mirror Y {'on' if axes.mirror_y else 'off'}"


class FieldCalWindow(QtWidgets.QDialog):
    """``decided(bool)`` when the user accepts (True) or discards (False) a result;
    ``abortRequested`` from the Abort button; ``axesFixRequested(StageAxes)`` from the mirror
    banner's button."""

    decided = QtCore.Signal(bool)
    abortRequested = QtCore.Signal()
    axesFixRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Field calibration")
        self.setModal(False)
        self.result: FieldCalResult | None = None
        self.axes = StageAxes()
        self.awaiting = False  # a result waits for Accept / Discard
        self.in_schedule = False

        self.header = QtWidgets.QLabel("–")
        self.header.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.step = QtWidgets.QLabel("")

        # --- images
        self.images = pg.GraphicsLayoutWidget()
        self.images.setBackground((18, 18, 22))
        self._img = {}
        for col, name in enumerate(("reference", "latest")):
            vb = self.images.addViewBox(row=0, col=col)
            vb.setAspectLocked(True)
            vb.invertY(True)
            vb.setMouseEnabled(False, False)
            it = pg.ImageItem(axisOrder="row-major")
            vb.addItem(it)
            self._img[name] = (vb, it)
            self.images.addLabel(name, row=1, col=col)
        self.images.setMinimumHeight(160)
        self.images.setMaximumHeight(230)

        # --- results
        self.banner = QtWidgets.QFrame()
        self.banner.setStyleSheet("QFrame{background:#5a3b10;border-radius:4px}")
        bl = QtWidgets.QHBoxLayout(self.banner)
        self.banner_text = QtWidgets.QLabel()
        self.banner_text.setWordWrap(True)
        self.b_fix = QtWidgets.QPushButton("Set the stage axes to match")
        self.b_fix.clicked.connect(self._fix_axes)
        bl.addWidget(self.banner_text, 1)
        bl.addWidget(self.b_fix)
        self.banner.hide()

        self.numbers = QtWidgets.QLabel()
        self.numbers.setTextFormat(QtCore.Qt.RichText)
        self.numbers.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.numbers.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["stage shift (µm)", "predicted (µm)", "measured (µm)", "fitted (µm)",
                                              "residual (nm)", "match"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.plot = pg.PlotWidget()
        self.plot.setBackground((18, 18, 22))
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "image shift x", "µm")
        self.plot.setLabel("left", "image shift y", "µm")
        self.plot.addLegend(offset=(5, 5))
        self._pred = pg.ScatterPlotItem(symbol="o", size=11, pen=pg.mkPen((150, 150, 170), width=1.5), brush=None,
                                        name="predicted (stage axes)")
        self._meas = pg.ScatterPlotItem(symbol="+", size=13, pen=pg.mkPen((90, 220, 120), width=2),
                                        name="measured")
        self._res = pg.PlotDataItem(pen=pg.mkPen((255, 120, 80), width=2), connect="pairs",
                                    name="residual × 100")
        for it in (self._pred, self._meas, self._res):
            self.plot.addItem(it)
        self.plot.setMinimumSize(300, 260)

        self.results = QtWidgets.QWidget()
        rl = QtWidgets.QGridLayout(self.results)
        rl.setContentsMargins(0, 0, 0, 0)
        self.numbers.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        self.table.setMinimumHeight(180)
        rl.addWidget(self.banner, 0, 0, 1, 2)
        rl.addWidget(self.numbers, 1, 0)
        rl.addWidget(self.plot, 1, 1)
        rl.addWidget(self.table, 2, 0, 1, 2)
        rl.setColumnStretch(0, 0)
        rl.setColumnStretch(1, 1)
        rl.setRowStretch(1, 3)
        rl.setRowStretch(2, 2)
        self.results.hide()

        # --- buttons
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

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.header)
        lay.addWidget(self.bar)
        lay.addWidget(self.step)
        lay.addWidget(self.images, 1)
        lay.addWidget(self.results, 2)
        lay.addLayout(bb)
        self._buttons("running")
        self.resize(900, 900)

    # ---------------------------------------------------------------- state
    def _buttons(self, mode: str) -> None:
        """running | ask (Accept / Discard) | done (Close)."""
        self.b_abort.setVisible(mode == "running")
        self.b_accept.setVisible(mode == "ask")
        self.b_discard.setVisible(mode == "ask")
        self.b_close.setVisible(mode != "ask")

    def start(self, n: int, position, condition, field_um: float, axes: StageAxes, in_schedule: bool = False):
        self.result = None
        self.awaiting = False
        self.axes = axes
        self.in_schedule = in_schedule
        self.header.setText(f"<b>Field calibration</b> at stage ({position[0]:.1f}, {position[1]:.1f}) µm · "
                            f"{condition.describe()} · field {field_um:.4g} µm · {n} images · stage axes: "
                            f"{_mirror_text(axes)}")
        self.bar.setRange(0, n)
        self.bar.setValue(0)
        self.bar.show()
        self.step.setText("moving to the calibration position…")
        self.step.setStyleSheet("")
        for _, it in self._img.values():
            it.clear()
        self.images.show()
        self.results.hide()
        self._buttons("running")

    def add_image(self, k: int, n: int, shift, image, reference) -> None:
        self.bar.setRange(0, n)
        self.bar.setValue(k + 1)
        self.step.setText(f"image {k + 1} / {n}: stage shift ({shift[0]:+.2f}, {shift[1]:+.2f}) µm")
        self._img["reference"][1].setImage(np.asarray(reference, float), autoLevels=True)
        self._img["latest"][1].setImage(np.asarray(image, float), autoLevels=True)
        for vb, _ in self._img.values():
            vb.autoRange(padding=0)

    def failed(self, message: str) -> None:
        """The calibration stopped before a result (aborted, stage error …)."""
        self.awaiting = False
        self.step.setText(message)
        self.step.setStyleSheet("color:#f77")
        self._buttons("done")

    def show_result(self, res: FieldCalResult, axes: StageAxes, ask: bool = True) -> None:
        """Show *res*; *ask*: wait for Accept / Discard (a refused result is only shown)."""
        self.result = res
        self.axes = axes
        self.bar.setValue(self.bar.maximum())
        refused = res.problem is not None
        self.awaiting = bool(ask and not refused)
        if refused:
            self.step.setText(f"REFUSED: {res.problem}")
            self.step.setStyleSheet("color:#f77")
        else:
            self.step.setText("measured: Accept to use it, Discard to drop it" if self.awaiting else "measured")
            self.step.setStyleSheet("color:#5d5")
        self._fill(res, axes)
        self.results.show()
        self._buttons("ask" if self.awaiting else "done")
        self.show()
        self.raise_()

    def set_axes(self, axes: StageAxes) -> None:
        self.axes = axes
        if self.result is not None:
            self._fill(self.result, axes)

    # ---------------------------------------------------------------- content
    def _fill(self, res: FieldCalResult, axes: StageAxes) -> None:
        d = res.diagnostics(axes)
        m = d["matrix"]

        def row(name, value, note=""):
            return (f"<tr><td style='padding-right:10px'>{name}</td><td align='right'><b>{value}</b></td>"
                    f"<td style='color:#9aa;padding-left:8px'>{note}</td></tr>")

        weak = d["weakest_match"]
        rows = [
            row("rotation", f"{d['rotation_mrad']:+.3f} mrad", f"{d['rotation_deg']:+.4f}°"),
            row("scale (mean)", f"{d['scale']:.5f}", f"{(d['scale'] - 1) * 100:+.3f} %"),
            row("scale X", f"{d['scale_x']:.5f}", f"{(d['scale_x'] - 1) * 100:+.3f} %"),
            row("scale Y", f"{d['scale_y']:.5f}", f"{(d['scale_y'] - 1) * 100:+.3f} %"),
            row("stretch (Y/X − 1)", f"{d['stretch'] * 1e3:+.2f} ‰"),
            row("shear", f"{d['shear_mrad']:+.3f} mrad"),
            row("drift", f"({d['drift_um'][0] * 1e3:+.0f}, {d['drift_um'][1] * 1e3:+.0f}) nm",
                "common to all images"),
            row("fit rms", f"{d['rms_um'] * 1e3:.1f} nm", f"{d['rms_px']:.2f} px"),
            row("worst residual", f"{d['max_residual_um'] * 1e3:.1f} nm"),
            row("weakest match", f"{weak:.2f}", "correlation peak 0…1" + (" – low" if weak < 0.05 else "")),
            row("stage axes", "MISMATCH" if d["mirror_mismatch"] else "agree", _mirror_text(axes)),
        ]
        if d["mirror_mismatch"]:
            rows.insert(0, row("<i>numbers for</i>", _mirror_text(d["axes_used"]), "as measured"))
        rows += [
            row("B·M", f"[[{m[0, 0]:+.5f}, {m[0, 1]:+.5f}],", ""),
            row("", f"[{m[1, 0]:+.5f}, {m[1, 1]:+.5f}]]", "identity: as nominal"),
        ]
        self.numbers.setText("<table>" + "".join(rows) + "</table>"
                             + f"<p style='color:#9aa'>{res.condition.describe()} · field {res.field_um:.4g} µm · "
                               f"pixel {res.pixel_um * 1e3:.1f} nm</p>")
        if d["mirror_mismatch"]:
            sug = d["suggested_axes"]
            self.banner_text.setText(
                f"The images move against the stage axes setting ({_mirror_text(axes)}).  The measurement says "
                f"<b>{_mirror_text(sug)}</b>.  The stage axes are a hardware setting: change them in the easyEBL "
                "menu, Hardware → Configure… → Devices → Microscope, then reopen the Lithography window "
                "(Discard this result).")
            self.b_fix.hide()
            self.banner.show()
        else:
            self.banner.hide()

        s = np.array([a for a, _ in res.shifts], float).reshape(-1, 2)
        meas = np.array([b for _, b in res.shifts], float).reshape(-1, 2)
        pred = np.array(res.predicted, float).reshape(-1, 2) if res.predicted else np.full_like(meas, np.nan)
        fitted = -(s @ res.stage_to_beam.T) + np.asarray(res.drift, float)
        resid = meas - fitted
        self.table.setRowCount(len(s))
        for r in range(len(s)):
            q = res.quality[r] if r < len(res.quality) else float("nan")
            cells = [f"({s[r, 0]:+.2f}, {s[r, 1]:+.2f})", f"({pred[r, 0]:+.3f}, {pred[r, 1]:+.3f})",
                     f"({meas[r, 0]:+.3f}, {meas[r, 1]:+.3f})", f"({fitted[r, 0]:+.3f}, {fitted[r, 1]:+.3f})",
                     f"({resid[r, 0] * 1e3:+.0f}, {resid[r, 1] * 1e3:+.0f})", f"{q:.2f}"]
            for c, t in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(t)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, c, it)
        self._pred.setData(pos=pred)
        self._meas.setData(pos=meas)
        tips = fitted + 100.0 * resid
        seg = np.empty((2 * len(fitted), 2))
        seg[0::2], seg[1::2] = fitted, tips
        self._res.setData(seg[:, 0], seg[:, 1])
        self.plot.autoRange()

    # ---------------------------------------------------------------- actions
    def _decide(self, ok: bool) -> None:
        if not self.awaiting:
            return
        self.awaiting = False
        self.step.setText("accepted: in use" if ok else "discarded")
        self.step.setStyleSheet("color:#5d5" if ok else "color:#fb4")
        self._buttons("done")
        self.decided.emit(bool(ok))

    def _fix_axes(self) -> None:
        sug = self.b_fix.property("axes")
        if sug is None:
            return
        self.axesFixRequested.emit(sug)
        if self.in_schedule and self.awaiting:
            self._decide(False)
        else:
            self.set_axes(sug)

    def cancel(self) -> None:
        """The run was aborted while a result waited."""
        if self.awaiting:
            self.awaiting = False
            self.step.setText("aborted")
            self._buttons("done")

    def closeEvent(self, ev):  # noqa: N802
        if self.awaiting:
            self._decide(False)
        super().closeEvent(ev)
