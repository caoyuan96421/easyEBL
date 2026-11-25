"""Dialogs of the litho window: stage settings, moving a chip, field / beam calibration settings."""
from __future__ import annotations

from PySide6 import QtWidgets

from ..beamcal import BeamCalConfig
from ..coords import StageAxes
from ..fieldcal import FieldCalConfig
from .tabs import _spin

__all__ = ["StageAxesDialog", "MoveFieldsDialog", "FieldCalDialog", "BeamCalDialog"]


class StageAxesDialog(QtWidgets.QDialog):
    """Stage settings of the litho module: the position tolerance.  The stage axes (mirror) are shown
    only: they are a hardware setting (Hardware → Configure… → Microscope)."""

    def __init__(self, axes: StageAxes, parent=None, tolerance_um: float = 0.5):
        super().__init__(parent)
        self.setWindowTitle("Stage settings")
        self.axes = axes
        text = QtWidgets.QLabel(
            f"Stage axes: {axes.describe()}.\nThey are set in the easyEBL menu: Hardware → Configure… → "
            "Devices → Microscope (mirror X: moving the stage to +X moves the image of the sample to +x).")
        text.setWordWrap(True)
        self.tol = _spin(tolerance_um, 0.01, 1000, 2, " µm", 0.1)
        self.tol.setToolTip("a schedule step fails when the stage reads back farther than this from its target")
        tf = QtWidgets.QFormLayout()
        tf.addRow("position tolerance", self.tol)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(text)
        lay.addLayout(tf)
        lay.addWidget(bb)
        self.resize(520, 200)

    def value(self) -> StageAxes:
        return self.axes

    def tolerance(self) -> float:
        return float(self.tol.value())


class MoveFieldsDialog(QtWidgets.QDialog):
    """Move a chip (its nominal CCS origin, with all its fields): a reference point → a target stage position.

    *anchors*: {key: (label, (x, y) | None)} – e.g. the first field's centre, the centre of the
    bbox of all fields, marker 1 of the first field with Global Align (None: not available).
    *stage* / *click*: the current stage position and the clicked plot position (None: unknown)."""

    def __init__(self, anchors: dict, stage=None, click=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Move the chip")
        self.anchors = anchors
        self.targets = {"stage": ("current stage position", stage), "click": ("the clicked position", click),
                        "abs": ("stage coordinate", None)}
        self.from_group = QtWidgets.QButtonGroup(self)
        self.to_group = QtWidgets.QButtonGroup(self)
        gf = QtWidgets.QGroupBox("Move")
        lf = QtWidgets.QVBoxLayout(gf)
        self.from_buttons = {}
        for k, (label, pt) in anchors.items():
            b = QtWidgets.QRadioButton(label + (f"  ({pt[0]:.2f}, {pt[1]:.2f})" if pt is not None else "  (none)"))
            b.setEnabled(pt is not None)
            self.from_group.addButton(b)
            self.from_buttons[k] = b
            lf.addWidget(b)
        gt = QtWidgets.QGroupBox("to")
        lt = QtWidgets.QVBoxLayout(gt)
        self.to_buttons = {}
        for k, (label, pt) in self.targets.items():
            text = label if k == "abs" else label + (f"  ({pt[0]:.2f}, {pt[1]:.2f})" if pt is not None else
                                                     "  (unknown)")
            b = QtWidgets.QRadioButton(text)
            b.setEnabled(k == "abs" or pt is not None)
            self.to_group.addButton(b)
            self.to_buttons[k] = b
            lt.addWidget(b)
        self.ax, self.ay = _spin(0, decimals=3), _spin(0, decimals=3)
        h = QtWidgets.QHBoxLayout()
        h.addSpacing(20)
        h.addWidget(self.ax)
        h.addWidget(self.ay)
        lt.addLayout(h)
        self.shift = QtWidgets.QLabel()
        for b in list(self.from_buttons.values()) + list(self.to_buttons.values()):
            b.toggled.connect(self._update)
        self.ax.valueChanged.connect(self._update)
        self.ay.valueChanged.connect(self._update)
        for k, b in self.from_buttons.items():
            if b.isEnabled():
                b.setChecked(True)
                pt = anchors[k][1]
                self.ax.setValue(pt[0])
                self.ay.setValue(pt[1])
                break
        for k in ("click", "stage", "abs"):
            if self.to_buttons[k].isEnabled():
                self.to_buttons[k].setChecked(True)
                break
        self.bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Move")
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(gf)
        lay.addWidget(gt)
        lay.addWidget(self.shift)
        lay.addWidget(self.bb)
        self._update()

    def _key(self, buttons: dict) -> str | None:
        return next((k for k, b in buttons.items() if b.isChecked()), None)

    def source(self):
        k = self._key(self.from_buttons)
        return None if k is None else self.anchors[k][1]

    def target(self):
        k = self._key(self.to_buttons)
        if k is None:
            return None
        if k == "abs":
            return (self.ax.value(), self.ay.value())
        return self.targets[k][1]

    def value(self) -> tuple | None:
        """(dx, dy) stage µm, or None."""
        a, b = self.source(), self.target()
        if a is None or b is None:
            return None
        return (float(b[0] - a[0]), float(b[1] - a[1]))

    def _update(self, *_):
        self.ax.setEnabled(self.to_buttons["abs"].isChecked())
        self.ay.setEnabled(self.to_buttons["abs"].isChecked())
        v = self.value()
        self.shift.setText("the chip origin moves by " + (f"({v[0]:+.3f}, {v[1]:+.3f}) µm" if v else "–"))
        if hasattr(self, "bb"):
            self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(v is not None)



class FieldCalDialog(QtWidgets.QDialog):
    """Field calibration settings: where (a named stage position or coordinates) and how."""

    def __init__(self, config: FieldCalConfig, positions: dict, stage_position=None, last=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Field calibration")
        self.stage_position = stage_position or (lambda: None)
        intro = QtWidgets.QLabel(
            "At a fixed stage position with a feature that has structure (a Faraday cup, a calibration sample), "
            "images are taken at known stage shifts; their shifts give the stage → beam map: the rotation between "
            "stage and beam (the images go through the beam calibration of the condition: run that first).  It is "
            "valid for the magnification, beam preset and WD it was measured at.")
        intro.setWordWrap(True)
        self.pos = QtWidgets.QComboBox()
        self.pos.addItem("the coordinates below", "")
        for n in positions:
            self.pos.addItem(n, n)
        if config.position and self.pos.findData(config.position) < 0:
            self.pos.addItem(f"{config.position} (unknown)", config.position)
        self.pos.setCurrentIndex(max(0, self.pos.findData(config.position)))
        self.x, self.y = _spin(config.x, 0, 1e6, 2), _spin(config.y, 0, 1e6, 2)
        b_cur = QtWidgets.QPushButton("Use current stage position")
        b_cur.clicked.connect(self._use_stage)
        self.pattern = QtWidgets.QComboBox()
        self.pattern.addItem("3 × 3 grid (9 images)", "grid")
        self.pattern.addItem("cross (5 images)", "cross")
        self.pattern.setCurrentIndex(max(0, self.pattern.findData(config.pattern)))
        self.step = _spin(config.step_fraction * 100, 1, 45, 1, " % of the field", 5)
        self.width = _spin(config.image_fraction * 100, 10, 100, 0, " % of the field", 5)
        self.px = QtWidgets.QComboBox()
        for n in (128, 256, 512, 1024):
            self.px.addItem(f"{n} × {n}", n)
        self.px.setCurrentIndex(max(0, self.px.findData(config.image_px)))
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, 64)
        self.frames.setValue(config.frames)
        self.settle = _spin(config.settle_s, 0, 60, 2, " s", 0.1)
        self.maxres = _spin(config.max_residual_px, 0.1, 50, 1, " px", 0.5)
        self.denoise = _spin(config.denoise_px, 0, 20, 1, " px", 0.5)
        self.denoise.setSpecialValueText("off")
        self.denoise.setToolTip("Gaussian low-pass (σ) before matching: about the size of the feature's detail; "
                                "more for noisier images")
        self.bgfrac = _spin(config.background_fraction * 100, 0, 50, 1, " % of the image", 2.5)
        self.bgfrac.setSpecialValueText("off")
        self.bgfrac.setToolTip("a blur this wide (σ) is subtracted: brightness gradients and charging do not match")
        self.whiten = _spin(config.whitening, 0, 1, 2, "", 0.1)
        self.whiten.setToolTip("1: pure phase correlation (sharp, but noise counts as much as the feature); "
                               "lower is more robust on noisy, low-contrast images")
        f = QtWidgets.QFormLayout()
        f.addRow("position", self.pos)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.x)
        h.addWidget(self.y)
        f.addRow("X, Y", h)
        f.addRow(b_cur)
        f.addRow("stage shifts", self.pattern)
        f.addRow("shift", self.step)
        f.addRow("image width", self.width)
        f.addRow("image", self.px)
        f.addRow("frames averaged", self.frames)
        f.addRow("settle after a move", self.settle)
        f.addRow("refuse a fit worse than", self.maxres)
        f.addRow("de-noise", self.denoise)
        f.addRow("flatten (background)", self.bgfrac)
        f.addRow("whitening", self.whiten)
        self.last = QtWidgets.QLabel("last calibration: " + (last if last else "none"))
        self.last.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(f)
        lay.addWidget(self.last)
        lay.addWidget(bb)
        self.pos.currentIndexChanged.connect(self._sync)
        self._sync()
        self.resize(520, 560)

    def _sync(self, *_):
        on = not self.pos.currentData()
        self.x.setEnabled(on)
        self.y.setEnabled(on)

    def _use_stage(self):
        pos = self.stage_position()
        if pos is not None:
            self.pos.setCurrentIndex(0)
            self.x.setValue(pos[0])
            self.y.setValue(pos[1])

    def value(self) -> FieldCalConfig:
        return FieldCalConfig(position=self.pos.currentData() or "", x=self.x.value(), y=self.y.value(),
                              pattern=self.pattern.currentData(), step_fraction=self.step.value() / 100.0,
                              image_fraction=self.width.value() / 100.0, image_px=int(self.px.currentData()),
                              frames=int(self.frames.value()), settle_s=self.settle.value(),
                              max_residual_px=self.maxres.value(), denoise_px=self.denoise.value(),
                              background_fraction=self.bgfrac.value() / 100.0, whitening=self.whiten.value())


class BeamCalDialog(QtWidgets.QDialog):
    """Beam calibration settings: the named stage position of the grid sample (X, Y, Z) and the image."""

    def __init__(self, config: BeamCalConfig, positions: dict, last=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Beam calibration")
        intro = QtWidgets.QLabel(
            "One image of a calibration sample with a square grid (1 µm pitch) over the whole field of view, at "
            "≤ 0.1 µm pixels (5000 × 5000 for a 500 µm field).  The fit gives the beam's X / Y scale, shear "
            "and pincushion distortion (the grid's rotation and shift are left out).  Valid for the magnification and "
            "WD it was measured at; run it before the field calibration.")
        intro.setWordWrap(True)
        self.pos = QtWidgets.QComboBox()
        for n, p in positions.items():
            z = p[2] if len(p) > 2 else None
            self.pos.addItem(f"{n}  ({p[0]:.1f}, {p[1]:.1f}" + (f", Z {z:.1f})" if z is not None else ", Z not stored)"), n)
        if config.position and self.pos.findData(config.position) < 0:
            self.pos.addItem(f"{config.position} (unknown)", config.position)
        if config.position:
            self.pos.setCurrentIndex(max(0, self.pos.findData(config.position)))
        self.px = QtWidgets.QSpinBox()
        self.px.setRange(200, 20000)
        self.px.setSingleStep(500)
        self.px.setValue(int(config.pixels))
        self.px.setSuffix(" px across")
        self.pitch = _spin(config.pitch_um, 0.05, 100, 3, " µm", 0.1)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, 64)
        self.frames.setValue(config.frames)
        self.maxrms = _spin(config.max_rms_um * 1e3, 1, 5000, 0, " nm", 10)
        f = QtWidgets.QFormLayout()
        f.addRow("grid sample at", self.pos)
        f.addRow("image", self.px)
        f.addRow("grid pitch", self.pitch)
        f.addRow("frames averaged", self.frames)
        f.addRow("refuse a fit worse than", self.maxrms)
        self.last = QtWidgets.QLabel("for the condition now: " + (last if last else "none"))
        self.last.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(intro)
        if not positions:
            w = QtWidgets.QLabel("No stage positions yet: add the grid sample's position in the Stage Positions tab.")
            w.setStyleSheet("color:#fb4")
            w.setWordWrap(True)
            lay.addWidget(w)
        lay.addLayout(f)
        lay.addWidget(self.last)
        lay.addWidget(bb)
        self.resize(560, 420)

    def value(self) -> BeamCalConfig:
        return BeamCalConfig(pitch_um=self.pitch.value(), pixels=int(self.px.value()), frames=int(self.frames.value()),
                             max_rms_um=self.maxrms.value() * 1e-3, position=self.pos.currentData() or "")
