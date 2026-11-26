"""Dialogs of the litho window: stage settings, moving a chip, field / beam calibration settings."""
from __future__ import annotations

from PySide6 import QtWidgets

from ..coords import StageAxes
from .tabs import _spin

__all__ = ["StageAxesDialog", "MoveFieldsDialog", "MoveStageDialog"]


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


class MoveStageDialog(QtWidgets.QDialog):
    """Move the stage to the clicked position, a marker of a chip, or a named stage position.

    Every target is ``(label, (x, y), z)`` in SCS µm, *z* the stage Z that has it in focus at the WD
    now (None: no height there, or the WD unknown – Z is kept).  *click*: a target or None;
    *markers*: {chip name: [targets]}; *positions*: [targets]; *chip*: the chip shown first."""

    def __init__(self, click=None, markers=None, positions=(), chip: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Move stage")
        self.click = click
        self.markers = dict(markers or {})
        self.positions = list(positions)
        self.group = QtWidgets.QButtonGroup(self)
        self.b_click = QtWidgets.QRadioButton("to the clicked position")
        self.b_marker = QtWidgets.QRadioButton("to a marker of the chip")
        self.b_position = QtWidgets.QRadioButton("to a stage position")
        for b in (self.b_click, self.b_marker, self.b_position):
            self.group.addButton(b)
        self.l_click = QtWidgets.QLabel(self._text(click) if click is not None else "(no click)")
        self.chip = QtWidgets.QComboBox()
        for n in self.markers:
            self.chip.addItem(n, n)
        self.chip.setCurrentIndex(max(0, self.chip.findData(chip)))
        self.marker = QtWidgets.QComboBox()
        self.position = QtWidgets.QComboBox()
        for k, t in enumerate(self.positions):
            self.position.addItem(self._text(t), k)
        self.focus = QtWidgets.QCheckBox("in focus: Z for the height there at the WD now (else Z stays)")
        self.focus.setChecked(True)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        f = QtWidgets.QGridLayout()
        f.addWidget(self.b_click, 0, 0)
        f.addWidget(self.l_click, 0, 1, 1, 2)
        f.addWidget(self.b_marker, 1, 0)
        f.addWidget(self.chip, 1, 1)
        f.addWidget(self.marker, 1, 2)
        f.addWidget(self.b_position, 2, 0)
        f.addWidget(self.position, 2, 1, 1, 2)
        f.setColumnStretch(2, 1)
        self.b_click.setEnabled(click is not None)
        self.b_marker.setEnabled(any(self.markers.values()))
        self.b_position.setEnabled(bool(self.positions))
        self.bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Move")
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(f)
        lay.addWidget(self.focus)
        lay.addWidget(self.info)
        lay.addWidget(self.bb)
        self.chip.currentIndexChanged.connect(self._fill_markers)
        for w in (self.b_click, self.b_marker, self.b_position, self.focus):
            w.toggled.connect(self._update)
        self.marker.currentIndexChanged.connect(lambda *_: self._select(self.b_marker))
        self.position.currentIndexChanged.connect(lambda *_: self._select(self.b_position))
        self.marker.activated.connect(lambda *_: self._select(self.b_marker))  # the same item picked again
        self.position.activated.connect(lambda *_: self._select(self.b_position))
        self._fill_markers()
        for b in (self.b_click, self.b_marker, self.b_position):
            if b.isEnabled():
                b.setChecked(True)
                break
        self._update()
        self.resize(560, 220)

    @staticmethod
    def _text(t) -> str:
        label, (x, y), z = t
        return f"{label}  ({x:.2f}, {y:.2f}" + (f", Z {z:.1f}" if z is not None else ", Z kept") + ") µm"

    def _select(self, b) -> None:
        if b.isEnabled() and not b.isChecked():
            b.setChecked(True)
        self._update()

    def _fill_markers(self, *_):
        self.marker.blockSignals(True)
        self.marker.clear()
        for k, t in enumerate(self.markers.get(self.chip.currentData(), [])):
            self.marker.addItem(self._text(t), k)
        self.marker.blockSignals(False)
        self._update()

    def target(self):
        """The chosen ``(label, (x, y), z)`` or None."""
        if self.b_click.isChecked():
            return self.click
        if self.b_marker.isChecked():
            ts = self.markers.get(self.chip.currentData(), [])
            k = self.marker.currentData()
            return ts[k] if k is not None and 0 <= k < len(ts) else None
        if self.b_position.isChecked():
            k = self.position.currentData()
            return self.positions[k] if k is not None else None
        return None

    def value(self):
        """(x, y, z | None) stage µm, or None."""
        t = self.target()
        if t is None:
            return None
        (x, y), z = t[1], t[2]
        return float(x), float(y), (None if z is None or not self.focus.isChecked() else float(z))

    def _update(self, *_):
        v = self.value()
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(v is not None)
        if v is None:
            self.info.setText("nothing to move to")
        else:
            self.info.setText(f"stage → ({v[0]:.2f}, {v[1]:.2f}) µm" + (f", Z {v[2]:.1f} µm" if v[2] is not None
                                                                         else ", Z stays"))
