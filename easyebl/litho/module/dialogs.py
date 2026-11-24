"""Dialogs of the litho window: stage axes, fractured-GDS import, moving all fields."""
from __future__ import annotations

import os

from PySide6 import QtWidgets

from ..coords import StageAxes
from ..fracture import ImportOptions, build_schedule, find_fields
from ..gds import GDSDoseMapping, gds_info
from .tabs import _spin, mag_combo

__all__ = ["StageAxesDialog", "ImportDialog", "MoveFieldsDialog"]


class StageAxesDialog(QtWidgets.QDialog):
    """Stage axes against the beam axes (until field correction SCS ↔ BCS exists)."""

    def __init__(self, axes: StageAxes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stage axes")
        self.mx = QtWidgets.QCheckBox("mirror X: moving the stage to +X moves the image of the sample to +x")
        self.my = QtWidgets.QCheckBox("mirror Y: moving the stage to +Y moves the image of the sample to +y")
        self.mx.setChecked(axes.mirror_x)
        self.my.setChecked(axes.mirror_y)
        text = QtWidgets.QLabel(
            "Normally moving the stage to +X brings sample features further in +X into view, so the image of the "
            "sample moves to −x.  Tick an axis where the SEM image moves the other way.  Field centres, markers and "
            "the GDS importer use this; otherwise the stage and beam axes are taken as parallel with equal scale "
            "(field correction is not implemented yet).")
        text.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        for w in (text, self.mx, self.my, bb):
            lay.addWidget(w)
        self.resize(520, 220)

    def value(self) -> StageAxes:
        return StageAxes(self.mx.isChecked(), self.my.isChecked())


class ImportDialog(QtWidgets.QDialog):
    """Fractured GDS → schedule (see :mod:`easyebl.litho.fracture`)."""

    def __init__(self, path: str, library, presets=None, axes: StageAxes | None = None, stage_position=None,
                 magnification: int = 1000, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import fractured GDS – {os.path.basename(path)}")
        self.path = path
        self.library = library
        self.axes = axes or StageAxes()
        self.stage_position = stage_position or (lambda: None)
        self.fields: list = []
        info = gds_info(path)
        self.source = QtWidgets.QComboBox()
        for k, t in (("auto", "auto"), ("references", "references in the top cell"), ("cells", "top-level cells")):
            self.source.addItem(t, k)
        self.top = QtWidgets.QComboBox()
        self.top.addItems(info["top"])
        self.order = QtWidgets.QComboBox()
        for k, t in (("file", "as in the file"), ("raster", "rows, left → right"), ("serpentine", "serpentine")):
            self.order.addItem(t, k)
        pairs = ", ".join(f"{l}/{d}" for l, d in info["layers"][:8]) + ("…" if len(info["layers"]) > 8 else "")
        self.layers = QtWidgets.QLineEdit()
        self.layers.setPlaceholderText(f"all ({pairs})")
        self.dose_mode = QtWidgets.QComboBox()
        self.dose_mode.addItems(["auto", "property", "datatype", "layer", "none"])
        self.dose_table = QtWidgets.QLineEdit()
        self.dose_table.setPlaceholderText("e.g. 0:1.0, 1:1.25 (datatype: dwell factor)")
        self.ox, self.oy = _spin(0, decimals=2), _spin(0, decimals=2)
        b_cur = QtWidgets.QPushButton("Use current stage position")
        b_cur.clicked.connect(self._use_stage)
        self.mag = mag_combo(magnification)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("Keep as is", "")
        for n in presets or []:
            self.preset.addItem(n, n)
        self.current = _spin(0, 0, 1e6, 2, " pA", 10)
        self.current.setSpecialValueText("unknown")
        self.glob = QtWidgets.QCheckBox("Global Align at the first field")
        self.m1x, self.m1y, self.m2x, self.m2y = _spin(-40), _spin(-40), _spin(40), _spin(40)
        self.loc = QtWidgets.QCheckBox("Local XY at every field")
        self.lx, self.ly = _spin(-40), _spin(-40)
        self.view = _spin(10, 0.01, 1e5)
        self.dwell = _spin(1, 0.001, 1e6, 3, " µs", 0.1)
        self.step = _spin(20, 0.1, 1e5, 2, " nm", 1)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        b_scan = QtWidgets.QPushButton("Find fields")
        b_scan.clicked.connect(self.scan)

        def pair(a, b):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(a)
            h.addWidget(b)
            return w

        f = QtWidgets.QFormLayout()
        f.addRow("fields", self.source)
        f.addRow("top cell", self.top)
        f.addRow("order", self.order)
        f.addRow("layers", self.layers)
        f.addRow("dose source", self.dose_mode)
        f.addRow("dose table", self.dose_table)
        f.addRow("layout origin at stage", pair(self.ox, self.oy))
        f.addRow(b_cur)
        f.addRow("magnification", self.mag)
        f.addRow("beam", self.preset)
        f.addRow("nominal current", self.current)
        f.addRow(self.glob)
        f.addRow("marker 1 (WCS)", pair(self.m1x, self.m1y))
        f.addRow("marker 2 (WCS)", pair(self.m2x, self.m2y))
        f.addRow(self.loc)
        f.addRow("local marker (WCS)", pair(self.lx, self.ly))
        f.addRow("marker view", self.view)
        f.addRow("dwell / point", self.dwell)
        f.addRow("point step", self.step)
        f.addRow(b_scan)
        f.addRow(self.info)
        self.replace = QtWidgets.QCheckBox("replace the current schedule (else append the fields)")
        self.replace.setChecked(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Import")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        w = QtWidgets.QWidget()
        w.setLayout(f)
        sc = QtWidgets.QScrollArea()
        sc.setWidget(w)
        sc.setWidgetResizable(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(sc, 1)
        lay.addWidget(self.replace)
        lay.addWidget(bb)
        self.resize(540, 800)
        self.scan()

    def _use_stage(self):
        pos = self.stage_position()
        if pos is None:
            self.info.setText("<span style='color:#f66'>the stage position is not known</span>")
            return
        self.ox.setValue(pos[0])
        self.oy.setValue(pos[1])

    def options(self) -> ImportOptions:
        layers = None
        txt = self.layers.text().strip()
        if txt:
            layers = []
            for part in txt.replace(";", ",").split(","):
                part = part.strip()
                if "/" in part:
                    a, b = part.split("/")
                    layers.append((int(a), int(b)))
                elif part:
                    layers.append(int(part))
        table = {}
        for part in self.dose_table.text().split(","):
            if ":" in part:
                k, v = part.split(":")
                table[int(k)] = float(v)
        return ImportOptions(
            source=self.source.currentData(), top=self.top.currentText() or None, layers=layers,
            dose=GDSDoseMapping(self.dose_mode.currentText(), None, table), order=self.order.currentData(),
            origin_scs=(self.ox.value(), self.oy.value()), axes=self.axes, magnification=self.mag.currentData(),
            beam_preset=self.preset.currentData(), current_pa=self.current.value() or None,
            global_align=self.glob.isChecked(),
            global_markers=((self.m1x.value(), self.m1y.value()), (self.m2x.value(), self.m2y.value())),
            local_xy=self.loc.isChecked(), local_marker=(self.lx.value(), self.ly.value()), view_um=self.view.value(),
            dwell_us=self.dwell.value(), step_nm=self.step.value())

    def scan(self) -> list:
        try:
            self.fields = find_fields(self.path, self.options(), self.library)
        except Exception as e:  # noqa: BLE001 - shown
            self.fields = []
            self.info.setText(f"<span style='color:#f66'>{type(e).__name__}: {e}</span>")
            return []
        if not self.fields:
            self.info.setText("<span style='color:#f66'>no non-empty fields</span>")
            return []
        w = max(max(f.size) for f in self.fields)
        self.info.setText(f"{len(self.fields)} field(s), {sum(f.n_elements for f in self.fields)} elements; "
                          f"largest field {w:.2f} µm")
        return self.fields

    def schedule(self):
        if not self.fields:
            self.scan()
        if not self.fields:
            raise ValueError("no fields to import")
        return build_schedule(self.fields, self.options(), os.path.splitext(os.path.basename(self.path))[0])


class MoveFieldsDialog(QtWidgets.QDialog):
    """Move all fields as a whole: a reference point of the schedule → a target stage position.

    *anchors*: {key: (label, (x, y) | None)} – e.g. the first field's centre, the centre of the
    bbox of all fields, marker 1 of the first field with Global Align (None: not available).
    *stage* / *click*: the current stage position and the clicked plot position (None: unknown)."""

    def __init__(self, anchors: dict, stage=None, click=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Move all fields")
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
        self.shift.setText("every field moves by " + (f"({v[0]:+.3f}, {v[1]:+.3f}) µm" if v else "–"))
        if hasattr(self, "bb"):
            self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(v is not None)
