"""Dialogs of the litho window: stage settings, moving a chip / the stage, the field sizes, the
field calibration."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..coords import StageAxes
from .tabs import _selected_row, _spin, _table

__all__ = ["StageAxesDialog", "MoveFieldsDialog", "MoveStageDialog", "FieldSizesDialog", "FieldCalSetupDialog"]


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
        self.setWindowTitle("Move pattern")
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

    Every target is ``(label, (x, y), z, wd)`` in SCS µm: *z* the stage Z there (a chip: Z_cal; a
    stage position: Z_cal + ΔZ; None: not known – Z stays) and *wd* the focus there (a chip: its
    WD2; a stage position: WD1; None: the focus stays).  *click*: a target or None; *markers*:
    {chip name: [targets]}; *positions*: [targets]; *chip*: the chip shown first."""

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
        self.focus = QtWidgets.QCheckBox("in focus: Z and the WD there (a chip: Z_cal and its WD2; a stage position: "
                                         "Z_cal + ΔZ and WD1; else both stay)")
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
        label, (x, y), z, wd = t
        return (f"{label}  ({x:.2f}, {y:.2f}" + (f", Z {z:.1f}" if z is not None else ", Z kept") + ") µm"
                + (f", WD {wd:.4f} mm" if wd is not None else ""))

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
        """The chosen ``(label, (x, y), z, wd)`` or None."""
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
        """(x, y, z | None, wd | None) – stage µm, WD mm – or None."""
        t = self.target()
        if t is None:
            return None
        (x, y), z, wd = t[1], t[2], t[3]
        on = self.focus.isChecked()
        return (float(x), float(y), None if z is None or not on else float(z),
                None if wd is None or not on else float(wd))

    def _update(self, *_):
        v = self.value()
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(v is not None)
        if v is None:
            self.info.setText("nothing to move to")
        else:
            self.info.setText(f"stage → ({v[0]:.2f}, {v[1]:.2f}) µm" + (f", Z {v[2]:.1f} µm" if v[2] is not None
                                                                         else ", Z stays")
                              + (f", WD {v[3]:.4f} mm" if v[3] is not None else ", the focus stays"))


class FieldSizesDialog(QtWidgets.QDialog):
    """The exposure field sizes: name, magnification and nominal WD (the Microscope window selects one;
    a schedule has one).  :meth:`value` is the edited list of
    :class:`~easyebl.litho.settings.FieldSize` (ValueError for a bad row)."""

    def __init__(self, sizes, field_at_1x_um: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Field sizes")
        self.k = field_at_1x_um
        info = QtWidgets.QLabel("Each field size sets the magnification; selecting it in the Microscope window also sets "
                                "the focus and the stage Z to its nominal WD.  The calibrations are kept per field size.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        self.table = _table(["Name", "Magnification", "Nominal WD (mm)", "Field (µm)"])
        self.table.itemChanged.connect(self._update)
        for s_ in sizes:
            self._add_row(s_.name, s_.magnification, s_.wd_mm)
        b_add = QtWidgets.QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_row("new", 1000, 15.0))
        b_del = QtWidgets.QPushButton("Delete")
        b_del.clicked.connect(self._delete)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(b_add)
        h.addWidget(b_del)
        h.addStretch(1)
        self.msg = QtWidgets.QLabel()
        self.msg.setStyleSheet("color:#f66")
        self.msg.setWordWrap(True)
        self.bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.table, 1)
        lay.addLayout(h)
        lay.addWidget(self.msg)
        lay.addWidget(self.bb)
        self._update()
        self.resize(560, 320)

    def _add_row(self, name, mag, wd) -> None:
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate((name, f"{int(mag)}", f"{float(wd):g}", "")):
            it = QtWidgets.QTableWidgetItem(v)
            if c == 3:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self._update()

    def _delete(self) -> None:
        r = _selected_row(self.table)
        if r >= 0:
            self.table.removeRow(r)
            self._update()

    def value(self) -> list:
        from ..schedule import MAGNIFICATIONS
        from ..settings import FieldSize

        out = []
        for r in range(self.table.rowCount()):
            try:
                name = self.table.item(r, 0).text()
                mag = int(float(self.table.item(r, 1).text()))
                wd = float(self.table.item(r, 2).text())
            except (ValueError, AttributeError) as e:
                raise ValueError(f"row {r + 1}: a name, a whole magnification and a WD in mm") from e
            if mag not in MAGNIFICATIONS:
                raise ValueError(f"row {r + 1}: ×{mag} is not a magnification the SEM accepts")
            try:
                out.append(FieldSize(name, mag, wd))
            except ValueError as e:
                raise ValueError(f"row {r + 1}: {e}") from e
        names = [f.name for f in out]
        if len(set(names)) != len(names):
            raise ValueError("two field sizes have the same name")
        return out

    def _update(self, *_):
        if not hasattr(self, "bb"):  # still being built
            return
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            try:
                mag = float(self.table.item(r, 1).text())
                f = f"{self.k / mag:.4g}" if self.k else "?"
            except (ValueError, AttributeError, ZeroDivisionError):
                f = "?"
            self.table.item(r, 3).setText(f)
        self.table.blockSignals(False)
        try:
            self.value()
            self.msg.clear()
            ok = True
        except ValueError as e:
            self.msg.setText(str(e))
            ok = False
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)


class FieldCalSetupDialog(QtWidgets.QDialog):
    """The field calibration (outside a schedule): the stage position with a feature with structure
    and how to measure, at the field size selected now (*field_size*).  :meth:`value` is the edited
    :class:`~easyebl.litho.fieldcal.FieldCalConfig`; OK (*Measure*) starts it."""

    def __init__(self, config, positions: dict, field_size: str = "", current=None, parent=None):
        super().__init__(parent)
        from ..fieldcal import FieldCalConfig

        self.setWindowTitle("Field calibration")
        self.cfg = FieldCalConfig.from_dict(config.to_dict())
        info = QtWidgets.QLabel(
            f"Field size {field_size or '(none selected in the Microscope window)'}: images at known stage shifts of a "
            "feature with structure give the stage ↔ beam rotation (through the beam calibration of the field size).  "
            "One is kept per field size, across sessions; the schedules use it automatically.  The stage goes to the "
            "position (Z_cal + ΔZ), the focus to WD1."
            + (f"\n\nIn force now: {current}" if current else "\n\nNone in force for this field size."))
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f = QtWidgets.QFormLayout()
        pos = QtWidgets.QComboBox()
        pos.addItem("(choose a stage position)", "")
        for n, p in positions.items():
            dz = getattr(p, "dz", None)
            pos.addItem(f"{n}  ({p.x:.1f}, {p.y:.1f}, " + ("ΔZ " + f"{dz:+.1f})" if dz is not None else "NO ΔZ)"), n)
        if self.cfg.position and pos.findData(self.cfg.position) < 0:
            pos.addItem(f"{self.cfg.position} (unknown)", self.cfg.position)
        pos.setCurrentIndex(max(0, pos.findData(self.cfg.position)))
        pos.currentIndexChanged.connect(lambda _: (setattr(self.cfg, "position", pos.currentData() or ""),
                                                   self._update()))
        self.position = pos
        f.addRow("at", pos)
        pat = QtWidgets.QComboBox()
        pat.addItem("3 × 3 grid (9 images)", "grid")
        pat.addItem("cross (5 images)", "cross")
        pat.setCurrentIndex(max(0, pat.findData(self.cfg.pattern)))
        pat.currentIndexChanged.connect(lambda _: setattr(self.cfg, "pattern", pat.currentData()))
        f.addRow("stage shifts", pat)

        def spin(label, name, scale=1.0, **kw):
            w = _spin(getattr(self.cfg, name) * scale, **kw)
            w.valueChanged.connect(lambda v: setattr(self.cfg, name, float(v) / scale))
            f.addRow(label, w)
            return w

        spin("shift", "step_fraction", 100.0, lo=1, hi=45, decimals=1, suffix=" % of the field", step=5)
        spin("image width", "image_fraction", 100.0, lo=10, hi=100, decimals=0, suffix=" % of the field", step=5)
        px = QtWidgets.QComboBox()
        for n in (128, 256, 512, 1024):
            px.addItem(f"{n} × {n}", n)
        px.setCurrentIndex(max(0, px.findData(self.cfg.image_px)))
        px.currentIndexChanged.connect(lambda _: setattr(self.cfg, "image_px", int(px.currentData())))
        f.addRow("image", px)
        fr = QtWidgets.QSpinBox()
        fr.setRange(1, 64)
        fr.setValue(int(self.cfg.frames))
        fr.valueChanged.connect(lambda v: setattr(self.cfg, "frames", int(v)))
        f.addRow("frames averaged", fr)
        spin("settle after a move", "settle_s", lo=0, hi=60, decimals=2, suffix=" s", step=0.1)
        spin("refuse a fit worse than", "max_residual_px", lo=0.1, hi=50, decimals=1, suffix=" px", step=0.5)
        dn = spin("de-noise", "denoise_px", lo=0, hi=20, decimals=1, suffix=" px", step=0.5)
        dn.setSpecialValueText("off")
        dn.setToolTip("Gaussian low-pass (σ) before matching: about the size of the feature's detail; more for "
                      "noisier images")
        bg = spin("flatten (background)", "background_fraction", 100.0, lo=0, hi=50, decimals=1,
                  suffix=" % of the image", step=2.5)
        bg.setSpecialValueText("off")
        wh = spin("whitening", "whitening", lo=0, hi=1, decimals=2, suffix="", step=0.1)
        wh.setToolTip("1: pure phase correlation; lower is more robust on noisy, low-contrast images")
        self.bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Measure")
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(f)
        lay.addWidget(self.bb)
        self.field_size = field_size
        self._update()
        self.resize(560, 560)

    def _update(self) -> None:
        self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(self.cfg.position and self.field_size))

    def value(self):
        return self.cfg
