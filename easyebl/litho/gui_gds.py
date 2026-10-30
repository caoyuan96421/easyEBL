"""GDS lithography debug GUI: load, map doses, check field fit, expose.

Run:  ``python -m easyebl.litho.gui_gds [file.gds]``
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from ..gui.common import QtCore, QtWidgets, Session, run_standalone
from .gds import GDSDoseMapping, gds_info, read_gds, write_gds
from .geometry import Pattern, Polygon, rectangle, regular_polygon
from .gui_common import ExposurePanel

__all__ = ["GDSWidget", "make_demo_gds", "main"]


def make_demo_gds(path: str) -> str:
    """Electrode-like demo layout with per-polygon dose factors (as properties and datatypes)."""
    els = []
    # four contact pads with tapered leads converging on a 2 µm gap
    for k, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        px, py = sx * 30, sy * 25
        els.append(rectangle(px - 8, py - 8, px + 8, py + 8, dose_factor=1.0, layer=1, datatype=0))
        tip = np.array([sx * 1.0, sy * 1.0])
        base = np.array([px - sx * 8, py])
        lead = Polygon(np.array([[base[0], base[1] - 1.5], [base[0], base[1] + 1.5], [tip[0], tip[1] + 0.2],
                                 [tip[0], tip[1] - 0.2]]), dose_factor=1.25, layer=1, datatype=1)
        els.append(lead)
    els.append(regular_polygon(0, 0, 0.6, 32, dose_factor=1.6, layer=1, datatype=2))
    write_gds(Pattern(els), path, datatype_from_dose={1.0: 0, 1.25: 1, 1.6: 2})
    return path


class GDSWidget(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None, path: str | None = None):
        super().__init__(parent)
        self.session = session
        self.path = None
        self._pattern: Pattern | None = None

        self.file_label = QtWidgets.QLabel("(no file)")
        b_open = QtWidgets.QPushButton("Open GDS…")
        b_open.clicked.connect(self._open_dialog)
        b_demo = QtWidgets.QPushButton("Generate demo GDS")
        b_demo.clicked.connect(self._demo)
        self.cell = QtWidgets.QComboBox()
        self.layers = QtWidgets.QListWidget()
        self.layers.setMaximumHeight(110)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["auto", "property", "datatype", "layer", "none"])
        self.attr = QtWidgets.QLineEdit("")
        self.attr.setPlaceholderText("any")
        self.table = QtWidgets.QLineEdit("")
        self.table.setPlaceholderText("e.g. 0:1.0, 1:1.25, 2:1.6")
        self.scale = QtWidgets.QDoubleSpinBox(decimals=4, minimum=1e-6, maximum=1e6, value=1.0)
        self.center = QtWidgets.QCheckBox("centre pattern in field")
        self.center.setChecked(False)
        b_load = QtWidgets.QPushButton("(Re)load pattern")
        b_load.clicked.connect(self.load)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)

        g = QtWidgets.QGroupBox("GDS source")
        f = QtWidgets.QFormLayout(g)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(b_open)
        hb.addWidget(b_demo)
        f.addRow(hb)
        f.addRow("file", self.file_label)
        f.addRow("cell", self.cell)
        f.addRow("layers", self.layers)
        f.addRow("dose source", self.mode)
        f.addRow("property attr", self.attr)
        f.addRow("datatype/layer table", self.table)
        f.addRow("value scale", self.scale)
        f.addRow(self.center)
        f.addRow(b_load)
        f.addRow(self.info)

        self.panel = ExposurePanel(session, self.pattern, owner="litho-gds")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(g)
        left.addWidget(self.panel.controls, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(self.panel.views, 1)
        if path:
            self.open(path)

    # ----------------------------------------------------------------- file
    def _open_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open GDS", "", "GDSII (*.gds *.gds2 *.gdsii);;All (*)")
        if p:
            self.open(p)

    def _demo(self):
        d = tempfile.mkdtemp(prefix="easyebl_")
        p = make_demo_gds(os.path.join(d, "demo_electrodes.gds"))
        self.table.setText("0:1.0, 1:1.25, 2:1.6")
        self.open(p)

    def open(self, path: str):
        try:
            info = gds_info(path)
        except Exception as e:  # noqa: BLE001
            self.panel.log.log(f"ERROR reading {path}: {e}")
            return
        self.path = path
        self.file_label.setText(os.path.basename(path))
        self.cell.clear()
        self.cell.addItems(info["top"] + [c for c in info["cells"] if c not in info["top"]])
        self.layers.clear()
        for l, d in info["layers"]:
            it = QtWidgets.QListWidgetItem(f"{l}/{d}")
            it.setData(QtCore.Qt.UserRole, (l, d))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.layers.addItem(it)
        self.load()

    def mapping(self) -> GDSDoseMapping:
        table = {}
        txt = self.table.text().strip()
        if txt:
            for part in txt.split(","):
                k, v = part.split(":")
                table[int(k)] = float(v)
        attr = self.attr.text().strip()
        return GDSDoseMapping(self.mode.currentText(), int(attr) if attr else None, table, 1.0, self.scale.value())

    def load(self):
        if not self.path:
            return
        try:
            layers = [self.layers.item(i).data(QtCore.Qt.UserRole) for i in range(self.layers.count())
                      if self.layers.item(i).checkState() == QtCore.Qt.Checked]
            st = {}
            pat = read_gds(self.path, self.cell.currentText() or None, layers, self.mapping(), stats=st)
            if self.center.isChecked():
                pat = pat.centered()
            self._pattern = pat
            s = pat.summary()
            cal = self.session.calibration
            pts = np.vstack([np.atleast_2d(e.points) for e in pat]) if len(pat) else np.zeros((0, 2))
            fits = bool(np.all(cal.contains(pts))) if len(pts) else True
            self.info.setText(
                f"{s['polygons']} polygons, {s['lines']} lines · area {s['area_um2']:.1f} µm² · "
                f"dose factors {s['dose_factor_range']} · sources {st} · "
                + ("fits field" if fits else "<b style='color:red'>OUTSIDE FIELD</b>")
                + f" ({cal.field_size_um:g} µm)"
            )
            self.panel.compile_preview()
        except Exception as e:  # noqa: BLE001
            self.panel.log.log(f"ERROR {type(e).__name__}: {e}")

    def pattern(self) -> Pattern:
        if self._pattern is None:
            raise RuntimeError("no GDS pattern loaded")
        return self._pattern


def main(argv=None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    path = None
    if argv and not argv[0].startswith("-"):
        path = argv.pop(0)
    return run_standalone(lambda s: GDSWidget(s, path=path), "GDS lithography", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
