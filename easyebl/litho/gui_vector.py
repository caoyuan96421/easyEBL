"""Vector-scan lithography debug GUI: single polygon (raster fill), lines and dots.

Run:  ``python -m easyebl.litho.gui_vector``
"""
from __future__ import annotations

import numpy as np

from ..gui.common import QtWidgets, Session, format_points, parse_points, run_standalone
from .geometry import Dot, Pattern, Polygon, Polyline, regular_polygon
from .gui_common import ExposurePanel

__all__ = ["VectorLithoWidget", "PRESETS", "main"]

PRESETS = {
    "rectangle 10×6 µm": [(-5, -3), (5, -3), (5, 3), (-5, 3)],
    "triangle": [(-6, -5), (6, -5), (0, 6)],
    "L-shape": [(-6, -6), (6, -6), (6, -2), (-2, -2), (-2, 6), (-6, 6)],
    "electrode pad + taper": [(-20, -10), (-5, -10), (5, -0.2), (15, -0.2), (15, 0.2), (5, 0.2), (-5, 10), (-20, 10)],
    "circle r=5 µm (64-gon)": regular_polygon(0, 0, 5, 64).points.tolist(),
}


def _ring(r_out: float = 6.0, r_in: float = 3.0, n: int = 48) -> list:
    """Annulus as a single keyhole polygon (outer loop, bridge, inner loop reversed)."""
    o = regular_polygon(0, 0, r_out, n).points
    i = regular_polygon(0, 0, r_in, n).points
    inner = np.vstack([i[:1], i[1:][::-1], i[:1]])
    return np.vstack([o, o[:1], inner]).tolist()


PRESETS["ring (keyhole)"] = _ring()

DEFAULT_LINES = """# one polyline per block, blank line separates lines
-15, -8
15, -8

-15, 8
0, 12
15, 8
"""


class VectorLithoWidget(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session

        # ---- polygon tab
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(PRESETS))
        self.poly_text = QtWidgets.QPlainTextEdit()
        self.preset.currentTextChanged.connect(lambda k: self.poly_text.setPlainText(format_points(PRESETS[k])))
        self.poly_text.setPlainText(format_points(PRESETS[self.preset.currentText()]))
        self.poly_dose = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0, maximum=100, value=1.0, singleStep=0.1)
        pt = QtWidgets.QWidget()
        pf = QtWidgets.QFormLayout(pt)
        pf.addRow("preset", self.preset)
        pf.addRow("vertices (x, y µm)", self.poly_text)
        pf.addRow("dose factor", self.poly_dose)

        # ---- lines / dots tab
        self.lines_text = QtWidgets.QPlainTextEdit(DEFAULT_LINES)
        self.line_dose = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0, maximum=100, value=1.0, singleStep=0.1)
        self.dot_nx = QtWidgets.QSpinBox(minimum=0, maximum=1000, value=5)
        self.dot_ny = QtWidgets.QSpinBox(minimum=0, maximum=1000, value=3)
        self.dot_pitch = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=1000, value=2.0, suffix=" µm")
        self.dot_center = QtWidgets.QLineEdit("0, -2")
        self.dot_dose_ramp = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0, maximum=10, value=0.25,
                                                      toolTip="dose factor increment per column (dose test)")
        lt = QtWidgets.QWidget()
        lf = QtWidgets.QFormLayout(lt)
        lf.addRow("polylines", self.lines_text)
        lf.addRow("line dose factor", self.line_dose)
        lf.addRow("dot array nx", self.dot_nx)
        lf.addRow("dot array ny", self.dot_ny)
        lf.addRow("dot pitch", self.dot_pitch)
        lf.addRow("array centre (µm)", self.dot_center)
        lf.addRow("dose step / column", self.dot_dose_ramp)

        self.mode = QtWidgets.QTabWidget()
        self.mode.addTab(pt, "Polygon (raster fill)")
        self.mode.addTab(lt, "Lines && dots")

        self.panel = ExposurePanel(session, self.pattern, owner="litho-vector")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.mode)
        left.addWidget(self.panel.controls, 1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(430)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.panel.views, 1)

    # ------------------------------------------------------------- patterns
    def polygon(self) -> Polygon:
        return Polygon(parse_points(self.poly_text.toPlainText()), dose_factor=self.poly_dose.value())

    def lines_and_dots(self) -> Pattern:
        els = []
        for block in self.lines_text.toPlainText().split("\n\n"):
            pts = parse_points(block)
            if len(pts) >= 2:
                els.append(Polyline(pts, self.line_dose.value()))
        cx, cy = (float(v) for v in self.dot_center.text().split(","))
        nx, ny, p = self.dot_nx.value(), self.dot_ny.value(), self.dot_pitch.value()
        for j in range(ny):
            for i in range(nx):
                els.append(Dot(cx + (i - (nx - 1) / 2) * p, cy + (j - (ny - 1) / 2) * p,
                               1.0 + i * self.dot_dose_ramp.value()))
        return Pattern(els)

    def pattern(self) -> Pattern:
        if self.mode.currentIndex() == 0:
            return Pattern([self.polygon()])
        return self.lines_and_dots()


def main(argv=None) -> int:
    return run_standalone(VectorLithoWidget, "Vector-scan lithography", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
