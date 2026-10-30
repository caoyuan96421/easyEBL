"""Dose / dwell-time calculator GUI.

Run:  ``python -m easyebl.litho.gui_dose``
"""
from __future__ import annotations

from ..gui.common import DataclassForm, QtWidgets, Session, ensure_app
from .dose import DoseCalculator

__all__ = ["DoseCalculatorWidget", "main"]

OVERRIDES = {
    "field_um": {"label": "field size", "suffix": " µm", "min": 0.001, "decimals": 3},
    "points": {"label": "points across field", "min": 1, "max": 2**24},
    "current_pA": {"label": "beam current", "suffix": " pA", "min": 0.001, "decimals": 3},
    "rate_hz": {"label": "DAQ clock", "suffix": " S/s", "min": 1, "max": 1e7, "decimals": 0},
    "area_dose_uC_cm2": {"label": "area dose", "suffix": " µC/cm²", "min": 0.001, "decimals": 2},
    "line_dose_pC_cm": {"label": "line dose", "suffix": " pC/cm", "min": 0.001, "decimals": 1},
    "dot_dose_fC": {"label": "dot dose", "suffix": " fC", "min": 0.0001, "decimals": 4},
    "min_dwell_s": {"label": "min dwell", "scale": 1e6, "suffix": " µs", "decimals": 3, "min": 0},
    "max_dose_error": {"label": "max quantisation error", "scale": 100, "suffix": " %", "decimals": 2},
    "dac_bits": {"min": 8, "max": 24},
}


class DoseCalculatorWidget(QtWidgets.QWidget):
    def __init__(self, session: Session | None = None, parent=None):
        super().__init__(parent)
        cal = session.calibration if session is not None else None
        init = DoseCalculator(field_um=cal.field_size_um if cal else 100.0)
        self.form = DataclassForm(init, OVERRIDES)
        self.out = QtWidgets.QTableWidget(0, 2)
        self.out.setHorizontalHeaderLabels(["quantity", "value"])
        self.out.horizontalHeader().setStretchLastSection(True)
        self.warn = QtWidgets.QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#c60")
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(self.form)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.out, 1)
        right.addWidget(self.warn)
        lay.addLayout(right, 1)
        self.form.changed.connect(self.update)
        self.update()

    def report(self):
        return self.form.value().report()

    def update(self):  # noqa: D401 - Qt slot name
        try:
            r = self.report()
        except Exception as e:  # noqa: BLE001
            self.warn.setText(str(e))
            return
        rows = [
            ("step (pixel pitch)", f"{r.step_um * 1e3:.3f} nm"),
            ("DAC resolution", f"{r.dac_step_um * 1e3:.3f} nm"),
            ("area dwell", f"{r.area_dwell_s * 1e6:.4g} µs → {r.area_dwell_samples} samples ({r.area_dwell_error * 100:+.2f}%)"),
            ("line dwell", f"{r.line_dwell_s * 1e6:.4g} µs → {r.line_dwell_samples} samples"),
            ("dot dwell", f"{r.dot_dwell_s * 1e6:.4g} µs → {r.dot_dwell_samples} samples"),
            ("exposure time / 100 µm²", f"{r.time_per_100um2_s:.4g} s"),
            ("beam speed in area fill", f"{r.step_um / r.area_dwell_s * 1e-3:.4g} mm/s"),
        ]
        self.out.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.out.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.out.setItem(i, 1, QtWidgets.QTableWidgetItem(v))
        self.out.resizeColumnToContents(0)
        self.warn.setText("<br>".join("⚠ " + w for w in r.warnings) or "✓ settings OK")


def main(argv=None) -> int:
    app = ensure_app()
    w = DoseCalculatorWidget()
    w.setWindowTitle("easyEBL – dose calculator")
    w.resize(800, 420)
    w.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
