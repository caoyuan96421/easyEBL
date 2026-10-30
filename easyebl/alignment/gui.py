"""Marker alignment debug GUI.

Edit nominal mark positions, scan + match each mark, fit a transform and
inspect per-mark images, scores, residuals and the measured lag shift.  On
the simulator the specimen can be deliberately misplaced to test recovery.

Run:  ``python -m easyebl.alignment.gui``
"""
from __future__ import annotations

import numpy as np

from ..core.transforms import AffineTransform
from ..gui.common import (
    DataclassForm,
    ImageView,
    LogBox,
    QtCore,
    QtWidgets,
    Session,
    Worker,
    format_points,
    parse_points,
    run_standalone,
)
from ..imaging.gui import RASTER_FORM_OVERRIDES
from ..imaging.raster import RasterConfig, RasterImager
from .aligner import MarkerAligner, MarkSpec
from .fitting import MODELS
from .matching import cross_template, square_template

__all__ = ["AlignmentWidget", "main"]

DEFAULT_MARKS = np.array([(-35.0, -35.0), (35.0, -35.0), (35.0, 35.0), (-35.0, 35.0)])


class AlignmentWidget(QtWidgets.QWidget):
    _mark = QtCore.Signal(object)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.handle = session.manager.get_handle("alignment")
        self.imager = RasterImager(self.handle, session.calibration)
        self.result = None
        self._worker = None

        # --- marks + template
        self.marks = QtWidgets.QPlainTextEdit(format_points(DEFAULT_MARKS))
        self.marks.setMaximumHeight(100)
        self.shape = QtWidgets.QComboBox()
        self.shape.addItems(["cross", "square"])
        self.size = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.1, maximum=500, value=8.0, suffix=" µm")
        self.width = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.05, maximum=100, value=1.0, suffix=" µm")
        self.window = QtWidgets.QDoubleSpinBox(decimals=1, minimum=1, maximum=1000, value=20.0, suffix=" µm")
        self.model = QtWidgets.QComboBox()
        self.model.addItems(list(MODELS))
        self.model.setCurrentText("similarity")
        self.bidir = QtWidgets.QCheckBox("bidirectional averaging (cancel lag)")
        self.bidir.setChecked(True)
        self.min_score = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0, maximum=1, value=0.5, singleStep=0.05)
        g = QtWidgets.QGroupBox("Marks / template")
        f = QtWidgets.QFormLayout(g)
        f.addRow("nominal marks (x, y µm)", self.marks)
        f.addRow("shape", self.shape)
        f.addRow("size", self.size)
        f.addRow("arm width", self.width)
        f.addRow("scan window", self.window)
        f.addRow("fit model", self.model)
        f.addRow("min score", self.min_score)
        f.addRow(self.bidir)

        scan_cfg = RasterConfig(nx=128, ny=128, dwell_s=4e-6, preline_s=30e-6, blank_preline=True,
                                ai_channels=tuple(session.manager.config.ai_channels)[:1])
        self.scan_form = DataclassForm(
            scan_cfg,
            RASTER_FORM_OVERRIDES,
            hidden=("center_um", "width_um", "height_um", "rotation_rad", "bidirectional", "x_direction",
                    "line_sync_marker", "frame_sync_marker", "y_direction"),
        )
        sg = QtWidgets.QGroupBox("Mark scan settings")
        QtWidgets.QVBoxLayout(sg).addWidget(self.scan_form)

        # --- simulator misplacement
        self.sim_box = QtWidgets.QGroupBox("Simulator: specimen misplacement")
        sf = QtWidgets.QFormLayout(self.sim_box)
        self.sim_tx = QtWidgets.QDoubleSpinBox(decimals=3, minimum=-20, maximum=20, value=1.2, suffix=" µm")
        self.sim_ty = QtWidgets.QDoubleSpinBox(decimals=3, minimum=-20, maximum=20, value=-0.8, suffix=" µm")
        self.sim_rot = QtWidgets.QDoubleSpinBox(decimals=2, minimum=-100, maximum=100, value=3.0, suffix=" mrad")
        self.sim_scale = QtWidgets.QDoubleSpinBox(decimals=5, minimum=0.9, maximum=1.1, value=1.001, singleStep=0.0005)
        b_apply = QtWidgets.QPushButton("Regenerate specimen")
        b_apply.clicked.connect(self._apply_sim)
        for lab, w in [("shift x", self.sim_tx), ("shift y", self.sim_ty), ("rotation", self.sim_rot), ("scale", self.sim_scale)]:
            sf.addRow(lab, w)
        sf.addRow(b_apply)
        self.sim_box.setVisible(session.simulator is not None)

        self.b_run = QtWidgets.QPushButton("Run alignment")
        self.b_run.clicked.connect(self.run_alignment)
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_stop.clicked.connect(self.handle.abort)

        left = QtWidgets.QVBoxLayout()
        for w in (g, sg, self.sim_box):
            left.addWidget(w)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.b_run)
        hb.addWidget(self.b_stop)
        left.addLayout(hb)
        left.addStretch(1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)

        # --- results
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["mark", "nominal x", "nominal y", "measured x", "measured y", "score", "residual (nm)", "lag shift (nm)"]
        )
        self.table.itemSelectionChanged.connect(self._show_selected)
        self.view = ImageView()
        self.score_view = ImageView(cmap="viridis")
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.log = LogBox()
        self.log.setMaximumHeight(120)
        imgs = QtWidgets.QSplitter()
        imgs.addWidget(self.view)
        imgs.addWidget(self.score_view)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(imgs)
        right.addWidget(self.table)
        rw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rw)
        rl.addWidget(right, 1)
        rl.addWidget(self.summary)
        rl.addWidget(self.log)

        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(rw, 1)
        self._mark.connect(self._on_mark)

    # ----------------------------------------------------------------- helpers
    def _template_factory(self):
        size, width = self.size.value(), self.width.value()
        if self.shape.currentText() == "cross":
            return lambda px: cross_template(size, width, px)
        return lambda px: square_template(size, px)

    def make_aligner(self) -> MarkerAligner:
        return MarkerAligner(
            self.imager,
            self.scan_form.value(),
            self._template_factory(),
            window_um=self.window.value(),
            min_score=self.min_score.value(),
            bidirectional_average=self.bidir.isChecked(),
        )

    def sim_truth(self) -> AffineTransform:
        return AffineTransform.from_params(
            tx=self.sim_tx.value(),
            ty=self.sim_ty.value(),
            rotation_rad=self.sim_rot.value() * 1e-3,
            sx=self.sim_scale.value(),
            sy=self.sim_scale.value(),
        )

    def _apply_sim(self):
        sim = self.session.simulator
        if sim is None:
            return
        from ..hal.simulated import make_demo_sample

        nominal = parse_points(self.marks.toPlainText())
        sim.sample = make_demo_sample(marks=self.sim_truth().apply(nominal), mark_size=self.size.value(),
                                      mark_width=self.width.value())
        self.log.log(f"simulated specimen placed with {self.sim_truth()}")

    # ----------------------------------------------------------------- actions
    def run_alignment(self):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            marks = [MarkSpec(tuple(p), name=f"M{i + 1}") for i, p in enumerate(parse_points(self.marks.toPlainText()))]
            aligner = self.make_aligner()
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {e}")
            return
        self.table.setRowCount(0)
        model = self.model.currentText()
        self._worker = Worker(aligner.align, marks, model, on_mark=self._mark.emit)
        self._worker.result.connect(self._done)
        self._worker.error.connect(lambda e: self.log.log("ERROR " + e))
        self._worker.start()

    def _on_mark(self, m):
        r = self.table.rowCount()
        self.table.insertRow(r)
        lag = "" if m.lag_shift_um is None else f"{m.lag_shift_um[0] * 1e3:+.1f}"
        vals = [m.name, *(f"{v:.4f}" for v in m.nominal_um), *(f"{v:.4f}" for v in m.measured_um),
                f"{m.score:.3f}" + ("" if m.ok else " ✗"), "", lag]
        for c, v in enumerate(vals):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self._show_mark(m)

    def _done(self, res):
        self.result = res
        for i, m in enumerate(res.measurements):
            good = [x for x in res.measurements if x.ok]
            if m.ok:
                k = good.index(m)
                e = res.residuals_um[k]
                self.table.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{np.hypot(*e) * 1e3:.1f}"))
        txt = res.summary()
        sim = self.session.simulator
        if sim is not None and self.sim_box.isVisible():
            truth = self.sim_truth()
            nominal = np.array([m.nominal_um for m in res.measurements])
            err = res.transform.apply(nominal) - truth.apply(nominal)
            txt += f"<br>vs simulator truth: max placement error {np.max(np.hypot(*err.T)) * 1e3:.1f} nm"
        self.summary.setText(txt)
        self.log.log(res.summary())

    def _show_mark(self, m):
        if not m.images:
            return
        img = m.images[0]
        self.view.set_image(img.image(), img.extent)
        self.view.clear_overlays()
        self.view.add_points([img.pixel_to_field(mt.col, mt.row) for mt in m.matches], color="r", size=14)
        self.view.add_points([m.nominal_um], color="c", size=10, symbol="o")
        sm = m.matches[0].score_map
        if sm is not None:
            px = img.pixel_size_um[0]
            h = (img.image().shape[0] - sm.shape[0]) / 2 * px
            x0, x1, y0, y1 = img.extent
            self.score_view.set_image(sm, (x0 + h, x1 - h, y0 + h, y1 - h))

    def _show_selected(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if self.result and rows:
            r = min(rows)
            if r < len(self.result.measurements):
                self._show_mark(self.result.measurements[r])


def main(argv=None) -> int:
    def make(session):
        w = AlignmentWidget(session)
        w._apply_sim()
        return w

    return run_standalone(make, "Marker alignment", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
