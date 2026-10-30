"""Shared exposure panel for the lithography GUIs."""
from __future__ import annotations

import numpy as np

from ..gui.common import (
    DataclassForm,
    ImageView,
    LogBox,
    QtCore,
    QtWidgets,
    Session,
    WaveformView,
    Worker,
    pg,
)
from .compiler import CompiledJob, VectorScanSettings
from .geometry import Dot, Pattern, Polygon, Polyline
from .writer import ExposureParameters, LithoWriter

__all__ = ["ExposurePanel", "PARAM_OVERRIDES", "SETTINGS_OVERRIDES", "draw_pattern"]

PARAM_OVERRIDES = {
    "beam_current_a": {"label": "beam current (pA)", "scale": 1e12, "decimals": 2, "min": 0.01},
    "area_dose_uC_cm2": {"label": "area dose (µC/cm²)", "min": 0.001, "decimals": 2},
    "line_dose_pC_cm": {"label": "line dose (pC/cm)", "min": 0.001, "decimals": 1},
    "dot_dose_fC": {"label": "dot dose (fC)", "min": 0.001, "decimals": 3},
    "area_step_um": {"label": "area step (nm)", "scale": 1e3, "decimals": 2, "min": 0.01},
    "line_step_um": {"label": "line step (nm)", "scale": 1e3, "decimals": 2, "min": 0.01},
    "fill_direction": {"choices": ["x", "y"]},
    "order": {"choices": ["as_is", "nearest", "kind"]},
}
SETTINGS_OVERRIDES = {
    "rate": {"min": 1e3, "max": 2e6, "decimals": 0, "suffix": " S/s", "step": 1e4},
    "settle_base_s": {"label": "settle base (µs)", "scale": 1e6, "decimals": 2, "min": 0},
    "settle_per_um_s": {"label": "settle per µm (µs)", "scale": 1e6, "decimals": 3, "min": 0},
    "settle_max_s": {"label": "settle max (µs)", "scale": 1e6, "decimals": 1, "min": 0},
    "blank_lead_s": {"label": "blank lead (µs)", "scale": 1e6, "decimals": 2, "min": 0},
    "blanker_delay_s": {"label": "blanker delay (µs)", "scale": 1e6, "decimals": 3, "min": 0},
    "point_settle_s": {"label": "point settle (µs)", "scale": 1e6, "decimals": 2, "min": 0},
    "chunk_samples": {"min": 1000, "max": 10_000_000},
    "min_samples_per_point": {"min": 1, "max": 1000},
    "park_samples": {"min": 1, "max": 100000},
}


def draw_pattern(plot: pg.PlotItem | pg.PlotWidget, pattern: Pattern, cmap: str = "viridis") -> None:
    """Outline polygons / lines / dots coloured by dose factor."""
    df = pattern.dose_factors()
    lo, hi = (float(df.min()), float(df.max())) if len(df) else (1.0, 1.0)
    cm = pg.colormap.get(cmap)

    def color(f):
        t = 0.5 if hi == lo else (f - lo) / (hi - lo)
        return cm.map(t, mode="qcolor")

    for e in pattern:
        if isinstance(e, Polygon):
            p = np.vstack([e.points, e.points[:1]])
            plot.plot(p[:, 0], p[:, 1], pen=pg.mkPen(color(e.dose_factor), width=1.5))
        elif isinstance(e, Polyline):
            plot.plot(e.points[:, 0], e.points[:, 1], pen=pg.mkPen(color(e.dose_factor), width=2))
        elif isinstance(e, Dot):
            plot.plot([e.x], [e.y], pen=None, symbol="o", symbolSize=6, symbolBrush=color(e.dose_factor))


class ExposurePanel(QtWidgets.QWidget):
    """Parameter forms + compile/preview/expose + result views.

    ``pattern_provider()`` must return the :class:`Pattern` to expose.
    """

    def __init__(self, session: Session, pattern_provider, owner: str = "litho", parent=None):
        super().__init__(parent)
        self.session = session
        self.pattern_provider = pattern_provider
        self.handle = session.manager.get_handle(owner)
        self.job: CompiledJob | None = None
        self._worker = None

        self.params_form = DataclassForm(ExposureParameters(), PARAM_OVERRIDES)
        self.settings_form = DataclassForm(VectorScanSettings(), SETTINGS_OVERRIDES)
        self.sync_current = QtWidgets.QCheckBox("simulator beam current = exposure current")
        self.sync_current.setChecked(True)
        self.sync_current.setVisible(session.simulator is not None)
        self.reset_dose = QtWidgets.QCheckBox("clear simulator dose map before exposure")
        self.reset_dose.setChecked(True)
        self.reset_dose.setVisible(session.simulator is not None)

        self.b_compile = QtWidgets.QPushButton("Compile + preview")
        self.b_expose = QtWidgets.QPushButton("Expose")
        self.b_stop = QtWidgets.QPushButton("Abort")
        self.b_stop.setStyleSheet("background:#a33;color:white")
        self.b_compile.clicked.connect(self.compile_preview)
        self.b_expose.clicked.connect(self.expose)
        self.b_stop.clicked.connect(self.handle.abort)
        self.progress = QtWidgets.QProgressBar()
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        self.log = LogBox()
        self.log.setMaximumHeight(130)

        tb = QtWidgets.QToolBox()
        tb.addItem(self.params_form, "Exposure parameters")
        tb.addItem(self.settings_form, "Vector-scan settings")
        self.controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(self.controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(tb, 1)
        cl.addWidget(self.sync_current)
        cl.addWidget(self.reset_dose)
        hb = QtWidgets.QHBoxLayout()
        for b in (self.b_compile, self.b_expose, self.b_stop):
            hb.addWidget(b)
        cl.addLayout(hb)
        cl.addWidget(self.progress)
        cl.addWidget(self.stats)
        cl.addWidget(self.log)

        self.preview = pg.PlotWidget()
        self.preview.setAspectLocked(True)
        self.preview.showGrid(x=True, y=True, alpha=0.2)
        self.preview.setLabel("bottom", "x", "µm")
        self.preview.setLabel("left", "y", "µm")
        self.wave = WaveformView()
        self.dose_view = ImageView(cmap="inferno")
        self.views = QtWidgets.QTabWidget()
        self.views.addTab(self.preview, "Pattern / path")
        self.views.addTab(self.wave, "Waveform")
        self.views.addTab(self.dose_view, "Dose map (sim)")

        self._sig = _Signals()
        self._sig.progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        self.progress.setRange(0, 1000)

    # --------------------------------------------------------------- actions
    def params(self) -> ExposureParameters:
        return self.params_form.value()

    def settings(self) -> VectorScanSettings:
        return self.settings_form.value()

    def writer(self) -> LithoWriter:
        return LithoWriter(self.handle, self.session.calibration, self.settings())

    def compile(self) -> CompiledJob:
        pat = self.pattern_provider()
        self.job = self.writer().compile(pat, self.params())
        st = self.job.stats
        self.stats.setText(
            f"{st.n_points:,} points in {st.n_runs:,} runs, {st.n_jumps:,} jumps · "
            f"exposure {st.exposure_time_s:.3f} s, total {st.total_time_s:.3f} s (duty {st.duty_cycle * 100:.0f}%) · "
            f"dose quantisation rms {st.dose_error_rms * 100:.2f}% max {st.dose_error_max * 100:.1f}%"
            + (f" · {st.clamped_points} points clamped to min dwell" if st.clamped_points else "")
        )
        return self.job

    def compile_preview(self):
        try:
            job = self.compile()
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {type(e).__name__}: {e}")
            return
        pat = self.pattern_provider()
        self.preview.clear()
        draw_pattern(self.preview, pat)
        # exposure points (subsampled) + blanked jump path
        pts = [r.points_um for r in job.runs]
        if pts:
            allp = np.vstack(pts)
            step = max(1, len(allp) // 50_000)
            self.preview.plot(allp[::step, 0], allp[::step, 1], pen=None, symbol="o", symbolSize=2,
                              symbolPen=None, symbolBrush=pg.mkBrush(80, 200, 255, 160))
            if len(job.runs) <= 2000:
                starts = np.array([r.points_um[0] for r in job.runs])
                ends = np.array([r.points_um[-1] for r in job.runs])
                jumps = np.column_stack([ends[:-1], starts[1:]]).reshape(-1, 2)
                conn = np.tile([1, 0], len(jumps) // 2).astype(bool) if len(jumps) else np.zeros(0, bool)
                if len(jumps):
                    self.preview.plot(jumps[:, 0], jumps[:, 1], connect=conn,
                                      pen=pg.mkPen((255, 80, 80, 120), style=QtCore.Qt.DashLine))
        try:
            if job.stats.total_samples <= 400_000:
                self.wave.show_chunk(job.preview_arrays(), job.settings.rate)
        except Exception as e:  # noqa: BLE001
            self.log.log(f"waveform preview skipped: {e}")
        self.views.setCurrentWidget(self.preview)

    def expose(self):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            job = self.compile()
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {type(e).__name__}: {e}")
            return
        sim = self.session.simulator
        if sim is not None:
            if self.sync_current.isChecked():
                sim.params.beam_current_a = self.params().beam_current_a
            if self.reset_dose.isChecked():
                sim.dose.reset()
        w = self.writer()
        self.progress.setValue(0)
        self._worker = Worker(w.expose_job, job, self._sig.progress.emit, 1.0)
        self._worker.result.connect(self._done)
        self._worker.error.connect(lambda e: self.log.log("ERROR " + e))
        self._worker.start()

    def _done(self, rep):
        self.log.log(
            f"exposure {'ABORTED' if rep.aborted else 'done'}: {rep.samples_written:,} samples in {rep.elapsed_s:.2f} s"
        )
        sim = self.session.simulator
        if sim is not None:
            d = sim.dose.dose_uC_cm2()
            self.dose_view.set_image(d, sim.dose.extent)
            self.dose_view.clear_overlays()
            pat = self.pattern_provider()
            for e in pat:
                if isinstance(e, Polygon):
                    self.dose_view.add_path(e.points, color="c", closed=True)
            nz = d[d > 0]
            if nz.size:
                self.log.log(f"simulated dose: median of exposed pixels {np.median(nz):.1f} µC/cm²")
            self.views.setCurrentWidget(self.dose_view)


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(float)
