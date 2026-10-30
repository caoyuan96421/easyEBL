"""Raster imaging debug GUI: scan settings, waveform preview, live image.

Run:  ``python -m easyebl.imaging.gui``
"""
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
    run_standalone,
)
from .raster import RasterConfig, RasterImager, RasterPlan

__all__ = ["ImagingWidget", "RASTER_FORM_OVERRIDES", "main"]

RASTER_FORM_OVERRIDES = {
    "nx": {"min": 1, "max": 8192},
    "ny": {"min": 1, "max": 8192},
    "width_um": {"min": 0.001, "max": 1e4, "suffix": " µm"},
    "rotation_rad": {"label": "rotation (mrad)", "scale": 1e3, "decimals": 2},
    "rate": {"min": 1e3, "max": 2e6, "decimals": 0, "suffix": " S/s", "step": 1e4},
    "dwell_s": {"label": "dwell (µs)", "scale": 1e6, "decimals": 3, "min": 0.001},
    "preline_s": {"label": "pre-line (µs)", "scale": 1e6, "decimals": 2, "min": 0},
    "flyback_s": {"label": "flyback (µs)", "scale": 1e6, "decimals": 2, "min": 0},
    "frame_settle_s": {"label": "frame settle (µs)", "scale": 1e6, "decimals": 1, "min": 0},
    "lag_comp_s": {"label": "lag comp (µs)", "scale": 1e6, "decimals": 3, "min": 0},
    "ai_latency_samples": {"label": "AI latency (samples)", "decimals": 2, "min": 0, "max": 100},
    "preline_mode": {"choices": ["ramp", "hold"]},
    "flyback_shape": {"choices": ["linear", "cosine", "step"]},
    "x_direction": {"choices": [1, -1]},
    "y_direction": {"choices": ["down", "up"]},
    "frames": {"min": 1, "max": 1000},
    "settle_samples": {"min": 0, "max": 1000},
}


class ImagingWidget(QtWidgets.QWidget):
    _row = QtCore.Signal(int, object)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.handle = session.manager.get_handle("imaging")
        self.imager = RasterImager(self.handle, session.calibration)
        self.last = None
        self._worker = None
        self._continuous = False

        cfg = RasterConfig(nx=256, ny=256, width_um=80.0, ai_channels=tuple(session.manager.config.ai_channels)[:1])
        self.form = DataclassForm(cfg, RASTER_FORM_OVERRIDES, hidden=("height_um", "line_sync_marker", "frame_sync_marker"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)

        self.b_prev = QtWidgets.QPushButton("Preview waveform")
        self.b_scan = QtWidgets.QPushButton("Single scan")
        self.b_cont = QtWidgets.QPushButton("Continuous")
        self.b_cont.setCheckable(True)
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_prev.clicked.connect(self.preview)
        self.b_scan.clicked.connect(self.scan)
        self.b_cont.toggled.connect(self._toggle_cont)
        self.b_stop.clicked.connect(self.stop)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.log = LogBox()
        self.log.setMaximumHeight(140)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(scroll, 1)
        bl = QtWidgets.QGridLayout()
        bl.addWidget(self.b_prev, 0, 0)
        bl.addWidget(self.b_scan, 0, 1)
        bl.addWidget(self.b_cont, 1, 0)
        bl.addWidget(self.b_stop, 1, 1)
        left.addLayout(bl)
        left.addWidget(self.summary)
        left.addWidget(self.log)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(380)

        self.view = ImageView()
        self.wave = WaveformView()
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.view, "Image")
        tabs.addTab(self.wave, "Waveform")
        self.tabs = tabs
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(tabs, 1)

        self.form.changed.connect(self._update_summary)
        self._row.connect(self._on_row)
        self._update_summary()

    # ----------------------------------------------------------------- helpers
    def config(self) -> RasterConfig:
        return self.form.value()

    def _update_summary(self):
        try:
            plan = RasterPlan(self.config(), self.session.calibration)
            s = plan.summary()
            self.summary.setText(
                f"{s['samples_per_pixel']} samples/pixel, dwell {s['dwell_s'] * 1e6:.2f} µs, "
                f"pixel {s['pixel_size_um'][0] * 1e3:.1f} nm; frame {s['frame_time_s']:.3f} s, "
                f"duty {s['duty_cycle'] * 100:.0f}%"
            )
        except Exception as e:  # noqa: BLE001
            self.summary.setText(f"<span style='color:red'>{e}</span>")

    # ----------------------------------------------------------------- actions
    def preview(self):
        try:
            cfg = self.config()
            small = cfg.with_(ny=min(cfg.ny, 4), frames=1)
            plan = RasterPlan(small, self.session.calibration)
            self.wave.show_chunk(plan.full_waveform(), small.rate)
            self.tabs.setCurrentWidget(self.wave)
            self.log.log(f"preview: first {small.ny} line(s)")
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {e}")

    def scan(self):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            cfg = self.config()
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {e}")
            return
        self.tabs.setCurrentWidget(self.view)
        self._extent = None
        self._cfg = cfg

        def job():
            # throttle in the acquisition thread: copy the partial image only every 16 rows
            def on_row(r, get_images):
                if r % 16 == 0 or r == cfg.ny - 1:
                    self._row.emit(r, get_images())

            return self.imager.acquire(cfg, on_row=on_row, timeout=1.0)

        self._worker = Worker(job)
        self._worker.result.connect(self._done)
        self._worker.error.connect(lambda e: (self.log.log("ERROR " + e), self.b_cont.setChecked(False)))
        self._worker.start()

    def _on_row(self, row: int, imgs: dict):
        cx, cy = self._cfg.center_um
        w, h = self._cfg.width_um, self._cfg.height
        self.view.set_image(next(iter(imgs.values())), (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2))

    def _done(self, img):
        self.last = img
        self.view.set_image(img.image(), img.extent)
        self.log.log(
            f"scan {'ABORTED ' if img.aborted else ''}done: {img.rows_completed} rows, frame {img.frame_time_s:.3f} s"
        )
        if self._continuous and not img.aborted:
            QtCore.QTimer.singleShot(10, self.scan)

    def _toggle_cont(self, on: bool):
        self._continuous = on
        if on:
            self.scan()

    def stop(self):
        self._continuous = False
        self.b_cont.setChecked(False)
        self.handle.abort()


def main(argv=None) -> int:
    return run_standalone(ImagingWidget, "Raster imaging", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
