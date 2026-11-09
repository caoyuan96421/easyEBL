"""Raster imaging debug GUI: scan settings, waveform preview, live image.

Run:  ``python -m easyebl.imaging.gui``
"""
from __future__ import annotations

import numpy as np

import threading
import time

from ..gui.common import (
    DataclassForm,
    ImageView,
    LogBox,
    QtCore,
    QtGui,
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
    """Raster imaging: single scans and a live, continuously streamed view.

    *Live* runs :meth:`RasterImager.live` – one hardware-timed stream that
    loops the frame waveform from the DAQ buffer – so frames follow each other
    without the per-run task setup.  Changing a setting while live restarts
    the stream with the new settings.
    """

    _row = QtCore.Signal(int, object)
    _frame = QtCore.Signal(int, object, float)

    GUI_FPS = 30.0  # max display refresh

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.handle = session.manager.get_handle("imaging")
        self.imager = RasterImager(self.handle, session.calibration)
        self.last = None
        self._worker = None
        self._live = False
        self._restart = False
        self._stop_evt = threading.Event()
        self._avg = None
        self._t_emit = 0.0
        self._fps_t = []

        cfg = RasterConfig(nx=256, ny=256, width_um=0.8 * session.calibration.field_size_um,
                           ai_channels=tuple(session.manager.config.ai_channels)[:1])
        self.form = DataclassForm(cfg, RASTER_FORM_OVERRIDES,
                                  hidden=("height_um", "line_sync_marker", "frame_sync_marker", "frames"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)

        # --- scan control (relay enables, e.g. PFI6/PFI7 on the USB-6363)
        self.static_names = list(session.manager.config.static_lines)
        self.cb_control = QtWidgets.QCheckBox(
            "External scan control (" + ", ".join(
                session.manager.config.static_lines[n] for n in self.static_names) + ")"
        )
        self.cb_control.setToolTip("Claims the relay-enable lines and drives them HIGH so the SEM "
                                   "takes X/Y from the DAQ. Unchecked = SEM internal scan.")
        self.cb_control.setVisible(bool(self.static_names))
        self.cb_control.toggled.connect(self._set_control)

        self.sp_avg = QtWidgets.QSpinBox()
        self.sp_avg.setRange(1, 256)
        self.sp_avg.setValue(1)
        self.sp_avg.setToolTip("Running average over ~N frames (exponential)")
        self.sp_avg.valueChanged.connect(lambda _: self._reset_avg())
        self.cb_autolev = QtWidgets.QCheckBox("Auto contrast")
        self.cb_autolev.setChecked(True)

        self.b_prev = QtWidgets.QPushButton("Preview waveform")
        self.b_scan = QtWidgets.QPushButton("Single scan")
        self.b_live = QtWidgets.QPushButton("Live")
        self.b_live.setCheckable(True)
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_save = QtWidgets.QPushButton("Save image…")
        self.b_prev.clicked.connect(self.preview)
        self.b_scan.clicked.connect(self.scan)
        self.b_live.toggled.connect(self._toggle_live)
        self.b_stop.clicked.connect(self.stop)
        self.b_save.clicked.connect(self.save)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.status = QtWidgets.QLabel("idle")
        self.log = LogBox()
        self.log.setMaximumHeight(140)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.cb_control)
        left.addWidget(scroll, 1)
        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(QtWidgets.QLabel("Average frames"))
        avg_row.addWidget(self.sp_avg)
        avg_row.addWidget(self.cb_autolev)
        left.addLayout(avg_row)
        bl = QtWidgets.QGridLayout()
        bl.addWidget(self.b_live, 0, 0)
        bl.addWidget(self.b_stop, 0, 1)
        bl.addWidget(self.b_scan, 1, 0)
        bl.addWidget(self.b_prev, 1, 1)
        bl.addWidget(self.b_save, 2, 0, 1, 2)
        left.addLayout(bl)
        left.addWidget(self.summary)
        left.addWidget(self.status)
        left.addWidget(self.log)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(400)

        self.view = ImageView()
        self.wave = WaveformView()
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.view, "Image")
        tabs.addTab(self.wave, "Waveform")
        self.tabs = tabs
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(tabs, 1)

        self.form.changed.connect(self._on_form_changed)
        self._row.connect(self._on_row)
        self._frame.connect(self._on_frame)
        self._update_summary()

    # ----------------------------------------------------------------- helpers
    def config(self) -> RasterConfig:
        return self.form.value()

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _update_summary(self):
        try:
            plan = RasterPlan(self.config(), self.session.calibration)
            s = plan.summary()
            self.summary.setText(
                f"{s['samples_per_pixel']} samples/pixel, dwell {s['dwell_s'] * 1e6:.2f} µs, "
                f"pixel {s['pixel_size_um'][0] * 1e3:.1f} nm; frame {s['frame_time_s']:.3f} s "
                f"({1 / s['frame_time_s']:.1f} fps max), duty {s['duty_cycle'] * 100:.0f}%"
            )
            return True
        except Exception as e:  # noqa: BLE001
            self.summary.setText(f"<span style='color:red'>{e}</span>")
            return False

    def _on_form_changed(self):
        ok = self._update_summary()
        if self._live and self._busy() and ok:
            self._restart = True  # restart the stream with the new settings
            self._stop_evt.set()
            self.handle.abort()

    def _reset_avg(self):
        self._avg = None

    def _set_control(self, on: bool):
        try:
            if on:
                self.handle.claim(*self.static_names)
                for n in self.static_names:
                    self.handle.set_line(n, True)
                self.log.log("external scan control ON: " + str(self.session.manager.line_state))
            else:
                if self._busy():
                    self.stop()
                    self._worker.wait(5000)
                for n in self.static_names:
                    if n in self.handle.claimed:
                        self.handle.set_line(n, False)
                self.handle.release(*[n for n in self.static_names if n in self.handle.claimed])
                self.log.log("external scan control OFF (SEM internal scan)")
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR scan control: {e}")
            self.cb_control.blockSignals(True)
            self.cb_control.setChecked(False)
            self.cb_control.blockSignals(False)

    def _check_control(self):
        if self.static_names and not self.cb_control.isChecked() and self.session.simulator is None:
            self.log.log("note: external scan control is OFF – the SEM is still on internal scan")

    def _extent(self, cfg):
        cx, cy = cfg.center_um
        w, h = cfg.width_um, cfg.height
        return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

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
        """One frame (averaged with the running average if N > 1)."""
        if self._busy():
            return
        try:
            cfg = self.config().with_(frames=1)
            RasterPlan(cfg, self.session.calibration)
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {e}")
            return
        self._check_control()
        self._start(cfg, max_frames=1)

    def _toggle_live(self, on: bool):
        if on:
            if self._busy():
                return
            try:
                cfg = self.config()
                RasterPlan(cfg, self.session.calibration)
            except Exception as e:  # noqa: BLE001
                self.log.log(f"ERROR {e}")
                self.b_live.setChecked(False)
                return
            self._check_control()
            self._live = True
            self._start(cfg, max_frames=None)
        else:
            self._live = False
            self._stop_evt.set()
            self.handle.abort()

    def _start(self, cfg, max_frames):
        self.tabs.setCurrentWidget(self.view)
        self._cfg = cfg
        self._stop_evt = threading.Event()
        self._fps_t = []
        stop_evt = self._stop_evt
        slow = RasterPlan(cfg, self.session.calibration).frame_time_s > 0.5

        def on_frame(k, img):
            # runs in the streaming thread: keep it cheap
            a = img.image()
            n = self.sp_avg.value()
            if self._avg is None or self._avg.shape != a.shape or n <= 1:
                self._avg = a.copy()
            else:
                self._avg += (a - self._avg) / n
            now = time.perf_counter()
            if now - self._t_emit >= 1.0 / self.GUI_FPS or max_frames == 1:
                self._t_emit = now
                self._frame.emit(k, self._avg.copy(), now)
            self.last = img

        def on_row(r, get_images):
            if r % 16 == 0 or r == cfg.ny - 1:
                self._row.emit(r, get_images())

        def job():
            return self.imager.live(cfg, on_frame, on_row=on_row if slow else None,
                                    stop=stop_evt, max_frames=max_frames, timeout=1.0)

        self._worker = Worker(job)
        self._worker.result.connect(self._done)
        self._worker.error.connect(self._error)
        self._worker.start()
        self.status.setText("live…" if max_frames is None else "scanning…")
        self.log.log(f"{'live' if max_frames is None else 'single scan'}: {cfg.nx}×{cfg.ny} @ {cfg.rate:g} S/s")

    def _on_row(self, row: int, imgs: dict):
        self.view.set_image(next(iter(imgs.values())), self._extent(self._cfg))

    def _on_frame(self, k: int, img, t: float):
        lv = None if self.cb_autolev.isChecked() else getattr(self, "_levels", None)
        self.view.set_image(img, self._extent(self._cfg), levels=lv)
        if self.cb_autolev.isChecked():
            self._levels = (float(np.nanmin(img)), float(np.nanmax(img)))
        self._fps_t.append(t)
        self._fps_t = self._fps_t[-20:]
        fps = (len(self._fps_t) - 1) / (self._fps_t[-1] - self._fps_t[0]) if len(self._fps_t) > 2 else 0.0
        self.status.setText(f"frame {k + 1}" + (f", display {fps:.1f} fps" if fps else ""))

    def _done(self, n_frames):
        self.log.log(f"stream ended after {n_frames} frame(s)")
        if self._restart:
            self._restart = False
            if self._live:
                self._reset_avg()
                QtCore.QTimer.singleShot(0, self._restart_live)
                return
        self._live = False
        self.b_live.blockSignals(True)
        self.b_live.setChecked(False)
        self.b_live.blockSignals(False)
        self.status.setText("idle")

    def _restart_live(self):
        if self._busy():  # result arrives just before the thread exits
            QtCore.QTimer.singleShot(10, self._restart_live)
            return
        if self._live:
            self._live = False
            self._toggle_live(True)

    def _error(self, e: str):
        self.log.log("ERROR " + e)
        self._live = False
        self._restart = False
        self.b_live.blockSignals(True)
        self.b_live.setChecked(False)
        self.b_live.blockSignals(False)
        self.status.setText("error")

    def stop(self):
        self._live = False
        self._restart = False
        self._stop_evt.set()
        self.handle.abort()
        self.b_live.blockSignals(True)
        self.b_live.setChecked(False)
        self.b_live.blockSignals(False)

    def save(self):
        if self._avg is None:
            self.log.log("nothing to save yet")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "image.png", "PNG (*.png)")
        if not path:
            return
        data = np.asarray(self._avg, float)
        np.save(path.rsplit(".", 1)[0] + ".npy", data)
        lo, hi = np.nanpercentile(data, [0.5, 99.5])
        u8 = np.clip((data - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        u8 = np.ascontiguousarray(u8)
        qimg = QtGui.QImage(u8.data, u8.shape[1], u8.shape[0], u8.shape[1], QtGui.QImage.Format_Grayscale8)
        qimg.save(path)
        self.log.log(f"saved {path} (+ .npy with raw values)")

    def closeEvent(self, ev):  # noqa: N802 - Qt
        self.stop()
        if self._worker is not None:
            self._worker.wait(5000)
        if self.cb_control.isChecked():
            self.cb_control.setChecked(False)
        super().closeEvent(ev)


def main(argv=None) -> int:
    return run_standalone(ImagingWidget, "Raster imaging", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
