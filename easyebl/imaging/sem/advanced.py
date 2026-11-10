"""'Advanced…' dialog: hardware, scan waveform, calibration and saving settings."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from ...core.calibration import FieldCalibration
from ..engine import AI_RANGES_V, ImagingEngine, valid_rates

__all__ = ["AdvancedDialog", "SaveSettings"]


class SaveSettings:
    """Where/how captures are saved (owned by the main window)."""

    def __init__(self, folder: str = "captures", auto_save: bool = True, formats=("tif", "png", "json"),
                 data_bar: bool = True):
        self.folder = folder
        self.auto_save = auto_save
        self.formats = tuple(formats)
        self.data_bar = data_bar


def _dspin(lo, hi, dec, suffix="", step=1.0):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSuffix(suffix)
    s.setSingleStep(step)
    s.setKeyboardTracking(False)
    return s


class AdvancedDialog(QtWidgets.QDialog):
    applied = QtCore.Signal()

    def __init__(self, engine: ImagingEngine, save: SaveSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.engine = engine
        self.save = save
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._hw_tab(), "Hardware")
        tabs.addTab(self._scan_tab(), "Scan waveform")
        tabs.addTab(self._cal_tab(), "Calibration")
        tabs.addTab(self._save_tab(), "Saving")
        tabs.addTab(self._phantom_tab(), "Test signal")
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close | QtWidgets.QDialogButtonBox.Reset
        )
        bb.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        bb.button(QtWidgets.QDialogButtonBox.Reset).setText("Reload")
        bb.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.load)
        bb.rejected.connect(self.close)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(self.msg)
        lay.addWidget(bb)
        self.resize(520, 520)
        self.load()

    # ------------------------------------------------------------------ tabs
    def _hw_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        dc = self.engine.daq_config
        self.rate = QtWidgets.QComboBox()
        self._fill_rates()
        self.rate.setToolTip("Sample clock (100 MHz / N). Every sample is digitised and averaged into its "
                             "pixel, so the fastest clock gives the best SNR at any scan speed.")
        f.addRow("Sample rate", self.rate)
        self.latency = _dspin(5, 1000, 0, " ms", 10)
        self.latency.setToolTip("Output buffer depth. Settings reach the beam after about this time. "
                                "Larger = more tolerant of PC stalls. Doubled automatically after an underflow.")
        f.addRow("Buffer / latency", self.latency)
        self.ai_range = QtWidgets.QComboBox()
        self.ai_range.addItem(f"configured (±{dc.ai_range_v:g} V)", None)
        for v in AI_RANGES_V:
            self.ai_range.addItem(f"±{v:g} V", v)
        self.ai_range.setToolTip("Video input range: the smallest range that contains the signal uses the "
                                 "full 16-bit resolution.")
        f.addRow("Video input range", self.ai_range)
        self.chunk = _dspin(0.5, 100, 1, " ms", 0.5)
        self.chunk.setToolTip("Lines are grouped into output blocks of about this length.")
        f.addRow("Output block", self.chunk)
        self.publish = _dspin(1, 120, 0, " Hz", 5)
        self.publish.setToolTip("Maximum rate of partial-image updates during slow scans.")
        f.addRow("Partial updates", self.publish)
        self.restart = QtWidgets.QCheckBox("Apply setting changes immediately (restart the frame)")
        f.addRow(self.restart)
        self.recover = QtWidgets.QCheckBox("Recover automatically from buffer underflows")
        f.addRow(self.recover)
        info = QtWidgets.QLabel(
            f"Device {dc.device}: AO ≤ {dc.max_ao_rate / 1e6:g} MS/s, AI ≤ {dc.max_ai_rate(1) / 1e6:g} MS/s "
            f"(1 ch), blanker {dc.blank_line} (active {'high' if dc.blank_active_high else 'low'}), "
            f"AI terminal {dc.ai_terminal}."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f.addRow(info)
        return w

    def _fill_rates(self):
        """Rates the device can do now (the limit depends on the AO channel count, e.g. the phantom)."""
        dc = self.engine.daq_config
        self.rate.clear()
        for i, r in enumerate(valid_rates(dc.max_scan_rate(len(self.engine.config.ai_channels)))):
            label = f"{r / 1e6:.4g} MS/s" if r >= 1e6 else f"{r / 1e3:g} kS/s"
            self.rate.addItem(label + ("  (fastest)" if i == 0 else ""), None if i == 0 else r)

    def _phantom_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        dc = self.engine.daq_config
        self.ph_on = QtWidgets.QCheckBox("Generate the phantom test chart (dummy image)")
        self.ph_on.setToolTip("Every scan also drives a synthetic test chart on an aux output, sample for "
                              "sample on the scan clock. Wire it to the video input to image it.")
        f.addRow(self.ph_on)
        self.ph_out = QtWidgets.QComboBox()
        for name, phys in dc.aux_outputs.items():
            self.ph_out.addItem(f"{name} = {phys}" if phys else f"{name} (dummy – no channel on this board)", name)
        f.addRow("Output", self.ph_out)
        self.ph_noise = _dspin(0.0, 2.0, 3, " V rms", 0.05)
        self.ph_noise.setToolTip("White noise added to every sample (applies immediately, even while scanning)")
        f.addRow("Synthetic noise", self.ph_noise)
        video = dc.ai_channels.get("video", "ai0")
        note = QtWidgets.QLabel(
            f"Wire the chosen output to the video input ({video}{'+, AO GND → ' + video + '−' if dc.ai_terminal == 'DIFF' else ''}). "
            "Switching on/off briefly restarts scanning: the extra AO channel lowers the fastest clock "
            f"(here {dc.max_ao_rate_for(2) / 1e6:g} → {dc.max_ao_rate_for(3) / 1e6:g} MS/s). "
            "The chart is fixed to the DAQ field: zoom and beam shift move across it."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa")
        f.addRow(note)
        return w

    def _scan_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.width = _dspin(0.001, 1e5, 3, " µm", 1)
        f.addRow("Scan width", self.width)
        self.dwell = _dspin(0.01, 100000, 3, " µs", 0.5)
        self.dwell.setToolTip("Pixel dwell (quantised to whole sample-clock periods)")
        f.addRow("Pixel dwell", self.dwell)
        self.rotation = _dspin(-180, 180, 2, " °", 1)
        f.addRow("Scan rotation", self.rotation)
        self.bidir = QtWidgets.QCheckBox("Bidirectional (serpentine) scan")
        f.addRow(self.bidir)
        self.flyback = _dspin(0, 10000, 1, " µs", 5)
        f.addRow("Flyback", self.flyback)
        self.fb_shape = QtWidgets.QComboBox()
        self.fb_shape.addItems(["linear", "cosine", "step"])
        f.addRow("Flyback shape", self.fb_shape)
        self.preline = _dspin(0, 10000, 1, " µs", 5)
        f.addRow("Pre-line (lead-in)", self.preline)
        self.pre_mode = QtWidgets.QComboBox()
        self.pre_mode.addItems(["ramp", "hold"])
        f.addRow("Pre-line mode", self.pre_mode)
        self.blank_pre = QtWidgets.QCheckBox("Blank during pre-line")
        f.addRow(self.blank_pre)
        self.fsettle = _dspin(0, 100000, 0, " µs", 50)
        f.addRow("Frame settle", self.fsettle)
        self.psettle = QtWidgets.QSpinBox()
        self.psettle.setRange(0, 10000)
        self.psettle.setSuffix(" samples")
        self.psettle.setToolTip("Samples discarded at the start of every pixel")
        f.addRow("Pixel settle", self.psettle)
        self.lag = _dspin(0, 10000, 3, " µs", 0.1)
        self.lag.setToolTip("Deflection + detector lag compensation (the alignment tool measures it)")
        f.addRow("Lag compensation", self.lag)
        self.ai_lat = _dspin(0, 100, 2, " samples", 0.1)
        self.ai_lat.setToolTip("AO→AI conversion latency (≈1 sample on X-series; scripts/measure_delay.py)")
        f.addRow("AI latency", self.ai_lat)
        return w

    def _cal_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.field = _dspin(0.01, 1e6, 3, " µm", 10)
        self.field.setToolTip("Field covered by the full AO range (±V) at the current SEM magnification")
        f.addRow("Full-scale field", self.field)
        self.inv_x = QtWidgets.QCheckBox("Invert X")
        self.inv_y = QtWidgets.QCheckBox("Invert Y")
        f.addRow(self.inv_x)
        f.addRow(self.inv_y)
        note = QtWidgets.QLabel("Recalibrate whenever the SEM magnification changes; the scale bar and "
                                "beam shift use this calibration.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa")
        f.addRow(note)
        return w

    def _save_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton("…")
        b.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        f.addRow("Capture folder", row)
        self.auto_save = QtWidgets.QCheckBox("Save every capture automatically")
        f.addRow(self.auto_save)
        self.f_tif = QtWidgets.QCheckBox("16-bit TIFF (raw volts)")
        self.f_png = QtWidgets.QCheckBox("PNG (display contrast)")
        self.f_npy = QtWidgets.QCheckBox("NumPy .npy (float volts)")
        self.f_json = QtWidgets.QCheckBox("JSON metadata")
        self.data_bar = QtWidgets.QCheckBox("Also save a PNG with the data bar / scale bar")
        for c in (self.f_tif, self.f_png, self.f_npy, self.f_json, self.data_bar):
            f.addRow(c)
        return w

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Capture folder", self.folder.text())
        if d:
            self.folder.setText(d)

    # ------------------------------------------------------------- load/apply
    def load(self):
        e = self.engine
        o, c, cal = e.options, e.config, e.calibration
        self._fill_rates()
        i = 0 if o.rate is None else self.rate.findData(c.rate)
        self.rate.setCurrentIndex(max(0, i))
        ph = e.handle.phantom
        self.ph_on.setChecked(ph["output"] is not None)
        self.ph_out.setCurrentIndex(max(0, self.ph_out.findData(ph["output"] or "aux0")))
        self.ph_noise.setValue(ph["noise_v"])
        self.latency.setValue(o.latency_s * 1e3)
        self.ai_range.setCurrentIndex(max(0, self.ai_range.findData(o.ai_range_v)))
        self.chunk.setValue(o.chunk_s * 1e3)
        self.publish.setValue(o.publish_hz)
        self.restart.setChecked(o.restart_on_change)
        self.recover.setChecked(o.auto_recover)
        self.width.setValue(c.width_um)
        self.dwell.setValue(c.dwell_s * 1e6)
        self.rotation.setValue(math.degrees(c.rotation_rad))
        self.bidir.setChecked(c.bidirectional)
        self.flyback.setValue(c.flyback_s * 1e6)
        self.fb_shape.setCurrentText(c.flyback_shape)
        self.preline.setValue(c.preline_s * 1e6)
        self.pre_mode.setCurrentText(c.preline_mode)
        self.blank_pre.setChecked(c.blank_preline)
        self.fsettle.setValue(c.frame_settle_s * 1e6)
        self.psettle.setValue(c.settle_samples)
        self.lag.setValue(c.lag_comp_s * 1e6)
        self.ai_lat.setValue(c.ai_latency_samples)
        self.field.setValue(cal.field_size_um)
        self.inv_x.setChecked(cal.invert_x)
        self.inv_y.setChecked(cal.invert_y)
        s = self.save
        self.folder.setText(s.folder)
        self.auto_save.setChecked(s.auto_save)
        for cb, k in ((self.f_tif, "tif"), (self.f_png, "png"), (self.f_npy, "npy"), (self.f_json, "json")):
            cb.setChecked(k in s.formats)
        self.data_bar.setChecked(s.data_bar)
        self.msg.setText("")

    def apply(self):
        e = self.engine
        try:
            cal = e.calibration
            new_cal = FieldCalibration(self.field.value(), cal.v_full_scale, cal.correction,
                                       self.inv_x.isChecked(), self.inv_y.isChecked(), cal.name)
            if new_cal != cal:
                e.set_calibration(new_cal)
            # phantom first: it changes the AO channel count and with it the fastest clock
            want = self.ph_out.currentData() if self.ph_on.isChecked() else None
            ph = e.handle.phantom
            if want != ph["output"] or abs(self.ph_noise.value() - ph["noise_v"]) > 1e-9:
                e.set_phantom(want, self.ph_noise.value())
                if want != ph["output"]:
                    self._fill_rates()
            rate = self.rate.currentData()
            if rate is not None and rate > e.daq_config.max_scan_rate(len(e.config.ai_channels)) * (1 + 1e-9):
                rate = None
            e.set_options(
                rate=rate,
                latency_s=self.latency.value() / 1e3,
                ai_range_v=self.ai_range.currentData(),
                chunk_s=self.chunk.value() / 1e3,
                publish_hz=self.publish.value(),
                restart_on_change=self.restart.isChecked(),
                auto_recover=self.recover.isChecked(),
            )
            e.update(
                width_um=self.width.value(),
                dwell_s=self.dwell.value() * 1e-6,
                rotation_rad=math.radians(self.rotation.value()),
                bidirectional=self.bidir.isChecked(),
                flyback_s=self.flyback.value() * 1e-6,
                flyback_shape=self.fb_shape.currentText(),
                preline_s=self.preline.value() * 1e-6,
                preline_mode=self.pre_mode.currentText(),
                blank_preline=self.blank_pre.isChecked(),
                frame_settle_s=self.fsettle.value() * 1e-6,
                settle_samples=self.psettle.value(),
                lag_comp_s=self.lag.value() * 1e-6,
                ai_latency_samples=self.ai_lat.value(),
            )
            s = self.save
            s.folder = self.folder.text().strip() or "captures"
            s.auto_save = self.auto_save.isChecked()
            s.formats = tuple(k for cb, k in ((self.f_tif, "tif"), (self.f_png, "png"), (self.f_npy, "npy"),
                                              (self.f_json, "json")) if cb.isChecked())
            s.data_bar = self.data_bar.isChecked()
            self.msg.setStyleSheet("color:#7c7")
            self.msg.setText("Applied.")
            self.applied.emit()
        except Exception as ex:  # noqa: BLE001
            self.msg.setStyleSheet("color:#f77")
            self.msg.setText(f"Not applied: {ex}")
