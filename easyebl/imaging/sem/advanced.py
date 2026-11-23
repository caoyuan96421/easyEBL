"""'Advanced…' dialog: hardware, scan waveform, field calibration and saving settings.

The scan *size, position and rotation* are in the main window (Field of view); the pixel
dwell and the line / frame timing come with the speed preset (shown in the status bar).
Here are only the knobs no preset sets.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..engine import AI_RANGES_V, ImagingEngine, valid_rates
from .display import format_length

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
    """*field_scale* is the global field calibration (:class:`~easyebl.microscope.coupling.FieldScale`);
    *measured* returns the length (µm) of the measuring line drawn on the image, or raises ValueError
    saying why there is none to calibrate with."""

    applied = QtCore.Signal()
    calibrated = QtCore.Signal(float)  # K was scaled by this ratio (true / measured length)

    def __init__(self, engine: ImagingEngine, save: SaveSettings, parent=None, field_scale=None, measured=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.engine = engine
        self.save = save
        self.fs = field_scale
        self.measured = measured
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._hw_tab(), "Hardware")
        tabs.addTab(self._scan_tab(), "Scan waveform")
        tabs.addTab(self._cal_tab(), "Calibration")
        tabs.addTab(self._save_tab(), "Saving")
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
        self.resize(520, 560)
        self._cal_timer = QtCore.QTimer(self)
        self._cal_timer.timeout.connect(self._refresh_cal)
        self._cal_timer.start(250)
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
        self.ai_mode = QtWidgets.QComboBox()
        for mode, text in (("RSE", "RSE – single-ended, AI+ vs AI GND (default)"),
                           ("DIFF", "DIFF – differential, AI+ and AI− both wired")):
            self.ai_mode.addItem(text, mode)
        if self.engine.handle.ai_terminal not in ("RSE", "DIFF"):  # set from config / console
            self.ai_mode.addItem(self.engine.handle.ai_terminal, self.engine.handle.ai_terminal)
        self.ai_mode.setToolTip("How the video input is wired. In DIFF mode an unconnected AI− floats and drives "
                                "the reading into the ±range rail (saturated image). Applying restarts scanning.")
        f.addRow("Video input mode", self.ai_mode)
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
            f"(1 ch), blanker {dc.blank_line} (active {'high' if dc.blank_active_high else 'low'})."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa")
        f.addRow(info)
        return w

    def _fill_rates(self):
        """Rates the device can do now."""
        dc = self.engine.daq_config
        self.rate.clear()
        for i, r in enumerate(valid_rates(dc.max_scan_rate(len(self.engine.config.ai_channels)))):
            label = f"{r / 1e6:.4g} MS/s" if r >= 1e6 else f"{r / 1e3:g} kS/s"
            self.rate.addItem(label + ("  (fastest)" if i == 0 else ""), None if i == 0 else r)

    def _scan_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.bidir = QtWidgets.QCheckBox("Bidirectional (serpentine) scan")
        f.addRow(self.bidir)
        self.pre_mode = QtWidgets.QComboBox()
        self.pre_mode.addItems(["ramp", "hold"])
        self.pre_mode.setToolTip("ramp: the beam already moves at line speed before the first pixel; "
                                 "hold: it waits at the line start")
        f.addRow("Pre-line mode", self.pre_mode)
        self.blank_pre = QtWidgets.QCheckBox("Blank during pre-line")
        f.addRow(self.blank_pre)
        self.fb_shape = QtWidgets.QComboBox()
        self.fb_shape.addItems(["linear", "cosine", "step"])
        f.addRow("Flyback shape", self.fb_shape)
        self.lag = _dspin(0, 10000, 3, " µs", 0.1)
        self.lag.setToolTip("Deflection + detector lag compensation (the alignment tool measures it)")
        f.addRow("Lag compensation", self.lag)
        self.ai_lat = _dspin(0, 100, 2, " samples", 0.1)
        self.ai_lat.setToolTip("AO→AI conversion latency (≈1 sample on X-series)")
        f.addRow("AI latency", self.ai_lat)
        note = QtWidgets.QLabel("Size, position and rotation: the Field of view group of the main window.  "
                                "Pixel dwell, pre-line, flyback and settling come with the scan speed "
                                "(status bar).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa")
        f.addRow(note)
        return w

    def _cal_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Global calibration (every magnification)")
        f = QtWidgets.QFormLayout(g)
        self.k_field = _dspin(1.0, 1e9, 0, " µm", 1000)
        self.k_field.setToolTip("field swept by the full deflection range at a hypothetical ×1;\n"
                                "the field at magnification M is this / M")
        f.addRow("Field at ×1", self.k_field)
        self.l_k_upv = QtWidgets.QLabel()
        f.addRow("Scale at ×1", self.l_k_upv)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Current setting")
        f = QtWidgets.QFormLayout(g)
        self.l_mag_src = QtWidgets.QLabel()
        self.mag = _dspin(1, 1e7, 0, "", 100)
        self.mag.setPrefix("×")
        self.mag.setToolTip("used while the microscope does not report its magnification")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.mag)
        row.addWidget(self.l_mag_src, 1)
        f.addRow("Magnification", row)
        self.l_field = QtWidgets.QLabel()
        f.addRow("Field", self.l_field)
        self.l_upv = QtWidgets.QLabel()
        f.addRow("Scale", self.l_upv)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Calibrate with the measuring line")
        f = QtWidgets.QFormLayout(g)
        self.l_line = QtWidgets.QLabel()
        self.l_line.setWordWrap(True)
        f.addRow("Line on the image", self.l_line)
        self.true_len = _dspin(0.0001, 1e6, 4, " µm", 1)
        self.true_len.setValue(10.0)
        self.true_len.setToolTip("the real length of the feature the line spans (e.g. a grating pitch)")
        f.addRow("Real length", self.true_len)
        self.b_cal = QtWidgets.QPushButton("Calibrate")
        self.b_cal.setToolTip("scale the global calibration so the line measures the real length")
        self.b_cal.clicked.connect(self.calibrate_from_line)
        f.addRow(self.b_cal)
        hint = QtWidgets.QLabel("Switch on Measure in the toolbar and drag a line across a feature of known "
                                "size; a long line (most of the image) gives the best accuracy.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa")
        f.addRow(hint)
        lay.addWidget(g)
        if self.fs is None:
            for gb in w.findChildren(QtWidgets.QGroupBox):
                gb.setEnabled(False)
        lay.addStretch(1)
        return w

    def _refresh_cal(self):
        """The read-only parts of the calibration tab (the magnification can change at any time)."""
        fs = self.fs
        if fs is None or not self.isVisible():
            return
        v = fs.v_full_scale
        self.l_k_upv.setText(f"{format_length(fs.k_um / (2 * v))}/V  (±{v:g} V full scale)")
        ext = fs.from_microscope
        self.mag.setEnabled(not ext)
        if ext:
            self.mag.blockSignals(True)
            self.mag.setValue(fs.magnification)
            self.mag.blockSignals(False)
        self.l_mag_src.setText("from the SEM" if ext else "not reported by the SEM: enter it")
        self.l_field.setText(f"{format_length(fs.field_um)} (×{fs.magnification:g})")
        self.l_upv.setText(f"{format_length(fs.um_per_volt())}/V")
        try:
            length = self.measured() if self.measured is not None else None
            self.l_line.setText("none" if length is None else format_length(length))
            ok = length is not None
        except ValueError as e:
            self.l_line.setText(str(e))
            ok = False
        self.b_cal.setEnabled(ok)

    def calibrate_from_line(self):
        try:
            length = self.measured()
            r = self.fs.calibrate(length, self.true_len.value())
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.msg.setText(f"<span style='color:#f66'>Not calibrated: {e}</span>")
            return
        self.k_field.setValue(self.fs.k_um)
        self.msg.setText(f"Calibrated: field at ×1 = {self.fs.k_um:.6g} µm (×{r:.5f}); the field is now "
                         f"{format_length(self.fs.field_um)}")
        self.calibrated.emit(r)
        self._refresh_cal()

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
        o, c = e.options, e.config
        self._fill_rates()
        i = 0 if o.rate is None else self.rate.findData(c.rate)
        self.rate.setCurrentIndex(max(0, i))
        self.latency.setValue(o.latency_s * 1e3)
        self.ai_range.setCurrentIndex(max(0, self.ai_range.findData(o.ai_range_v)))
        self.ai_mode.setCurrentIndex(max(0, self.ai_mode.findData(e.handle.ai_terminal)))
        self.chunk.setValue(o.chunk_s * 1e3)
        self.publish.setValue(o.publish_hz)
        self.restart.setChecked(o.restart_on_change)
        self.recover.setChecked(o.auto_recover)
        self.bidir.setChecked(c.bidirectional)
        self.fb_shape.setCurrentText(c.flyback_shape)
        self.pre_mode.setCurrentText(c.preline_mode)
        self.blank_pre.setChecked(c.blank_preline)
        self.lag.setValue(c.lag_comp_s * 1e6)
        self.ai_lat.setValue(c.ai_latency_samples)
        if self.fs is not None:
            self.k_field.setValue(self.fs.k_um)
            self.mag.setValue(self.fs.magnification)
            self._refresh_cal()
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
            fs = self.fs
            if fs is not None:
                if not fs.from_microscope and abs(self.mag.value() - fs.manual_magnification) > 1e-9:
                    fs.set_manual_magnification(self.mag.value())
                if abs(self.k_field.value() - fs.k_um) > 0.5:  # the box shows whole µm: not a change
                    fs.set_k(self.k_field.value())
            if self.ai_mode.currentData() != e.handle.ai_terminal:
                e.set_ai_terminal(self.ai_mode.currentData())
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
                bidirectional=self.bidir.isChecked(),
                flyback_shape=self.fb_shape.currentText(),
                preline_mode=self.pre_mode.currentText(),
                blank_preline=self.blank_pre.isChecked(),
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
