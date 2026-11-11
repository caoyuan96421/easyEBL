"""Energy calibration dialog: MCA channel → keV from the peaks of a known standard."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..analysis import STANDARDS, auto_assign, find_peaks, fit_calibration, fit_resolution, line_energy, standard_lines
from ..spectrum import EnergyCalibration, Spectrum

__all__ = ["CalibrationDialog"]


class CalibrationDialog(QtWidgets.QDialog):
    """Calibrate channels → energy.

    * spectrum: the current one, or acquire a new one (callback ``acquire(seconds) -> Future[Spectrum]``);
    * standard: lines expected in the spectrum (or type your own: "Cu Ka1" or a number in keV);
    * peaks: found automatically, assigned to lines (editable table);
    * fit: linear (offset + gain) or quadratic; residuals in eV; detector FWHM at Mn Kα.

    ``calibrationApplied(EnergyCalibration, fwhm_mn_ev, fano)`` is emitted on Apply.
    """

    calibrationApplied = QtCore.Signal(object, float, float)

    COLS = ["Use", "Line", "Energy (keV)", "Channel", "FWHM (eV)", "Residual (eV)"]

    def __init__(self, spectrum: Spectrum | None, current: EnergyCalibration, acquire=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Energy calibration")
        self.resize(980, 680)
        self.spectrum = spectrum
        self.current = current
        self.acquire = acquire
        self._fit = None
        self._future = None
        self._peaks = []

        # --- spectrum source
        src = QtWidgets.QGroupBox("Calibration spectrum")
        sl = QtWidgets.QHBoxLayout(src)
        self.l_src = QtWidgets.QLabel()
        self.sp_time = QtWidgets.QDoubleSpinBox()
        self.sp_time.setRange(1, 3600)
        self.sp_time.setValue(60)
        self.sp_time.setSuffix(" s")
        self.b_acq = QtWidgets.QPushButton("Acquire new")
        self.b_acq.setEnabled(acquire is not None)
        self.b_acq.clicked.connect(self._acquire)
        sl.addWidget(self.l_src, 1)
        sl.addWidget(QtWidgets.QLabel("Live:"))
        sl.addWidget(self.sp_time)
        sl.addWidget(self.b_acq)

        # --- standard
        std = QtWidgets.QGroupBox("Standard")
        stl = QtWidgets.QHBoxLayout(std)
        self.cb_std = QtWidgets.QComboBox()
        self.cb_std.addItems(list(STANDARDS))
        self.b_auto = QtWidgets.QPushButton("Find peaks && assign")
        self.b_auto.clicked.connect(self.auto)
        self.b_add = QtWidgets.QPushButton("Add row")
        self.b_add.clicked.connect(lambda: self._add_row("", float("nan"), float("nan"), float("nan"), True))
        self.b_del = QtWidgets.QPushButton("Remove row")
        self.b_del.clicked.connect(self._remove_row)
        stl.addWidget(self.cb_std, 1)
        stl.addWidget(self.b_auto)
        stl.addWidget(self.b_add)
        stl.addWidget(self.b_del)

        # --- plot (channels)
        self.plot = pg.PlotWidget(background="#1e1f22")
        self.plot.setLabel("bottom", "Channel")
        self.plot.setLabel("left", "Counts")
        self.plot.setLogMode(y=False)
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#ffd54f", width=1.2), stepMode="center")
        self.peak_marks = pg.ScatterPlotItem(symbol="t", size=12, brush=pg.mkBrush("#ff6b6b"), pen=None)
        self.plot.addItem(self.peak_marks)
        self._labels = []
        self.plot.scene().sigMouseClicked.connect(self._plot_click)

        # --- table
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.itemChanged.connect(lambda *_: self.refit())

        # --- fit
        fitg = QtWidgets.QGroupBox("Fit")
        fl = QtWidgets.QGridLayout(fitg)
        self.cb_order = QtWidgets.QComboBox()
        self.cb_order.addItems(["Linear (offset + gain)", "Quadratic"])
        self.cb_order.currentIndexChanged.connect(lambda *_: self.refit())
        self.l_res = QtWidgets.QLabel("–")
        self.l_res.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.l_cur = QtWidgets.QLabel(self._cal_text(current))
        fl.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
        fl.addWidget(self.cb_order, 0, 1)
        fl.addWidget(QtWidgets.QLabel("Current:"), 1, 0)
        fl.addWidget(self.l_cur, 1, 1)
        fl.addWidget(QtWidgets.QLabel("New:"), 2, 0)
        fl.addWidget(self.l_res, 2, 1)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close)
        self.b_apply = bb.button(QtWidgets.QDialogButtonBox.Apply)
        self.b_apply.setEnabled(False)
        self.b_apply.clicked.connect(self.apply)
        bb.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(src)
        lay.addWidget(std)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([380, 200])
        lay.addWidget(split, 1)
        lay.addWidget(fitg)
        lay.addWidget(bb)
        self.l_hint = QtWidgets.QLabel("Click a peak in the plot to add it; type a line label (\"Cu Ka1\") or an "
                                       "energy in keV in the Line column.")
        self.l_hint.setStyleSheet("color:#999")
        lay.addWidget(self.l_hint)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(200)
        self._show_spectrum()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _cal_text(c: EnergyCalibration) -> str:
        s = f"E = {c.offset_kev * 1e3:+.2f} eV + {c.gain_kev * 1e3:.5f} eV × ch"
        if c.quad_kev:
            s += f" {c.quad_kev * 1e3:+.3e} eV × ch²"
        return s

    def _show_spectrum(self):
        sp = self.spectrum
        if sp is None:
            self.l_src.setText("No spectrum – acquire one of the standard.")
            return
        self.l_src.setText(f"{sp.title or 'spectrum'}: {sp.total:,} counts, {sp.real_time:.1f} s real, "
                           f"{sp.n_channels} channels")
        k = np.arange(sp.n_channels + 1) - 0.5
        self.curve.setData(k, sp.counts.astype(float))
        self.plot.autoRange()

    def _acquire(self):
        if self.acquire is None or self._future is not None:
            return
        try:
            self._future = self.acquire(float(self.sp_time.value()))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Acquire", str(e))
            return
        self.b_acq.setEnabled(False)
        self.l_src.setText("Acquiring…")

    def _poll(self):
        f = self._future
        if f is None:
            return
        part = getattr(f, "partial", None)
        if isinstance(part, Spectrum) and part is not self.spectrum:
            self.spectrum = part
            self._show_spectrum()
        if f.done():
            self._future = None
            self.b_acq.setEnabled(True)
            try:
                self.spectrum = f.result()
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Acquire", str(e))
                return
            self._show_spectrum()
            self.auto()

    def _spectrum_now(self) -> Spectrum | None:
        sp = self.spectrum
        return None if sp is None else sp.with_calibration(self.current)

    # ------------------------------------------------------------ table
    def _add_row(self, label, energy, channel, fwhm_ev, use=True):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk.setCheckState(QtCore.Qt.Checked if use else QtCore.Qt.Unchecked)
        self.table.setItem(r, 0, chk)
        vals = [label, "" if not np.isfinite(energy) else f"{energy:.4f}",
                "" if not np.isfinite(channel) else f"{channel:.3f}",
                "" if not np.isfinite(fwhm_ev) else f"{fwhm_ev:.0f}", ""]
        for c, v in enumerate(vals, start=1):
            it = QtWidgets.QTableWidgetItem(v)
            if c in (4, 5):
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, c, it)
        self.table.blockSignals(False)

    def _remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows or [self.table.rowCount() - 1]:
            if r >= 0:
                self.table.removeRow(r)
        self.refit()

    def rows(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            def txt(c):
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            label, e_txt, ch_txt = txt(1), txt(2), txt(3)
            energy = float("nan")
            if e_txt:
                try:
                    energy = float(e_txt)
                except ValueError:
                    pass
            if not np.isfinite(energy) and label:
                try:
                    energy = float(label)
                except ValueError:
                    try:
                        energy = line_energy(label)
                    except Exception:  # noqa: BLE001
                        pass
            try:
                ch = float(ch_txt)
            except ValueError:
                ch = float("nan")
            use = self.table.item(r, 0).checkState() == QtCore.Qt.Checked
            out.append({"row": r, "label": label, "energy": energy, "channel": ch, "use": use})
        return out

    # ------------------------------------------------------------ actions
    def auto(self):
        sp = self._spectrum_now()
        if sp is None:
            return
        self._peaks = find_peaks(sp)
        lines = standard_lines(self.cb_std.currentText())
        pairs = auto_assign(sp, lines, peaks=self._peaks)
        self.table.setRowCount(0)
        for lab, e, p in pairs:
            self._add_row(lab, e, p.channel, p.fwhm_ch * sp.calibration.channel_width(p.channel) * 1e3)
        self._mark_peaks()
        self.refit()
        if len(pairs) < 2:
            self.l_hint.setText("Fewer than two lines matched – click the peaks to add them by hand.")

    def _mark_peaks(self):
        for t in self._labels:
            self.plot.removeItem(t)
        self._labels = []
        sp = self.spectrum
        if sp is None or not self._peaks:
            self.peak_marks.setData([], [])
            return
        xs = [p.channel for p in self._peaks]
        ys = [float(sp.counts[int(round(p.channel))]) for p in self._peaks]
        self.peak_marks.setData(xs, ys)
        for x, y in zip(xs, ys):
            t = pg.TextItem(f"{x:.1f}", color="#ff8787", anchor=(0.5, 1.6))
            t.setPos(x, y)
            self.plot.addItem(t)
            self._labels.append(t)

    def _plot_click(self, ev):
        sp = self.spectrum
        if sp is None:
            return
        p = self.plot.getPlotItem().vb.mapSceneToView(ev.scenePos())
        from ..analysis import refine_peak
        from ..analysis.background import snip_background

        bg = snip_background(sp.counts, self.current)
        k = float(p.x())
        sig = 0.13 / 2.355 / max(self.current.gain_kev, 1e-6)
        pk = refine_peak(sp.counts, k, sig, bg)
        if pk is None:
            return
        e_guess = self.current.energy(pk.channel)
        from ... import xray

        cand = xray.lines_near(e_guess, tol=0.15, min_weight=0.3)
        lab = cand[0].label if cand else ""
        self._add_row(lab, line_energy(lab) if lab else float("nan"), pk.channel,
                      pk.fwhm_ch * self.current.gain_kev * 1e3)
        self.refit()

    def refit(self):
        rows = [r for r in self.rows() if r["use"] and np.isfinite(r["energy"]) and np.isfinite(r["channel"])]
        self.b_apply.setEnabled(False)
        self._fit = None
        if not rows:
            self.l_res.setText("need ≥ 1 assigned peak (≥ 2 for offset + gain)")
            return
        order = 2 if self.cb_order.currentIndex() == 1 else 1
        try:
            fit = fit_calibration([r["channel"] for r in rows], [r["energy"] for r in rows], order=order,
                                  labels=[r["label"] for r in rows], name=self.cb_std.currentText())
        except ValueError as e:
            self.l_res.setText(str(e))
            return
        self._fit = fit
        self.table.blockSignals(True)
        for r, res in zip(rows, fit.residuals_ev):
            self.table.item(r["row"], 5).setText(f"{res:+.1f}")
        self.table.blockSignals(False)
        txt = self._cal_text(fit.calibration) + f"   rms {fit.rms_ev:.1f} eV ({len(rows)} peaks)"
        fw = self._resolution(fit.calibration)
        if fw is not None:
            txt += f"   FWHM@Mn Kα ≈ {fw[0]:.0f} eV"
        self.l_res.setText(txt)
        self.b_apply.setEnabled(True)

    def _resolution(self, cal: EnergyCalibration):
        pts = []
        for r in self.rows():
            it = self.table.item(r["row"], 4)
            if r["use"] and it and it.text() and np.isfinite(r["energy"]):
                try:
                    fw_ch = float(it.text()) / 1e3 / max(self.current.gain_kev, 1e-9)
                except ValueError:
                    continue
                pts.append((r["energy"], fw_ch * cal.gain_kev))
        if not pts:
            return None
        return fit_resolution(pts)

    def apply(self):
        if self._fit is None:
            return
        fw = self._resolution(self._fit.calibration) or (float("nan"), float("nan"))
        self.calibrationApplied.emit(self._fit.calibration, float(fw[0]), float(fw[1]))
        self.current = self._fit.calibration
        self.l_cur.setText(self._cal_text(self.current))
