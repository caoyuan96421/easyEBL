"""Energy calibration window: a linear channel → energy line from peaks of a known sample.

* **Offset / slope** are shown and can be typed in directly (eV, eV per channel).
* **Spectrum** of the calibration sample: the current one, or *Acquire*.  Click a peak to
  put the marker on it (it snaps to the peak centre).
* **Periodic table**: click the element that made the marked peak.  Its line nearest to the
  marker (by the current calibration) is paired with the peak; the line can be changed in
  the table.  One pair sets the slope (offset kept); two or more fit offset and slope.
* **Pairs persist** across spectra (and sessions): e.g. Si Kα from one sample, then acquire on
  a Cu sample and add Cu Kα – the fit uses both.  ``pointsChanged(list)`` reports every change
  (as dicts, see :meth:`CalibrationDialog.points`).
* **Clear calibration**: offset 0, 10 eV per channel, no pairs.

``calibrationApplied(EnergyCalibration)`` is emitted on *Apply*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ... import xray
from ..analysis import find_peaks, refine_peak, snip_background
from ..spectrum import DEFAULT_GAIN_KEV, EnergyCalibration, Spectrum
from .views import PeriodicTableWidget

__all__ = ["CalibrationDialog", "PeakLine", "element_peaks"]

_GREEK = {"a": "α", "b": "β", "g": "γ", "l": "ℓ", "n": "η"}


@dataclass(frozen=True)
class PeakLine:
    """One peak an element produces on an SDD (lines closer than the resolution merged)."""

    label: str  # "Cu Kα"
    energy_kev: float


def element_peaks(element: str, e_min_kev: float = 0.2, merge_kev: float = 0.06) -> list[PeakLine]:
    """The resolvable peaks of *element*: lines of one family within *merge_kev* are one peak
    at their weighted mean energy (Kα1 + Kα2, Lα + Lβ1 …), labelled by the strongest line."""
    ls = sorted((ln for ln in xray.lines(element, min_weight=0.05) if ln.energy >= e_min_kev),
                key=lambda ln: ln.energy)
    groups: list[list] = []
    for ln in ls:
        g = groups[-1] if groups else None
        if g and g[0].family == ln.family and ln.energy - g[-1].energy <= merge_kev:
            g.append(ln)
        else:
            groups.append([ln])
    out = []
    for g in groups:
        w = np.array([ln.weight for ln in g])
        e = float(np.dot(w, [ln.energy for ln in g]) / w.sum())
        main = max(g, key=lambda ln: ln.weight)
        name = main.name[0] + _GREEK.get(main.name[1:2].lower(), main.name[1:2]) if len(main.name) > 1 else main.name
        out.append(PeakLine(f"{element} {name}", e))
    return out


@dataclass
class _Pair:
    element: str
    line: PeakLine
    channel: float
    source: str = ""  # the spectrum the peak was marked in

    def to_dict(self) -> dict:
        return {"element": self.element, "label": self.line.label, "energy_kev": self.line.energy_kev,
                "channel": self.channel, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "_Pair":
        return cls(str(d["element"]), PeakLine(str(d["label"]), float(d["energy_kev"])), float(d["channel"]),
                   str(d.get("source", "")))


def _source(sp: Spectrum | None) -> str:
    if sp is None:
        return ""
    t = (sp.timestamp or "").replace("T", " ")[5:16]
    return f"{sp.title or 'spectrum'} {t}".strip()


class CalibrationDialog(QtWidgets.QDialog):
    """Linear energy calibration (see the module docstring)."""

    calibrationApplied = QtCore.Signal(object)
    pointsChanged = QtCore.Signal(list)

    COLS = ["Line", "Energy (keV)", "Channel", "Residual (eV)", "Spectrum"]

    def __init__(self, spectrum: Spectrum | None, current: EnergyCalibration, acquire=None, parent=None,
                 points: list | None = None):
        super().__init__(parent)
        self.setWindowTitle("Energy calibration")
        self.resize(1150, 760)
        self.spectrum = spectrum
        self.applied = current
        self.acquire = acquire
        self.pairs: list[_Pair] = []
        for d in points or []:
            try:
                self.pairs.append(_Pair.from_dict(d))
            except (KeyError, TypeError, ValueError):
                continue
        self.pairs.sort(key=lambda p: p.line.energy_kev)
        self.marker: float | None = None  # marked peak (fractional channel)
        self._future = None
        self._syncing = False

        # --- calibration (offset / slope)
        g = QtWidgets.QGroupBox("Calibration   E = offset + slope × channel")
        cl = QtWidgets.QHBoxLayout(g)
        self.sp_offset = QtWidgets.QDoubleSpinBox()
        self.sp_offset.setRange(-5000, 5000)
        self.sp_offset.setDecimals(2)
        self.sp_offset.setSuffix(" eV")
        self.sp_slope = QtWidgets.QDoubleSpinBox()
        self.sp_slope.setRange(0.001, 1000)
        self.sp_slope.setDecimals(4)
        self.sp_slope.setSuffix(" eV/ch")
        for sp in (self.sp_offset, self.sp_slope):
            sp.setKeyboardTracking(False)
            sp.setMinimumWidth(130)
            sp.valueChanged.connect(self._typed)
        self.b_clear = QtWidgets.QPushButton("Clear calibration")
        self.b_clear.setToolTip(f"offset 0, {DEFAULT_GAIN_KEV * 1e3:g} eV per channel; removes all pairs")
        self.b_clear.clicked.connect(self.clear)
        self.l_applied = QtWidgets.QLabel()
        self.l_applied.setStyleSheet("color:#9aa")
        cl.addWidget(QtWidgets.QLabel("Offset"))
        cl.addWidget(self.sp_offset)
        cl.addSpacing(12)
        cl.addWidget(QtWidgets.QLabel("Slope"))
        cl.addWidget(self.sp_slope)
        cl.addSpacing(12)
        cl.addWidget(self.b_clear)
        cl.addStretch(1)
        cl.addWidget(self.l_applied)

        # --- spectrum
        sg = QtWidgets.QGroupBox("Calibration spectrum (click a peak to mark it)")
        sl = QtWidgets.QVBoxLayout(sg)
        row = QtWidgets.QHBoxLayout()
        self.l_src = QtWidgets.QLabel()
        self.sp_time = QtWidgets.QDoubleSpinBox()
        self.sp_time.setRange(1, 3600)
        self.sp_time.setValue(30)
        self.sp_time.setSuffix(" s live")
        self.b_acq = QtWidgets.QPushButton("Acquire")
        self.b_acq.setEnabled(acquire is not None)
        self.b_acq.clicked.connect(self._acquire)
        self.cb_log = QtWidgets.QCheckBox("Log")
        self.cb_log.toggled.connect(lambda on: self.plot.setLogMode(y=on))
        row.addWidget(self.l_src, 1)
        row.addWidget(self.cb_log)
        row.addWidget(self.sp_time)
        row.addWidget(self.b_acq)
        sl.addLayout(row)
        self.plot = pg.PlotWidget(background="#1e1f22")
        self.plot.setLabel("bottom", "Channel")
        self.plot.setLabel("left", "Counts")
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#ffd54f", width=1.2), stepMode="center")
        self.mark_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#4fc3f7", width=2,
                                                                                 style=QtCore.Qt.DashLine))
        self.mark_line.setVisible(False)
        self.plot.addItem(self.mark_line, ignoreBounds=True)
        self._pair_items: list = []
        self.plot.scene().sigMouseClicked.connect(self._plot_click)
        sl.addWidget(self.plot, 1)
        self.l_mark = QtWidgets.QLabel("no peak marked")
        sl.addWidget(self.l_mark)

        # --- periodic table + pairs
        pg_ = QtWidgets.QGroupBox("Element of the marked peak")
        pl = QtWidgets.QVBoxLayout(pg_)
        self.ptable = PeriodicTableWidget(checkable=False)
        self.ptable.elementClicked.connect(self.add_element)
        pl.addWidget(self.ptable)
        pl.addStretch(1)
        tg = QtWidgets.QGroupBox("Calibration pairs")
        tl = QtWidgets.QVBoxLayout(tg)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        tl.addWidget(self.table)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Remove selected")
        b.clicked.connect(self.remove_selected)
        r.addWidget(b)
        b = QtWidgets.QPushButton("Remove all")
        b.clicked.connect(self.remove_all)
        r.addWidget(b)
        r.addStretch(1)
        tl.addLayout(r)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(pg_)
        bottom.addWidget(tg, 1)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close)
        self.b_apply = bb.button(QtWidgets.QDialogButtonBox.Apply)
        self.b_apply.clicked.connect(self.apply)
        bb.rejected.connect(self.reject)
        self.l_msg = QtWidgets.QLabel("1. take a spectrum of the calibration sample   2. click a peak   "
                                      "3. click its element   (one pair: slope; two or more: offset and slope)")
        self.l_msg.setStyleSheet("color:#9aa")
        self.l_msg.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(g)
        lay.addWidget(sg, 1)
        lay.addLayout(bottom)
        lay.addWidget(self.l_msg)
        lay.addWidget(bb)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(200)
        self._set_spins(current)
        self._show_applied()
        self._show_spectrum(mark_strongest=True)

    # ------------------------------------------------------------ calibration
    @property
    def calibration(self) -> EnergyCalibration:
        """The calibration shown (not applied yet)."""
        return EnergyCalibration(self.sp_offset.value() / 1e3, self.sp_slope.value() / 1e3)

    def _set_spins(self, cal: EnergyCalibration) -> None:
        self._syncing = True
        try:
            self.sp_offset.setValue(cal.offset_kev * 1e3)
            self.sp_slope.setValue(cal.gain_kev * 1e3)
        finally:
            self._syncing = False
        self._refresh()

    def _typed(self, *_):
        if not self._syncing:
            self._refresh()

    def _show_applied(self):
        c = self.applied
        self.l_applied.setText(f"in use: {c.offset_kev * 1e3:+.2f} eV + {c.gain_kev * 1e3:.4f} eV/ch")

    def points(self) -> list:
        """The calibration pairs as dicts (element, label, energy_kev, channel, source)."""
        return [p.to_dict() for p in self.pairs]

    def _pairs_changed(self):
        self.pointsChanged.emit(self.points())

    def clear(self):
        self.pairs.clear()
        self._pairs_changed()
        self._set_spins(EnergyCalibration())
        self._msg(f"cleared: offset 0, {DEFAULT_GAIN_KEV * 1e3:g} eV per channel")

    def refit(self) -> None:
        """Offset and slope from the pairs (one pair: slope only, through the current offset)."""
        if not self.pairs:
            self._refresh()
            return
        try:
            cal = EnergyCalibration.fit([p.channel for p in self.pairs], [p.line.energy_kev for p in self.pairs],
                                        offset_kev=self.sp_offset.value() / 1e3)
        except ValueError as e:
            self._msg(str(e), error=True)
            self._refresh()
            return
        self._set_spins(cal)

    def apply(self):
        cal = self.calibration
        self.applied = cal
        self._show_applied()
        self.calibrationApplied.emit(cal)
        self._msg("calibration applied")

    # ------------------------------------------------------------ pairs
    def add_element(self, element: str):
        """Pair the marked peak with *element*'s line nearest to it (by the shown calibration)."""
        if self.spectrum is None or self.marker is None:
            self._msg("take a spectrum and click a peak first", error=True)
            return
        peaks = element_peaks(element)
        if not peaks:
            self._msg(f"{element}: no lines above 0.2 keV", error=True)
            return
        e_mark = self.calibration.energy(self.marker)
        line = min(peaks, key=lambda p: abs(p.energy_kev - e_mark))
        src = _source(self.spectrum)
        # re-assigning a peak of this spectrum replaces its pair
        self.pairs = [p for p in self.pairs if not (p.source == src and abs(p.channel - self.marker) <= 1.0)]
        self.pairs.append(_Pair(element, line, float(self.marker), src))
        self.pairs.sort(key=lambda p: p.line.energy_kev)
        self._pairs_changed()
        self._msg(f"{line.label} ({line.energy_kev:.3f} keV) ↔ channel {self.marker:.2f}")
        self.refit()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.pairs):
                del self.pairs[r]
        self._pairs_changed()
        self.refit()

    def remove_all(self):
        self.pairs.clear()
        self._pairs_changed()
        self._refresh()

    def _line_changed(self, row: int, combo: QtWidgets.QComboBox):
        if 0 <= row < len(self.pairs):
            self.pairs[row].line = combo.currentData()
            self._pairs_changed()
            self.refit()

    # ------------------------------------------------------------ spectrum
    def _acquire(self):
        if self.acquire is None or self._future is not None:
            return
        try:
            self._future = self.acquire(float(self.sp_time.value()))
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._msg(f"acquire: {e}", error=True)
            return
        self.b_acq.setEnabled(False)
        self._msg("acquiring …")

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
                self._msg(f"acquire: {e}", error=True)
                return
            self._show_spectrum(mark_strongest=True)  # the pairs so far are kept
            self._msg("spectrum acquired: click a peak, then its element (earlier pairs are kept)")

    def _show_spectrum(self, mark_strongest: bool = False):
        sp = self.spectrum
        if sp is None:
            self.l_src.setText("no spectrum – acquire one on the calibration sample")
            self.curve.setData([], [])
            self._refresh()
            return
        y = np.asarray(sp.counts, dtype=float)
        x = np.arange(y.size + 1) - 0.5
        self.curve.setData(x, y)
        self.l_src.setText(f"{sp.title or 'spectrum'}: {y.size} channels, {int(y.sum()):,} counts, "
                           f"live {sp.live_time:.1f} s")
        if mark_strongest:
            self.marker = None
            try:
                pk = find_peaks(sp.with_calibration(self.calibration), min_significance=5.0)
            except Exception:  # noqa: BLE001 - empty or odd spectrum
                pk = []
            if pk:
                self.marker = float(max(pk, key=lambda p: p.height).channel)
        self._refresh()

    def _plot_click(self, ev):
        if self.spectrum is None or ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.mark_peak(vb.mapSceneToView(ev.scenePos()).x())

    def mark_peak(self, channel: float) -> float | None:
        """Put the marker on the peak nearest *channel* (centroid of the local maximum)."""
        sp = self.spectrum
        if sp is None:
            return None
        y = np.asarray(sp.counts, dtype=float)
        n = y.size
        cal = self.calibration
        k0 = int(np.clip(round(channel), 0, n - 1))
        # expected peak width: about 130 eV FWHM at Mn Kα, at least 1.5 channels
        sigma = max(1.5, 0.13 / 2.355 / cal.gain_kev)
        w = int(max(3, round(2 * sigma)))
        lo, hi = max(0, k0 - w), min(n, k0 + w + 1)
        k = lo + int(np.argmax(y[lo:hi]))
        try:
            bg = snip_background(y, cal)
        except Exception:  # noqa: BLE001
            bg = None
        p = refine_peak(y, float(k), sigma, bg)
        self.marker = float(p.channel) if p is not None else float(k)
        self._refresh()
        return self.marker

    # ------------------------------------------------------------ display
    def _refresh(self):
        cal = self.calibration
        # marker
        if self.marker is not None and self.spectrum is not None:
            self.mark_line.setPos(self.marker)
            self.mark_line.setVisible(True)
            self.l_mark.setText(f"marked peak: channel {self.marker:.2f} = {cal.energy(self.marker):.3f} keV  "
                                "– now click its element")
        else:
            self.mark_line.setVisible(False)
            self.l_mark.setText("no peak marked")
        # pair markers in the plot
        for it in self._pair_items:
            self.plot.removeItem(it)
        self._pair_items = []
        src = _source(self.spectrum)
        for p in self.pairs:
            here = p.source == src
            ln = pg.InfiniteLine(p.channel, angle=90,
                                 pen=pg.mkPen("#ff6b6b" if here else "#8a6b6b", width=1,
                                              style=QtCore.Qt.SolidLine if here else QtCore.Qt.DotLine),
                                 label=p.line.label, labelOpts={"position": 0.92 if here else 0.8,
                                                                "color": "#ff9b9b" if here else "#a08080"})
            self.plot.addItem(ln, ignoreBounds=True)
            self._pair_items.append(ln)
        # table
        self.table.setRowCount(len(self.pairs))
        for r, p in enumerate(self.pairs):
            combo = QtWidgets.QComboBox()
            for pl in element_peaks(p.element):
                combo.addItem(f"{pl.label}  {pl.energy_kev:.3f} keV", pl)
            labels = [combo.itemData(i).label for i in range(combo.count())]
            if p.line.label in labels:
                combo.setCurrentIndex(labels.index(p.line.label))
            else:  # a line no longer offered: keep it as it is
                combo.insertItem(0, f"{p.line.label}  {p.line.energy_kev:.3f} keV", p.line)
                combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(lambda _i, r=r, c=combo: self._line_changed(r, c))
            self.table.setCellWidget(r, 0, combo)
            res = (cal.energy(p.channel) - p.line.energy_kev) * 1e3
            for c, txt in ((1, f"{p.line.energy_kev:.4f}"), (2, f"{p.channel:.2f}"), (3, f"{res:+.1f}"),
                           (4, p.source)):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _msg(self, text: str, error: bool = False):
        self.l_msg.setText(text)
        self.l_msg.setStyleSheet("color:#f77" if error else "color:#9aa")
