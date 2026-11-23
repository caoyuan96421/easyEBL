"""The EDS main window – an Oxford-AZtec-like front end for :class:`EDSController`.

The window only *displays* and *commands*: acquisitions run in the
controller's job threads, analysis (AutoID, fitting, quantification, maps) in
a worker thread; results are picked up by timers.  Nothing here touches the
DAQ or the pulse processor directly.

Layout: electron image with acquisition sites (left) · spectrum with element
markers and fit (centre) · maps / line profile (bottom) ·
control tabs Acquire / Map / Composition (right) with the live MCA panel
(delta-mode spectrum, ICR, OCR, dead time) always visible below them.
"""
from __future__ import annotations

import concurrent.futures
import csv
import os
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ... import xray
from .. import analysis as A
from ..controller import EDSController
from ..settings import EDSSettings
from ..spectrum import ElectronImage, Spectrum, SpectrumMap, load_map, load_spectrum
from .calibration import CalibrationDialog
from .processor import ProcessorDialog
from .views import SITE_COLORS, ElectronImageView, MapTile, PeriodicTable, SpectrumView

__all__ = ["EDSWindow"]

MAP_COLORS = [(255, 90, 90), (90, 200, 255), (110, 230, 110), (255, 210, 80), (220, 120, 255), (255, 150, 60),
              (80, 255, 220), (255, 120, 200), (200, 200, 200)]


class _Site:
    def __init__(self, sid: int, kind: str, params: tuple):
        self.id, self.kind, self.params = sid, kind, tuple(float(p) for p in params)
        self.spectrum: Spectrum | None = None
        self.text = ""  # fixed description (map pixel, file …)
        self.color = SITE_COLORS[(sid - 1) % len(SITE_COLORS)]

    @property
    def name(self) -> str:
        return f"Spectrum {self.id}"

    def describe(self, field_um: float = 1.0) -> str:
        """*params* are fractions of the deflection field; shown in µm at *field_um*."""
        if self.text:
            return f"{self.name} – {self.text}"
        p = [v * field_um for v in self.params]
        if self.kind == "point":
            return f"{self.name} – point ({p[0]:.2f}, {p[1]:.2f}) µm"
        if self.kind in ("rect", "ellipse"):
            return f"{self.name} – {self.kind} {p[2]:.2f}×{p[3]:.2f} µm"
        if self.kind == "field":
            return f"{self.name} – whole image"
        return f"{self.name} – SEM spot"


class EDSWindow(QtWidgets.QMainWindow):
    TICK_MS = 100

    def __init__(self, controller: EDSController, settings: EDSSettings | None = None, sem_engine=None,
                 simulated: bool = False, owns_controller: bool = True, microscope=None, microscope_bundle=None):
        super().__init__()
        self.ctrl = controller
        self.settings = settings or EDSSettings()
        self.sem = sem_engine
        # MicroscopeControl: the SEM's accelerating voltage becomes the nominal beam energy; a bundle
        # (easyebl.microscope.bundle) adds the microscope window and the stage readback for metadata
        self.bundle = microscope_bundle
        if microscope is None and microscope_bundle is not None and microscope_bundle.connected:
            microscope = microscope_bundle.sem
        self.microscope = microscope
        self._mic_kv = None
        self.simulated = simulated
        self.owns_controller = owns_controller
        self._sub = controller.events.subscribe(maxlen=2000)
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="easyebl-eds-analysis")
        self._analysis = None  # (future, callback)
        self._job = None
        self._sites: list[_Site] = []
        self._site_ids = iter(range(1, 10 ** 9))
        self._current: _Site | None = None
        self.elements: list = []
        self.image: ElectronImage | None = None
        self.map: SpectrumMap | None = None
        self.map_region = None
        self.line = None
        self._fit = None
        self._quant = None
        self._maps: list = []
        self._map_live_t = 0.0
        # element maps: computed in their own worker, during a map scan too; any change of the
        # inputs (elements, kind, binning, oxides, calibration) or new pixels triggers a refresh
        self._maps_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="easyebl-eds-maps")
        self._maps_fut = None  # (future, inputs generation, map)
        self._maps_gen = 0  # incremented on every input change
        self._maps_done_gen = -1  # generation of the maps shown
        self._maps_px = -1  # scanned pixels of the maps shown
        self._maps_t = 0.0  # when the last computation started
        self._maps_dt = 0.0  # how long it took
        self._map_running = None  # the SpectrumMap of a running map scan
        self._live_prev = None  # last cumulative partial spectrum of the running job (for the live delta)
        self._live_px = None  # map pixels already counted into the live delta
        self._live_px_t = 0.0
        self._calib_dialog = None
        self._apply_settings_to_controller()

        # ------------------------------------------------ central area
        self.imgview = ElectronImageView()
        self.imgview.siteRequested.connect(self._site_requested)
        self.imgview.pointerMoved.connect(
            lambda x, y: self.l_ptr.setText(f"x {x * self._field():.2f} µm  y {y * self._field():.2f} µm"))
        self.specview = SpectrumView()
        self.specview.cursorMoved.connect(self._cursor)
        self.site_list = QtWidgets.QListWidget()
        self.site_list.currentRowChanged.connect(self._select_site)
        self.site_list.setMaximumHeight(130)

        tools = QtWidgets.QToolBar()
        tools.setIconSize(QtCore.QSize(16, 16))
        self._tool_group = QtGui.QActionGroup(self)
        for key, label, tip in [("pan", "Pan", "drag / wheel: move and zoom the image"),
                                ("point", "Point", "click: spectrum at a point"),
                                ("rect", "Rectangle", "drag: spectrum averaged over a rectangle"),
                                ("ellipse", "Ellipse", "drag: spectrum averaged over an ellipse"),
                                ("line", "Line", "drag: line for a line scan"),
                                ("map", "Map area", "drag: region for maps (default: the whole image)")]:
            a = tools.addAction(label)
            a.setCheckable(True)
            a.setToolTip(tip)
            a.setData(key)
            self._tool_group.addAction(a)
            if key == "point":
                a.setChecked(True)
        self._tool_group.triggered.connect(lambda a: setattr(self.imgview, "tool", a.data()))
        self.b_clear = QtWidgets.QPushButton("Clear all")
        f_ = self.b_clear.font()
        f_.setBold(True)
        f_.setPointSizeF(f_.pointSizeF() * 1.4)
        self.b_clear.setFont(f_)
        self.b_clear.setToolTip("remove all sites and the line; the map region goes back to the whole image")
        self.b_clear.clicked.connect(self.clear_all)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QtWidgets.QLabel("<b>Electron image</b>"))
        ll.addWidget(tools)
        ll.addWidget(self.b_clear)
        ll.addWidget(self._image_controls())
        ll.addWidget(self.imgview, 1)
        ll.addWidget(self.site_list)

        # bottom tabs: maps / line profile (the quantification is in the Composition tab)
        self.maps_area = QtWidgets.QScrollArea()
        self.maps_area.setWidgetResizable(True)
        self.maps_holder = QtWidgets.QWidget()
        self.maps_grid = QtWidgets.QGridLayout(self.maps_holder)
        self.maps_area.setWidget(self.maps_holder)
        self.profile = pg.PlotWidget(background="#1e1f22")
        self.profile.setLabel("bottom", "Distance", units="µm")
        self.profile.setLabel("left", "Counts")
        self.profile.addLegend()
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(self.maps_area, "Maps")
        self.bottom.addTab(self.profile, "Line profile")

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.specview)
        right.addWidget(self.bottom)
        right.setSizes([520, 300])
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([520, 900])
        self.setCentralWidget(split)

        # ------------------------------------------------ controls dock: tabs + live MCA panel
        self.controls = QtWidgets.QTabWidget()
        self.controls.addTab(self._acquire_tab(), "Acquire")
        self.controls.addTab(self._map_tab(), "Map")
        self.controls.addTab(self._composition_tab(), "Composition")
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.controls, 1)
        sl.addWidget(self._live_panel())
        dock = QtWidgets.QDockWidget("EDS")
        dock.setObjectName("eds_controls")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setWidget(side)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        ldock = QtWidgets.QDockWidget("Messages")
        ldock.setObjectName("eds_messages")
        ldock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, ldock)
        ldock.setMaximumHeight(120)

        self._make_menus()
        self._make_status()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.TICK_MS)
        cap = controller.handle.capabilities()
        self.setWindowTitle(f"easyEBL EDS – {controller.mca.name} + {cap.name}")
        self.notify(f"Pulse processor: {controller.mca.name} ({controller.mca.channels} channels); "
                    f"DAQ: {cap.name}; beam {controller.beam_kv:g} kV")
        self._emin_pending = True
        self.apply_min_energy()
        if self.settings.idle_monitor:
            self.ctrl.start_idle_monitor(self.settings.idle_update_s)
        self._update_status()

    # ================================================================ settings
    def _apply_settings_to_controller(self):
        s = self.settings
        self.ctrl.set_energy_calibration(s.energy_calibration)
        self.ctrl.beam_kv = float(s.beam_kv)
        self.ctrl.takeoff_deg = float(s.takeoff_deg)

    def _save_settings(self):
        try:
            self.settings.save()
        except OSError as e:
            self.notify(f"could not save settings: {e}")

    @property
    def detector(self):
        return self.settings.detector

    # ================================================================ panels
    def _image_controls(self):
        """Electron image acquisition – one row directly above the image."""
        s = self.settings
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        self.im_res = QtWidgets.QComboBox()
        for nx, ny in [(256, 192), (512, 384), (768, 576), (1024, 768), (256, 256), (512, 512)]:
            self.im_res.addItem(f"{nx} × {ny}", f"{nx}x{ny}")
        i = self.im_res.findData(f"{s.image_nx}x{s.image_ny}")
        self.im_res.setCurrentIndex(max(0, i))
        self.im_res.setToolTip("resolution")
        self.im_dwell = QtWidgets.QDoubleSpinBox()
        self.im_dwell.setRange(0.5, 1000)
        self.im_dwell.setValue(s.image_dwell_us)
        self.im_dwell.setSuffix(" µs")
        self.im_dwell.setToolTip("dwell per pixel")
        self.b_image = QtWidgets.QPushButton("Acquire image")
        self.b_image.clicked.connect(self.acquire_image)
        self.b_semimg = QtWidgets.QPushButton("Use SEM image")
        self.b_semimg.setEnabled(self.sem is not None)
        self.b_semimg.setToolTip("take the latest frame of the SEM window (same field calibration)")
        self.b_semimg.clicked.connect(self.use_sem_image)
        self.b_image.setText("Acquire")
        self.b_image.setToolTip("acquire an electron image with the DAQ")
        self.b_semimg.setText("From SEM")
        for wd in (self.im_res, self.im_dwell, self.b_image, self.b_semimg):
            wd.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            wd.setMinimumWidth(60)
        grid.addWidget(self.im_res, 0, 0)
        grid.addWidget(self.im_dwell, 0, 1)
        grid.addWidget(self.b_image, 1, 0)
        grid.addWidget(self.b_semimg, 1, 1)
        return w

    def _acquire_tab(self):
        s = self.settings
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Spectrum")
        f = QtWidgets.QFormLayout(g)
        self.sp_time = QtWidgets.QDoubleSpinBox()
        self.sp_time.setRange(0.05, 36000)
        self.sp_time.setDecimals(2)
        self.sp_time.setValue(s.spectrum_time_s)
        self.sp_time.setSuffix(" s")
        self.sp_live = QtWidgets.QCheckBox("live time")
        self.sp_live.setChecked(s.live_time_mode)
        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(self.sp_time)
        tr.addWidget(self.sp_live)
        f.addRow("Time", tr)
        self.sp_emin = QtWidgets.QDoubleSpinBox()
        self.sp_emin.setRange(0.02, 5.0)
        self.sp_emin.setDecimals(3)
        self.sp_emin.setSingleStep(0.01)
        self.sp_emin.setValue(s.min_energy_kev)
        self.sp_emin.setSuffix(" keV")
        self.sp_emin.setToolTip("pulses below this energy are rejected (DP5 slow threshold) – cuts off the "
                                "noise peak at 0 keV.  Default just below C Kα (0.277 keV).")
        self._emin_timer = QtCore.QTimer(self)
        self._emin_timer.setSingleShot(True)
        self._emin_timer.setInterval(400)
        self._emin_timer.timeout.connect(self.apply_min_energy)
        self.sp_emin.valueChanged.connect(lambda *_: self._emin_timer.start())
        f.addRow("Min. energy", self.sp_emin)
        self.sp_speed = QtWidgets.QComboBox()
        self.sp_speed.setToolTip("pulse-processor shaping: 1 = fastest (high count rates, wider peaks) … "
                                 "5 = best resolution (low count rates).  Written to the processor at once.")
        self._speed_pending = None
        self._fill_speeds()
        self.sp_speed.activated.connect(self._speed_chosen)
        f.addRow("Processor speed", self.sp_speed)
        hint = QtWidgets.QLabel("Draw a point or area on the image (or select a site), then Acquire. "
                                "With no site selected the whole image is measured.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa")
        f.addRow(hint)
        row = QtWidgets.QHBoxLayout()
        self.b_acquire = QtWidgets.QPushButton("Acquire")
        self.b_acquire.setStyleSheet("QPushButton{background:#2e7d32;color:white;font-weight:bold;padding:6px}")
        self.b_acquire.setToolTip("acquire a spectrum at the selected site (none selected: the whole image)")
        self.b_acquire.clicked.connect(self.acquire)
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_stop.setStyleSheet("QPushButton{background:#8a2c2c;color:white;font-weight:bold;padding:6px}")
        self.b_stop.clicked.connect(self.stop)
        row.addWidget(self.b_acquire, 2)
        row.addWidget(self.b_stop, 1)
        f.addRow(row)
        self.l_e0 = QtWidgets.QLabel("–")
        self.l_e0.setToolTip("landing energy from the end of the continuum (Duane–Hunt limit) of the "
                             "current spectrum; the nominal value (Advanced) is used when it cannot be measured")
        f.addRow("Beam energy", self.l_e0)

        # advanced detector / analysis parameters (collapsed)
        self.c_kv = QtWidgets.QDoubleSpinBox()
        self.c_kv.setRange(1, 40)
        self.c_kv.setDecimals(1)
        self.c_kv.setValue(s.beam_kv)
        self.c_kv.setSuffix(" kV")
        self.c_kv.setToolTip("used only when the spectrum does not show the end of the continuum")
        self.c_kv.valueChanged.connect(self._conditions_changed)
        self.b_kv_sync = QtWidgets.QPushButton("Sync from SEM")
        self.b_kv_sync.setToolTip("read the accelerating voltage from the SEM")
        self.b_kv_sync.setEnabled(self.microscope is not None)
        self.b_kv_sync.clicked.connect(self.sync_beam_kv)
        kv_row = QtWidgets.QHBoxLayout()
        kv_row.addWidget(self.c_kv, 1)
        kv_row.addWidget(self.b_kv_sync)
        self.b_adv = QtWidgets.QToolButton()
        self.b_adv.setText("Advanced")
        self.b_adv.setCheckable(True)
        self.b_adv.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.b_adv.setArrowType(QtCore.Qt.RightArrow)
        self.b_adv.setStyleSheet("QToolButton{border:none}")
        f.addRow(self.b_adv)
        self.adv = QtWidgets.QWidget()
        af = QtWidgets.QFormLayout(self.adv)
        af.setContentsMargins(12, 0, 0, 0)
        self.c_to = QtWidgets.QDoubleSpinBox()
        self.c_to.setRange(5, 90)
        self.c_to.setValue(s.takeoff_deg)
        self.c_to.setSuffix(" °")
        self.c_to.valueChanged.connect(self._conditions_changed)
        self.c_fwhm = QtWidgets.QDoubleSpinBox()
        self.c_fwhm.setRange(100, 300)
        self.c_fwhm.setValue(s.fwhm_mn_ev)
        self.c_fwhm.setSuffix(" eV @ Mn Kα")
        self.c_fwhm.valueChanged.connect(self._conditions_changed)
        self.c_win = QtWidgets.QComboBox()
        self.c_win.addItems(["AP3.3", "Be8", "none"])
        self.c_win.setCurrentText(s.window)
        self.c_win.currentTextChanged.connect(self._conditions_changed)
        self.c_idle = QtWidgets.QDoubleSpinBox()
        self.c_idle.setRange(0.1, 60)
        self.c_idle.setDecimals(1)
        self.c_idle.setSingleStep(0.5)
        self.c_idle.setValue(s.idle_update_s)
        self.c_idle.setSuffix(" s")
        self.c_idle.setToolTip("read-out interval of the live MCA (delta mode) between acquisitions")
        self.c_idle.valueChanged.connect(self._idle_interval_changed)
        af.addRow("Nominal beam energy", kv_row)
        af.addRow("Take-off", self.c_to)
        af.addRow("Resolution", self.c_fwhm)
        af.addRow("Window", self.c_win)
        af.addRow("Live MCA update", self.c_idle)
        self.adv.setVisible(False)
        self.b_adv.toggled.connect(self._toggle_advanced)
        f.addRow(self.adv)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Display")
        f = QtWidgets.QFormLayout(g)
        self.d_log = QtWidgets.QCheckBox("log scale")
        self.d_log.toggled.connect(self.specview.set_log)
        f.addRow(self.d_log)
        self.d_overlay = QtWidgets.QCheckBox("overlay all site spectra")
        self.d_overlay.toggled.connect(lambda *_: self._show_current())
        f.addRow(self.d_overlay)
        self.d_norm = QtWidgets.QCheckBox("normalise overlays (counts/s)")
        self.d_norm.toggled.connect(lambda *_: self._show_current())
        f.addRow(self.d_norm)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _toggle_advanced(self, on: bool):
        self.adv.setVisible(on)
        self.b_adv.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)

    def _live_panel(self):
        """Live MCA (delta mode between jobs), rates and dead time – always visible."""
        g = QtWidgets.QGroupBox("Live MCA")
        v = QtWidgets.QVBoxLayout(g)
        v.setContentsMargins(4, 4, 4, 4)
        self.live_plot = pg.PlotWidget(background="#1e1f22")
        self.live_plot.setMinimumHeight(130)
        self.live_plot.setMaximumHeight(170)
        self.live_plot.showGrid(x=True, y=True, alpha=0.15)
        self.live_plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        self.live_plot.setLabel("bottom", "keV")
        self.live_plot.getPlotItem().getAxis("left").setWidth(36)
        self.live_plot.setMouseEnabled(x=True, y=False)
        self.live_plot.setXRange(0, 10, padding=0)
        self._live_curve = self.live_plot.plot([], [], stepMode="center", fillLevel=0,
                                               brush=pg.mkBrush(79, 195, 247, 70), pen=pg.mkPen("#4fc3f7"))
        v.addWidget(self.live_plot)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        self.l_icr = QtWidgets.QLabel("ICR –")
        self.l_ocr = QtWidgets.QLabel("OCR –")
        self.l_live_t = QtWidgets.QLabel("")
        self.l_live_t.setStyleSheet("color:#9aa")
        for lb in (self.l_icr, self.l_ocr):
            lb.setStyleSheet("font-weight:bold")
        grid.addWidget(self.l_icr, 0, 0)
        grid.addWidget(self.l_ocr, 0, 1)
        grid.addWidget(self.l_live_t, 0, 2)
        self.l_dead = QtWidgets.QProgressBar()
        self.l_dead.setRange(0, 100)
        self.l_dead.setFormat("dead time %p %")
        self.l_dead.setAlignment(QtCore.Qt.AlignCenter)
        self.l_dead.setToolTip("green zone 5–50 %: good count rate; red above 50 %: too much current")
        self._dead_zone = None
        self._style_dead(0)
        grid.addWidget(self.l_dead, 1, 0, 1, 3)
        v.addLayout(grid)
        r = QtWidgets.QHBoxLayout()
        self.cb_live = QtWidgets.QCheckBox("Live between acquisitions")
        self.cb_live.setChecked(self.settings.idle_monitor)
        self.cb_live.toggled.connect(self._live_toggled)
        r.addWidget(self.cb_live, 1)
        b = QtWidgets.QPushButton("Processor…")
        b.setToolTip("pulse processor settings")
        b.clicked.connect(self.open_processor)
        r.addWidget(b)
        v.addLayout(r)
        return g

    def _live_toggled(self, on: bool):
        self.settings.idle_monitor = bool(on)
        if on:
            self.ctrl.start_idle_monitor(self.settings.idle_update_s)
        else:
            self.ctrl.stop_idle_monitor()

    def _idle_interval_changed(self, v: float):
        self.settings.idle_update_s = float(v)
        self.ctrl.set_idle_interval(float(v))

    #: dead-time bar zones (%): below the green zone the count rate is low; above it, too high
    DEAD_GREEN = (5, 50)

    def _style_dead(self, pct: int) -> None:
        """Zones behind the bar (grey < 5 %, green 5–50 %, red > 50 %); the fill takes the colour
        of the zone the dead time is in."""
        g0, g1 = (v / 100 for v in self.DEAD_GREEN)
        zone = 0 if pct < self.DEAD_GREEN[0] else (1 if pct <= self.DEAD_GREEN[1] else 2)
        if zone == self._dead_zone:
            return
        self._dead_zone = zone
        fill = ("rgba(200,200,200,150)", "rgba(80,200,90,210)", "rgba(240,70,60,220)")[zone]
        self.l_dead.setStyleSheet(
            "QProgressBar{border:1px solid #555;border-radius:3px;color:white;font-weight:bold;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #2a2c30,stop:{g0:.4f} #2a2c30,stop:{g0 + 1e-4:.4f} #1f4d27,stop:{g1:.4f} #1f4d27,"
            f"stop:{g1 + 1e-4:.4f} #5a1f1f,stop:1 #5a1f1f)}}"
            f"QProgressBar::chunk{{background:{fill};border-radius:2px}}")

    def _show_rates(self, sp: Spectrum):
        if sp is None or sp.real_time <= 0:
            return
        self.l_icr.setText(f"ICR {sp.icr / 1e3:.2f} kcps")
        self.l_ocr.setText(f"OCR {sp.ocr / 1e3:.2f} kcps")
        pct = int(round(100 * min(max(sp.dead_fraction, 0.0), 1.0)))
        self.l_dead.setValue(pct)
        self._style_dead(pct)

    def _live_from_partial(self, sp: Spectrum):
        """During a spectrum acquisition: the live panel shows what arrived since the previous
        read-out (current partial − previous partial), with its own ICR / OCR / dead time."""
        prev = self._live_prev
        self._live_prev = sp
        if prev is not None and len(prev.counts) == len(sp.counts) and sp.real_time > prev.real_time:
            sp = Spectrum(np.clip(sp.counts.astype(np.int64) - prev.counts, 0, None), sp.calibration,
                          sp.real_time - prev.real_time, max(sp.live_time - prev.live_time, 0.0),
                          max(sp.fast_count - prev.fast_count, 0), max(sp.slow_count - prev.slow_count, 0),
                          sp.beam_kv, sp.takeoff_deg)
        self._show_live(sp)

    #: how often a running map updates the live panel (s)
    LIVE_MAP_S = 0.5

    def _live_from_map(self, sm):
        """During a map / line scan: the live panel shows the pixels finished since its last
        update (summed), with their ICR / OCR / dead time."""
        now = time.monotonic()
        if now - self._live_px_t < self.LIVE_MAP_S:
            return
        done = np.asarray(sm.done, dtype=bool)
        prev = self._live_px if self._live_px is not None and self._live_px.shape == done.shape else             np.zeros_like(done)
        new = done & ~prev
        if not new.any():
            return
        self._live_px, self._live_px_t = done.copy(), now
        counts = np.asarray(sm.cube[new], dtype=np.int64).sum(axis=0)
        real = float(np.asarray(sm.real_time)[new].sum())
        if real <= 0:
            return
        sp = Spectrum(counts, sm.calibration, real, float(np.asarray(sm.live_time)[new].sum()),
                      int(np.asarray(sm.fast_count)[new].sum()), int(counts.sum()))
        self._show_live(sp)

    def _show_live(self, sp: Spectrum):
        cal = self.ctrl.energy_calibration
        n = len(sp.counts)
        edges = cal.energy(np.arange(n + 1) - 0.5)
        self._live_curve.setData(edges, sp.counts.astype(float))
        hi = max(float(sp.counts.max()), 1.0) if n else 1.0
        self.live_plot.setYRange(0, hi * 1.1, padding=0)
        self.live_plot.setXRange(0, float(edges[-1]), padding=0)
        self.l_live_t.setText(f"Δ {sp.real_time:.2f} s · {sp.total:,} cts")
        self._show_rates(sp)

    def _map_tab(self):
        s = self.settings
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Spectrum map (step scan)")
        f = QtWidgets.QFormLayout(g)
        self.m_nx = QtWidgets.QSpinBox()
        self.m_nx.setRange(1, 1024)
        self.m_nx.setValue(s.map_nx)
        self.m_ny = QtWidgets.QSpinBox()
        self.m_ny.setRange(1, 1024)
        self.m_ny.setValue(s.map_ny)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.m_nx)
        r.addWidget(QtWidgets.QLabel("×"))
        r.addWidget(self.m_ny)
        f.addRow("Pixels", r)
        self.m_dwell = QtWidgets.QDoubleSpinBox()
        self.m_dwell.setRange(1, 60000)
        self.m_dwell.setValue(s.map_dwell_ms)
        self.m_dwell.setSuffix(" ms")
        f.addRow("Dwell / pixel", self.m_dwell)
        self.m_region = QtWidgets.QLabel("whole image")
        f.addRow("Region", self.m_region)
        self.m_eta = QtWidgets.QLabel()
        f.addRow("Estimated", self.m_eta)
        for sb in (self.m_nx, self.m_ny, self.m_dwell):
            sb.valueChanged.connect(self._update_eta)
        self.b_map = QtWidgets.QPushButton("Start map")
        self.b_map.clicked.connect(self.start_map)
        self.b_map_stop = QtWidgets.QPushButton("Stop")
        self.b_map_stop.clicked.connect(self.stop)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.b_map)
        r.addWidget(self.b_map_stop)
        f.addRow(r)
        self.m_progress = QtWidgets.QProgressBar()
        f.addRow(self.m_progress)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Line scan")
        f = QtWidgets.QFormLayout(g)
        self.l_n = QtWidgets.QSpinBox()
        self.l_n.setRange(2, 4096)
        self.l_n.setValue(100)
        self.l_dwell = QtWidgets.QDoubleSpinBox()
        self.l_dwell.setRange(1, 60000)
        self.l_dwell.setValue(100)
        self.l_dwell.setSuffix(" ms")
        self.l_line = QtWidgets.QLabel("draw a line on the image")
        f.addRow("Points", self.l_n)
        f.addRow("Dwell / point", self.l_dwell)
        f.addRow("Line", self.l_line)
        self.b_line = QtWidgets.QPushButton("Start line scan")
        self.b_line.clicked.connect(self.start_linescan)
        f.addRow(self.b_line)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Map display")
        f = QtWidgets.QFormLayout(g)
        self.m_kind = QtWidgets.QComboBox()
        self.m_kind.addItems(["Window (fast)", "Fitted (deconvolved)", "Quantitative (wt%)"])
        self.m_kind.currentIndexChanged.connect(lambda *_: self._maps_changed())
        f.addRow("Maps", self.m_kind)
        self.m_bin = QtWidgets.QSpinBox()
        self.m_bin.setRange(1, 16)
        self.m_bin.valueChanged.connect(lambda *_: self._maps_changed())
        f.addRow("Binning", self.m_bin)
        self.m_live = QtWidgets.QCheckBox("Update maps while scanning")
        self.m_live.setChecked(True)
        self.m_live.setToolTip("element maps are recomputed about once a second from the pixels scanned so far")
        f.addRow(self.m_live)
        self.b_maps = QtWidgets.QPushButton("Compute maps")
        self.b_maps.clicked.connect(self.compute_maps)
        f.addRow(self.b_maps)
        self.ph_k = QtWidgets.QSpinBox()
        self.ph_k.setRange(2, 12)
        self.ph_k.setValue(4)
        self.b_phase = QtWidgets.QPushButton("Phase map")
        self.b_phase.clicked.connect(self.compute_phases)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(QtWidgets.QLabel("phases"))
        r.addWidget(self.ph_k)
        r.addWidget(self.b_phase)
        f.addRow(r)
        self.m_overlay = QtWidgets.QComboBox()
        self.m_overlay.addItem("(no overlay)")
        self.m_overlay.currentIndexChanged.connect(self._overlay_changed)
        f.addRow("Overlay on image", self.m_overlay)
        self.b_mapsum = QtWidgets.QPushButton("Sum spectrum")
        self.b_mapsum.clicked.connect(self.show_map_sum)
        f.addRow(self.b_mapsum)
        lay.addWidget(g)
        lay.addStretch(1)
        self._update_eta()
        return w

    def _composition_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Elements")
        v = QtWidgets.QVBoxLayout(g)
        self.el_label = QtWidgets.QLabel("–")
        self.el_label.setWordWrap(True)
        self.el_label.setStyleSheet("font-size:16px;font-weight:bold")
        v.addWidget(self.el_label)
        note = QtWidgets.QLabel("Selected elements stay for every spectrum until cleared; "
                                "AutoID only adds to them.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa")
        v.addWidget(note)
        big = "QPushButton{padding:8px;font-weight:bold}"
        self.b_ptable = QtWidgets.QPushButton("Periodic table…")
        self.b_ptable.setStyleSheet(big)
        self.b_ptable.clicked.connect(self.pick_elements)
        self.b_autoid = QtWidgets.QPushButton("AutoID")
        self.b_autoid.setStyleSheet(big)
        self.b_autoid.setToolTip("identify elements in the current spectrum and add them to the selection")
        self.b_autoid.clicked.connect(lambda: self.analyse(autoid=True))
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.b_ptable)
        r.addWidget(self.b_autoid)
        v.addLayout(r)
        self.b_clear_el = QtWidgets.QPushButton("Clear elements")
        self.b_clear_el.clicked.connect(lambda: self.set_elements([]))
        v.addWidget(self.b_clear_el)
        self.sp_autoid = QtWidgets.QCheckBox("AutoID + quant when a spectrum is done")
        self.sp_autoid.setChecked(True)
        v.addWidget(self.sp_autoid)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("Quantification")
        f = QtWidgets.QFormLayout(g)
        self.q_oxide = QtWidgets.QCheckBox("oxygen by stoichiometry (oxides)")
        self.q_oxide.setChecked(self.settings.oxygen_by_stoichiometry)
        self.q_oxide.toggled.connect(lambda on: (setattr(self.settings, "oxygen_by_stoichiometry", on),
                                                 self._maps_changed()))
        f.addRow(self.q_oxide)
        b = QtWidgets.QPushButton("Fit && quantify")
        b.clicked.connect(lambda: self.analyse(autoid=False))
        f.addRow(b)
        self.q_fit = QtWidgets.QCheckBox("show fit")
        self.q_fit.setChecked(True)
        self.q_fit.toggled.connect(self._show_fit)
        f.addRow(self.q_fit)
        self.quant_table = QtWidgets.QTableWidget(0, 6)
        self.quant_table.setHorizontalHeaderLabels(["Element", "Line", "Wt%", "± Wt%", "At%", "Oxide wt%"])
        self.quant_table.verticalHeader().setVisible(False)
        self.quant_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        hh = self.quant_table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.quant_table.setMinimumHeight(180)
        f.addRow(self.quant_table)
        self.quant_note = QtWidgets.QLabel()
        self.quant_note.setWordWrap(True)
        self.quant_note.setStyleSheet("color:#aaa")
        f.addRow(self.quant_note)
        lay.addWidget(g, 1)
        return w

    def _make_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction("Open spectrum…", self.open_spectrum)
        m.addAction("Save spectrum…", self.save_spectrum, QtGui.QKeySequence.Save)
        m.addSeparator()
        m.addAction("Open map…", self.open_map)
        m.addAction("Save map…", self.save_map)
        m.addAction("Export maps (PNG)…", self.export_maps)
        m.addSeparator()
        m.addAction("Export quantification (CSV)…", self.export_quant)
        m.addAction("Save electron image…", self.save_image)
        m.addSeparator()
        m.addAction("Close", self.close)
        m = mb.addMenu("&Tools")
        m.addAction("Energy calibration…", self.open_calibration)
        m.addAction("Pulse processor…", self.open_processor)
        if self.bundle is not None and self.bundle.connected:
            m.addSeparator()
            m.addAction("Microscope…", self.bundle.show_window)

    def _make_status(self):
        sb = self.statusBar()
        self.l_state = QtWidgets.QLabel("Idle")
        self.l_cursor = QtWidgets.QLabel()
        self.l_ptr = QtWidgets.QLabel()
        self.p_job = QtWidgets.QProgressBar()
        self.p_job.setMaximumWidth(220)
        self.p_job.setRange(0, 1000)
        for w_ in (self.l_state, self.p_job, self.l_cursor):
            sb.addPermanentWidget(w_)
        sb.addPermanentWidget(self.l_ptr, 1)

    # ================================================================ helpers
    def notify(self, msg: str):
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    # Sites, the line and the map region are kept in fractions of the deflection field
    # (centre 0, full field ±0.5), so they stay where they were drawn relative to the
    # field of view when the magnification changes; µm = fraction × the present field.
    def _field(self) -> float:
        return float(self.ctrl.calibration.field_size_um)

    def _um(self, params) -> tuple:
        f = self._field()
        return tuple(float(p) * f for p in params)

    def _frac(self, region_um, field_um=None) -> tuple:
        f = float(field_um or self._field())
        return tuple(float(p) / f for p in region_um)

    def _frac_of(self, obj) -> tuple:
        """Region of an ElectronImage / SpectrumMap in field fractions (field at its acquisition)."""
        return self._frac(obj.region_um, (obj.metadata or {}).get("field_um"))

    def _image_region(self) -> tuple:
        """The SEM / EDS image in field fractions (the whole field without an image)."""
        if self.image is None:
            return (0.0, 0.0, 1.0, 1.0)
        return self._frac_of(self.image)

    def _show_image(self, im: ElectronImage) -> None:
        self.image = im
        self.imgview.set_image(im.data, self._image_region())

    def _clip(self, kind: str, params: tuple):
        """*params* limited to the image; None if nothing of it is inside."""
        cx, cy, w, h = self._image_region()
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        cl = lambda v, a, b: min(max(v, a), b)  # noqa: E731
        if kind == "point":
            x, y = params
            return params if (x0 <= x <= x1 and y0 <= y <= y1) else None
        if kind == "line":
            a, b, c, d = params
            out = (cl(a, x0, x1), cl(b, y0, y1), cl(c, x0, x1), cl(d, y0, y1))
            return out if out[:2] != out[2:] else None
        pcx, pcy, pw, ph = params
        l, r = max(pcx - pw / 2, x0), min(pcx + pw / 2, x1)
        b_, t = max(pcy - ph / 2, y0), min(pcy + ph / 2, y1)
        if r - l <= 0 or t - b_ <= 0:
            return None
        return ((l + r) / 2, (b_ + t) / 2, r - l, t - b_)

    def _line_text(self) -> str:
        if self.line is None:
            return "draw a line on the image"
        x0, y0, x1, y1 = self._um(self.line)
        return f"({x0:.1f}, {y0:.1f}) → ({x1:.1f}, {y1:.1f}) µm, {np.hypot(x1 - x0, y1 - y0):.2f} µm"

    def _region_text(self) -> str:
        if self.map_region is None:
            return "whole image"
        cx, cy, w, h = self._um(self.map_region)
        return f"{w:.2f} × {h:.2f} µm at ({cx:.1f}, {cy:.1f})"

    def _field_changed(self) -> None:
        """New magnification: same places in the field of view, new µm values."""
        for i, s in enumerate(self._sites):
            it = self.site_list.item(i)
            if it is not None:
                it.setText(s.describe(self._field()))
        self.l_line.setText(self._line_text())
        self.m_region.setText(self._region_text())

    def _cursor(self, e, c):
        near = xray.lines_near(e, tol=0.04, beam_kv=self._kv(), min_weight=0.3)[:3]
        self.l_cursor.setText(f"{e:.3f} keV  {c:.0f} cts  " + " ".join(ln.label for ln in near))

    def _busy(self) -> bool:
        if self.ctrl.busy:
            self.notify(f"busy: {self.ctrl.job.description}")
            return True
        return False

    def _start(self, fn, *a, **kw):
        if self._busy():
            return None
        try:
            job = fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.notify(f"cannot start: {e}")
            QtWidgets.QMessageBox.warning(self, "EDS", str(e))
            return None
        self._job = job
        return job

    def _update_eta(self):
        n = self.m_nx.value() * self.m_ny.value()
        per = self.m_dwell.value() / 1e3 + 0.02
        self.m_eta.setText(f"≈ {n * per / 60:.1f} min ({n} px)")

    # ================================================================ acquisitions
    def acquire_image(self):
        nx, ny = (int(v) for v in self.im_res.currentData().split("x"))
        self.settings.image_nx, self.settings.image_ny = nx, ny
        self.settings.image_dwell_us = self.im_dwell.value()
        self._start(self.ctrl.acquire_image, nx, ny, self.im_dwell.value() * 1e-6)

    def use_sem_image(self):
        if self.sem is None:
            return
        f = self.sem.frames.latest() if hasattr(self.sem, "frames") else None
        if f is None:
            self.notify("the SEM has no image yet")
            return
        cfg = f.config
        if abs(getattr(cfg, "rotation_rad", 0.0)) > 1e-6:
            self.notify("note: the SEM image is rotated – EDS sites use unrotated field coordinates")
        region = (cfg.center_um[0], cfg.center_um[1], cfg.width_um, cfg.height)
        cal = getattr(f, "calibration", None)
        field = float(cal.field_size_um) if cal is not None else self._field()
        self._show_image(ElectronImage(np.asarray(f.image, float), region, cfg.dwell_s,
                                       metadata={"source": "SEM", "field_um": field}))
        self._limit_map_region()
        self.notify(f"electron image from the SEM: {cfg.nx}×{cfg.ny}, {cfg.width_um:g} µm field")

    def _site_requested(self, kind: str, params: tuple):
        """A site drawn on the image (field fractions), limited to the image."""
        clipped = self._clip(kind, tuple(float(p) for p in params))
        if clipped is None:
            self.notify(f"{kind}: outside the image – ignored")
            return
        params = clipped
        if kind == "line":
            self.line = params
            self.l_line.setText(self._line_text())
            self.imgview.add_site("line", "line", params, "#ffffff", "line")
            return
        if kind == "map":
            self.map_region = params
            self.m_region.setText(self._region_text())
            self.imgview.add_site("mapregion", "map", params, "#ffffff", "map")
            return
        self._new_site(kind, params)

    def _limit_map_region(self) -> None:
        """Keep the map region inside the (new) image."""
        if self.map_region is None:
            return
        r = self._clip("map", self.map_region)
        if r != self.map_region:
            self.map_region = r
            if r is None:
                self.imgview.remove_site("mapregion")
                self.notify("the map region was outside the new image: maps cover the whole image")
            else:
                self.imgview.add_site("mapregion", "map", r, "#ffffff", "map")
            self.m_region.setText(self._region_text())

    def _new_site(self, kind: str, params: tuple):
        site = _Site(next(self._site_ids), kind, params)
        self._sites.append(site)
        self.site_list.addItem(site.describe(self._field()))
        self.site_list.item(self.site_list.count() - 1).setForeground(QtGui.QColor(site.color))
        if kind in ("point", "rect", "ellipse"):
            self.imgview.add_site(site.id, kind, params, site.color, str(site.id))
        self.site_list.setCurrentRow(self.site_list.count() - 1)
        return site

    def acquire(self):
        """Acquire button: a spectrum at the selected site, or – none selected – over the whole image."""
        site = self._current
        if site is None or site.kind == "spot":
            site = self._new_site("field", self._image_region())
        self.acquire_site(site)

    def acquire_site(self, site: _Site | None):
        if site is None:
            return
        t = self.sp_time.value()
        live = self.sp_live.isChecked()
        self.settings.spectrum_time_s, self.settings.live_time_mode = t, live
        kw = {"live_time": live, "title": site.name}
        if site.kind == "point":
            kw["at_um"] = self._um(site.params)
        elif site.kind in ("rect", "ellipse"):
            kw["region_um"], kw["shape"] = self._um(site.params), site.kind
        elif site.kind == "field":
            kw["region_um"], kw["shape"] = self._um(site.params), "rect"
        job = self._start(self.ctrl.acquire_spectrum, t, **kw)
        if job is not None:
            job._site = site

    def start_map(self):
        self._limit_map_region()
        region = self._um(self.map_region or self._image_region())
        s = self.settings
        s.map_nx, s.map_ny, s.map_dwell_ms = self.m_nx.value(), self.m_ny.value(), self.m_dwell.value()
        job = self._start(self.ctrl.acquire_map, s.map_nx, s.map_ny, s.map_dwell_ms / 1e3, region)
        if job is not None:
            self.bottom.setCurrentWidget(self.maps_area)
            self.m_progress.setValue(0)

    def start_linescan(self):
        if self.line is None:
            self.notify("draw a line on the electron image first (Line tool)")
            return
        x0, y0, x1, y1 = self._um(self.line)
        self._start(self.ctrl.acquire_linescan, (x0, y0), (x1, y1), self.l_n.value(), self.l_dwell.value() / 1e3)

    def stop(self):
        self.ctrl.cancel()

    # ================================================================ sites / spectra
    def clear_sites(self):
        self._sites.clear()
        self.site_list.clear()
        self.imgview.clear_sites()
        self._current = None
        self.specview.clear_spectra()

    def clear_all(self):
        """Clear all: sites, the line, and the map region (back to the whole image)."""
        self.clear_sites()  # also removes the line and map outlines from the image
        self.line = None
        self.l_line.setText(self._line_text())
        self.map_region = None
        self.m_region.setText(self._region_text())
        self.imgview.set_overlay(None)

    def _select_site(self, row: int):
        self._current = self._sites[row] if 0 <= row < len(self._sites) else None
        self._show_current()

    def _show_current(self):
        self.specview.clear_spectra()
        sites = self._sites if self.d_overlay.isChecked() else ([self._current] if self._current else [])
        cal = self.ctrl.energy_calibration
        for s in sites:
            if s is None or s.spectrum is None:
                continue
            sp = s.spectrum.with_calibration(cal)
            if self.d_norm.isChecked() and sp.live_time > 0:
                sp = Spectrum(sp.counts / sp.live_time, cal, sp.real_time, sp.live_time, beam_kv=sp.beam_kv)
            self.specview.set_spectrum(f"site{s.id}", sp, s.color, s.name, primary=(s is self._current))
        self.specview.set_markers(self.elements, self._kv())
        self._show_fit()
        self.specview.autoscale()
        self._show_e0()

    def _kv(self) -> float | None:
        """Beam energy of the current spectrum (None: no spectrum – every line is shown)."""
        sp = self._spectrum_of_current()
        return float(sp.beam_kv) if sp is not None else None

    def _show_e0(self):
        sp = self._spectrum_of_current()
        if sp is None:
            self.l_e0.setText("–")
            return
        src = sp.metadata.get("beam_kv_source", "")
        txt = {"spectrum": "from the spectrum", "nominal": "nominal (end of continuum not measurable)",
               "file": "from the file"}.get(src, src)
        self.l_e0.setText(f"{sp.beam_kv:.1f} kV  {txt}")

    def _assign_beam_kv(self, sp) -> None:
        """E0 of *sp* from its Duane–Hunt limit, else the nominal value (Advanced)."""
        est = A.estimate_beam_kv(sp.with_calibration(self.ctrl.energy_calibration), detector=self.detector)
        sp.metadata = dict(sp.metadata or {})
        if est is not None:
            sp.beam_kv = float(est)
            sp.metadata["beam_kv_source"] = "spectrum"
        else:
            sp.beam_kv = float(self.settings.beam_kv)
            sp.metadata["beam_kv_source"] = "nominal"

    def _show_fit(self, *_):
        if self.q_fit.isChecked() and self._fit is not None and self._fit[0] is self._spectrum_of_current():
            f = self._fit[1]
            self.specview.set_fit(f.model, f.background)
        else:
            self.specview.set_fit(None, None)

    def _spectrum_of_current(self):
        return self._current.spectrum if self._current else None

    def current_spectrum(self) -> Spectrum | None:
        sp = self._spectrum_of_current()
        return None if sp is None else sp.with_calibration(self.ctrl.energy_calibration)

    # ================================================================ analysis
    def set_elements(self, els):
        new = sorted(set(els), key=xray.z_of)
        changed = new != getattr(self, "elements", None)
        self.elements = new
        if changed:
            self._maps_changed()
        self.el_label.setText(" ".join(self.elements) if self.elements else "–")
        self.specview.set_markers(self.elements, self._kv())
        self.specview.autoscale()

    def pick_elements(self):
        d = PeriodicTable(self.elements, self)
        if d.exec() == QtWidgets.QDialog.Accepted:
            self.set_elements(d.selected())
            if self.current_spectrum() is not None and self.elements:
                self.analyse(autoid=False)

    def analyse(self, autoid: bool = True):
        sp = self.current_spectrum()
        if sp is None:
            self.notify("no spectrum selected")
            return
        if self._analysis is not None:
            return
        raw = self._spectrum_of_current()
        det, els, oxide = self.detector, list(self.elements), self.q_oxide.isChecked()

        def work():
            ids = A.identify_elements(sp, detector=det) if autoid else []
            added = [i.element for i in ids if i.element not in els]
            elements = els + added  # AutoID only adds; the user's selection is kept
            if not elements:
                return elements, added, None, None
            fit = A.fit_spectrum(sp, elements, detector=det)
            q = A.quantify(fit, sp.beam_kv, sp.takeoff_deg, det, oxygen_by_stoichiometry=oxide)
            return elements, added, fit, q

        def done(res):
            elements, added, fit, q = res
            # merge with the current selection (the user may have changed it while the analysis ran)
            self.set_elements(list(self.elements) + list(added))
            if autoid:
                self.notify("AutoID: " + (f"added {', '.join(added)}" if added else "nothing new"))
            self._fit = (raw, fit) if fit is not None else None
            self._quant = q
            self._show_quant(q)
            self._show_fit()
            if fit is not None:
                self.notify(f"{raw.title}: " + ", ".join(f"{r.element} {r.wt:.1f}" for r in q.rows) +
                            f" wt%  (χ²ᵣ {fit.chi2_reduced:.2f})")

        self._run_analysis(work, done, "analysing spectrum")

    def _run_analysis(self, work, done, what: str):
        if self._analysis is not None:
            self.notify("analysis busy")
            return
        self.l_state.setText(what + "…")
        self._analysis = (self._pool.submit(work), done, what)

    def _show_quant(self, q):
        t = self.quant_table
        t.setRowCount(0)
        if q is None:
            self.quant_note.setText("")
            return
        for r in q.rows:
            i = t.rowCount()
            t.insertRow(i)
            vals = [r.element, r.family, f"{r.wt:.2f}", "–" if not np.isfinite(r.wt_sigma) else f"{r.wt_sigma:.2f}",
                    f"{r.at:.2f}", f"{r.oxide} {r.oxide_wt:.2f}" if r.oxide else ("stoich." if r.by_stoichiometry else "")]
            for c, v in enumerate(vals):
                t.setItem(i, c, QtWidgets.QTableWidgetItem(v))
        tot = sum(r.wt for r in q.rows)
        self.quant_note.setText(f"Total {tot:.1f} wt% (normalised) – {q.method}; E0 {q.beam_kv:.1f} kV. "
                                + " ".join(q.notes))

    def sync_beam_kv(self):
        """Beam energy from the SEM (its remote control reports the accelerating voltage)."""
        kv = None
        if self.microscope is not None:
            try:
                kv = self.microscope.accelerating_voltage_kv()
            except Exception as e:  # noqa: BLE001
                self.notify(f"SEM beam voltage: {e}")
                return
        if kv is None:
            self.notify("the SEM does not report its accelerating voltage "
                        f"({getattr(self.microscope, 'name', 'no SEM link')}) – enter the beam energy by hand")
            return
        self.c_kv.setValue(float(kv))
        self.notify(f"nominal beam energy {float(kv):g} kV (from the SEM)")

    def _conditions_changed(self, *_):
        s = self.settings
        s.beam_kv, s.takeoff_deg = self.c_kv.value(), self.c_to.value()
        s.fwhm_mn_ev, s.window = self.c_fwhm.value(), self.c_win.currentText()
        self.ctrl.beam_kv, self.ctrl.takeoff_deg = s.beam_kv, s.takeoff_deg
        for site in self._sites:
            sp = site.spectrum
            if sp is not None:
                sp.takeoff_deg = s.takeoff_deg
                if (sp.metadata or {}).get("beam_kv_source") == "nominal":  # measured E0 stays
                    sp.beam_kv = s.beam_kv
        self.specview.set_markers(self.elements, self._kv())
        self._show_e0()
        self._maps_changed()

    # ================================================================ maps
    def _map_elements(self):
        if not self.elements and self.map is not None:
            ids = A.identify_elements(self.map.sum_spectrum().with_calibration(self.ctrl.energy_calibration),
                                      detector=self.detector)
            return [i.element for i in ids]
        return list(self.elements)

    def compute_maps(self):
        """Recompute the element maps now (of the finished map, or of the running one)."""
        if self.map is None and self._map_running is None:
            self.notify("no map acquired")
            return
        self._maps_changed()
        self._update_maps(force=True)

    def _maps_changed(self):
        """An input of the maps changed: the next refresh recomputes them."""
        self._maps_gen += 1

    #: live maps: at least this long between two computations (longer if a computation is slow)
    LIVE_MAPS_S = 1.0

    def _update_maps(self, force: bool = False):
        """Start a map computation when there is something new: pixels scanned since the
        maps shown, or changed inputs.  One computation at a time, in its own worker."""
        if self._maps_fut is not None:
            return
        running = self._map_running is not None
        sm = self._map_running if running else self.map
        if sm is None or (running and not self.m_live.isChecked() and not force):
            return
        px = int(sm.done.sum()) if sm.done is not None else int(np.prod(sm.shape))
        if px == 0 or (not force and self._maps_gen == self._maps_done_gen and px == self._maps_px):
            return
        now = time.monotonic()
        if running and not force and now - self._maps_t < max(self.LIVE_MAPS_S, 2.0 * self._maps_dt):
            return
        done = None if sm.done is None else sm.done.copy()  # the pixels this computation covers
        sm = SpectrumMap(sm.cube, self.ctrl.energy_calibration, sm.region_um, sm.dwell_s, sm.real_time,
                         sm.live_time, sm.fast_count, sm.se, done, sm.beam_kv, self.ctrl.takeoff_deg,
                         sm.timestamp, sm.metadata)
        kind, binning, det = self.m_kind.currentIndex(), self.m_bin.value(), self.detector
        oxide, els0 = self.q_oxide.isChecked(), list(self.elements)
        cal = self.ctrl.energy_calibration
        gen = self._maps_gen

        def work():
            t0 = time.perf_counter()
            m = A.bin_map(sm, binning)
            els = els0
            if not els:  # nothing selected: AutoID on the sum (once enough counts arrived)
                s = m.sum_spectrum().with_calibration(cal)
                els = [i.element for i in A.identify_elements(s, detector=det)] if s.total > 5000 else []
            if not els:
                return els, [], time.perf_counter() - t0
            if kind == 0:
                maps = A.window_maps(m, els, det)
            elif kind == 1:
                maps, _ = A.fitted_maps(m, els, det)
            else:
                maps, _ = A.quant_maps(m, els, det, oxygen_by_stoichiometry=oxide)
            if m.done is not None and not m.done.all():
                for mp in maps:  # pixels not scanned yet: blank
                    mp.data = np.where(m.done, mp.data, np.nan)
            return els, maps, time.perf_counter() - t0

        self._maps_t = now
        self._maps_fut = (self._maps_pool.submit(work), gen, px)

    def _maps_poll(self):
        f = self._maps_fut
        if f is None or not f[0].done():
            return
        self._maps_fut = None
        fut, gen, px = f
        try:
            els, maps, dt = fut.result()
        except Exception as e:  # noqa: BLE001 - e.g. too few counts for a fit early in a scan
            if self._map_running is None:
                self.notify(f"maps failed: {type(e).__name__}: {e}")
            self._maps_done_gen, self._maps_px = gen, px  # do not retry the same inputs at once
            return
        self._maps_dt = dt
        self._maps_done_gen, self._maps_px = gen, px
        if els and not self.elements:
            self.set_elements(els)  # AutoID result (bumps the generation: recomputed with them)
        if maps:
            self._show_maps(maps)

    def compute_phases(self):
        if not self._maps:
            self.notify("compute element maps first")
            return
        maps, k = [m for m in self._maps if m.unit != "phase"], self.ph_k.value()

        def work():
            return A.phase_map(maps, k=k)

        def done(ph):
            self._phase = ph
            lab = ph.labels.astype(float)
            from ..analysis.maps import ElementMap

            self._show_maps(maps + [ElementMap("Phases", lab, "phase")])
            txt = []
            for j, (c, fr) in enumerate(zip(ph.centroids, ph.fractions)):
                comp = ", ".join(f"{lbl.split()[0]} {v:.0f}" for lbl, v in zip(ph.elements, c) if v > 0.05 * c.max())
                txt.append(f"phase {j}: {100 * fr:.0f} % of area – {comp}")
            self.notify("; ".join(txt))

        self._run_analysis(work, done, "phase map")

    def _show_maps(self, maps):
        self._maps = list(maps)
        sm = self._map_running if self._map_running is not None else self.map
        tiles = []
        if sm is not None and sm.se is not None:
            se = sm.se if self.m_bin.value() <= 1 else None
            if se is not None:
                tiles.append(("SE", np.where(sm.done, se, np.nan) if sm.done is not None else se))
        tiles += [(m.label, m.data) for m in maps]
        cur = getattr(self, "_tiles", [])
        if [t.label for t in cur] == [lbl for lbl, _ in tiles]:  # same maps: new data in place
            for t, (_, d) in zip(cur, tiles):
                t.set_data(d)
            self._refresh_overlay()
            return
        self._tiles = []
        while self.maps_grid.count():
            it = self.maps_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        cols = 4
        tiles = []
        if sm is not None and sm.se is not None and self.m_bin.value() <= 1:
            se = np.where(sm.done, sm.se, np.nan) if sm.done is not None else sm.se
            tiles.append(("SE", se, "V", (220, 220, 220)))
        for i, m in enumerate(maps):
            col = (255, 255, 255) if m.unit == "phase" else MAP_COLORS[i % len(MAP_COLORS)]
            tiles.append((m.label, m.data, m.unit, col))
        for i, (lbl, d, unit, col) in enumerate(tiles):
            t = MapTile(lbl, d, unit, col)
            t.clicked.connect(self._map_pixel)
            self.maps_grid.addWidget(t, i // cols, i % cols)
            self._tiles.append(t)
        self.m_overlay.blockSignals(True)
        self.m_overlay.clear()
        self.m_overlay.addItem("(no overlay)")
        for m in maps:
            self.m_overlay.addItem(m.label)
        self.m_overlay.blockSignals(False)
        self.bottom.setCurrentWidget(self.maps_area)

    def _refresh_overlay(self):
        """Live maps: the element overlay on the image follows the new data."""
        i = self.m_overlay.currentIndex()
        if i > 0 and self._map_running is None:
            self._overlay_changed(i)

    def _overlay_changed(self, i):
        if i <= 0 or self.map is None:
            self.imgview.set_overlay(None)
            return
        m = self._maps[i - 1]
        col = MAP_COLORS[(i - 1) % len(MAP_COLORS)]
        region = self._frac_of(self.map)
        if m.data.shape != self.map.shape:
            pass  # binned: same region, coarser
        self.imgview.set_overlay(m.data, region, col)

    def _map_pixel(self, row, col):
        sm = self._map_running if self._map_running is not None else self.map
        if sm is None:
            return
        b = self.m_bin.value()
        r0, c0 = row * b, col * b
        mask = np.zeros(sm.shape, bool)
        mask[r0:r0 + b, c0:c0 + b] = True
        sp = sm.sum_spectrum(mask).with_calibration(self.ctrl.energy_calibration)
        sp.title = f"map pixel ({col}, {row})"
        sp.metadata = dict(sp.metadata or {}, beam_kv_source=sm.metadata.get("beam_kv_source", "nominal"))
        site = _Site(next(self._site_ids), "point",
                     self._frac(sm.pixel_to_um(c0 + (b - 1) / 2, r0 + (b - 1) / 2), sm.metadata.get("field_um")))
        site.text = sp.title
        site.spectrum = sp
        self._sites.append(site)
        self.site_list.addItem(f"{site.name} – {sp.title}")
        self.site_list.setCurrentRow(self.site_list.count() - 1)

    def show_map_sum(self):
        if self.map is None:
            return
        sp = self.map.sum_spectrum().with_calibration(self.ctrl.energy_calibration)
        sp.metadata = dict(sp.metadata or {}, beam_kv_source=self.map.metadata.get("beam_kv_source", "nominal"))
        site = _Site(next(self._site_ids), "field", self._frac_of(self.map))
        site.text = f"map sum ({int(self.map.done.sum())} px)"
        site.spectrum = sp
        self._sites.append(site)
        self.site_list.addItem(f"{site.name} – map sum ({int(self.map.done.sum())} px)")
        self.site_list.setCurrentRow(self.site_list.count() - 1)

    def _show_linescan(self, sm: SpectrumMap):
        self.profile.clear()
        meta = sm.metadata
        p0, p1 = np.asarray(meta["p0_um"]), np.asarray(meta["p1_um"])
        n = sm.shape[1]
        d = (np.arange(n) + 0.5) / n * float(np.hypot(*(p1 - p0)))
        els = self.elements or self._map_elements()
        det = self.detector
        for i, el in enumerate(els):
            fam = A.choose_family(el, sm.beam_kv)
            ln = xray.main_line(el, fam) if fam else None
            if ln is None:
                continue
            fw = float(det.fwhm_kev(ln.energy))
            y = sm.window_image(ln.energy - fw, ln.energy + fw)[0]
            self.profile.plot(d, y, pen=pg.mkPen(MAP_COLORS[i % len(MAP_COLORS)], width=2), name=f"{el} {fam}")
        if sm.se is not None:
            se = sm.se[0]
            if np.isfinite(se).any() and np.nanmax(se) > 0:
                top = max((it.yData.max() for it in self.profile.getPlotItem().listDataItems()
                           if it.yData is not None and len(it.yData)), default=1.0)
                self.profile.plot(d, se / np.nanmax(se) * top, pen=pg.mkPen("#bbb", style=QtCore.Qt.DashLine),
                                  name="SE (scaled)")
        self.bottom.setCurrentWidget(self.profile)

    # ================================================================ dialogs
    def open_calibration(self):
        sp = self._spectrum_of_current()

        def acquire(seconds):
            site = self._current
            kw = {"live_time": True, "title": "calibration"}
            if site is not None and site.kind == "point":
                kw["at_um"] = self._um(site.params)
            elif site is not None and site.kind in ("rect", "ellipse", "field"):
                kw["region_um"], kw["shape"] = self._um(site.params), ("ellipse" if site.kind == "ellipse" else "rect")
            elif self.image is not None:
                kw["region_um"] = self._um(self._image_region())
            return self.ctrl.acquire_spectrum(seconds, **kw)

        d = CalibrationDialog(sp, self.ctrl.energy_calibration, acquire=acquire, parent=self,
                              points=self.settings.calibration_points)
        d.calibrationApplied.connect(self._calibration_applied)
        d.pointsChanged.connect(self._calibration_points_changed)
        self._calib_dialog = d
        d.show()

    def _fill_speeds(self):
        """The speed presets, with the one the processor is on selected ("custom" otherwise)."""
        try:
            speeds = self.ctrl.mca.speeds()
            cur = self.ctrl.mca.current_speed()
        except Exception:  # noqa: BLE001 - a processor without presets
            speeds, cur = {}, None
        self.sp_speed.blockSignals(True)
        self.sp_speed.clear()
        for k, desc in speeds.items():
            label = {min(speeds): " – fastest", max(speeds): " – best resolution"}.get(k, "")
            self.sp_speed.addItem(f"{desc}{label}", k)
        if cur in speeds:
            self.sp_speed.setCurrentIndex(self.sp_speed.findData(cur))
        else:
            self.sp_speed.addItem("custom (set in the processor dialog)", None)
            self.sp_speed.setCurrentIndex(self.sp_speed.count() - 1)
        self.sp_speed.setEnabled(bool(speeds))
        self.sp_speed.blockSignals(False)

    def _speed_chosen(self, i: int):
        k = self.sp_speed.itemData(i)
        if k is None:
            return
        if self.ctrl.busy:  # written when the acquisition is done
            self._speed_pending = k
            self.notify(f"processor speed {k}: applied after the acquisition")
            return
        self._apply_speed(k)

    def _apply_speed(self, k: int):
        self._speed_pending = None
        try:
            now = self.ctrl.set_speed(k)
            self.notify(f"processor speed {now if now is not None else k} "
                        "(peak positions are kept; the slow threshold follows Min. energy)")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.notify(f"processor speed not applied: {e}")
        self._fill_speeds()

    def apply_min_energy(self):
        """Write the slow threshold for the minimum energy (deferred while an acquisition runs)."""
        e = float(self.sp_emin.value())
        self.settings.min_energy_kev = e
        if self.ctrl.busy:
            self._emin_pending = True
            return
        try:
            self.ctrl.set_min_energy(e)
            self._emin_pending = False
        except Exception as e_:  # noqa: BLE001 - shown to the user
            self.notify(f"min. energy not applied: {e_}")

    def _calibration_points_changed(self, points: list):
        self.settings.calibration_points = list(points)
        self._save_settings()

    def _calibration_applied(self, cal):
        self.ctrl.set_energy_calibration(cal)
        self._maps_changed()
        self.settings.energy_calibration = cal
        self.settings.calibration_source = f"{self.ctrl.mca.name} {self.ctrl.mca.info().get('serial', '')}"
        self._save_settings()
        self.apply_min_energy()  # the threshold in % depends on the calibration
        self.notify(f"energy calibration applied: offset {cal.offset_kev * 1e3:+.2f} eV, "
                    f"slope {cal.gain_kev * 1e3:.4f} eV/ch")
        self._show_current()

    def open_processor(self):
        if self._busy():
            return
        d = ProcessorDialog(self.ctrl.mca, self)
        d.channelsChanged.connect(lambda o, n: self.notify(
            f"MCA channels {o} → {n}: the energy calibration changed – recalibrate (Tools → Energy calibration)"))
        with self.ctrl.idle_paused():  # the dialog owns the processor while it is open
            d.exec()
        self.apply_min_energy()  # channels may have changed (or THSL written by hand)
        self._fill_speeds()  # the shaping may have been changed by hand

    # ================================================================ files
    def _folder(self):
        return self.settings.save_folder or os.getcwd()

    def open_spectrum(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open spectrum", self._folder(),
                                                        "Spectra (*.msa *.emsa *.txt *.npz)")
        if not path:
            return
        try:
            sp = load_spectrum(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Open", str(e))
            return
        site = _Site(next(self._site_ids), "spot", ())
        site.text = os.path.basename(path)
        site.spectrum = sp
        sp.title = sp.title or os.path.basename(path)
        sp.metadata = dict(sp.metadata or {}, beam_kv_source="file")
        self._sites.append(site)
        self.site_list.addItem(f"{site.name} – {os.path.basename(path)}")
        self.site_list.setCurrentRow(self.site_list.count() - 1)

    def save_spectrum(self):
        sp = self.current_spectrum()
        if sp is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save spectrum",
                                                        os.path.join(self._folder(), f"{sp.title or 'spectrum'}.msa"),
                                                        "EMSA (*.msa);;NumPy (*.npz)")
        if path:
            sp.save(path)
            self.settings.save_folder = os.path.dirname(path)
            self.notify(f"saved {path}")

    def open_map(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open map", self._folder(), "Maps (*.npz)")
        if path:
            try:
                self.map = load_map(path)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Open", str(e))
                return
            self._map_done(self.map)

    def save_map(self):
        if self.map is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save map", os.path.join(self._folder(), "map.npz"),
                                                        "Map (*.npz)")
        if path:
            self.map.save(path)
            self.notify(f"saved {path}")

    def export_maps(self):
        if not self._maps:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export maps to", self._folder())
        if not folder:
            return
        from ...imaging.export import write_png

        for m in self._maps:
            d = np.nan_to_num(m.data)
            hi = np.percentile(d, 99.5) if d.size else 1
            img = (np.clip(d / (hi if hi > 0 else 1), 0, 1) * 255).astype(np.uint8)
            write_png(os.path.join(folder, f"{m.label.replace(' ', '_')}.png"), img)
        self.notify(f"exported {len(self._maps)} maps to {folder}")

    def export_quant(self):
        q = self._quant
        if q is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export quantification",
                                                        os.path.join(self._folder(), "quant.csv"), "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["element", "line", "wt%", "wt% sigma", "at%", "oxide", "oxide wt%", "net counts"])
            for r in q.rows:
                w.writerow([r.element, r.family, f"{r.wt:.3f}", f"{r.wt_sigma:.3f}", f"{r.at:.3f}", r.oxide,
                            f"{r.oxide_wt:.3f}" if r.oxide else "", f"{r.intensity:.0f}"])
        self.notify(f"saved {path}")

    def save_image(self):
        if self.image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save electron image",
                                                        os.path.join(self._folder(), "electron.png"),
                                                        "PNG (*.png);;NumPy (*.npy)")
        if not path:
            return
        if path.lower().endswith(".npy"):
            np.save(path, self.image.data)
        else:
            from ...imaging.export import write_png

            d = np.nan_to_num(self.image.data)
            lo, hi = np.percentile(d, [0.5, 99.5])
            write_png(path, (np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))
        self.notify(f"saved {path}")

    # ================================================================ periodic updates
    def _follow_microscope_kv(self):
        """The SEM's accelerating voltage (while HV is on) is the nominal beam energy."""
        if self.microscope is None:
            return
        try:
            kv = self.microscope.accelerating_voltage_kv()
        except Exception:  # noqa: BLE001
            return
        if kv and kv != self._mic_kv:
            self._mic_kv = kv
            if abs(self.c_kv.value() - kv) > 1e-6:
                self.c_kv.setValue(float(kv))
                self.notify(f"nominal beam energy {kv:g} kV (from the SEM)")

    def _microscope_metadata(self) -> dict:
        if self.bundle is not None:
            return self.bundle.metadata()
        return self.microscope.snapshot() if self.microscope is not None else {}

    def _tick(self):
        self._follow_microscope_kv()
        f = self._field()
        if f != getattr(self, "_field_shown", None):
            self._field_changed()
        self._field_shown = f
        for ev in self._sub.drain():
            self._event(ev)
        job = self._job
        if job is not None and job.done():
            self._job = None
            self._job_done(job)
        self._maps_poll()
        if self._job is None and self._map_running is not None:
            self._map_running = None  # the map job ended without a result (cancelled / failed)
        self._update_maps()
        a = self._analysis
        if a is not None and a[0].done():
            self._analysis = None
            fut, cb, what = a
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                self.notify(f"{what} failed: {type(e).__name__}: {e}")
            else:
                try:
                    cb(res)
                except Exception as e:  # noqa: BLE001
                    self.notify(f"{what}: display failed: {e}")
            self.l_state.setText("Idle" if not self.ctrl.busy else self.l_state.text())
        self._update_status()

    def _event(self, ev):
        d = ev.data
        if ev.kind in ("error", "warning", "job_started", "job_done", "settings", "info"):
            self.notify(ev.message)
        if ev.kind in ("job_started", "job_done"):
            self._live_prev = self._live_px = None
        if ev.kind == "spectrum":
            sp = d.get("spectrum")
            job = self._job
            site = getattr(job, "_site", None) if job is not None and d.get("job") == job.id else None
            if site is not None and sp is not None:
                site.spectrum = sp
                if site is self._current:
                    self.specview.set_spectrum(f"site{site.id}", sp.with_calibration(self.ctrl.energy_calibration),
                                               site.color, site.name)
                    if time.monotonic() - getattr(self, "_last_auto", 0) > 1.0:
                        self._last_auto = time.monotonic()
                        self.specview.autoscale()
            if sp is not None and sp.real_time > 0:
                self._live_from_partial(sp)
        elif ev.kind == "idle_spectrum":
            sp = d.get("spectrum")
            if sp is not None:
                self._show_live(sp)
        elif ev.kind == "image":
            im = d.get("image")
            if im is not None:
                self._show_image(im)
        elif ev.kind == "pixel":
            sm = d.get("map")
            if sm is not None:
                self.m_progress.setValue(int(100 * d.get("progress", 0)))
                self._live_from_map(sm)
                job = self._job
                if job is not None and job.kind == "map" and d.get("job") in (None, job.id):
                    if self._map_running is not sm:
                        self._map_running = sm
                        self._maps_changed()
                if time.monotonic() - self._map_live_t > 1.0 and sm.se is not None:
                    self._map_live_t = time.monotonic()
                    se = np.where(sm.done, sm.se, np.nan)
                    if np.isfinite(se).any() and sm.shape[0] > 1:
                        self.imgview.set_overlay(np.nan_to_num(se - np.nanmin(se)), self._frac_of(sm),
                                                (120, 200, 255))
        if "progress" in d:
            self.p_job.setValue(int(1000 * float(d["progress"])))

    def _job_done(self, job):
        try:
            res = job.result(0)
        except Exception as e:  # noqa: BLE001
            self.notify(f"{job.description} failed: {e}")
            return
        if job.kind == "spectrum" and isinstance(res, Spectrum):
            site = getattr(job, "_site", None)
            if site is not None:
                site.spectrum = res
                res.takeoff_deg = self.ctrl.takeoff_deg
                self._assign_beam_kv(res)
                mic = self._microscope_metadata()
                if mic:
                    res.metadata = dict(res.metadata or {}, microscope=mic)
                if site is self._current:
                    self._show_current()
                if self.sp_autoid.isChecked() and site is self._current and res.total > 200 and not job.cancelled:
                    self.analyse(autoid=True)
        elif job.kind == "image" and isinstance(res, ElectronImage):
            self._show_image(res)
            self._limit_map_region()
        elif job.kind == "map" and isinstance(res, SpectrumMap):
            self._map_running = None
            self._map_beam_kv(res)
            self.map = res
            self._map_done(res)
        elif job.kind == "linescan" and isinstance(res, SpectrumMap):
            self._map_beam_kv(res)
            self.map = res
            self._show_linescan(res)
        self.p_job.setValue(1000)
        if self._speed_pending is not None:
            self._apply_speed(self._speed_pending)
        if self._emin_pending:
            self.apply_min_energy()

    def _map_beam_kv(self, sm: SpectrumMap):
        """E0 of a map / line scan from its sum spectrum (Duane–Hunt), else the nominal value."""
        s = sm.sum_spectrum()
        self._assign_beam_kv(s)
        sm.beam_kv = s.beam_kv
        sm.metadata = dict(sm.metadata or {}, beam_kv_source=s.metadata["beam_kv_source"])
        self.notify(f"map E0 {sm.beam_kv:.1f} kV ({sm.metadata['beam_kv_source']})")

    def _map_done(self, sm: SpectrumMap):
        self.imgview.set_overlay(None)
        self.m_progress.setValue(int(100 * sm.done.mean()))
        self.imgview.add_site("mapregion", "map", self._frac_of(sm), "#ffffff", "map")
        oh = sm.metadata.get("overhead_s")
        self.notify(f"map {sm.shape[1]}×{sm.shape[0]}: {int(sm.done.sum())} px, "
                    f"{sm.sum_spectrum().total:,} counts" + (f", overhead {oh * 1e3:.0f} ms/px" if oh else ""))
        self._maps_changed()  # final maps of the whole scan (E0 from its sum spectrum)
        self._update_maps(force=True)

    def _update_status(self):
        st = self.ctrl.idle_state
        live_delta = st == "job" and (self._live_prev is not None or self._live_px is not None)
        if st != "live" and not live_delta:  # a running job shows its own deltas
            txt = {"job": "acquisition running", "off": "live read-out off", "paused": "paused"}.get(st, st)
            self.l_live_t.setText(txt if len(txt) < 48 else txt[:45] + "…")
            self.l_live_t.setToolTip(st)
        j = self.ctrl.job
        if j is not None and not j.done():
            self.l_state.setText(f"{j.description} – {j.message}")
            self.l_state.setStyleSheet("color:#fb4")
        elif self._analysis is None:
            self.l_state.setText("Idle")
            self.l_state.setStyleSheet("color:#4c4")
        lease = self.ctrl.handle.manager.lease
        if lease is not None and lease.owner != self.ctrl.handle.owner:
            self.l_state.setText(f"DAQ reserved by '{lease.owner}'")
            self.l_state.setStyleSheet("color:#f90")

    # ================================================================ lifecycle
    def closeEvent(self, ev):
        self.timer.stop()
        self.ctrl.abort()  # stop counting now: cancel the job, stop the live read-out, MCA off
        self._save_settings()
        if self.ctrl.busy:
            self.ctrl.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._maps_pool.shutdown(wait=False, cancel_futures=True)
        self._sub.close()
        if self.owns_controller:
            self.ctrl.close()
        super().closeEvent(ev)
