"""The EDS main window – an Oxford-AZtec-like front end for :class:`EDSController`.

The window only *displays* and *commands*: acquisitions run in the
controller's job threads, analysis (AutoID, fitting, quantification, maps) in
a worker thread; results are picked up by timers.  Nothing here touches the
DAQ or the pulse processor directly.

Layout: electron image with acquisition sites (left) · spectrum with element
markers and fit (centre) · quantification / maps / line profile (bottom) ·
acquisition and analysis controls (right).
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
        self.color = SITE_COLORS[(sid - 1) % len(SITE_COLORS)]

    @property
    def name(self) -> str:
        return f"Spectrum {self.id}"

    def describe(self) -> str:
        p = self.params
        if self.kind == "point":
            return f"{self.name} – point ({p[0]:.2f}, {p[1]:.2f}) µm"
        if self.kind in ("rect", "ellipse"):
            return f"{self.name} – {self.kind} {p[2]:.2f}×{p[3]:.2f} µm"
        if self.kind == "field":
            return f"{self.name} – whole field"
        return f"{self.name} – SEM spot"


class EDSWindow(QtWidgets.QMainWindow):
    TICK_MS = 100

    def __init__(self, controller: EDSController, settings: EDSSettings | None = None, sem_engine=None,
                 simulated: bool = False, owns_controller: bool = True):
        super().__init__()
        self.ctrl = controller
        self.settings = settings or EDSSettings()
        self.sem = sem_engine
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
        self._calib_dialog = None
        self._apply_settings_to_controller()

        # ------------------------------------------------ central area
        self.imgview = ElectronImageView()
        self.imgview.siteRequested.connect(self._site_requested)
        self.imgview.pointerMoved.connect(lambda x, y: self.l_ptr.setText(f"x {x:.2f} µm  y {y:.2f} µm"))
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
        tools.addSeparator()
        tools.addAction("Clear sites", self.clear_sites)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QtWidgets.QLabel("<b>Electron image</b>"))
        ll.addWidget(tools)
        ll.addWidget(self.imgview, 1)
        ll.addWidget(self.site_list)

        # bottom tabs: quant / maps / line profile
        self.quant_table = QtWidgets.QTableWidget(0, 6)
        self.quant_table.setHorizontalHeaderLabels(["Element", "Line", "Wt%", "Wt% σ", "At%", "Oxide wt%"])
        self.quant_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.quant_note = QtWidgets.QLabel()
        self.quant_note.setStyleSheet("color:#aaa")
        qw = QtWidgets.QWidget()
        ql = QtWidgets.QVBoxLayout(qw)
        ql.setContentsMargins(2, 2, 2, 2)
        ql.addWidget(self.quant_table)
        ql.addWidget(self.quant_note)
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
        self.bottom.addTab(qw, "Quantification")
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

        # ------------------------------------------------ controls dock
        self.controls = QtWidgets.QTabWidget()
        self.controls.addTab(self._acquire_tab(), "Acquire")
        self.controls.addTab(self._map_tab(), "Map")
        self.controls.addTab(self._analysis_tab(), "Analyse")
        dock = QtWidgets.QDockWidget("EDS")
        dock.setObjectName("eds_controls")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setWidget(self.controls)
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
        dc = controller.daq_config
        if dc.phantom_kind == "eds":
            self.notify(f"EDS phantom: SE image on {dc.phantom_output} ({dc.phantom_physical or 'sim'}), "
                        f"SDD signal on {dc.sdd_output} ({dc.sdd_physical or 'sim'})")
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
    def _acquire_tab(self):
        s = self.settings
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Electron image")
        f = QtWidgets.QFormLayout(g)
        self.im_res = QtWidgets.QComboBox()
        for nx, ny in [(256, 192), (512, 384), (768, 576), (1024, 768), (256, 256), (512, 512)]:
            self.im_res.addItem(f"{nx} × {ny}", f"{nx}x{ny}")
        i = self.im_res.findData(f"{s.image_nx}x{s.image_ny}")
        self.im_res.setCurrentIndex(max(0, i))
        self.im_dwell = QtWidgets.QDoubleSpinBox()
        self.im_dwell.setRange(0.5, 1000)
        self.im_dwell.setValue(s.image_dwell_us)
        self.im_dwell.setSuffix(" µs")
        self.b_image = QtWidgets.QPushButton("Acquire image")
        self.b_image.clicked.connect(self.acquire_image)
        self.b_semimg = QtWidgets.QPushButton("Use SEM image")
        self.b_semimg.setEnabled(self.sem is not None)
        self.b_semimg.setToolTip("take the latest frame of the SEM window (same field calibration)")
        self.b_semimg.clicked.connect(self.use_sem_image)
        f.addRow("Resolution", self.im_res)
        f.addRow("Dwell", self.im_dwell)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.b_image)
        row.addWidget(self.b_semimg)
        f.addRow(row)
        lay.addWidget(g)

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
        self.sp_autoacq = QtWidgets.QCheckBox("acquire when a site is drawn")
        self.sp_autoacq.setChecked(True)
        self.sp_autoid = QtWidgets.QCheckBox("AutoID + quant when done")
        self.sp_autoid.setChecked(True)
        f.addRow(self.sp_autoacq)
        f.addRow(self.sp_autoid)
        row = QtWidgets.QHBoxLayout()
        self.b_field = QtWidgets.QPushButton("Whole field")
        self.b_field.setToolTip("spectrum while the beam rasters the whole image")
        self.b_field.clicked.connect(lambda: self._new_site("field", self._image_region()))
        self.b_spot = QtWidgets.QPushButton("SEM spot")
        self.b_spot.setToolTip("acquire without moving the beam (SEM in spot mode / its own scan)")
        self.b_spot.clicked.connect(lambda: self._new_site("spot", ()))
        self.b_again = QtWidgets.QPushButton("Re-acquire")
        self.b_again.clicked.connect(lambda: self.acquire_site(self._current) if self._current else None)
        row.addWidget(self.b_field)
        row.addWidget(self.b_spot)
        row.addWidget(self.b_again)
        f.addRow(row)
        self.b_stop = QtWidgets.QPushButton("Stop")
        self.b_stop.setStyleSheet("QPushButton{background:#8a2c2c;color:white;font-weight:bold}")
        self.b_stop.clicked.connect(self.stop)
        f.addRow(self.b_stop)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Pulse processor")
        f = QtWidgets.QFormLayout(g)
        self.l_rates = QtWidgets.QLabel("–")
        self.l_dead = QtWidgets.QProgressBar()
        self.l_dead.setRange(0, 100)
        self.l_dead.setFormat("dead time %p %")
        f.addRow("Rates", self.l_rates)
        f.addRow(self.l_dead)
        b = QtWidgets.QPushButton("Settings…")
        b.clicked.connect(self.open_processor)
        f.addRow(b)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

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
        f.addRow("Maps", self.m_kind)
        self.m_bin = QtWidgets.QSpinBox()
        self.m_bin.setRange(1, 16)
        f.addRow("Binning", self.m_bin)
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

    def _analysis_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Elements")
        v = QtWidgets.QVBoxLayout(g)
        self.el_label = QtWidgets.QLabel("–")
        self.el_label.setWordWrap(True)
        self.el_label.setStyleSheet("font-size:14px;font-weight:bold")
        v.addWidget(self.el_label)
        r = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("AutoID")
        b.clicked.connect(lambda: self.analyse(autoid=True))
        r.addWidget(b)
        b = QtWidgets.QPushButton("Periodic table…")
        b.clicked.connect(self.pick_elements)
        r.addWidget(b)
        b = QtWidgets.QPushButton("Clear")
        b.clicked.connect(lambda: self.set_elements([]))
        r.addWidget(b)
        v.addLayout(r)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("Quantification")
        f = QtWidgets.QFormLayout(g)
        self.q_oxide = QtWidgets.QCheckBox("oxygen by stoichiometry (oxides)")
        self.q_oxide.setChecked(self.settings.oxygen_by_stoichiometry)
        self.q_oxide.toggled.connect(lambda on: setattr(self.settings, "oxygen_by_stoichiometry", on))
        f.addRow(self.q_oxide)
        b = QtWidgets.QPushButton("Fit && quantify")
        b.clicked.connect(lambda: self.analyse(autoid=False))
        f.addRow(b)
        self.q_fit = QtWidgets.QCheckBox("show fit")
        self.q_fit.setChecked(True)
        self.q_fit.toggled.connect(self._show_fit)
        f.addRow(self.q_fit)
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
        g = QtWidgets.QGroupBox("Conditions")
        f = QtWidgets.QFormLayout(g)
        self.c_kv = QtWidgets.QDoubleSpinBox()
        self.c_kv.setRange(1, 40)
        self.c_kv.setValue(self.settings.beam_kv)
        self.c_kv.setSuffix(" kV")
        self.c_kv.valueChanged.connect(self._conditions_changed)
        self.c_to = QtWidgets.QDoubleSpinBox()
        self.c_to.setRange(5, 90)
        self.c_to.setValue(self.settings.takeoff_deg)
        self.c_to.setSuffix(" °")
        self.c_to.valueChanged.connect(self._conditions_changed)
        self.c_fwhm = QtWidgets.QDoubleSpinBox()
        self.c_fwhm.setRange(100, 300)
        self.c_fwhm.setValue(self.settings.fwhm_mn_ev)
        self.c_fwhm.setSuffix(" eV @ Mn Kα")
        self.c_fwhm.valueChanged.connect(self._conditions_changed)
        self.c_win = QtWidgets.QComboBox()
        self.c_win.addItems(["AP3.3", "Be8", "none"])
        self.c_win.setCurrentText(self.settings.window)
        self.c_win.currentTextChanged.connect(self._conditions_changed)
        f.addRow("Beam", self.c_kv)
        f.addRow("Take-off", self.c_to)
        f.addRow("Resolution", self.c_fwhm)
        f.addRow("Window", self.c_win)
        lay.addWidget(g)
        lay.addStretch(1)
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

    def _image_region(self) -> tuple:
        return self.image.region_um if self.image is not None else self.ctrl.full_field_um()

    def _cursor(self, e, c):
        near = xray.lines_near(e, tol=0.04, beam_kv=self.ctrl.beam_kv, min_weight=0.3)[:3]
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
        self.image = ElectronImage(np.asarray(f.image, float), region, cfg.dwell_s, metadata={"source": "SEM"})
        self.imgview.set_image(self.image)
        self.notify(f"electron image from the SEM: {cfg.nx}×{cfg.ny}, {cfg.width_um:g} µm field")

    def _site_requested(self, kind: str, params: tuple):
        if kind == "line":
            self.line = params
            self.l_line.setText(f"({params[0]:.1f}, {params[1]:.1f}) → ({params[2]:.1f}, {params[3]:.1f}) µm, "
                                f"{np.hypot(params[2] - params[0], params[3] - params[1]):.2f} µm")
            self.imgview.add_site("line", "line", params, "#ffffff", "line")
            return
        if kind == "map":
            self.map_region = params
            self.m_region.setText(f"{params[2]:.2f} × {params[3]:.2f} µm at ({params[0]:.1f}, {params[1]:.1f})")
            self.imgview.add_site("mapregion", "map", params, "#ffffff", "map")
            return
        self._new_site(kind, params)

    def _new_site(self, kind: str, params: tuple):
        site = _Site(next(self._site_ids), kind, params)
        self._sites.append(site)
        self.site_list.addItem(site.describe())
        self.site_list.item(self.site_list.count() - 1).setForeground(QtGui.QColor(site.color))
        if kind in ("point", "rect", "ellipse"):
            self.imgview.add_site(site.id, kind, params, site.color, str(site.id))
        self.site_list.setCurrentRow(self.site_list.count() - 1)
        if self.sp_autoacq.isChecked() or kind in ("field", "spot"):
            self.acquire_site(site)

    def acquire_site(self, site: _Site | None):
        if site is None:
            return
        t = self.sp_time.value()
        live = self.sp_live.isChecked()
        self.settings.spectrum_time_s, self.settings.live_time_mode = t, live
        kw = {"live_time": live, "title": site.name}
        if site.kind == "point":
            kw["at_um"] = site.params
        elif site.kind in ("rect", "ellipse"):
            kw["region_um"], kw["shape"] = site.params, site.kind
        elif site.kind == "field":
            kw["region_um"], kw["shape"] = site.params, "rect"
        job = self._start(self.ctrl.acquire_spectrum, t, **kw)
        if job is not None:
            job._site = site

    def start_map(self):
        region = self.map_region or self._image_region()
        s = self.settings
        s.map_nx, s.map_ny, s.map_dwell_ms = self.m_nx.value(), self.m_ny.value(), self.m_dwell.value()
        job = self._start(self.ctrl.acquire_map, s.map_nx, s.map_ny, s.map_dwell_ms / 1e3, region)
        if job is not None:
            self.bottom.setCurrentIndex(1)
            self.m_progress.setValue(0)

    def start_linescan(self):
        if self.line is None:
            self.notify("draw a line on the electron image first (Line tool)")
            return
        x0, y0, x1, y1 = self.line
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
        self.specview.set_markers(self.elements, self.ctrl.beam_kv)
        self._show_fit()
        self.specview.autoscale()

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
        self.elements = sorted(set(els), key=xray.z_of)
        self.el_label.setText(" ".join(self.elements) if self.elements else "–")
        self.specview.set_markers(self.elements, self.ctrl.beam_kv)
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
            ids = A.identify_elements(sp, detector=det) if autoid else None
            elements = [i.element for i in ids] if ids is not None else els
            if not elements:
                return elements, None, None
            fit = A.fit_spectrum(sp, elements, detector=det)
            q = A.quantify(fit, sp.beam_kv, sp.takeoff_deg, det, oxygen_by_stoichiometry=oxide)
            return elements, fit, q

        def done(res):
            elements, fit, q = res
            self.set_elements(elements)
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
        self.quant_note.setText(f"Total {tot:.1f} wt% (normalised) – {q.method}; {q.beam_kv:g} kV. "
                                + " ".join(q.notes))
        self.bottom.setCurrentIndex(0)

    def _conditions_changed(self, *_):
        s = self.settings
        s.beam_kv, s.takeoff_deg = self.c_kv.value(), self.c_to.value()
        s.fwhm_mn_ev, s.window = self.c_fwhm.value(), self.c_win.currentText()
        self.ctrl.beam_kv, self.ctrl.takeoff_deg = s.beam_kv, s.takeoff_deg
        for site in self._sites:
            if site.spectrum is not None:
                site.spectrum.beam_kv, site.spectrum.takeoff_deg = s.beam_kv, s.takeoff_deg
        self.specview.set_markers(self.elements, s.beam_kv)

    # ================================================================ maps
    def _map_elements(self):
        if not self.elements and self.map is not None:
            ids = A.identify_elements(self.map.sum_spectrum().with_calibration(self.ctrl.energy_calibration),
                                      detector=self.detector)
            return [i.element for i in ids]
        return list(self.elements)

    def compute_maps(self):
        if self.map is None:
            self.notify("no map acquired")
            return
        sm = self.map
        sm = SpectrumMap(sm.cube, self.ctrl.energy_calibration, sm.region_um, sm.dwell_s, sm.real_time, sm.live_time,
                         sm.fast_count, sm.se, sm.done, self.ctrl.beam_kv, self.ctrl.takeoff_deg, sm.timestamp,
                         sm.metadata)
        kind, binning, det = self.m_kind.currentIndex(), self.m_bin.value(), self.detector
        oxide = self.q_oxide.isChecked()

        def work():
            m = A.bin_map(sm, binning)
            els = self._map_elements()
            if not els:
                return els, []
            if kind == 0:
                maps = A.window_maps(m, els, det)
            elif kind == 1:
                maps, _ = A.fitted_maps(m, els, det)
            else:
                maps, _ = A.quant_maps(m, els, det, oxygen_by_stoichiometry=oxide)
            return els, maps

        def done(res):
            els, maps = res
            if els and not self.elements:
                self.set_elements(els)
            self._show_maps(maps)

        self._run_analysis(work, done, "computing maps")

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
        while self.maps_grid.count():
            it = self.maps_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        cols = 4
        tiles = []
        if self.map is not None and self.map.se is not None:
            tiles.append(("SE", self.map.se, "V", (220, 220, 220)))
        for i, m in enumerate(maps):
            col = (255, 255, 255) if m.unit == "phase" else MAP_COLORS[i % len(MAP_COLORS)]
            tiles.append((m.label, m.data, m.unit, col))
        for i, (lbl, d, unit, col) in enumerate(tiles):
            t = MapTile(lbl, d, unit, col)
            t.clicked.connect(self._map_pixel)
            self.maps_grid.addWidget(t, i // cols, i % cols)
        self.m_overlay.blockSignals(True)
        self.m_overlay.clear()
        self.m_overlay.addItem("(no overlay)")
        for m in maps:
            self.m_overlay.addItem(m.label)
        self.m_overlay.blockSignals(False)
        self.bottom.setCurrentIndex(1)

    def _overlay_changed(self, i):
        if i <= 0 or self.map is None:
            self.imgview.set_overlay(None)
            return
        m = self._maps[i - 1]
        col = MAP_COLORS[(i - 1) % len(MAP_COLORS)]
        region = self.map.region_um
        if m.data.shape != self.map.shape:
            pass  # binned: same region, coarser
        self.imgview.set_overlay(m.data, region, col)

    def _map_pixel(self, row, col):
        if self.map is None:
            return
        sm = self.map
        b = self.m_bin.value()
        r0, c0 = row * b, col * b
        mask = np.zeros(sm.shape, bool)
        mask[r0:r0 + b, c0:c0 + b] = True
        sp = sm.sum_spectrum(mask).with_calibration(self.ctrl.energy_calibration)
        sp.title = f"map pixel ({col}, {row})"
        site = _Site(next(self._site_ids), "point", sm.pixel_to_um(c0 + (b - 1) / 2, r0 + (b - 1) / 2))
        site.spectrum = sp
        self._sites.append(site)
        self.site_list.addItem(f"{site.name} – {sp.title}")
        self.site_list.setCurrentRow(self.site_list.count() - 1)

    def show_map_sum(self):
        if self.map is None:
            return
        sp = self.map.sum_spectrum().with_calibration(self.ctrl.energy_calibration)
        site = _Site(next(self._site_ids), "field", self.map.region_um)
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
        self.bottom.setCurrentIndex(2)

    # ================================================================ dialogs
    def open_calibration(self):
        sp = self._spectrum_of_current()

        def acquire(seconds):
            site = self._current
            kw = {"live_time": True, "title": "calibration"}
            if site is not None and site.kind == "point":
                kw["at_um"] = site.params
            elif site is not None and site.kind in ("rect", "ellipse", "field"):
                kw["region_um"], kw["shape"] = site.params, ("ellipse" if site.kind == "ellipse" else "rect")
            elif self.image is not None:
                kw["region_um"] = self.image.region_um
            return self.ctrl.acquire_spectrum(seconds, **kw)

        d = CalibrationDialog(sp, self.ctrl.energy_calibration, acquire=acquire, parent=self)
        d.calibrationApplied.connect(self._calibration_applied)
        self._calib_dialog = d
        d.show()

    def _calibration_applied(self, cal, fwhm_ev, fano):
        self.ctrl.set_energy_calibration(cal)
        self.settings.energy_calibration = cal
        self.settings.calibration_source = f"{self.ctrl.mca.name} {self.ctrl.mca.info().get('serial', '')}"
        if np.isfinite(fwhm_ev) and 90 < fwhm_ev < 400:
            self.settings.fwhm_mn_ev = round(float(fwhm_ev), 1)
            self.c_fwhm.setValue(self.settings.fwhm_mn_ev)
        if np.isfinite(fano):
            self.settings.fano = float(fano)
        self._save_settings()
        self.notify(f"energy calibration applied: offset {cal.offset_kev * 1e3:+.2f} eV, "
                    f"gain {cal.gain_kev * 1e3:.4f} eV/ch; FWHM@Mn {self.settings.fwhm_mn_ev:g} eV")
        self._show_current()

    def open_processor(self):
        d = ProcessorDialog(self.ctrl.mca, self)
        d.channelsChanged.connect(lambda o, n: self.notify(
            f"MCA channels {o} → {n}: the energy calibration changed – recalibrate (Tools → Energy calibration)"))
        d.exec()

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
        site.spectrum = sp
        sp.title = sp.title or os.path.basename(path)
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
    def _tick(self):
        for ev in self._sub.drain():
            self._event(ev)
        job = self._job
        if job is not None and job.done():
            self._job = None
            self._job_done(job)
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
                self.l_rates.setText(f"ICR {sp.icr / 1e3:.1f} k  OCR {sp.ocr / 1e3:.1f} kcps")
                self.l_dead.setValue(int(round(100 * sp.dead_fraction)))
        elif ev.kind == "image":
            im = d.get("image")
            if im is not None:
                self.image = im
                self.imgview.set_image(im)
        elif ev.kind == "pixel":
            sm = d.get("map")
            if sm is not None:
                self.m_progress.setValue(int(100 * d.get("progress", 0)))
                if time.monotonic() - self._map_live_t > 1.0 and sm.se is not None:
                    self._map_live_t = time.monotonic()
                    se = np.where(sm.done, sm.se, np.nan)
                    if np.isfinite(se).any() and sm.shape[0] > 1:
                        self.imgview.set_overlay(np.nan_to_num(se - np.nanmin(se)), sm.region_um, (120, 200, 255))
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
                res.beam_kv, res.takeoff_deg = self.ctrl.beam_kv, self.ctrl.takeoff_deg
                if site is self._current:
                    self._show_current()
                if self.sp_autoid.isChecked() and site is self._current and res.total > 200 and not job.cancelled:
                    self.analyse(autoid=True)
        elif job.kind == "image" and isinstance(res, ElectronImage):
            self.image = res
            self.imgview.set_image(res)
        elif job.kind == "map" and isinstance(res, SpectrumMap):
            self.map = res
            self._map_done(res)
        elif job.kind == "linescan" and isinstance(res, SpectrumMap):
            self.map = res
            self._show_linescan(res)
        self.p_job.setValue(1000)

    def _map_done(self, sm: SpectrumMap):
        self.imgview.set_overlay(None)
        self.m_progress.setValue(int(100 * sm.done.mean()))
        self.imgview.add_site("mapregion", "map", sm.region_um, "#ffffff", "map")
        oh = sm.metadata.get("overhead_s")
        self.notify(f"map {sm.shape[1]}×{sm.shape[0]}: {int(sm.done.sum())} px, "
                    f"{sm.sum_spectrum().total:,} counts" + (f", overhead {oh * 1e3:.0f} ms/px" if oh else ""))
        self.compute_maps()

    def _update_status(self):
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
        self._save_settings()
        if self.ctrl.busy:
            self.ctrl.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._sub.close()
        if self.owns_controller:
            self.ctrl.close()
        super().closeEvent(ev)
