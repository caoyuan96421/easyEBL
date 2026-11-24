"""The easyEBL entry point: a small menu window, a tray icon and the subsystem windows.

* :class:`EasyEBLApp` owns the :class:`~easyebl.app.backend.SystemBackend` and opens
  one window per subsystem (SEM imaging, EDS, microscope, lithography and debug tools).
* :class:`MenuWindow` shows whether each piece of hardware is connected; its menus
  configure the hardware and open the subsystems.  Closing it only hides it.
* The tray icon (right click) brings the menu back, opens subsystems directly, and
  is the place to shut the whole system down.

Nothing is disconnected when windows close: only *Shut down easyEBL* stops the backend.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .backend import COMPONENTS, ComponentStatus, SystemBackend
from .config import BAUD_RATES, SystemConfig

__all__ = ["EasyEBLApp", "MenuWindow", "HardwareDialog", "SUBSYSTEMS", "make_icon"]

STATE_COLORS = {"ok": "#4c4", "sim": "#6af", "waiting": "#fb4", "warn": "#fb4", "error": "#f55", "off": "#777"}
STATE_WORDS = {"ok": "connected", "sim": "simulator", "waiting": "waiting", "warn": "check", "error": "failed",
               "off": "not used"}


@dataclass(frozen=True)
class Subsystem:
    key: str
    title: str
    needs: str  # backend component it uses ("daq", "eds", "microscope")
    tool: bool = False  # listed under "Tools"


SUBSYSTEMS = (
    Subsystem("sem", "SEM imaging", "daq"),
    Subsystem("eds", "EDS", "eds"),
    Subsystem("microscope", "Microscope (column && stage)", "microscope"),
    Subsystem("litho", "Lithography", "daq"),
    Subsystem("vector", "Vector lithography", "daq", tool=True),
    Subsystem("gds", "GDS lithography", "daq", tool=True),
    Subsystem("dose", "Dose calculator", "daq", tool=True),
    Subsystem("alignment", "Alignment", "daq", tool=True),
    Subsystem("raster", "Raster imaging (debug)", "daq", tool=True),
    Subsystem("hal", "HAL debug", "daq", tool=True),
)
_BY_KEY = {s.key: s for s in SUBSYSTEMS}


def make_icon(state: str | None = None) -> QtGui.QIcon:
    """The easyEBL icon: an electron-beam spot; a small status dot in one corner."""
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64):
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = size / 64.0
        grad = QtGui.QRadialGradient(32 * r, 32 * r, 30 * r)
        grad.setColorAt(0.0, QtGui.QColor("#e8f4ff"))
        grad.setColorAt(0.35, QtGui.QColor("#4aa3ff"))
        grad.setColorAt(1.0, QtGui.QColor("#123a6b"))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor("#0b2442"), max(1.0, 2 * r)))
        p.drawEllipse(QtCore.QRectF(3 * r, 3 * r, 58 * r, 58 * r))
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), max(1.0, 4 * r)))
        p.drawLine(QtCore.QPointF(32 * r, 6 * r), QtCore.QPointF(32 * r, 26 * r))  # the beam
        if state in STATE_COLORS and state not in ("ok", "sim"):
            p.setBrush(QtGui.QColor(STATE_COLORS[state]))
            p.setPen(QtGui.QPen(QtGui.QColor("#111"), max(1.0, 2 * r)))
            p.drawEllipse(QtCore.QRectF(38 * r, 38 * r, 24 * r, 24 * r))
        p.end()
        icon.addPixmap(pm)
    return icon


def _overall(status: dict[str, ComponentStatus]) -> str:
    states = {s.state for s in status.values()}
    for s in ("error", "warn", "waiting"):
        if s in states:
            return s
    return "ok"


# ==================================================================== tool windows
class ToolWindow(QtWidgets.QMainWindow):
    """A lithography / debug widget as a window of its own; its DAQ handles close with it."""

    def __init__(self, widget: QtWidgets.QWidget, title: str):
        super().__init__()
        self.widget = widget
        self.setCentralWidget(widget)
        self.setWindowTitle(f"easyEBL – {title}")

    def closeEvent(self, ev):  # noqa: N802
        from ..hal.manager import DAQHandle

        self.widget.close()  # its own clean-up (stop streams, release external control)
        # every DAQ handle the tool holds: on its widgets, also panels kept as attributes only
        todo, seen = [self.widget, *self.widget.findChildren(QtWidgets.QWidget)], set()
        while todo:
            w = todo.pop()
            if id(w) in seen:
                continue
            seen.add(id(w))
            for v in list(vars(w).values()):
                if isinstance(v, DAQHandle):
                    if not v.closed:
                        v.close()
                elif isinstance(v, QtWidgets.QWidget):
                    todo.append(v)
        super().closeEvent(ev)


# ==================================================================== hardware dialog
class HardwareDialog(QtWidgets.QDialog):
    """Edit a :class:`SystemConfig`: scan DAQ, EDS pulse processor, microscope port (*Devices* tab)
    and the DAQ signal → channel map (*Wiring* tab, :class:`~easyebl.app.wiring_gui.WiringEditor`)."""

    def __init__(self, config: SystemConfig, parent=None, persistent: bool = True):
        super().__init__(parent)
        self.setWindowTitle("easyEBL – hardware")
        self._base = config
        lay = QtWidgets.QVBoxLayout(self)
        if not persistent:
            note = QtWidgets.QLabel("Started with --sim: changes apply to this session and are not saved.")
            note.setStyleSheet("color:#fb4")
            lay.addWidget(note)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)
        devices = QtWidgets.QWidget()
        dlay = QtWidgets.QVBoxLayout(devices)
        self.tabs.addTab(devices, "Devices")

        # --- scan DAQ
        d = config.daq
        g = QtWidgets.QGroupBox("Scan DAQ (deflection, blanker, video)")
        f = QtWidgets.QFormLayout(g)
        self.daq_backend = QtWidgets.QComboBox()
        self.daq_backend.addItem("NI-DAQmx device", "nidaqmx")
        self.daq_backend.addItem("Simulator", "sim")
        self.daq_backend.setCurrentIndex(max(0, self.daq_backend.findData(d.backend)))
        self.daq_device = QtWidgets.QLineEdit(d.device)
        self.daq_device.setPlaceholderText("Dev1")
        self.daq_preset = QtWidgets.QComboBox()
        self.daq_preset.addItem("USB-6363 wiring", "usb6363")
        self.daq_preset.addItem("Generic default", "default")
        self.daq_preset.setCurrentIndex(max(0, self.daq_preset.findData(d.preset)))
        self.daq_file = QtWidgets.QLineEdit(d.config_file)
        self.daq_file.setPlaceholderText("optional DAQConfig JSON (replaces the preset)")
        b = QtWidgets.QToolButton()
        b.setText("…")
        b.clicked.connect(self._browse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.daq_file, 1)
        row.addWidget(b)
        self.daq_rate = QtWidgets.QDoubleSpinBox()
        self.daq_rate.setRange(0, 10000)
        self.daq_rate.setDecimals(0)
        self.daq_rate.setSuffix(" kS/s")
        self.daq_rate.setSpecialValueText("device maximum")
        self.daq_rate.setValue((d.rate or 0) / 1e3)
        self.daq_latency = QtWidgets.QDoubleSpinBox()
        self.daq_latency.setRange(5, 2000)
        self.daq_latency.setDecimals(0)
        self.daq_latency.setSuffix(" ms")
        self.daq_latency.setValue(d.latency_ms)
        self.daq_latency.setToolTip("output buffer depth = how long a setting change takes to reach the beam")
        f.addRow("Device", self.daq_backend)
        f.addRow("NI device name", self.daq_device)
        f.addRow("Wiring", self.daq_preset)
        f.addRow("Config file", row)
        f.addRow("Sample clock", self.daq_rate)
        f.addRow("Buffer latency", self.daq_latency)
        self.daq_inv_x = QtWidgets.QCheckBox("Invert X")
        self.daq_inv_y = QtWidgets.QCheckBox("Invert Y")
        self.daq_swap = QtWidgets.QCheckBox("Swap X / Y")
        self.daq_inv_x.setChecked(bool(d.invert_x))
        self.daq_inv_y.setChecked(bool(d.invert_y))
        self.daq_swap.setChecked(bool(d.swap_xy))
        self.daq_swap.setToolTip("image X drives the Y deflection and image Y the X deflection\n"
                                 "(the inversions refer to the outputs, after the swap)")
        row2 = QtWidgets.QHBoxLayout()
        for cb in (self.daq_inv_x, self.daq_inv_y, self.daq_swap):
            row2.addWidget(cb)
        row2.addStretch(1)
        f.addRow("Scan orientation", row2)
        dlay.addWidget(g)
        self._daq_hw = (self.daq_device,)

        # --- EDS
        g = QtWidgets.QGroupBox("EDS pulse processor")
        f = QtWidgets.QFormLayout(g)
        self.eds_mca = QtWidgets.QComboBox()
        self.eds_mca.addItem("Amptek DP5 (USB)", "dp5")
        self.eds_mca.addItem("DP5 mock device (simulator)", "mock")
        self.eds_mca.addItem("Not used", "none")
        self.eds_mca.setCurrentIndex(max(0, self.eds_mca.findData(config.eds.mca)))
        self.eds_external = QtWidgets.QCheckBox("Switch the SEM to external scan for EDS jobs")
        self.eds_external.setChecked(config.eds.external_control)
        f.addRow("Device", self.eds_mca)
        f.addRow("", self.eds_external)
        dlay.addWidget(g)

        # --- microscope
        m = config.microscope
        g = QtWidgets.QGroupBox("Microscope remote control (Hitachi S-3000, RS-232)")
        f = QtWidgets.QFormLayout(g)
        self.mic_port = QtWidgets.QComboBox()
        self.mic_port.setEditable(True)
        self.mic_port.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        b = QtWidgets.QToolButton()
        b.setText("↻")
        b.setToolTip("list the serial ports again")
        b.clicked.connect(lambda: self._fill_ports(self.port()))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.mic_port, 1)
        row.addWidget(b)
        self.mic_baud = QtWidgets.QComboBox()
        for r in BAUD_RATES:
            self.mic_baud.addItem(f"{r} baud", r)
        self.mic_baud.setCurrentIndex(max(0, self.mic_baud.findData(m.baudrate)))
        self.mic_baud.setToolTip("the S-3000 uses 9600 baud, 8N1")
        f.addRow("Port", row)
        f.addRow("Baud rate", self.mic_baud)
        dlay.addWidget(g)
        dlay.addStretch(1)
        self._fill_ports(m.port)

        # --- wiring (the one place the DAQ channels are edited)
        from .wiring_gui import WiringEditor

        page = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(page)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("NI USB-6363: which signal is on which channel."))
        top.addStretch(1)
        b = QtWidgets.QPushButton("Reset to preset")
        b.setToolTip("the wiring of the selected preset (or DAQ config file)")
        b.clicked.connect(self.reset_wiring)
        top.addWidget(b)
        wl.addLayout(top)
        self.wiring = WiringEditor(config.effective_wiring() if self._safe_base() else self._fallback_wiring(),
                                   d.ai_terminal, self._preset_terminal(), ai_range_v=d.ai_range_v,
                                   preset_range_v=self._preset_range())
        wl.addWidget(self.wiring, 1)
        self.tabs.addTab(page, "Wiring")

        # --- buttons
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("All simulators")
        b.setToolTip("simulated DAQ, DP5 mock device and S-3000 simulator")
        b.clicked.connect(lambda: self.set_config(self.config().as_simulated()))
        row.addWidget(b)
        row.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._ok = bb.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok.setText("Apply && reconnect")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)
        self.daq_backend.currentIndexChanged.connect(self._sync)
        self.daq_preset.currentIndexChanged.connect(self.reset_wiring)
        self.daq_file.editingFinished.connect(self._file_changed)
        self._file_seen = self.daq_file.text().strip()
        self.wiring.changed.connect(self._sync_wiring)
        self._sync()
        self._sync_wiring()
        self.resize(620, 720)

    # ---------------------------------------------------------------- helpers
    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "DAQ configuration", self.daq_file.text(),
                                                        "JSON (*.json);;All files (*)")
        if path:
            self.daq_file.setText(path)

    def _fill_ports(self, current: str):
        self.mic_port.clear()
        self.mic_port.addItem("Simulator", "sim")
        self.mic_port.addItem("Not used", "none")
        try:
            from serial.tools import list_ports

            for p in sorted(list_ports.comports(), key=lambda p: p.device):
                self.mic_port.addItem(f"{p.device} – {p.description}", p.device)
        except Exception:  # noqa: BLE001 - pyserial missing: the port can still be typed
            pass
        i = self.mic_port.findData(current)
        if i < 0:
            self.mic_port.addItem(current, current)
            i = self.mic_port.count() - 1
        self.mic_port.setCurrentIndex(i)

    def port(self) -> str:
        i = self.mic_port.currentIndex()
        text = self.mic_port.currentText().strip()
        if i >= 0 and self.mic_port.itemText(i) == text:
            return self.mic_port.itemData(i)
        return text.split(" ")[0] if text else "none"  # typed by hand

    def _sync(self):
        hw = self.daq_backend.currentData() == "nidaqmx"
        for w in self._daq_hw:
            w.setEnabled(hw)

    # ---------------------------------------------------------------- wiring
    def _with_daq(self, **kw) -> SystemConfig:
        out = replace(self._base, daq=replace(self._base.daq, **kw))
        out.path = self._base.path
        return out

    def _selected_base(self) -> SystemConfig:
        return self._with_daq(preset=self.daq_preset.currentData(), config_file=self.daq_file.text().strip(),
                              ai_terminal=None)

    def _safe_base(self) -> bool:
        try:
            self._base.base_daq_config()
            return True
        except (OSError, ValueError, TypeError):
            return False

    def _fallback_wiring(self):
        from ..hal.wiring import Wiring

        return self._base.wiring.copy() if self._base.wiring is not None else Wiring()

    def _preset_terminal(self) -> str:
        try:
            return (self._selected_base() if hasattr(self, "daq_preset") else
                    self._with_daq(ai_terminal=None)).base_daq_config().ai_terminal
        except (OSError, ValueError, TypeError):
            return "RSE"

    def _preset_range(self) -> float:
        try:
            return self._with_daq(preset=self.daq_preset.currentData(), config_file=self.daq_file.text().strip(),
                                  ai_range_v=None).base_daq_config().ai_range_v
        except (OSError, ValueError, TypeError):
            return 10.0

    def preset_wiring(self):
        """The wiring of the selected preset / config file (None if the file cannot be read)."""
        try:
            return self._selected_base().preset_wiring()
        except (OSError, ValueError, TypeError):
            return None

    def reset_wiring(self):
        w = self.preset_wiring()
        self.wiring.set_preset_terminal(self._preset_terminal())
        self.wiring.set_preset_range(self._preset_range())
        if w is not None:
            self.wiring.set_wiring(w, keep_terminal=True)

    def _file_changed(self):
        if self.daq_file.text().strip() != self._file_seen:
            self._file_seen = self.daq_file.text().strip()
            self.reset_wiring()

    def _sync_wiring(self):
        n = len(self.wiring.issues())
        i = self.tabs.indexOf(self.wiring.parentWidget())
        self.tabs.setTabText(i, f"Wiring ⚠ {n}" if n else "Wiring")
        self._ok.setEnabled(n == 0)
        self._ok.setToolTip("fix the wiring conflicts first" if n else "")

    def wiring_value(self):
        """The wiring to save: None while it equals the preset's and the file had no wiring section."""
        w = self.wiring.wiring()
        if self._base.wiring is None and w == self.preset_wiring():
            return None
        return w

    # ---------------------------------------------------------------- value
    def config(self) -> SystemConfig:
        b = self._base
        daq = replace(b.daq, backend=self.daq_backend.currentData(), device=self.daq_device.text().strip() or "Dev1",
                      preset=self.daq_preset.currentData(), config_file=self.daq_file.text().strip(),
                      rate=(self.daq_rate.value() * 1e3) or None, latency_ms=float(self.daq_latency.value()),
                      ai_terminal=self.wiring.selected_terminal(), ai_range_v=self.wiring.selected_range(),
                      invert_x=self.daq_inv_x.isChecked(),
                      invert_y=self.daq_inv_y.isChecked(), swap_xy=self.daq_swap.isChecked())
        eds = replace(b.eds, mca=self.eds_mca.currentData(), external_control=self.eds_external.isChecked())
        mic = replace(b.microscope, port=self.port(), baudrate=int(self.mic_baud.currentData()))
        out = SystemConfig(daq, eds, mic, b.monitor, self.wiring_value())
        out.path = b.path
        return out

    def set_config(self, c: SystemConfig):
        self.daq_backend.setCurrentIndex(max(0, self.daq_backend.findData(c.daq.backend)))
        if c.wiring is not None:
            self.wiring.set_wiring(c.wiring, c.daq.ai_terminal)
        self.wiring._set_range(c.daq.ai_range_v)
        self.eds_mca.setCurrentIndex(max(0, self.eds_mca.findData(c.eds.mca)))
        self._fill_ports(c.microscope.port)


# ==================================================================== menu window
class MenuWindow(QtWidgets.QMainWindow):
    """Small status window: one line per piece of hardware, menus for everything else."""

    def __init__(self, app: "EasyEBLApp"):
        super().__init__()
        self.app = app
        self.setWindowTitle("easyEBL")
        self.setWindowIcon(make_icon())
        w = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QWidget()
        outer.addWidget(top)
        grid = QtWidgets.QGridLayout(top)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        self.rows: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        for r, (key, name) in enumerate(COMPONENTS.items()):
            dot = QtWidgets.QLabel("●")
            dot.setFixedWidth(14)
            label = QtWidgets.QLabel(name)
            label.setStyleSheet("font-weight:bold")
            text = QtWidgets.QLabel("—")
            text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            text.setMinimumWidth(240)
            grid.addWidget(dot, r, 0)
            grid.addWidget(label, r, 1)
            grid.addWidget(text, r, 2)
            self.rows[key] = (dot, text)
        self.l_config = QtWidgets.QLabel()
        self.l_config.setStyleSheet("color:#9aa")
        grid.addWidget(self.l_config, len(COMPONENTS), 0, 1, 3)
        # --- scan monitor: X/Y/blanker as an oscilloscope in XY mode with persistence
        from ..gui.scanmonitor import ScanMonitorWidget

        mc = app.config.monitor
        self.monitor_box = QtWidgets.QGroupBox("Scan monitor (image coordinates, blanked samples hidden)")
        ml = QtWidgets.QVBoxLayout(self.monitor_box)
        d = app.config.daq
        self.monitor = ScanMonitorWidget(app.scan_tap, mc.resolution, mc.retention_s, mc.fade,
                                         on_settings=app.set_monitor_settings,
                                         orientation=(d.invert_x, d.invert_y, d.swap_xy))
        ml.addWidget(self.monitor)
        outer.addWidget(self.monitor_box, 1)
        outer.setContentsMargins(0, 0, 8, 8)
        self.setCentralWidget(w)
        self._build_menus()
        self._compact = top.sizeHint().expandedTo(QtCore.QSize(460, 0))
        self.show_monitor(mc.visible, save=False)
        self._hint_shown = False

    def show_monitor(self, on: bool, save: bool = True):
        """Show or hide the scan monitor (the window is a fixed-size status panel without it)."""
        on = bool(on)
        self.monitor_box.setVisible(on)
        blocked = self.a_monitor.blockSignals(True)
        self.a_monitor.setChecked(on)
        self.a_monitor.blockSignals(blocked)
        self.statusBar().setSizeGripEnabled(on)
        if on:
            self.setMinimumSize(460, 0)
            self.setMaximumSize(16777215, 16777215)
            self.resize(max(self.width(), 480), 640)
        else:
            self.setFixedSize(self._compact.width(), self._compact.height() + self.menuBar().sizeHint().height()
                              + self.statusBar().sizeHint().height())
        if save:
            self.app.set_monitor_settings(visible=on)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&easyEBL")
        m.addAction("Hide to tray", self.close)
        m.addSeparator()
        self.a_shutdown = m.addAction("Shut down easyEBL…", self.app.shutdown)
        m = mb.addMenu("&Hardware")
        m.addAction("Configure…", self.app.configure)
        m.addAction("Reconnect all", self.app.reconnect)
        m = mb.addMenu("&View")
        self.a_monitor = m.addAction("Scan monitor")
        self.a_monitor.setCheckable(True)
        self.a_monitor.toggled.connect(self.show_monitor)
        m.addAction("Clear scan monitor", lambda: self.monitor.clear())
        self.m_open = mb.addMenu("&Open")
        self.app.fill_open_menu(self.m_open)

    def update_status(self, status: dict[str, ComponentStatus], config: SystemConfig):
        for key, (dot, text) in self.rows.items():
            s = status[key]
            dot.setStyleSheet(f"color:{STATE_COLORS.get(s.state, '#ccc')};font-size:15px")
            text.setText(s.text)
            text.setToolTip(f"{STATE_WORDS.get(s.state, s.state)}: {s.text}")
        mode = "simulators" if config.simulated else "instrument"
        self.l_config.setText(f"Configuration: {mode}" + ("" if self.app.persistent else " (--sim, not saved)")
                              + "  ·  Open a subsystem from the Open menu")
        self.statusBar().showMessage(time.strftime("updated %H:%M:%S"))

    def closeEvent(self, ev):  # noqa: N802
        """Closing the menu hides it: the backend keeps running in the tray."""
        if self.app.closing:
            return super().closeEvent(ev)
        if not self.app.has_tray:
            ev.ignore()
            self.showMinimized()  # no tray to bring it back from
            return
        ev.ignore()
        self.hide()
        if not self._hint_shown:
            self._hint_shown = True
            self.app.tray_message("easyEBL is still running. Right-click the tray icon to open subsystems "
                                  "or to shut it down.")


# ==================================================================== application
class EasyEBLApp(QtCore.QObject):
    """The backend, the menu window, the tray icon and every subsystem window of one session."""

    STATUS_MS = 1000

    def __init__(self, config: SystemConfig, persistent: bool = True, eds_settings_path: str | None = None,
                 microscope_settings=None, microscope_delay_s: float = 0.25, use_tray: bool = True,
                 confirm: Callable[[str, str], bool] | None = None):
        super().__init__()
        self.config = config
        self.persistent = persistent
        self._backend_kw = dict(eds_settings_path=eds_settings_path, microscope_settings=microscope_settings,
                                microscope_delay_s=microscope_delay_s)
        self.confirm = confirm or self._ask
        self.closing = False
        self.windows: dict[str, QtWidgets.QWidget] = {}
        qapp = QtWidgets.QApplication.instance()
        qapp.setQuitOnLastWindowClosed(False)  # the backend outlives every window
        qapp.setWindowIcon(make_icon())
        self.backend = SystemBackend(config, **self._backend_kw)
        from ..hal.scanmonitor import ScanTap

        self.scan_tap = ScanTap()  # outlives backend restarts; attached to each new DAQ backend
        self.menu = MenuWindow(self)
        self.tray = None
        if use_tray and QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QtWidgets.QSystemTrayIcon(make_icon(), self)
            self.tray.setToolTip("easyEBL")
            tm = QtWidgets.QMenu()
            tm.aboutToShow.connect(lambda: self._fill_tray_menu(tm))
            self._fill_tray_menu(tm)
            self.tray.setContextMenu(tm)
            self._tray_menu = tm
            self.tray.activated.connect(self._tray_activated)
            self.tray.show()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_status)

    @property
    def has_tray(self) -> bool:
        return self.tray is not None

    # ------------------------------------------------------------------ lifecycle
    def start(self, show_menu: bool = True) -> "EasyEBLApp":
        self.backend.start()
        self._wire_backend()
        self.refresh_status()
        self.timer.start(self.STATUS_MS)
        if show_menu or not self.has_tray:
            self.show_menu()
        return self

    def _wire_backend(self):
        if self.backend.manager is not None:
            self.backend.manager.backend.output_tap = self.scan_tap
        d = self.config.daq
        self.menu.monitor.set_orientation(d.invert_x, d.invert_y, d.swap_xy)  # same coordinates as the images
        if self.backend.microscope is not None:
            self.backend.microscope.opener = lambda: self.open("microscope")

    def shutdown(self, ask: bool = True) -> bool:
        """Close every window and disconnect all hardware; the program ends."""
        if ask:
            busy = self.backend.busy
            msg = "Shut down easyEBL?\n\nAll windows close and the hardware is disconnected."
            if busy:
                msg += "\n\nStill running: " + ", ".join(busy) + " (it will be stopped)."
            if not self.confirm("Shut down easyEBL", msg):
                return False
        self.closing = True
        self.timer.stop()
        self.menu.monitor.stop()
        self.close_windows()
        self.backend.close()
        self.menu.close()
        if self.tray is not None:
            self.tray.hide()
        self.shut_down.emit()
        QtWidgets.QApplication.instance().quit()
        return True

    shut_down = QtCore.Signal()

    def close_windows(self):
        for key in list(self.windows):
            w = self.windows.pop(key, None)
            if w is not None:
                try:
                    w.close()
                except RuntimeError:  # already deleted
                    pass

    def restart_backend(self, config: SystemConfig | None = None, ask: bool = True) -> bool:
        """Reconnect everything (with a new configuration): the subsystem windows are closed first."""
        if ask and (self.windows or self.backend.busy):
            what = ", ".join(self.backend.busy) or "the subsystem windows"
            if not self.confirm("Reconnect hardware", f"Reconnecting closes {what}. Continue?"):
                return False
        self.close_windows()
        self.backend.close()
        if config is not None:
            self.config = config
            if self.persistent:
                try:
                    config.save()
                except OSError as e:
                    QtWidgets.QMessageBox.warning(self.menu, "easyEBL", f"Configuration not saved: {e}")
        self.backend = SystemBackend(self.config, **self._backend_kw)
        self.backend.start()
        self._wire_backend()
        self.refresh_status()
        return True

    def set_monitor_settings(self, resolution: int | None = None, retention_s: float | None = None,
                             fade: bool | None = None, visible: bool | None = None):
        """Remember the scan-monitor settings (saved with the configuration unless --sim)."""
        m = self.config.monitor
        if resolution is not None:
            m.resolution = int(resolution)
        if retention_s is not None:
            m.retention_s = float(retention_s)
        if fade is not None:
            m.fade = bool(fade)
        if visible is not None:
            m.visible = bool(visible)
        if self.persistent:
            try:
                self.config.save()
            except OSError:
                pass

    def reconnect(self):
        return self.restart_backend()

    def configure(self):
        d = HardwareDialog(self.config, self.menu if self.menu.isVisible() else None, persistent=self.persistent)
        if d.exec() == QtWidgets.QDialog.Accepted:
            return self.restart_backend(d.config())
        return False

    # ------------------------------------------------------------------ status
    def refresh_status(self):
        st = self.backend.status()
        self.menu.update_status(st, self.config)
        if self.tray is not None:
            state = _overall(st)
            if state != getattr(self, "_tray_state", None):
                self._tray_state = state
                self.tray.setIcon(make_icon(state))
            self.tray.setToolTip("easyEBL\n" + "\n".join(f"{COMPONENTS[k]}: {s.text}" for k, s in st.items()))
        self._sync_actions(st)
        return st

    def available(self, key: str, status: dict | None = None) -> tuple[bool, str]:
        """(can open, why not) for a subsystem."""
        sub = _BY_KEY[key]
        st = (status or self.backend.status())[sub.needs]
        if not st.usable:
            return False, f"{COMPONENTS[sub.needs]}: {st.text}"
        return True, ""

    # ------------------------------------------------------------------ menus
    def fill_open_menu(self, menu: QtWidgets.QMenu):
        self._open_actions = getattr(self, "_open_actions", [])
        tools = None
        for sub in SUBSYSTEMS:
            if sub.tool and tools is None:
                menu.addSeparator()
                tools = menu.addMenu("Tools")
            a = (tools or menu).addAction(sub.title, lambda k=sub.key: self.open(k))
            a.setData(sub.key)
            self._open_actions.append(a)

    def _sync_actions(self, status):
        for a in getattr(self, "_open_actions", []):
            try:
                ok, why = self.available(a.data(), status)
            except RuntimeError:
                continue
            a.setEnabled(ok)
            a.setToolTip(why)

    def _fill_tray_menu(self, tm: QtWidgets.QMenu):
        tm.clear()
        self._open_actions = [a for a in getattr(self, "_open_actions", []) if _alive(a) and a.parent() is not tm]
        tm.addAction("Show easyEBL menu", self.show_menu)
        tm.addSeparator()
        self.fill_open_menu(tm)
        tm.addSeparator()
        tm.addAction("Hardware…", self.configure)
        tm.addSeparator()
        tm.addAction("Shut down easyEBL…", self.shutdown)
        self._sync_actions(self.backend.status())

    def _tray_activated(self, reason):
        if reason in (QtWidgets.QSystemTrayIcon.Trigger, QtWidgets.QSystemTrayIcon.DoubleClick):
            self.show_menu()

    def tray_message(self, text: str):
        if self.tray is not None:
            self.tray.showMessage("easyEBL", text, make_icon(), 5000)

    def show_menu(self):
        self.menu.show()
        self.menu.setWindowState(self.menu.windowState() & ~QtCore.Qt.WindowMinimized)
        self.menu.raise_()
        self.menu.activateWindow()
        return self.menu

    # ------------------------------------------------------------------ subsystem windows
    def open(self, key: str):
        """Show the window of subsystem *key* (created once; again after it was closed)."""
        w = self.windows.get(key)
        if w is not None and _alive(w):
            w.show()
            w.setWindowState(w.windowState() & ~QtCore.Qt.WindowMinimized)
            w.raise_()
            w.activateWindow()
            return w
        ok, why = self.available(key)
        if not ok:
            QtWidgets.QMessageBox.information(self.menu if self.menu.isVisible() else None, "easyEBL",
                                              f"{_BY_KEY[key].title.replace('&&', '&')} is not available.\n\n{why}")
            return None
        try:
            w = self._create(key)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self.menu if self.menu.isVisible() else None, "easyEBL",
                                          f"Cannot open {_BY_KEY[key].title.replace('&&', '&')}:\n{type(e).__name__}: {e}")
            return None
        w.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        w.setWindowIcon(make_icon())
        w.destroyed.connect(lambda _=None, k=key, ww=w: self._forget(k, ww))
        self.windows[key] = w
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    def _forget(self, key, w):
        if self.windows.get(key) is w:
            self.windows.pop(key, None)

    def _create(self, key: str) -> QtWidgets.QWidget:
        b = self.backend
        sim = self.config.daq.simulated
        if key == "sem":
            from ..imaging.sem.window import SEMWindow

            w = SEMWindow(b.engine, owns_engine=False, simulated=sim, microscope_bundle=b.microscope)
            w.resize(1500, 950)
            return w
        if key == "eds":
            from ..eds.gui.window import EDSWindow

            w = EDSWindow(b.eds, b.eds_settings, sem_engine=b.engine, simulated=sim, owns_controller=False,
                          microscope_bundle=b.microscope)
            w.resize(1600, 980)
            return w
        if key == "microscope":
            from ..microscope.gui import MicroscopeWindow

            m = b.microscope
            w = MicroscopeWindow(m.link, m.sem, m.stage, m.settings, owns_link=False)
            w.resize(1200, 860)
            return w
        if key == "litho":
            from ..litho.module import LithoWindow
            from ..litho.module.hardware import BackendHardware

            def hardware(win):
                return BackendHardware(b.engine, b.manager, b.microscope, win.bridge, open_sem=lambda: self.open("sem"),
                                       parent_widget=win)

            w = LithoWindow(hardware)
            w.resize(1500, 950)
            return w
        from ..gui.common import Session

        session = Session(b.manager, b.engine.calibration)  # never closed here: the backend owns the manager
        title = _BY_KEY[key].title
        if key == "vector":
            from ..litho.gui_vector import VectorLithoWidget as cls
        elif key == "gds":
            from ..litho.gui_gds import GDSWidget as cls
        elif key == "dose":
            from ..litho.gui_dose import DoseCalculatorWidget as cls
        elif key == "alignment":
            from ..alignment.gui import AlignmentWidget as cls
        elif key == "raster":
            from ..imaging.gui import ImagingWidget as cls
        elif key == "hal":
            from ..hal.gui import HALDebugWidget as cls
        else:
            raise KeyError(key)
        widget = cls(session)
        if key == "alignment" and session.simulator is not None:
            widget._apply_sim()
        w = ToolWindow(widget, f"{title} [{b.manager.capabilities().name}]")
        w.resize(1300, 850)
        return w

    # ------------------------------------------------------------------ misc
    def _ask(self, title: str, text: str) -> bool:
        parent = self.menu if self.menu.isVisible() else None
        return QtWidgets.QMessageBox.question(parent, title, text) == QtWidgets.QMessageBox.Yes

    def handle_command(self, cmd: str):
        """A message from a second ``easyebl`` start: ``show`` or ``open <key>[,<key>…]``."""
        cmd = cmd.strip()
        if cmd.startswith("open "):
            for k in cmd[5:].split(","):
                if k.strip() in _BY_KEY:
                    self.open(k.strip())
        else:
            self.show_menu()


def _alive(obj) -> bool:
    try:
        from shiboken6 import isValid

        return isValid(obj)
    except ImportError:  # pragma: no cover
        return True
