"""The easyEBL entry point: :class:`EasyEBLApp` owns the :class:`~easyebl.app.backend.SystemBackend`
and the one window, :class:`~easyebl.app.mainwindow.MainWindow` (every subsystem in it: SEM, EDS
and Lithography tabs, the microscope side bar).  The program runs in the foreground: closing the
window shuts easyEBL down and disconnects the hardware (it asks first).

* :class:`HardwareDialog` – *Hardware → Configure…*: the devices and the DAQ wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .backend import COMPONENTS, SystemBackend
from .config import BAUD_RATES, SystemConfig

__all__ = ["EasyEBLApp", "HardwareDialog", "SUBSYSTEMS", "make_icon"]

STATE_COLORS = {"ok": "#4c4", "sim": "#6af", "waiting": "#fb4", "warn": "#fb4", "error": "#f55", "off": "#777"}


@dataclass(frozen=True)
class Subsystem:
    key: str
    title: str
    needs: str  # backend component it uses ("daq", "eds", "microscope")


SUBSYSTEMS = (
    Subsystem("sem", "SEM imaging", "daq"),
    Subsystem("eds", "EDS", "eds"),
    Subsystem("microscope", "Microscope (column && stage)", "microscope"),
    Subsystem("litho", "Lithography", "daq"),
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
        self.mic_mx = QtWidgets.QCheckBox("Mirror X: moving the stage to +X moves the SEM image of the sample to +x")
        self.mic_my = QtWidgets.QCheckBox("Mirror Y: moving the stage to +Y moves the SEM image of the sample to +y")
        self.mic_mx.setChecked(m.stage_mirror_x)
        self.mic_my.setChecked(m.stage_mirror_y)
        for w in (self.mic_mx, self.mic_my):
            w.setToolTip("stage axes against the SEM image, used by the lithography (S-3000: mirror X on, Y off)")
        f.addRow("Stage axes", self.mic_mx)
        f.addRow("", self.mic_my)
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
        mic = replace(b.microscope, port=self.port(), baudrate=int(self.mic_baud.currentData()),
                      stage_mirror_x=self.mic_mx.isChecked(), stage_mirror_y=self.mic_my.isChecked())
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
        self.mic_mx.setChecked(c.microscope.stage_mirror_x)
        self.mic_my.setChecked(c.microscope.stage_mirror_y)


# ==================================================================== menu window
class EasyEBLApp(QtCore.QObject):
    """The backend and the main window of one session (every subsystem is a part of it)."""

    STATUS_MS = 1000

    def __init__(self, config: SystemConfig, persistent: bool = True, eds_settings_path: str | None = None,
                 microscope_settings=None, microscope_delay_s: float = 0.25,
                 confirm: Callable[[str, str], bool] | None = None, layout_path: str | None = None):
        super().__init__()
        from .mainwindow import MainWindow

        self.config = config
        self.persistent = persistent
        self._backend_kw = dict(eds_settings_path=eds_settings_path, microscope_settings=microscope_settings,
                                microscope_delay_s=microscope_delay_s)
        self.confirm = confirm or self._ask
        self.closing = False
        qapp = QtWidgets.QApplication.instance()
        qapp.setQuitOnLastWindowClosed(False)  # the program ends with shutdown(), not with a window
        qapp.setWindowIcon(make_icon())
        self.backend = SystemBackend(config, **self._backend_kw)
        from ..hal.scanmonitor import ScanTap

        self.scan_tap = ScanTap()  # outlives backend restarts; attached to each new DAQ backend
        self.main = MainWindow(self, layout_path)  # every subsystem in one window
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_status)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "EasyEBLApp":
        self.backend.start()
        self.main.build()
        self.main.restore_layout()
        self._wire_backend()
        self.refresh_status()
        self.timer.start(self.STATUS_MS)
        self.show_main()
        return self

    def _wire_backend(self):
        if self.backend.manager is not None:
            self.backend.manager.backend.output_tap = self.scan_tap
        d = self.config.daq
        if self.main.monitor is not None:  # same coordinates as the images
            self.main.monitor.set_orientation(d.invert_x, d.invert_y, d.swap_xy)

    def shutdown(self, ask: bool = True) -> bool:
        """Close the window and disconnect all hardware; the program ends."""
        if ask:
            busy = self.backend.busy
            msg = "Shut down easyEBL?\n\nThe window closes and the hardware is disconnected."
            if busy:
                msg += "\n\nStill running: " + ", ".join(busy) + " (it will be stopped)."
            if not self.confirm("Shut down easyEBL", msg):
                return False
        if not self.main.teardown():  # e.g. unsaved data kept
            return False
        self.closing = True
        self.timer.stop()
        self.backend.close()
        self.main.close()
        self.shut_down.emit()
        QtWidgets.QApplication.instance().quit()
        return True

    shut_down = QtCore.Signal()

    def restart_backend(self, config: SystemConfig | None = None, ask: bool = True) -> bool:
        """Reconnect everything (with a new configuration): the subsystems are closed and built again."""
        if ask:
            what = ", ".join(self.backend.busy) or "every subsystem"
            if not self.confirm("Reconnect hardware", f"Reconnecting closes {what}. Continue?"):
                return False
        if not self.main.teardown():
            return False
        self.backend.close()
        if config is not None:
            self.config = config
            if self.persistent:
                try:
                    config.save()
                except OSError as e:
                    QtWidgets.QMessageBox.warning(self.main, "easyEBL", f"Configuration not saved: {e}")
        self.backend = SystemBackend(self.config, **self._backend_kw)
        self.backend.start()
        self.main.build()
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
        d = HardwareDialog(self.config, self.main, persistent=self.persistent)
        if d.exec() == QtWidgets.QDialog.Accepted:
            return self.restart_backend(d.config())
        return False

    # ------------------------------------------------------------------ status
    def refresh_status(self):
        st = self.backend.status()
        self.main.update_status(st, self.config)
        return st

    def available(self, key: str, status: dict | None = None) -> tuple[bool, str]:
        """(can be built, why not) for a subsystem."""
        sub = _BY_KEY[key]
        st = (status or self.backend.status())[sub.needs]
        if not st.usable:
            return False, f"{COMPONENTS[sub.needs]}: {st.text}"
        return True, ""

    # ------------------------------------------------------------------ the window
    def show_main(self):
        self.main.bring_up()
        return self.main

    def open(self, key: str):
        """Show subsystem *key*: its tab (SEM, EDS, Lithography) or the microscope side bar."""
        if key in ("sem", "eds", "litho"):
            return self.main.show_tab(key)
        if key == "microscope":
            return self.main.show_microscope()
        raise KeyError(key)

    def _create(self, key: str) -> QtWidgets.QWidget:
        """The subsystem *key* (``sem``, ``eds``, ``litho``) on the backend, for its tab."""
        b = self.backend
        sim = self.config.daq.simulated
        if key == "sem":
            from ..imaging.sem.window import SEMWindow

            w = SEMWindow(b.engine, owns_engine=False, simulated=sim, microscope_bundle=b.microscope)
            w.stage_mirror = (b.config.microscope.stage_mirror_x, b.config.microscope.stage_mirror_y)
            return w
        if key == "eds":
            from ..eds.gui.window import EDSWindow

            return EDSWindow(b.eds, b.eds_settings, sem_engine=b.engine, simulated=sim, owns_controller=False,
                             microscope_bundle=b.microscope)
        if key == "litho":
            from ..litho.module import LithoWindow
            from ..litho.module.hardware import BackendHardware

            def hardware(win):
                from ..litho.coords import StageAxes

                mc = b.config.microscope
                return BackendHardware(b.engine, b.manager, b.microscope, win.bridge, open_sem=self._sem_for_litho,
                                       parent_widget=win, axes=StageAxes(mc.stage_mirror_x, mc.stage_mirror_y))

            return LithoWindow(hardware)
        raise KeyError(key)

    def _sem_for_litho(self):
        """The SEM the lithography picks markers in: the SEM tab's."""
        return self.main.sem_for_litho()

    # ------------------------------------------------------------------ misc
    def _ask(self, title: str, text: str) -> bool:
        return QtWidgets.QMessageBox.question(self.main, title, text) == QtWidgets.QMessageBox.Yes

    def handle_command(self, cmd: str):
        """A message from a second ``easyebl`` start: ``show`` or ``open <key>[,<key>…]``."""
        cmd = cmd.strip()
        self.show_main()
        if cmd.startswith("open "):
            for k in cmd[5:].split(","):
                if k.strip() in _BY_KEY:
                    self.open(k.strip())
