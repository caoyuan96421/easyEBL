"""The easyEBL backend: every hardware connection, opened once per process (no Qt).

* the scan DAQ (:class:`~easyebl.hal.DAQManager`) and the SEM imaging engine on it;
* the EDS pulse processor and :class:`~easyebl.eds.EDSController` (same DAQ);
* the microscope's serial port (:class:`~easyebl.microscope.bundle.MicroscopeBundle`),
  with the imaging / EDS field following the SEM magnification.

The subsystem windows only borrow these objects: closing a window never
closes hardware, and the backend keeps running until :meth:`SystemBackend.close`.
A component that cannot be opened (DP5 unplugged, port in use …) is reported
by :meth:`SystemBackend.status`; the others work without it.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading

from .config import SystemConfig

__all__ = ["SystemBackend", "ComponentStatus", "COMPONENTS"]

#: component key -> display name
COMPONENTS = {"daq": "Scan DAQ", "eds": "EDS pulse processor", "microscope": "Microscope"}


@dataclass(frozen=True)
class ComponentStatus:
    #: "ok" (hardware connected), "sim" (simulator running), "waiting" (not there yet / reconnecting),
    #: "warn" (connected, needs attention), "error" (could not be opened), "off" (not configured)
    state: str
    text: str

    @property
    def usable(self) -> bool:
        return self.state in ("ok", "sim", "waiting", "warn")


class SystemBackend:
    def __init__(self, config: SystemConfig, eds_settings_path: str | None = None, microscope_settings=None,
                 microscope_delay_s: float = 0.25):
        self.config = config
        self._eds_settings_path = eds_settings_path
        self._mic_settings = microscope_settings
        self._mic_delay = microscope_delay_s
        self.manager = self.engine = None
        self.mca = self.eds = self.eds_settings = None
        self.microscope = None  # MicroscopeBundle (a stub bundle when not configured)
        self.errors: dict[str, str] = {}
        self.started = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ start / stop
    def start(self) -> "SystemBackend":
        """Open everything the configuration names; failures are recorded per component."""
        with self._lock:
            if self.started:
                return self
            self.started = True
            self._open_daq()
            self._open_eds()
            self._open_microscope()
            return self

    def _fail(self, key: str, e: BaseException) -> None:
        self.errors[key] = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__

    def _open_daq(self) -> None:
        from ..core.calibration import FieldCalibration
        from ..hal import DAQConfig, DAQManager, SimParams, SimulatedBackend, create_backend
        from ..imaging.engine import EngineOptions, ImagingEngine

        c = self.config.daq
        try:
            if c.config_file:
                with open(c.config_file, encoding="utf-8") as fh:
                    cfg = DAQConfig.from_json(fh.read())
            else:
                cfg = DAQConfig.usb6363() if c.preset == "usb6363" else DAQConfig()
            if c.device:
                cfg.device = c.device
            if c.ai_terminal:
                cfg.ai_terminal = c.ai_terminal
            cal = FieldCalibration(float(c.field_um), cfg.ao_limit_v, invert_x=bool(c.invert_x),
                                   invert_y=bool(c.invert_y), swap_xy=bool(c.swap_xy))
            if c.simulated:
                be = SimulatedBackend(cfg, SimParams(field_size_um=float(c.field_um), v_full_scale=cfg.ao_limit_v,
                                                     realtime=True, noise_sigma=0.05))
            else:
                be = create_backend(c.backend, cfg)
            # a missing / unplugged device is not fatal: the HAL connects as soon as it appears
            self.manager = DAQManager(be).open(allow_missing=True)
            rate = c.rate if c.rate else (500e3 if c.simulated else None)  # the simulator is lighter at 500 kS/s
            opts = EngineOptions(rate=rate, latency_s=max(0.005, float(c.latency_ms) / 1e3))
            self.engine = ImagingEngine(self.manager, cal, options=opts)
        except Exception as e:  # noqa: BLE001 - reported in the menu
            self._fail("daq", e)
            if self.manager is not None:
                self.manager.close()
            self.manager = self.engine = None

    def _open_eds(self) -> None:
        kind = self.config.eds.mca
        if kind == "none" or self.manager is None:
            if kind != "none":
                self.errors["eds"] = "needs the scan DAQ"
            return
        from ..eds.controller import EDSController, EDSOptions
        from ..eds.settings import EDSSettings
        from ..hal.mca import create_pulse_processor

        try:
            self.eds_settings = EDSSettings.load(self._eds_settings_path)
            self.mca = create_pulse_processor(kind).open()
            s = self.eds_settings
            self.eds = EDSController(self.manager, self.mca, self.engine.calibration, s.energy_calibration,
                                     s.beam_kv, s.takeoff_deg,
                                     options=EDSOptions(external_control=self.config.eds.external_control),
                                     owner="eds")
        except Exception as e:  # noqa: BLE001
            self._fail("eds", e)
            if self.mca is not None:
                try:
                    self.mca.close()
                except Exception:  # noqa: BLE001
                    pass
            self.mca = self.eds = None

    def _open_microscope(self) -> None:
        from ..microscope.bundle import MicroscopeBundle, open_bundle

        m = self.config.microscope
        try:
            self.microscope = open_bundle(m.port, m.baudrate, self._mic_settings, command_delay_s=self._mic_delay)
        except Exception as e:  # noqa: BLE001
            self._fail("microscope", e)
            self.microscope = MicroscopeBundle(settings=self._mic_settings)
            return
        targets = [t for t in (self.engine, self.eds) if t is not None]
        sim_be = self.manager.backend if self.manager is not None and self.config.daq.simulated else None
        if self.microscope.connected and targets:
            self.microscope.follow_magnification(targets, sim_backend=sim_be)

    def close(self) -> None:
        """Stop and disconnect everything (the end of the easyEBL session)."""
        with self._lock:
            if self.microscope is not None:
                self.microscope.close()
            for obj in (self.eds, self.engine, self.mca, self.manager):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:  # noqa: BLE001 - closing as much as possible
                        pass
            self.manager = self.engine = self.mca = self.eds = None
            self.started = False

    # ------------------------------------------------------------------ status
    @property
    def busy(self) -> list[str]:
        """What is running now (asked before a shutdown or restart)."""
        out = []
        if self.engine is not None and self.engine.is_scanning:
            out.append("SEM scanning")
        if self.eds is not None and self.eds.busy:
            out.append("EDS acquisition")
        st = self.microscope.stage if self.microscope is not None else None
        if st is not None and st.moving:
            out.append("stage moving")
        return out

    def status(self) -> dict[str, ComponentStatus]:
        return {"daq": self._daq_status(), "eds": self._eds_status(), "microscope": self._mic_status()}

    def _daq_status(self) -> ComponentStatus:
        c = self.config.daq
        if self.manager is None:
            return ComponentStatus("error", self.errors.get("daq", "not started"))
        if c.simulated:
            return ComponentStatus("sim", "simulator")
        name = f"NI {c.device}"
        if self.manager.connected:
            return ComponentStatus("ok", f"{name} connected")
        why = self.manager.lost_reason or "device not found"
        return ComponentStatus("waiting", f"{name}: {why} – connects when it appears")

    def _eds_status(self) -> ComponentStatus:
        kind = self.config.eds.mca
        if kind == "none":
            return ComponentStatus("off", "not used")
        if self.eds is None:
            return ComponentStatus("error", self.errors.get("eds", "not started"))
        idle = self.eds.idle_state
        if idle.startswith("error") or idle == "not connected":
            return ComponentStatus("warn", f"DP5: {idle}")
        if kind == "mock":
            return ComponentStatus("sim", "DP5 mock device")
        sn = ""
        try:
            sn = str(self.mca.info().get("serial", "") or "")
        except Exception:  # noqa: BLE001
            pass
        return ComponentStatus("ok", "DP5 connected" + (f" (S/N {sn})" if sn else ""))

    def _mic_status(self) -> ComponentStatus:
        port = self.config.microscope.port
        if port.lower() == "none":
            return ComponentStatus("off", "not used")
        b = self.microscope
        if b is None or not b.connected:
            return ComponentStatus("error", self.errors.get("microscope", "not started"))
        state = b.link.state
        if state == "connected":
            return ComponentStatus("sim", "S-3000 simulator") if b.link.simulated else \
                ComponentStatus("ok", f"S-3000 on {port}")
        if state == "uncertain":
            return ComponentStatus("warn", f"{port}: no reply – check the SEM, then Resync (Microscope window)")
        return ComponentStatus("error", f"{port}: {state}" + (f" – {b.link.error}" if b.link.error else ""))
