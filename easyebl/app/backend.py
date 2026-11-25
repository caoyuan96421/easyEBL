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

__all__ = ["SystemBackend", "ComponentStatus", "COMPONENTS", "couple_sim_stage", "SIM_STAGE_TO_BEAM"]

#: the simulators' hidden stage → beam map (what a field calibration should find): stage X against
#: the image's x as on the S-3000, the beam axes turned 0.35° against the stage and the global field
#: calibration 0.6 % off
SIM_STAGE_TO_BEAM = ((-1.006 * 0.99998134, -1.006 * 0.00610863), (-1.006 * 0.00610863, 1.006 * 0.99998134))


def couple_sim_stage(sim_backend, stage, stage_to_beam=SIM_STAGE_TO_BEAM, period_um: float = 120.0) -> None:
    """Make the simulated SEM image follow the simulated stage: moving the stage by ``d`` moves the
    sample by ``−B·d`` in the field (relative to where the stage was first read; the demo sample
    repeats every *period_um*, so there is always something to see)."""
    import numpy as np

    b = np.asarray(stage_to_beam, float)
    ref: dict = {}

    def offset():
        x, y = stage.positions.get("X"), stage.positions.get("Y")
        if x is None or y is None:
            return (0.0, 0.0)
        if not ref:
            ref["s"] = np.array([x, y], float)
        o = b @ (np.array([x, y], float) - ref["s"])
        return (float(o[0]), float(o[1]))

    sim_backend.params.stage_offset = offset
    sim_backend.params.sample_period_um = float(period_um)


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
        from ..hal import DAQManager, SimParams, SimulatedBackend, create_backend
        from ..imaging.engine import EngineOptions, ImagingEngine

        c = self.config.daq
        try:
            cfg = self.config.daq_config()  # preset / file with the Wiring tab's channels applied
            # the field itself is set by the global field calibration (FieldScale) once the microscope is open
            cal = FieldCalibration(100.0, cfg.full_scale_v, invert_x=bool(c.invert_x),
                                   invert_y=bool(c.invert_y), swap_xy=bool(c.swap_xy))
            if c.simulated:
                be = SimulatedBackend(cfg, SimParams(field_size_um=100.0, v_full_scale=cfg.full_scale_v,
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
                                     options=EDSOptions(external_control=self.config.eds.external_control,
                                                        video_channel=self._eds_video()),
                                     owner="eds")
        except Exception as e:  # noqa: BLE001
            self._fail("eds", e)
            if self.mca is not None:
                try:
                    self.mca.close()
                except Exception:  # noqa: BLE001
                    pass
            self.mca = self.eds = None

    def _eds_video(self) -> str | None:
        """The SE input of EDS (Wiring tab), if the open DAQ has it."""
        ch = self.config.eds_video_channel
        return ch if ch in self.manager.config.ai_channels else None

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
        if targets:  # also without a microscope: then the manual magnification is used
            self.microscope.tie_field(targets, sim_backend=sim_be, persist=not self.config.daq.simulated)
        if sim_be is not None and self.microscope.stage is not None and self.microscope.link.simulated:
            couple_sim_stage(sim_be, self.microscope.stage)

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
