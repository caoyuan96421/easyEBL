"""Hardware configuration of the easyEBL system (JSON, no Qt).

Stored per user in ``%APPDATA%\\easyEBL\\system.json`` (Windows) or
``~/.config/easyebl/system.json``; ``EASYEBL_SYSTEM_CONFIG`` overrides the path.
Edited in the menu window (*Hardware → Configure…*).

The ``wiring`` section (:class:`~easyebl.hal.wiring.Wiring`, the *Wiring* tab)
maps every DAQ signal to its physical channel.  A file without it (written
before the tab existed) keeps the wiring of the DAQ preset / config file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
import os

from ..hal.wiring import Wiring

__all__ = ["Wiring", "DAQSection", "AI_RANGES_V", "EDSSection", "MicroscopeSection", "ScanMonitorSection", "SystemConfig", "default_config_path",
           "AI_TERMINALS", "BAUD_RATES"]

AI_TERMINALS = ("DIFF", "RSE", "NRSE", "PSEUDO_DIFF")
#: USB-6363 AI input ranges (±V): the smallest that holds the signal uses the full 16-bit resolution
AI_RANGES_V = (10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1)
BAUD_RATES = (1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200)


def default_config_path() -> str:
    env = os.environ.get("EASYEBL_SYSTEM_CONFIG")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "easyEBL", "system.json")
    return os.path.join(os.path.expanduser("~"), ".config", "easyebl", "system.json")


@dataclass
class DAQSection:
    """Scan DAQ (X/Y deflection, blanker, video input)."""

    backend: str = "sim"  # "sim" (simulator) or "nidaqmx"
    device: str = "Dev1"
    preset: str = "usb6363"  # wiring preset: "usb6363" or "default"
    config_file: str = ""  # optional DAQConfig JSON; replaces the preset
    rate: float | None = None  # sample clock (S/s); None: device maximum (500 kS/s on the simulator)
    latency_ms: float = 100.0  # output buffer depth (setting-change latency)
    ai_terminal: str | None = None  # video input mode; None: the preset's
    ai_range_v: float | None = None  # video input range (±V); None: the preset's
    invert_x: bool = False  # reverse the X deflection output
    invert_y: bool = False  # reverse the Y deflection output
    swap_xy: bool = False  # image X drives the Y deflection (before the inversions)

    @property
    def simulated(self) -> bool:
        return self.backend == "sim"


@dataclass
class EDSSection:
    """EDS pulse processor."""

    mca: str = "mock"  # "dp5" (USB), "mock" (the DP5 driver's mock device) or "none"
    external_control: bool = True  # switch the SEM to external scan for EDS jobs


@dataclass
class MicroscopeSection:
    """SEM remote control (Hitachi S-3000, RS-232)."""

    port: str = "sim"  # "COM4" …, "sim" (simulator) or "none"
    baudrate: int = 9600  # the S-3000 uses 9600 8N1
    #: stage axes against the SEM image (the lithography's stage → beam map): moving the stage to +X
    #: moves the image of the sample to +x (S-3000: X yes, Y no)
    stage_mirror_x: bool = True
    stage_mirror_y: bool = False


@dataclass
class ScanMonitorSection:
    """X/Y/blanker monitor in the menu window (oscilloscope XY mode with persistence)."""

    visible: bool = True
    resolution: int = 256  # pixels per axis over the full ±10 V field
    retention_s: float = 2.0  # persistence; 0 = keep until cleared
    fade: bool = True  # older samples fade out over the retention time


@dataclass
class SystemConfig:
    daq: DAQSection = field(default_factory=DAQSection)
    eds: EDSSection = field(default_factory=EDSSection)
    microscope: MicroscopeSection = field(default_factory=MicroscopeSection)
    monitor: ScanMonitorSection = field(default_factory=ScanMonitorSection)
    #: signal → channel map; None = the preset's / config file's wiring (files without a wiring section)
    wiring: Wiring | None = None
    path: str = field(default="", repr=False, compare=False)

    _SECTIONS = {"daq": DAQSection, "eds": EDSSection, "microscope": MicroscopeSection,
                 "monitor": ScanMonitorSection}

    @classmethod
    def simulators(cls) -> "SystemConfig":
        """Everything simulated (the default configuration)."""
        return cls()

    @classmethod
    def hardware(cls, device: str = "Dev1", port: str = "COM4") -> "SystemConfig":
        """The instrument: NI USB-6363, DP5 over USB, S-3000 on a serial port."""
        return cls(DAQSection(backend="nidaqmx", device=device), EDSSection(mca="dp5"),
                   MicroscopeSection(port=port))

    def as_simulated(self) -> "SystemConfig":
        """The same settings with every device replaced by its simulator."""
        return replace(self, daq=replace(self.daq, backend="sim"),
                       eds=replace(self.eds, mca="none" if self.eds.mca == "none" else "mock"),
                       microscope=replace(self.microscope,
                                          port="none" if self.microscope.port == "none" else "sim"))

    # ---------------------------------------------------------------- DAQ wiring
    def base_daq_config(self, preset: str | None = None, config_file: str | None = None):
        """The DAQConfig of the preset / config file, before the wiring section is applied."""
        from ..hal.config import DAQConfig

        c = self.daq
        path = c.config_file if config_file is None else config_file
        if path:
            with open(path, encoding="utf-8") as fh:
                cfg = DAQConfig.from_json(fh.read())
        else:
            cfg = DAQConfig.usb6363() if (preset or c.preset) == "usb6363" else DAQConfig()
        if c.device:
            cfg.device = c.device
        if c.ai_terminal:
            cfg.ai_terminal = c.ai_terminal
        if c.ai_range_v:
            cfg.ai_range_v = float(c.ai_range_v)
        return cfg

    def preset_wiring(self, preset: str | None = None, config_file: str | None = None) -> Wiring:
        """The wiring used when there is no wiring section (today's defaults)."""
        return Wiring.from_daq_config(self.base_daq_config(preset, config_file))

    def effective_wiring(self) -> Wiring:
        return self.wiring.copy() if self.wiring is not None else self.preset_wiring()

    def daq_config(self):
        """The DAQConfig the HAL is opened with: preset / file + device + input mode + wiring."""
        cfg = self.base_daq_config()
        if self.wiring is not None:
            self.wiring.validate(cfg.ai_terminal)
            self.wiring.apply_to(cfg)
        cfg.validate()
        return cfg

    @property
    def eds_video_channel(self) -> str | None:
        """The video input EDS co-acquires as its SE signal."""
        return self.effective_wiring().eds_video

    @property
    def simulated(self) -> bool:
        return self.daq.simulated and self.eds.mca != "dp5" and self.microscope.port.lower() in ("sim", "none")

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SystemConfig":
        out = cls()
        for name, sec_cls in cls._SECTIONS.items():
            known = {f.name for f in fields(sec_cls)}
            vals = {k: v for k, v in (d.get(name) or {}).items() if k in known}
            setattr(out, name, sec_cls(**vals))
        w = d.get("wiring")
        out.wiring = Wiring.from_dict(w) if isinstance(w, dict) else None
        return out

    @classmethod
    def load(cls, path: str | None = None) -> "SystemConfig":
        path = path or default_config_path()
        try:
            with open(path, encoding="utf-8") as fh:
                s = cls.from_dict(json.load(fh))
        except (OSError, ValueError, TypeError):
            s = cls()
        s.path = path
        return s

    def save(self, path: str | None = None) -> str:
        path = path or self.path or default_config_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)
        self.path = path
        return path
