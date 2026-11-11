"""Persistent EDS settings (energy calibration, detector model, defaults) – JSON, no Qt.

Stored per user in ``%APPDATA%\\easyEBL\\eds.json`` (Windows) or
``~/.config/easyebl/eds.json``; ``EASYEBL_EDS_SETTINGS`` overrides the path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os

from ..xray.generation import Detector
from .spectrum import EnergyCalibration

__all__ = ["EDSSettings", "default_settings_path"]


def default_settings_path() -> str:
    env = os.environ.get("EASYEBL_EDS_SETTINGS")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "easyEBL", "eds.json")
    return os.path.join(os.path.expanduser("~"), ".config", "easyebl", "eds.json")


@dataclass
class EDSSettings:
    calibration: dict = field(default_factory=lambda: EnergyCalibration(0.0, 0.01).to_dict())
    calibration_source: str = ""  # which processor / standard the calibration was made with
    window: str = "AP3.3"
    fwhm_mn_ev: float = 130.0
    fano: float = 0.115
    si_thickness_um: float = 450.0
    beam_kv: float = 20.0
    takeoff_deg: float = 35.0
    field_um: float = 100.0
    spectrum_time_s: float = 30.0
    live_time_mode: bool = False
    map_nx: int = 64
    map_ny: int = 48
    map_dwell_ms: float = 50.0
    image_nx: int = 512
    image_ny: int = 384
    image_dwell_us: float = 4.0
    oxygen_by_stoichiometry: bool = False
    save_folder: str = ""
    path: str = field(default="", repr=False, compare=False)

    # ------------------------------------------------------------- helpers
    @property
    def energy_calibration(self) -> EnergyCalibration:
        return EnergyCalibration.from_dict(self.calibration)

    @energy_calibration.setter
    def energy_calibration(self, cal: EnergyCalibration) -> None:
        self.calibration = cal.to_dict()

    @property
    def detector(self) -> Detector:
        return Detector(window=self.window, si_thickness_um=self.si_thickness_um, fwhm_mn_ev=self.fwhm_mn_ev,
                        fano=self.fano)

    # ------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path: str | None = None) -> "EDSSettings":
        path = path or default_settings_path()
        s = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            known = {f.name for f in fields(cls)} - {"path"}
            for k, v in d.items():
                if k in known:
                    setattr(s, k, v)
        except (OSError, ValueError):
            pass
        s.path = path
        return s

    def save(self, path: str | None = None) -> str:
        path = path or self.path or default_settings_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        d.pop("path", None)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
        os.replace(tmp, path)
        self.path = path
        return path
