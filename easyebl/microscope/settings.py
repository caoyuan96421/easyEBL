"""Persistent microscope settings (port, polling, magnification calibration, beam presets) – JSON, no Qt.

Stored per user in ``%APPDATA%\\easyEBL\\microscope.json`` (Windows) or
``~/.config/easyebl/microscope.json``; ``EASYEBL_MICROSCOPE_SETTINGS`` overrides
the path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os

__all__ = ["MicroscopeSettings", "default_settings_path"]


def default_settings_path() -> str:
    env = os.environ.get("EASYEBL_MICROSCOPE_SETTINGS")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "easyEBL", "microscope.json")
    return os.path.join(os.path.expanduser("~"), ".config", "easyebl", "microscope.json")


@dataclass
class MicroscopeSettings:
    port: str = "COM4"
    stage_poll_s: float = 30.0  # never below StageControl.MIN_POLL_S (20 s)
    sem_status_poll_s: float = 10.0  # MAG / HV / WD background read (0: off)
    #: the global field calibration: the field (µm) the full deflection range sweeps at ×1;
    #: the field at magnification M is this / M (easyebl.microscope.coupling.FieldScale)
    field_at_1x_um: float = 100_000.0
    #: the magnification used while the microscope does not report one (not connected)
    manual_magnification: float = 1000.0
    jog_step_um: float = 100.0
    jog_step_deg: float = 1.0
    #: named beam presets: [{"name": ..., "beam_coarse": ..., "beam_fine": ..., "stig_x": ...,
    #: "stig_y": ..., "contrast": ..., "brightness": ...}, ...] (raw SEM settings)
    beam_presets: list = field(default_factory=list)
    path: str = field(default="", repr=False, compare=False)

    @classmethod
    def load(cls, path: str | None = None) -> "MicroscopeSettings":
        path = path or default_settings_path()
        s = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            known = {f.name for f in fields(cls)} - {"path"}
            if "field_at_1x_um" not in d and d.get("field_mag_product_um"):  # the same constant, renamed
                d["field_at_1x_um"] = d["field_mag_product_um"]
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
