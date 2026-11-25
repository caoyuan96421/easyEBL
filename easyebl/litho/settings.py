"""Persistent lithography settings: named stage positions and the stage axes (JSON, no Qt).

Stored per user in ``%APPDATA%\\easyEBL\\litho.json`` (Windows) or
``~/.config/easyebl/litho.json``; ``EASYEBL_LITHO_SETTINGS`` overrides the path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

from .coords import StageAxes
from .fieldcal import FieldCalConfig

__all__ = ["LithoSettings", "StagePosition", "default_litho_settings_path"]


def default_litho_settings_path() -> str:
    env = os.environ.get("EASYEBL_LITHO_SETTINGS")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "easyEBL", "litho.json")
    return os.path.join(os.path.expanduser("~"), ".config", "easyebl", "litho.json")


@dataclass
class StagePosition:
    """A fixed object on the stage (calibration sample, Faraday cup …), SCS µm."""

    name: str
    x: float
    y: float
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "StagePosition":
        return cls(str(d["name"]), float(d["x"]), float(d["y"]), str(d.get("note", "")))


@dataclass
class LithoSettings:
    stage_positions: list = field(default_factory=list)  # [StagePosition]
    axes: StageAxes = field(default_factory=StageAxes)
    #: a schedule step fails when the stage reads back farther than this from its target (µm)
    stage_tolerance_um: float = 1.0
    fieldcal: FieldCalConfig = field(default_factory=FieldCalConfig)
    last_dir: str = ""
    path: str = field(default="", repr=False, compare=False)

    # ---------------------------------------------------------------- positions
    def positions(self) -> dict:
        """{name: (x, y)}."""
        return {p.name: (p.x, p.y) for p in self.stage_positions}

    def find(self, name: str) -> int:
        return next((i for i, p in enumerate(self.stage_positions) if p.name == name), -1)

    def unique_name(self, base: str = "Position") -> str:
        names = {p.name for p in self.stage_positions}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def add_position(self, name: str, x: float, y: float, note: str = "") -> StagePosition:
        if not name.strip():
            raise ValueError("a stage position needs a name")
        if self.find(name) >= 0:
            raise ValueError(f"a stage position named {name!r} exists already")
        p = StagePosition(name.strip(), float(x), float(y), note)
        self.stage_positions.append(p)
        return p

    # ---------------------------------------------------------------- file
    def to_dict(self) -> dict:
        return {"stage_positions": [p.to_dict() for p in self.stage_positions],
                "axes": self.axes.to_dict(), "stage_tolerance_um": self.stage_tolerance_um,
                "fieldcal": self.fieldcal.to_dict(),
                "last_dir": self.last_dir}

    @classmethod
    def load(cls, path: str | None = None) -> "LithoSettings":
        path = path or default_litho_settings_path()
        s = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            s.stage_positions = [StagePosition.from_dict(p) for p in d.get("stage_positions", [])]
            s.axes = StageAxes.from_dict(d.get("axes"))
            s.stage_tolerance_um = float(d.get("stage_tolerance_um", 1.0))
            s.fieldcal = FieldCalConfig.from_dict(d.get("fieldcal"))
            s.last_dir = str(d.get("last_dir", ""))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        s.path = path
        return s

    def save(self, path: str | None = None) -> str:
        path = path or self.path or default_litho_settings_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)
        self.path = path
        return path
