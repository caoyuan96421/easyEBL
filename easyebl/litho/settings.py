"""Persistent lithography settings (JSON, no Qt): the exposure field sizes, the named stage
positions, the beam calibrations and the field calibration of each field size.

* **Field sizes** – each a magnification and a nominal WD (editable; three to start with).  The
  Microscope window selects one (:attr:`LithoSettings.field_size`); a schedule runs only at the
  field size selected there.
* **Stage positions** – X, Y and ΔZ (µm, relative to the calibration sample's Z; positive moves
  the stage away from the gun).  Two are always there (replaced, never deleted): the
  **Calibration sample** – ΔZ 0, its Z (``Z_cal``) and the WD in focus there (``WD1``, *Record
  WD*), both kept per field size (its :attr:`StagePosition.z_um` / :attr:`StagePosition.wd_mm`
  are the selected field size's; every ΔZ is against that Z_cal) – and the **Faraday cup**.
  Replacing the calibration sample at another Z changes every other ΔZ so their Z stays; *Record
  WD* changes only the calibration sample.
* **Field calibrations** – one per field size, kept across sessions (only its stage ↔ beam
  rotation is used).

Stored per user in ``%APPDATA%\\easyEBL\\litho.json`` (Windows) or
``~/.config/easyebl/litho.json``; ``EASYEBL_LITHO_SETTINGS`` overrides the path.  The stage axes
are hardware settings (Hardware → Configure… → Microscope), not stored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

from .beamcal import BeamCalibration
from .fieldcal import Condition, FieldCalConfig, FieldCalResult

__all__ = ["LithoSettings", "StagePosition", "FieldSize", "DEFAULT_FIELD_SIZES", "default_litho_settings_path",
           "CALIBRATION", "FARADAY", "ROLE_NAMES"]

CALIBRATION = "calibration"
FARADAY = "faraday"
ROLE_NAMES = {CALIBRATION: "Calibration sample", FARADAY: "Faraday cup"}


def default_litho_settings_path() -> str:
    env = os.environ.get("EASYEBL_LITHO_SETTINGS")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "easyEBL", "litho.json")
    return os.path.join(os.path.expanduser("~"), ".config", "easyebl", "litho.json")


@dataclass
class FieldSize:
    """An exposure field size: *name* (e.g. "100 µm"), the *magnification* and the nominal WD."""

    name: str
    magnification: int
    wd_mm: float

    def __post_init__(self):
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("a field size needs a name")
        self.magnification = int(self.magnification)
        if self.magnification <= 0:
            raise ValueError("the magnification must be positive")
        self.wd_mm = float(self.wd_mm)
        if not 4.0 <= self.wd_mm <= 60.0:
            raise ValueError(f"the nominal WD must be 4–60 mm (got {self.wd_mm:g})")

    def describe(self) -> str:
        return f"{self.name}: ×{self.magnification:g}, WD {self.wd_mm:g} mm"

    def to_dict(self) -> dict:
        return {"name": self.name, "magnification": self.magnification, "wd_mm": self.wd_mm}

    @classmethod
    def from_dict(cls, d: dict) -> "FieldSize":
        return cls(d["name"], d["magnification"], d["wd_mm"])


def DEFAULT_FIELD_SIZES() -> list:  # noqa: N802 - a fresh list each time
    return [FieldSize("100 µm", 1200, 7.0), FieldSize("250 µm", 500, 11.0), FieldSize("500 µm", 250, 24.0)]


@dataclass
class StagePosition:
    """A fixed object on the stage: X, Y (SCS µm) and *dz* – its Z relative to the calibration
    sample's (µm, positive away from the gun; None: not recorded).  *role* calibration | faraday |
    "" (a user's position).  The calibration sample also has its stage Z *z_um* (``Z_cal``) and the
    WD in focus there *wd_mm* (``WD1``), both of the field size selected now; *wd1* / *zcal*:
    {field size: WD1 / Z_cal}, kept for each – the settings keep *wd_mm* and *z_um* in step with
    the selection."""

    name: str
    x: float
    y: float
    dz: float | None = None
    note: str = ""
    role: str = ""
    z_um: float | None = None
    wd_mm: float | None = None
    wd1: dict = field(default_factory=dict)
    zcal: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.role == CALIBRATION:
            self.dz = 0.0
        self.wd1 = {str(k): float(v) for k, v in (self.wd1 or {}).items()}
        self.zcal = {str(k): float(v) for k, v in (self.zcal or {}).items()}

    @property
    def fixed(self) -> bool:
        """The calibration sample or the Faraday cup (replaced, never deleted)."""
        return bool(self.role)

    def to_dict(self) -> dict:
        d = {"name": self.name, "x": self.x, "y": self.y, "dz": self.dz, "note": self.note, "role": self.role,
             "z_um": self.z_um, "wd_mm": self.wd_mm}
        if self.wd1:
            d["wd1"] = dict(self.wd1)
        if self.zcal:
            d["zcal"] = dict(self.zcal)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StagePosition":
        def f(k):
            v = d.get(k)
            return None if v is None else float(v)
        # an older file's "h" (= WD − Z) or "z" cannot be converted: ΔZ not recorded
        return cls(str(d["name"]), float(d["x"]), float(d["y"]), f("dz"), str(d.get("note", "")),
                   str(d.get("role", "")), f("z_um"), f("wd_mm"), dict(d.get("wd1") or {}),
                   dict(d.get("zcal") or {}))


_SHARED: dict = {}


@dataclass
class LithoSettings:
    stage_positions: list = field(default_factory=list)  # [StagePosition], the two fixed ones first
    #: a schedule step fails when the stage reads back farther than this from its target (µm)
    stage_tolerance_um: float = 0.5
    beamcals: list = field(default_factory=list)  # [BeamCalibration], one per condition (field size, magnification)
    field_sizes: list = field(default_factory=DEFAULT_FIELD_SIZES)  # [FieldSize]
    field_size: str = ""  # the one selected (Microscope window)
    fieldcals: dict = field(default_factory=dict)  # {field size name: FieldCalResult}
    fieldcal_config: FieldCalConfig = field(default_factory=FieldCalConfig)
    last_dir: str = ""
    path: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        self._ensure_fixed()
        self._sync_wd1()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "field_size" and "stage_positions" in self.__dict__:
            self._sync_wd1()

    def _sync_wd1(self) -> None:
        """The calibration sample's WD1 and Z_cal := those recorded for the selected field size (None: none)."""
        try:
            cal = self.calibration
        except StopIteration:
            return
        cal.wd_mm = cal.wd1.get(self.field_size)
        cal.z_um = cal.zcal.get(self.field_size)

    def z_cal(self, name: str | None = None) -> float | None:
        """Z_cal recorded for field size *name* (None: the selected one)."""
        return self.calibration.zcal.get(self.field_size if name is None else name)

    def wd1(self, name: str | None = None) -> float | None:
        """WD1 recorded for field size *name* (None: the selected one)."""
        return self.calibration.wd1.get(self.field_size if name is None else name)

    @classmethod
    def shared(cls, path: str | None = None) -> "LithoSettings":
        """The one instance of the settings file *path* in this process (the Lithography and the
        Microscope windows share it, so their Stage Positions and field sizes agree)."""
        key = os.path.normcase(os.path.abspath(path or default_litho_settings_path()))
        s = _SHARED.get(key)
        if s is None:
            s = _SHARED[key] = cls.load(path)
        return s

    # ---------------------------------------------------------------- field sizes
    def find_field_size(self, name: str | None) -> FieldSize | None:
        return next((f for f in self.field_sizes if f.name == name), None)

    @property
    def selected_field_size(self) -> FieldSize | None:
        return self.find_field_size(self.field_size)

    def set_field_sizes(self, sizes: list) -> None:
        names = [s.name for s in sizes]
        if len(set(names)) != len(names):
            raise ValueError("two field sizes have the same name")
        self.field_sizes = list(sizes)
        if self.field_size not in names:
            self.field_size = ""

    def fieldcal(self, name: str | None = None) -> FieldCalResult | None:
        """The field calibration of field size *name* (None: the selected one)."""
        return self.fieldcals.get(self.field_size if name is None else name)

    # ---------------------------------------------------------------- positions
    def _ensure_fixed(self) -> None:
        for k, role in enumerate((CALIBRATION, FARADAY)):
            i = next((j for j, p in enumerate(self.stage_positions) if p.role == role), -1)
            p = self.stage_positions.pop(i) if i >= 0 else None
            if p is None:
                name = ROLE_NAMES[role]
                j = self.find(name)  # an older file's position of that name becomes the fixed one
                if j >= 0:
                    old = self.stage_positions.pop(j)
                    p = StagePosition(old.name, old.x, old.y, None, old.note, role)
                else:
                    p = StagePosition(name, 0.0, 0.0, None, "", role)
                if role == CALIBRATION:
                    p.dz = 0.0
            self.stage_positions.insert(k, p)

    def positions(self) -> dict:
        """{name: StagePosition}."""
        return {p.name: p for p in self.stage_positions}

    def role(self, role: str) -> StagePosition:
        return next(p for p in self.stage_positions if p.role == role)

    @property
    def calibration(self) -> StagePosition:
        """The calibration sample (``Z_cal``: its *z_um*, ``WD1``: its *wd_mm*)."""
        return self.role(CALIBRATION)

    @property
    def faraday(self) -> StagePosition:
        return self.role(FARADAY)

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

    def dz_here(self, z_um: float | None) -> float | None:
        """ΔZ of the stage Z *z_um* against the calibration sample (None: either not known)."""
        zc = self.calibration.z_um
        return None if z_um is None or zc is None else float(z_um) - zc

    def z_of(self, p: StagePosition) -> float | None:
        """The stage Z of position *p*: ``Z_cal + ΔZ`` (None: not known)."""
        zc = self.calibration.z_um
        if p.role == CALIBRATION:
            return zc
        return None if zc is None or p.dz is None else zc + p.dz

    def add_position(self, name: str, x: float, y: float, dz: float | None = None, note: str = "") -> StagePosition:
        if not name.strip():
            raise ValueError("a stage position needs a name")
        if self.find(name) >= 0:
            raise ValueError(f"a stage position named {name!r} exists already")
        p = StagePosition(name.strip(), float(x), float(y), None if dz is None else float(dz), note)
        self.stage_positions.append(p)
        return p

    def delete_position(self, name: str) -> None:
        i = self.find(name)
        if i < 0:
            raise ValueError(f"no stage position named {name!r}")
        if self.stage_positions[i].fixed:
            raise ValueError(f"{name!r} is always kept (replace it instead)")
        del self.stage_positions[i]

    def replace_position(self, name: str, x: float, y: float, z_um: float | None) -> StagePosition:
        """Position *name* to the stage at (x, y, z_um).  The calibration sample: its Z becomes
        ``Z_cal`` of the selected field size and every other ΔZ changes so their stage Z stays (WD1
        is kept – record it again at the new place); another position: ΔZ from the selected field
        size's Z_cal."""
        i = self.find(name)
        if i < 0:
            raise ValueError(f"no stage position named {name!r}")
        p = self.stage_positions[i]
        p.x, p.y = float(x), float(y)
        if p.role == CALIBRATION:
            if z_um is not None:
                old = p.z_um
                if old is not None:
                    shift = float(z_um) - old
                    for q in self.stage_positions:
                        if q is not p and q.dz is not None:
                            q.dz -= shift
                p.z_um = p.zcal[self.field_size] = float(z_um)
        else:
            p.dz = self.dz_here(z_um)
        return p

    def record_wd(self, wd_mm: float, z_um: float | None, field_size: str | None = None) -> StagePosition:
        """WD1 of the field size *field_size* (None: the selected one): the WD in focus on the
        calibration sample, with the stage at *z_um* (its ``Z_cal``; no other ΔZ changes)."""
        p = self.calibration
        name = self.field_size if field_size is None else field_size
        p.wd1[name] = round(float(wd_mm), 4)
        if name == self.field_size:
            p.wd_mm = p.wd1[name]
        if z_um is not None:
            p.zcal[name] = float(z_um)
            if name == self.field_size:
                p.z_um = p.zcal[name]
        return p

    # ---------------------------------------------------------------- beam calibrations
    def set_beamcal(self, bc: BeamCalibration) -> None:
        """Keep *bc*, replacing the one of the same condition (field size, magnification)."""
        self.beamcals = [b for b in self.beamcals
                         if b.condition is None or not b.condition.matches(bc.condition, preset=False)] + [bc]

    def forget_beamcal(self, condition: Condition) -> bool:
        n = len(self.beamcals)
        self.beamcals = [b for b in self.beamcals if b.condition is None or not b.condition.matches(condition, preset=False)]
        return len(self.beamcals) != n

    # ---------------------------------------------------------------- file
    def to_dict(self) -> dict:
        return {"stage_positions": [p.to_dict() for p in self.stage_positions],
                "stage_tolerance_um": self.stage_tolerance_um,
                "beamcals": [b.to_dict() for b in self.beamcals],
                "field_sizes": [f.to_dict() for f in self.field_sizes],
                "field_size": self.field_size,
                "fieldcals": {k: v.to_dict() for k, v in self.fieldcals.items()},
                "fieldcal_config": self.fieldcal_config.to_dict(),
                "last_dir": self.last_dir}

    @classmethod
    def load(cls, path: str | None = None) -> "LithoSettings":
        path = path or default_litho_settings_path()
        s = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = {}
        for key, fn in (
                ("stage_positions", lambda v: [StagePosition.from_dict(p) for p in v]),
                ("stage_tolerance_um", float),
                ("beamcals", lambda v: [BeamCalibration.from_dict(b, Condition.from_dict) for b in v]),
                ("field_sizes", lambda v: [FieldSize.from_dict(f) for f in v] or DEFAULT_FIELD_SIZES()),
                ("field_size", str),
                ("fieldcals", lambda v: {str(k): FieldCalResult.from_dict(c) for k, c in v.items()}),
                ("fieldcal_config", FieldCalConfig.from_dict),
                ("last_dir", str)):
            if key in d:
                try:  # as stored: the sync to the selected field size comes after the migration below
                    object.__setattr__(s, key, fn(d[key]))
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
        s._ensure_fixed()
        cal = s.calibration
        if not cal.wd1 and cal.wd_mm is not None:  # an older file: one WD1, the selected field size's
            cal.wd1[s.field_size] = cal.wd_mm
        if not cal.zcal and cal.z_um is not None:  # … and one Z_cal
            cal.zcal[s.field_size] = cal.z_um
        if s.field_size and s.find_field_size(s.field_size) is None:
            s.field_size = ""
        s._sync_wd1()
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
