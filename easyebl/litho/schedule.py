"""The exposure schedule (no Qt): Column setting, Field correction, Expose field steps.

* **Column setting** (``column``) – beam preset of the microscope module (or keep as is),
  magnification (sets the field from the global field calibration), optional nominal beam
  current (for the dose readout).  Every schedule starts with one; more may follow.
* **Field correction** (``field_correction``) – the field calibration (stage → beam map) at a
  fixed stage position; valid until the magnification or the beam preset changes.
* **Expose field** (``expose``) – the field centre (SCS), optionally corrected by the last
  Global Align; the pattern (GDS cell centred in its cell, a dummy test pattern, or none);
  dwell per point and point step; optional Global Align (2–4 markers) and Local XY (1 marker).
  Markers are WCS points (µm from the field centre): they move with the field.

Saved as JSON (``format: easyebl-schedule``, version 2).  :func:`plan` walks a schedule
without hardware: stage targets, field size, transform and warnings per step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os
from typing import ClassVar

import numpy as np

from ..core.transforms import AffineTransform
from ..microscope.s3000.commands import ALLOWED_MAGNIFICATIONS
from . import dose as dosemod
from .coords import FieldTransform, GlobalAlignment, StageAxes, StageFrame, check_field
from .gds import GDSDoseMapping, load_library, pattern_from_cell
from .geometry import Dot, Pattern, Polygon, Polyline, rectangle

__all__ = ["Operation", "ColumnSetting", "FieldCorrection", "ExposeField", "PatternSpec", "PatternLibrary",
           "ScheduleSettings", "Schedule", "StepPlan", "plan", "OP_TYPES", "SCHEDULE_FORMAT", "ScheduleError",
           "dummy_pattern", "KEEP_AS_IS", "dose_readout", "MAGNIFICATIONS", "nearest_magnification"]

SCHEDULE_FORMAT = "easyebl-schedule"
SCHEDULE_VERSION = 2
KEEP_AS_IS = ""  # the column setting's preset: keep the beam as it is
#: the magnifications the S-3000 accepts (manual section 8)
MAGNIFICATIONS: tuple = ALLOWED_MAGNIFICATIONS


def nearest_magnification(mag: float) -> int:
    """The permitted magnification closest to *mag* (on a log scale)."""
    return min(MAGNIFICATIONS, key=lambda m: abs(np.log(m / float(mag))))


class ScheduleError(ValueError):
    pass


def _xy(v) -> tuple:
    a = tuple(float(x) for x in v)
    if len(a) != 2:
        raise ScheduleError("a point is (x, y)")
    return a


# ====================================================================== patterns
def dummy_pattern(size_um: float = 20.0) -> Pattern:
    """Test pattern centred on (0, 0): a cross of lines, four squares, a dot in the middle."""
    s = float(size_um)
    h, q = s / 2.0, s / 10.0
    els = [Polyline(np.array([[-h, 0.0], [h, 0.0]])), Polyline(np.array([[0.0, -h], [0.0, h]])), Dot(0.0, 0.0)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * s / 3.0, sy * s / 3.0
            els.append(rectangle(cx - q, cy - q, cx + q, cy + q))
    return Pattern(els, name=f"dummy {s:g} µm")


def _mapping_to_dict(m: GDSDoseMapping) -> dict:
    d = asdict(m)
    d["table"] = {str(k): v for k, v in m.table.items()}
    return d


def _mapping_from_dict(d: dict | None) -> GDSDoseMapping:
    d = dict(d or {})
    d["table"] = {int(k): float(v) for k, v in (d.get("table") or {}).items()}
    known = {f.name for f in fields(GDSDoseMapping)}
    return GDSDoseMapping(**{k: v for k, v in d.items() if k in known})


@dataclass
class PatternSpec:
    """An Expose field's pattern.

    * ``"gds"`` – *cell* of GDS file *path* (relative paths: relative to the schedule file),
      flattened, *layers* kept (None: all), doses from *dose*.  The cell is assumed centred:
      its coordinates are WCS.  *placement* (cell → WCS) is set by the importer only (a field
      cut from a layout is recentred on its bbox centre).
    * ``"dummy"`` – :func:`dummy_pattern` of *size_um*.
    * ``"none"`` – nothing: the step aligns only.
    """

    kind: str = "dummy"
    path: str = ""
    cell: str = ""
    layers: list | None = None
    dose: GDSDoseMapping = field(default_factory=GDSDoseMapping)
    placement: AffineTransform = field(default_factory=AffineTransform.identity)
    size_um: float = 20.0

    def __post_init__(self):
        if self.kind not in ("gds", "dummy", "none"):
            raise ScheduleError("pattern kind must be 'gds', 'dummy' or 'none'")

    def describe(self) -> str:
        if self.kind == "gds":
            s = f"{os.path.basename(self.path) or '(no file)'}:{self.cell or '?'}"
            if not self.placement.is_identity():
                o = self.placement.offset
                s += f" (recentred by {o[0]:+.2f}, {o[1]:+.2f})"
            return s
        if self.kind == "dummy":
            return f"dummy {self.size_um:g} µm"
        return "none (alignment only)"

    def to_dict(self, base_dir: str | None = None) -> dict:
        if self.kind == "none":
            return {"kind": "none"}
        if self.kind == "dummy":
            return {"kind": "dummy", "size_um": self.size_um}
        path = self.path
        if base_dir and path and os.path.isabs(path):
            try:
                path = os.path.relpath(path, base_dir)
            except ValueError:  # another drive
                pass
        d = {"kind": "gds", "path": path.replace("\\", "/"), "cell": self.cell,
             "layers": [list(x) if isinstance(x, (tuple, list)) else x for x in self.layers]
             if self.layers is not None else None, "dose": _mapping_to_dict(self.dose)}
        if not self.placement.is_identity():
            d["placement"] = self.placement.to_list()
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "PatternSpec":
        kind = d.get("kind", "dummy")
        if kind in ("none", "dummy"):
            return cls(kind, size_um=float(d.get("size_um", 20.0)))
        path = d.get("path", "")
        if base_dir and path and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(base_dir, path))
        layers = d.get("layers")
        if layers is not None:
            layers = [tuple(x) if isinstance(x, list) else x for x in layers]
        placement = AffineTransform.from_list(d["placement"]) if "placement" in d else AffineTransform.identity()
        return cls("gds", path, d.get("cell", ""), layers, _mapping_from_dict(d.get("dose")), placement)


class PatternLibrary:
    """Loads :class:`PatternSpec` patterns (WCS), reading each GDS file once."""

    def __init__(self):
        self._libs: dict = {}
        self._cells: dict = {}

    def library(self, path: str):
        key = os.path.normcase(os.path.abspath(path))
        if key not in self._libs:
            if not os.path.exists(path):
                raise ScheduleError(f"GDS file not found: {path}")
            self._libs[key] = load_library(path)
        return self._libs[key]

    def cells(self, path: str) -> list:
        lib = self.library(path)
        tops = [c.name for c in lib.top_level()]
        return tops + [c.name for c in lib.cells if c.name not in tops]

    def get(self, spec: PatternSpec) -> Pattern:
        if spec.kind == "none":
            return Pattern([])
        if spec.kind == "dummy":
            return dummy_pattern(spec.size_um)
        if not spec.path or not spec.cell:
            raise ScheduleError("no GDS file / cell chosen")
        key = (os.path.normcase(os.path.abspath(spec.path)), spec.cell,
               None if spec.layers is None else tuple(spec.layers), json.dumps(_mapping_to_dict(spec.dose),
                                                                                sort_keys=True))
        pat = self._cells.get(key)
        if pat is None:
            lib = self.library(spec.path)
            found = [c for c in lib.cells if c.name == spec.cell]
            if not found:
                raise ScheduleError(f"cell {spec.cell!r} not in {os.path.basename(spec.path)}")
            pat = pattern_from_cell(found[0], spec.layers, spec.dose)
            pat.source = spec.path
            self._cells[key] = pat
        if not spec.placement.is_identity():
            return pat.transformed(spec.placement)
        return Pattern(list(pat.elements), pat.name, pat.source)

    def clear(self) -> None:
        self._libs.clear()
        self._cells.clear()


# ====================================================================== operations
@dataclass
class Operation:
    TYPE: ClassVar[str] = ""
    TITLE: ClassVar[str] = ""
    label: str = ""
    enabled: bool = True

    def summary(self) -> str:
        return ""

    def title(self) -> str:
        s = self.summary()
        return f"{self.TITLE}: {s}" if s else self.TITLE

    def to_dict(self, base_dir: str | None = None) -> dict:
        d = {"type": self.TYPE}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = list(v) if isinstance(v, tuple) else v
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Operation":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ColumnSetting(Operation):
    TYPE: ClassVar[str] = "column"
    TITLE: ClassVar[str] = "Column setting"
    preset: str = KEEP_AS_IS  # beam preset name; "" keeps the beam as it is
    magnification: int = 1000  # one of MAGNIFICATIONS
    current_pa: float | None = None  # nominal beam current (dose readout only)

    def __post_init__(self):
        if not self.magnification or self.magnification <= 0:
            raise ScheduleError("the magnification must be positive")
        if float(self.magnification).is_integer():
            self.magnification = int(self.magnification)  # the SEM driver takes whole numbers only

    @property
    def permitted(self) -> bool:
        return self.magnification in MAGNIFICATIONS

    def summary(self) -> str:
        s = f"×{self.magnification:g}, beam {repr(self.preset) if self.preset else 'kept as is'}"
        return s + (f", {self.current_pa:g} pA" if self.current_pa else "")


@dataclass
class FieldCorrection(Operation):
    """Field calibration (stage → beam map) at a fixed stage position, valid for the column
    condition it is measured at (see :mod:`easyebl.litho.fieldcal`)."""

    TYPE: ClassVar[str] = "field_correction"
    TITLE: ClassVar[str] = "Field correction"
    position: str = ""  # a named stage position; "": the one in the field calibration settings
    notes: str = ""

    def summary(self) -> str:
        return f"calibrate at {self.position!r}" if self.position else "calibrate (configured position)"


@dataclass
class ExposeField(Operation):
    TYPE: ClassVar[str] = "expose"
    TITLE: ClassVar[str] = "Expose field"
    centre: tuple = (0.0, 0.0)  # SCS µm
    correct_centre: bool = True  # move to the centre corrected by the last Global Align
    pattern: PatternSpec = field(default_factory=PatternSpec)
    dwell_us: float = 1.0  # per exposure point (× the GDS dose factor)
    step_nm: float = 20.0  # point spacing (area fill and lines)
    global_align: bool = False
    markers: list = field(default_factory=list)  # 2–4 WCS points (µm from the field centre)
    local_xy: bool = False
    local_marker: tuple = (0.0, 0.0)  # WCS
    view_um: float = 10.0  # width of the zoomed SEM scan around a predicted marker

    def __post_init__(self):
        self.centre = _xy(self.centre)
        self.local_marker = _xy(self.local_marker)
        self.markers = [_xy(m) for m in self.markers]
        if isinstance(self.pattern, dict):
            self.pattern = PatternSpec.from_dict(self.pattern)
        if not self.dwell_us > 0 or not self.step_nm > 0:
            raise ScheduleError("dwell and step must be positive")

    @property
    def active_markers(self) -> list:
        return list(self.markers) if self.global_align else []

    def nominal_scs(self, wcs, axes: StageAxes) -> np.ndarray:
        """Nominal stage position of WCS points of this field."""
        return axes.wcs_to_scs(wcs, self.centre)

    def summary(self) -> str:
        al = []
        if self.global_align:
            al.append(f"global {len(self.markers)}")
        if self.local_xy:
            al.append("local XY")
        return (f"{self.label or self.pattern.describe()} @ ({self.centre[0]:.1f}, {self.centre[1]:.1f})"
                + (f" [{', '.join(al)}]" if al else "") + f", {self.dwell_us:g} µs")

    def to_dict(self, base_dir: str | None = None) -> dict:
        d = super().to_dict(base_dir)
        d["pattern"] = self.pattern.to_dict(base_dir)
        d["markers"] = [list(m) for m in self.markers]
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "ExposeField":
        d = dict(d)
        d["pattern"] = PatternSpec.from_dict(d.get("pattern") or {}, base_dir)
        return super().from_dict(d, base_dir)


OP_TYPES: dict = {c.TYPE: c for c in (ColumnSetting, FieldCorrection, ExposeField)}


def op_from_dict(d: dict, base_dir: str | None = None) -> Operation:
    t = d.get("type")
    if t not in OP_TYPES:
        raise ScheduleError(f"unknown operation type {t!r}")
    return OP_TYPES[t].from_dict({k: v for k, v in d.items() if k != "type"}, base_dir)


def dose_readout(dwell_us: float, step_nm: float, current_pa: float | None) -> str:
    """Equivalent doses of a dwell / step at a nominal current."""
    if not current_pa:
        return "no nominal current (Column setting): dose unknown"
    dw, st, i = dwell_us * 1e-6, step_nm * 1e-3, current_pa * 1e-12
    return (f"≙ {dosemod.area_dose_uC_cm2(dw, st, i):.4g} µC/cm², {dosemod.line_dose_pC_cm(dw, st, i):.4g} pC/cm, "
            f"{dosemod.dot_dose_fC(dw, i):.4g} fC/dot at {current_pa:g} pA")


# ====================================================================== schedule
@dataclass
class ScheduleSettings:
    fill_direction: str = "x"
    serpentine: bool = False
    order: str = "as_is"
    on_out_of_field: str = "ask"  # ask (skip the field or abort) | skip | abort


@dataclass
class Schedule:
    operations: list = field(default_factory=list)
    name: str = ""
    settings: ScheduleSettings = field(default_factory=ScheduleSettings)
    vector_settings: dict = field(default_factory=dict)  # VectorScanSettings overrides
    path: str = field(default="", compare=False)

    def __post_init__(self):
        if not self.operations or not isinstance(self.operations[0], ColumnSetting):
            self.operations.insert(0, ColumnSetting())

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    def __getitem__(self, i):
        return self.operations[i]

    def add(self, *ops) -> "Schedule":
        self.operations.extend(ops)
        return self

    def can_delete(self, i: int) -> bool:
        return 0 < i < len(self.operations)

    def can_move(self, i: int, j: int) -> bool:
        return 0 < i < len(self.operations) and 0 < j < len(self.operations)

    @property
    def fields(self) -> list:
        return [o for o in self.operations if isinstance(o, ExposeField)]

    def shift_fields(self, dx: float, dy: float) -> int:
        """Move every field (centre; its markers are WCS and move along) by (dx, dy) µm of stage."""
        for o in self.fields:
            o.centre = (o.centre[0] + float(dx), o.centre[1] + float(dy))
        return len(self.fields)

    # ---------------------------------------------------------------- file
    def to_dict(self, base_dir: str | None = None) -> dict:
        return {"format": SCHEDULE_FORMAT, "version": SCHEDULE_VERSION, "name": self.name,
                "settings": asdict(self.settings), "vector_settings": dict(self.vector_settings),
                "operations": [o.to_dict(base_dir) for o in self.operations]}

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Schedule":
        if d.get("format") != SCHEDULE_FORMAT:
            raise ScheduleError("not an easyEBL schedule file")
        v = int(d.get("version", 1))
        if v != SCHEDULE_VERSION:
            raise ScheduleError(f"schedule file version {v} is not supported (this program reads version "
                                f"{SCHEDULE_VERSION})")
        known = {f.name for f in fields(ScheduleSettings)}
        st = ScheduleSettings(**{k: v for k, v in (d.get("settings") or {}).items() if k in known})
        return cls([op_from_dict(o, base_dir) for o in d.get("operations", [])], d.get("name", ""), st,
                   dict(d.get("vector_settings") or {}))

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        if not path:
            raise ScheduleError("no file name")
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(os.path.dirname(path)), fh, indent=1)
        os.replace(tmp, path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str) -> "Schedule":
        path = os.path.abspath(path)
        with open(path, encoding="utf-8") as fh:
            try:
                d = json.load(fh)
            except ValueError as e:
                raise ScheduleError(f"not a schedule file: {e}") from e
        s = cls.from_dict(d, os.path.dirname(path))
        s.path = path
        return s


# ====================================================================== plan
@dataclass
class StepPlan:
    """What a step will do, computed without hardware."""

    index: int
    field_um: float
    current_pa: float | None
    target: tuple | None = None  # stage target of an Expose field (SCS)
    transform: FieldTransform | None = None  # WCS → BCS (assuming the alignments reproduce the guess)
    warnings: list = field(default_factory=list)
    fov_check: object = None
    frame: StageFrame | None = None  # stage axes + the stage → beam map in force at this step
    calibrated: bool = False  # a field calibration applies to this step


def field_target(op: ExposeField, glob: GlobalAlignment | None, frame) -> np.ndarray:
    """Stage target of an Expose field: its centre, corrected by the last Global Align if asked
    (*frame*: StageFrame, or StageAxes for the nominal stage → beam map)."""
    c = np.array(op.centre, float)
    if op.correct_centre and glob is not None:
        return glob.corrected_centre(c, frame)
    return c


def plan(schedule: Schedule, field_at_1x_um: float, axes: StageAxes | None = None, presets=None,
         library: PatternLibrary | None = None, glob: GlobalAlignment | None = None, calibration=None,
         preset: str | None = None, positions: dict | None = None, fieldcal=None) -> list:
    """Walk *schedule* without hardware (Global Aligns are assumed to measure *glob*'s A, or the
    nominal map, and no shift; a Field correction to measure *calibration*'s map, or the nominal
    one).  *field_at_1x_um*: the global field calibration (field = K / mag); *preset*: the beam
    preset in force before the schedule (None: not known); *positions* / *fieldcal*: named stage
    positions and the field calibration settings.  Returns one :class:`StepPlan` per operation."""
    from .fieldcal import Condition

    axes = axes or StageAxes()
    library = library or PatternLibrary()
    positions = positions or {}
    glob = glob.copy() if glob is not None else None
    field_um = float(field_at_1x_um) / schedule[0].magnification
    cur = None
    mag = None
    mag_changed = False
    cal_b = calibration.stage_to_beam if calibration is not None else None
    cal_cond = calibration.condition if calibration is not None else None
    out = []
    for i, op in enumerate(schedule.operations):
        sp = StepPlan(i, field_um, cur)
        out.append(sp)
        if not op.enabled:
            if i == 0:
                sp.warnings.append("the first Column setting is always applied")
            else:
                continue
        try:
            if isinstance(op, ColumnSetting):
                preset = op.preset or preset
                if not op.permitted:
                    sp.warnings.append(f"magnification ×{op.magnification:g} is not one the SEM accepts "
                                       f"(nearest: ×{nearest_magnification(op.magnification)})")
                if presets is not None and op.preset and op.preset not in presets:
                    sp.warnings.append(f"no beam preset named {op.preset!r} in the microscope module")
                if mag is not None and op.magnification != mag and glob is not None:
                    mag_changed = True  # a global alignment measured at another magnification
                mag = op.magnification
                field_um = float(field_at_1x_um) / op.magnification
                cur = op.current_pa or cur
            elif isinstance(op, FieldCorrection):
                name = op.position or (fieldcal.position if fieldcal is not None else "")
                if name and name not in positions:
                    sp.warnings.append(f"no stage position named {name!r} for the field calibration")
                elif not name:
                    x, y = (fieldcal.x, fieldcal.y) if fieldcal is not None else (0.0, 0.0)
                    d = (fieldcal.step_fraction if fieldcal is not None else 0.3) * field_um
                    if min(x, y) - d < 0:
                        sp.warnings.append("the field calibration position is not set (Field → Field calibration "
                                           "settings…)")
                cal_cond = Condition(mag, preset)
            elif isinstance(op, ExposeField):
                ok = cal_cond is not None and cal_cond == Condition(mag, preset)
                frame = StageFrame.make(axes, cal_b if ok else None)
                sp.frame, sp.calibrated = frame, ok
                if not ok:
                    sp.warnings.append(f"no field calibration for {Condition(mag, preset).describe()} (a Field "
                                       "correction step, or Field → Run field calibration): nominal stage ↔ beam map")
                _plan_expose(sp, op, i, glob, frame, field_um, library, mag_changed, mag)
                if op.global_align:
                    a = glob.linear if glob is not None else frame.nominal_transform().linear
                    t = np.array(sp.target)
                    glob = GlobalAlignment(a, op.centre, t, (0.0, 0.0), mag, i)
                    mag_changed = False
        except Exception as e:  # noqa: BLE001 - reported per step
            sp.warnings.append(f"{type(e).__name__}: {e}")
        sp.field_um, sp.current_pa = field_um, cur
    return out


def _plan_expose(sp, op: ExposeField, i, glob, frame, field_um, library, mag_changed, mag) -> None:
    target = field_target(op, glob, frame)
    sp.target = (float(target[0]), float(target[1]))
    t = FieldTransform(glob.linear, (0.0, 0.0)) if glob is not None else frame.nominal_transform()
    sp.transform = t
    half = field_um / 2.0
    if target[0] < 0 or target[1] < 0:
        sp.warnings.append("stage target below 0 (the stage refuses negative coordinates)")
    if mag_changed and not op.global_align:
        sp.warnings.append("the magnification changed since the last Global Align and this field has none")
    if op.global_align:
        if not 2 <= len(op.markers) <= 4:
            sp.warnings.append(f"Global Align needs 2–4 markers (has {len(op.markers)})")
        elif len({m for m in op.markers}) != len(op.markers):
            sp.warnings.append("two Global Align markers are at the same position")
    marks = [(f"Global Align marker {k + 1}", m) for k, m in enumerate(op.active_markers)]
    if op.local_xy:
        marks.append(("Local XY marker", op.local_marker))
    for name, m in marks:
        b = t.to_bcs(m)
        if np.any(np.abs(b) > half):
            sp.warnings.append(f"{name} (WCS {m[0]:g}, {m[1]:g}) is outside the field (±{half:.1f} µm)")
    if marks and op.view_um > field_um:
        sp.warnings.append("the marker view is wider than the field")
    pat = library.get(op.pattern)
    if op.pattern.kind != "none":
        fc = check_field(pat, t.wcs_to_bcs, field_um)
        sp.fov_check = fc
        if not fc.ok:
            sp.warnings.append("pattern outside the field of view: " + fc.message())
        if len(pat) == 0:
            sp.warnings.append("empty pattern")
    elif not marks:
        sp.warnings.append("no pattern and no alignment: the step only moves the stage")
