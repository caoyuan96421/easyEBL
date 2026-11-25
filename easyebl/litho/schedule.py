"""The exposure schedule (no Qt): chips, and Column setting / Field calibration / Exposure steps.

* **Chip** – a sample with its own chip coordinates (CCS, :mod:`easyebl.litho.coords`): a name
  and the stage position of the CCS origin (nominal, through the stage axes).  A Global Align
  measures where the chip really is (CCS → SCS, affine); that holds for every later Exposure on
  the chip.
* **Column setting** (``column``) – beam preset of the microscope module (or keep as is),
  magnification (sets the field from the global field calibration), optional nominal beam
  current (for the dose readout).  Every schedule starts with one; more may follow.
* **Field calibration** (``field_correction``) – the stage → beam map at a stage position; valid
  for the column condition (magnification, beam preset, WD) and beam calibration it was measured at.
* **Exposure** (``exposure``) – a whole GDS file on a chip.  Its origin is the CCS origin + an
  offset (no rotation).  Each cell of the top level is one *field*: its centre (the cell's
  bounding-box centre, or its origin) is the field centre, its coordinates relative to that are
  FCS.  Markers are CCS points (= GDS top-level coordinates + offset):

  - Global Align (2–4 markers) – all inside the field of view of one field, which is written
    first; it measures the chip's CCS → SCS map;
  - Local markers – each assigned to the field whose field of view holds it (nearest centre),
    at most one per field: a shift for that field only.

  An Exposure without a GDS file aligns only (one field at the Global Align markers).

Saved as JSON (``format: easyebl-schedule``, version 3).  :func:`plan` walks a schedule without
hardware: fields, stage targets, maps and warnings per step.
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
from .coords import S3000_AXES, ChipTransform, FieldMap, StageAxes, check_field, stage_rotation
from .gds import GDSDoseMapping, load_library, pattern_from_cell
from .geometry import Dot, Pattern, Polyline, rectangle

__all__ = ["Operation", "ColumnSetting", "FieldCorrection", "Exposure", "Chip", "PatternSpec", "PatternLibrary",
           "FieldDef", "MarkerAssignment", "exposure_fields", "assign_markers", "ScheduleSettings", "Schedule",
           "StepPlan", "FieldPlan", "plan", "OP_TYPES", "SCHEDULE_FORMAT", "ScheduleError", "dummy_pattern",
           "KEEP_AS_IS", "dose_readout", "MAGNIFICATIONS", "nearest_magnification", "op_from_dict", "find_beamcal"]

SCHEDULE_FORMAT = "easyebl-schedule"
SCHEDULE_VERSION = 3
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


def _layers_from(v):
    if v is None:
        return None
    return [tuple(x) if isinstance(x, list) else x for x in v]


@dataclass
class PatternSpec:
    """A pattern to load: ``"gds"`` – *cell* of GDS file *path*, flattened, *layers* kept (None: all),
    doses from *dose*, then *placement* (cell → wanted coordinates); ``"dummy"`` –
    :func:`dummy_pattern` of *size_um*; ``"none"`` – nothing."""

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


class PatternLibrary:
    """Loads :class:`PatternSpec` patterns, reading each GDS file once."""

    def __init__(self):
        self._libs: dict = {}
        self._cells: dict = {}
        self._fields: dict = {}

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

    def top_cells(self, path: str) -> list:
        return [c.name for c in self.library(path).top_level()]

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

    def cached_fields(self, key, make):
        if key not in self._fields:
            self._fields[key] = make()
        return self._fields[key]

    def clear(self) -> None:
        self._libs.clear()
        self._cells.clear()
        self._fields.clear()


# ====================================================================== chips
@dataclass
class Chip:
    """A sample with its own CCS: *origin* is the nominal stage position of the CCS origin."""

    name: str = "Chip 1"
    origin: tuple = (0.0, 0.0)
    note: str = ""

    def __post_init__(self):
        self.origin = _xy(self.origin)

    def nominal(self, axes: StageAxes) -> ChipTransform:
        return ChipTransform.nominal(self.origin, axes)

    def to_dict(self) -> dict:
        return {"name": self.name, "origin": list(self.origin), "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Chip":
        return cls(str(d.get("name", "Chip 1")), d.get("origin", (0.0, 0.0)), str(d.get("note", "")))


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
    """Field calibration (stage → beam map) at a stage position, valid for the column condition it
    is measured at (see :mod:`easyebl.litho.fieldcal`)."""

    TYPE: ClassVar[str] = "field_correction"
    TITLE: ClassVar[str] = "Field calibration"
    position: str = ""  # a named stage position; "": the one in the field calibration settings
    notes: str = ""

    def summary(self) -> str:
        return f"at {self.position!r}" if self.position else "at the configured position"


ORDERS = ("file", "raster", "serpentine")


@dataclass
class Exposure(Operation):
    TYPE: ClassVar[str] = "exposure"
    TITLE: ClassVar[str] = "Exposure"
    chip: str = ""  # Chip.name
    path: str = ""  # GDS file ("": alignment only); relative paths: relative to the schedule file
    top: str = ""  # the top cell placing the fields ("": the only top-level cell)
    layers: list | None = None  # [(layer, datatype)] or [layer]; None: all
    dose: GDSDoseMapping = field(default_factory=GDSDoseMapping)
    offset: tuple = (0.0, 0.0)  # GDS origin in CCS
    cell_origin: bool = False  # field centre = the cell's origin (else its bounding-box centre)
    order: str = "file"  # file | raster | serpentine (after the Global Align field)
    dwell_us: float = 1.0  # per exposure point (× the GDS dose factor)
    step_nm: float = 20.0  # point spacing (area fill and lines)
    global_align: bool = False
    global_markers: list = field(default_factory=list)  # 2–4 CCS points, inside one field
    local_align: bool = False
    local_markers: list = field(default_factory=list)  # CCS points, at most one per field
    view_um: float = 10.0  # width of the zoomed SEM scan around a predicted marker

    def __post_init__(self):
        self.offset = _xy(self.offset)
        self.global_markers = [_xy(m) for m in self.global_markers]
        self.local_markers = [_xy(m) for m in self.local_markers]
        self.layers = _layers_from(self.layers)
        if isinstance(self.dose, dict):
            self.dose = _mapping_from_dict(self.dose)
        if self.order not in ORDERS:
            raise ScheduleError(f"order must be one of {', '.join(ORDERS)}")
        if not self.dwell_us > 0 or not self.step_nm > 0:
            raise ScheduleError("dwell and step must be positive")

    @property
    def alignment_only(self) -> bool:
        return not self.path

    def summary(self) -> str:
        al = []
        if self.global_align:
            al.append(f"global {len(self.global_markers)}")
        if self.local_align:
            al.append(f"local {len(self.local_markers)}")
        what = self.label or (os.path.basename(self.path) if self.path else "alignment only")
        return (f"{what} on {self.chip or '?'}" + (f" [{', '.join(al)}]" if al else "")
                + ("" if self.alignment_only else f", {self.dwell_us:g} µs"))

    def to_dict(self, base_dir: str | None = None) -> dict:
        d = super().to_dict(base_dir)
        path = self.path
        if base_dir and path and os.path.isabs(path):
            try:
                path = os.path.relpath(path, base_dir)
            except ValueError:  # another drive
                pass
        d["path"] = path.replace("\\", "/")
        d["dose"] = _mapping_to_dict(self.dose)
        d["layers"] = [list(x) if isinstance(x, (tuple, list)) else x for x in self.layers] \
            if self.layers is not None else None
        d["global_markers"] = [list(m) for m in self.global_markers]
        d["local_markers"] = [list(m) for m in self.local_markers]
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Exposure":
        d = dict(d)
        path = d.get("path", "")
        if base_dir and path and not os.path.isabs(path):
            d["path"] = os.path.normpath(os.path.join(base_dir, path))
        d["dose"] = _mapping_from_dict(d.get("dose"))
        return super().from_dict(d, base_dir)


OP_TYPES: dict = {c.TYPE: c for c in (ColumnSetting, FieldCorrection, Exposure)}


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


# ====================================================================== fields of an exposure
@dataclass
class FieldDef:
    """One field of an Exposure: a top-level cell (placement) with its centre."""

    name: str
    cell: str
    placement: AffineTransform  # cell → GDS
    bbox_gds: tuple
    centre_gds: tuple
    centre_ccs: tuple
    n_elements: int
    path: str = ""

    @property
    def size(self) -> tuple:
        b = self.bbox_gds
        return (b[2] - b[0], b[3] - b[1])

    @property
    def center(self) -> tuple:  # for fracture.order_fields
        return self.centre_gds

    def spec(self, exp: "Exposure") -> PatternSpec:
        """The field's pattern in FCS."""
        if not self.path:
            return PatternSpec("none")
        cx, cy = self.centre_gds
        return PatternSpec("gds", self.path, self.cell, exp.layers, exp.dose,
                           AffineTransform.translation(-cx, -cy) @ self.placement)

    def contains(self, ccs, field_um: float) -> bool:
        h = field_um / 2.0
        return bool(abs(ccs[0] - self.centre_ccs[0]) <= h and abs(ccs[1] - self.centre_ccs[1]) <= h)


def exposure_fields(exp: Exposure, library: PatternLibrary | None = None) -> list:
    """The non-empty fields of *exp* in file order (an alignment-only Exposure: one empty field
    at the centroid of its Global Align markers)."""
    from .fracture import ImportOptions, find_fields

    library = library or PatternLibrary()
    ox, oy = exp.offset
    if exp.alignment_only:
        pts = exp.global_markers or exp.local_markers
        if not pts:
            return []
        c = tuple(float(v) for v in np.mean(np.asarray(pts, float), axis=0))
        return [FieldDef("alignment", "", AffineTransform.identity(), (c[0], c[1], c[0], c[1]), (c[0] - ox, c[1] - oy),
                         c, 0)]
    key = (os.path.normcase(os.path.abspath(exp.path)), exp.top, None if exp.layers is None else tuple(exp.layers),
           json.dumps(_mapping_to_dict(exp.dose), sort_keys=True))
    raw = library.cached_fields(key, lambda: find_fields(
        exp.path, ImportOptions(source="auto", top=exp.top or None, layers=exp.layers, dose=exp.dose), library))
    out = []
    for f in raw:
        c = tuple(map(float, f.placement.offset)) if exp.cell_origin else f.center
        out.append(FieldDef(f.name, f.cell, f.placement, f.bbox, c, (c[0] + ox, c[1] + oy), f.n_elements, f.path))
    return out


@dataclass
class MarkerAssignment:
    global_field: int | None  # index into the fields (None: no Global Align / not in one field)
    local: dict  # field index -> CCS marker
    order: list  # field indices in writing order (the Global Align field first)
    warnings: list


def assign_markers(fields_: list, exp: Exposure, field_um: float) -> MarkerAssignment:
    """Global Align markers → the field whose field of view holds all of them (nearest centre);
    each Local marker → the field whose field of view holds it (nearest centre), one per field."""
    from .fracture import order_fields

    warns = []
    g = None
    if exp.global_align:
        mk = np.asarray(exp.global_markers, float).reshape(-1, 2)
        if not 2 <= len(mk) <= 4:
            warns.append(f"Global Align needs 2–4 markers (has {len(mk)})")
        elif len({tuple(m) for m in exp.global_markers}) != len(exp.global_markers):
            warns.append("two Global Align markers are at the same position")
        if len(mk):
            cand = [i for i, f in enumerate(fields_) if all(f.contains(m, field_um) for m in mk)]
            if cand:
                c = mk.mean(0)
                g = min(cand, key=lambda i: float(np.hypot(*(np.asarray(fields_[i].centre_ccs) - c))))
            else:
                warns.append(f"the Global Align markers are not all inside the field of view (±{field_um / 2:.1f} µm) "
                             "of one field")
    local: dict = {}
    if exp.local_align:
        for m in exp.local_markers:
            cand = [i for i, f in enumerate(fields_) if f.contains(m, field_um)]
            if not cand:
                warns.append(f"Local marker ({m[0]:g}, {m[1]:g}) is in no field's field of view")
                continue
            i = min(cand, key=lambda i: float(np.hypot(fields_[i].centre_ccs[0] - m[0], fields_[i].centre_ccs[1] - m[1])))
            if i in local:
                old = local[i]
                d_old = np.hypot(fields_[i].centre_ccs[0] - old[0], fields_[i].centre_ccs[1] - old[1])
                d_new = np.hypot(fields_[i].centre_ccs[0] - m[0], fields_[i].centre_ccs[1] - m[1])
                keep = m if d_new < d_old else old
                warns.append(f"field {fields_[i].name} has two Local markers ({old[0]:g}, {old[1]:g}) and "
                             f"({m[0]:g}, {m[1]:g}): using ({keep[0]:g}, {keep[1]:g})")
                local[i] = keep
            else:
                local[i] = m
        if g is not None and g in local:
            warns.append(f"field {fields_[g].name} is the Global Align field: its Local marker is not used")
            del local[g]
        missing = [f.name for i, f in enumerate(fields_) if i != g and i not in local]
        if missing and fields_ and not exp.alignment_only:
            warns.append(f"{len(missing)} field(s) without a Local marker: " + ", ".join(missing[:5])
                         + (" …" if len(missing) > 5 else ""))
    rest = [f for i, f in enumerate(fields_) if i != g]
    ordered = order_fields(rest, exp.order)
    idx = {id(f): i for i, f in enumerate(fields_)}
    order = ([g] if g is not None else []) + [idx[id(f)] for f in ordered]
    return MarkerAssignment(g, local, order, warns)


def find_beamcal(beamcals, condition):
    """The beam calibration for *condition* (magnification and WD), or None."""
    for bc in reversed(list(beamcals or [])):
        c = getattr(bc, "condition", None)
        if c is not None and c.matches(condition, preset=False):
            return bc
    return None


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
    chips: list = field(default_factory=list)  # [Chip]
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
    def exposures(self) -> list:
        return [o for o in self.operations if isinstance(o, Exposure)]

    # ---------------------------------------------------------------- chips
    def chip(self, name: str) -> Chip | None:
        return next((c for c in self.chips if c.name == name), None)

    def unique_chip_name(self, base: str = "Chip") -> str:
        names = {c.name for c in self.chips}
        k = 1
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def add_chip(self, name: str = "", origin=(0.0, 0.0), note: str = "") -> Chip:
        name = (name or self.unique_chip_name()).strip()
        if self.chip(name) is not None:
            raise ScheduleError(f"a chip named {name!r} exists already")
        c = Chip(name, origin, note)
        self.chips.append(c)
        return c

    def rename_chip(self, old: str, new: str) -> None:
        new = new.strip()
        if not new:
            raise ScheduleError("a chip needs a name")
        if new != old and self.chip(new) is not None:
            raise ScheduleError(f"a chip named {new!r} exists already")
        c = self.chip(old)
        if c is None:
            raise ScheduleError(f"no chip {old!r}")
        c.name = new
        for e in self.exposures:
            if e.chip == old:
                e.chip = new

    def shift_chip(self, name: str, dx: float, dy: float) -> None:
        c = self.chip(name)
        if c is None:
            raise ScheduleError(f"no chip {name!r}")
        c.origin = (c.origin[0] + float(dx), c.origin[1] + float(dy))

    # ---------------------------------------------------------------- file
    def to_dict(self, base_dir: str | None = None) -> dict:
        return {"format": SCHEDULE_FORMAT, "version": SCHEDULE_VERSION, "name": self.name,
                "settings": asdict(self.settings), "vector_settings": dict(self.vector_settings),
                "chips": [c.to_dict() for c in self.chips],
                "operations": [o.to_dict(base_dir) for o in self.operations]}

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Schedule":
        if d.get("format") != SCHEDULE_FORMAT:
            raise ScheduleError("not an easyEBL schedule file")
        v = int(d.get("version", 1))
        if v != SCHEDULE_VERSION:
            raise ScheduleError(f"schedule file version {v} is not supported (this program reads version "
                                f"{SCHEDULE_VERSION}: chips and Exposures of whole GDS files)")
        known = {f.name for f in fields(ScheduleSettings)}
        st = ScheduleSettings(**{k: v for k, v in (d.get("settings") or {}).items() if k in known})
        return cls([op_from_dict(o, base_dir) for o in d.get("operations", [])], d.get("name", ""), st,
                   dict(d.get("vector_settings") or {}), [Chip.from_dict(c) for c in d.get("chips", [])])

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
class FieldPlan:
    """One field of an Exposure step, as planned (assuming the alignments measure what is known)."""

    index: int  # into the Exposure's fields
    name: str
    centre_ccs: tuple
    target: tuple  # SCS
    map: FieldMap  # FCS → XCS
    local_marker: tuple | None = None
    is_global: bool = False
    fov_check: object = None
    warnings: list = field(default_factory=list)


@dataclass
class StepPlan:
    """What a step will do, computed without hardware."""

    index: int
    field_um: float
    current_pa: float | None
    warnings: list = field(default_factory=list)
    fields: list = field(default_factory=list)  # [FieldPlan] of an Exposure, in writing order
    chip: ChipTransform | None = None  # the chip map in force for the Exposure
    stage_to_beam: np.ndarray | None = None  # B used (SCS → BCS)
    calibrated: bool = False  # a field calibration applies
    beamcal: object = None  # the beam calibration that applies (None: none)
    condition: object = None

    @property
    def target(self) -> tuple | None:  # the first field's stage target
        return self.fields[0].target if self.fields else None


def plan(schedule: Schedule, field_at_1x_um: float, axes: StageAxes | None = None, presets=None,
         library: PatternLibrary | None = None, chips: dict | None = None, calibration=None,
         preset: str | None = None, positions: dict | None = None, fieldcal=None, beamcals=None,
         wd_mm: float | None = None) -> list:
    """Walk *schedule* without hardware.  *chips*: {chip name: measured ChipTransform} (else the
    nominal map from the chip origin; a Global Align is assumed to measure what is known);
    *calibration*: the field calibration, *beamcals*: the beam calibrations, *wd_mm*: the working
    distance now; *field_at_1x_um*: the global field calibration (field = K / mag); *preset*: the
    beam preset in force before the schedule (None: not known); *positions* / *fieldcal*: named
    stage positions and the field calibration settings.  One :class:`StepPlan` per operation."""
    from .fieldcal import Condition

    axes = axes or S3000_AXES
    library = library or PatternLibrary()
    positions = positions or {}
    chips = {k: v.copy() for k, v in (chips or {}).items()}
    field_um = float(field_at_1x_um) / schedule[0].magnification
    cur = None
    mag = None
    cal = calibration
    cal_measured_cond = None  # a Field calibration step in the schedule
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
            cond = Condition(mag, preset, wd_mm)
            if isinstance(op, ColumnSetting):
                preset = op.preset or preset
                if not op.permitted:
                    sp.warnings.append(f"magnification ×{op.magnification:g} is not one the SEM accepts "
                                       f"(nearest: ×{nearest_magnification(op.magnification)})")
                if presets is not None and op.preset and op.preset not in presets:
                    sp.warnings.append(f"no beam preset named {op.preset!r} in the microscope module")
                mag = op.magnification
                field_um = float(field_at_1x_um) / op.magnification
                cur = op.current_pa or cur
                cond = Condition(mag, preset, wd_mm)
                if find_beamcal(beamcals, cond) is None:
                    sp.warnings.append(f"no beam calibration for {cond.describe()} (Field → Run beam calibration): "
                                       "nominal beam")
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
                if find_beamcal(beamcals, cond) is None:
                    sp.warnings.append("measured without a beam calibration for this condition")
                cal_measured_cond = cond
            elif isinstance(op, Exposure):
                bc = find_beamcal(beamcals, cond)
                if cal_measured_cond is not None and cal_measured_cond.matches(cond):
                    ok, b = True, (cal.stage_to_beam if cal is not None else axes.matrix)
                else:
                    ok = cal is not None and cal.valid_for(cond, bc)
                    b = cal.stage_to_beam if ok else None
                bmat = stage_rotation(b, axes) if b is not None else axes.matrix
                sp.calibrated, sp.beamcal, sp.condition, sp.stage_to_beam = ok, bc, cond, bmat
                if not ok:
                    sp.warnings.append(f"no field calibration for {cond.describe()} (a Field calibration step, or "
                                       "Field → Run field calibration): nominal stage ↔ beam rotation")
                _plan_exposure(sp, op, schedule, chips, axes, bmat, bc, field_um, library, i)
        except Exception as e:  # noqa: BLE001 - reported per step
            sp.warnings.append(f"{type(e).__name__}: {e}")
        sp.field_um, sp.current_pa = field_um, cur
    return out


def _plan_exposure(sp: StepPlan, op: Exposure, schedule: Schedule, chips: dict, axes, bmat, bc, field_um,
                   library, i) -> None:
    chip = schedule.chip(op.chip)
    if chip is None:
        sp.warnings.append(f"no chip named {op.chip!r} (Chips tab)" if op.chip else "no chip chosen")
        return
    g = chips.get(chip.name) or chip.nominal(axes)
    sp.chip = g
    flds = exposure_fields(op, library)
    if not flds:
        sp.warnings.append("no fields: the GDS has no non-empty top-level cells" if op.path
                           else "alignment only, but no markers")
        return
    asg = assign_markers(flds, op, field_um)
    sp.warnings.extend(asg.warnings)
    if op.alignment_only and not op.global_align:
        sp.warnings.append("no GDS file and no Global Align: the step does nothing")
    if (op.global_align or op.local_align) and op.view_um > field_um:
        sp.warnings.append("the marker view is wider than the field")
    for k in asg.order:
        f = flds[k]
        tgt = g.to_scs(np.asarray(f.centre_ccs, float))
        fm = FieldMap.make(g, f.centre_ccs, tgt, bmat, beamcal=bc)
        fp = FieldPlan(k, f.name, f.centre_ccs, (float(tgt[0]), float(tgt[1])), fm, asg.local.get(k),
                       k == asg.global_field)
        if tgt[0] < 0 or tgt[1] < 0:
            fp.warnings.append("stage target below 0 (the stage refuses negative coordinates)")
        if not op.alignment_only:
            pat = library.get(f.spec(op))
            fc = check_field(pat, fm, field_um)
            fp.fov_check = fc
            if not fc.ok:
                fp.warnings.append("pattern outside the field of view: " + fc.message())
        sp.fields.append(fp)
        sp.warnings.extend(f"field {f.name}: {w}" for w in fp.warnings)
    if op.global_align and asg.global_field is not None:
        chips[chip.name] = g.copy()  # later Exposures on the chip use the (assumed) measured map
        chips[chip.name].measured = True
