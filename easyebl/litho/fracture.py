"""Fractured-GDS importer: one Expose field per cell.

Large patterns are fractured beforehand (e.g. by BEAMER) into cells that each fit in the
field of view.  The fields of a file are either

* ``"references"`` – the cell references placed in the single top cell (each placement,
  array elements included, is one field, at its placed position), or
* ``"cells"`` – cells used as they are (layout coordinates), by default every top cell;

``"auto"`` picks *references* for one top cell that places other cells, else *cells*.

:func:`build_schedule` makes a Column setting, then one Expose field per field:

* field centre (SCS) = the stage position of the layout origin + the field's bbox centre
  (through the stage axes), corrected by the last Global Align;
* pattern = the field, recentred on its bbox centre (WCS);
* optional Global Align on the first field and Local XY on every field; markers are WCS
  (µm from the field centre, in layout orientation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import numpy as np

from ..core.transforms import AffineTransform
from .coords import StageAxes
from .gds import GDSDoseMapping, GDSError
from .schedule import ColumnSetting, ExposeField, PatternLibrary, PatternSpec, Schedule

__all__ = ["FractureField", "ImportOptions", "find_fields", "order_fields", "build_schedule",
           "import_fractured_gds", "reference_transform"]


@dataclass
class FractureField:
    name: str
    path: str
    cell: str
    placement: AffineTransform  # cell → layout
    bbox: tuple  # layout µm
    n_elements: int

    @property
    def center(self) -> tuple:
        b = self.bbox
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def size(self) -> tuple:
        b = self.bbox
        return (b[2] - b[0], b[3] - b[1])


@dataclass
class ImportOptions:
    source: str = "auto"  # auto | references | cells
    top: str | None = None
    cells: list | None = None
    layers: list | None = None
    dose: GDSDoseMapping = field(default_factory=GDSDoseMapping)
    order: str = "file"  # file | raster (rows top → bottom, left → right) | serpentine
    origin_scs: tuple = (0.0, 0.0)  # stage position of the layout origin (0, 0)
    axes: StageAxes = field(default_factory=StageAxes)
    magnification: int = 1000  # one of the SEM's permitted magnifications
    beam_preset: str = ""  # "" keeps the beam as it is
    current_pa: float | None = None
    global_align: bool = False
    global_markers: tuple = ((-40.0, -40.0), (40.0, 40.0))  # WCS of the first field
    local_xy: bool = False
    local_marker: tuple = (-40.0, -40.0)  # WCS of every field
    view_um: float = 10.0
    dwell_us: float = 1.0
    step_nm: float = 20.0


def reference_transform(ref) -> AffineTransform:
    """gdstk reference placement (x reflection, magnification, rotation, then origin) as a transform."""
    m = AffineTransform.translation(*ref.origin) @ AffineTransform.rotation(ref.rotation or 0.0) \
        @ AffineTransform.scaling(ref.magnification or 1.0)
    if ref.x_reflection:
        m = m @ AffineTransform.scaling(1.0, -1.0)
    return m


def _placements(ref) -> list:
    """Cell → layout transform of every placement of *ref* (arrays: one per element; the library is not modified)."""
    t = reference_transform(ref)
    rep = ref.repetition
    if rep is None or rep.size <= 1:
        return [t]
    return [AffineTransform.translation(float(x), float(y)) @ t for x, y in rep.get_offsets()]


def find_fields(path: str, options: ImportOptions | None = None, library: PatternLibrary | None = None) -> list:
    """The non-empty fields of a fractured GDS file (see module docstring), in file order."""
    o = options or ImportOptions()
    library = library or PatternLibrary()
    lib = library.library(path)
    tops = lib.top_level()
    source = o.source
    top = None
    if source in ("auto", "references"):
        if o.top:
            top = next((c for c in lib.cells if c.name == o.top), None)
            if top is None:
                raise GDSError(f"cell {o.top!r} not found")
        elif len(tops) == 1:
            top = tops[0]
        if source == "auto":
            source = "references" if top is not None and top.references else "cells"
        elif top is None:
            raise GDSError(f"file has {len(tops)} top cells; name the one that places the fields")
    path = os.path.abspath(path)
    items = []
    if source == "references":
        k = 0
        for r in top.references:
            name = r.cell.name if hasattr(r.cell, "name") else str(r.cell)
            for t in _placements(r):
                k += 1
                items.append((f"{name} #{k}", name, t))
    elif source == "cells":
        for name in (o.cells if o.cells is not None else [c.name for c in tops]):
            items.append((name, name, AffineTransform.identity()))
    else:
        raise ValueError("source must be 'auto', 'references' or 'cells'")
    out = []
    for label, cell, t in items:
        pat = library.get(PatternSpec("gds", path, cell, o.layers, o.dose, t))
        if len(pat):
            out.append(FractureField(label, path, cell, t, pat.bbox, len(pat)))
    return out


def order_fields(fields: list, order: str = "file") -> list:
    if order == "file" or len(fields) < 2:
        return list(fields)
    if order not in ("raster", "serpentine"):
        raise ValueError("order must be 'file', 'raster' or 'serpentine'")
    h = float(np.median([max(f.size[1], 1e-9) for f in fields]))
    rows: dict = {}
    for f in fields:
        rows.setdefault(int(math.floor(-f.center[1] / (h / 2.0) + 0.5)), []).append(f)
    out = []
    for i, key in enumerate(sorted(rows)):
        row = sorted(rows[key], key=lambda f: f.center[0])
        if order == "serpentine" and i % 2:
            row.reverse()
        out.extend(row)
    return out


def build_schedule(fields: list, options: ImportOptions | None = None, name: str = "") -> Schedule:
    o = options or ImportOptions()
    s = Schedule([ColumnSetting(preset=o.beam_preset, magnification=o.magnification,
                                current_pa=o.current_pa)], name=name)
    origin = np.asarray(o.origin_scs, float)

    def scs(layout_pt):
        return tuple(float(v) for v in o.axes.wcs_to_scs([layout_pt], origin)[0])

    for i, f in enumerate(order_fields(fields, o.order)):
        cx, cy = f.center
        spec = PatternSpec("gds", f.path, f.cell, o.layers, o.dose,
                           AffineTransform.translation(-cx, -cy) @ f.placement)
        op = ExposeField(label=f.name, centre=scs((cx, cy)), correct_centre=True, pattern=spec, dwell_us=o.dwell_us,
                         step_nm=o.step_nm, view_um=o.view_um)
        if i == 0 and o.global_align:
            op.global_align = True
            op.markers = [(float(x), float(y)) for x, y in o.global_markers]
        if o.local_xy:
            op.local_xy = True
            op.local_marker = (float(o.local_marker[0]), float(o.local_marker[1]))
        s.add(op)
    return s


def import_fractured_gds(path: str, options: ImportOptions | None = None,
                         library: PatternLibrary | None = None) -> Schedule:
    o = options or ImportOptions()
    fields = find_fields(path, o, library)
    if not fields:
        raise GDSError("no non-empty fields found")
    return build_schedule(fields, o, os.path.splitext(os.path.basename(path))[0])
