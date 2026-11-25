"""Fields of a fractured GDS file: one field per top-level cell.

Large patterns are fractured beforehand (e.g. by BEAMER) into cells that each fit in the
field of view.  The fields of a file are either

* ``"references"`` – the cell references placed in the single top cell (each placement,
  array elements included, is one field, at its placed position), or
* ``"cells"`` – cells used as they are (layout coordinates), by default every top cell;

``"auto"`` picks *references* for one top cell that places other cells, else *cells*.
An :class:`~easyebl.litho.schedule.Exposure` writes all of them (:func:`find_fields`,
:func:`order_fields`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import numpy as np

from ..core.transforms import AffineTransform
from .gds import GDSDoseMapping, GDSError
from .schedule import PatternLibrary, PatternSpec

__all__ = ["FractureField", "ImportOptions", "find_fields", "order_fields", "reference_transform"]


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
