"""GDSII import (via ``gdstk``) with per-polygon dose factors.

Dose information can come from:

* ``"property"`` – a GDS property (PROPATTR/PROPVALUE) on each element whose
  value is (or contains) a number, e.g. ``"1.25"`` or ``"dose=1.25"``;
* ``"datatype"`` / ``"layer"`` – a lookup table ``{datatype: factor}``.  This
  is how dose classes are commonly carried in fractured/PEC'd GDS exports
  (e.g. from BEAMER): configure the table from the export's dose list;
* ``"auto"`` – property if present, else the datatype table, else default;
* ``"none"`` – every element gets ``default``.

Coordinates are returned in µm.  Hierarchy and repetitions are flattened.
Paths with non-zero width become polygons; zero-width paths become
single-pass lines (if the reader keeps them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import numpy as np

from .geometry import Pattern, Polygon, Polyline

__all__ = ["GDSDoseMapping", "read_gds", "write_gds", "gds_info", "parse_dose_value", "GDSError"]

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class GDSError(RuntimeError):
    pass


def _gdstk():
    try:
        import gdstk
    except ImportError as e:  # pragma: no cover
        raise GDSError("gdstk is required for GDS support (pip install gdstk)") from e
    return gdstk


def parse_dose_value(value) -> float | None:
    """Extract the first number from a property value (str/bytes)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("latin-1", "ignore")
    m = _NUM.search(str(value).replace("\x00", ""))
    return float(m.group()) if m else None


@dataclass
class GDSDoseMapping:
    mode: str = "auto"  # none | property | datatype | layer | auto
    property_attr: int | None = None  # None = first property with a number
    table: dict = field(default_factory=dict)  # datatype (or layer) -> factor
    default: float = 1.0
    value_scale: float = 1.0  # e.g. 0.01 if the property holds percent

    def __post_init__(self):
        if self.mode not in ("none", "property", "datatype", "layer", "auto"):
            raise ValueError("invalid dose mapping mode")

    def _from_property(self, obj) -> float | None:
        props = getattr(obj, "properties", None) or []
        for entry in props:
            # gdstk: ['S_GDS_PROPERTY', attr, b'value\x00']
            if len(entry) >= 3 and entry[0] == "S_GDS_PROPERTY":
                attr, val = entry[1], entry[2]
                if self.property_attr is not None and attr != self.property_attr:
                    continue
                v = parse_dose_value(val)
                if v is not None:
                    return v * self.value_scale
        return None

    def factor(self, obj, layer: int, datatype: int) -> tuple[float, str]:
        """Return (dose_factor, source)."""
        if self.mode in ("property", "auto"):
            v = self._from_property(obj)
            if v is not None:
                return v, "property"
            if self.mode == "property":
                return self.default, "default"
        if self.mode in ("datatype", "auto") and datatype in self.table:
            return float(self.table[datatype]), "datatype"
        if self.mode == "layer" and layer in self.table:
            return float(self.table[layer]), "layer"
        return self.default, "default"


def _layer_ok(layer: int, datatype: int, layers) -> bool:
    if layers is None:
        return True
    for sel in layers:
        if isinstance(sel, (tuple, list)):
            if (layer, datatype) == tuple(sel):
                return True
        elif layer == sel:
            return True
    return False


def gds_info(path: str) -> dict:
    """Cells, top cells and (layer, datatype) pairs present in a file."""
    gdstk = _gdstk()
    lib = gdstk.read_gds(str(path), unit=1e-6)
    pairs = set()
    for c in lib.cells:
        for p in c.polygons:
            pairs.add((p.layer, p.datatype))
        for p in c.paths:
            for l, d in zip(p.layers, p.datatypes):
                pairs.add((l, d))
    return {
        "cells": [c.name for c in lib.cells],
        "top": [c.name for c in lib.top_level()],
        "layers": sorted(pairs),
        "unit_um": 1.0,
    }


def read_gds(
    path: str,
    cell: str | None = None,
    layers: Iterable | None = None,
    dose: GDSDoseMapping | None = None,
    stats: dict | None = None,
) -> Pattern:
    """Load a GDS file into a :class:`Pattern` (µm, flattened).

    ``layers``: iterable of layer numbers or (layer, datatype) tuples to keep.
    ``stats``: optional dict filled with counts per dose source.
    """
    gdstk = _gdstk()
    dose = dose or GDSDoseMapping()
    lib = gdstk.read_gds(str(path), unit=1e-6)
    if cell is None:
        tops = lib.top_level()
        if len(tops) != 1:
            raise GDSError(f"file has {len(tops)} top cells {[c.name for c in tops]}; specify one")
        src = tops[0]
    else:
        found = [c for c in lib.cells if c.name == cell]
        if not found:
            raise GDSError(f"cell '{cell}' not found")
        src = found[0]
    flat = src.copy(f"{src.name}__flat", deep_copy=True)
    flat.flatten()
    layers = list(layers) if layers is not None else None
    els = []
    counts: dict = {}

    def add(e, src_name):
        els.append(e)
        counts[src_name] = counts.get(src_name, 0) + 1

    for p in flat.polygons:
        if not _layer_ok(p.layer, p.datatype, layers):
            continue
        f, s = dose.factor(p, p.layer, p.datatype)
        try:
            add(Polygon(np.array(p.points), f, p.layer, p.datatype), s)
        except ValueError:
            continue  # degenerate
    for path_obj in flat.paths:
        widths = np.atleast_2d(path_obj.widths())
        zero = bool(np.all(widths == 0))
        if zero:
            spine = np.array(path_obj.spine())
            for l, d in zip(path_obj.layers, path_obj.datatypes):
                if _layer_ok(l, d, layers):
                    f, s = dose.factor(path_obj, l, d)
                    add(Polyline(spine, f, l, d), s)
        else:
            for poly in path_obj.to_polygons():
                if not _layer_ok(poly.layer, poly.datatype, layers):
                    continue
                f, s = dose.factor(path_obj, poly.layer, poly.datatype)
                try:
                    add(Polygon(np.array(poly.points), f, poly.layer, poly.datatype), s)
                except ValueError:
                    continue
    if stats is not None:
        stats.update(counts)
    return Pattern(els, name=src.name, source=str(path))


def write_gds(
    pattern: Pattern,
    path: str,
    cell_name: str = "TOP",
    dose_property_attr: int | None = 1,
    layer: int | None = None,
    datatype_from_dose: dict | None = None,
) -> None:
    """Write a pattern (polygons + lines as zero-width paths are skipped) to GDS.

    Mainly for tests/demos.  Dose factors are stored as a property and/or
    encoded as datatype via ``datatype_from_dose`` ({factor: datatype}).
    """
    gdstk = _gdstk()
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell(cell_name)
    for e in pattern:
        if not isinstance(e, Polygon):
            continue
        dt = e.datatype or 0
        if datatype_from_dose is not None:
            dt = datatype_from_dose.get(e.dose_factor, dt)
        p = gdstk.Polygon(e.points, layer=e.layer if layer is None and e.layer is not None else (layer or 0), datatype=dt)
        if dose_property_attr is not None:
            p.set_gds_property(dose_property_attr, f"{e.dose_factor:.6g}")
        c.add(p)
    lib.write_gds(str(path))
