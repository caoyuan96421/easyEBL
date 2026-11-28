"""GDSII import (via ``gdstk``) with per-shape dose factors from the datatype.

The dose is decided **layer by layer** over the whole file: a layer whose shapes all have
datatype 0 is written at the plain dose (factor 1); a layer with any other datatype carries
its dose in it – factor = datatype / :data:`DOSE_SCALE` (2/500 → 0.500, 2/1000 → 1.000,
2/1250 → 1.250).  Datatype 0 on such a layer is factor 0 (not written; :func:`dose_summary`
reports it).

Coordinates are returned in µm.  Hierarchy and repetitions are flattened.
Paths with non-zero width become polygons; zero-width paths become
single-pass lines (if the reader keeps them).
"""
from __future__ import annotations


from typing import Iterable

import numpy as np

from .geometry import Pattern, Polygon, Polyline

__all__ = ["DOSE_SCALE", "dose_factor", "dose_summary", "dose_text", "dose_warnings", "encoded_layers", "read_gds",
           "load_library", "pattern_from_cell", "write_gds", "gds_info", "GDSError"]

#: datatype / DOSE_SCALE = the dose factor on a dose-encoded layer
DOSE_SCALE = 1000
MAX_DATATYPE = 65535


class GDSError(RuntimeError):
    pass


def _gdstk():
    try:
        import gdstk
    except ImportError as e:  # pragma: no cover
        raise GDSError("gdstk is required for GDS support (pip install gdstk)") from e
    return gdstk


def _shapes_ld(lib):
    """(layer, datatype) of every polygon and path of every cell of a gdstk library (not flattened)."""
    for c in lib.cells:
        for p in c.polygons:
            yield p.layer, p.datatype
        for p in c.paths:
            yield from zip(p.layers, p.datatypes)


def encoded_layers(lib) -> frozenset:
    """The layers of *lib* that carry the dose in the datatype (any datatype other than 0)."""
    return frozenset(l for l, d in _shapes_ld(lib) if d != 0)


def dose_factor(layer: int, datatype: int, encoded) -> float:
    """The dose factor of a shape on *layer* / *datatype* (*encoded*: :func:`encoded_layers`)."""
    return datatype / DOSE_SCALE if layer in encoded else 1.0


def dose_summary(lib, layers: Iterable | None = None) -> dict:
    """{layer: {"levels": sorted dose factors, "uniform": bool, "zero": shapes with datatype 0 on an
    encoded layer}} of the layers kept by *layers* (None: all) – counted per cell definition."""
    layers = list(layers) if layers is not None else None
    enc = encoded_layers(lib)
    out: dict = {}
    for l, d in _shapes_ld(lib):
        if not _layer_ok(l, d, layers):
            continue
        e = out.setdefault(l, {"levels": set(), "uniform": l not in enc, "zero": 0})
        e["levels"].add(dose_factor(l, d, enc))
        if l in enc and d == 0:
            e["zero"] += 1
    for e in out.values():
        e["levels"] = sorted(e["levels"])
    return dict(sorted(out.items()))


def dose_warnings(summary: dict) -> list:
    """The warnings of a :func:`dose_summary`: datatype 0 among dose-encoded datatypes."""
    return [f"layer {l}: {e['zero']} shape(s) with datatype 0 among dose-encoded datatypes → dose 0, not written"
            for l, e in summary.items() if e["zero"]]


def dose_text(summary: dict) -> str:
    """One line per layer: "layer 1: uniform (datatype 0)", "layer 2: dose factor 0.500 … 1.000 (2 levels)"."""
    parts = []
    for l, e in summary.items():
        lv = [v for v in e["levels"] if v > 0]
        if e["uniform"]:
            t = "uniform (datatype 0)"
        elif len(lv) == 1:
            t = f"dose factor {lv[0]:.3f}"
        elif lv:
            t = f"dose factor {lv[0]:.3f} … {lv[-1]:.3f} ({len(lv)} levels)"
        else:
            t = "dose 0"
        if e["zero"]:
            t += f"; {e['zero']} shape(s) datatype 0 → dose 0 ⚠"
        parts.append(f"layer {l}: {t}")
    return "\n".join(parts) or "no shapes on the selected layers"


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
    stats: dict | None = None,
) -> Pattern:
    """Load a GDS file into a :class:`Pattern` (µm, flattened).

    ``layers``: iterable of layer numbers or (layer, datatype) tuples to keep.
    ``stats``: optional dict filled with counts per dose source ("uniform" / "datatype").
    """
    lib = load_library(path)
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
    pat = pattern_from_cell(src, layers, encoded_layers(lib), stats)
    pat.source = str(path)
    return pat


def load_library(path: str):
    """The gdstk library of a file (µm units) – read once, then :func:`pattern_from_cell` per cell."""
    return _gdstk().read_gds(str(path), unit=1e-6)


def pattern_from_cell(
    src,
    layers: Iterable | None = None,
    encoded=None,
    stats: dict | None = None,
) -> Pattern:
    """Flatten a gdstk cell (of a library read in µm) into a :class:`Pattern`; *src* is not modified.
    *encoded*: the dose-encoded layers of the file (:func:`encoded_layers`; None: those of *src*)."""
    flat = src.copy(f"{src.name}__flat", deep_copy=True)
    flat.flatten()
    if encoded is None:
        pairs = [(p.layer, p.datatype) for p in flat.polygons] + [x for p in flat.paths
                                                                  for x in zip(p.layers, p.datatypes)]
        encoded = frozenset(l for l, d in pairs if d != 0)

    def factor(l, d):
        return dose_factor(l, d, encoded), ("datatype" if l in encoded else "uniform")

    layers = list(layers) if layers is not None else None
    els = []
    counts: dict = {}

    def add(e, src_name):
        els.append(e)
        counts[src_name] = counts.get(src_name, 0) + 1

    for p in flat.polygons:
        if not _layer_ok(p.layer, p.datatype, layers):
            continue
        f, s = factor(p.layer, p.datatype)
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
                    f, s = factor(l, d)
                    add(Polyline(spine, f, l, d), s)
        else:
            for poly in path_obj.to_polygons():
                if not _layer_ok(poly.layer, poly.datatype, layers):
                    continue
                f, s = factor(poly.layer, poly.datatype)
                try:
                    add(Polygon(np.array(poly.points), f, poly.layer, poly.datatype), s)
                except ValueError:
                    continue
    if stats is not None:
        stats.update(counts)
    return Pattern(els, name=src.name)


def write_gds(pattern: Pattern, path: str, cell_name: str = "TOP", layer: int | None = None) -> None:
    """Write a pattern's polygons to GDS (lines are skipped), each with its dose factor as the
    datatype (factor × :data:`DOSE_SCALE`: the file reads back with the same doses)."""
    gdstk = _gdstk()
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    c = lib.new_cell(cell_name)
    for e in pattern:
        if not isinstance(e, Polygon):
            continue
        dt = int(round(e.dose_factor * DOSE_SCALE))
        if not 0 <= dt <= MAX_DATATYPE:
            raise GDSError(f"dose factor {e.dose_factor:g} does not fit a datatype (0 … {MAX_DATATYPE / DOSE_SCALE:g})")
        lay = e.layer if layer is None and e.layer is not None else (layer or 0)
        c.add(gdstk.Polygon(e.points, layer=lay, datatype=dt))
    lib.write_gds(str(path))
