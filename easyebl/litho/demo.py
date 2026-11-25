"""Demo files for the Lithography module: a fractured multi-field GDS and schedules for it.

    python -m easyebl.litho.demo [folder]      # default: examples/litho

writes

* ``multifield_demo.gds`` – a 3 × 3 array of 80 µm fields on a 100 µm pitch, placed by the
  top cell ``TOP`` (references, as a fracturing tool would write them).  Layer 1/0: the
  exposure pattern of every field (grating, dot array, ring, a single-pixel line, field
  label, corner marks at (±40, ±40) µm); layer 2/0: alignment crosses at (±45, ±45) µm around each field centre (meant to be
  on the sample already – not exposed);
* ``multifield_demo.ebls.json`` – the imported schedule: a Column setting (×1000 → 100 µm
  field, beam kept as is), a Field correction (at the position set in Field → Field
  calibration settings…), Global Align on the first field (two crosses), Local XY on every
  field, layout origin at stage (5000, 5000) µm, layer 1 only;
* ``dummy_demo.ebls.json`` – two dummy-pattern fields without alignment and one
  alignment-only field (no pattern), for trying the run without a GDS.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .fracture import ImportOptions, import_fractured_gds
from .schedule import ColumnSetting, ExposeField, FieldCorrection, PatternSpec, Schedule

__all__ = ["make_multifield_gds", "make_demo_files", "DEMO_ORIGIN", "FIELD_PITCH"]

DEMO_ORIGIN = (5000.0, 5000.0)  # stage position of the layout origin
FIELD_PITCH = 100.0
MARK = 45.0  # alignment crosses at (±MARK, ±MARK) around each field centre


def _cross(gdstk, cx, cy, arm=4.0, w=0.5, layer=2):
    return [gdstk.rectangle((cx - arm, cy - w / 2), (cx + arm, cy + w / 2), layer=layer),
            gdstk.rectangle((cx - w / 2, cy - arm), (cx + w / 2, cy + arm), layer=layer)]


def make_multifield_gds(path: str, rows: int = 3, cols: int = 3, pitch: float = FIELD_PITCH) -> str:
    """Write the multi-field demo GDS (see module docstring); fields at (c·pitch, r·pitch), r = 0 at the top."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("TOP")
    for r in range(rows):
        for c in range(cols):
            cell = lib.new_cell(f"FIELD_R{r + 1}C{c + 1}")
            # grating: 8 lines 0.5 µm wide on a 2 µm pitch (upper left)
            for k in range(8):
                x0 = -35.0 + 2.0 * k
                cell.add(gdstk.rectangle((x0, 10.0), (x0 + 0.5, 35.0), layer=1))
            # dot array 6 × 6 on 3 µm (upper right): tiny squares
            for i in range(6):
                for j in range(6):
                    x, y = 12.0 + 3.0 * i, 12.0 + 3.0 * j
                    cell.add(gdstk.rectangle((x - 0.1, y - 0.1), (x + 0.1, y + 0.1), layer=1))
            # ring (lower left), dose factor 1.2 as a property
            ring = gdstk.ellipse((-22.0, -22.0), 10.0, inner_radius=7.0, layer=1, tolerance=0.05)
            ring.set_gds_property(1, "1.2")
            cell.add(ring)
            # single-pixel line (zero-width path, lower middle)
            cell.add(gdstk.FlexPath([(-5.0, -35.0), (5.0, -10.0)], 0, layer=1))
            # field label (lower right)
            for poly in gdstk.text(f"R{r + 1}C{c + 1}", 6.0, (8.0, -30.0), layer=1):
                cell.add(poly)
            # corner marks: 2 µm L-shapes at (±40, ±40), so the field's bbox is centred on its origin
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = sx * 40.0, sy * 40.0
                    cell.add(gdstk.rectangle((x, y), (x - sx * 2.0, y - sy * 0.4), layer=1))
                    cell.add(gdstk.rectangle((x, y), (x - sx * 0.4, y - sy * 2.0), layer=1))
            # alignment crosses (layer 2: on the sample already)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for p in _cross(gdstk, sx * MARK, sy * MARK):
                        cell.add(p)
            top.add(gdstk.Reference(cell, (c * pitch, -r * pitch)))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lib.write_gds(str(path))
    return path


def make_demo_files(folder: str) -> dict:
    """Write the demo GDS and schedules into *folder*; returns their paths."""
    os.makedirs(folder, exist_ok=True)
    gds = make_multifield_gds(os.path.join(folder, "multifield_demo.gds"))
    opts = ImportOptions(layers=[(1, 0)], order="raster", origin_scs=DEMO_ORIGIN, magnification=1000,
                         global_align=True, global_markers=((-MARK, -MARK), (MARK, MARK)),
                         local_xy=True, local_marker=(-MARK, MARK), view_um=10.0, dwell_us=2.0, step_nm=20.0)
    s = import_fractured_gds(gds, opts)
    s.operations.insert(1, FieldCorrection())  # at the position in Field → Field calibration settings…
    s.name = "multifield demo"
    s.save(os.path.join(folder, "multifield_demo.ebls.json"))

    ox, oy = DEMO_ORIGIN
    d = Schedule([ColumnSetting(magnification=1000, current_pa=100.0), FieldCorrection(),
                  ExposeField(label="dummy 1", centre=(ox, oy), pattern=PatternSpec("dummy", size_um=40.0)),
                  ExposeField(label="dummy 2", centre=(ox + 100.0, oy), pattern=PatternSpec("dummy", size_um=60.0),
                              dwell_us=0.5),
                  ExposeField(label="align only", centre=(ox + 200.0, oy), pattern=PatternSpec("none"),
                              global_align=True, markers=[(-MARK, -MARK), (MARK, MARK)])],
                 name="dummy demo")
    d.save(os.path.join(folder, "dummy_demo.ebls.json"))
    return {"gds": gds, "schedule": s.path, "dummy": d.path}


if __name__ == "__main__":  # pragma: no cover
    out = make_demo_files(sys.argv[1] if len(sys.argv) > 1 else os.path.join("examples", "litho"))
    for k, v in out.items():
        print(f"{k}: {v}")
