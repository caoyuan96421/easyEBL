"""Demo files for the Lithography module: a fractured multi-field GDS and schedules for it.

    python -m easyebl.litho.demo [folder]      # default: examples/litho

writes

* ``multifield_demo.gds`` – a 3 × 3 array of 80 µm fields on a 100 µm pitch, placed by the
  top cell ``TOP`` (references, as a fracturing tool would write them).  Layer 1/0: the
  exposure pattern of every field (grating, dot array, ring, a single-pixel line, field
  label, corner marks at (±40, ±40) µm); layer 2/0: alignment crosses at (±45, ±45) µm around
  each field centre (meant to be on the sample already – not exposed);
* ``multifield_demo.ebls.json`` – chip "Demo chip" (CCS origin at stage (5000, 5000) µm, no
  height: Z kept), the fixed steps – Column setting (×1000 → 100 µm field, beam kept as is, WD
  15 mm), Beam calibration at the stage position "Calibration grid", Field calibration at
  "Faraday cup" (record both in the Stage Positions tab, in focus) – and one Exposure of the
  whole GDS (layer 1): Global Align on two crosses of the first field, a Local marker (the
  upper-left cross) in every field;
* ``dummy_demo.ebls.json`` – the same GDS written without alignment, then an alignment-only
  Exposure (two crosses), for trying the run.
"""
from __future__ import annotations

import os
import sys

from .schedule import BeamCalibrationStep, Chip, ColumnSetting, Exposure, FieldCorrection, Schedule

__all__ = ["make_multifield_gds", "make_demo_files", "DEMO_ORIGIN", "FIELD_PITCH", "GRID_POSITION", "CUP_POSITION"]

DEMO_ORIGIN = (5000.0, 5000.0)  # stage position of the CCS (layout) origin
GRID_POSITION = "Calibration grid"  # the stage positions the calibration steps use
CUP_POSITION = "Faraday cup"
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

def demo_local_markers(rows: int = 3, cols: int = 3, pitch: float = FIELD_PITCH) -> list:
    """The upper-left cross of every field (CCS)."""
    return [(c * pitch - MARK, -r * pitch + MARK) for r in range(rows) for c in range(cols)]

def make_demo_files(folder: str) -> dict:
    """Write the demo GDS and schedules into *folder*; returns their paths."""
    os.makedirs(folder, exist_ok=True)
    gds = os.path.abspath(make_multifield_gds(os.path.join(folder, "multifield_demo.gds")))
    chip = Chip("Demo chip", DEMO_ORIGIN)

    def fixed(**column):
        b, f = BeamCalibrationStep(), FieldCorrection()
        b.position, f.position = GRID_POSITION, CUP_POSITION
        return [ColumnSetting(magnification=1000, **column), b, f]

    s = Schedule(fixed() + [
                  Exposure(chip=chip.name, path=gds, layers=[(1, 0)], order="raster", dwell_us=2.0, step_nm=20.0,
                           global_align=True, global_markers=[(-MARK, -MARK), (MARK, MARK)],
                           local_align=True, local_markers=demo_local_markers(), view_um=10.0)],
                 wd_mm=15.0,
                 name="multifield demo", chips=[chip])
    s.save(os.path.join(folder, "multifield_demo.ebls.json"))

    d = Schedule(fixed(current_pa=100.0) + [
                  Exposure(label="no alignment", chip=chip.name, path=gds, layers=[(1, 0)], dwell_us=0.5),
                  Exposure(label="align only", chip=chip.name, global_align=True,
                           global_markers=[(-MARK, -MARK), (MARK, MARK)])],
                 name="dummy demo", chips=[Chip("Demo chip", DEMO_ORIGIN)])
    d.save(os.path.join(folder, "dummy_demo.ebls.json"))
    return {"gds": gds, "schedule": s.path, "dummy": d.path}

if __name__ == "__main__":  # pragma: no cover
    out = make_demo_files(sys.argv[1] if len(sys.argv) > 1 else os.path.join("examples", "litho"))
    for k, v in out.items():
        print(f"{k}: {v}")
