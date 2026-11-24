"""EDS data manager: every acquisition of a session (spectra, line scans, maps) kept in memory.

Pure bookkeeping (no Qt, no hardware): the EDS window lists the items, draws their
geometry on the electron image and runs the acquisitions.

* Every item has a **number** (one counter for all kinds, never reused) and an optional
  **name**.
* Its **geometry** is in fractions of the deflection field (centre 0, full field ±0.5):
  ``point`` (x, y) · ``rect`` / ``ellipse`` / ``field`` (cx, cy, w, h) · ``line``
  (x0, y0, x1, y1) · ``map`` (cx, cy, w, h) · ``spot`` () – the beam where the SEM put it.
* Its **conditions** hold the acquisition request (``request``: what *Remeasure* repeats)
  and the context (field, processor settings, microscope readbacks …).
* **Dirty** = data not saved since it was acquired (or renamed).  An item without data yet
  (a drawn site, a running acquisition) has nothing to save and is never dirty.

Files are the existing EDS formats: spectra EMSA ``.msa`` (or ``.npz``), maps and line scans
``.npz``.  The item's name, kind, geometry and conditions travel in the data's metadata
(``metadata["data_manager"]``), so an opened file can be remeasured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re

import numpy as np

from .spectrum import Spectrum, SpectrumMap, _jsonable, load_map, load_spectrum

__all__ = ["DataItem", "DataManager", "SPECTRUM", "LINE", "MAP", "open_file", "default_filename", "flatten"]

SPECTRUM, LINE, MAP = "spectrum", "line", "map"
KINDS = (SPECTRUM, LINE, MAP)
#: geometry shapes per item kind
SHAPES = {SPECTRUM: ("point", "rect", "ellipse", "field", "spot"), LINE: ("line",), MAP: ("map",)}
META_KEY = "data_manager"


@dataclass
class DataItem:
    number: int
    kind: str  # spectrum | line | map
    shape: str  # point | rect | ellipse | field | spot | line | map
    params: tuple = ()  # field fractions
    name: str = ""
    conditions: dict = field(default_factory=dict)
    data: Spectrum | SpectrumMap | None = None
    path: str | None = None  # where it was last saved / opened from
    dirty: bool = False
    source: str = ""  # fixed description: "map 3 pixel (2, 1)", a file name …
    remeasure_of: int | None = None  # the item this one repeats
    acquiring: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.shape not in SHAPES[self.kind]:
            raise ValueError(f"a {self.kind} cannot have the shape {self.shape!r}")
        self.params = tuple(float(p) for p in self.params)

    # ------------------------------------------------------------------ names
    @property
    def label(self) -> str:
        """The default name: ``Spectrum 3``, ``Line 4``, ``Map 5``."""
        return f"{self.kind.capitalize()} {self.number}"

    @property
    def title(self) -> str:
        """The name shown everywhere: the user's name, else the default label."""
        return self.name or self.label

    @property
    def spectrum(self) -> Spectrum | None:
        return self.data if self.kind == SPECTRUM else None

    @property
    def has_data(self) -> bool:
        return self.data is not None

    @property
    def can_remeasure(self) -> bool:
        return bool(self.conditions.get("request")) and (self.shape != "spot" or self.kind == SPECTRUM)

    def describe(self, field_um: float = 1.0) -> str:
        """Geometry in µm at *field_um* (the fractions times the present field)."""
        if self.source:
            return self.source
        p = [v * field_um for v in self.params]
        if self.shape == "point":
            return f"point ({p[0]:.2f}, {p[1]:.2f}) µm"
        if self.shape in ("rect", "ellipse"):
            return f"{self.shape} {p[2]:.2f}×{p[3]:.2f} µm at ({p[0]:.1f}, {p[1]:.1f})"
        if self.shape == "field":
            return "whole image"
        if self.shape == "line":
            return f"({p[0]:.1f}, {p[1]:.1f}) → ({p[2]:.1f}, {p[3]:.1f}) µm, {np.hypot(p[2] - p[0], p[3] - p[1]):.2f} µm"
        if self.shape == "map":
            return f"{p[2]:.2f} × {p[3]:.2f} µm at ({p[0]:.1f}, {p[1]:.1f})"
        return "SEM spot"

    def status(self) -> str:
        if self.acquiring:
            return "acquiring"
        if self.data is None:
            return "not acquired"
        return "unsaved" if self.dirty else "saved"

    # ------------------------------------------------------------------ metadata
    def record(self) -> dict:
        """What is stored in the data's metadata (and read back by :func:`open_file`)."""
        return _jsonable({"number": self.number, "name": self.name, "kind": self.kind, "shape": self.shape,
                          "params": list(self.params), "conditions": self.conditions, "source": self.source,
                          "remeasure_of": self.remeasure_of})

    def stamp(self) -> None:
        """Write :meth:`record` (and the title) into the data's metadata."""
        d = self.data
        if d is None:
            return
        d.metadata = dict(d.metadata or {})
        d.metadata[META_KEY] = self.record()
        if isinstance(d, Spectrum):
            d.title = self.title

    def properties(self) -> dict:
        """Everything known about how the item was taken (for the Properties dialog)."""
        out: dict = {"Item": {"number": self.number, "name": self.name or "–", "kind": self.kind,
                              "geometry": self.shape, "geometry (field fractions)": list(self.params),
                              "status": self.status()}}
        if self.path:
            out["Item"]["file"] = self.path
        if self.source:
            out["Item"]["source"] = self.source
        if self.remeasure_of is not None:
            out["Item"]["remeasure of"] = self.remeasure_of
        d = self.data
        if isinstance(d, Spectrum):
            out["Spectrum"] = {"timestamp": d.timestamp, "real time (s)": d.real_time, "live time (s)": d.live_time,
                               "dead time (%)": round(100 * d.dead_fraction, 2), "ICR (cps)": round(d.icr, 1),
                               "OCR (cps)": round(d.ocr, 1), "total counts": d.total, "channels": d.n_channels,
                               "beam energy (kV)": d.beam_kv, "take-off (°)": d.takeoff_deg,
                               "energy calibration": d.calibration.to_dict()}
        elif isinstance(d, SpectrumMap):
            ny, nx = d.shape
            done = int(d.done.sum()) if d.done is not None else nx * ny
            out["Map" if self.kind == MAP else "Line scan"] = {
                "timestamp": d.timestamp, "pixels": f"{nx} × {ny}", "pixels acquired": done,
                "dwell (ms)": d.dwell_s * 1e3, "region (µm)": list(d.region_um), "channels": d.n_channels,
                "real time total (s)": float(np.nansum(d.real_time)),
                "live time total (s)": float(np.nansum(d.live_time)),
                "beam energy (kV)": d.beam_kv, "take-off (°)": d.takeoff_deg,
                "energy calibration": d.calibration.to_dict()}
        cond = {k: v for k, v in self.conditions.items()}
        if cond:
            out["Conditions"] = cond
        meta = {k: v for k, v in ((d.metadata or {}) if d is not None else {}).items() if k != META_KEY}
        if meta:
            out["Metadata"] = meta
        return _jsonable(out)


def flatten(d, prefix: str = "") -> list:
    """Nested dict → [(dotted key, text)] rows (for a property table)."""
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            rows += flatten(v, key + " › ")
        else:
            if isinstance(v, float):
                txt = f"{v:.6g}"
            elif isinstance(v, (list, tuple)):
                txt = ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v) if len(v) <= 16 \
                    else f"[{len(v)} values]"
            elif v is None or v == {}:
                txt = "–"
            else:
                txt = str(v)
            rows.append((key, txt))
    return rows


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", text).strip() or "item"


def default_filename(item: DataItem) -> str:
    ext = ".msa" if item.kind == SPECTRUM else ".npz"
    stem = f"{item.number:03d}_{_safe(item.title)}"
    return stem + ext


def save_item(item: DataItem, path: str) -> str:
    """Write *item*'s data (EMSA / npz by extension; maps and lines always npz)."""
    if item.data is None:
        raise ValueError(f"{item.title} has no data")
    path = str(path)
    if item.kind != SPECTRUM and not path.lower().endswith(".npz"):
        path += ".npz"
    item.stamp()
    item.data.save(path)
    return path


def open_file(path: str) -> DataItem:
    """A spectrum (``.msa``/``.emsa``/``.txt``/spectrum ``.npz``), map or line scan (``.npz``) as an
    item (number 0: :meth:`DataManager.add` numbers it)."""
    path = str(path)
    data = None
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as f:
            keys = set(f.files)
        data = load_map(path) if "cube" in keys else load_spectrum(path)
    else:
        data = load_spectrum(path)
    rec = dict((data.metadata or {}).get(META_KEY) or {})
    if isinstance(data, SpectrumMap):
        meta = data.metadata or {}
        kind = LINE if meta.get("type") == "linescan" or data.shape[0] == 1 and "p0_um" in meta else MAP
    else:
        kind = SPECTRUM
    shape, params = rec.get("shape"), rec.get("params") or ()
    if rec.get("kind") != kind or shape not in SHAPES[kind]:
        shape, params = _geometry_from(data, kind)
    item = DataItem(0, kind, shape, params, name=str(rec.get("name") or ""),
                    conditions=dict(rec.get("conditions") or {}), data=data, path=path,
                    source="" if shape != "spot" else os.path.basename(path))
    if not item.name and kind == SPECTRUM and data.title and not rec:
        item.name = data.title  # a foreign spectrum: its own title
    return item


def _geometry_from(data, kind) -> tuple:
    """Geometry of a file without the data manager's record (older easyEBL files)."""
    meta = data.metadata or {}
    f = meta.get("field_um")
    if kind == SPECTRUM:
        return "spot", ()
    if not f:
        return ("line", ()) if kind == LINE else ("map", ())
    if kind == LINE and meta.get("p0_um") and meta.get("p1_um"):
        return "line", tuple(float(v) / f for v in list(meta["p0_um"]) + list(meta["p1_um"]))
    return "map", tuple(float(v) / f for v in data.region_um)


class DataManager:
    """The items of a session, in acquisition order.  ``listeners`` are called with
    ``(event, item)`` – ``added``, ``changed``, ``removed`` – after every change."""

    def __init__(self):
        self._items: list[DataItem] = []
        self._next = 1
        self.listeners: list = []

    # ---------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(list(self._items))

    def __contains__(self, item) -> bool:
        return item in self._items

    @property
    def items(self) -> list:
        return list(self._items)

    def of_kind(self, kind: str) -> list:
        return [i for i in self._items if i.kind == kind]

    def get(self, number: int) -> DataItem | None:
        return next((i for i in self._items if i.number == number), None)

    def index(self, item: DataItem) -> int:
        return self._items.index(item)

    def dirty_items(self) -> list:
        return [i for i in self._items if i.dirty and i.data is not None]

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_items())

    def _emit(self, event: str, item: DataItem) -> None:
        for fn in list(self.listeners):
            fn(event, item)

    # ---------------------------------------------------------------- changes
    def new(self, kind: str, shape: str, params=(), conditions=None, **kw) -> DataItem:
        """A new item (no data yet) with the next number."""
        return self.add(DataItem(0, kind, shape, params, conditions=dict(conditions or {}), **kw))

    def add(self, item: DataItem) -> DataItem:
        item.number = self._next
        self._next += 1
        if item.data is not None and item.path is None:
            item.dirty = True
        self._items.append(item)
        item.stamp()
        self._emit("added", item)
        return item

    def set_data(self, item: DataItem, data, final: bool = True) -> None:
        """New (or partial, ``final=False``) data of *item*; final data is unsaved."""
        item.data = data
        item.acquiring = not final
        if final and data is not None:
            item.dirty = True
            item.stamp()
        self._emit("changed", item)

    def finish(self, item: DataItem) -> None:
        """The acquisition of *item* ended (with or without data)."""
        item.acquiring = False
        if item.data is not None:
            item.dirty = True
            item.stamp()
        self._emit("changed", item)

    def rename(self, item: DataItem, name: str) -> None:
        name = (name or "").strip()
        if name == item.name:
            return
        item.name = name
        if item.data is not None:
            item.dirty = True
            item.stamp()
        self._emit("changed", item)

    def remove(self, item: DataItem) -> None:
        if item in self._items:
            self._items.remove(item)
            self._emit("removed", item)

    def clear(self) -> None:
        for it in list(self._items):
            self.remove(it)

    def touch(self, item: DataItem) -> None:
        """*item* was changed in place (e.g. its beam energy): listeners refresh it."""
        self._emit("changed", item)

    def remeasure(self, item: DataItem) -> DataItem:
        """A new item with *item*'s geometry and conditions (no data yet): fill it with the new
        acquisition.  The original is kept unchanged."""
        if not item.can_remeasure:
            raise ValueError(f"{item.title} has no recorded acquisition to repeat")
        new = DataItem(0, item.kind, item.shape, item.params, conditions=json.loads(json.dumps(item.conditions)),
                       remeasure_of=item.number)
        if item.name:
            new.name = f"{item.name} (re)"
        return self.add(new)

    # ---------------------------------------------------------------- files
    def save(self, item: DataItem, path: str) -> str:
        path = save_item(item, path)
        item.path, item.dirty = path, False
        self._emit("changed", item)
        return path

    def save_all(self, folder: str, items=None) -> list:
        """Save *items* (default: the unsaved ones) – each to its own file if it has one, else
        to ``folder/<number>_<name>.<ext>``.  Returns the paths."""
        out = []
        for it in list(items if items is not None else self.dirty_items()):
            if it.data is None:
                continue
            path = it.path or os.path.join(folder, default_filename(it))
            out.append(self.save(it, path))
        return out

    def open(self, path: str) -> DataItem:
        return self.add(open_file(path))
