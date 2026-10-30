"""Hardware-independent description of sample-clocked output streams.

Every hardware-timed operation (imaging, lithography, test patterns) is
expressed as an iterable of :class:`OutputChunk`.  A chunk holds, for each
tick of the shared sample clock:

* ``x`` / ``y``  – deflection voltages in volts,
* ``blank``      – logical blanker state (``True`` = beam blanked),
* ``markers``    – optional extra hardware-timed digital lines by logical
  name (e.g. ``"line_sync"`` for a detector/MCA), ``True`` = asserted.

Polarity, channel names and physical lines are resolved by the HAL, so
builders never deal with hardware details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

__all__ = ["OutputChunk", "concat_chunks", "rechunk", "chunk_arrays", "total_samples"]


@dataclass
class OutputChunk:
    x: np.ndarray
    y: np.ndarray
    blank: np.ndarray
    markers: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.y = np.ascontiguousarray(self.y, dtype=np.float64)
        self.blank = np.ascontiguousarray(self.blank, dtype=bool)
        n = self.x.shape[0]
        if self.x.ndim != 1 or self.y.shape != (n,) or self.blank.shape != (n,):
            raise ValueError("x, y and blank must be 1-D arrays of equal length")
        for k, v in list(self.markers.items()):
            v = np.ascontiguousarray(v, dtype=bool)
            if v.shape != (n,):
                raise ValueError(f"marker '{k}' length mismatch")
            self.markers[k] = v

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def constant(cls, n: int, x: float, y: float, blank: bool = True, markers: dict | None = None):
        mk = {k: np.full(n, bool(v)) for k, v in (markers or {}).items()}
        return cls(np.full(n, float(x)), np.full(n, float(y)), np.full(n, bool(blank)), mk)

    @classmethod
    def empty(cls) -> "OutputChunk":
        return cls(np.empty(0), np.empty(0), np.empty(0, dtype=bool))

    def max_abs_voltage(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(max(np.max(np.abs(self.x)), np.max(np.abs(self.y))))

    def slice(self, start: int, stop: int) -> "OutputChunk":
        return OutputChunk(
            self.x[start:stop],
            self.y[start:stop],
            self.blank[start:stop],
            {k: v[start:stop] for k, v in self.markers.items()},
        )


def concat_chunks(chunks: Iterable[OutputChunk]) -> OutputChunk:
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return OutputChunk.empty()
    names = set()
    for c in chunks:
        names.update(c.markers)
    markers = {
        k: np.concatenate([c.markers.get(k, np.zeros(len(c), bool)) for c in chunks]) for k in names
    }
    return OutputChunk(
        np.concatenate([c.x for c in chunks]),
        np.concatenate([c.y for c in chunks]),
        np.concatenate([c.blank for c in chunks]),
        markers,
    )


def rechunk(chunks: Iterable[OutputChunk], size: int) -> Iterator[OutputChunk]:
    """Re-slice a chunk stream into pieces of exactly *size* samples (last may be shorter)."""
    if size <= 0:
        raise ValueError("size must be positive")
    pending: list[OutputChunk] = []
    n_pending = 0
    for c in chunks:
        if not len(c):
            continue
        pending.append(c)
        n_pending += len(c)
        if n_pending >= size:
            merged = concat_chunks(pending)
            start = 0
            while len(merged) - start >= size:
                yield merged.slice(start, start + size)
                start += size
            rest = merged.slice(start, len(merged))
            pending = [rest] if len(rest) else []
            n_pending = len(rest)
    if n_pending:
        yield concat_chunks(pending)


def chunk_arrays(x, y, blank, size: int, markers: dict | None = None) -> Iterator[OutputChunk]:
    """Split full arrays into an iterator of chunks."""
    n = len(x)
    markers = markers or {}
    for s in range(0, n, size):
        e = min(n, s + size)
        yield OutputChunk(x[s:e], y[s:e], blank[s:e], {k: v[s:e] for k, v in markers.items()})


def total_samples(chunks: Iterable[OutputChunk]) -> int:
    return sum(len(c) for c in chunks)
