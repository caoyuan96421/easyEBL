"""Turning the continuous AI sample stream back into image rows.

The stream program (:mod:`.program`) registers a :class:`FrameRecord` for
every frame it emits – where the frame starts in the sample stream, which
waveform layout it uses and (if it was cut short by a settings change) how
many lines it really has.  Because AI sample *k* is taken on the same clock
edge as output sample *k*, these records are all that is needed to map
incoming samples to pixels, even though the settings can change from one
frame to the next inside one uninterrupted hardware stream.

Memory is bounded: AI samples live in a :class:`SampleRing` that only has to
span a few lines, however large the frame or however large the AI blocks
delivered after a host stall.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
import math
import time
from typing import Callable

import numpy as np

from .source import FrameSource

__all__ = ["FrameRecord", "SampleRing", "StreamAssembler", "ImageAccumulator"]


@dataclass
class FrameRecord:
    """One frame as scheduled in the output stream."""

    index: int  # frame number within the stream
    start: int  # absolute sample index of the frame-settle start
    source: FrameSource
    kind: str = "live"  # "live" | "single" | "capture"
    gen: int = 0  # settings generation the frame was built from
    n_lines: int = 0  # lines actually emitted (== ny unless cut)
    t_request: float | None = None  # perf_counter of the settings change (latency stats)
    capture_id: int | None = None
    lines_done: int = 0
    t_first_line: float | None = None

    def __post_init__(self):
        if not self.n_lines:
            self.n_lines = self.source.ny

    @property
    def cut(self) -> bool:
        return self.n_lines < self.source.ny

    def pixel_start(self, line: int) -> int:
        s = self.source
        return self.start + s.n_fs + line * s.line_len + s.n_fb + s.n_pre

    @property
    def end(self) -> int:
        """Absolute index one past the last sample of the (possibly cut) frame."""
        s = self.source
        return self.start + s.n_fs + self.n_lines * s.line_len


class SampleRing:
    """Circular buffer of the most recent AI samples, addressed by absolute index."""

    def __init__(self, n_ch: int, capacity: int):
        self.n_ch = n_ch
        self.cap = int(capacity)
        self.buf = np.zeros((n_ch, self.cap))
        self.end = 0  # absolute index one past the newest sample

    @property
    def start(self) -> int:
        return max(0, self.end - self.cap)

    def ensure(self, capacity: int) -> None:
        """Grow (never shrink) while keeping the newest samples."""
        if capacity <= self.cap:
            return
        keep = min(self.cap, self.end)
        old = self.get(self.end - keep, self.end) if keep else None
        self.cap = int(capacity)
        self.buf = np.zeros((self.n_ch, self.cap))
        if old is not None:
            e = self.end
            self.end = e - keep
            self.write(old)
            assert self.end == e

    def write(self, block: np.ndarray) -> None:
        n = block.shape[1]
        if n > self.cap:  # only the newest samples can be kept
            block = block[:, n - self.cap :]
            self.end += n - self.cap
            n = self.cap
        i = self.end % self.cap
        first = min(n, self.cap - i)
        self.buf[:, i : i + first] = block[:, :first]
        if first < n:
            self.buf[:, : n - first] = block[:, first:]
        self.end += n

    def get(self, lo: int, hi: int) -> np.ndarray:
        """Copy of samples ``lo … hi-1`` (must be within the ring)."""
        if lo < self.start or hi > self.end or hi < lo:
            raise IndexError(f"samples [{lo}, {hi}) not in ring [{self.start}, {self.end})")
        n = hi - lo
        i = lo % self.cap
        if i + n <= self.cap:
            return self.buf[:, i : i + n].copy()
        first = self.cap - i
        return np.concatenate([self.buf[:, i:], self.buf[:, : n - first]], axis=1)


LineCallback = Callable[[FrameRecord, int, int, np.ndarray], None]
FrameCallback = Callable[[FrameRecord], None]


class StreamAssembler:
    """Cuts the AI stream into rows according to the scheduled frame records.

    ``on_line(record, line, row, values)`` gets one row (values: n_ch × nx,
    image column order); ``on_frame(record)`` fires when a record's last
    emitted line is done.  Both run in the stream thread.
    """

    MIN_RING = 1 << 18

    def __init__(self, n_ch: int, on_line: LineCallback, on_frame: FrameCallback):
        self.n_ch = n_ch
        self.on_line = on_line
        self.on_frame = on_frame
        self.records: collections.deque[FrameRecord] = collections.deque()
        self.ring = SampleRing(n_ch, self.MIN_RING)
        self.t_first_sample: float | None = None
        self.lines_total = 0

    @property
    def samples_in(self) -> int:
        return self.ring.end

    def add_record(self, rec: FrameRecord) -> None:
        s = rec.source
        d = s.cfg.lag_comp_samples
        need = 4 * (s.line_len + int(math.ceil(d)) + 4)
        self.ring.ensure(max(self.MIN_RING, need))
        self.records.append(rec)

    def _sub(self) -> int:
        return max(1024, self.ring.cap // 4)

    def feed(self, block: np.ndarray) -> None:
        if self.t_first_sample is None and block.shape[1]:
            self.t_first_sample = time.perf_counter()
        sub = self._sub()
        n = block.shape[1]
        for s in range(0, n, sub):
            self.ring.write(block[:, s : s + sub])
            self._process()
        if n == 0:
            self._process()

    def _process(self) -> None:
        ring = self.ring
        while self.records:
            rec = self.records[0]
            if rec.lines_done >= rec.n_lines:
                self.records.popleft()
                self.on_frame(rec)
                continue
            src = rec.source
            cfg = src.cfg
            d = cfg.lag_comp_samples
            d_int = int(math.floor(d))
            frac = d - d_int
            j = rec.lines_done
            lo = rec.pixel_start(j) + d_int
            hi = lo + src.n_pix + 1
            if ring.end < hi:
                return
            if lo < ring.start:  # cannot happen with a correctly sized ring – skip, don't crash
                rec.lines_done += 1
                continue
            seg = ring.get(lo, hi)
            if frac > 1e-9:
                vals = (1.0 - frac) * seg[:, :-1] + frac * seg[:, 1:]
            else:
                vals = seg[:, :-1]
            spp = src.plan.spp
            vals = vals.reshape(self.n_ch, cfg.nx, spp)
            if cfg.settle_samples:
                vals = vals[:, :, cfg.settle_samples :]
            row_vals = vals.mean(axis=2)
            if src.plan.line_dirs[j] < 0:
                row_vals = row_vals[:, ::-1]
            row = int(src.plan.line_rows[j])
            if rec.t_first_line is None:
                rec.t_first_line = time.perf_counter()
            rec.lines_done += 1
            self.lines_total += 1
            self.on_line(rec, j, row, row_vals)

    def pending_frames(self) -> int:
        return len(self.records)


class ImageAccumulator:
    """Display composite: running average over ≈ N frames, updated row by row.

    Pixels of a row are only replaced when that row is scanned again, so after
    a settings change the new image builds up over the old one (no blank
    screen), exactly like the analogue SEM display.  Geometry changes (pixel
    count, field, shift, rotation) restart the average.
    """

    def __init__(self, channels: tuple):
        self.channels = tuple(channels)
        self.images: dict[str, np.ndarray] = {}
        self.row_n = np.zeros(0, int)
        self._key = None
        self._geo = None

    def ensure(self, cfg) -> bool:
        """Adapt to *cfg*; returns True if the average was restarted."""
        key = (cfg.nx, cfg.ny, tuple(cfg.ai_channels))
        geo = (cfg.width_um, cfg.height, tuple(cfg.center_um), cfg.rotation_rad)
        if key != self._key:
            self._key, self._geo = key, geo
            self.channels = tuple(cfg.ai_channels)
            self.images = {c: np.full((cfg.ny, cfg.nx), np.nan, dtype=np.float32) for c in self.channels}
            self.row_n = np.zeros(cfg.ny, int)
            return True
        if geo != self._geo:
            self._geo = geo
            self.row_n[:] = 0
            return True
        return False

    def reset(self) -> None:
        self.row_n[:] = 0

    def add_row(self, row: int, vals: np.ndarray, n_avg: int) -> None:
        n = self.row_n[row]
        if n == 0 or n_avg <= 1:
            for ci, c in enumerate(self.channels):
                self.images[c][row] = vals[ci]
        else:
            w = 1.0 / min(n + 1, n_avg)
            for ci, c in enumerate(self.channels):
                img = self.images[c][row]
                img += (vals[ci] - img) * w
        self.row_n[row] = n + 1

    def snapshot(self) -> dict:
        return {c: a.copy() for c, a in self.images.items()}

    def frames_averaged(self) -> int:
        return int(self.row_n.min()) if self.row_n.size else 0


@dataclass
class CaptureAccumulator:
    """Exact average of a capture's frames (photo integration)."""

    channels: tuple
    ny: int
    nx: int
    frames: int
    sums: dict = field(default_factory=dict)
    counts: np.ndarray | None = None
    frames_done: int = 0

    def __post_init__(self):
        self.sums = {c: np.zeros((self.ny, self.nx)) for c in self.channels}
        self.counts = np.zeros(self.ny, int)

    def add_row(self, row: int, vals: np.ndarray) -> None:
        for ci, c in enumerate(self.channels):
            self.sums[c][row] += vals[ci]
        self.counts[row] += 1

    def images(self) -> dict:
        c = np.maximum(self.counts, 1)[:, None]
        out = {}
        for ch, s in self.sums.items():
            img = (s / c).astype(np.float32)
            img[self.counts == 0] = np.nan
            out[ch] = img
        return out
