"""Beam control for EDS: one long HAL stream whose beam position is commanded live.

A DP5 has no list mode, so spectra cannot be time-tagged against a fast
raster.  EDS acquisitions therefore *step* the beam: hold it on a point (or a
small repeating pattern, for area spectra), let the pulse processor count for
a preset real time, then move on.  :class:`BeamProgram` is the chunk source of
that stream.  The acquisition thread calls :meth:`move` / :meth:`pattern` /
:meth:`blank`; the stream thread picks the command up at its next chunk and
records the stream sample index at which it took effect (a :class:`Ticket`).
The beam is really there once the device has *generated* that sample
(``StreamMonitor.generated``), one output buffer (≈ ``latency_s``) later.

:class:`AIWindows` turns the video AI stream into per-pixel means over
sample windows [start, stop), so the SE signal is co-acquired with every
spectrum on the same clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from ..core.waveform import OutputChunk

__all__ = ["Ticket", "BeamProgram", "AIWindows", "raster_pattern", "grid_points"]


class Ticket:
    """When a beam command took effect in the stream (sample index)."""

    def __init__(self):
        self._ev = threading.Event()
        self.index: int | None = None

    def _set(self, index: int) -> None:
        self.index = int(index)
        self._ev.set()

    def wait(self, timeout: float | None = None) -> int:
        if not self._ev.wait(timeout):
            raise TimeoutError("the beam command was not picked up by the stream")
        return self.index

    @property
    def done(self) -> bool:
        return self._ev.is_set()


@dataclass
class _Cmd:
    kind: str  # hold | pattern
    x: np.ndarray
    y: np.ndarray
    blank: bool
    ticket: Ticket


class BeamProgram:
    """Endless chunk generator for a HAL stream; thread-safe commands.

    ``chunk_n`` samples per chunk bounds how quickly a command is picked up
    (the HAL keeps ``latency_s`` of output queued on top of that).
    """

    def __init__(self, rate: float, start=(0.0, 0.0), chunk_n: int = 1024, blank: bool = True):
        self.rate = float(rate)
        self.chunk_n = int(chunk_n)
        self._lock = threading.Lock()
        self._pending: _Cmd | None = None
        self._stop = threading.Event()
        t = Ticket()
        t._set(0)
        self._cur = _Cmd("hold", np.array([float(start[0])]), np.array([float(start[1])]), bool(blank), t)
        self._phase = 0
        self.produced = 0
        self.started = threading.Event()  # the first chunk has been generated

    # ------------------------------------------------------------- commands
    def _submit(self, cmd: _Cmd) -> Ticket:
        if self._stop.is_set():
            raise RuntimeError("beam program stopped")
        with self._lock:
            self._pending = cmd  # the newest command wins
        return cmd.ticket

    def move(self, x_v: float, y_v: float, blank: bool = False) -> Ticket:
        """Hold the beam at (x_v, y_v) volts, unblanked unless *blank*."""
        return self._submit(_Cmd("hold", np.array([float(x_v)]), np.array([float(y_v)]), bool(blank), Ticket()))

    def blank(self) -> Ticket:
        """Blank the beam where it is."""
        with self._lock:
            cur = self._pending or self._cur
            x, y = float(cur.x[0]), float(cur.y[0])
        return self._submit(_Cmd("hold", np.array([x]), np.array([y]), True, Ticket()))

    def pattern(self, x_v, y_v, blank: bool = False) -> Ticket:
        """Repeat a sample pattern (one position per sample), e.g. a fast raster over an area."""
        x = np.ascontiguousarray(x_v, dtype=float)
        y = np.ascontiguousarray(y_v, dtype=float)
        if x.shape != y.shape or x.ndim != 1 or x.size == 0:
            raise ValueError("pattern x and y must be equal-length 1-D arrays")
        return self._submit(_Cmd("pattern", x, y, bool(blank), Ticket()))

    def stop(self) -> None:
        """End the stream after the current chunk (the HAL then blanks and parks)."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def position(self) -> tuple:
        with self._lock:
            c = self._cur
        return float(c.x[0]), float(c.y[0]), bool(c.blank)

    # ------------------------------------------------------------- generator
    def chunks(self):
        n = self.chunk_n
        while not self._stop.is_set():
            with self._lock:
                cmd, self._pending = self._pending, None
            if cmd is not None:
                self._cur, self._phase = cmd, 0
                cmd.ticket._set(self.produced)
            c = self._cur
            if c.kind == "hold":
                chunk = OutputChunk.constant(n, float(c.x[0]), float(c.y[0]), blank=c.blank)
            else:
                idx = (self._phase + np.arange(n)) % c.x.size
                self._phase = int((self._phase + n) % c.x.size)
                chunk = OutputChunk(c.x[idx], c.y[idx], np.full(n, c.blank))
            self.produced += n
            self.started.set()
            yield chunk


def raster_pattern(x0_v: float, x1_v: float, y0_v: float, y1_v: float, nx: int, ny: int,
                   mask=None, repeat: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Serpentine grid over a rectangle (volts), *repeat* samples per position; *mask* (ny, nx) keeps a subset."""
    xs = np.linspace(x0_v, x1_v, nx) if nx > 1 else np.array([(x0_v + x1_v) / 2])
    ys = np.linspace(y0_v, y1_v, ny) if ny > 1 else np.array([(y0_v + y1_v) / 2])
    px, py = [], []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        keep = np.ones(nx, bool) if mask is None else (mask[j] if j % 2 == 0 else mask[j][::-1])
        px.append(row[keep])
        py.append(np.full(int(keep.sum()), y))
    x = np.concatenate(px)
    y = np.concatenate(py)
    if x.size == 0:
        raise ValueError("empty pattern")
    return np.repeat(x, repeat), np.repeat(y, repeat)


def grid_points(center, width, height, nx: int, ny: int) -> np.ndarray:
    """(ny·nx, 2) pixel centres of a grid, row 0 at the top (+y), in the units of the inputs."""
    pw, ph = width / nx, height / ny
    u = center[0] - width / 2 + (np.arange(nx) + 0.5) * pw
    v = center[1] + height / 2 - (np.arange(ny) + 0.5) * ph
    uu, vv = np.meshgrid(u, v)
    return np.column_stack([uu.ravel(), vv.ravel()])


class AIWindows:
    """Means of AI samples over sample windows [start, stop), fed with the stream's AI blocks.

    Keeps the last *history* samples, so a window may start in the (recent)
    past – the acquisition thread usually learns where a pixel began only
    after the samples have arrived.  A window completes once data past its
    end has arrived.  Thread-safe: ``feed`` runs in the stream thread,
    ``request``/``result`` in the acquisition thread.
    """

    def __init__(self, n_channels: int, history: int = 1 << 21):
        self.n = int(n_channels)
        self.received = 0
        self._hist = np.zeros((self.n, int(history)), np.float32)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._open: list = []  # [key, start, stop, sum (n,), count]
        self._done: dict = {}

    def _from_history(self, s: int, e: int) -> np.ndarray:
        H = self._hist.shape[1]
        a, b = s % H, e % H
        if e - s <= 0:
            return np.zeros(self.n)
        if a < b:
            return self._hist[:, a:b].sum(axis=1, dtype=np.float64)
        return self._hist[:, a:].sum(axis=1, dtype=np.float64) + self._hist[:, :b].sum(axis=1, dtype=np.float64)

    def request(self, key, start: int, stop: int) -> None:
        if stop <= start:
            raise ValueError("empty AI window")
        start, stop = int(start), int(stop)
        with self._lock:
            H = self._hist.shape[1]
            acc = np.zeros(self.n)
            cnt = 0
            if start < self.received:
                lo = max(start, self.received - H)
                hi = min(stop, self.received)
                acc += self._from_history(lo, hi)
                cnt = (hi - lo) if lo == start else -1  # -1: part of the window is lost
            if stop <= self.received:
                self._done[key] = acc / cnt if cnt > 0 else np.full(self.n, np.nan)
                self._cv.notify_all()
            else:
                self._open.append([key, max(start, self.received), stop, acc, cnt])

    def feed(self, block: np.ndarray) -> None:
        block = np.atleast_2d(block)
        m = block.shape[1]
        b0 = self.received
        b1 = b0 + m
        with self._lock:
            H = self._hist.shape[1]
            if m >= H:
                self._hist[:] = block[:, m - H:]
                # rotate so that index (b1 - H) % H holds the oldest sample
                self._hist = np.roll(self._hist, (b1 - H) % H, axis=1)
            else:
                a = b0 % H
                k = min(m, H - a)
                self._hist[:, a:a + k] = block[:, :k]
                if k < m:
                    self._hist[:, :m - k] = block[:, k:]
            keep = []
            for w in self._open:
                key, s, e, acc, cnt = w
                lo, hi = max(s, b0), min(e, b1)
                if hi > lo and cnt >= 0:
                    acc += block[:, lo - b0:hi - b0].sum(axis=1)
                    w[4] = cnt + (hi - lo)
                if e <= b1:
                    self._done[key] = acc / w[4] if w[4] > 0 else np.full(self.n, np.nan)
                else:
                    keep.append(w)
            self._open = keep
            self.received = b1
            if self._done:
                self._cv.notify_all()

    def result(self, key, timeout: float | None = None, pop: bool = True):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while key not in self._done:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise TimeoutError(f"AI window {key!r} not complete")
                self._cv.wait(left)
            return self._done.pop(key) if pop else self._done[key]

    def ready(self) -> dict:
        """Pop every completed window: {key: means}."""
        with self._lock:
            d, self._done = self._done, {}
        return d
