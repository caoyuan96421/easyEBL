"""The stream program: an output-sample generator that follows the engine's commands.

The engine runs **one** hardware-timed stream for as long as scanning goes
on.  The HAL pulls output chunks from :meth:`StreamProgram.chunks` only as
fast as the output buffer drains, so the program always runs exactly one
buffer depth (``latency_s``) ahead of the beam.  At every line boundary it
re-reads the engine's :class:`Directive`:

* new scan settings → the current frame is cut after this line and a new
  frame (new :class:`FrameSource`) starts – the hardware never stops, so a
  change reaches the beam after ≈ ``latency_s`` instead of a task restart;
* freeze → the frame is finished, then the program ends;
* stop → the program ends at once (after this line);
* new stream parameters (sample rate, buffer, AI range) → the program ends
  and the engine restarts the stream.

When the program ends, the HAL appends its clocked safe tail (blank, park).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ...core.calibration import FieldCalibration
from ...core.waveform import OutputChunk
from ..raster import RasterConfig, RasterPlan
from .assembler import FrameRecord, StreamAssembler
from .source import FrameSource

__all__ = ["Directive", "CaptureRequest", "StreamProgram"]


@dataclass
class CaptureRequest:
    cfg: RasterConfig
    frames: int
    id: int
    future: object  # concurrent.futures.Future[Frame]
    t_request: float = 0.0


@dataclass(frozen=True)
class Directive:
    """What the engine wants right now (an immutable snapshot)."""

    gen: int
    cfg: RasterConfig
    mode: str  # "live" | "single" | "capture" | "idle"
    mode_gen: int
    stop: bool = False
    freeze: bool = False
    stream_gen: int = 0  # changes when the stream itself must restart
    capture: CaptureRequest | None = None
    restart_on_change: bool = True
    chunk_s: float = 0.004
    t_change: float | None = None
    calibration: FieldCalibration | None = None


class StreamProgram:
    """Generates the sample stream for one hardware stream (see module doc)."""

    def __init__(
        self,
        engine,  # provides directive() and source_for(cfg, gen)
        assembler: StreamAssembler,
        calibration: FieldCalibration,
        rate: float,
        park_v: tuple,
        stream_gen: int,
    ):
        self.engine = engine
        self.asm = assembler
        self.cal = calibration
        self.rate = rate
        self.park_v = np.asarray(park_v, float)
        self.stream_gen = stream_gen
        self.pos = 0  # absolute index of the next output sample
        self.frames_started = 0
        self.frames_full = 0
        self.end_reason: str | None = None
        self.last_mode_gen: int | None = None
        self._cache: dict = {}
        self._cap_count: dict = {}

    # ------------------------------------------------------------- helpers
    def _source(self, cfg: RasterConfig, cal: FieldCalibration | None, key) -> FrameSource:
        src = self._cache.get(key)
        if src is None:
            if len(self._cache) > 4:  # bounded: only recent layouts are kept
                self._cache.clear()
            src = FrameSource(RasterPlan(cfg, cal or self.cal))
            self._cache[key] = src
        return src

    def _emit(self, c: OutputChunk) -> OutputChunk:
        self.pos += len(c)
        return c

    def _end_now(self, d: Directive) -> str | None:
        self.last_mode_gen = d.mode_gen
        if d.stream_gen != self.stream_gen:
            return "restart"
        if d.stop or d.mode == "idle":
            return "stopped"
        return None

    # ------------------------------------------------------------- program
    def chunks(self) -> Iterator[OutputChunk]:
        prev_end = self.park_v
        src: FrameSource | None = None
        try:
            while True:
                d = self.engine.directive()
                reason = self._end_now(d)
                if reason:
                    self.end_reason = reason
                    break
                # ---- what to scan next
                if d.mode == "capture" and d.capture is not None:
                    req = d.capture
                    kind, cfg, key = "capture", req.cfg, ("cap", req.id)
                    gen = -req.id
                else:
                    kind = "single" if d.mode == "single" else "live"
                    cfg, key, gen = d.cfg, ("set", d.gen), d.gen
                    req = None
                src = self._source(cfg, d.calibration, key)
                rec = FrameRecord(
                    index=self.frames_started,
                    start=self.pos,
                    source=src,
                    kind=kind,
                    gen=gen,
                    t_request=d.t_change,
                    capture_id=None if req is None else req.id,
                )
                self.asm.add_record(rec)
                self.frames_started += 1
                yield self._emit(src.settle(prev_end))
                prev_end = src.line_start_v[0]
                L = src.line_len
                k = max(1, int(d.chunk_s * self.rate) // L)
                j = 0
                cut = None
                while j < src.ny:
                    j1 = min(src.ny, j + k)
                    yield self._emit(src.lines(j, j1, prev_end))
                    prev_end = src.line_end_v[j1 - 1]
                    j = j1
                    if j >= src.ny:
                        break
                    d2 = self.engine.directive()
                    r = self._end_now(d2)
                    if r:
                        cut = r
                    elif d2.mode_gen != d.mode_gen:
                        cut = "mode"
                    elif kind != "capture" and d2.gen != d.gen and d2.restart_on_change:
                        cut = "settings"
                    if cut:
                        rec.n_lines = j
                        break
                if cut in ("restart", "stopped"):
                    self.end_reason = cut
                    break
                if cut:
                    continue  # start a new frame with the new settings / mode
                self.frames_full += 1
                # ---- after a complete frame
                d3 = self.engine.directive()
                r = self._end_now(d3)
                if r:
                    self.end_reason = r
                    break
                if d3.mode_gen != d.mode_gen:
                    continue
                if kind == "single":
                    self.end_reason = "single_done"
                    break
                if kind == "capture":
                    if self._capture_frames(req.id) >= req.frames:
                        self.end_reason = "capture_done"
                        break
                    continue
                if d3.freeze:
                    self.end_reason = "frozen"
                    break
            # blanked hold so the lag-compensated AI of the last line arrives
            if src is not None:
                yield self._emit(src.tail(prev_end))
        finally:
            if self.end_reason is None:
                self.end_reason = "aborted"

    def _capture_frames(self, capture_id: int) -> int:
        """Full frames *emitted* for this capture (the frame just finished included)."""
        self._cap_count[capture_id] = self._cap_count.get(capture_id, 0) + 1
        return self._cap_count[capture_id]

    @property
    def hardware_time_s(self) -> float:
        return self.pos / self.rate
