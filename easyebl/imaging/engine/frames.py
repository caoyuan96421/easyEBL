"""Published images and engine events – the engine's thread-safe outputs.

Consumers (GUI, command line, scripts) never receive data through callbacks
running in the scan thread.  Instead:

* :class:`FrameSlot` keeps only the *latest* frame.  Readers poll
  :meth:`FrameSlot.latest` or block in :meth:`FrameSlot.wait`.  A slow
  reader therefore skips frames instead of queuing them – memory use is
  bounded no matter how slow the consumer is.
* :class:`EventBus` hands each subscriber a bounded queue of
  :class:`EngineEvent` (state changes, errors, messages).  Overflowing
  subscribers lose their oldest events, never block the engine.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
import enum
import threading
import time

import numpy as np

from ...core.calibration import FieldCalibration
from ..raster import RasterConfig

__all__ = ["EngineState", "Frame", "FrameSlot", "EngineEvent", "EventBus", "Subscription"]


class EngineState(str, enum.Enum):
    IDLE = "idle"  # no stream, beam blanked and parked
    STARTING = "starting"  # setting up the hardware tasks
    LIVE = "live"  # continuous scanning
    FREEZING = "freezing"  # finishing the current frame, then stop
    SINGLE = "single"  # one frame, then freeze
    CAPTURE = "capture"  # photo scan (integrated frames), then freeze
    FROZEN = "frozen"  # scanning stopped, last image kept
    DISCONNECTED = "disconnected"  # DAQ connection lost: waiting for it, then scanning resumes
    ERROR = "error"  # last run failed (see status().last_error)
    CLOSED = "closed"


@dataclass(frozen=True)
class Frame:
    """One published image (a snapshot – arrays are never modified afterwards).

    ``images`` hold the display composite per channel (volts): the running
    average, with rows of the frame in progress already replaced by new data
    (like a commercial SEM, the scan line moves down over the old image).
    """

    images: dict
    seq: int  # publication counter (increments for every publish)
    frame_index: int  # number of frames completed in this run before this one
    complete: bool  # True when a frame has just been completed
    rows_done: int  # rows of the current frame already scanned
    scan_row: int | None  # image row most recently scanned (scan-line indicator)
    config: RasterConfig
    calibration: FieldCalibration
    t_sample: float  # hardware time of the last sample used (s since stream start)
    timestamp: float = field(default_factory=time.time)
    frames_averaged: int = 1
    kind: str = "live"  # "live" | "single" | "capture"
    run_id: int = 0

    @property
    def image(self) -> np.ndarray:
        return next(iter(self.images.values()))

    @property
    def pixel_size_um(self) -> tuple:
        return self.config.pixel_size_um

    @property
    def field_width_um(self) -> float:
        return self.config.width_um

    def metadata(self) -> dict:
        c = self.config
        return {
            "nx": c.nx,
            "ny": c.ny,
            "width_um": c.width_um,
            "height_um": c.height,
            "pixel_size_um": list(c.pixel_size_um),
            "center_um": list(c.center_um),
            "rotation_rad": c.rotation_rad,
            "dwell_s": c.actual_dwell_s,
            "sample_rate": c.rate,
            "samples_per_pixel": c.samples_per_pixel,
            "bidirectional": c.bidirectional,
            "frames_averaged": self.frames_averaged,
            "kind": self.kind,
            "complete": self.complete,
            "timestamp": self.timestamp,
            "field_size_um": self.calibration.field_size_um,
            "channels": list(self.images),
        }


class FrameSlot:
    """Latest-value slot with a condition variable (single writer, many readers)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._frame: Frame | None = None
        self._last_complete: Frame | None = None

    def publish(self, frame: Frame) -> None:
        with self._cv:
            self._frame = frame
            if frame.complete:
                self._last_complete = frame
            self._cv.notify_all()

    def latest(self) -> Frame | None:
        with self._cv:
            return self._frame

    def latest_complete(self) -> Frame | None:
        with self._cv:
            return self._last_complete

    @property
    def seq(self) -> int:
        with self._cv:
            return self._frame.seq if self._frame else 0

    def wait(self, after_seq: int = 0, timeout: float | None = None, complete: bool = False) -> Frame | None:
        """Block until a frame newer than *after_seq* (and complete, if asked) is published."""
        deadline = None if timeout is None else time.monotonic() + timeout

        def ok():
            f = self._last_complete if complete else self._frame
            return f is not None and f.seq > after_seq

        with self._cv:
            while not ok():
                rem = None if deadline is None else deadline - time.monotonic()
                if rem is not None and rem <= 0:
                    return None
                self._cv.wait(rem)
            return self._last_complete if complete else self._frame

    def clear(self) -> None:
        with self._cv:
            self._frame = None
            self._last_complete = None


@dataclass(frozen=True)
class EngineEvent:
    kind: str  # "state" | "error" | "warning" | "info" | "frame" | "settings"
    message: str = ""
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


class Subscription:
    """Bounded event queue of one consumer.  Use :meth:`get` or :meth:`drain`."""

    def __init__(self, bus: "EventBus", maxlen: int):
        self._bus = bus
        self._q: collections.deque = collections.deque(maxlen=maxlen)
        self._cv = threading.Condition()
        self.dropped = 0

    def _put(self, ev: EngineEvent) -> None:
        with self._cv:
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(ev)
            self._cv.notify_all()

    def get(self, timeout: float | None = None) -> EngineEvent | None:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    def drain(self) -> list[EngineEvent]:
        with self._cv:
            out = list(self._q)
            self._q.clear()
            return out

    def close(self) -> None:
        self._bus._remove(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EventBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self.history: collections.deque = collections.deque(maxlen=200)

    def subscribe(self, maxlen: int = 500) -> Subscription:
        s = Subscription(self, maxlen)
        with self._lock:
            self._subs.append(s)
        return s

    def _remove(self, s: Subscription) -> None:
        with self._lock:
            if s in self._subs:
                self._subs.remove(s)

    def emit(self, kind: str, message: str = "", **data) -> EngineEvent:
        ev = EngineEvent(kind, message, data)
        with self._lock:
            subs = list(self._subs)
            self.history.append(ev)
        for s in subs:
            s._put(ev)
        return ev
