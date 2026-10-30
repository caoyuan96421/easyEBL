"""Shared access to one DAQ device.

There is exactly one :class:`DAQManager` per physical device.  Application
modules (imaging, alignment, lithography, GUIs, …) each obtain their own
:class:`DAQHandle`::

    mgr = DAQManager(SimulatedBackend(cfg)).open()
    imaging = mgr.get_handle("imaging")
    litho = mgr.get_handle("litho")

Rules enforced here (not in the backends):

* **One hardware-timed operation at a time.**  ``run_stream`` takes the device
  operation lock.  A second caller either waits (``timeout``) or gets
  :class:`DeviceBusyError`.  The manager records which handle is active.
* **Exclusive claims for auxiliary resources.**  Software-timed lines (e.g.
  the scan relay) can only be driven by the handle that claimed them.
  Arbitrary named resources (``"stage"``, ``"sem_serial"`` …) can be claimed
  too, so the same mechanism coordinates future subsystems.
* **Safe state.**  After every stream – normal end, abort or exception – the
  beam is blanked and parked.  Static writes are rejected while a stream is
  running.
* **Output validation.**  Rate limits, AO voltage limits, AI channel names and
  marker line names are checked before/while streaming.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import threading
import time
from typing import Callable, Iterable, Iterator, Sequence
import weakref

import numpy as np

from ..core.waveform import OutputChunk
from .backend import (
    AIConsumer,
    DAQBackend,
    DeviceBusyError,
    HALError,
    ResourceConflictError,
    StreamResult,
    VoltageRangeError,
)

__all__ = ["DAQManager", "DAQHandle", "BeamState"]

Listener = Callable[[str, dict], None]


@dataclass
class BeamState:
    x_v: float
    y_v: float
    blank: bool


class DAQManager:
    """Owner of a backend; hands out :class:`DAQHandle` objects."""

    def __init__(self, backend: DAQBackend):
        self.backend = backend
        self._op_lock = threading.Lock()
        self._state = threading.RLock()
        self._claims: dict[str, "DAQHandle"] = {}
        self._handles: "weakref.WeakSet[DAQHandle]" = weakref.WeakSet()
        self._active: DAQHandle | None = None
        self._active_abort: threading.Event | None = None
        self._listeners: list[Listener] = []
        self._ids = itertools.count(1)
        self._beam = BeamState(backend.config.park_x_v, backend.config.park_y_v, True)
        self._line_state: dict[str, bool] = {}

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> "DAQManager":
        if not self.backend.is_open:
            self.backend.open()
            self.backend.safe_state()
            self._emit("opened", {})
        return self

    def close(self) -> None:
        with self._state:
            if self._active_abort is not None:
                self._active_abort.set()
        # wait for a running stream to finish its cleanup
        with self._op_lock:
            if self.backend.is_open:
                try:
                    self.backend.safe_state()
                finally:
                    self.backend.close()
                    self._emit("closed", {})

    def __enter__(self) -> "DAQManager":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------------- queries
    @property
    def config(self):
        return self.backend.config

    def capabilities(self):
        return self.backend.capabilities()

    @property
    def is_busy(self) -> bool:
        return self._op_lock.locked()

    @property
    def active_owner(self) -> str | None:
        with self._state:
            return self._active.owner if self._active else None

    @property
    def beam_state(self) -> BeamState:
        with self._state:
            return BeamState(self._beam.x_v, self._beam.y_v, self._beam.blank)

    @property
    def line_state(self) -> dict:
        with self._state:
            return dict(self._line_state)

    def handles(self) -> list["DAQHandle"]:
        return [h for h in list(self._handles) if not h.closed]

    def claims(self) -> dict[str, str]:
        with self._state:
            return {r: h.owner for r, h in self._claims.items()}

    # ------------------------------------------------------------------ handles
    def get_handle(self, owner: str) -> "DAQHandle":
        h = DAQHandle(self, owner, next(self._ids))
        self._handles.add(h)
        self._emit("handle_created", {"owner": owner})
        return h

    def abort_all(self) -> None:
        """Abort whatever stream is running (emergency stop)."""
        with self._state:
            if self._active_abort is not None:
                self._active_abort.set()

    # ---------------------------------------------------------------- listeners
    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, event: str, data: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(event, data)
            except Exception:  # listeners must never break hardware control
                pass

    # -------------------------------------------------------------- internals
    def _claim(self, handle: "DAQHandle", resources: Sequence[str]) -> None:
        with self._state:
            for r in resources:
                holder = self._claims.get(r)
                if holder is not None and holder is not handle:
                    raise ResourceConflictError(f"'{r}' is claimed by '{holder.owner}'")
            for r in resources:
                self._claims[r] = handle
        self._emit("claim", {"owner": handle.owner, "resources": list(resources)})

    def _release(self, handle: "DAQHandle", resources: Sequence[str] | None) -> None:
        with self._state:
            if resources is None:
                resources = [r for r, h in self._claims.items() if h is handle]
            for r in resources:
                if self._claims.get(r) is handle:
                    del self._claims[r]
        self._emit("release", {"owner": handle.owner, "resources": list(resources)})

    def _holds(self, handle: "DAQHandle", resource: str) -> bool:
        with self._state:
            return self._claims.get(resource) is handle

    def _static(self, handle: "DAQHandle", x_v: float, y_v: float, blank: bool) -> None:
        lim = self.config.ao_limit_v
        if abs(x_v) > lim or abs(y_v) > lim:
            raise VoltageRangeError(f"static position ({x_v:.4g}, {y_v:.4g}) V exceeds ±{lim} V")
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError(f"device busy (stream owned by '{self.active_owner}')")
        try:
            self.backend.write_static(float(x_v), float(y_v), bool(blank))
            with self._state:
                self._beam = BeamState(float(x_v), float(y_v), bool(blank))
        finally:
            self._op_lock.release()
        self._emit("static", {"owner": handle.owner, "x_v": x_v, "y_v": y_v, "blank": blank})

    def _set_line(self, handle: "DAQHandle", name: str, value: bool) -> None:
        if name not in self.config.static_lines:
            raise HALError(f"unknown static line '{name}'")
        if not self._holds(handle, name):
            holder = self.claims().get(name)
            raise ResourceConflictError(
                f"line '{name}' must be claimed first" + (f" (held by '{holder}')" if holder else "")
            )
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError("cannot switch static lines while a stream is running")
        try:
            # the relay may switch deflection sources: always blank first
            self.backend.write_static(self._beam.x_v, self._beam.y_v, True)
            self.backend.set_static_line(name, bool(value))
            with self._state:
                self._beam.blank = True
                self._line_state[name] = bool(value)
        finally:
            self._op_lock.release()
        self._emit("line", {"owner": handle.owner, "line": name, "value": bool(value)})

    def _validate_stream(self, rate: float, ai_channels: Sequence[str]) -> None:
        cfg = self.config
        if rate <= 0:
            raise ValueError("rate must be positive")
        if rate > cfg.max_ao_rate:
            raise HALError(f"rate {rate:g} S/s exceeds AO limit {cfg.max_ao_rate:g} S/s")
        unknown = [c for c in ai_channels if c not in cfg.ai_channels]
        if unknown:
            raise HALError(f"unknown AI channel(s) {unknown}; configured: {list(cfg.ai_channels)}")
        if ai_channels and rate * len(ai_channels) > cfg.max_ai_aggregate_rate:
            raise HALError(
                f"AI aggregate rate {rate * len(ai_channels):g} S/s exceeds "
                f"{cfg.max_ai_aggregate_rate:g} S/s"
            )

    def _checked(self, chunks: Iterable[OutputChunk]) -> Iterator[OutputChunk]:
        lim = self.config.ao_limit_v * (1 + 1e-9)
        allowed = set(self.config.timed_lines)
        for c in chunks:
            if c.max_abs_voltage() > lim:
                raise VoltageRangeError(
                    f"output sample {c.max_abs_voltage():.4g} V exceeds ±{self.config.ao_limit_v} V"
                )
            extra = set(c.markers) - allowed
            if extra:
                raise HALError(f"marker line(s) {sorted(extra)} not configured as timed_lines")
            yield c

    def _run(
        self,
        handle: "DAQHandle",
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str],
        on_ai: AIConsumer | None,
        progress: Callable[[int], None] | None,
        timeout: float,
    ) -> StreamResult:
        self._validate_stream(rate, ai_channels)
        if self._active is handle:
            raise DeviceBusyError("handle already running a stream (re-entrant call)")
        if timeout is None or timeout < 0:
            ok = self._op_lock.acquire()
        else:
            ok = self._op_lock.acquire(timeout=timeout) if timeout > 0 else self._op_lock.acquire(False)
        if not ok:
            raise DeviceBusyError(f"device busy (stream owned by '{self.active_owner}')")
        abort = threading.Event()
        with self._state:
            self._active = handle
            self._active_abort = abort
        self._emit("stream_started", {"owner": handle.owner, "rate": rate})
        t0 = time.perf_counter()
        result: StreamResult | None = None
        try:
            result = self.backend.run_stream(
                rate, self._checked(chunks), tuple(ai_channels), on_ai, abort, progress
            )
            return result
        finally:
            try:
                self.backend.safe_state()
            finally:
                with self._state:
                    self._beam = BeamState(self.config.park_x_v, self.config.park_y_v, True)
                    self._active = None
                    self._active_abort = None
                self._op_lock.release()
                self._emit(
                    "stream_finished",
                    {
                        "owner": handle.owner,
                        "aborted": bool(result.aborted) if result else True,
                        "elapsed_s": time.perf_counter() - t0,
                        "samples": result.samples_written if result else 0,
                    },
                )

    def _abort(self, handle: "DAQHandle") -> bool:
        with self._state:
            if self._active is handle and self._active_abort is not None:
                self._active_abort.set()
                return True
        return False


class DAQHandle:
    """Per-module access token for a shared :class:`DAQManager`."""

    def __init__(self, manager: DAQManager, owner: str, uid: int):
        self._mgr = manager
        self.owner = owner
        self.uid = uid
        self.closed = False

    def __repr__(self) -> str:
        return f"<DAQHandle {self.owner!r} #{self.uid}{' closed' if self.closed else ''}>"

    def _check(self) -> None:
        if self.closed:
            raise HALError(f"handle '{self.owner}' is closed")
        if not self._mgr.backend.is_open:
            raise HALError("device is not open")

    # ----------------------------------------------------------------- queries
    @property
    def manager(self) -> DAQManager:
        return self._mgr

    @property
    def config(self):
        return self._mgr.config

    def capabilities(self):
        return self._mgr.capabilities()

    @property
    def is_active(self) -> bool:
        return self._mgr._active is self

    @property
    def claimed(self) -> set[str]:
        return {r for r, o in self._mgr._claims.items() if o is self}

    # ----------------------------------------------------------------- claims
    def claim(self, *resources: str) -> "DAQHandle":
        self._check()
        self._mgr._claim(self, resources)
        return self

    def release(self, *resources: str) -> None:
        self._mgr._release(self, resources if resources else None)

    # -------------------------------------------------------------- static I/O
    def park(self, x_v: float | None = None, y_v: float | None = None, blank: bool = True) -> None:
        """Move the beam statically (default: configured park position, blanked)."""
        self._check()
        cfg = self._mgr.config
        self._mgr._static(
            self, cfg.park_x_v if x_v is None else x_v, cfg.park_y_v if y_v is None else y_v, blank
        )

    def set_blank(self, blank: bool) -> None:
        self._check()
        st = self._mgr.beam_state
        self._mgr._static(self, st.x_v, st.y_v, blank)

    def set_line(self, name: str, value: bool) -> None:
        self._check()
        self._mgr._set_line(self, name, value)

    # ----------------------------------------------------------------- streams
    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        progress: Callable[[int], None] | None = None,
        timeout: float = 0.0,
    ) -> StreamResult:
        """Run a hardware-timed stream.

        ``timeout``: seconds to wait for the device if another handle is
        streaming (0 = fail immediately, negative/None = wait forever).
        """
        self._check()
        return self._mgr._run(self, rate, chunks, ai_channels, on_ai, progress, timeout)

    def abort(self) -> bool:
        """Abort this handle's running stream.  Returns True if one was running."""
        return self._mgr._abort(self)

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if not self.closed:
            self.abort()
            self.release()
            self.closed = True

    def __enter__(self) -> "DAQHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
