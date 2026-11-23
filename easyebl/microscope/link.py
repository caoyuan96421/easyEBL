"""One RS-232 port shared by the SEM and the stage: a prioritised, single-file request queue.

The S-3000 answers one request at a time (0.25–0.75 s per SEM transaction,
0.5–1.1 s per stage query).  :class:`MicroscopeLink` owns the port (one
:class:`~easyebl.microscope.s3000.Session`) and a single worker thread that
executes queued requests in priority order:

==============  =====  ==============================================
priority        value  used for
==============  =====  ==============================================
``SEM_SET``     0      SEM settings and actions (always first)
``STAGE_CMD``   1      stage moves and the refresh after a move
``READ``        2      explicit SEM / stage reads
``POLL``        3      background polling (:mod:`~easyebl.microscope.poll`)
==============  =====  ==============================================

A request is one or a few transactions, so a SEM setting waits at most for the
transaction in progress.  Stage **halts bypass the queue** (they go straight to
the stage client, which sends S0 even while the bus is busy).

The link never retries on its own.  After a timeout the driver refuses further
commands (``uncertain``) until :meth:`MicroscopeLink.resync` – call it after
checking the instrument.
"""
from __future__ import annotations

import concurrent.futures
import itertools
import queue
import threading
import time
from typing import Callable

from ..imaging.engine.frames import EventBus
from .s3000 import ConnectionUncertain, Session, TransportError

__all__ = ["MicroscopeLink", "SEM_SET", "STAGE_CMD", "READ", "POLL"]

SEM_SET, STAGE_CMD, READ, POLL = 0, 1, 2, 3
_NAMES = {SEM_SET: "SEM", STAGE_CMD: "stage", READ: "read", POLL: "poll"}


class _Request:
    __slots__ = ("priority", "seq", "fn", "future", "what", "key")

    def __init__(self, priority, seq, fn, what, key):
        self.priority, self.seq, self.fn, self.what, self.key = priority, seq, fn, what, key
        self.future: concurrent.futures.Future = concurrent.futures.Future()

    def __lt__(self, other):
        return (self.priority, self.seq) < (other.priority, other.seq)


class MicroscopeLink:
    """Owner of the microscope port: queue, worker thread, connection state and events.

    ``events`` (an :class:`~easyebl.imaging.engine.frames.EventBus`) carries
    ``state`` (connection), ``error``, ``sem`` (readback changes), ``stage``
    (positions / motion) and ``stage_event`` (limits, faults) events.
    """

    def __init__(self, session: Session, port: str = "", simulated: bool = False):
        self.session = session
        self.port = port
        self.simulated = simulated
        self.events = EventBus()
        self._q: "queue.PriorityQueue[_Request]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._keys: dict = {}  # key -> pending request (de-duplication)
        self._klock = threading.Lock()
        self._state = "connected"
        self._error: str | None = None
        self._busy: _Request | None = None
        self.completed = 0
        self.executed: list[tuple[int, str]] = []  # (priority, what) – last 200, for tests and the log
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, name="easyebl-microscope", daemon=True)
        self._worker.start()
        self.sem = None  # SEMControl (set by its constructor)
        self.stage = None  # StageControl
        self._poller = None

    @property
    def poller(self):
        """The background poller every module registers its readbacks with (created on first use)."""
        if self._poller is None:
            from .poll import Poller

            self._poller = Poller(self)
        return self._poller

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, port: str, **session_kw) -> "MicroscopeLink":
        """Real instrument on *port* (9600 8N1); SEM and stage share it."""
        return cls(Session.open(port, **session_kw), port=port)

    @classmethod
    def simulated(cls, command_delay_s: float = 0.25, **sim_kw) -> "MicroscopeLink":
        """The S-3000 simulator with the instrument's response time (``command_delay_s`` per transaction)."""
        from .sim import simulated_session

        return cls(simulated_session(command_delay_s=command_delay_s, **sim_kw), port="simulator", simulated=True)

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        """``connected``, ``uncertain`` (a reply timed out – resync after checking),
        ``disconnected`` (port error) or ``closed``."""
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def connected(self) -> bool:
        return self._state == "connected"

    @property
    def pending(self) -> int:
        return self._q.qsize()

    @property
    def busy_with(self) -> str | None:
        r = self._busy
        return r.what if r is not None else None

    def _set_state(self, state: str, error: str | None = None) -> None:
        if state == self._state and error == self._error:
            return
        self._state, self._error = state, error
        self.events.emit("state", f"microscope link {state}" + (f": {error}" if error else ""),
                         state=state, error=error)

    def log_entries(self):
        """Timestamped raw TX/RX lines of the port."""
        return self.session.channels[0].log_entries()

    # ------------------------------------------------------------------ queue
    def submit(self, fn: Callable[[Session], object], priority: int = READ, what: str = "",
               key=None) -> concurrent.futures.Future:
        """Queue ``fn(session)``.  With *key*, a request with the same key that is still
        waiting is reused (its future is returned) instead of queueing a duplicate."""
        if self._closed.is_set():
            raise RuntimeError("microscope link closed")
        with self._klock:
            if key is not None:
                old = self._keys.get(key)
                if old is not None and not old.future.done() and old is not self._busy:
                    return old.future
            r = _Request(priority, next(self._seq), fn, what or getattr(fn, "__name__", "request"), key)
            if key is not None:
                self._keys[key] = r
        self._q.put(r)
        return r.future

    def call(self, fn, priority: int = READ, what: str = "", timeout: float | None = 60.0):
        """Queue and wait (for scripts and tests – not for GUI threads)."""
        return self.submit(fn, priority, what).result(timeout)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                r = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if r is None or self._closed.is_set():
                if r is not None:
                    r.future.cancel()
                break
            if not r.future.set_running_or_notify_cancel():
                continue
            self._busy = r
            try:
                res = r.fn(self.session)
            except BaseException as e:  # noqa: BLE001 - delivered to the caller
                r.future.set_exception(e)
                self._failed(r, e)
            else:
                r.future.set_result(res)
                if self._state == "disconnected":
                    self._set_state("connected")
            finally:
                self._busy = None
                self.completed += 1
                self.executed.append((r.priority, r.what))
                del self.executed[:-200]
                with self._klock:
                    if r.key is not None and self._keys.get(r.key) is r:
                        del self._keys[r.key]
        # drain: whatever is left is cancelled
        while True:
            try:
                r = self._q.get_nowait()
            except queue.Empty:
                break
            if r is not None:
                r.future.cancel()

    def _failed(self, r: _Request, e: BaseException) -> None:
        if isinstance(e, ConnectionUncertain):
            self._set_state("uncertain", str(e))
        elif isinstance(e, TransportError):
            self._set_state("disconnected", str(e))
        self.events.emit("error", f"{r.what}: {type(e).__name__}: {e}", what=r.what, priority=r.priority)

    def resync(self) -> list:
        """Clear the *uncertain* state after checking the instrument (nothing is resent).
        Returns the late replies the SEM sent in the meantime."""
        s = self.session
        late = s.sem.resync(confirm=True)
        if s.stage.uncertain:
            s.stage.resync(confirm=True)
        self._set_state("connected")
        self.events.emit("info", "microscope link resynchronised" + (f" ({len(late)} late replies)" if late else ""),
                         late=[t for _, t in late])
        return late

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._closed.is_set():
            return
        for mod in (self.stage, self.sem):
            if mod is not None:
                try:
                    mod.close()
                except Exception:  # noqa: BLE001
                    pass
        if self._poller is not None:
            self._poller.close()
        self._closed.set()
        self._q.put(_Request(-99, -1, None, "stop", None))
        self._worker.join(5.0)
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            pass
        self._set_state("closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def describe_priority(p: int) -> str:
    return _NAMES.get(p, str(p))


def wait_all(futures, timeout: float = 30.0) -> None:
    """Wait for several futures (tests / scripts)."""
    deadline = time.monotonic() + timeout
    for f in futures:
        f.result(max(0.0, deadline - time.monotonic()))
