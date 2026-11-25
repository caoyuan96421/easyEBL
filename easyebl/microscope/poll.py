"""One background poller for everything the microscope can be asked about – by priority level.

Every pollable quantity (each SEM readback item, each stage controller) is a
:class:`PollTask` with a priority *level* (1 = most important).  The poller sends
one request at a time, at the lowest link priority (``POLL``), the next as soon as
the last is back, and always picks the task with the largest

    age / RATIO ** (level - 1)        (age: time since its last read)

so a task of level n comes round :attr:`Poller.RATIO` (2) times less often than one
of level n - 1, whatever their read times: lower levels are slowed, never starved.
A task whose age reaches :attr:`Poller.MAX_PERIOD_S` (15 s) goes before all others
(the most overdue first), so nothing is read less often than that while the port
can carry it.  Read times are measured on every read (``cost``, a running average)
for the record only.

Levels in use (S-3000, measured 2026-11-23: a SEM item 0.2–0.5 s, a stage
controller 1.0–1.3 s): 1 MAG, HV (≈2.7 s); 2 focus, stigmator, beam, WD (≈5 s);
3 stage X/Y and Z/T (≈11 s); 4 stage R and the other SEM readbacks (15 s).

Settings and moves (higher link priorities) wait at most for the read in progress.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .link import POLL

__all__ = ["Poller", "PollTask"]


@dataclass
class PollTask:
    name: str
    read: Callable  # read(session) – runs on the link worker
    level: int = 1  # priority level: 1 most important; each level is read RATIO x less often
    key: object = None  # de-duplication key shared with explicit reads of the same thing
    what: str = ""
    owner: object = None
    enabled: Callable[[], bool] = lambda: True
    cost: float = 0.3  # s, running average of the measured read time (for the record)
    last: float | None = None  # monotonic time of the last poll
    period: float | None = None  # s, running average of the time between polls
    reads: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def score(self, now: float, ratio: float = 2.0, max_period_s: float | None = None) -> float:
        if self.last is None:
            return float("inf")
        age = now - self.last
        if max_period_s and age >= max_period_s:
            return 1e12 + age  # overdue: before every task within its period, the oldest first
        return age / ratio ** (max(1, int(self.level)) - 1)


class Poller:
    """Priority-level round of :class:`PollTask` reads over a :class:`MicroscopeLink`."""

    #: a level is read this many times less often than the level above it
    RATIO = 2.0
    #: no task is read less often than this (s) while the port can carry it
    MAX_PERIOD_S = 15.0
    #: weight of the newest measurement in the running averages
    ALPHA = 0.3

    def __init__(self, link):
        self.link = link
        self._tasks: list[PollTask] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.reads = 0
        self._thread = threading.Thread(target=self._loop, name="easyebl-microscope-poll", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ tasks
    def add(self, name: str, read: Callable, *, level: int = 1, owner=None, key=None, what: str = "",
            enabled=None, cost_s: float = 0.3) -> PollTask:
        t = PollTask(name, read, level=int(level), key=key, what=what or f"poll {name}", owner=owner,
                     enabled=enabled or (lambda: True), cost=float(cost_s))
        with self._lock:
            self._tasks = [x for x in self._tasks if x.name != name] + [t]
        return t

    def remove(self, owner) -> None:
        with self._lock:
            self._tasks = [t for t in self._tasks if t.owner is not owner]

    def tasks(self) -> list[PollTask]:
        with self._lock:
            return list(self._tasks)

    def task(self, name: str) -> PollTask | None:
        return next((t for t in self.tasks() if t.name == name), None)

    def stats(self) -> list[dict]:
        """Per task: name, level, measured read time (s), measured period (s), reads, failures."""
        return [{"name": t.name, "level": t.level, "cost_s": t.cost, "period_s": t.period, "reads": t.reads,
                 "failures": t.failures} for t in self.tasks()]

    def _next(self) -> PollTask | None:
        now = time.monotonic()
        cands = []
        for t in self.tasks():
            try:
                if t.enabled():
                    cands.append(t)
            except Exception:  # noqa: BLE001
                pass
        if not cands:
            return None
        # never-read tasks first, most important level first
        return max(cands, key=lambda t: (t.score(now, self.RATIO, self.MAX_PERIOD_S), -t.level))

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            # nothing is queued behind an exclusive job (a stage move): its result could wait for minutes
            t = self._next() if self.link.connected and not self.link.exclusive else None
            if t is None:
                self._stop.wait(0.25)
                continue
            box = {}

            def run(session, t=t, box=box):
                t0 = time.perf_counter()
                try:
                    return t.read(session)
                finally:
                    box["dt"] = time.perf_counter() - t0

            try:
                fut = self.link.submit(run, POLL, t.what, key=t.key)
            except RuntimeError:  # link closed
                return
            ok = True
            deadline = time.monotonic() + 120
            while True:
                try:
                    fut.result(timeout=1.0)
                    break
                except concurrent.futures.TimeoutError:
                    if self.link.exclusive:  # waiting behind a stage move is not a failure
                        deadline = time.monotonic() + 120
                    if time.monotonic() > deadline or self._stop.is_set():
                        ok = False
                        break
                except Exception:  # noqa: BLE001 - failed / cancelled
                    ok = False
                    break
            now = time.monotonic()
            a = self.ALPHA
            if "dt" in box:  # it ran (an explicit read of the same key may have served it instead)
                t.cost = box["dt"] if t.reads == 0 else (1 - a) * t.cost + a * box["dt"]
            if t.last is not None:
                d = now - t.last
                t.period = d if t.period is None else (1 - a) * t.period + a * d
            t.last = now
            if ok:
                t.reads += 1
                self.reads += 1
            else:
                t.failures += 1
                self._stop.wait(1.0)

    def close(self) -> None:
        self._stop.set()
