"""The microscope (column) interface the other subsystems use – no I/O here.

Imaging and EDS only see :class:`MicroscopeControl`: cached readbacks that
return at once, and setters that *queue* a request and return immediately (a
:class:`concurrent.futures.Future`).  RS-232 transactions take 0.25–1 s, so
nothing may block a GUI thread on them.

:class:`StubMicroscope` is the "no remote control" implementation: readbacks
are None and every action raises :class:`NotAvailable`.  The real one is
:class:`easyebl.microscope.sem.SEMControl`.
"""
from __future__ import annotations

from concurrent.futures import Future
from typing import Callable

__all__ = ["MicroscopeControl", "StubMicroscope", "NotAvailable"]


class NotAvailable(NotImplementedError):
    """The microscope function is not remote-controllable (or no microscope is connected)."""


class MicroscopeControl:
    """Column control API used by the SEM and EDS windows and scripts.

    Readbacks (``magnification()`` …) return the last value read from the SEM, or
    None if unknown.  Actions return a Future (resolved when the SEM has
    answered) or raise :class:`NotAvailable` at once.
    """

    name = "microscope"

    @property
    def connected(self) -> bool:
        return False

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        """*fn(changes)* is called (from a worker thread) whenever readbacks change."""

    def remove_listener(self, fn) -> None:
        """Undo :meth:`add_listener`."""

    def snapshot(self) -> dict:
        """Current readbacks (for image metadata)."""
        return {}

    # magnification
    def magnification(self) -> float | None:
        return None

    def set_magnification(self, mag) -> Future:
        raise NotAvailable("magnification")

    def step_magnification(self, steps: int) -> Future:
        raise NotAvailable("magnification")

    # focus / working distance
    def working_distance_mm(self) -> float | None:
        return None

    def focus(self, delta: float, coarse: bool = False) -> Future:
        raise NotAvailable("focus")

    def auto_focus(self) -> Future:
        raise NotAvailable("auto focus")

    # stigmation
    def stigmate(self, dx: float, dy: float) -> Future:
        raise NotAvailable("stigmation")

    def auto_stigma(self) -> Future:
        raise NotAvailable("auto stigma")

    # beam
    def accelerating_voltage_kv(self) -> float | None:
        """The accelerating voltage while HV is on, else None."""
        return None

    def alignment(self, kind: str) -> Future:
        raise NotAvailable(f"{kind} alignment")


class StubMicroscope(MicroscopeControl):
    name = "not connected"
