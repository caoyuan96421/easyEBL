"""Microscope (column) control interface – stubs for now.

Magnification, focus / working distance, stigmation and beam alignment are
set on the Hitachi itself.  Remote control (S-3000N RS-232) will implement
:class:`MicroscopeControl`; until then :class:`StubMicroscope` keeps the GUI
wiring in place and reports "not implemented".
"""
from __future__ import annotations

__all__ = ["MicroscopeControl", "StubMicroscope", "NotAvailable"]


class NotAvailable(NotImplementedError):
    """The microscope function is not (yet) remote-controllable."""


class MicroscopeControl:
    """Column control API used by the SEM GUI and scripts."""

    name = "microscope"

    # magnification
    def magnification(self) -> float | None:
        return None

    def set_magnification(self, mag: float) -> None:
        raise NotAvailable("magnification")

    def step_magnification(self, steps: int) -> None:
        raise NotAvailable("magnification")

    # focus / working distance
    def working_distance_mm(self) -> float | None:
        return None

    def focus(self, delta: float, coarse: bool = False) -> None:
        raise NotAvailable("focus")

    def auto_focus(self) -> None:
        raise NotAvailable("auto focus")

    # stigmation
    def stigmate(self, dx: float, dy: float) -> None:
        raise NotAvailable("stigmation")

    def auto_stigma(self) -> None:
        raise NotAvailable("auto stigma")

    # beam
    def accelerating_voltage_kv(self) -> float | None:
        return None

    def alignment(self, kind: str) -> None:
        raise NotAvailable(f"{kind} alignment")


class StubMicroscope(MicroscopeControl):
    name = "not connected"
