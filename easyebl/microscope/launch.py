"""Opening the microscope from command-line arguments, for the SEM, EDS and suite programs.

The serial port can be opened by one program at a time, so each program opens
it once (:func:`open_from_args`) and hands the same :class:`MicroscopeBundle`
to every window that needs it: the SEM window (column panel, data bar, image
metadata), the EDS window (beam voltage) and the microscope window.
"""
from __future__ import annotations

import argparse

from .control import StubMicroscope
from .settings import MicroscopeSettings

__all__ = ["add_microscope_args", "open_from_args", "MicroscopeBundle"]


def add_microscope_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--microscope", default=None, metavar="PORT",
                   help="SEM remote control (Hitachi S-3000, RS-232): a serial port (e.g. COM4), 'sim' or 'none'; "
                        "default: 'sim' with --backend sim, else 'none'")
    return p


class MicroscopeBundle:
    """The link, the SEM and stage HAL modules, the settings – or a stub when not connected."""

    def __init__(self, link=None, sem=None, stage=None, settings: MicroscopeSettings | None = None):
        self.link, self.stage = link, stage
        self.sem = sem if sem is not None else StubMicroscope()
        self.settings = settings or MicroscopeSettings()
        self.follower = None
        self._window = None

    @property
    def connected(self) -> bool:
        return self.link is not None

    def metadata(self) -> dict:
        """Column + stage readbacks for image / spectrum metadata."""
        d = dict(self.sem.snapshot())
        if self.stage is not None:
            d.update(self.stage.snapshot())
        return d

    def follow_magnification(self, targets, sim_backend=None, notify=None):
        """Tie the field calibration of *targets* (imaging engine, EDS controller) to the magnification."""
        if self.link is None:
            return None
        from .coupling import FieldFollower

        # the simulator's K is kept in memory: it must not overwrite the instrument's calibration on disk
        settings = None if self.link.simulated else self.settings
        self.follower = FieldFollower(self.sem, targets, settings, sim_backend=sim_backend, notify=notify)
        return self.follower

    def window(self, parent=None):
        """The microscope window (created once; the bundle keeps owning the port)."""
        if self.link is None:
            return None
        if self._window is None:
            from .gui import MicroscopeWindow

            self._window = MicroscopeWindow(self.link, self.sem, self.stage, self.settings, owns_link=False)
            self._window.resize(1200, 860)
        return self._window

    def show_window(self):
        w = self.window()
        if w is not None:
            w.show()
            w.raise_()
            w.activateWindow()
        return w

    def close(self) -> None:
        if self.follower is not None:
            self.follower.close()
        if self._window is not None:
            self._window.close()
        if self.link is not None:
            try:
                self.settings.save()
            except OSError:
                pass
            self.link.close()


def open_from_args(args, settings: MicroscopeSettings | None = None) -> MicroscopeBundle:
    """Open the microscope named by ``--microscope`` (a stub bundle for 'none')."""
    port = getattr(args, "microscope", None)
    if port is None:
        port = "sim" if getattr(args, "backend", "sim") == "sim" else "none"
    settings = settings or MicroscopeSettings.load()
    if str(port).lower() in ("none", "off", ""):
        return MicroscopeBundle(settings=settings)
    from . import open_microscope

    link, sem, stage = open_microscope(port, stage_poll_s=settings.stage_poll_s,
                                       sem_status_poll_s=settings.sem_status_poll_s)
    if str(port).lower() not in ("sim", "simulator"):
        settings.port = port
    return MicroscopeBundle(link, sem, stage, settings)
