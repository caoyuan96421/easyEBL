"""The opened microscope as one object: link, SEM and stage HAL modules, settings.

The serial port can be opened by one program at a time, so the easyEBL
backend opens it once (:func:`open_bundle`) and hands the same
:class:`MicroscopeBundle` to every window that needs it: the SEM window
(column panel, data bar, image metadata), the EDS window (beam voltage) and
the microscope window.
"""
from __future__ import annotations

from .control import StubMicroscope
from .settings import MicroscopeSettings

__all__ = ["MicroscopeBundle", "open_bundle"]


class MicroscopeBundle:
    """The link, the SEM and stage HAL modules, the settings – or a stub when not connected."""

    def __init__(self, link=None, sem=None, stage=None, settings: MicroscopeSettings | None = None):
        self.link, self.stage = link, stage
        self.sem = sem if sem is not None else StubMicroscope()
        self.settings = settings or MicroscopeSettings()
        self.follower = None
        #: ``opener()`` shows the microscope window (set by the easyEBL app, which manages the windows)
        self.opener = None
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
        """A microscope window of its own (when no app manages the windows)."""
        if self.link is None:
            return None
        if self._window is None:
            from .gui import MicroscopeWindow

            self._window = MicroscopeWindow(self.link, self.sem, self.stage, self.settings, owns_link=False)
            self._window.resize(1200, 860)
        return self._window

    def show_window(self):
        if self.opener is not None:
            return self.opener()
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


def open_bundle(port: str | None, baudrate: int = 9600, settings: MicroscopeSettings | None = None,
                command_delay_s: float = 0.25) -> MicroscopeBundle:
    """Open the microscope on *port* (``"COM4"``), the simulator (``"sim"``) or nothing (``"none"``)."""
    from . import open_microscope

    settings = settings or MicroscopeSettings.load()
    if not port or str(port).lower() in ("none", "off"):
        return MicroscopeBundle(settings=settings)
    kw = {} if str(port).lower() in ("sim", "simulator") else {"baudrate": int(baudrate)}
    link, sem, stage = open_microscope(port, command_delay_s=command_delay_s, stage_poll_s=settings.stage_poll_s,
                                       sem_status_poll_s=settings.sem_status_poll_s, **kw)
    if not link.simulated:
        settings.port = port
    return MicroscopeBundle(link, sem, stage, settings)
