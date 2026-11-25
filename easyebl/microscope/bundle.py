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
        #: the global field calibration (field = K / magnification), see :meth:`tie_field`
        self.field_scale = None
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

    def tie_field(self, targets, sim_backend=None, persist: bool = True, notify=None):
        """The global field calibration for *targets* (imaging engine, EDS controller): their field is
        ``K / magnification`` (the manual magnification while the microscope reports none).  With
        *persist* (and a real microscope or none) ``K`` is kept in the settings file; simulators keep it
        in memory so they never overwrite the instrument's calibration."""
        from .coupling import FieldScale

        if self.field_scale is not None:
            self.field_scale.close()
        keep = persist and not (self.link is not None and self.link.simulated)
        self.field_scale = FieldScale(self.sem, targets, self.settings if keep else None,
                                      sim_backend=sim_backend, notify=notify)
        return self.field_scale

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
        if self.field_scale is not None:
            self.field_scale.close()
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
    link, sem, stage = open_microscope(port, command_delay_s=command_delay_s, sem_polling=settings.sem_polling,
                                       stage_polling=settings.stage_polling, **kw)
    if not link.simulated:
        settings.port = port
    stage.backlash_um = float(settings.stage_backlash_um)
    return MicroscopeBundle(link, sem, stage, settings)
