"""Hitachi S-3000 remote control (RS-232): SEM column + five-axis stage HAL.

One serial port carries both the SEM text protocol and the stage protocol.
:class:`MicroscopeLink` owns it and executes requests one at a time in
priority order (SEM settings first, background polling last).  Two HAL modules
share the link:

* :class:`SEMControl` – magnification, HV, beam, focus, stigmation, contrast …
  (implements :class:`MicroscopeControl`, the interface imaging and EDS use);
* :class:`StageControl` – stage positions (polled with the SEM readbacks, by priority level), moves, stop.

Quick start (simulator)::

    from easyebl.microscope import open_microscope
    link, sem, stage = open_microscope("sim")
    sem.set_magnification(2000).result()
    stage.engage().result()          # take the stage from the SEM PC (#STAGE ON)
    stage.move({"X": 100}, relative=True).result()
    link.close()

The driver (:mod:`easyebl.microscope.s3000`) is vendored from
hitachi_s3000_control_lib.
"""
from .control import MicroscopeControl, NotAvailable, StubMicroscope
from .link import POLL, READ, SEM_SET, STAGE_CMD, MicroscopeLink
from .sem import MAGNIFICATIONS, PRESET_KEYS, SEMControl
from .poll import Poller
from .stage import AXES, StageControl

__all__ = ["MicroscopeControl", "NotAvailable", "StubMicroscope", "MicroscopeLink", "SEMControl", "StageControl",
           "AXES", "Poller", "MAGNIFICATIONS", "PRESET_KEYS", "SEM_SET", "STAGE_CMD", "READ", "POLL",
           "open_microscope"]


def open_microscope(port: str, command_delay_s: float = 0.25, sem_polling: bool = True,
                    stage_polling: bool = True, **session_kw):
    """(link, SEMControl, StageControl) on *port* (e.g. ``"COM4"``) or ``"sim"`` (simulator)."""
    if str(port).lower() in ("sim", "simulator"):
        link = MicroscopeLink.simulated(command_delay_s=command_delay_s)
    else:
        link = MicroscopeLink.open(port, **session_kw)
    sem = SEMControl(link, polling=sem_polling)
    stage = StageControl(link, polling=stage_polling)
    return link, sem, stage
