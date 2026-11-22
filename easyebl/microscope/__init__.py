"""Hitachi S-3000 remote control (RS-232): SEM column + five-axis stage HAL.

One serial port carries both the SEM text protocol and the stage protocol.
:class:`MicroscopeLink` owns it and executes requests one at a time in
priority order (SEM settings first, background polling last).  Two HAL modules
share the link:

* :class:`SEMControl` – magnification, HV, beam, focus, stigmation, contrast …
  (implements :class:`MicroscopeControl`, the interface imaging and EDS use);
* :class:`StageControl` – stage positions (polled at most every 20 s), moves, stop.

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
from .stage import AXES, MIN_POLL_S, StageControl

__all__ = ["MicroscopeControl", "NotAvailable", "StubMicroscope", "MicroscopeLink", "SEMControl", "StageControl",
           "AXES", "MIN_POLL_S", "MAGNIFICATIONS", "PRESET_KEYS", "SEM_SET", "STAGE_CMD", "READ", "POLL",
           "open_microscope"]


def open_microscope(port: str, command_delay_s: float = 0.25, stage_poll_s: float = 30.0,
                    sem_status_poll_s: float = 10.0, **session_kw):
    """(link, SEMControl, StageControl) on *port* (e.g. ``"COM4"``) or ``"sim"`` (simulator)."""
    if str(port).lower() in ("sim", "simulator"):
        link = MicroscopeLink.simulated(command_delay_s=command_delay_s)
    else:
        link = MicroscopeLink.open(port, **session_kw)
    sem = SEMControl(link, status_poll_s=sem_status_poll_s)
    stage = StageControl(link, poll_s=stage_poll_s)
    return link, sem, stage
