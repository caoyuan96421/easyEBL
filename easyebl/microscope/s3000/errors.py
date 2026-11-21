"""Exception hierarchy for the Hitachi S-3000 control library."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .sem import SemReply


class S3000Error(Exception):
    """Base class for every error raised by this library."""


class ValidationError(S3000Error, ValueError):
    """A command, argument or range check failed before anything was sent."""


class ConfigurationError(S3000Error):
    """The library was asked to do something that needs explicit configuration or confirmation.

    Examples: a VAC command with ``low_vacuum_capable=False``, or ``resync()``
    without ``confirm=True``.
    """


class TransportError(S3000Error):
    """The serial port (or simulated transport) failed or was closed."""


class ProtocolError(S3000Error):
    """A reply did not match the documented grammar (bad echo, missing result...)."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class ConnectionUncertain(S3000Error):
    """The link is no longer known to be synchronized.

    Raised when a reply timed out or could not be matched to its request.
    Further commands are refused until :meth:`resync` is called deliberately,
    because a timeout does not prove that the instrument did not act.
    """


class CommandFailed(S3000Error):
    """The SEM answered with a result code other than ``G0``."""

    def __init__(self, reply: "SemReply"):
        super().__init__(f"{reply.request!r} -> {reply.result} ({reply.result_meaning}); raw reply {reply.raw!r}")
        self.reply = reply


class StageError(S3000Error):
    """Base class for five-axis stage errors."""


class StageCommandRejected(StageError):
    """The stage controller answered ``uE1`` (command error) or ``uE2`` (communication error)."""

    def __init__(self, command: str, controller: int, code: int):
        meaning = {1: "command error", 2: "communication error"}.get(code, "unknown")
        super().__init__(f"{command!r} rejected by stage controller {controller}: E{code} ({meaning})")
        self.command = command
        self.controller = controller
        self.code = code


class StageRejectedBySem(StageError):
    """The SEM itself answered a stage command in SEM grammar (``#00R1 E2``).

    On the S-3000 the stage protocol is relayed through the SEM text port. While
    stage control is off the SEM rejects every ``#uabc`` line with ``E2``
    instead of forwarding it; send ``#STAGE ON`` (``sem.stage_control(True)``)
    first. ``E6`` means the S3 motor-drive stage is not connected.
    """

    HINTS = {
        "E2": "stage control is off; send '#STAGE ON' first (sem.stage_control(True))",
        "E6": "S3 motor drive stage unconnected",
    }

    def __init__(self, command: str, result: str, raw: str):
        hint = self.HINTS.get(result, "SEM result code")
        super().__init__(f"{command!r} answered by the SEM with {result} ({hint}); raw reply {raw!r}")
        self.command = command
        self.result = result
        self.raw = raw


class MotionError(StageError):
    """A move ended in a state other than normal completion (halt, limit, motor error, timeout)."""

    def __init__(self, message: str, states: dict | None = None):
        super().__init__(message)
        self.states = states or {}
