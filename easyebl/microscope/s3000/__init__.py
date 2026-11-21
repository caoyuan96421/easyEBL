"""Hitachi S-3000 RS-232C driver (SEM text protocol + five-axis stage), vendored.

Copied from hitachi_s3000_control_lib (commit 2c6eb28, verified on the S-3000
2026-11-15: every SEM read, k-notation MAG, stage reads, X moves and halts) –
the core only, without its GUI.  easyEBL talks to it through
:mod:`easyebl.microscope` (shared-port link with priorities, SEM and stage HAL
modules); use this package directly only for commissioning.

See the library's docs/commissioning.md for what is verified and what is not.
"""
from . import commands
from .channel import LineChannel, LogEntry
from .errors import (
    CommandFailed,
    ConfigurationError,
    ConnectionUncertain,
    MotionError,
    ProtocolError,
    S3000Error,
    StageCommandRejected,
    StageRejectedBySem,
    StageError,
    TransportError,
    ValidationError,
)
from .sem import RESULT_CODES, SemClient, SemReply, Timeouts
from .session import Session
from .simulator import SimOptions, SimulatedInstrument
from .stage import PULSE_SCALES, Axis, CoordinateCodec, MotionState, StageClient, axis_unit, from_pulses, to_pulses
from .transport import SerialTransport, list_serial_ports

__version__ = "0.1.0+2c6eb28"

__all__ = [
    "PULSE_SCALES",
    "axis_unit",
    "from_pulses",
    "to_pulses",
    "Axis",
    "CommandFailed",
    "ConfigurationError",
    "ConnectionUncertain",
    "CoordinateCodec",
    "LineChannel",
    "LogEntry",
    "MotionError",
    "MotionState",
    "ProtocolError",
    "RESULT_CODES",
    "S3000Error",
    "SemClient",
    "SemReply",
    "SerialTransport",
    "Session",
    "SimOptions",
    "SimulatedInstrument",
    "StageClient",
    "StageCommandRejected",
    "StageRejectedBySem",
    "StageError",
    "Timeouts",
    "TransportError",
    "ValidationError",
    "commands",
    "list_serial_ports",
]
