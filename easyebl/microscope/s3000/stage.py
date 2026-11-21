"""Five-axis stage motor-drive protocol (manual section 13 / source chapter 8).

Commands are ``#{u}{a}{b}{c} {params}<CR>`` with hexadecimal parameters.
Replies are compact (``0E0``, ``0P17077D``, ``000``...) and do not echo the
request.

Observed on the S-3000 (2026-11-15). The stage lines travel over the SEM
text port, and the SEM relays them only after ``#STAGE ON``. Until then it
answers each one in SEM grammar (``#00R1 E2``), which raises
:class:`~easyebl.microscope.s3000.errors.StageRejectedBySem`. R1 replies carry no
direction sign and no ``uE0``. R0 and R4 *are* followed by ``uE0``; R3 and R5
are not. R3 has no field separators. Command *acceptance* (``uE0``) is not motion *completion*
(``ua0``); limit, touch and motor-error events can arrive at any time.

:class:`StageClient` keeps those concerns apart:

* one receive handler classifies every stage line;
* acknowledgements and query responses go to the single outstanding transaction;
* motion events update a per-axis :class:`MotionState` and are fanned out to listeners;
* :meth:`StageClient.halt` can always be sent, even during a long wait.
"""

from __future__ import annotations

import enum
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Union

from .channel import LineChannel
from .errors import (
    ConfigurationError,
    ConnectionUncertain,
    MotionError,
    StageCommandRejected,
    StageRejectedBySem,
    ValidationError,
)

# --------------------------------------------------------------------------- axes & scales


class Axis(enum.Enum):
    """Stage axis with its controller number ``u`` and motor selector ``a``."""

    X = (0, 0)
    Y = (0, 1)
    R = (1, 0)
    Z = (2, 0)
    T = (2, 1)

    @property
    def controller(self) -> int:
        return self.value[0]

    @property
    def motor(self) -> int:
        return self.value[1]

    @property
    def address(self) -> str:
        return f"{self.controller}{self.motor}"

    @classmethod
    def from_address(cls, controller: int, motor: int) -> "Axis":
        for ax in cls:
            if ax.value == (controller, motor):
                return ax
        raise ValidationError(f"No axis at controller {controller}, motor {motor}")

    @classmethod
    def parse(cls, name: "Axis | str") -> "Axis":
        if isinstance(name, Axis):
            return name
        try:
            return cls[str(name).strip().upper()]
        except KeyError as exc:
            raise ValidationError(f"Unknown axis {name!r}; use X, Y, R, Z or T") from exc


#: Motors driven by each controller; C0 (simultaneous start) acts on all of them.
CONTROLLER_AXES: dict[int, tuple[Axis, ...]] = {0: (Axis.X, Axis.Y), 1: (Axis.R,), 2: (Axis.Z, Axis.T)}

#: Pulse scales per axis as (pulses per unit, unit).
#:
#: The printed values are correct, but the manual swaps the T and Z units: it prints
#: "T 3528 pulses/µm" and "Z 22.32 pulses/deg". The real scales are T 3528 pulses/deg and
#: Z 22.32 pulses/µm (confirmed by the operator, 2026-11-15). The instrument agrees: Z read
#: 334827 pulses = 15.0 mm while the SEM reported WD 15.
PULSE_SCALES: dict[Axis, tuple[float, str]] = {
    Axis.X: (25.6, "µm"),
    Axis.Y: (25.6, "µm"),
    Axis.R: (1849.0, "deg"),
    Axis.Z: (22.32, "µm"),
    Axis.T: (3528.0, "deg"),
}

XY_PULSES_PER_UM = 25.6
R_PULSES_PER_DEG = 1849
Z_PULSES_PER_UM = 22.32
T_PULSES_PER_DEG = 3528


def axis_unit(axis) -> str:
    """Physical unit of an axis: ``"µm"`` for X/Y/Z, ``"deg"`` for R/T."""
    return PULSE_SCALES[Axis.parse(axis)][1]


def to_pulses(axis, value: float) -> int:
    """Convert a physical value (µm for X/Y/Z, degrees for R/T) to pulses.

    Rounds half away from zero.
    """
    p = value * PULSE_SCALES[Axis.parse(axis)][0]
    return int(p + 0.5) if p >= 0 else -int(-p + 0.5)


def from_pulses(axis, pulses: int) -> float:
    """Convert pulses to the axis's physical unit (µm for X/Y/Z, degrees for R/T)."""
    return pulses / PULSE_SCALES[Axis.parse(axis)][0]


def xy_um_to_pulses(um: float) -> int:
    """Convert X/Y micrometres to pulses (25.6 pulses/µm), rounding half away from zero."""
    return to_pulses(Axis.X, um)


def xy_pulses_to_um(pulses: int) -> float:
    return from_pulses(Axis.X, pulses)


def rotary_enhance_upper_bound() -> int:
    """I2 value for O2 enhance mode on R: pulses for 360° minus one (1849 pulses/deg → 0x0A2827)."""
    return R_PULSES_PER_DEG * 360 - 1


# --------------------------------------------------------------------------- hex encoding

_HEX6 = re.compile(r"^[0-9A-F]{6}$")


def hex_field(value: int, width: int) -> str:
    """Unsigned, zero-padded uppercase hex of exactly ``width`` characters."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Expected an integer, got {value!r}")
    if not 0 <= value < 16**width:
        raise ValidationError(f"{value} does not fit in {width} hex characters")
    return f"{value:0{width}X}"


class CoordinateCodec:
    """Encode/decode the six-character coordinate fields.

    The manual does **not** define how negative coordinates are encoded (the
    soft-limit bounds 3FFFFF / C00000 only *suggest* a signed form). By default
    (``bits=None``) only non-negative values 0..0xFFFFFF are accepted and
    decoding returns the unsigned value. Set ``bits=24`` (or ``22``) after
    confirming the encoding on your controller to enable two's-complement.
    """

    def __init__(self, bits: int | None = None):
        if bits not in (None, 22, 24):
            raise ConfigurationError("bits must be None, 22 or 24")
        self.bits = bits

    def encode(self, value: "int | str") -> str:
        if isinstance(value, str):
            s = value.strip().upper()
            if not _HEX6.match(s):
                raise ValidationError(f"Raw coordinate must be exactly six hex characters, got {value!r}")
            return s
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"Coordinate must be an int or six-character hex string, got {value!r}")
        if value >= 0:
            limit = 0xFFFFFF if self.bits is None else (1 << (self.bits - 1)) - 1
            if value > limit:
                raise ValidationError(f"Coordinate {value} exceeds {limit:#X}")
            return f"{value:06X}"
        if self.bits is None:
            raise ConfigurationError(
                "Negative coordinates need a confirmed signed encoding: create CoordinateCodec(bits=24) "
                "(or 22) after verifying, or pass the raw six-character hex"
            )
        lo = -(1 << (self.bits - 1))
        if value < lo:
            raise ValidationError(f"Coordinate {value} below {lo}")
        return f"{value & 0xFFFFFF:06X}"

    def decode(self, field: str) -> int:
        n = int(field, 16)
        if self.bits is None:
            return n
        n &= (1 << self.bits) - 1
        if n & (1 << (self.bits - 1)):
            n -= 1 << self.bits
        return n


def coefficient_field(alpha: "int | str") -> str:
    """Encode one raster-rotation coefficient as two hex characters.

    The source prints ``80H < α < 7FH (two's complement)`` with *strict*
    inequalities, so -127..126 are accepted. A raw two-character hex string is
    checked against the same bounds.
    """
    if isinstance(alpha, str):
        s = alpha.strip().upper()
        if not re.fullmatch(r"[0-9A-F]{2}", s):
            raise ValidationError(f"Coefficient must be two hex characters, got {alpha!r}")
        v = int(s, 16)
        v = v - 256 if v >= 128 else v
    elif isinstance(alpha, int) and not isinstance(alpha, bool):
        v = alpha
    else:
        raise ValidationError(f"Coefficient must be an int or two hex characters, got {alpha!r}")
    if not -128 < v < 127:
        raise ValidationError(f"Coefficient {v} violates 80H < α < 7FH (strict, as printed)")
    return f"{v & 0xFF:02X}"


# --------------------------------------------------------------------------- command builders


def _hdr(controller: int, motor: int, op: str) -> str:
    if controller not in CONTROLLER_AXES or motor not in (0, 1):
        raise ValidationError(f"Invalid controller/motor {controller}/{motor}")
    return f"#{controller}{motor}{op}"


def _ax(axis) -> Axis:
    return Axis.parse(axis)


def cmd_trapezoid(axis, max_speed: int, accel_time_10ms: int) -> str:
    """Item 1, I1: maximum speed (pulses/s) and acceleration time (10 ms units), 0..0xFFFF each."""
    a = _ax(axis)
    return f"{_hdr(a.controller, a.motor, 'I1')} 00{hex_field(max_speed, 4)} 00{hex_field(accel_time_10ms, 4)}"


def cmd_cw_soft_limit(axis, coordinate: str) -> str:
    """Item 2, I2 (raw six-hex coordinate)."""
    a = _ax(axis)
    return f"{_hdr(a.controller, a.motor, 'I2')} {CoordinateCodec().encode(coordinate)}"


def cmd_ccw_soft_limit(axis, coordinate: str) -> str:
    """Item 3, I3 (raw six-hex coordinate)."""
    a = _ax(axis)
    return f"{_hdr(a.controller, a.motor, 'I3')} {CoordinateCodec().encode(coordinate)}"


def cmd_target(axis, coordinate: str) -> str:
    """Item 4, M0: target designation (start with C0)."""
    a = _ax(axis)
    return f"{_hdr(a.controller, a.motor, 'M0')} {CoordinateCodec().encode(coordinate)}"


def cmd_constant_speed(axis, speed: int, direction: str) -> str:
    """Item 5, M3: constant-speed movement; ``direction`` '+' (CW) or '-' (CCW)."""
    a = _ax(axis)
    if direction not in ("+", "-"):
        raise ValidationError("Direction must be '+' (CW) or '-' (CCW)")
    return f"{_hdr(a.controller, a.motor, 'M3')} {hex_field(speed, 6)} {direction}"


def cmd_halt(axis) -> str:
    """Item 6, S0: intermediate halt."""
    a = _ax(axis)
    return _hdr(a.controller, a.motor, "S0")


def cmd_motor_free(controller: int) -> str:
    """Item 7, S2 (fixed motor field 0)."""
    return _hdr(controller, 0, "S2")


def cmd_simultaneous_start(controller: int) -> str:
    """Item 8, C0: start both axes of one controller after M0."""
    return _hdr(controller, 0, "C0")


def cmd_set_coordinate(axis, coordinate: str) -> str:
    """Item 9, C2: assign the current coordinate (not a move)."""
    a = _ax(axis)
    return f"{_hdr(a.controller, a.motor, 'C2')} {CoordinateCodec().encode(coordinate)}"


def cmd_read_limits(axis) -> str:
    a = _ax(axis)
    return _hdr(a.controller, a.motor, "R0")


def cmd_read_position(axis) -> str:
    a = _ax(axis)
    return _hdr(a.controller, a.motor, "R1")


def cmd_read_touch(axis) -> str:
    a = _ax(axis)
    return _hdr(a.controller, a.motor, "R4")


def cmd_raster_link(controller: int, a11, a12, a21, a22) -> str:
    """Item 13, A0: raster-rotation linkage coefficients."""
    f = coefficient_field
    return f"{_hdr(controller, 0, 'A0')} 00{f(a11)}{f(a12)} 00{f(a21)}{f(a22)}"


TRACKBALL_MASKS = {"FF": "data unchanged", "FE": "ignore low bit (1/2)", "FC": "ignore low two bits (1/4)"}


def cmd_trackball_sensitivity(axis, mask: str, gain: int) -> str:
    """Item 14, I0: ``{mm}00{gg}``; mask FF/FE/FC (documented), gain 0..255."""
    a = _ax(axis)
    m = str(mask).strip().upper()
    if m not in TRACKBALL_MASKS:
        raise ValidationError(f"Mask must be one of {', '.join(TRACKBALL_MASKS)} (documented values)")
    return f"{_hdr(a.controller, a.motor, 'I0')} {m}00{hex_field(gain, 2)}"


def cmd_trackball_speed(axis, factor: int) -> str:
    """Item 15, O1: 0 slow, 1 medium, 2 fast."""
    a = _ax(axis)
    if factor not in (0, 1, 2):
        raise ValidationError("Trackball speed factor must be 0, 1 or 2")
    return f"{_hdr(a.controller, a.motor, 'O1')} {factor}"


def cmd_read_both(controller: int) -> str:
    """Item 16, R3: both positions and trackball speed."""
    return _hdr(controller, 0, "R3")


def cmd_read_trackball_speed(controller: int) -> str:
    """Item 17, R5."""
    return _hdr(controller, 0, "R5")


def cmd_limit_mode(axis, mode: int) -> str:
    """Item 18, O2: 0 normal (default), 1 enhance (endless rotary)."""
    a = _ax(axis)
    if mode not in (0, 1):
        raise ValidationError("Limit mode must be 0 (normal) or 1 (enhance)")
    return f"{_hdr(a.controller, a.motor, 'O2')} {mode}"


# --------------------------------------------------------------------------- messages


@dataclass(frozen=True)
class Ack:
    controller: int
    code: int  # 0 accepted, 1 command error, 2 communication error
    raw: str


@dataclass(frozen=True)
class PositionReply:
    controller: int
    coordinate: str  # six hex characters, as received
    raw: str


@dataclass(frozen=True)
class BothPositionsReply:
    controller: int
    axis0: str
    axis1: str
    speed: int
    raw: str


@dataclass(frozen=True)
class TrackballSpeedReply:
    controller: int
    speed: int
    raw: str


@dataclass(frozen=True)
class LimitSensors:
    controller: int
    motor: int
    cw_on: bool
    ccw_on: bool
    raw: str


@dataclass(frozen=True)
class TouchSensor:
    controller: int
    motor: int
    on: bool
    raw: str


AXIS_EVENT_MEANINGS = {
    0: "Normal halt after MOVE",
    1: "Beyond CW limit",
    2: "Beyond CCW limit",
    4: "Halted by S0 during movement",
    6: "Within CW limit",
    7: "Within CCW limit",
    8: "Overcurrent protection activated",
    9: "Motor error (no encoder return pulse)",
}


@dataclass(frozen=True)
class AxisEvent:
    controller: int
    motor: int
    code: int
    raw: str

    @property
    def meaning(self) -> str:
        return AXIS_EVENT_MEANINGS.get(self.code, "Undocumented event code")


@dataclass(frozen=True)
class UnknownData:
    raw: str


StageMessage = Union[
    Ack, PositionReply, BothPositionsReply, TrackballSpeedReply, LimitSensors, TouchSensor, AxisEvent, UnknownData
]

_RE_ACK = re.compile(r"^([0-2])E([0-2])$")
# Observed: six hex characters, no direction sign (the manual's +/- is not sent).
_RE_POS = re.compile(r"^([0-2])P([0-9A-Fa-f]{6})$")
# R3 separators are not established in the source: accept none or whitespace between fields.
_RE_BOTH = re.compile(r"^([0-2])Q\s*([0-9A-Fa-f]{6})\s*([0-9A-Fa-f]{6})\s*([0-2])$")
_RE_TB = re.compile(r"^([0-2])S([0-2])$")
_RE_LIMIT = re.compile(r"^([0-2])([01])([ABCD])$")
_RE_TOUCH = re.compile(r"^([0-2])([01])([KL])$")
_RE_EVENT = re.compile(r"^([0-2])([01])([0-9])$")

#: A stage command answered by the SEM itself: ``<echo of #uabc ...> <result>``.
_RE_SEM_REJECT = re.compile(r"^(#?[0-2][01][A-Z][0-9](?: .*)?) ([GE][0-9])$")

#: Queries the S-3000 firmware follows with ``uE0`` (observed 2026-11-15). The manual says
#: R1 only; the observed behaviour is R0 and R4 but not R1, R3 or R5.
OBSERVED_E0_AFTER_QUERIES = frozenset({"R0", "R4"})

_LIMIT_BITS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


def parse_stage_message(text: str) -> StageMessage | None:
    """Classify one CR-stripped line, or return ``None`` if it is not stage grammar."""
    if text.startswith("SYS00-"):
        return UnknownData(text)
    if m := _RE_ACK.match(text):
        return Ack(int(m[1]), int(m[2]), text)
    if m := _RE_POS.match(text):
        return PositionReply(int(m[1]), m[2].upper(), text)
    if m := _RE_BOTH.match(text):
        return BothPositionsReply(int(m[1]), m[2].upper(), m[3].upper(), int(m[4]), text)
    if m := _RE_TB.match(text):
        return TrackballSpeedReply(int(m[1]), int(m[2]), text)
    if m := _RE_LIMIT.match(text):
        cw, ccw = _LIMIT_BITS[m[3]]
        return LimitSensors(int(m[1]), int(m[2]), cw, ccw, text)
    if m := _RE_TOUCH.match(text):
        return TouchSensor(int(m[1]), int(m[2]), m[3] == "K", text)
    if m := _RE_EVENT.match(text):
        return AxisEvent(int(m[1]), int(m[2]), int(m[3]), text)
    return None


# --------------------------------------------------------------------------- client


class MotionState(enum.Enum):
    IDLE = "idle"
    MOVING = "moving"
    COMPLETED = "completed"  # ua0 received
    HALTED = "halted"  # ua4 received (S0)
    LIMIT = "limit"  # ua1 / ua2 received
    FAULT = "fault"  # ua8 / ua9 received
    UNCERTAIN = "uncertain"  # timed out waiting; actual state unknown


_TERMINAL_EVENT_STATE = {
    0: MotionState.COMPLETED,
    4: MotionState.HALTED,
    1: MotionState.LIMIT,
    2: MotionState.LIMIT,
    8: MotionState.FAULT,
    9: MotionState.FAULT,
}

_RESPONSE_TYPES = {
    "R0": LimitSensors,
    "R1": PositionReply,
    "R3": BothPositionsReply,
    "R4": TouchSensor,
    "R5": TrackballSpeedReply,
}


@dataclass
class PositionReading:
    axis: Axis
    coordinate: str
    value: int  # pulses, decoded with the client's CoordinateCodec
    raw: str

    @property
    def unit(self) -> str:
        """``"µm"`` (X/Y/Z) or ``"deg"`` (R/T)."""
        return axis_unit(self.axis)

    @property
    def physical(self) -> float:
        """Position in :attr:`unit` (µm or degrees)."""
        return from_pulses(self.axis, self.value)


class _Pending:
    def __init__(self, command: str, controller: int, motor: int, kind: str | None, want_ack: bool):
        self.command = command
        self.controller = controller
        self.motor = motor
        self.kind = kind  # query op code (R0..R5) or None for a setting command
        self.want_ack = want_ack
        self.response: StageMessage | None = None
        self.ack: Ack | None = None
        self.sem_reject: tuple[str, str] | None = None  # (result, raw) if the SEM answered instead
        self.stray_acks: list[Ack] = []


class StageClient:
    """Event-aware client for the stage controllers.

    Parameters
    ----------
    ack_timeout, query_timeout:
        Seconds to wait for an acknowledgement / query response. Not specified
        by the manual; validate on hardware.
    extra_ack_window:
        After a query that is *not* followed by ``uE0`` (R1/R3/R5 on the
        S-3000) the client keeps listening this long and logs any stray
        ``uE0`` as a protocol observation instead of attributing it to the next request.
    expected_ack_timeout:
        How long to wait for the ``uE0`` that follows R0/R4. It usually arrives
        with the reply, but on the instrument it was seen more than 0.3 s later.
        Its absence is logged, not fatal.
    e0_after_queries:
        Query op codes that are followed by ``uE0``. Defaults to the observed
        S-3000 behaviour (R0, R4).
    """

    def __init__(
        self,
        channel: LineChannel,
        *,
        codec: CoordinateCodec | None = None,
        ack_timeout: float = 2.0,
        query_timeout: float = 2.0,
        extra_ack_window: float = 0.3,
        expected_ack_timeout: float = 1.5,
        halt_lock_timeout: float = 0.2,
        e0_after_queries: Iterable[str] = OBSERVED_E0_AFTER_QUERIES,
    ):
        self.channel = channel
        self.codec = codec or CoordinateCodec()
        self.ack_timeout = ack_timeout
        self.query_timeout = query_timeout
        self.extra_ack_window = extra_ack_window
        self.expected_ack_timeout = expected_ack_timeout
        self.halt_lock_timeout = halt_lock_timeout
        self.e0_after_queries = frozenset(e0_after_queries)
        self._cond = threading.Condition()
        self._pending: _Pending | None = None
        self._motion: dict[Axis, MotionState] = {a: MotionState.IDLE for a in Axis}
        self._last_event: dict[Axis, AxisEvent | None] = {a: None for a in Axis}
        self._listeners: list[Callable[[StageMessage], None]] = []
        self.observations: deque[str] = deque(maxlen=500)
        self.events: deque[tuple[float, StageMessage]] = deque(maxlen=2000)
        self._uncertain: str | None = None
        self._ambiguous_acks: dict[int, int] = {0: 0, 1: 0, 2: 0}
        channel.add_handler(self._on_line, first=True)

    # ------------------------------------------------------------------ public state
    def add_listener(self, fn: Callable[[StageMessage], None]) -> None:
        """Called (on the reader thread) for every asynchronous event and unmatched message."""
        self._listeners.append(fn)

    def motion_state(self, axis) -> MotionState:
        with self._cond:
            return self._motion[_ax(axis)]

    def motion_states(self) -> dict[Axis, MotionState]:
        with self._cond:
            return dict(self._motion)

    def last_event(self, axis) -> AxisEvent | None:
        with self._cond:
            return self._last_event[_ax(axis)]

    @property
    def uncertain(self) -> str | None:
        return self._uncertain

    def resync(self, *, confirm: bool = False) -> None:
        """Clear the uncertain state after checking the stage. Motion states stay as they are."""
        if not confirm:
            raise ConfigurationError("resync(confirm=True) is required")
        with self._cond:
            self._uncertain = None
            self._pending = None
        self.channel.note("Stage link resynchronized by user")

    def _observe(self, text: str) -> None:
        self.observations.append(text)
        self.channel.note("stage observation: " + text)

    # ------------------------------------------------------------------ receive path
    def _on_line(self, text: str, raw: bytes) -> bool:
        if m := _RE_SEM_REJECT.match(text):
            with self._cond:
                p = self._pending
                echo = m[1].upper().lstrip("#")
                if p is not None and p.sem_reject is None and echo == p.command.upper().lstrip("#"):
                    p.sem_reject = (m[2], text)
                    self._cond.notify_all()
                    return True
            return False  # not ours: leave it to the SEM client
        msg = parse_stage_message(text)
        if msg is None:
            return False
        notify = True
        with self._cond:
            p = self._pending
            if isinstance(msg, AxisEvent):
                axis = self._axis_or_none(msg.controller, msg.motor)
                if axis is not None:
                    self._last_event[axis] = msg
                    new = _TERMINAL_EVENT_STATE.get(msg.code)
                    if new is not None and (
                        self._motion[axis] in (MotionState.MOVING, MotionState.UNCERTAIN)
                        or new in (MotionState.FAULT, MotionState.LIMIT)
                    ):
                        self._motion[axis] = new
                self.events.append((time.time(), msg))
            elif isinstance(msg, Ack):
                same = p is not None and p.controller == msg.controller
                if same and msg.code != 0 and p.ack is None:
                    p.ack = msg  # an error answers the outstanding request
                    notify = False
                elif same and p.want_ack and p.ack is None and (p.kind is None or p.response is not None):
                    p.ack = msg
                    notify = False
                    if self._ambiguous_acks[msg.controller]:
                        self._ambiguous_acks[msg.controller] -= 1
                        self._observe(f"E0 for {p.command!r} may belong to a halt sent concurrently")
                elif same and not p.want_ack and p.response is not None:
                    p.stray_acks.append(msg)
                    self._observe(f"extra {msg.raw!r} after {p.command!r} (not expected for this query)")
                    notify = False
                elif self._ambiguous_acks[msg.controller]:
                    self._ambiguous_acks[msg.controller] -= 1
                    self.channel.note(f"acknowledgement {msg.raw!r} attributed to concurrent halt")
                    notify = False
                else:
                    self._observe(f"unexpected acknowledgement {msg.raw!r} with no matching request")
                    if msg.code:
                        self._uncertain = f"unsolicited stage error {msg.raw!r}"
            elif isinstance(msg, TouchSensor) and not (
                p is not None
                and p.kind == "R4"
                and p.response is None
                and (p.controller, p.motor) == (msg.controller, msg.motor)
            ):
                self.events.append((time.time(), msg))  # asynchronous touch event
            elif isinstance(msg, UnknownData):
                self.events.append((time.time(), msg))
                self._observe(f"SYS00 unknown-data message {msg.raw!r}")
            elif (
                p is not None
                and p.kind
                and isinstance(msg, _RESPONSE_TYPES[p.kind])
                and p.response is None
                and msg.controller == p.controller
            ):
                p.response = msg
                notify = False
            else:
                self._observe(f"unmatched stage message {msg.raw!r}")
            self._cond.notify_all()
        if notify:
            for fn in list(self._listeners):
                try:
                    fn(msg)
                except Exception:  # pragma: no cover
                    pass
        return True

    @staticmethod
    def _axis_or_none(controller: int, motor: int) -> Axis | None:
        try:
            return Axis.from_address(controller, motor)
        except ValidationError:
            return None

    # ------------------------------------------------------------------ transactions
    def _wait(self, pred, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not pred():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._cond.wait(remaining)
        return True

    def _fail(self, reason: str):
        self._uncertain = reason
        self._pending = None
        self.channel.note("stage link uncertain: " + reason)
        return ConnectionUncertain(reason)

    def command(self, text: str, controller: int, motor: int = 0) -> Ack:
        """Send a setting/motion command and wait for its acknowledgement (acceptance only)."""
        with self.channel.bus_lock:
            return self._command_locked(text, controller, motor)

    def _command_locked(self, text: str, controller: int, motor: int) -> Ack:
        if self._uncertain:
            raise ConnectionUncertain(f"Refusing {text!r}: {self._uncertain}. Call resync(confirm=True).")
        p = _Pending(text, controller, motor, None, True)
        with self._cond:
            self._pending = p
        self.channel.send_line(text)
        with self._cond:
            if not self._wait(lambda: p.ack is not None or p.sem_reject is not None, self.ack_timeout):
                raise self._fail(f"no acknowledgement to {text!r} within {self.ack_timeout:g} s")
            self._pending = None
        if p.sem_reject is not None:
            raise StageRejectedBySem(text, *p.sem_reject)
        assert p.ack is not None
        if p.ack.code != 0:
            raise StageCommandRejected(text, controller, p.ack.code)
        return p.ack

    def query(self, text: str, controller: int, motor: int, kind: str) -> StageMessage:
        """Send R0/R1/R3/R4/R5 and return the response message."""
        expects_e0 = kind in self.e0_after_queries
        with self.channel.bus_lock:
            if self._uncertain:
                raise ConnectionUncertain(f"Refusing {text!r}: {self._uncertain}. Call resync(confirm=True).")
            p = _Pending(text, controller, motor, kind, expects_e0)
            with self._cond:
                self._pending = p
            self.channel.send_line(text)
            with self._cond:
                if not self._wait(
                    lambda: p.response is not None or p.ack is not None or p.sem_reject is not None,
                    self.query_timeout,
                ):
                    raise self._fail(f"no response to {text!r} within {self.query_timeout:g} s")
                if p.sem_reject is not None:
                    self._pending = None
                    raise StageRejectedBySem(text, *p.sem_reject)
                # Listen briefly for the E0 (R1) or a stray E0 (others) so it is not misattributed.
                if expects_e0:
                    if not self._wait(lambda: p.ack is not None, self.expected_ack_timeout):
                        self._observe(f"no E0 after {text!r} (expected for {kind})")
                else:
                    self._wait(lambda: False, self.extra_ack_window)
                self._pending = None
        if p.ack is not None and p.ack.code != 0:
            raise StageCommandRejected(text, controller, p.ack.code)
        if p.response is None:  # pragma: no cover - guarded above
            raise self._fail(f"{text!r} acknowledged but returned no data")
        return p.response

    def raw(self, text: str) -> StageMessage:
        """Send an arbitrary ``#uabc ...`` line (commissioning). Queries return their response,
        other commands their acknowledgement. Motion state is not tracked for raw moves."""
        m = re.match(r"^#([0-2])([01])([A-Z][0-9])(?: |$)", text)
        if not m:
            raise ValidationError(f"Not a stage command: {text!r}")
        u, a, op = int(m[1]), int(m[2]), m[3]
        if op in _RESPONSE_TYPES:
            return self.query(text, u, a, op)
        return self.command(text, u, a)

    # ------------------------------------------------------------------ reads
    def read_position(self, axis) -> PositionReading:
        a = _ax(axis)
        msg = self.query(cmd_read_position(a), a.controller, a.motor, "R1")
        assert isinstance(msg, PositionReply)
        return PositionReading(a, msg.coordinate, self.codec.decode(msg.coordinate), msg.raw)

    def read_limits(self, axis) -> LimitSensors:
        a = _ax(axis)
        msg = self.query(cmd_read_limits(a), a.controller, a.motor, "R0")
        assert isinstance(msg, LimitSensors)
        return msg

    def read_touch(self, axis) -> TouchSensor:
        a = _ax(axis)
        msg = self.query(cmd_read_touch(a), a.controller, a.motor, "R4")
        assert isinstance(msg, TouchSensor)
        return msg

    def read_both(self, controller: int) -> BothPositionsReply:
        msg = self.query(cmd_read_both(controller), controller, 0, "R3")
        assert isinstance(msg, BothPositionsReply)
        return msg

    def read_trackball_speed(self, controller: int) -> int:
        msg = self.query(cmd_read_trackball_speed(controller), controller, 0, "R5")
        assert isinstance(msg, TrackballSpeedReply)
        return msg.speed

    # ------------------------------------------------------------------ motion
    def halt(self, axis) -> None:
        """Send S0 for one axis. Never blocked for long by another transaction.

        If the bus is busy (e.g. a long SEM photograph on a shared port) S0 is
        sent anyway; its acknowledgement is then logged as possibly ambiguous.
        """
        a = _ax(axis)
        text = cmd_halt(a)
        if self.channel.bus_lock.acquire(timeout=self.halt_lock_timeout):
            try:
                if not self._uncertain:
                    self._command_locked(text, a.controller, a.motor)
                    return
                # Link uncertain: still send the halt, but do not wait for an attributable ack.
                with self._cond:
                    self._ambiguous_acks[a.controller] += 1
                self.channel.send_line(text, note="halt sent while stage link uncertain")
                return
            finally:
                self.channel.bus_lock.release()
        with self._cond:
            self._ambiguous_acks[a.controller] += 1
        self.channel.send_line(text, note="emergency halt sent while bus busy")

    def halt_all(self) -> list[str]:
        """Send S0 to every axis; returns a list of problems (empty if all were accepted)."""
        problems = []
        for a in Axis:
            try:
                self.halt(a)
            except Exception as exc:  # keep going: every axis must get its S0
                problems.append(f"{a.name}: {exc}")
        return problems

    def move(
        self,
        targets: dict,
        *,
        hold_partners: bool = False,
        wait: bool = True,
        timeout: float = 120.0,
    ) -> dict[Axis, MotionState]:
        """Absolute move: M0 for every axis on each involved controller, then C0 per controller.

        C0 starts *both* motors of a controller. Every motor on an involved
        controller therefore needs a target: give it explicitly, or pass
        ``hold_partners=True`` to read the partner's current position (R1) and
        re-target it there. Targets are ints (encoded by the client's codec) or
        raw six-character hex strings.

        With ``wait=True`` raises :class:`MotionError` unless every axis reports
        normal completion (``ua0``).
        """
        if not targets:
            raise ValidationError("No targets given")
        resolved: dict[Axis, str] = {_ax(k): self.codec.encode(v) for k, v in targets.items()}
        controllers = sorted({a.controller for a in resolved})
        for u in controllers:
            for partner in CONTROLLER_AXES[u]:
                if partner in resolved:
                    continue
                if not hold_partners:
                    raise ValidationError(
                        f"C0 starts every motor on controller {u}; give a target for {partner.name} "
                        "or use hold_partners=True"
                    )
                resolved[partner] = self.read_position(partner).coordinate
        with self._cond:
            busy = [a.name for a in resolved if self._motion[a] == MotionState.MOVING]
        if busy:
            raise MotionError(f"Axes already moving: {', '.join(busy)}")
        for a, coord in resolved.items():
            self.command(cmd_target(a, coord), a.controller, a.motor)
        started: list[Axis] = []
        for u in controllers:
            axes = CONTROLLER_AXES[u]
            with self._cond:
                for a in axes:
                    self._motion[a] = MotionState.MOVING
            try:
                self.command(cmd_simultaneous_start(u), u, 0)
            except Exception:
                with self._cond:
                    for a in axes:
                        self._motion[a] = MotionState.UNCERTAIN
                raise
            started.extend(axes)
        if not wait:
            return self.motion_states()
        states = self.wait_for_motion(started, timeout=timeout)
        bad = {a: s for a, s in states.items() if s != MotionState.COMPLETED}
        if bad:
            raise MotionError(
                "Move did not complete normally: " + ", ".join(f"{a.name}={s.value}" for a, s in bad.items()), states
            )
        return states

    def move_constant_speed(self, axis, speed: int, direction: str) -> None:
        """M3: run continuously until :meth:`halt`. Speed in pulses/s."""
        a = _ax(axis)
        with self._cond:
            self._motion[a] = MotionState.MOVING
        try:
            self.command(cmd_constant_speed(a, speed, direction), a.controller, a.motor)
        except Exception:
            with self._cond:
                self._motion[a] = MotionState.UNCERTAIN
            raise

    def wait_for_motion(self, axes: Iterable, timeout: float) -> dict[Axis, MotionState]:
        """Wait until none of ``axes`` is MOVING. Timed-out axes become UNCERTAIN.

        Does not hold the bus lock, so :meth:`halt` remains available.
        """
        ax = [_ax(a) for a in axes]
        with self._cond:
            done = self._wait(lambda: all(self._motion[a] != MotionState.MOVING for a in ax), timeout)
            if not done:
                for a in ax:
                    if self._motion[a] == MotionState.MOVING:
                        self._motion[a] = MotionState.UNCERTAIN
                self.channel.note("motion wait timed out; state uncertain (no command resent)")
            return {a: self._motion[a] for a in ax}

    def acknowledge_motion(self, axis) -> None:
        """Return a finished/uncertain axis to IDLE after the operator has checked it."""
        a = _ax(axis)
        with self._cond:
            if self._motion[a] == MotionState.MOVING:
                raise MotionError(f"{a.name} is still moving")
            self._motion[a] = MotionState.IDLE

    # ------------------------------------------------------------------ configuration (guarded)
    def set_profile(self, axis, max_speed: int, accel_time_10ms: int) -> Ack:
        a = _ax(axis)
        return self.command(cmd_trapezoid(a, max_speed, accel_time_10ms), a.controller, a.motor)

    def set_soft_limits(self, axis, cw: "int | str", ccw: "int | str", *, confirm_overwrite: bool = False) -> None:
        """I2/I3. There is no readback for soft limits, so overwriting them needs explicit confirmation."""
        if not confirm_overwrite:
            raise ConfigurationError("Soft limits cannot be read back; pass confirm_overwrite=True deliberately")
        a = _ax(axis)
        self.command(cmd_cw_soft_limit(a, self.codec.encode(cw)), a.controller, a.motor)
        self.command(cmd_ccw_soft_limit(a, self.codec.encode(ccw)), a.controller, a.motor)

    def set_coordinate(self, axis, value: "int | str", *, confirm: bool = False) -> Ack:
        """C2: redefine the current coordinate (no motion). Requires ``confirm=True``."""
        if not confirm:
            raise ConfigurationError("C2 changes the coordinate reference; pass confirm=True deliberately")
        a = _ax(axis)
        return self.command(cmd_set_coordinate(a, self.codec.encode(value)), a.controller, a.motor)

    def motor_free(self, controller: int, *, confirm: bool = False) -> Ack:
        if not confirm:
            raise ConfigurationError("S2 releases the motors for manual rotation; pass confirm=True deliberately")
        return self.command(cmd_motor_free(controller), controller, 0)

    def set_raster_link(self, controller: int, a11, a12, a21, a22) -> Ack:
        return self.command(cmd_raster_link(controller, a11, a12, a21, a22), controller, 0)

    def set_trackball_sensitivity(self, axis, mask: str, gain: int) -> Ack:
        a = _ax(axis)
        return self.command(cmd_trackball_sensitivity(a, mask, gain), a.controller, a.motor)

    def set_trackball_speed(self, axis, factor: int) -> Ack:
        a = _ax(axis)
        return self.command(cmd_trackball_speed(a, factor), a.controller, a.motor)

    def set_limit_mode(self, axis, mode: int) -> Ack:
        a = _ax(axis)
        return self.command(cmd_limit_mode(a, mode), a.controller, a.motor)

    def configure_rotary_enhance(self, *, confirm: bool = False) -> None:
        """Put R in endless mode: I2 = 360° − 1 pulse (printed scale), I3 = 000000, O2 1.

        Uses 1849 pulses/deg; confirm that continuous rotation is permitted first.
        """
        if not confirm:
            raise ConfigurationError("Rotary enhance mode overwrites R soft limits; pass confirm=True deliberately")
        upper = hex_field(rotary_enhance_upper_bound(), 6)
        self.command(cmd_cw_soft_limit(Axis.R, upper), 1, 0)
        self.command(cmd_ccw_soft_limit(Axis.R, "000000"), 1, 0)
        self.command(cmd_limit_mode(Axis.R, 1), 1, 0)
