"""SEM text-protocol client: one request at a time, echo + result-code checking.

Protocol summary (manual section 2)::

    setting/execution  ->  <request> <result><CR>
    data read          ->  <request> <data> <result><CR>

The SEM replies after execution completes. This client therefore holds the
channel's bus lock from transmission until the complete reply arrives, and
refuses further commands after a timeout or an unmatched reply until
:meth:`SemClient.resync` is called deliberately.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import commands as C
from .channel import LineChannel, validate_line_text
from .errors import (
    CommandFailed,
    ConfigurationError,
    ConnectionUncertain,
    ProtocolError,
    ValidationError,
)

#: SEM result codes, complete Table 6-1 (manual section 2).
RESULT_CODES: dict[str, str] = {
    "G0": "Normal end",
    "E1": "Abnormal end",
    "E2": "Unexecuted (unexecutable)",
    "E6": "S3 motor drive stage unconnected",
    "E7": "Data reading error",
    "E8": "Undefined command received",
    "E9": "Unacceptable set value 1 or set value 2 received",
}

_RESULT_RE = re.compile(r"^[GE][0-9]$")
_STAGE_LIKE_RE = re.compile(r"^(?:[0-9]|SYS00-)")


@dataclass(frozen=True)
class SemReply:
    """A parsed SEM reply."""

    request: str
    raw: str
    result: str
    data: str | None = None

    @property
    def ok(self) -> bool:
        return self.result == "G0"

    @property
    def result_meaning(self) -> str:
        return RESULT_CODES.get(self.result, "Undocumented result code")


def echo_variants(request: str) -> tuple[str, ...]:
    """Forms in which the SEM may echo ``request``.

    Observed on the S-3000 (2026-11-15): the echo is upper-cased
    (``MAG 10k`` -> ``MAG 10K G0``) and a leading ``#`` is dropped
    (``#STAGE ON`` -> ``STAGE ON G0``). The verbatim form is accepted too.
    """
    out = [request, request.upper()]
    if request.startswith("#"):
        out += [request[1:], request[1:].upper()]
    return tuple(dict.fromkeys(out))


def parse_reply(request: str, line: str, *, is_read: bool) -> SemReply:
    """Parse one CR-stripped reply line against the request that produced it.

    Raises :class:`ProtocolError` if the echo does not match, the result field
    is missing or malformed, or a successful read carries no data. Read data
    is returned as sent (``R MAG`` returns ``9k`` with a lower-case k even
    though the echo is upper-cased).
    """
    for echo in echo_variants(request):
        if line.startswith(echo + " "):
            break
    else:
        raise ProtocolError(f"Reply does not echo request {request!r}", raw=line)
    remainder = line[len(echo) + 1 :]
    if not remainder:
        raise ProtocolError("Reply has no result field", raw=line)
    if " " in remainder:
        data, result = remainder.rsplit(" ", 1)
    else:
        data, result = None, remainder
    if not _RESULT_RE.match(result):
        raise ProtocolError(f"Malformed result field {result!r}", raw=line)
    if is_read and result == "G0" and not data:
        raise ProtocolError("Successful read reply carries no data field", raw=line)
    if not is_read and data is not None and result == "G0":
        raise ProtocolError(f"Unexpected data {data!r} in a setting/execution reply", raw=line)
    return SemReply(request=request, raw=line, result=result, data=data)


def as_int(data: str | None) -> int | None:
    """Return ``int(data)`` if the read data is a plain integer, else ``None`` (e.g. ``OFF``)."""
    if data is not None and re.fullmatch(r"-?\d+", data.strip()):
        return int(data)
    return None


@dataclass
class Timeouts:
    """Per-operation reply timeouts in seconds.

    The manual gives **no** response-time limits; these defaults are
    placeholders to be validated on the instrument. Photographing uses the
    configured exposure time plus ``photo_margin``.
    """

    default: float = 5.0
    hv: float = 60.0
    automatic: float = 180.0  # ABC/AFC/ASF/AFS/AGA
    startup: float = 600.0  # AUTO1/AUTO2
    vacuum: float = 600.0
    photo_margin: float = 120.0
    by_command: dict[str, float] = field(default_factory=dict)

    def for_request(self, request: str, photo_seconds: int | None) -> float:
        head = request.split(" ", 1)[0]
        if head in self.by_command:
            return self.by_command[head]
        if head == "#PHOTO":
            return (photo_seconds or max(C.PHOTO_SPEEDS)) + self.photo_margin
        if head in ("ABC", "AFC", "ASF", "AFS", "AGA"):
            return self.automatic
        if head in ("AUTO1", "AUTO2"):
            return self.startup
        if head == "HV" and request in ("HV ON", "HV OFF"):
            return self.hv
        if head == "VAC":
            return self.vacuum
        return self.default


class SemClient:
    """High-level client for the SEM text interface."""

    def __init__(
        self,
        channel: LineChannel,
        *,
        timeouts: Timeouts | None = None,
        low_vacuum_capable: bool = False,
    ):
        self.channel = channel
        self.timeouts = timeouts or Timeouts()
        self.low_vacuum_capable = low_vacuum_capable
        self._state = threading.Lock()
        self._pending: str | None = None
        self._reply: str | None = None
        self._event = threading.Event()
        self._uncertain: str | None = None
        self._late: list[tuple[str | None, str]] = []
        self._photo_seconds: int | None = None
        self.history: list[tuple[float, str, str]] = []  # (time, request, raw reply)
        self.on_uncertain: Callable[[str], None] | None = None
        channel.add_handler(self._on_line)

    # ------------------------------------------------------------------ line handling
    def _on_line(self, text: str, raw: bytes) -> bool:
        if _STAGE_LIKE_RE.match(text):
            return False  # leave stage-grammar lines to the stage client
        with self._state:
            if self._pending is not None and self._reply is None and self._uncertain is None:
                self._reply = text
                self._event.set()
                return True
            # Late (after a timeout) or unsolicited: keep it, associate it with any pending request.
            self._late.append((self._pending, text))
        self.channel.note(f"SEM line with no request waiting (late or unsolicited): {text!r}")
        return True

    # ------------------------------------------------------------------ state
    @property
    def uncertain(self) -> str | None:
        """Reason the link is unsynchronized, or ``None`` if it is synchronized."""
        return self._uncertain

    @property
    def late_replies(self) -> list[tuple[str | None, str]]:
        """(request it probably belongs to, raw text) for lines that arrived late or unsolicited."""
        with self._state:
            return list(self._late)

    def _mark_uncertain(self, reason: str) -> None:
        self._uncertain = reason
        self.channel.note("SEM link uncertain: " + reason)
        if self.on_uncertain:
            try:
                self.on_uncertain(reason)
            except Exception:  # pragma: no cover
                pass

    def resync(self, *, confirm: bool = False) -> list[tuple[str | None, str]]:
        """Clear the *uncertain* state after you have checked the instrument.

        Pass ``confirm=True`` to acknowledge that a timed-out command may still
        have executed. Returns (and clears) the late/unsolicited lines captured
        so far so they can be reviewed. Nothing is resent.
        """
        if not confirm:
            raise ConfigurationError("resync(confirm=True) is required: a timeout does not prove a command failed")
        with self._state:
            late = list(self._late)
            self._late.clear()
            self._pending = None
            self._reply = None
            self._event.clear()
            self._uncertain = None
        self.channel.note("SEM link resynchronized by user")
        return late

    # ------------------------------------------------------------------ transaction
    def transact(self, request: str, *, timeout: float | None = None, is_read: bool | None = None) -> SemReply:
        """Send one request and wait for its complete reply.

        Raises :class:`CommandFailed` for any result other than ``G0``,
        :class:`ConnectionUncertain` on timeout or protocol mismatch.
        """
        validate_line_text(request)
        if is_read is None:
            is_read = request.startswith("R ")
        if timeout is None:
            timeout = self.timeouts.for_request(request, self._photo_seconds)
        with self.channel.bus_lock:
            if self._uncertain:
                raise ConnectionUncertain(f"Refusing {request!r}: {self._uncertain}. Call resync(confirm=True).")
            with self._state:
                self._pending = request
                self._reply = None
                self._event.clear()
            self.channel.send_line(request)
            got = self._event.wait(timeout)
            with self._state:
                line = self._reply
                if got:
                    self._pending = None
            if not got or line is None:
                partial = self.channel.partial
                self._mark_uncertain(
                    f"no complete reply to {request!r} within {timeout:g} s"
                    + (f" (partial bytes retained: {partial!r})" if partial else "")
                )
                raise ConnectionUncertain(self._uncertain or "timeout")
            self.history.append((time.time(), request, line))
            try:
                reply = parse_reply(request, line, is_read=is_read)
            except ProtocolError as exc:
                self._mark_uncertain(f"{exc} (raw {line!r})")
                raise ConnectionUncertain(self._uncertain or str(exc)) from exc
        if not reply.ok:
            raise CommandFailed(reply)
        return reply

    def command(self, request: str, **kw) -> SemReply:
        """Send a setting/execution command; raise unless the result is ``G0``."""
        return self.transact(request, is_read=False, **kw)

    def raw(self, request: str, **kw) -> SemReply:
        """Send arbitrary (syntax-checked) text. For commissioning; bypasses value validation."""
        return self.transact(request, **kw)

    # ------------------------------------------------------------------ reads
    def read(self, item: str) -> str:
        """Read one documented item and return its raw data string (e.g. ``"50"`` or ``"OFF"``)."""
        key = C.normalize_read_item(item)
        reply = self.transact("R " + key, is_read=True)
        assert reply.data is not None
        return reply.data

    def read_int(self, item: str) -> int | None:
        return as_int(self.read(item))

    def read_all(self, items=None) -> dict[str, SemReply | Exception]:
        """Read every documented item (or ``items``). Failures are returned, not raised.

        Stops early (re-raising) if the link becomes uncertain.
        """
        out: dict[str, SemReply | Exception] = {}
        keys = list(items) if items else list(C.READ_COMMANDS)
        for key in keys:
            key = C.normalize_read_item(key)
            try:
                out[key] = self.transact("R " + key, is_read=True)
            except CommandFailed as exc:
                out[key] = exc
        return out

    # ------------------------------------------------------------------ scan & display
    def scan(self, mode: str) -> SemReply:
        return self.command(C.scan(mode))

    def monitor(self, mode: str) -> SemReply:
        return self.command(C.monitor(mode))

    def run(self) -> SemReply:
        return self.command(C.run(True))

    def freeze(self) -> SemReply:
        """``RUN OFF`` (FREEZE). This does not switch off HV or blank the beam."""
        return self.command(C.run(False))

    def reduce_speed(self, speed: str) -> SemReply:
        return self.command(C.reduce_speed(speed))

    def split(self, mode) -> SemReply:
        return self.command(C.split(mode))

    def set_rotation(self, degrees: int | None, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.command(C.rotation_enabled(enabled))
        if degrees is not None:
            self.command(C.rotation(degrees))

    def set_dynamic_focus(self, value: int | None, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.command(C.dynamic_focus_enabled(enabled))
        if value is not None:
            self.command(C.dynamic_focus(value))

    def set_tilt_compensation(self, value: int | None, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.command(C.tilt_compensation_enabled(enabled))
        if value is not None:
            self.command(C.tilt_compensation(value))

    def set_contrast(self, n: int) -> SemReply:
        return self.command(C.contrast(n))

    def set_brightness(self, n: int) -> SemReply:
        return self.command(C.brightness(n))

    # ------------------------------------------------------------------ beam & optics
    def hv_on(self) -> SemReply:
        return self.command(C.hv_on())

    def hv_off(self) -> SemReply:
        return self.command(C.hv_off())

    def set_hv(self, kv) -> SemReply:
        """Set the accelerating-voltage setpoint (does not switch HV on)."""
        return self.command(C.hv_setting(kv))

    def set_beam(self, current: int, stage: str, n: int) -> SemReply:
        return self.command(C.beam(current, stage, n))

    def set_focus(self, stage: str, n: int) -> SemReply:
        return self.command(C.focus(stage, n))

    def set_stigmator(self, axis: str, n: int) -> SemReply:
        return self.command(C.stigmator(axis, n))

    def set_magnification(self, m, *, verify: bool = True) -> str | None:
        """Request a listed magnification; returns the ``R MAG`` readback if ``verify``.

        The source warns the SEM may limit magnification, so the readback can
        differ from the request.
        """
        self.command(C.magnification(m))
        return self.read("MAG") if verify else None

    def read_magnification(self) -> int:
        """``R MAG`` as a number (the SEM returns e.g. ``500``, ``9k`` or ``1.2k``)."""
        return C.parse_magnification(self.read("MAG"))

    def abc(self) -> SemReply:
        return self.command(C.abc())

    def afc(self, stage: str) -> SemReply:
        return self.command(C.afc(stage))

    def asf(self) -> SemReply:
        return self.command(C.asf())

    def afs(self, level: str) -> SemReply:
        return self.command(C.afs(level))

    def aga(self) -> SemReply:
        return self.command(C.aga())

    # ------------------------------------------------------------------ detectors & photo
    def set_input(self, destination: str, signal: str) -> SemReply:
        return self.command(C.inputs(destination, signal))

    def set_photo_speed(self, seconds: int) -> SemReply:
        reply = self.command(C.photo_speed(seconds))
        self._photo_seconds = int(seconds)
        return reply

    def set_memory_photo_speed(self, seconds: int) -> SemReply:
        return self.command(C.memory_photo_speed(seconds))

    def photo(self, *, timeout: float | None = None) -> SemReply:
        """``#PHOTO ON``. Waits up to the photographing time plus a margin.

        If the photographing time was not set through this client it is read
        first (``R PSPEED``) so the timeout covers the exposure.
        """
        if self._photo_seconds is None and timeout is None:
            n = as_int(self.read("PSPEED"))
            self._photo_seconds = n if n in C.PHOTO_SPEEDS else None
        return self.command(C.photo(), timeout=timeout)

    # ------------------------------------------------------------------ vacuum, WD, startup
    def _require_low_vacuum(self) -> None:
        if not self.low_vacuum_capable:
            raise ConfigurationError(
                "VAC commands are only valid on low-vacuum-capable models (low_vacuum_capable=False)"
            )

    def vacuum_high(self) -> SemReply:
        self._require_low_vacuum()
        return self.command(C.vacuum_mode("HIGH"))

    def vacuum_low(self) -> SemReply:
        self._require_low_vacuum()
        return self.command(C.vacuum_mode("LOW"))

    def set_vacuum_level(self, level) -> SemReply:
        self._require_low_vacuum()
        return self.command(C.vacuum_level(level))

    def set_working_distance(self, n: int) -> SemReply:
        return self.command(C.working_distance(n))

    def auto_startup(self, routine: int) -> SemReply:
        return self.command(C.auto_startup(routine))

    def stage_control(self, on: bool) -> SemReply:
        return self.command(C.stage_control(on))

    # ------------------------------------------------------------------ analysis
    def set_line_position(self, n: int) -> SemReply:
        return self.command(C.line_position(n))

    def start_line_analysis(self) -> SemReply:
        return self.command(C.line_analysis())

    def line_image(self, on: bool) -> SemReply:
        return self.command(C.line_image(on))

    def set_spot(self, x: int, y: int) -> None:
        # Validate both before sending either.
        cx, cy = C.spot_position("X", x), C.spot_position("Y", y)
        self.command(cx)
        self.command(cy)

    def start_spot_analysis(self) -> SemReply:
        return self.command(C.spot_analysis())

    def read_area(self) -> C.Area:
        vals = {}
        for f, item in (("x", "AREAP X"), ("y", "AREAP Y"), ("size_x", "AREAI X"), ("size_y", "AREAI Y")):
            v = self.read_int(item)
            if v is None:
                raise ProtocolError(f"R {item} did not return an integer")
            vals[f] = v
        return C.Area(**vals)

    def set_area(self, x: int, y: int, size_x: int, size_y: int) -> list[str]:
        """Read the current area, then apply the target in an order that keeps every step valid.

        Returns the commands that were sent.
        """
        target = C.Area(x, y, size_x, size_y)
        if not target.is_valid:
            raise ValidationError("Target area is invalid: " + "; ".join(target.problems()))
        plan = C.plan_area_update(self.read_area(), target)
        for cmd in plan:
            self.command(cmd)
        return plan

    def start_area_analysis(self) -> SemReply:
        return self.command(C.area_analysis())

    def end_analysis(self) -> SemReply:
        return self.command(C.end_analysis())
