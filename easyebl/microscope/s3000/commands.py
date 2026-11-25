"""SEM text-command builders and the documented value tables.

Every function here is pure: it validates its arguments against the ranges
transcribed in the manual (sections 3-9) and returns the command text
*without* the trailing CR. Nothing is sent. Use these with
:class:`easyebl.microscope.s3000.sem.SemClient`, or on their own to check a command.

Ranges are inclusive. Only documented tokens are produced; nothing is
rounded or snapped to a nearest allowed value.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

from .errors import ValidationError

# --------------------------------------------------------------------------- tables

#: Every read request listed in source section 5.2 (manual section 9), with a description.
READ_COMMANDS: dict[str, str] = {
    "RUN": "RUN/FREEZE status",
    "EMI": "Emission current (units and scaling unspecified)",
    "HV": "Accelerating voltage or OFF",
    "BEAM C": "Beam current 1 coarse setting (0-99, raw)",
    "BEAM F": "Beam current 1 fine setting (0-99, raw)",
    "BEAM2 C": "Beam current 2 coarse setting (0-99, raw)",
    "BEAM2 F": "Beam current 2 fine setting (0-99, raw)",
    "FOCUS C": "Coarse focus value",
    "FOCUS F": "Fine focus value",
    "MAG": "Magnification",
    "STG X": "X stigmator",
    "STG Y": "Y stigmator",
    "CONT": "Contrast",
    "BRT": "Brightness",
    "DYNF": "Dynamic focus data",
    "TILTC": "Tilt compensation data",
    "INPUTS AF": "Signal for full/left display",
    "INPUTS AR": "Signal for right display",
    "INPUTS AL": "Signal for line display",
    "SPLIT": "Split mode",
    "ROTATION": "Raster rotation data",
    "PSPEED": "Direct photographing time",
    "MPSPEED": "Memory photographing time",
    "VAC": "Vacuum level (low-vacuum-capable models only)",
    "WD": "Working distance",
    "LINEP": "Line analysis position",
    "SPOTP X": "Spot analysis X position",
    "SPOTP Y": "Spot analysis Y position",
    "AREAP X": "Area analysis X position",
    "AREAP Y": "Area analysis Y position",
    "AREAI X": "Area analysis X size",
    "AREAI Y": "Area analysis Y size",
    # not in the manual's list, but answered by the S-3000 ("R SCAN 1 G0"; checked 2026-11-25)
    "SCAN": "Scan mode (F, S, 1-4, R as for SCAN)",
}

#: Magnifications exactly as printed in the source table (manual section 8).
MAG_PRINTED: tuple[str, ...] = (
    "15",
    "18",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "60",
    "70",
    "80",
    "90",
    "100",
    "120",
    "150",
    "180",
    "200",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "600",
    "700",
    "800",
    "900",
    "1k",
    "1.2k",
    "1.5k",
    "1.8k",
    "2k",
    "2.5k",
    "3k",
    "3.5k",
    "4k",
    "4.5k",
    "5k",
    "6k",
    "7k",
    "8k",
    "9k",
    "10k",
    "12k",
    "15k",
    "18k",
    "20k",
    "25k",
    "30k",
    "35k",
    "40k",
    "45k",
    "50k",
    "60k",
    "70k",
    "80k",
    "90k",
    "100k",
    "120k",
    "150k",
    "180k",
    "200k",
    "250k",
    "300k",
)


#: Numeric magnification -> wire token (k notation from 1000 up, as the S-3000 requires).
MAG_TABLE: dict[int, str] = {(round(float(t[:-1]) * 1000) if t.endswith("k") else int(t)): t for t in MAG_PRINTED}
#: Wire token (as sent and as read back by ``R MAG``) -> numeric magnification.
MAG_VALUES: dict[str, int] = {t: v for v, t in MAG_TABLE.items()}
ALLOWED_MAGNIFICATIONS: tuple[int, ...] = tuple(sorted(MAG_TABLE))

#: Signal display name -> wire token (manual section 5).
SIGNAL_TOKENS: dict[str, str] = {
    "SE": "SE",
    "BSE1": "BSE",
    "BSE2": "ANU",
    "X-RAY1": "X1",
    "X-RAY2": "X2",
    "AUX1": "A1",
    "AE": "AE",
    "EBIC": "EB",
    "CL": "CL",
    "TE": "TE",
    "IP (EP)": "EP",
    "AUX2": "A2",
}
WIRE_SIGNAL_TOKENS = frozenset(SIGNAL_TOKENS.values())
DISPLAY_DESTINATIONS = {"AF": "Full or left", "AR": "Right", "AL": "Line"}

PHOTO_SPEEDS = (40, 80, 200, 400)
MEMORY_PHOTO_SPEEDS = (40, 80)

#: Low-vacuum levels in Pa plus the literal CLOSE (manual section 6).
VACUUM_LEVELS: tuple[str, ...] = (
    "CLOSE",
    "1",
    "3",
    "5",
    "7",
    "10",
    "15",
    "20",
    "25",
    "30",
    "40",
    "50",
    "60",
    "70",
    "80",
    "90",
    "100",
    "110",
    "120",
    "130",
    "140",
    "150",
    "170",
    "200",
    "240",
    "270",
)

SCAN_MODES = {
    "F": "TV scan 1",
    "S": "TV scan 2",
    "1": "Slow scan 1",
    "2": "Slow scan 2",
    "3": "Slow scan 3",
    "4": "Slow scan 4",
    "R": "Reduced-area scan",
}
SPLIT_MODES = ("1", "5", "10", "R", "L", "OFF")

# --------------------------------------------------------------------------- helpers


def _int_in(name: str, value: object, lo: int, hi: int) -> int:
    """Accept an int (or a string of decimal digits) within [lo, hi]."""
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, not a boolean")
    if isinstance(value, int):
        n = value
    elif isinstance(value, str) and value.strip().isdigit():
        n = int(value.strip())
    elif isinstance(value, float) and value.is_integer():
        n = int(value)
    else:
        raise ValidationError(f"{name} must be an integer in {lo}-{hi}, got {value!r}")
    if not lo <= n <= hi:
        raise ValidationError(f"{name} must be in {lo}-{hi}, got {n}")
    return n


def _choice(name: str, value: object, allowed) -> str:
    token = str(value).strip().upper()
    if token not in allowed:
        raise ValidationError(f"{name} must be one of {', '.join(map(str, allowed))}; got {value!r}")
    return token


def _on_off(on: bool) -> str:
    if not isinstance(on, bool):
        raise ValidationError("Expected True (ON) or False (OFF)")
    return "ON" if on else "OFF"


# --------------------------------------------------------------------------- reads


def normalize_read_item(item: str) -> str:
    key = " ".join(str(item).upper().split())
    if key.startswith("R "):
        key = key[2:]
    if key not in READ_COMMANDS:
        raise ValidationError(f"{item!r} is not a documented read request; do not invent queries by prefixing R")
    return key


def read(item: str) -> str:
    """``R <item>``; ``item`` must be one of :data:`READ_COMMANDS`."""
    return "R " + normalize_read_item(item)


# --------------------------------------------------------------------------- scan & display


def scan(mode: str) -> str:
    return "SCAN " + _choice("Scan mode", mode, SCAN_MODES)


def monitor(mode: str) -> str:
    """``MONIT S`` (signal monitor) or ``MONIT W`` (waveform monitor)."""
    return "MONIT " + _choice("Monitor mode", mode, ("S", "W"))


def run(on: bool) -> str:
    """``RUN ON`` (RUN) or ``RUN OFF`` (FREEZE). FREEZE does not blank the beam."""
    return "RUN " + _on_off(on)


def reduce_speed(speed: str) -> str:
    return "REDUCE " + _choice("Reduced-area scan speed", speed, ("F", "M", "S"))


def split(mode: str | int) -> str:
    return "SPLIT " + _choice("Split mode", mode, SPLIT_MODES)


def rotation_enabled(on: bool) -> str:
    return "ROTATION " + _on_off(on)


def rotation(n: int) -> str:
    return f"ROTATION {_int_in('Raster rotation', n, 0, 359)}"


def dynamic_focus_enabled(on: bool) -> str:
    return "DYNF " + _on_off(on)


def dynamic_focus(n: int) -> str:
    return f"DYNF {_int_in('Dynamic focus', n, 0, 255)}"


def tilt_compensation_enabled(on: bool) -> str:
    return "TILTC " + _on_off(on)


def tilt_compensation(n: int) -> str:
    return f"TILTC {_int_in('Tilt compensation', n, 0, 180)}"


def contrast(n: int) -> str:
    return f"CONT {_int_in('Contrast', n, 0, 4095)}"


def brightness(n: int) -> str:
    return f"BRT {_int_in('Brightness', n, 0, 4095)}"


# --------------------------------------------------------------------------- beam & optics


def hv_on() -> str:
    return "HV ON"


def hv_off() -> str:
    return "HV OFF"


def format_hv(kv: float | str | Decimal) -> str:
    """Format an accelerating voltage in kV with the documented step.

    0.30-9.99 kV in 0.01 kV steps (``5.00``); 10.0-30.0 kV in 0.1 kV steps (``15.0``).
    Values that are not exactly on a step are rejected, not rounded.
    """
    try:
        v = Decimal(str(kv).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"Accelerating voltage must be a number of kV, got {kv!r}") from exc
    if Decimal("0.30") <= v <= Decimal("9.99"):
        q = v.quantize(Decimal("0.01"))
        if q != v:
            raise ValidationError(f"{kv} kV is not on the 0.01 kV step for 0.30-9.99 kV")
        return f"{q:.2f}"
    if Decimal("10.0") <= v <= Decimal("30.0"):
        q = v.quantize(Decimal("0.1"))
        if q != v:
            raise ValidationError(f"{kv} kV is not on the 0.1 kV step for 10.0-30.0 kV")
        return f"{q:.1f}"
    raise ValidationError(f"Accelerating voltage must be 0.30-9.99 or 10.0-30.0 kV, got {kv}")


def hv_setting(kv: float | str | Decimal) -> str:
    """``HV v``: voltage setpoint only. It does not switch HV on."""
    return "HV " + format_hv(kv)


def beam(current: int, stage: str, n: int) -> str:
    """``BEAM C n`` / ``BEAM F n`` (current=1) or ``BEAM2 C n`` / ``BEAM2 F n`` (current=2).

    Values are 0-99 control settings, not calibrated probe currents.
    """
    if current not in (1, 2):
        raise ValidationError("Beam current family must be 1 (BEAM) or 2 (BEAM2)")
    s = _choice("Beam stage", stage, ("C", "F"))
    name = "BEAM" if current == 1 else "BEAM2"
    return f"{name} {s} {_int_in(name, n, 0, 99)}"


def focus(stage: str, n: int) -> str:
    s = _choice("Focus stage", stage, ("C", "F"))
    hi = 16383 if s == "C" else 4095
    return f"FOCUS {s} {_int_in('Focus ' + s, n, 0, hi)}"


def parse_magnification(m: int | str) -> int:
    """Numeric magnification for a listed number (``1200``) or its wire token (``"1.2k"``)."""
    value = m if type(m) is int else MAG_VALUES.get(str(m).strip().lower())
    if value not in MAG_TABLE:
        raise ValidationError(f"Magnification {m!r} is not in the permitted list (manual section 8)")
    return value


def format_magnification(m: int | str) -> str:
    """Wire token for a permitted magnification: ``500``, ``1k``, ``1.2k``... (always k notation from 1000 up)."""
    return MAG_TABLE[parse_magnification(m)]


def magnification(m: int | str) -> str:
    return "MAG " + format_magnification(m)


def stigmator(axis: str, n: int) -> str:
    a = _choice("Stigmator axis", axis, ("X", "Y"))
    return f"STG {a} {_int_in('Stigmator ' + a, n, 0, 4095)}"


def abc() -> str:
    return "ABC ON"


def afc(stage: str) -> str:
    return "AFC " + _choice("AFC stage", stage, ("C", "F"))


def asf() -> str:
    """``ASF ON``. Distinct from :func:`afs`."""
    return "ASF ON"


def afs(level: str) -> str:
    """``AFS H`` / ``AFS M`` / ``AFS L``. Distinct from :func:`asf`."""
    return "AFS " + _choice("AFS level", level, ("H", "M", "L"))


def aga() -> str:
    return "AGA ON"


# --------------------------------------------------------------------------- detectors & photo


def signal_token(signal: str) -> str:
    """Map a display name (``BSE1``) or wire token (``BSE``) to the wire token."""
    s = str(signal).strip().upper()
    if s in SIGNAL_TOKENS:
        return SIGNAL_TOKENS[s]
    if s in ("IP", "EP"):
        return "EP"
    if s in WIRE_SIGNAL_TOKENS:
        return s
    raise ValidationError(f"Unknown signal {signal!r}; use one of {', '.join(SIGNAL_TOKENS)}")


def inputs(destination: str, signal: str) -> str:
    d = _choice("Display destination", destination, DISPLAY_DESTINATIONS)
    return f"INPUTS {d} {signal_token(signal)}"


def photo() -> str:
    return "#PHOTO ON"


def photo_speed(seconds: int) -> str:
    return f"PSPEED {_choice('Direct photographing time', seconds, tuple(map(str, PHOTO_SPEEDS)))}"


def memory_photo_speed(seconds: int) -> str:
    return f"MPSPEED {_choice('Memory photographing time', seconds, tuple(map(str, MEMORY_PHOTO_SPEEDS)))}"


# --------------------------------------------------------------------------- vacuum, WD, startup, stage toggle


def vacuum_mode(mode: str) -> str:
    """``VAC HIGH`` / ``VAC LOW`` (low-vacuum-capable models only)."""
    return "VAC " + _choice("Vacuum mode", mode, ("HIGH", "LOW"))


def vacuum_level(level: int | str) -> str:
    """``VAC value`` in Pa, or ``VAC CLOSE``. Low-vacuum mode only."""
    return "VAC " + _choice("Low-vacuum level", level, VACUUM_LEVELS)


def working_distance(n: int) -> str:
    return f"WD {_int_in('Working distance', n, 3, 60)}"


def auto_startup(routine: int) -> str:
    if routine not in (1, 2):
        raise ValidationError("Full auto startup routine must be 1 or 2")
    return f"AUTO{routine} ON"


def stage_control(on: bool) -> str:
    return "#STAGE " + _on_off(on)


# --------------------------------------------------------------------------- analysis

LINE_POS = (50, 460)
SPOT_X = (64, 574)
SPOT_Y = (50, 460)
AREA_SIZE = (40, 320)


def line_position(n: int) -> str:
    return f"LINEP {_int_in('Line position', n, *LINE_POS)}"


def line_analysis() -> str:
    return "LINEA ON"


def line_image(on: bool) -> str:
    return "LINIM " + _on_off(on)


def spot_position(axis: str, n: int) -> str:
    a = _choice("Spot axis", axis, ("X", "Y"))
    lo, hi = SPOT_X if a == "X" else SPOT_Y
    return f"SPOTP {a} {_int_in('Spot ' + a, n, lo, hi)}"


def spot_analysis() -> str:
    return "SPOTA ON"


def area_position(axis: str, n: int) -> str:
    """Single-field area position command. Check the full rectangle with :class:`Area`."""
    a = _choice("Area axis", axis, ("X", "Y"))
    lo, hi = (44, 595 - AREA_SIZE[0]) if a == "X" else (50 + AREA_SIZE[0], 460)
    return f"AREAP {a} {_int_in('Area position ' + a, n, lo, hi)}"


def area_size(axis: str, n: int) -> str:
    a = _choice("Area axis", axis, ("X", "Y"))
    return f"AREAI {a} {_int_in('Area size ' + a, n, *AREA_SIZE)}"


def area_analysis() -> str:
    return "AREAA ON"


def end_analysis() -> str:
    return "ANAL OFF"


@dataclass(frozen=True)
class Area:
    """Area-analysis rectangle in raw interface coordinates.

    Constraints (manual section 7)::

        44 <= x <= 595 - size_x
        50 + size_y <= y <= 460
        40 <= size_x, size_y <= 320
    """

    x: int
    y: int
    size_x: int
    size_y: int

    def problems(self) -> list[str]:
        out = []
        if not AREA_SIZE[0] <= self.size_x <= AREA_SIZE[1]:
            out.append(f"size_x {self.size_x} outside 40-320")
        if not AREA_SIZE[0] <= self.size_y <= AREA_SIZE[1]:
            out.append(f"size_y {self.size_y} outside 40-320")
        if not 44 <= self.x <= 595 - self.size_x:
            out.append(f"x {self.x} outside 44-{595 - self.size_x} (595 - size_x)")
        if not 50 + self.size_y <= self.y <= 460:
            out.append(f"y {self.y} outside {50 + self.size_y}-460 (50 + size_y to 460)")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems()


_AREA_FIELDS = {
    "x": ("AREAP", "X"),
    "y": ("AREAP", "Y"),
    "size_x": ("AREAI", "X"),
    "size_y": ("AREAI", "Y"),
}


def plan_area_update(current: Area, target: Area) -> list[str]:
    """Return the commands that move ``current`` to ``target`` one field at a time.

    The order is chosen so that every intermediate rectangle also satisfies the
    constraints, because the interface has no atomic four-field update and a new
    size can invalidate the present position. Raises :class:`ValidationError`
    if the target is invalid or no safe order exists.
    """
    if not target.is_valid:
        raise ValidationError("Target area is invalid: " + "; ".join(target.problems()))
    changed = [f for f in _AREA_FIELDS if getattr(current, f) != getattr(target, f)]
    for order in itertools.permutations(changed):
        state = current
        ok = True
        for field in order:
            state = replace(state, **{field: getattr(target, field)})
            if not state.is_valid:
                ok = False
                break
        if ok:
            cmds = []
            for field in order:
                name, axis = _AREA_FIELDS[field]
                cmds.append(f"{name} {axis} {getattr(target, field)}")
            return cmds
    raise ValidationError(
        "No single-field update order keeps every intermediate area valid; move through an intermediate rectangle first"
    )
