"""In-process simulator of the SEM text interface and the five-axis stage.

:class:`SimulatedInstrument` implements the :class:`~easyebl.microscope.s3000.transport.Transport`
interface, so it can replace a serial port in tests and in the GUI's demo mode.
By default it reproduces the behaviour observed on a real S-3000 on 2026-11-15,
where that differs from the manual:

* the SEM echo is upper-cased and drops a leading ``#``;
* ``MAG`` needs k notation at 1000 and above (``MAG 1200`` -> ``E9``);
* stage lines get ``<line> E2`` until ``#STAGE ON``;
* R1 has no sign and no ``uE0``, while R0/R4 are followed by ``uE0``.

It does not model real firmware timing. Fault-injection hooks let tests
exercise timeouts and bad echoes.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field

from . import commands as C


@dataclass
class _AxisSim:
    position: int = 0
    target: int | None = None
    moving: bool = False
    start_pos: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    direction: str = "+"
    timer: threading.Timer | None = None
    #: after the completion report: read-back closes in from settle_from to position until settle_end
    settle_from: int = 0
    settle_start: float = 0.0
    settle_end: float = 0.0


@dataclass
class SimOptions:
    """Behaviour switches for the simulator."""

    low_vacuum_capable: bool = False
    #: Seconds to "execute" ordinary commands and automatic routines.
    command_delay: float = 0.0
    auto_delay: float = 0.3
    #: Photographing takes PSPEED * photo_time_scale seconds.
    photo_time_scale: float = 0.002
    #: Stage speed in pulses per second for simulated moves.
    stage_speed: float = 20000.0
    #: Stage queries followed by uE0. Observed: R0 and R4. The manual's general rule: R1 only.
    e0_after_queries: frozenset = frozenset({"R0", "R4"})
    #: R3 separator between fields ("" observed, or " ").
    r3_separator: str = ""
    #: X / Y (controller 0) report completion this many pulses short of the target, then the
    #: encoder loop closes in over settle_time_s (observed on the S-3000: seconds after "stopped").
    settle_error_pulses: int = 0
    settle_time_s: float = 0.0
    #: {(controller, motor): pulses} – a lasting error at the end of every move (lost steps), for tests
    final_error_pulses: dict = field(default_factory=dict)
    #: Initial #STAGE state. While off, stage lines are answered "<line> E2" by the SEM.
    stage_enabled: bool = False
    #: Line terminator for replies (CR per the manual; "\r\n" to test LF tolerance).
    terminator: str = "\r"


@dataclass
class _Faults:
    drop_next_reply: int = 0
    garble_next_echo: int = 0
    delay_next_reply: float = 0.0


@dataclass
class SimulatedInstrument:
    """Thread-safe loopback transport answering like the SEM and stage controllers."""

    options: SimOptions = field(default_factory=SimOptions)

    def __post_init__(self) -> None:
        self._rx: "queue.Queue[bytes]" = queue.Queue()
        self._inbuf = bytearray()
        self._lock = threading.RLock()
        self._closed = False
        self.faults = _Faults()
        self.received: list[str] = []
        self.sem = {
            "RUN": "ON",
            "EMI": "10",
            "HV": "OFF",
            "HV_SET": "15.0",
            "HV_ON": False,
            "BEAM C": "50",
            "BEAM F": "50",
            "BEAM2 C": "50",
            "BEAM2 F": "50",
            "FOCUS C": "10670",  # WD 15 (see focus_wd)
            "FOCUS F": "2048",
            "MAG": "100",
            "STG X": "2048",
            "STG Y": "2048",
            "CONT": "2048",
            "BRT": "2048",
            "DYNF": "OFF",
            "DYNF_SET": "0",
            "TILTC": "OFF",
            "TILTC_SET": "0",
            "INPUTS AF": "SE",
            "INPUTS AR": "SE",
            "INPUTS AL": "SE",
            "SPLIT": "OFF",
            "ROTATION": "OFF",
            "ROTATION_SET": "0",
            "PSPEED": "40",
            "MPSPEED": "40",
            "VAC": "HIGH",
            "WD": "15",
            "LINEP": "255",
            "SPOTP X": "319",
            "SPOTP Y": "255",
            "AREAP X": "100",
            "AREAP Y": "300",
            "AREAI X": "100",
            "AREAI Y": "100",
            "SCAN": "F",
            "VAC_MODE": "HIGH",
            "STAGE": "ON" if self.options.stage_enabled else "OFF",
        }
        self.axes = {(u, a): _AxisSim() for (u, a) in [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]}
        self.trackball_speed = {0: 1, 1: 1, 2: 1}
        self.touch = {k: False for k in self.axes}
        self.limits = {k: (False, False) for k in self.axes}
        self.soft_limits = {k: (0x3FFFFF, 0xC00000) for k in self.axes}

    # ------------------------------------------------------------------ Transport API
    def write(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                raise OSError("simulator closed")
            self._inbuf.extend(data)
            while (idx := self._inbuf.find(b"\r")) >= 0:
                line = bytes(self._inbuf[:idx]).decode("ascii", errors="replace")
                del self._inbuf[: idx + 1]
                self.received.append(line)
                threading.Thread(target=self._handle, args=(line,), daemon=True).start()

    def read(self, timeout: float) -> bytes:
        try:
            first = self._rx.get(timeout=timeout)
        except queue.Empty:
            return b""
        out = bytearray(first)
        while True:
            try:
                out.extend(self._rx.get_nowait())
            except queue.Empty:
                return bytes(out)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for ax in self.axes.values():
                if ax.timer:
                    ax.timer.cancel()

    # ------------------------------------------------------------------ helpers
    def _send(self, text: str, *, sem_reply: bool = False) -> None:
        with self._lock:
            if sem_reply or text[:1].isdigit():
                if self.faults.drop_next_reply:
                    self.faults.drop_next_reply -= 1
                    return
            if sem_reply and self.faults.garble_next_echo:
                self.faults.garble_next_echo -= 1
                text = "X" + text
            delay = self.faults.delay_next_reply
            self.faults.delay_next_reply = 0.0
        if delay:
            time.sleep(delay)
        self._rx.put((text + self.options.terminator).encode("ascii"))

    def emit_event(self, text: str) -> None:
        """Inject an asynchronous line (e.g. ``"011"`` limit event, ``"SYS00-1234"``)."""
        self._rx.put((text + self.options.terminator).encode("ascii"))

    def _handle(self, line: str) -> None:
        try:
            if line.startswith("#") and len(line) >= 5 and line[1:3].isdigit():
                if self.sem["STAGE"] != "ON":
                    self._send(f"{line} E2", sem_reply=True)
                    return
                self._stage(line)
            else:
                if self.options.command_delay:
                    time.sleep(self.options.command_delay)
                result, data = self._sem(line.upper())
                # Observed: the echo is upper-cased and drops a leading '#'.
                echo = line.upper()
                if echo.startswith("#"):
                    echo = echo[1:]
                reply = f"{echo} {data} {result}" if data is not None else f"{echo} {result}"
                self._send(reply, sem_reply=True)
        except Exception as exc:  # pragma: no cover - simulator bug guard
            self._send(f"{line} E1", sem_reply=True)
            print("simulator error", exc)

    # ------------------------------------------------------------------ SEM
    def _sem(self, line: str) -> tuple[str, str | None]:
        s = self.sem
        tok = line.split(" ")
        head, args = tok[0], tok[1:]
        if head == "R":
            key = " ".join(args)
            if key not in C.READ_COMMANDS:
                return "E8", None
            if key == "HV":
                return "G0", (s["HV_SET"] if s["HV_ON"] else "OFF")
            if key in ("ROTATION", "DYNF", "TILTC"):
                return "G0", (s[key + "_SET"] if s[key] == "ON" else "OFF")
            if key == "WD":  # like the S-3000: the focus read through the WD table, whole mm
                from .focus_wd import wd_from_focus

                return "G0", str(int(round(wd_from_focus(int(s["FOCUS C"]), int(s["FOCUS F"])))))
            return "G0", s[key]

        def num(v: str, lo: int, hi: int) -> bool:
            return v.isdigit() and lo <= int(v) <= hi

        def setv(key: str, v: str, lo: int, hi: int) -> str:
            if not num(v, lo, hi):
                return "E9"
            s[key] = str(int(v))
            return "G0"

        n = len(args)
        if head == "SCAN" and n == 1 and args[0] in C.SCAN_MODES:
            s["SCAN"] = args[0]
            return "G0", None
        if head == "MONIT" and args in (["S"], ["W"]):
            return "G0", None
        if head == "RUN" and args in (["ON"], ["OFF"]):
            s["RUN"] = args[0]
            return "G0", None
        if head == "REDUCE" and n == 1 and args[0] in ("F", "M", "S"):
            return "G0", None
        if head == "SPLIT" and n == 1:
            if args[0] not in C.SPLIT_MODES:
                return "E9", None
            s["SPLIT"] = args[0]
            return "G0", None
        for key, hi in (("ROTATION", 359), ("DYNF", 255), ("TILTC", 180)):
            if head == key and n == 1:
                if args[0] in ("ON", "OFF"):
                    s[key] = args[0]
                    return "G0", None
                return setv(key + "_SET", args[0], 0, hi), None
        if head in ("CONT", "BRT") and n == 1:
            return setv(head, args[0], 0, 4095), None
        if head == "HV" and n == 1:
            if args[0] == "ON":
                time.sleep(self.options.auto_delay)
                s["HV_ON"] = True
                return "G0", None
            if args[0] == "OFF":
                s["HV_ON"] = False
                return "G0", None
            try:
                C.format_hv(args[0])
            except Exception:
                return "E9", None
            s["HV_SET"] = args[0]
            return "G0", None
        if head in ("BEAM", "BEAM2") and n == 2 and args[0] in ("C", "F"):
            return setv(f"{head} {args[0]}", args[1], 0, 99), None
        if head == "FOCUS" and n == 2 and args[0] in ("C", "F"):
            r = setv(f"FOCUS {args[0]}", args[1], 0, 16383 if args[0] == "C" else 4095)
            if r == "G0" and args[0] == "C":
                from .focus_wd import COARSE_MIN

                s["FOCUS C"] = str(max(int(s["FOCUS C"]), COARSE_MIN))  # the S-3000 holds it there
            return r, None
        if head == "STG" and n == 2 and args[0] in ("X", "Y"):
            return setv(f"STG {args[0]}", args[1], 0, 4095), None
        if head == "MAG" and n == 1:
            return self._set_mag(args[0]), None
        if (head, tuple(args)) in {
            ("ABC", ("ON",)),
            ("AFC", ("C",)),
            ("AFC", ("F",)),
            ("ASF", ("ON",)),
            ("AFS", ("H",)),
            ("AFS", ("M",)),
            ("AFS", ("L",)),
            ("AGA", ("ON",)),
            ("AUTO1", ("ON",)),
            ("AUTO2", ("ON",)),
        }:
            time.sleep(self.options.auto_delay)
            return "G0", None
        if head == "INPUTS" and n == 2 and args[0] in C.DISPLAY_DESTINATIONS:
            if args[1] not in C.WIRE_SIGNAL_TOKENS:
                return "E9", None
            s[f"INPUTS {args[0]}"] = args[1]
            return "G0", None
        if head == "#PHOTO" and args == ["ON"]:
            time.sleep(int(s["PSPEED"]) * self.options.photo_time_scale)
            return "G0", None
        if head == "PSPEED" and n == 1:
            if args[0] not in ("40", "80", "200", "400"):
                return "E9", None
            s["PSPEED"] = args[0]
            return "G0", None
        if head == "MPSPEED" and n == 1:
            if args[0] not in ("40", "80"):
                return "E9", None
            s["MPSPEED"] = args[0]
            return "G0", None
        if head == "VAC" and n == 1:
            if not self.options.low_vacuum_capable:
                return "E2", None
            if args[0] in ("HIGH", "LOW"):
                s["VAC_MODE"] = args[0]
                s["VAC"] = args[0]
                return "G0", None
            if args[0] not in C.VACUUM_LEVELS:
                return "E9", None
            if s["VAC_MODE"] != "LOW":
                return "E2", None
            s["VAC"] = args[0]
            return "G0", None
        if head == "WD" and n == 1:
            r = setv("WD", args[0], 3, 60)
            if r == "G0":  # like the S-3000: WD n sets the coarse focus and centres the fine focus
                from .focus_wd import WD_TABLE

                c, f = WD_TABLE.get(int(args[0]), (16383, 4095))
                s["FOCUS C"], s["FOCUS F"] = str(c), str(f)
            return r, None
        if head == "#STAGE" and args in (["ON"], ["OFF"]):
            if args[0] == "OFF" and s["STAGE"] == "OFF":
                return "E6", None  # like the S-3000: #STAGE OFF while already off
            s["STAGE"] = args[0]
            return "G0", None
        if head == "LINEP" and n == 1:
            return setv("LINEP", args[0], 50, 460), None
        if (head, tuple(args)) in {
            ("LINEA", ("ON",)),
            ("SPOTA", ("ON",)),
            ("AREAA", ("ON",)),
            ("ANAL", ("OFF",)),
            ("LINIM", ("ON",)),
            ("LINIM", ("OFF",)),
        }:
            return "G0", None
        if head == "SPOTP" and n == 2 and args[0] in ("X", "Y"):
            lo, hi = C.SPOT_X if args[0] == "X" else C.SPOT_Y
            return setv(f"SPOTP {args[0]}", args[1], lo, hi), None
        if head in ("AREAP", "AREAI") and n == 2 and args[0] in ("X", "Y"):
            if not args[1].isdigit():
                return "E9", None
            area = C.Area(int(s["AREAP X"]), int(s["AREAP Y"]), int(s["AREAI X"]), int(s["AREAI Y"]))
            field_name = {("AREAP", "X"): "x", ("AREAP", "Y"): "y", ("AREAI", "X"): "size_x", ("AREAI", "Y"): "size_y"}[
                (head, args[0])
            ]
            from dataclasses import replace

            new = replace(area, **{field_name: int(args[1])})
            if not new.is_valid:
                return "E9", None
            s[f"{head} {args[0]}"] = args[1]
            return "G0", None
        return "E8", None

    def _set_mag(self, token: str) -> str:
        if token.lower() not in C.MAG_VALUES:
            return "E9"  # unlisted, or not in k notation at 1000 and above (observed: MAG 1200 -> E9)
        self.sem["MAG"] = token.lower()  # read back as e.g. "500", "1.2k", "9k"
        return "G0"


    # ------------------------------------------------------------------ stage
    def _pos_now(self, key) -> int:
        ax = self.axes[key]
        if not ax.moving:
            now = time.monotonic()
            if now < ax.settle_end:
                frac = (now - ax.settle_start) / max(1e-9, ax.settle_end - ax.settle_start)
                return int(round(ax.settle_from + (ax.position - ax.settle_from) * frac))
            return ax.position
        frac = min(1.0, (time.monotonic() - ax.start_time) / max(1e-9, ax.end_time - ax.start_time))
        target = ax.target if ax.target is not None else ax.position
        return int(ax.start_pos + (target - ax.start_pos) * frac)

    def _stage(self, line: str) -> None:
        u, a, op = int(line[1]), int(line[2]), line[3:5]
        params = line[6:].split(" ") if len(line) > 5 else []
        key = (u, a)
        ok = f"{u}E0"
        err = f"{u}E1"
        if key not in self.axes and op not in ("S2", "C0", "R3", "R5", "A0"):
            self._send(err)
            return
        hex6 = re.compile(r"^[0-9A-F]{6}$")
        with self._lock:
            if op == "R0":
                cw, ccw = self.limits[key]
                letter = {(False, False): "A", (True, False): "B", (False, True): "C", (True, True): "D"}[(cw, ccw)]
                self._send(f"{u}{a}{letter}")
                if op in self.options.e0_after_queries:
                    self._send(ok)
                return
            if op == "R1":
                ax = self.axes[key]
                self._send(f"{u}P{self._pos_now(key) & 0xFFFFFF:06X}")  # observed: no sign
                if op in self.options.e0_after_queries:
                    self._send(ok)
                return
            if op == "R4":
                self._send(f"{u}{a}{'K' if self.touch[key] else 'L'}")
                if op in self.options.e0_after_queries:
                    self._send(ok)
                return
            if op == "R3":
                p0 = self._pos_now((u, 0)) & 0xFFFFFF
                p1 = self._pos_now((u, 1)) & 0xFFFFFF if (u, 1) in self.axes else 0
                sep = self.options.r3_separator
                self._send(f"{u}Q{sep}{p0:06X}{sep}{p1:06X}{sep}{self.trackball_speed[u]}")
                if op in self.options.e0_after_queries:
                    self._send(ok)
                return
            if op == "R5":
                self._send(f"{u}S{self.trackball_speed[u]}")
                if op in self.options.e0_after_queries:
                    self._send(ok)
                return
            if op == "M0" and len(params) == 1 and hex6.match(params[0]):
                self.axes[key].target = _signed24(int(params[0], 16))
                self._send(ok)
                return
            if op == "C0":
                self._send(ok)
                for k in [k for k in self.axes if k[0] == u]:
                    self._start_move(k)
                return
            if op == "M3" and len(params) == 2 and hex6.match(params[0]) and params[1] in "+-":
                self._send(ok)
                ax = self.axes[key]
                ax.direction = params[1]
                ax.moving = True
                ax.start_pos, ax.start_time = ax.position, time.monotonic()
                ax.end_time = ax.start_time + 3600
                ax.target = None
                return
            if op == "S0":
                self._send(ok)
                ax = self.axes[key]
                if ax.moving:
                    if ax.timer:
                        ax.timer.cancel()
                    ax.position = self._pos_now(key) if ax.target is not None else ax.position
                    ax.moving = False
                    self._send(f"{u}{a}4")
                return
            if op == "C2" and len(params) == 1 and hex6.match(params[0]):
                self.axes[key].position = _signed24(int(params[0], 16))
                self._send(ok)
                return
            if op == "O1" and params and params[0] in ("0", "1", "2"):
                self.trackball_speed[u] = int(params[0])
                self._send(ok)
                return
            if op in ("I0", "I1", "I2", "I3", "O2", "A0", "S2"):
                self._send(ok)
                return
        self._send(err)

    def _start_move(self, key) -> None:
        ax = self.axes[key]
        if ax.target is None:
            ax.target = ax.position
        ax.position = self._pos_now(key)  # a correction still in progress ends here
        ax.settle_end = 0.0
        ax.start_pos = ax.position
        dist = abs(ax.target - ax.position)
        ax.direction = "+" if ax.target >= ax.position else "-"
        ax.start_time = time.monotonic()
        ax.end_time = ax.start_time + dist / self.options.stage_speed
        ax.moving = True
        ax.timer = threading.Timer(max(0.01, ax.end_time - ax.start_time), self._finish_move, args=(key,))
        ax.timer.daemon = True
        ax.timer.start()

    def _finish_move(self, key) -> None:
        with self._lock:
            ax = self.axes[key]
            if not ax.moving or ax.target is None:
                return
            travel = ax.target - ax.start_pos
            ax.position = ax.target + self.options.final_error_pulses.get(key, 0)
            ax.moving = False
            err = self.options.settle_error_pulses
            if key[0] == 0 and err and travel and self.options.settle_time_s > 0:
                ax.settle_from = ax.position - (err if travel > 0 else -err)
                ax.settle_start = time.monotonic()
                ax.settle_end = ax.settle_start + self.options.settle_time_s
        self._send(f"{key[0]}{key[1]}0")


def _signed24(n: int) -> int:
    return n - (1 << 24) if n & 0x800000 else n
