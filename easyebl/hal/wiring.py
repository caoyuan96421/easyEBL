"""Signal → physical-channel wiring of the scan DAQ (NI USB-6363), edited in one place.

:class:`Wiring` is the part of :class:`~easyebl.hal.config.DAQConfig` that says
which wire carries which signal: X/Y deflection and the auxiliary outputs (AO),
the blanker and hardware-timed marker lines (port0), the video / SE inputs (AI),
the external-scan relay lines and other static lines (port1/2 = PFI), and the
optional sample-clock / start-trigger exports (PFI).  It is stored in the
``wiring`` section of ``system.json`` (*Hardware → Configure… → Wiring*) and
applied over the wiring preset when the DAQ is opened.  No Qt here.

USB-6363 channels (:data:`USB6363`):

* AO ``ao0``–``ao3``;
* AI ``ai0``–``ai31`` single ended (RSE / NRSE).  Differential pairs AI *n* with
  AI *n*\\+8, so only ``ai0``–``ai7`` and ``ai16``–``ai23`` exist in DIFF and each
  also uses its pair;
* ``port0/line0``–``line31``: hardware-timed DIO (the stream drives the whole port);
* ``port1/line0``–``7`` = ``PFI0``–``7`` and ``port2/line0``–``7`` = ``PFI8``–``15``:
  static DIO / PFI.  The two names are the same pin, so both are checked as one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import re

__all__ = ["Wiring", "WiringIssue", "USB6363", "ChannelCatalog", "pin_key", "canonical_line", "ai_pins"]


@dataclass(frozen=True)
class ChannelCatalog:
    """The physical channels of a DAQ device, by function."""

    name: str
    ao: tuple
    ai: tuple
    timed_lines: tuple  # hardware-timed DIO (port0)
    static_lines: tuple  # software-timed DIO usable for relays / static levels (PFI names)
    pfi: tuple  # PFI terminals usable for clock / trigger exports
    diff_pair_offset: int = 8

    def ai_for(self, terminal: str) -> tuple:
        """AI channels that exist in this input mode."""
        if _differential(terminal):
            return tuple(c for c in self.ai if (int(c[2:]) % 16) < self.diff_pair_offset)
        return self.ai


USB6363 = ChannelCatalog(
    name="NI USB-6363",
    ao=tuple(f"ao{i}" for i in range(4)),
    ai=tuple(f"ai{i}" for i in range(32)),
    timed_lines=tuple(f"port0/line{i}" for i in range(32)),
    static_lines=tuple(f"PFI{i}" for i in range(16)),
    pfi=tuple(f"PFI{i}" for i in range(16)),
)


def _differential(terminal: str | None) -> bool:
    return str(terminal or "").upper() in ("DIFF", "PSEUDO_DIFF")


_PORT_LINE = re.compile(r"^port(\d+)/line(\d+)$", re.I)
_PFI = re.compile(r"^pfi(\d+)$", re.I)


def canonical_line(line: str) -> str:
    """``port1/line3`` → ``PFI3``, ``port2/line0`` → ``PFI8`` (USB-6363 aliases); others unchanged."""
    s = str(line).strip()
    m = _PORT_LINE.match(s)
    if m:
        port, ln = int(m.group(1)), int(m.group(2))
        if port in (1, 2) and ln < 8:
            return f"PFI{(port - 1) * 8 + ln}"
        return f"port{port}/line{ln}"
    m = _PFI.match(s)
    if m:
        return f"PFI{int(m.group(1))}"
    return s


def pin_key(terminal: str) -> str:
    """One key per physical pin (aliases collapse): used to find two signals on one wire."""
    return canonical_line(terminal).lower()


def ai_pins(channel: str, terminal: str | None) -> tuple:
    """Physical AI pins used by *channel* in input mode *terminal* (DIFF: AI+ and AI−)."""
    c = str(channel).strip().lower()
    if not _differential(terminal) or not c.startswith("ai"):
        return (c,)
    try:
        n = int(c[2:])
    except ValueError:
        return (c,)
    return (c, f"ai{n + 8}") if n % 16 < 8 else (c,)


@dataclass(frozen=True)
class WiringIssue:
    """A wiring problem and the signals it concerns (keys as in :meth:`Wiring.signals`)."""

    message: str
    signals: tuple = ()


@dataclass
class Wiring:
    """Which physical channel carries each DAQ signal (logical names → terminals)."""

    ao_x: str = "ao0"
    ao_y: str = "ao1"
    aux0: str | None = "ao2"
    aux1: str | None = "ao3"
    blank_line: str = "port0/line23"
    blank_active_high: bool = False
    timed_lines: dict = field(default_factory=dict)  # name -> port0 line (hardware-timed markers)
    external_control_lines: dict = field(default_factory=lambda: {"PFI6": True, "PFI7": True})  # line -> active high
    static_lines: dict = field(default_factory=dict)  # name -> PFI / port1-2 line
    static_lines_active_high: dict = field(default_factory=dict)
    ai_channels: dict = field(default_factory=lambda: {"video": "ai0"})  # name -> AI; the first is the SEM video
    eds_video: str | None = "video"  # AI input co-acquired by EDS as its SE signal (None: none)
    sample_clock_out: str | None = None  # export the scan sample clock on this PFI
    start_trigger_out: str | None = None  # export the stream start trigger on this PFI

    # ------------------------------------------------------------ conversion
    @classmethod
    def from_daq_config(cls, cfg, eds_video: str | None = "video") -> "Wiring":
        """The wiring of a :class:`DAQConfig` (port1/2 lines named as PFI)."""
        aux = dict(cfg.aux_outputs or {})
        ai = dict(cfg.ai_channels or {})
        if eds_video is not None and eds_video not in ai:
            eds_video = next(iter(ai), None)
        return cls(
            ao_x=cfg.ao_x, ao_y=cfg.ao_y, aux0=aux.get("aux0"), aux1=aux.get("aux1"),
            blank_line=cfg.blank_line, blank_active_high=bool(cfg.blank_active_high),
            timed_lines=dict(cfg.timed_lines or {}),
            external_control_lines={canonical_line(k): v for k, v in cfg.external_control_map().items()},
            static_lines={k: canonical_line(v) for k, v in (cfg.static_lines or {}).items()},
            # only the exceptions are stored: a line not listed is active high
            static_lines_active_high={k: False for k, v in (cfg.static_lines_active_high or {}).items()
                                      if not v and k in (cfg.static_lines or {})},
            ai_channels=ai, eds_video=eds_video,
            sample_clock_out=getattr(cfg, "sample_clock_out", None),
            start_trigger_out=getattr(cfg, "start_trigger_out", None),
        )

    def apply_to(self, cfg) -> None:
        """Write this wiring into *cfg* (a :class:`DAQConfig`), in place."""
        cfg.ao_x, cfg.ao_y = self.ao_x, self.ao_y
        cfg.aux_outputs = {"aux0": self.aux0 or None, "aux1": self.aux1 or None}
        cfg.blank_line, cfg.blank_active_high = self.blank_line, bool(self.blank_active_high)
        cfg.timed_lines = dict(self.timed_lines)
        cfg.external_control_lines = {k: bool(v) for k, v in self.external_control_lines.items()}
        cfg.static_lines = dict(self.static_lines)
        cfg.static_lines_active_high = {k: bool(v) for k, v in self.static_lines_active_high.items()
                                        if k in self.static_lines}
        cfg.ai_channels = dict(self.ai_channels)
        cfg.sample_clock_out = self.sample_clock_out or None
        cfg.start_trigger_out = self.start_trigger_out or None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Wiring":
        known = {f.name for f in fields(cls)}
        out = cls(**{k: v for k, v in (d or {}).items() if k in known})
        for name in ("timed_lines", "external_control_lines", "static_lines", "static_lines_active_high",
                     "ai_channels"):
            v = getattr(out, name)
            if not isinstance(v, dict):
                raise TypeError(f"wiring.{name} must be an object")
        return out

    def copy(self) -> "Wiring":
        return Wiring.from_dict(self.to_dict())

    # ------------------------------------------------------------ checking
    def signals(self) -> dict:
        """``{signal key: physical channel}`` of every assigned signal (for display and checks)."""
        s = {"ao_x": self.ao_x, "ao_y": self.ao_y}
        for k in ("aux0", "aux1"):
            if getattr(self, k):
                s[k] = getattr(self, k)
        s["blank"] = self.blank_line
        for n, line in self.timed_lines.items():
            s[f"timed:{n}"] = line
        for line in self.external_control_lines:
            s[f"ext:{line}"] = line
        for n, line in self.static_lines.items():
            s[f"static:{n}"] = line
        for n, ch in self.ai_channels.items():
            s[f"ai:{n}"] = ch
        if self.sample_clock_out:
            s["clock_out"] = self.sample_clock_out
        if self.start_trigger_out:
            s["trigger_out"] = self.start_trigger_out
        return s

    def check(self, ai_terminal: str = "RSE", catalog: ChannelCatalog = USB6363) -> list:
        """Every problem of this wiring on *catalog* (empty list: usable)."""
        issues: list[WiringIssue] = []

        def bad(msg, *sig):
            issues.append(WiringIssue(msg, tuple(sig)))

        lower = lambda seq: {str(x).lower() for x in seq}  # noqa: E731
        ao_ok, timed_ok = lower(catalog.ao), lower(catalog.timed_lines)
        static_ok = lower(catalog.static_lines) | {pin_key(x) for x in catalog.static_lines}
        pfi_ok = lower(catalog.pfi)
        ai_all, ai_mode = lower(catalog.ai), lower(catalog.ai_for(ai_terminal))

        # --- each signal on a channel of the right kind
        for k in ("ao_x", "ao_y", "aux0", "aux1"):
            v = getattr(self, k)
            if (v or k in ("ao_x", "ao_y")) and str(v).lower() not in ao_ok:
                bad(f"{_label(k)}: '{v}' is not an analog output of the {catalog.name}", k)
        if str(self.blank_line).lower() not in timed_ok:
            bad(f"Blanker: '{self.blank_line}' is not a hardware-timed line (port0/line0–31)", "blank")
        for n, line in self.timed_lines.items():
            if str(line).lower() not in timed_ok:
                bad(f"Timed line '{n}': '{line}' is not on port0", f"timed:{n}")
        for line in self.external_control_lines:
            if pin_key(line) not in static_ok:
                bad(f"External control: '{line}' is not a static line (PFI0–15 / port1–2)", f"ext:{line}")
        for n, line in self.static_lines.items():
            if pin_key(line) not in static_ok:
                bad(f"Static line '{n}': '{line}' is not a static line (PFI0–15 / port1–2)", f"static:{n}")
        for k in ("sample_clock_out", "start_trigger_out"):
            v = getattr(self, k)
            if v and pin_key(v) not in pfi_ok:
                bad(f"{_label(k)}: '{v}' is not a PFI terminal", _short(k))
        mode = str(ai_terminal or "RSE").upper()
        for n, ch in self.ai_channels.items():
            c = str(ch).lower()
            if c not in ai_all:
                bad(f"Video input '{n}': '{ch}' is not an analog input of the {catalog.name}", f"ai:{n}")
            elif c not in ai_mode:
                bad(f"Video input '{n}': {ch} is the negative input of a differential pair in {mode}", f"ai:{n}")

        # --- names
        if not self.ai_channels:
            bad("At least one video input is needed (the SEM image)")
        for kind, names in (("Video input", self.ai_channels), ("Timed line", self.timed_lines),
                            ("Static line", self.static_lines)):
            for n in names:
                if not str(n).strip():
                    bad(f"{kind} without a name", f"{_prefix(kind)}{n}")
        if self.eds_video is not None and self.eds_video not in self.ai_channels:
            bad(f"EDS SE input '{self.eds_video}' is not one of the video inputs", "eds_video")

        # --- two signals on one output (aliases are one pin); AI inputs on one pin
        def shared(sig: dict, what: str):
            by_pin: dict[str, list] = {}
            for key, pins in sig.items():
                for p in pins:
                    by_pin.setdefault(p, []).append(key)
            for p, keys in by_pin.items():
                if len(keys) > 1:
                    names = ", ".join(_label(k) for k in keys)
                    bad(f"{names[:1].upper()}{names[1:]} share {what} {_pin_label(p)}", *keys)

        shared({k: (str(v).lower(),) for k, v in self.signals().items() if k in ("ao_x", "ao_y", "aux0", "aux1")},
               "the analog output")
        digital = {k: (pin_key(v),) for k, v in self.signals().items()
                   if k == "blank" or k.split(":")[0] in ("timed", "ext", "static") or k in ("clock_out", "trigger_out")}
        shared(digital, "the line")
        shared({f"ai:{n}": ai_pins(ch, ai_terminal) for n, ch in self.ai_channels.items()}, "the analog input")

        # --- the stream writes the blanker's whole port: static lines must be elsewhere
        port = str(self.blank_line).split("/")[0].lower()
        for k, v in self.signals().items():
            if k.split(":")[0] in ("ext", "static", "clock_out", "trigger_out") and \
                    pin_key(v).split("/")[0] == port:
                bad(f"{_label(k)} is on {port}, which the scan stream drives as a whole – use PFI0–15", k)
        return issues

    def problems(self, ai_terminal: str = "RSE", catalog: ChannelCatalog = USB6363) -> list:
        return [i.message for i in self.check(ai_terminal, catalog)]

    def validate(self, ai_terminal: str = "RSE", catalog: ChannelCatalog = USB6363) -> None:
        """Raise :class:`ValueError` listing every problem."""
        p = self.problems(ai_terminal, catalog)
        if p:
            raise ValueError("wiring: " + "; ".join(p))


_LABELS = {"ao_x": "Scan X", "ao_y": "Scan Y", "aux0": "Aux 0", "aux1": "Aux 1", "blank": "Blanker",
           "sample_clock_out": "Sample clock out", "start_trigger_out": "Start trigger out",
           "clock_out": "Sample clock out", "trigger_out": "Start trigger out", "eds_video": "EDS SE input"}


def _short(k: str) -> str:
    return {"sample_clock_out": "clock_out", "start_trigger_out": "trigger_out"}.get(k, k)


def _prefix(kind: str) -> str:
    return {"Video input": "ai:", "Timed line": "timed:", "Static line": "static:"}[kind]


def _label(key: str) -> str:
    if key in _LABELS:
        return _LABELS[key]
    kind, _, name = key.partition(":")
    return {"timed": f"timed line '{name}'", "ext": f"external control {name}", "static": f"static line '{name}'",
            "ai": f"video input '{name}'"}.get(kind, key)


def _pin_label(p: str) -> str:
    return p.upper() if p.startswith("pfi") else p
