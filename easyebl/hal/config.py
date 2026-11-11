"""Static description of how the DAQ is wired to the SEM."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json

__all__ = ["DAQConfig", "AUX_OUTPUTS", "AI_TERMINALS"]

#: AI input modes.  DIFF needs both AI+ and AI− wired: an unconnected AI− floats and
#: drives the reading into the rail.  RSE measures AI+ against AI GND.
AI_TERMINALS = ("DIFF", "RSE", "NRSE", "PSEUDO_DIFF")

#: Logical auxiliary analog outputs every HAL configuration provides.  On a board
#: without the matching AO channel they are dummies (accepted, no hardware effect).
AUX_OUTPUTS = ("aux0", "aux1")
PHANTOM_KINDS = ("chart", "eds")


@dataclass
class DAQConfig:
    """Wiring/limits of the scan DAQ.

    Logical names are used everywhere above the HAL; this object maps them to
    physical NI-DAQmx terminals.

    * Deflection: two AO channels (``ao_x``, ``ao_y``).
    * Blanker: one hardware-timed DO line clocked with the AO sample clock.
      ``blank_active_high`` describes the electrical level that *blanks* the
      beam.  The wiring should be fail-safe (undriven = blanked); this flag only
      tells software which level to drive.
    * ``timed_lines``: additional hardware-timed DO lines (same port as the
      blanker, port0 on X-series), e.g. ``{"line_sync": "port0/line1"}`` for
      EDS pulse-processor synchronisation.
    * ``external_control_lines``: the wire(s) that switch the SEM from its
      internal scan to the DAQ (external scan), with their polarity::

          {"PFI6": True, "PFI7": False}   # PFI6 active high, PFI7 active low

      "Active" is the level that selects *external* scan; the opposite level
      selects the SEM's internal scan.  A plain list/tuple of wires is also
      accepted and uses ``external_control_active_high`` for all of them.
      Applications only use the logical HAL function
      (``handle.set_external_control(True)``) – they never see these wires.
    * ``static_lines``: other software-timed auxiliary DO lines by logical
      name – put these on port1/port2/PFI.
    * ``ai_channels``: logical video channel name → physical AI channel.
    * ``aux_outputs``: the auxiliary analog outputs ``aux0`` / ``aux1`` → physical
      AO channel, or ``None`` for a dummy output (boards with only two AO
      channels).  Applications use the logical names only.
    * ``phantom_output``: route the phantom specimen (synthetic video, see
      :mod:`easyebl.hal.phantom`) to this aux output during streams, e.g.
      ``"aux0"`` with AO2 wired to the video input.  ``phantom_noise_v`` adds
      white noise.  Ignored on a dummy output.
    * ``phantom_kind``: ``"chart"`` (test chart, :mod:`easyebl.hal.phantom`) or
      ``"eds"`` (multi-phase EDS specimen, :mod:`easyebl.hal.eds_phantom`): its SE
      image goes to ``phantom_output`` and the signal of a reset-type SDD preamp
      to ``sdd_output`` (e.g. ``"aux1"`` = AO3, wired to the pulse processor).
      ``eds_phantom`` holds its parameters (``cps``, ``beam_kv``, ``mv_per_kev``,
      ``noise_mv``, ``leak_v_per_s``; see :class:`EDSPhantomSpecimen`).
    * ``ao_rate_limits``: maximum AO update rate by number of AO channels in a
      stream (X-series share one DAC timing engine); None = ``max_ao_rate``.
    """

    device: str = "Dev1"
    ao_x: str = "ao0"
    ao_y: str = "ao1"
    ao_limit_v: float = 10.0
    blank_line: str = "port0/line0"
    blank_active_high: bool = True
    timed_lines: dict = field(default_factory=dict)
    external_control_lines: dict = field(default_factory=lambda: {"port1/line0": True})
    external_control_active_high: bool = True  # default polarity for the list form
    static_lines: dict = field(default_factory=lambda: {"aux": "port1/line1"})
    static_lines_active_high: dict = field(default_factory=dict)
    ai_channels: dict = field(default_factory=lambda: {"video": "ai0"})
    ai_range_v: float = 10.0
    # RSE: AI+ against AI GND (one wire + common ground).  DIFF rejects ground loops but
    # needs AI− wired too – an unconnected AI− floats and drifts into the rail.
    ai_terminal: str = "RSE"  # RSE | DIFF | NRSE | PSEUDO_DIFF
    aux_outputs: dict = field(default_factory=lambda: {"aux0": None, "aux1": None})
    phantom_output: str | None = None
    phantom_noise_v: float = 0.0
    phantom_kind: str = "chart"  # chart | eds
    sdd_output: str | None = None  # eds phantom: aux output carrying the SDD preamp signal
    eds_phantom: dict = field(default_factory=dict)
    ao_rate_limits: dict | None = None
    park_x_v: float = 0.0
    park_y_v: float = 0.0
    max_ao_rate: float = 840e3  # USB-6341, two channels
    max_ai_aggregate_rate: float = 500e3  # USB-6341, multiplexed
    max_ai_single_rate: float | None = None  # one AI channel (None = aggregate limit)
    buffer_samples: int = 200_000  # minimum output buffer (samples) for default streams
    min_buffer_s: float = 1.0  # …and at least this much time at the stream rate
    min_latency_samples: int = 16_384  # floor for low-latency streams (≥ 2× the AO FIFO)
    usb_xfer_req_size: int | None = None  # USB transfer request size (samples); None = driver default
    tail_hold_s: float = 20e-6  # end of stream: hold last position blanked before parking
    abort_drain_timeout_s: float = 3.0  # abort: max wait for the safe tail to play out
    write_timeout_s: float = 10.0

    def validate(self) -> None:
        if self.ao_limit_v <= 0:
            raise ValueError("ao_limit_v must be positive")
        if not (abs(self.park_x_v) <= self.ao_limit_v and abs(self.park_y_v) <= self.ao_limit_v):
            raise ValueError("park position outside AO limits")
        port = self.blank_line.split("/")[0]
        for name, line in self.timed_lines.items():
            if line.split("/")[0] != port:
                raise ValueError(
                    f"timed line '{name}' ({line}) must be on the same port as the blanker ({port})"
                )
            if line == self.blank_line:
                raise ValueError(f"timed line '{name}' collides with the blank line")
        self.external_control_lines = self.external_control_map()
        for line in self.external_control_lines:
            if line.split("/")[0] == port:
                raise ValueError(
                    f"external-control line {line} is on the hardware-timed port {port}; "
                    "the stream drives that whole port – move it to another port or a PFI line"
                )
            if line in self.static_lines.values():
                raise ValueError(f"external-control line {line} is also configured as a static line")
        for name, line in self.static_lines.items():
            if line.split("/")[0] == port:
                raise ValueError(
                    f"static line '{name}' ({line}) is on the hardware-timed port {port}; "
                    "the stream drives that whole port – move it to another port"
                )
        aux = {k: None for k in AUX_OUTPUTS}
        aux.update(self.aux_outputs or {})
        unknown = set(aux) - set(AUX_OUTPUTS)
        if unknown:
            raise ValueError(f"unknown aux output(s) {sorted(unknown)}; available: {list(AUX_OUTPUTS)}")
        phys = [v for v in aux.values() if v]
        if len(phys) != len(set(phys)) or {self.ao_x, self.ao_y} & set(phys):
            raise ValueError("aux outputs must use distinct AO channels, different from X/Y")
        self.aux_outputs = aux
        if self.phantom_output is not None and self.phantom_output not in AUX_OUTPUTS:
            raise ValueError(f"phantom_output must be one of {list(AUX_OUTPUTS)} or None")
        if self.phantom_noise_v < 0:
            raise ValueError("phantom_noise_v must be ≥ 0")
        if self.phantom_kind not in PHANTOM_KINDS:
            raise ValueError(f"phantom_kind must be one of {list(PHANTOM_KINDS)}")
        if self.sdd_output is not None:
            if self.sdd_output not in AUX_OUTPUTS:
                raise ValueError(f"sdd_output must be one of {list(AUX_OUTPUTS)} or None")
            if self.sdd_output == self.phantom_output:
                raise ValueError("sdd_output and phantom_output must be different aux outputs")
        self.eds_phantom = dict(self.eds_phantom or {})
        if self.ao_rate_limits:
            self.ao_rate_limits = {int(k): float(v) for k, v in self.ao_rate_limits.items()}
        if self.ai_terminal not in AI_TERMINALS:
            raise ValueError(f"ai_terminal must be one of {list(AI_TERMINALS)}")
        if self.buffer_samples < 1000:
            raise ValueError("buffer_samples too small")
        if self.min_latency_samples < 1000:
            raise ValueError("min_latency_samples too small")
        if self.min_buffer_s <= 0 or self.tail_hold_s < 0:
            raise ValueError("min_buffer_s must be > 0 and tail_hold_s ≥ 0")

    @classmethod
    def usb6363(cls, device: str = "Dev1") -> "DAQConfig":
        """Wiring for an NI USB-6363 (X Series) on the S-3000N scan break-in.

        * ``ao0``/``ao1`` = X/Y deflection (DBC X-IN / Y-IN).  Full scale is set by
          ``ao_limit_v``; check it against the SEM's external-scan level before use.
        * Blanker on ``port0/line23`` (P0.23), clocked by the AO sample clock.  The
          blanker input is active LOW: a low level blanks the beam, and an
          undriven input must also blank (fail-safe wiring).
        * PFI6 + PFI7 = external-control (relay-enable) lines, both HIGH = the SEM
          takes X/Y from the DAQ.  They are the HAL's ``external control``
          function; they are written from software between streams.
        * AI ``ai0`` = video, single-ended (RSE) by default; DIFF needs AI0− wired.
        * ``aux0`` = AO2, ``aux1`` = AO3.  For a dummy-image test wire AO2 → AI0 and
          set ``phantom_output="aux0"``.
        """
        return cls(
            device=device,
            ao_x="ao0",
            ao_y="ao1",
            ao_limit_v=5.0,
            blank_line="port0/line23",
            blank_active_high=False,
            timed_lines={},
            external_control_lines={"PFI6": True, "PFI7": True},  # both HIGH = external scan
            static_lines={},
            ai_channels={"video": "ai0"},
            ai_terminal="RSE",  # switch to DIFF (Advanced… / --ai-terminal) when AI0− is wired
            aux_outputs={"aux0": "ao2", "aux1": "ao3"},
            max_ao_rate=2e6,  # 2 AO channels: 2.00 MS/s (2.86 MS/s is the 1-channel limit)
            ao_rate_limits={1: 2.857e6, 2: 2.0e6, 3: 1.54e6, 4: 1.25e6},
            max_ai_aggregate_rate=1e6,  # multi-channel AI: 1.00 MS/s aggregate
            max_ai_single_rate=2e6,  # single-channel AI: 2.00 MS/s
        )

    def external_control_map(self) -> dict:
        """``{wire: active_high}`` of the external-control function (list form normalised)."""
        lines = self.external_control_lines or {}
        if isinstance(lines, str):
            lines = [lines]
        if isinstance(lines, dict):
            return {str(k): bool(v) for k, v in lines.items()}
        return {str(k): bool(self.external_control_active_high) for k in lines}

    def max_ai_rate(self, n_channels: int) -> float:
        """Highest per-channel AI sample rate for *n_channels* channels."""
        if n_channels <= 0:
            return float("inf")
        if n_channels == 1 and self.max_ai_single_rate:
            return float(self.max_ai_single_rate)
        return float(self.max_ai_aggregate_rate) / n_channels

    def aux_physical(self, name: str) -> str | None:
        """Physical AO channel of aux output *name* (None = dummy)."""
        if name not in AUX_OUTPUTS:
            raise KeyError(f"unknown aux output '{name}'; available: {list(AUX_OUTPUTS)}")
        return (self.aux_outputs or {}).get(name)

    @property
    def phantom_physical(self) -> str | None:
        """AO channel driven by the phantom during streams (None = no phantom / dummy output)."""
        return self.aux_physical(self.phantom_output) if self.phantom_output else None

    @property
    def sdd_physical(self) -> str | None:
        """AO channel carrying the EDS phantom's SDD signal during streams (None = off / dummy)."""
        if self.phantom_kind != "eds" or not self.sdd_output:
            return None
        return self.aux_physical(self.sdd_output)

    @property
    def phantom_outputs(self) -> set:
        """Aux outputs the phantom drives during streams (cannot be set statically)."""
        out = set()
        if self.phantom_physical:
            out.add(self.phantom_output)
        if self.sdd_physical:
            out.add(self.sdd_output)
        return out

    @property
    def stream_ao_channels(self) -> int:
        """AO channels in a hardware-timed stream: X, Y (+ phantom image, + SDD signal)."""
        return 2 + (1 if self.phantom_physical else 0) + (1 if self.sdd_physical else 0)

    def max_ao_rate_for(self, n_channels: int) -> float:
        if self.ao_rate_limits:
            lim = {int(k): float(v) for k, v in self.ao_rate_limits.items()}
            if n_channels in lim:
                return lim[n_channels]
        return float(self.max_ao_rate)

    def max_scan_rate(self, n_ai: int = 1) -> float:
        """Fastest sample clock usable for imaging with *n_ai* channels, as 100 MHz / N."""
        lim = min(self.max_ao_rate_for(self.stream_ao_channels), self.max_ai_rate(n_ai))
        div = int(-(-100e6 // lim))  # ceil
        return 100e6 / div

    def physical(self, terminal: str) -> str:
        return f"{self.device}/{terminal}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DAQConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DAQConfig":
        return cls.from_dict(json.loads(s))
