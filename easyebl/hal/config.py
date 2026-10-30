"""Static description of how the DAQ is wired to the SEM."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json

__all__ = ["DAQConfig"]


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
    * ``static_lines``: software-timed DO lines (e.g. the internal/external
      scan relay) – put these on port1/port2.
    * ``ai_channels``: logical video channel name → physical AI channel.
    """

    device: str = "Dev1"
    ao_x: str = "ao0"
    ao_y: str = "ao1"
    ao_limit_v: float = 10.0
    blank_line: str = "port0/line0"
    blank_active_high: bool = True
    timed_lines: dict = field(default_factory=dict)
    static_lines: dict = field(default_factory=lambda: {"scan_relay": "port1/line0"})
    static_lines_active_high: dict = field(default_factory=dict)
    ai_channels: dict = field(default_factory=lambda: {"video": "ai0"})
    ai_range_v: float = 10.0
    ai_terminal: str = "DIFF"  # DIFF | RSE | NRSE | PSEUDO_DIFF
    park_x_v: float = 0.0
    park_y_v: float = 0.0
    max_ao_rate: float = 840e3  # USB-6341, two channels
    max_ai_aggregate_rate: float = 500e3  # USB-6341, multiplexed
    buffer_samples: int = 200_000
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
        if self.buffer_samples < 1000:
            raise ValueError("buffer_samples too small")

    def physical(self, terminal: str) -> str:
        return f"{self.device}/{terminal}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DAQConfig":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DAQConfig":
        return cls.from_dict(json.loads(s))
