"""User-facing scan presets and engine (hardware) options.

The engine runs the DAQ at one fixed, fast sample clock (the USB-6363 does
2 MS/s with one video channel) and sets the *pixel dwell* by the number of
clock ticks per pixel.  Every tick is digitised and averaged into the pixel,
so slower scans oversample the video and gain SNR ∝ √(samples per pixel) –
the hardware is used at full speed whatever the scan speed, and changing the
speed never requires restarting the hardware tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "ScanSpeed",
    "SCAN_SPEEDS",
    "RESOLUTIONS",
    "EngineOptions",
    "speed_by_name",
    "valid_rates",
    "AI_RANGES_V",
]


@dataclass(frozen=True)
class ScanSpeed:
    name: str
    dwell_s: float  # pixel dwell time

    def label(self) -> str:
        d = self.dwell_s * 1e6
        return f"{self.name}  ({d:g} µs)"


#: Commercial-SEM-like speed ladder: TV (live focusing) → fast → slow → photo.
SCAN_SPEEDS: tuple = (
    ScanSpeed("TV", 0.5e-6),
    ScanSpeed("Fast 1", 1e-6),
    ScanSpeed("Fast 2", 2e-6),
    ScanSpeed("Slow 1", 5e-6),
    ScanSpeed("Slow 2", 10e-6),
    ScanSpeed("Slow 3", 20e-6),
    ScanSpeed("Photo 1", 40e-6),
    ScanSpeed("Photo 2", 80e-6),
    ScanSpeed("Photo 3", 160e-6),
)

#: Scan resolutions (pixels), 4:3 like the S-3000N's photo formats, plus squares.
RESOLUTIONS: tuple = (
    (320, 240),
    (640, 480),
    (1280, 960),
    (2560, 1920),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
)

#: Input ranges of the X-series AI (±V); a range matching the video swing uses
#: the full 16-bit resolution.
AI_RANGES_V: tuple = (10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1)


def speed_by_name(name: str) -> ScanSpeed:
    key = name.strip().lower().replace(" ", "").replace("_", "")
    for s in SCAN_SPEEDS:
        if s.name.lower().replace(" ", "") == key:
            return s
    raise KeyError(f"unknown scan speed '{name}'; choose from {[s.name for s in SCAN_SPEEDS]}")


def valid_rates(max_rate: float, min_rate: float = 10e3) -> list[float]:
    """Sample clocks the X-series timebase produces exactly (100 MHz / N), fastest first."""
    out = []
    for div in (50, 64, 80, 100, 125, 160, 200, 250, 400, 500, 1000, 2000, 4000, 10000):
        r = 100e6 / div
        if min_rate <= r <= max_rate * (1 + 1e-9):
            out.append(r)
    return out


@dataclass(frozen=True)
class EngineOptions:
    """Hardware/stream options of the imaging engine ("Advanced…").

    * ``rate``: sample clock (None = fastest the device supports for the
      configured video channels).  Changing it restarts the stream.
    * ``latency_s``: output buffer depth – how far the host runs ahead of the
      beam.  A setting change reaches the beam after ≈ this time.  Smaller is
      snappier but less tolerant of host stalls.  After an underflow the
      engine doubles it (up to ``max_latency_s``) and restarts by itself.
    * ``averaging``: running average over ≈ N frames (1 = off).
    * ``ai_range_v``: video input range (None = configured).
    * ``chunk_s``: lines are grouped into output chunks of about this length;
      setting changes are applied at line boundaries.
    * ``publish_hz``: maximum rate of partial-frame updates during slow scans.
    * ``restart_on_change``: apply settings immediately (cut the frame at the
      next line, like a commercial SEM) instead of at the end of the frame.
    """

    rate: float | None = None
    latency_s: float = 0.1
    max_latency_s: float = 1.0
    averaging: int = 1
    ai_range_v: float | None = None
    chunk_s: float = 0.004
    publish_hz: float = 20.0
    restart_on_change: bool = True
    auto_recover: bool = True
    max_recoveries: int = 5  # consecutive failed (re)starts before giving up
    device_timeout_s: float = 2.0  # wait for another module's stream to finish
    # Python hands the GIL between threads every "switch interval" (default 5 ms).  The scan
    # thread must get it promptly to keep the DAQ buffer topped up while the GUI or a script is
    # busy, so the engine lowers it (process-wide, never raises it).  None = leave unchanged.
    gil_switch_interval_s: float | None = 0.001

    def validate(self) -> None:
        if self.rate is not None and self.rate <= 0:
            raise ValueError("rate must be positive")
        if not (0.005 <= self.latency_s <= self.max_latency_s):
            raise ValueError(f"latency must be between 5 ms and {self.max_latency_s:g} s")
        if not (1 <= int(self.averaging) <= 1024):
            raise ValueError("averaging must be 1…1024 frames")
        if self.ai_range_v is not None and self.ai_range_v <= 0:
            raise ValueError("ai_range_v must be positive")
        if not (0.0005 <= self.chunk_s <= 0.1):
            raise ValueError("chunk_s must be 0.5…100 ms")
        if not (1.0 <= self.publish_hz <= 200.0):
            raise ValueError("publish_hz must be 1…200")

    def with_(self, **kw) -> "EngineOptions":
        o = replace(self, **kw)
        o.validate()
        return o
