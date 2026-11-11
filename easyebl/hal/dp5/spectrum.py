"""
dp5.spectrum - Spectrum container, decoding of spectrum packets and Amptek .mca file I/O.

Spectrum packets (PID1 0x81): channel count from PID2 (256 << ((PID2-1)>>1)), 3 bytes per
channel little-endian; even PID2 = a 64-byte status block follows the counts.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field

import numpy as np

from .protocol import Packet, RSP_SPECTRUM, DP5ProtocolError, spectrum_channels, spectrum_has_status
from .status import Status, decode_status


@dataclass
class Spectrum:
    """One spectrum read from the DP5 (or an accumulation of several in Delta mode).

    counts        numpy int64 array, one entry per channel
    status        decoded status that came with the spectrum (None for spectrum-only reads)
    mode          "mca" (cumulative since last clear) or "delta" (since the previous read)
    seq           poll sequence number since the acquisition was started
    timestamp     host time (time.time()) when the packet was received
    real_time     seconds covered (from the status block)
    acc_time      accumulation time, s
    fast_count    input counts during the period covered
    slow_count    counts in the spectrum during the period covered
    """
    counts: np.ndarray
    status: Status | None = None
    mode: str = "mca"
    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    real_time: float = 0.0
    acc_time: float = 0.0
    fast_count: int = 0
    slow_count: int = 0

    @property
    def channels(self) -> int:
        return int(self.counts.size)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    @property
    def input_rate(self) -> float:
        return self.fast_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def output_rate(self) -> float:
        return self.slow_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def dead_time(self) -> float:
        return 1.0 - self.slow_count / self.fast_count if self.fast_count > 0 else 0.0

    def copy(self) -> "Spectrum":
        return Spectrum(self.counts.copy(), self.status, self.mode, self.seq, self.timestamp,
                        self.real_time, self.acc_time, self.fast_count, self.slow_count)


def decode_spectrum_packet(pkt: Packet) -> tuple[np.ndarray, Status | None]:
    """Decode a 0x81 packet into (counts, status-or-None)."""
    if pkt.pid1 != RSP_SPECTRUM:
        raise DP5ProtocolError(f"not a spectrum packet: {pkt}")
    nch = spectrum_channels(pkt.pid2)
    need = 3 * nch + (64 if spectrum_has_status(pkt.pid2) else 0)
    if len(pkt.data) < need:
        raise DP5ProtocolError(f"spectrum packet too short ({len(pkt.data)}/{need})")
    raw = np.frombuffer(pkt.data[:3 * nch], np.uint8).reshape(nch, 3).astype(np.int64)
    counts = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    status = decode_status(pkt.data[3 * nch: 3 * nch + 64]) if spectrum_has_status(pkt.pid2) else None
    return counts, status


def save_mca(path: str, spec: Spectrum, config: dict | None = None, description: str = "",
             tag: str = "dp5py"):
    """Write an Amptek-style .mca text file (readable by DPPMCA / PyMCA)."""
    st = spec.status
    start = _dt.datetime.fromtimestamp(spec.timestamp - spec.real_time)
    lines = [
        "<<PMCA SPECTRUM>>",
        f"TAG - {tag}",
        f"DESCRIPTION - {description}",
        f"GAIN - {config.get('GAIN', '') if config else ''}",
        "THRESHOLD - 0",
        "LIVE_MODE - 0",
        f"PRESET_TIME - {config.get('PRET', '') if config else ''}",
        f"LIVE_TIME - {spec.acc_time:.3f}",
        f"REAL_TIME - {spec.real_time:.3f}",
        f"START_TIME - {start:%m/%d/%Y %H:%M:%S}",
        f"SERIAL_NUMBER - {st.serial_number if st else ''}",
        "<<DATA>>",
        *[str(int(c)) for c in spec.counts],
        "<<END>>",
    ]
    if config:
        lines += ["<<DP5 CONFIGURATION>>", *[f"{k}={v};" for k, v in config.items()],
                  "<<DP5 CONFIGURATION END>>"]
    if st:
        lines += ["<<DPP STATUS>>",
                  f"Device Type: {st.device_name}", f"Serial Number: {st.serial_number}",
                  f"Firmware: {st.firmware}", f"FPGA: {st.fpga}",
                  f"Fast Count: {spec.fast_count}", f"Slow Count: {spec.slow_count}",
                  f"Accumulation Time: {spec.acc_time:.3f}", f"Real Time: {spec.real_time:.3f}",
                  f"Board Temp: {st.board_temp_c}",
                  "<<DPP STATUS END>>"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_mca(path: str) -> np.ndarray:
    """Read the <<DATA>> block of an Amptek .mca file."""
    out, on = [], False
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s == "<<DATA>>":
                on = True
            elif s == "<<END>>":
                break
            elif on and s:
                out.append(int(s))
    return np.array(out, np.int64)
