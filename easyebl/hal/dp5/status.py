"""
dp5.status - decoding of the 64-byte DP5 status block.

The same 64-byte block is returned by the status request (0x01 0x01 -> 0x80 0x01) and is
appended to "spectrum + status" responses (0x81, even PID2).
Layout from the SDK (``FW6DppStatus.cs``, ``DP5Status.cpp``). Counters are little-endian.

=======  ======  ===================================================================
Offset   Size    Field
=======  ======  ===================================================================
0-3      u32     fast count (fast-channel triggers = input counts)
4-7      u32     slow count (events into the spectrum)
8-11     u32     general-purpose counter
12       u8      accumulation time, ms part
13-15    u24     accumulation time, 100 ms units
16-19    u32     live time, ms (MCA8000D only)
20-23    u32     real time, ms
24       u8      firmware version (major<<4 | minor)
25       u8      FPGA version (major<<4 | minor)
26-29    u32     serial number (valid if byte 29 < 0x80)
30-31    i16 BE  HV setting, 0.5 V/count
32-33    u12 BE  detector temperature, 0.1 K/count
34       i8      board temperature, degC
35       flags   D7 preset real time done, D6 fast thresh locked, D5 MCA enabled,
                 D4 preset count reached, D2 scope data ready, D1 DP5 configured
36       flags   D7 0 = input offset locked, D6 MCS done, D5 reboot flag,
                 D1 80 MHz mode, D0 FPGA auto clock
37       u8      firmware build (low nibble)
38       flags   D7 PC5 present, D6 PC5 HV polarity, D5 PC5 8.5 V preamp supply
39       u8      device type (0 DP5, 1 PX5, 2 DP5G, 3 MCA8000D, 4 TB5, 5 DP5-X)
40-41    u12 BE  TEC voltage (/758.5 V)
42       u8      DPP options
49       u8      DPP_ECO (board revision / configuration code)
=======  ======  ===================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEVICE_NAMES = {0: "DP5", 1: "PX5", 2: "DP5G", 3: "MCA8000D", 4: "TB5", 5: "DP5-X"}
DEVICE_DP5 = 0
STATUS_LEN = 64


def _u32(b, o):
    return int.from_bytes(b[o:o + 4], "little")


@dataclass
class Status:
    """Decoded DP5 status. ``raw`` keeps the original 64 bytes."""

    raw: bytes = field(repr=False)
    fast_count: int = 0
    slow_count: int = 0
    gp_count: int = 0
    accumulation_time: float = 0.0      # s
    live_time: float = 0.0              # s (MCA8000D only; 0 on DP5)
    real_time: float = 0.0              # s
    firmware: str = ""
    fpga: str = ""
    firmware_raw: int = 0
    fpga_raw: int = 0
    build: int = 0
    serial_number: int = 0
    hv: float = 0.0                     # V (only meaningful with PC5)
    detector_temp_k: float = 0.0        # K (only meaningful with PC5)
    board_temp_c: int = 0
    preset_real_time_done: bool = False
    fast_threshold_locked: bool = False
    mca_enabled: bool = False
    preset_count_reached: bool = False
    scope_data_ready: bool = False
    configured: bool = False
    input_offset_locked: bool = False
    mcs_done: bool = False
    reboot_flag: bool = False
    clock_80mhz: bool = False
    fpga_auto_clock: bool = False
    pc5_present: bool = False
    pc5_hv_positive: bool = False
    pc5_8v5: bool = False
    device_type: int = 0
    tec_voltage: float = 0.0
    dpp_options: int = 0
    dpp_eco: int = 0

    # ---- derived ----
    @property
    def device_name(self) -> str:
        return DEVICE_NAMES.get(self.device_type, f"type {self.device_type}")

    @property
    def is_rev_dx(self) -> bool:
        """DP5 Rev Dx board (24-step coarse gain table), per ``DP5Status.cpp``."""
        fw_build = (self.firmware_raw << 8) + self.build
        return self.device_type == DEVICE_DP5 and fw_build >= 0x686 and self.dpp_eco < 0xFF

    @property
    def has_papz(self) -> bool:
        """DP5 Rev Dx configuration K or L supports the PAPZ (pole-zero) command."""
        return self.is_rev_dx and (self.dpp_eco & 0x0F) in (0x0A, 0x0B)

    @property
    def board_revision(self) -> str:
        """Human-readable DP5 Rev Dx board string (D7-6 rev, D5-4 minor, D3-0 config)."""
        if not self.is_rev_dx:
            return "DP5 (pre-Rev Dx gain table)"
        e = self.dpp_eco
        return f"DP5 Rev {chr(ord('D') + ((e >> 6) & 3))}{(e >> 4) & 3} Configuration {chr(ord('A') + (e & 0x0F))}"

    @property
    def input_count_rate(self) -> float:
        """Average fast-channel rate over the real time (cps)."""
        return self.fast_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def output_count_rate(self) -> float:
        """Average slow-channel (spectrum) rate over the real time (cps)."""
        return self.slow_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def dead_time_fraction(self) -> float:
        """1 - OCR/ICR (includes pile-up rejection losses)."""
        return 1.0 - self.slow_count / self.fast_count if self.fast_count > 0 else 0.0

    def summary(self) -> str:
        return (f"{self.device_name} S/N {self.serial_number}  FW {self.firmware}  FPGA {self.fpga}  "
                f"{self.board_revision}  board {self.board_temp_c} C  "
                f"MCA {'ON' if self.mca_enabled else 'off'}  "
                f"fast {self.fast_count}  slow {self.slow_count}  "
                f"real {self.real_time:.3f} s  acc {self.accumulation_time:.3f} s")


def decode_status(b: bytes) -> Status:
    """Decode a 64-byte status block."""
    if len(b) < STATUS_LEN:
        raise ValueError(f"status block needs 64 bytes, got {len(b)}")
    b = bytes(b[:STATUS_LEN])
    fw, fp = b[24], b[25]
    build = (b[37] & 0x0F) if fw > 0x65 else 0
    hv_raw = int.from_bytes(b[30:32], "big", signed=True)
    return Status(
        raw=b,
        fast_count=_u32(b, 0),
        slow_count=_u32(b, 4),
        gp_count=_u32(b, 8),
        accumulation_time=b[12] * 0.001 + int.from_bytes(b[13:16], "little") * 0.1,
        live_time=_u32(b, 16) * 0.001 if (b[39] == 3 and fw >= 0x67) else 0.0,
        real_time=_u32(b, 20) * 0.001,
        firmware=f"{fw >> 4}.{fw & 0x0F:02d}.{build:02d}",
        fpga=f"{fp >> 4}.{fp & 0x0F:02d}",
        firmware_raw=fw,
        fpga_raw=fp,
        build=build,
        serial_number=_u32(b, 26) if b[29] < 128 else 0,
        hv=hv_raw * 0.5,
        detector_temp_k=(((b[32] & 0x0F) << 8) + b[33]) * 0.1,
        board_temp_c=int.from_bytes(b[34:35], "little", signed=True),
        preset_real_time_done=bool(b[35] & 0x80),
        fast_threshold_locked=bool(b[35] & 0x40),
        mca_enabled=bool(b[35] & 0x20),
        preset_count_reached=bool(b[35] & 0x10),
        scope_data_ready=bool(b[35] & 0x04),
        configured=bool(b[35] & 0x02),
        input_offset_locked=not (b[36] & 0x80),
        mcs_done=bool(b[36] & 0x40),
        reboot_flag=bool(b[36] & 0x20) if fw >= 0x65 else False,
        clock_80mhz=bool(b[36] & 0x02),
        fpga_auto_clock=bool(b[36] & 0x01),
        pc5_present=bool(b[38] & 0x80),
        pc5_hv_positive=bool(b[38] & 0x40),
        pc5_8v5=bool(b[38] & 0x20),
        device_type=b[39],
        tec_voltage=(((b[40] & 0x0F) << 8) + b[41]) / 758.5,
        dpp_options=b[42] & 0x0F,
        dpp_eco=b[49],
    )
