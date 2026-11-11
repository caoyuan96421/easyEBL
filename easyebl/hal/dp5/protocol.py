"""
dp5.protocol - Amptek DP5 (FW6) packet format, PID tables and ACK codes.

Every exchange with the DP5 is one request packet followed by one response packet:

    +------+------+------+------+---------+---------+------------------+---------+---------+
    | 0xF5 | 0xFA | PID1 | PID2 | LEN MSB | LEN LSB | DATA (LEN bytes) | CHK MSB | CHK LSB |
    +------+------+------+------+---------+---------+------------------+---------+---------+

* LEN is the number of DATA bytes (big-endian, < 32768).
* CHK is the two's complement of the 16-bit sum of every preceding byte, so that
  ``(sum(packet[:-2]) + CHK) & 0xFFFF == 0``.

Commands that return no data are answered with an ACK packet (PID1 = 0xFF, code in PID2).
Source: Amptek DP5 SDK, ``VC++/.../DP5Protocol.h`` and ``DotNet/CSharp/csDP5/FW6/FW6Packet.cs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

SYNC1 = 0xF5
SYNC2 = 0xFA
HEADER_LEN = 6
CHECKSUM_LEN = 2
MAX_CONFIG_DATA = 512          # max ASCII bytes per configuration packet


# ---- requests: (PID1, PID2), host -> DP5 -----------------------------------------------
REQ_STATUS = (0x01, 0x01)                    # -> 0x80 0x01 (64-byte status)
REQ_SPECTRUM = (0x02, 0x01)                  # -> 0x81 odd
REQ_SPECTRUM_CLEAR = (0x02, 0x02)            # -> 0x81 odd, then clears spectrum
REQ_SPECTRUM_STATUS = (0x02, 0x03)           # -> 0x81 even (spectrum + status)
REQ_SPECTRUM_STATUS_CLEAR = (0x02, 0x04)     # -> 0x81 even, then clears spectrum + status
REQ_SCOPE = (0x03, 0x01)                     # -> 0x82 0x01 / 0x03
REQ_MISC_DATA = (0x03, 0x02)                 # -> 0x82 0x02 (512 bytes user data)
REQ_SCOPE_REARM = (0x03, 0x03)               # -> 0x82 0x01
REQ_DIAGNOSTIC = (0x03, 0x05)                # -> 0x82 0x05 (takes up to 2.5 s)
REQ_HW_DESCRIPTION = (0x03, 0x06)            # -> 0x82 0x06
REQ_SCA = (0x04, 0x01)                       # -> 0x83 0x01
REQ_SCA_LATCH = (0x04, 0x02)
REQ_SCA_LATCH_CLEAR = (0x04, 0x03)
REQ_CONFIG_TEXT = (0x20, 0x02)               # ASCII "CMD=VAL;..." -> ACK
REQ_CONFIG_READBACK = (0x20, 0x03)           # ASCII "CMD=?;..." -> 0x82 0x07
REQ_CLEAR_SPECTRUM = (0xF0, 0x01)            # -> ACK
REQ_ENABLE_MCA = (0xF0, 0x02)                # -> ACK
REQ_DISABLE_MCA = (0xF0, 0x03)               # -> ACK
REQ_ARM_SCOPE = (0xF0, 0x04)                 # -> ACK
REQ_AUTOSET_INPUT_OFFSET = (0xF0, 0x05)      # -> ACK
REQ_AUTOSET_FAST_THRESHOLD = (0xF0, 0x06)    # -> ACK
REQ_WRITE_MISC_DATA = (0xF0, 0x09)           # 512 bytes -> ACK
REQ_CLEAR_GP_COUNTER = (0xF0, 0x10)          # -> ACK
REQ_COMM_TEST = (0xF1, 0x00)                 # -> ACK

# ---- responses -------------------------------------------------------------------------
RSP_STATUS = 0x80
RSP_SPECTRUM = 0x81
RSP_SCOPE_MISC = 0x82
RSP_SCA = 0x83
RSP_ACK = 0xFF

# PID2 values under RSP_SCOPE_MISC
RSP2_SCOPE = 0x01
RSP2_MISC_DATA = 0x02
RSP2_SCOPE_OVERFLOW = 0x03
RSP2_DIAGNOSTIC = 0x05
RSP2_HW_DESCRIPTION = 0x06
RSP2_CONFIG_READBACK = 0x07


class Ack(IntEnum):
    """ACK codes (PID2 of a 0xFF packet)."""
    OK = 0x00
    SYNC_ERROR = 0x01
    PID_ERROR = 0x02
    LEN_ERROR = 0x03
    CHECKSUM_ERROR = 0x04
    BAD_PARAMETER = 0x05
    BAD_HEX_RECORD = 0x06
    UNRECOGNIZED_COMMAND = 0x07
    FPGA_ERROR = 0x08               # FPGA not initialised / not programmed
    CP2201_NOT_FOUND = 0x09
    SCOPE_DATA_NOT_AVAILABLE = 0x0A
    PC5_NOT_PRESENT = 0x0B
    OK_SHARING_REQUEST = 0x0C
    BUSY_OTHER_INTERFACE = 0x0D
    I2C_ERROR = 0x0E
    OK_FPGA_UPLOAD_ADDRESS = 0x0F
    FEATURE_NOT_SUPPORTED = 0x10
    CAL_DATA_NOT_PRESENT = 0x11


ACK_OK_CODES = {Ack.OK, Ack.OK_SHARING_REQUEST, Ack.OK_FPGA_UPLOAD_ADDRESS}


class DP5Error(Exception):
    """Base class for driver errors."""


class DP5ProtocolError(DP5Error):
    """Malformed / unexpected packet."""


class DP5AckError(DP5Error):
    """The DP5 answered with a non-OK ACK."""

    def __init__(self, code: int, data: bytes = b"", request=None):
        self.code = code
        self.data = data
        self.request = request
        try:
            name = Ack(code).name
        except ValueError:
            name = f"0x{code:02X}"
        msg = f"DP5 ACK error {name}"
        if data:
            txt = data.decode("ascii", "replace").strip("\x00 ")
            if txt:
                msg += f": {txt}"          # BAD_PARAMETER echoes the rejected command
        super().__init__(msg)


def checksum(data: bytes) -> int:
    """16-bit two's-complement checksum of ``data``."""
    return (-sum(data)) & 0xFFFF


def build_packet(pid1: int, pid2: int, data: bytes = b"") -> bytes:
    """Encode one request packet."""
    if len(data) > 0x7FFF:
        raise ValueError("packet data too long")
    head = bytes([SYNC1, SYNC2, pid1 & 0xFF, pid2 & 0xFF, len(data) >> 8, len(data) & 0xFF])
    body = head + bytes(data)
    c = checksum(body)
    return body + bytes([c >> 8, c & 0xFF])


@dataclass
class Packet:
    pid1: int
    pid2: int
    data: bytes

    @property
    def is_ack(self) -> bool:
        return self.pid1 == RSP_ACK

    def __repr__(self):
        return f"Packet(0x{self.pid1:02X}, 0x{self.pid2:02X}, {len(self.data)} bytes)"


def expected_length(buf: bytes) -> int | None:
    """Total packet length once the 6-byte header is available, else None."""
    if len(buf) < HEADER_LEN:
        return None
    return HEADER_LEN + ((buf[4] << 8) | buf[5]) + CHECKSUM_LEN


def parse_packet(buf: bytes) -> Packet:
    """Validate and decode one complete response packet."""
    if len(buf) < HEADER_LEN + CHECKSUM_LEN:
        raise DP5ProtocolError(f"short packet ({len(buf)} bytes)")
    if buf[0] != SYNC1 or buf[1] != SYNC2:
        raise DP5ProtocolError(f"bad sync bytes {buf[:2].hex()}")
    if buf[4] & 0x80:
        raise DP5ProtocolError("length field too large")
    n = (buf[4] << 8) | buf[5]
    total = HEADER_LEN + n + CHECKSUM_LEN
    if len(buf) < total:
        raise DP5ProtocolError(f"truncated packet ({len(buf)}/{total} bytes)")
    chk = (buf[total - 2] << 8) | buf[total - 1]
    if (sum(buf[: total - 2]) + chk) & 0xFFFF:
        raise DP5ProtocolError("checksum error")
    return Packet(buf[2], buf[3], bytes(buf[HEADER_LEN: HEADER_LEN + n]))


def spectrum_channels(pid2: int) -> int:
    """Channel count encoded in a spectrum response PID2 (1..12)."""
    if not 1 <= pid2 <= 12:
        raise DP5ProtocolError(f"not a spectrum PID2: 0x{pid2:02X}")
    return 256 << ((pid2 - 1) >> 1)


def spectrum_has_status(pid2: int) -> bool:
    """Even spectrum PID2 values carry a 64-byte status block after the counts."""
    return pid2 % 2 == 0
