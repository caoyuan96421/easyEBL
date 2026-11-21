"""Byte transports: a real RS-232C port (pyserial) or anything with the same shape.

A transport only moves bytes. Line assembly, logging and dispatch live in
:mod:`easyebl.microscope.s3000.channel`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .errors import TransportError

#: Serial settings from the manual (section 1): 9600 baud, 8 data bits, no parity, 1 stop bit.
BAUD_RATE = 9600
DATA_BITS = 8
PARITY = "N"
STOP_BITS = 1


@runtime_checkable
class Transport(Protocol):
    """Minimal byte-stream interface used by :class:`~easyebl.microscope.s3000.channel.LineChannel`."""

    def write(self, data: bytes) -> None:
        """Send all bytes."""

    def read(self, timeout: float) -> bytes:
        """Return whatever bytes arrive within ``timeout`` seconds (``b""`` if none)."""

    def close(self) -> None:
        """Release the underlying resource. Must be safe to call twice."""


class SerialTransport:
    """RS-232C transport using pyserial, fixed at 9600 8N1.

    Flow control is **off** by default because the photographed manual does not
    specify it. Turn on ``rtscts``/``dsrdtr``/``xonxoff`` only after confirming
    the cable and handshake arrangement on your instrument.
    """

    def __init__(
        self,
        port: str,
        *,
        rtscts: bool = False,
        dsrdtr: bool = False,
        xonxoff: bool = False,
        write_timeout: float = 2.0,
    ):
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise TransportError("pyserial is required for SerialTransport: pip install pyserial") from exc
        self._serial_mod = serial
        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                write_timeout=write_timeout,
                rtscts=rtscts,
                dsrdtr=dsrdtr,
                xonxoff=xonxoff,
            )
        except Exception as exc:  # serial.SerialException and OS errors
            raise TransportError(f"Could not open serial port {port!r}: {exc}") from exc
        self.port = port

    def write(self, data: bytes) -> None:
        try:
            self._ser.write(data)
            self._ser.flush()
        except Exception as exc:
            raise TransportError(f"Serial write failed on {self.port}: {exc}") from exc

    def read(self, timeout: float) -> bytes:
        try:
            self._ser.timeout = timeout
            first = self._ser.read(1)
            if not first:
                return b""
            waiting = self._ser.in_waiting
            return first + (self._ser.read(waiting) if waiting else b"")
        except Exception as exc:
            raise TransportError(f"Serial read failed on {self.port}: {exc}") from exc

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # pragma: no cover
            pass


def list_serial_ports() -> list[str]:
    """Return the serial port device names the operating system reports.

    Returns an empty list if pyserial is not installed.
    """
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return []
    return sorted(p.device for p in list_ports.comports())
