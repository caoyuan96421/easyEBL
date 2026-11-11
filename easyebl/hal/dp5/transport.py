"""
dp5.transport - low-level USB transport for the DP5 (pyusb + libusb-1.0).

USB facts (SDK ``DppLibUsb.h`` and the DP5 firmware descriptors):
  * VID 0x10C4, PID 0x842A, configuration 1, interface 0 (vendor class)
  * bulk OUT endpoint 0x02, bulk IN endpoint 0x81, 64-byte packets
  * one request -> one response; the response is read with a large buffer (32 KiB)
  * default timeout 500 ms, diagnostic data 2.5 s
  * a request whose length is an exact multiple of 64 bytes must be followed by a zero-length
    packet (ZLP), otherwise the DP5 keeps waiting for the rest of the transfer and never answers
    (found on hardware: a 120-byte config string -> 128-byte packet -> no ACK). WinUSB only adds
    the ZLP itself when the SHORT_PACKET_TERMINATE pipe policy is set, and libusb never does, so
    ``transact`` sends it explicitly.

On Windows the DP5 must be bound to WinUSB (Amptek's driver or "WinUSB Device" in Device
Manager); libusb-1.0 talks to WinUSB directly, so no extra driver is needed. The libusb DLL is
provided by the ``libusb-package`` pip package. The device can only be opened by one program at a
time - close DPPMCA / Firmware Manager first.

Install:  pip install pyusb libusb-package
"""

from __future__ import annotations

import time

from .protocol import (DP5ProtocolError, build_packet, expected_length, parse_packet, Packet,
                       REQ_DIAGNOSTIC)

VID = 0x10C4
PID = 0x842A
INTERFACE = 0
EP_OUT = 0x02
EP_IN = 0x81
READ_SIZE = 32768
DEFAULT_TIMEOUT_MS = 500
DIAGNOSTIC_TIMEOUT_MS = 2500


def _backend():
    """libusb-1.0 backend; prefers the DLL shipped with libusb-package."""
    try:
        import libusb_package
        return libusb_package.get_libusb1_backend()
    except Exception:
        import usb.backend.libusb1
        return usb.backend.libusb1.get_backend()


def list_devices():
    """Return a list of (bus, address) for attached DP5s."""
    import usb.core
    out = []
    for d in usb.core.find(find_all=True, idVendor=VID, idProduct=PID, backend=_backend()):
        out.append((d.bus, d.address))
    return out


class USBTransport:
    """Blocking request/response transport over USB. Not thread-safe by itself:
    the DP5 class serialises all access on its worker thread."""

    def __init__(self, index: int = 0, read_delay_s: float = 0.0):
        """
        index       which DP5 to open if several are attached (0 = first).
        read_delay_s optional pause between write and read (the C++ SDK uses 50 ms; not needed
                    with libusb because the read simply blocks until the response arrives).
        """
        self.index = index
        self.read_delay_s = read_delay_s
        self.dev = None
        self.max_packet = 64                # bulk OUT wMaxPacketSize (read from the descriptor)

    # -------------------------------------------------------------------------------------
    def open(self):
        import usb.core
        import usb.util
        devs = list(usb.core.find(find_all=True, idVendor=VID, idProduct=PID, backend=_backend()))
        if not devs:
            raise IOError("No DP5 found (USB VID 10C4 / PID 842A). Is it plugged in, powered and "
                          "bound to WinUSB? Is another program (DPPMCA) using it?")
        if self.index >= len(devs):
            raise IOError(f"DP5 index {self.index} requested but only {len(devs)} found")
        dev = devs[self.index]
        try:
            dev.set_configuration()
        except Exception:
            pass                            # WinUSB: already configured, set_configuration unsupported
        usb.util.claim_interface(dev, INTERFACE)
        try:
            intf = dev.get_active_configuration()[(INTERFACE, 0)]
            ep = usb.util.find_descriptor(intf, bEndpointAddress=EP_OUT)
            if ep is not None and ep.wMaxPacketSize:
                self.max_packet = int(ep.wMaxPacketSize)
        except Exception:
            pass
        self.dev = dev
        self.flush()

    def close(self):
        if self.dev is not None:
            import usb.util
            try:
                usb.util.release_interface(self.dev, INTERFACE)
            except Exception:
                pass
            try:
                usb.util.dispose_resources(self.dev)
            except Exception:
                pass
            self.dev = None

    @property
    def is_open(self) -> bool:
        return self.dev is not None

    def flush(self):
        """Discard any stale bytes waiting on the IN endpoint."""
        for _ in range(16):
            try:
                self.dev.read(EP_IN, READ_SIZE, timeout=20)
            except Exception:
                break

    # -------------------------------------------------------------------------------------
    def transact(self, pid1: int, pid2: int, data: bytes = b"", timeout_ms: int | None = None) -> Packet:
        """Send one packet and return the parsed response packet (data or ACK)."""
        if self.dev is None:
            raise IOError("transport not open")
        if timeout_ms is None:
            timeout_ms = DIAGNOSTIC_TIMEOUT_MS if (pid1, pid2) == REQ_DIAGNOSTIC else DEFAULT_TIMEOUT_MS
        pkt = build_packet(pid1, pid2, data)
        self.dev.write(EP_OUT, pkt, timeout=timeout_ms)
        if len(pkt) % self.max_packet == 0:
            self.dev.write(EP_OUT, b"", timeout=timeout_ms)     # ZLP ends the transfer
        if self.read_delay_s:
            time.sleep(self.read_delay_s)
        buf = bytearray()
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            chunk = self.dev.read(EP_IN, READ_SIZE, timeout=remaining)
            buf += bytes(chunk)
            need = expected_length(buf)
            if need is not None and len(buf) >= need:
                return parse_packet(bytes(buf[:need]))
            if time.monotonic() > deadline:
                raise DP5ProtocolError(f"incomplete response ({len(buf)} bytes)")
