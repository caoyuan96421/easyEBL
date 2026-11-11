"""
dp5 - Python driver for the Amptek DP5 digital pulse processor (FW6, USB).

    from dp5 import DP5
    with DP5() as dp:                         # first DP5 on USB
        print(dp.status().summary())
        dp.write_config({"MCAC": 1024, "AINP": "POS", "TPEA": 4.0})
        dp.start(mode="delta", interval=0.5)  # or mode="mca"
        ...
        spec = dp.accumulated()               # Spectrum (numpy counts + status)
        dp.stop()

Vendored into easyEBL (``easyebl.hal.dp5``) from the standalone driver in
``Dropbox/Hitachi/EDS test/driver`` (2026-11-10, verified on DP5 S/N 15791,
FW 6.10.02, FPGA 7.07).  Applications use it through the pulse-processor HAL
(:mod:`easyebl.hal.mca`), not directly.  Changes from the standalone copy: the
mock honours ``PRER`` (preset real time) and sets the "preset real time done"
status flag.

Modules
    protocol   packet format, PID tables, ACK codes, exceptions
    transport  USB (pyusb + libusb-1.0, works with the WinUSB driver)
    device     DP5: worker thread + thread-safe API, MCA / Delta acquisition
    status     64-byte status block decoding
    spectrum   Spectrum container, packet decoding, .mca file I/O
    commands   catalogue of all ASCII configuration commands + config-string helpers
    gain       coarse gain tables (DP5, DP5 Rev Dx, ...)
    presets    SDD speed presets 1 (fastest) .. 5 (best resolution), derived from the SDK
    mock       simulated DP5 for testing without hardware
"""

from .protocol import Ack, DP5Error, DP5AckError, DP5ProtocolError
from .status import Status, decode_status
from .spectrum import Spectrum, save_mca, load_mca
from .commands import COMMANDS, Command
from .device import DP5, MODES
from .presets import SPEEDS

__all__ = ["DP5", "MODES", "Status", "Spectrum", "COMMANDS", "Command", "Ack",
           "SPEEDS", "DP5Error", "DP5AckError", "DP5ProtocolError", "decode_status", "save_mca", "load_mca"]
__version__ = "0.1.0"
