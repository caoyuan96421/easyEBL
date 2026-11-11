"""
dp5.mock - a simulated DP5 speaking the real packet protocol, for testing without hardware.

``MockTransport`` implements the same ``open/close/transact`` interface as ``USBTransport``.
It accumulates a Fe-55 (Mn K) spectrum at ~``cps`` counts/s while the MCA is enabled, honours
MCAC / GAIN / THSL / PRET, read-and-clear requests, configuration write/readback and ACKs.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import protocol as P
from .commands import COMMANDS, parse_config_string


class MockTransport:
    def __init__(self, cps: float = 40000.0, seed: int | None = None):
        self.cps = cps
        self.rng = np.random.default_rng(seed)
        self.cfg = {c.name: c.sdd_default for c in COMMANDS.values() if c.sdd_default}
        self.cfg.update({"MCAC": "1024", "GAIN": "20", "AINP": "POS", "PRET": "OFF", "TPEA": "4.000"})
        self._lock = threading.Lock()
        self.is_open = False
        self.reject: set[str] = set()
        self.misc = bytearray(b"mock DP5 misc data".ljust(512, b"\x00"))           # commands to answer with BAD_PARAMETER (tests)
        self._reset_counters()

    def _reset_counters(self):
        self.mca_on = False
        self.counts = np.zeros(8192, np.int64)
        self.fast = self.slow = 0
        self.real = self.acc = 0.0
        self.prer_done = False
        self._t_last = time.monotonic()

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    # ---------------- simulation ----------------
    def _advance(self):
        now = time.monotonic()
        dt = now - self._t_last
        self._t_last = now
        if not self.mca_on or self.prer_done or dt <= 0:
            return
        pret = self.cfg.get("PRET", "OFF")
        if pret not in ("OFF", "OF", "") and self.acc >= float(pret):
            self.mca_on = False
            return
        prer = self.cfg.get("PRER", "OFF")
        if prer not in ("OFF", "OF", ""):
            # like the DP5 (FW 6.10.02, measured): counting stops exactly at the preset and the
            # "preset real time done" bit is set, but the "MCA enabled" bit stays set
            left = float(prer) - self.real
            if left <= 0:
                self.prer_done = True
                return
            if dt > left:
                dt = left
                self.prer_done = True
        nch = int(self.cfg.get("MCAC", "1024"))
        gain = float(self.cfg.get("GAIN", "20"))
        kev_per_ch = 40.0 * 20.0 / gain / nch               # GAIN 20 -> 0-40 keV full scale
        try:   # paralyzable dead time ~1.07*(2*TPEA+TFLA), like the real board
            tau = 1.07 * (2 * float(self.cfg.get("TPEA", "4")) + float(self.cfg.get("TFLA", "0.2"))) * 1e-6
        except ValueError:
            tau = 8.8e-6
        n_in = self.rng.poisson(self.cps * dt)
        n_out = self.rng.poisson(self.cps * np.exp(-self.cps * tau) * dt)
        e = np.where(self.rng.random(n_out) < 0.88, 5.895, 6.490)
        try:   # FWHM: Fano ~118 eV (+) electronic noise falling with peaking time
            tpea = float(self.cfg.get("TPEA", "4"))
        except ValueError:
            tpea = 4.0
        sig = np.sqrt((0.118 / 2.355) ** 2 * e / 5.895 + (0.12 / 2.355) ** 2 / tpea)
        e = e + self.rng.normal(0, 1, n_out) * sig
        tail = self.rng.random(n_out) < 0.03
        e[tail] = self.rng.uniform(0.5, 6.0, tail.sum())
        ch = np.clip((e / kev_per_ch).astype(int), 0, nch - 1)
        thsl = float(self.cfg.get("THSL", "0.976"))
        np.add.at(self.counts, ch[ch > thsl / 100 * nch], 1)
        self.fast += n_in
        self.slow += ch.size
        self.real += dt
        self.acc += dt * 0.97

    def _status(self) -> bytes:
        s = bytearray(64)
        s[0:4] = int(self.fast & 0xFFFFFFFF).to_bytes(4, "little")
        s[4:8] = int(self.slow & 0xFFFFFFFF).to_bytes(4, "little")
        acc_ms = int(self.acc * 1000)
        s[12] = acc_ms % 100
        s[13:16] = (acc_ms // 100).to_bytes(3, "little")
        s[20:24] = int(self.real * 1000).to_bytes(4, "little")
        s[24], s[25] = 0x6A, 0x77
        s[26:30] = (123456).to_bytes(4, "little")
        s[34] = 35
        s[35] = (0x20 if self.mca_on else 0) | (0x80 if getattr(self, "prer_done", False) else 0) | 0x02
        s[36] = 0x02
        s[37] = 2
        s[39] = 0
        s[49] = 0x00
        return bytes(s)

    def _spectrum(self, with_status: bool, clear: bool) -> P.Packet:
        nch = int(self.cfg.get("MCAC", "1024"))
        c = np.clip(self.counts[:nch], 0, 0xFFFFFF).astype(np.uint32)
        data = np.zeros((nch, 3), np.uint8)
        data[:, 0], data[:, 1], data[:, 2] = c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF
        payload = data.tobytes() + (self._status() if with_status else b"")
        pid2 = (2 * (nch.bit_length() - 9) + 1) + (1 if with_status else 0)
        if clear:
            self.counts[:] = 0
            self.fast = self.slow = 0
            self.real = self.acc = 0.0
            self.prer_done = False
        return P.Packet(P.RSP_SPECTRUM, pid2, payload)

    # ---------------- protocol ----------------
    def transact(self, pid1, pid2, data=b"", timeout_ms=None) -> P.Packet:
        with self._lock:
            time.sleep(0.003)
            self._advance()
            ack = lambda code=0, d=b"": P.Packet(P.RSP_ACK, code, d)
            req = (pid1, pid2)
            if req == P.REQ_STATUS:
                return P.Packet(P.RSP_STATUS, 0x01, self._status())
            if pid1 == 0x02 and pid2 in (1, 2, 3, 4):
                return self._spectrum(with_status=pid2 in (3, 4), clear=pid2 in (2, 4))
            if req == P.REQ_ENABLE_MCA:
                self.mca_on = True
                self._t_last = time.monotonic()
                return ack()
            if req == P.REQ_DISABLE_MCA:
                self.mca_on = False
                return ack()
            if req == P.REQ_CLEAR_SPECTRUM:
                self.counts[:] = 0
                return ack()
            if req == P.REQ_MISC_DATA:
                return P.Packet(P.RSP_SCOPE_MISC, P.RSP2_MISC_DATA, bytes(self.misc))
            if req == P.REQ_WRITE_MISC_DATA:
                self.misc = bytearray(data[:512].ljust(512, b"\x00"))
                return ack()
            if req == P.REQ_DIAGNOSTIC:
                return P.Packet(P.RSP_SCOPE_MISC, P.RSP2_DIAGNOSTIC, bytes(range(192)))
            if req == P.REQ_AUTOSET_FAST_THRESHOLD:
                self.cfg["THFA"] = "18.50"
                return ack()
            if req in (P.REQ_AUTOSET_INPUT_OFFSET, P.REQ_CLEAR_GP_COUNTER):
                return ack()
            if req == P.REQ_CONFIG_TEXT:
                cfg = parse_config_string(data.decode("ascii"))
                for k, v in cfg.items():
                    bad_num = False
                    if k in ("TPEA", "TFLA", "GAIN", "THSL", "THFA"):
                        lim = {"TPEA": 102.4, "TFLA": 51.2, "GAIN": 250, "THSL": 24.9, "THFA": 255.93}[k]
                        try:
                            bad_num = not (0 <= float(v) <= lim)
                        except ValueError:
                            bad_num = True
                    if k in ("PAPS", "HVSE", "TECS"):          # no PC5 fitted, like the real DP5
                        return ack(P.Ack.PC5_NOT_PRESENT, f"{k}={v};".encode())
                    if k not in COMMANDS or k in self.reject or bad_num:
                        return ack(P.Ack.BAD_PARAMETER, f"{k}={v};".encode())
                    if k == "RESC":
                        continue
                    self.cfg[k] = v
                return ack()
            if req == P.REQ_CONFIG_READBACK:
                names = [k for k in parse_config_string(data.decode("ascii"))]
                unreadable = {"RESC": "?", "CUSP": "??"}       # like FPGA 7.07
                txt = "".join(f"{k}={unreadable.get(k, self.cfg.get(k, 'OFF'))};" for k in names)
                return P.Packet(P.RSP_SCOPE_MISC, P.RSP2_CONFIG_READBACK, txt.encode())
            return ack(P.Ack.UNRECOGNIZED_COMMAND)
