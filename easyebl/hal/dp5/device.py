"""
dp5.device - threaded, thread-safe DP5 driver.

Architecture
------------
One worker thread owns the transport (USB). Every public method is safe to call from any
thread (CLI, Qt GUI thread, scripts): it puts a job on a queue and waits for the result
(``concurrent.futures.Future``). The worker executes jobs strictly one at a time, which is
exactly what the DP5's one-request/one-response protocol requires.

When an acquisition is running, the worker also polls the spectrum every ``poll_interval``
seconds between jobs:

* **MCA mode** - request "spectrum + status" (0x02 0x03). Each read is cumulative since the
  last clear; ``latest()`` and ``accumulated()`` are the same spectrum.
* **Delta mode** - request "spectrum + status, then clear" (0x02 0x04). Each read contains only
  the counts since the previous read (with its own real/accumulation time). ``latest()`` is the
  last interval, ``accumulated()`` is the host-side sum since ``start()`` / ``clear()``.

Delta vs MCA is purely the host's choice of request (as in Amptek's own GUIs); the DP5 has no
setting for it. MCA enable/disable (0xF0 0x02 / 0x03) controls whether the DP5 acquires at all.

Listeners registered with ``add_listener(callback)`` are called **on the worker thread** as
``callback(event, payload)`` with events:

    "spectrum"   payload = dict(latest=Spectrum, accumulated=Spectrum)
    "status"     payload = Status (after status())
    "started"    payload = mode
    "stopped"    payload = reason ("user", "preset", "error")
    "config"     payload = dict of settings written
    "error"      payload = exception

GUI code should marshal these to its own thread (e.g. emit a Qt signal).

Example
-------
>>> from dp5 import DP5
>>> with DP5() as dp:
...     print(dp.status().summary())
...     dp.write_config({"MCAC": 2048, "AINP": "POS", "PRET": "OFF"})
...     dp.apply_speed(3)                      # SDD speed preset 1 (fast) .. 5 (best resolution)
...     dp.start(mode="delta", interval=0.5)
...     dp.auto_speed(max_dead=0.4)            # optional: pick speed from measured dead time
...     dp.wait_for_spectra(10)
...     spec = dp.accumulated()
...     dp.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable, Iterable

import numpy as np

from . import protocol as P
from .commands import (COMMANDS, allowed_commands, apply_gain_style, build_config_string,
                       full_readback_list, load_config_file, parse_config_string, remove_commands,
                       split_config)
from .gain import table_for
from .spectrum import Spectrum, decode_spectrum_packet, save_mca
from .status import Status, decode_status

log = logging.getLogger("dp5")

MODES = ("mca", "delta")


class DP5:
    """Thread-safe Amptek DP5 driver.

    Parameters
    ----------
    transport      object with open()/close()/transact(pid1, pid2, data, timeout_ms) -> Packet.
                   Defaults to ``USBTransport()`` (first DP5 on USB).
    poll_interval  seconds between spectrum reads while acquiring (default 0.5 s).
    call_timeout   default seconds a public call waits for the worker (default 10 s).
    """

    def __init__(self, transport=None, poll_interval: float = 0.5, call_timeout: float = 10.0):
        if transport is None:
            from .transport import USBTransport
            transport = USBTransport()
        self._t = transport
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()            # protects the shared state below
        self._listeners: list[Callable] = []
        self.call_timeout = call_timeout

        # shared state (read via properties / getters, written by the worker)
        self._poll_interval = float(poll_interval)
        self._mode = "mca"
        self._running = False
        self._next_poll = 0.0
        self._seq = 0
        self._status: Status | None = None
        self._latest: Spectrum | None = None
        self._accum: Spectrum | None = None
        self._prev_status: Status | None = None
        self._config_cache: dict[str, str] = {}
        self._fpga_ok: bool | None = None
        self._last_error: Exception | None = None
        self.last_preset_skipped: list[str] = []
        self.last_write_skipped: list[str] = []
        self.speed_overrides: dict = {}          # on top of every apply_speed(): {"TPFA": 800} or {1: {"TPFA": 100}}
        self.auto_thresholds: dict[int, dict] = {}  # speed -> {"THFA", "THSL"} from auto_threshold()
        self._speed: int | None = None

    # ===================================================================================
    # lifecycle
    # ===================================================================================
    def open(self) -> "DP5":
        """Open the transport, start the worker thread and read the device status."""
        if self._thread is not None:
            return self
        self._stop_evt.clear()
        ready: Future = Future()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="DP5-worker", daemon=True)
        self._thread.start()
        ready.result(timeout=self.call_timeout)        # re-raises open errors
        return self

    def close(self):
        """Stop acquisition (disables the MCA), stop the worker and release the device."""
        if self._thread is None:
            return
        try:
            if self._running:
                self.stop()
        except Exception as ex:                        # device may already be gone
            log.warning("stop on close failed: %s", ex)
        self._stop_evt.set()
        self._q.put(None)
        self._thread.join(timeout=5)
        self._thread = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def is_open(self) -> bool:
        """True while the worker thread is running (between open() and close())."""
        return self._thread is not None and self._thread.is_alive()

    # ===================================================================================
    # listeners
    # ===================================================================================
    def add_listener(self, cb: Callable[[str, object], None]):
        """Register ``cb(event, payload)``; called on the worker thread."""
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb):
        """Unregister a callback added with add_listener()."""
        with self._lock:
            if cb in self._listeners:
                self._listeners.remove(cb)

    def _emit(self, event, payload=None):
        with self._lock:
            ls = list(self._listeners)
        for cb in ls:
            try:
                cb(event, payload)
            except Exception:
                log.exception("listener raised")

    # ===================================================================================
    # worker
    # ===================================================================================
    def _call(self, fn, *args, timeout=None, **kw):
        """Run ``fn(*args, **kw)`` on the worker thread and return its result."""
        if threading.current_thread() is self._thread:
            return fn(*args, **kw)                     # re-entrant call from a listener / job
        if not self.is_open:
            raise P.DP5Error("DP5 not open")
        fut: Future = Future()
        self._q.put((fn, args, kw, fut))
        return fut.result(timeout=timeout or self.call_timeout)

    def _run(self, ready: Future):
        try:
            self._t.open()
            st = self._w_status()
            self._fpga_ok = self._w_check_fpga()
            log.info("connected: %s (FPGA %s)", st.summary(), "ok" if self._fpga_ok else "NOT PROGRAMMED")
            ready.set_result(True)
        except Exception as ex:
            try:
                self._t.close()
            except Exception:
                pass
            ready.set_exception(ex)
            return

        while not self._stop_evt.is_set():
            with self._lock:
                running, next_poll = self._running, self._next_poll
            wait = max(0.0, next_poll - time.monotonic()) if running else 0.25
            try:
                item = self._q.get(timeout=wait)
            except queue.Empty:
                item = None
            if item is not None:
                fn, args, kw, fut = item
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(fn(*args, **kw))
                    except Exception as ex:
                        fut.set_exception(ex)
            with self._lock:
                due = self._running and time.monotonic() >= self._next_poll
            if due:
                self._w_poll()

        try:
            self._t.close()
        except Exception:
            pass

    # ---- worker-side primitives (only call on the worker thread) ----------------------
    def _w_req(self, req, data=b"", timeout_ms=None) -> P.Packet:
        pkt = self._t.transact(req[0], req[1], data, timeout_ms)
        if pkt.is_ack and pkt.pid2 not in P.ACK_OK_CODES:
            raise P.DP5AckError(pkt.pid2, pkt.data, req)
        return pkt

    def _w_ack(self, req, data=b"", timeout_ms=None):
        pkt = self._w_req(req, data, timeout_ms)
        if not pkt.is_ack:
            raise P.DP5ProtocolError(f"expected ACK for {req}, got {pkt}")

    def _w_status(self) -> Status:
        pkt = self._w_req(P.REQ_STATUS)
        if pkt.pid1 != P.RSP_STATUS:
            raise P.DP5ProtocolError(f"unexpected status response {pkt}")
        st = decode_status(pkt.data)
        with self._lock:
            self._status = st
        return st

    def _w_check_fpga(self) -> bool:
        """The SDK's test: a spectrum request fails (ACK FPGA_ERROR) if the FPGA is unprogrammed."""
        try:
            pkt = self._w_req(P.REQ_SPECTRUM)
            return pkt.pid1 == P.RSP_SPECTRUM
        except P.DP5AckError as ex:
            if ex.code == P.Ack.FPGA_ERROR:
                return False
            raise

    def _w_read_spectrum(self, clear: bool) -> Spectrum:
        pkt = self._w_req(P.REQ_SPECTRUM_STATUS_CLEAR if clear else P.REQ_SPECTRUM_STATUS)
        counts, st = decode_spectrum_packet(pkt)
        sp = Spectrum(counts=counts, status=st, mode="delta" if clear else "mca",
                      real_time=st.real_time, acc_time=st.accumulation_time,
                      fast_count=st.fast_count, slow_count=st.slow_count)
        with self._lock:
            self._status = st
        return sp

    def _w_reset_accum(self):
        with self._lock:
            self._accum = None
            self._latest = None
            self._prev_status = None
            self._seq = 0

    def _w_poll(self):
        with self._lock:
            mode, interval = self._mode, self._poll_interval
            self._next_poll = max(self._next_poll + interval, time.monotonic() + 0.2 * interval)
        try:
            sp = self._w_read_spectrum(clear=(mode == "delta"))
        except Exception as ex:
            log.error("poll failed: %s", ex)
            with self._lock:
                self._last_error = ex
                self._running = False
            self._emit("error", ex)
            self._emit("stopped", "error")
            return
        with self._lock:
            if mode == "delta" and getattr(self, "_delta_prime", False):
                self._delta_prime = False              # baseline read after MCA->delta switch
                return
            self._seq += 1
            sp.seq = self._seq
            if mode == "delta":
                acc = self._accum
                if acc is None or acc.counts.size != sp.counts.size or acc.mode != "delta":
                    acc = Spectrum(np.zeros_like(sp.counts), mode="delta")
                acc.counts = acc.counts + sp.counts
                acc.real_time += sp.real_time
                acc.acc_time += sp.acc_time
                acc.fast_count += sp.fast_count
                acc.slow_count += sp.slow_count
                acc.status, acc.seq, acc.timestamp = sp.status, sp.seq, sp.timestamp
                self._accum = acc
            else:
                # interval quantities from successive cumulative statuses
                prev = self._prev_status
                if prev is not None and sp.status.real_time >= prev.real_time:
                    d_real = sp.status.real_time - prev.real_time
                    d_fast = sp.status.fast_count - prev.fast_count
                    d_slow = sp.status.slow_count - prev.slow_count
                else:
                    d_real, d_fast, d_slow = sp.real_time, sp.fast_count, sp.slow_count
                self._interval_rates = (d_fast / d_real if d_real > 0 else 0.0,
                                        d_slow / d_real if d_real > 0 else 0.0)
                self._accum = sp
            self._prev_status = sp.status
            self._latest = sp
            latest, accum = sp, self._accum.copy()
            mca_on = sp.status.mca_enabled
        self._emit("spectrum", {"latest": latest, "accumulated": accum})
        if not mca_on:                                 # preset reached (or disabled elsewhere)
            with self._lock:
                self._running = False
            self._emit("stopped", "preset")

    # ===================================================================================
    # public API - device information
    # ===================================================================================
    def status(self) -> Status:
        """Request and return a fresh Status (also emits a "status" event)."""
        st = self._call(self._w_status)
        self._emit("status", st)
        return st

    @property
    def cached_status(self) -> Status | None:
        """Last status received (from status() or any spectrum+status read)."""
        with self._lock:
            return self._status

    @property
    def fpga_programmed(self) -> bool | None:
        """True if the FPGA answered a spectrum request at open(); False = load the FPGA image."""
        return self._fpga_ok

    @property
    def gain_table(self) -> tuple:
        """Coarse-gain table for this board (16-step DP5 or 24-step Rev Dx)."""
        return table_for(self.cached_status)

    def request(self, pid1: int, pid2: int, data: bytes = b"", timeout_ms: int | None = None) -> P.Packet:
        """Send an arbitrary packet and return the response (escape hatch for unlisted commands)."""
        return self._call(lambda: self._t.transact(pid1, pid2, data, timeout_ms))

    def diagnostic_data(self) -> bytes:
        """Raw diagnostic block (0x03 0x05 -> 0x82 0x05); takes up to 2.5 s."""
        return self._call(lambda: self._w_req(P.REQ_DIAGNOSTIC).data)

    def misc_data(self) -> bytes:
        """512-byte user 'misc data' area stored in the DP5 (0x03 0x02)."""
        return self._call(lambda: self._w_req(P.REQ_MISC_DATA).data)

    def write_misc_data(self, data: bytes):
        """Store up to 512 bytes of user data in the DP5 (0xF0 0x09)."""
        d = bytes(data)[:512].ljust(512, b"\x00")
        self._call(self._w_ack, P.REQ_WRITE_MISC_DATA, d)

    # ===================================================================================
    # public API - configuration
    # ===================================================================================
    def allowed_commands(self) -> set[str]:
        """Configuration commands valid for this device (SDK rules: no HVSE/TECS without a PC5,
        no CON1/CON2/VOLU/SCTC/PREL on a DP5, PAPZ only on Rev Dx-K/L boards)."""
        st = self.cached_status
        if st is None:
            return set(COMMANDS)
        return allowed_commands(st.device_type, st.pc5_present, st.has_papz)

    def read_config(self, names: Iterable[str] | None = None) -> dict[str, str]:
        """Read settings back from the DP5. ``names=None`` reads the SDK's full list."""
        if names is None:
            st = self.cached_status
            names = full_readback_list(st.device_type, st.pc5_present, st.has_papz) if st else list(COMMANDS)
        names = [n.upper() for n in names]

        def job():
            out: dict[str, str] = {}
            for chunk in split_config("".join(f"{n}=?;" for n in names)):
                pkt = self._w_req(P.REQ_CONFIG_READBACK, chunk.encode("ascii"))
                if not (pkt.pid1 == P.RSP_SCOPE_MISC and pkt.pid2 == P.RSP2_CONFIG_READBACK):
                    raise P.DP5ProtocolError(f"unexpected readback response {pkt}")
                out.update(parse_config_string(pkt.data.decode("ascii", "replace")))
            # The DP5 answers "?" / "??" for commands it cannot read back on this FPGA
            # (e.g. RESC, and CUSP on FPGA 7.07); drop them.
            unsupported = [k for k, v in out.items() if v.strip("?") == ""]
            for k in unsupported:
                out.pop(k)
            if unsupported:
                log.debug("not readable on this device: %s", unsupported)
            with self._lock:
                self._config_cache.update(out)
            return out
        return self._call(job)

    def get(self, name: str) -> str:
        """Read one setting (e.g. ``dp.get("TPEA")`` -> "9.600")."""
        return self.read_config([name]).get(name.upper(), "")

    def set(self, name: str, value):
        """Write one setting (e.g. ``dp.set("AINP", "NEG")``)."""
        self.write_config({name: value})

    def write_config(self, settings, gain_style: str | None = None, filter_device: bool = True) -> dict:
        """Write settings to the DP5.

        settings       dict {"TPEA": 4.0, ...}, list of pairs, or a "CMD=VAL;..." string.
        gain_style     "total" keeps GAIN, "coarse_fine" keeps GAIA/GAIF, None = whichever is present
                       (GAIN wins if both). The DP5 must never receive both styles.
        filter_device  drop commands not valid for this device (HVSE/TECS without PC5, CON1, ...).
        Returns the dict actually sent. Raises DP5AckError (with the rejected command) on a bad
        value. Commands the DP5 refuses with PC5_NOT_PRESENT (they need a PC5 detector supply)
        are skipped and listed in ``self.last_write_skipped``.
        """
        cfg = parse_config_string(build_config_string(settings))
        unknown = [k for k in cfg if k not in COMMANDS]
        if unknown:
            log.warning("commands not in catalogue (sent anyway): %s", unknown)
        if gain_style is not None or ("GAIN" in cfg and ("GAIA" in cfg or "GAIF" in cfg)):
            cfg = apply_gain_style(cfg, None if gain_style is None else gain_style == "coarse_fine")
        if filter_device:
            allowed = self.allowed_commands()
            dropped = [k for k in cfg if k in COMMANDS and k not in allowed]
            if dropped:
                log.info("not sent (not valid for this device): %s", dropped)
            cfg = {k: v for k, v in cfg.items() if k not in dropped}
        if "RESC" in cfg:                              # reset must come first
            cfg = {"RESC": cfg.pop("RESC"), **cfg}
        s = build_config_string(cfg)
        if not s:
            return {}

        def job():
            skipped = []
            for chunk in split_config(s):
                while chunk:
                    try:
                        self._w_ack(P.REQ_CONFIG_TEXT, chunk.encode("ascii"))
                        break
                    except P.DP5AckError as ex:
                        # Commands needing hardware that isn't fitted (PC5 supply) are skipped, not fatal.
                        bad = ex.data.decode("ascii", "replace").split("=")[0].strip("\x00 ;").upper()
                        if ex.code != P.Ack.PC5_NOT_PRESENT or not bad or bad not in cfg:
                            raise
                        skipped.append(f"{bad}={cfg.pop(bad)}")
                        chunk = build_config_string(remove_commands(parse_config_string(chunk), [bad]))
            if skipped:
                log.warning("skipped (needs PC5): %s", skipped)
            self.last_write_skipped = skipped
            with self._lock:
                self._config_cache.update(cfg)
                if "MCAC" in cfg:                      # channel count change invalidates sums
                    self._accum = None
            return cfg
        sent = self._call(job)
        self._emit("config", sent)
        return sent

    def reset_config(self):
        """``RESC=YES`` - restore the DP5 factory default configuration."""
        self.write_config({"RESC": "YES"}, filter_device=False)

    def load_config_file(self, path: str, send: bool = True) -> dict:
        """Load an Amptek configuration file (e.g. 'SDD Default DP5.txt'); optionally send it."""
        cfg = load_config_file(path)
        return self.write_config(cfg) if send else cfg

    # ---- SDD speed presets 1 (fastest) .. 5 (best resolution) --------------------------
    @staticmethod
    def speeds() -> dict[int, dict]:
        """Speed presets 1..5 -> {"config": {...}, "source": ...}. See dp5.presets."""
        from .presets import SPEEDS
        return {k: {"config": dict(v["config"]), "source": v["source"]} for k, v in SPEEDS.items()}

    def apply_speed(self, speed, overrides: dict | None = None, autoset_fast_threshold: bool = False,
                    clear: bool = True, verify: bool = True) -> dict:
        """Apply SDD speed preset 1 (fastest) .. 5 (best resolution).

        Only pulse-processing settings are written (TPEA, TFLA, TPFA, thresholds THSL/THFA,
        BLR, PUR, RTD, clock); gain, polarity, input offset, reset lockout and channel count are
        kept, so peaks stay put. ``speed`` may also be an old name ("fast"=1, "default"=3,
        "high_resolution"=5).

        overrides  extra settings on top for this call, e.g. {"THFA": 25}.
        ``self.speed_overrides`` is applied first, every time: string keys apply to all speeds
        ({"TPFA": 200}), integer keys to one speed ({1: {"TPFA": 100}}). Use it for detector-
        specific settings, e.g. the NI 6363 simulator (0.5 us step rise) is too slow for speed 1's
        50 ns fast peaking: dp.speed_overrides = {1: {"TPFA": 100}}.
        Thresholds found by ``auto_threshold()`` for this speed (``self.auto_thresholds``) are
        applied after speed_overrides, so measured noise thresholds win over the preset values.
        The total GAIN is read before and re-sent after the shaping change (the DP5 re-derives
        its gain when TPEA changes), so peak positions do not move.
        autoset_fast_threshold  afterwards let the DP5 set THFA from the noise (beam off!).
        clear      if acquiring, clear the spectrum so two shapings are not mixed.
        verify     read the written settings back and return them (else return what was sent).
        Commands the DP5 rejects as BAD_PARAMETER are skipped -> ``self.last_preset_skipped``.
        """
        from .presets import resolve, speed_config
        n = resolve(speed)
        cfg = self.speed_settings(n)
        if overrides:
            cfg.update({k.upper(): str(v) for k, v in overrides.items()})
        # The DP5 re-derives its digital gain when TPEA/CLCK change (seen on FW 6.10.02: GAIN read
        # back 9.997 / 15.996 / 11.997 for the same request). Re-send the total gain *after* the
        # shaping settings, as Amptek's configuration files do, so peaks stay where they are.
        keep_gain = "GAIN" not in cfg and "GAIA" not in cfg
        gain_before = self.get("GAIN") if keep_gain else None
        if keep_gain and gain_before:
            cfg["GAIN"] = gain_before
        skipped = []
        sent = {}
        for _ in range(len(cfg) + 1):
            try:
                sent = self.write_config(cfg)
                break
            except P.DP5AckError as ex:
                # BAD_PARAMETER echoes the rejected command ("XXXX=..;"): drop it and retry
                bad = ex.data.decode("ascii", "replace").split("=")[0].strip("\x00 ;").upper()
                if ex.code != P.Ack.BAD_PARAMETER or bad not in cfg:
                    raise
                skipped.append(f"{bad}={cfg.pop(bad)}")
        if skipped:
            log.warning("speed %d: DP5 rejected %s - skipped", n, skipped)
        if keep_gain and gain_before:
            g = self.get("GAIN")
            try:
                if abs(float(g) - float(gain_before)) > 0.005 * float(gain_before):
                    self.write_config({"GAIN": gain_before})          # second attempt, on its own
                    g = self.get("GAIN")
                    if abs(float(g) - float(gain_before)) > 0.005 * float(gain_before):
                        log.warning("speed %d: GAIN %s could not be restored (now %s)", n, gain_before, g)
            except ValueError:
                pass
        self.last_preset_skipped = skipped
        with self._lock:
            self._speed = n
        log.info("applied speed %d: %s", n, sent)
        if autoset_fast_threshold:
            self.autoset_fast_threshold()
        if clear and self.is_running:
            self.clear()
        if verify:
            return self.read_config(list(sent) + (["THFA"] if autoset_fast_threshold else []))
        return sent

    def speed_settings(self, speed) -> dict[str, str]:
        """What apply_speed(speed) writes (before GAIN): the preset, then ``speed_overrides``
        (all-speed string keys, then this speed's integer-key entry), then the thresholds
        ``auto_threshold()`` measured for this speed."""
        from .presets import resolve, speed_config
        n = resolve(speed)
        cfg = speed_config(n)
        ov = self.speed_overrides or {}
        cfg.update({k.upper(): str(v) for k, v in ov.items() if isinstance(k, str)})
        cfg.update({k.upper(): str(v) for k, v in (ov.get(n) or {}).items()})
        cfg.update({k: str(v) for k, v in self.auto_thresholds.get(n, {}).items()})
        return cfg

    def current_speed(self) -> int | None:
        """Speed 1..5 whose peaking/flat-top/fast-peaking times match the device (including
        ``speed_overrides``), else None."""
        from .presets import SPEEDS
        keys = ["TPEA", "TFLA", "TPFA"]
        cur = self.read_config(keys)
        for k in SPEEDS:
            c = self.speed_settings(k)
            try:
                if all(abs(float(cur[x]) - float(c[x])) <= 0.02 * float(c[x]) + 1e-3 for x in keys):
                    return k
            except (KeyError, ValueError):
                return None
        return None

    def measure_dead_time(self, seconds: float = 1.0) -> dict:
        """Measure input rate (fast channel), output rate (slow channel) and dead fraction.

        While acquiring, uses the running polls (no disturbance). Otherwise runs a short
        acquisition of ``seconds`` itself (the DP5 spectrum is cleared and the MCA left disabled).
        Returns dict(icr, ocr, dead, real_time, fast_ok). ``fast_ok`` is False when OCR > ICR,
        i.e. the fast channel misses pulses (THFA too high or TPFA too short for the signal's
        rise time) - then ICR, dead time and pile-up rejection are all unreliable.
        """
        if self.is_running:
            with self._lock:
                start_seq = self._seq
            acc_f = acc_s = acc_t = 0.0
            t0 = time.monotonic()
            last = start_seq
            while time.monotonic() - t0 < max(seconds, 2.5 * self.poll_interval):
                time.sleep(0.02)
                with self._lock:
                    seq, sp = self._seq, self._latest
                    rates = getattr(self, "_interval_rates", None)
                if sp is None or seq == last:
                    continue
                last = seq
                if sp.mode == "delta":
                    acc_f += sp.fast_count
                    acc_s += sp.slow_count
                    acc_t += sp.real_time
                elif rates is not None:
                    dt = self.poll_interval
                    acc_f += rates[0] * dt
                    acc_s += rates[1] * dt
                    acc_t += dt
            if acc_t <= 0:
                raise P.DP5Error("no spectra received while measuring dead time")
            icr, ocr = acc_f / acc_t, acc_s / acc_t
            real = acc_t
        else:
            def job():
                self._w_ack(P.REQ_DISABLE_MCA)
                self._w_read_spectrum(clear=True)
                self._w_ack(P.REQ_ENABLE_MCA)
                time.sleep(seconds)
                sp = self._w_read_spectrum(clear=True)
                self._w_ack(P.REQ_DISABLE_MCA)
                return sp
            sp = self._call(job, timeout=seconds + self.call_timeout)
            real = sp.real_time
            icr, ocr = sp.input_rate, sp.output_rate
        dead = max(0.0, 1.0 - ocr / icr) if icr > 0 else 0.0
        # The fast channel must see every photon for ICR (and hence dead time and pile-up
        # rejection) to be right. OCR > ICR means it is missing pulses: THFA too high, or TPFA
        # too short for the signal's rise time.
        fast_ok = not (ocr > 1.05 * icr + 50)
        if not fast_ok:
            log.warning("fast channel undercounts (ICR %.1f < OCR %.1f kcps): raise TPFA or lower THFA; "
                        "dead time is not reliable", icr / 1e3, ocr / 1e3)
        return {"icr": icr, "ocr": ocr, "dead": dead, "real_time": real, "fast_ok": fast_ok}

    def auto_speed(self, max_dead: float = 0.40, seconds: float = 1.0, apply: bool = True,
                   measurement: dict | None = None) -> dict:
        # (speed_overrides are honoured because the chosen speed is applied with apply_speed)
        """Pick the best-resolution speed whose dead time stays <= ``max_dead``, from a measurement.

        1. Measure ICR / OCR / dead fraction at the current settings (or use ``measurement``).
        2. Calibrate the dead-time model tau = f * (2*TPEA + TFLA) from it:
           dead = 1 - exp(-ICR * tau)  ->  f = -ln(1 - dead) / (ICR * (2*TPEA + TFLA)).
        3. Predict the dead fraction of every speed at this ICR and choose the slowest (best
           resolution) one with predicted dead <= max_dead (speed 1 if none qualifies - then reduce
           the beam current).
        4. Apply it (``apply=True``), clearing the spectrum if acquiring.

        Returns dict(speed, previous_speed, measured={icr, ocr, dead}, factor, predicted={1: d1, ...},
        applied, note).
        """
        from .presets import DEADTIME_FACTOR, SPEEDS, choose_speed, predicted_dead_fraction
        import math
        m = measurement or self.measure_dead_time(seconds)
        cur = self.read_config(["TPEA", "TFLA"])
        width = 2 * float(cur.get("TPEA", 4)) + float(cur.get("TFLA", 0.2))
        factor = DEADTIME_FACTOR
        if m["icr"] > 0 and 0.02 < m["dead"] < 0.98:
            factor = -math.log(1 - m["dead"]) / (m["icr"] * width * 1e-6)
            factor = min(max(factor, 0.5), 3.0)          # keep the model sane
        predicted = {k: predicted_dead_fraction(k, m["icr"], factor) for k in SPEEDS}
        speed = choose_speed(m["icr"], max_dead, factor)
        note = ""
        if m["icr"] <= 0:
            note = "no input counts measured - choosing the best-resolution speed"
            speed = max(SPEEDS)
        elif not m.get("fast_ok", True):
            note = (f"fast channel undercounts (ICR {m['icr'] / 1e3:.1f} < OCR {m['ocr'] / 1e3:.1f} kcps) - "
                    "the dead time is unreliable; run auto_threshold() with the beam off, or raise TPFA "
                    "(dp.speed_overrides)")
        elif predicted[speed] > max_dead:
            note = (f"even speed 1 gives ~{100 * predicted[speed]:.0f} % dead time at "
                    f"{m['icr'] / 1e3:.1f} kcps - reduce the beam current")
        prev = self.current_speed()
        applied = False
        if apply and speed != prev:
            self.apply_speed(speed)
            applied = True
        out = {"speed": speed, "previous_speed": prev, "measured": m, "factor": factor,
               "predicted": predicted, "applied": applied, "note": note}
        log.info("auto_speed: ICR %.1f kcps, dead %.0f %% -> speed %d (%s)", m["icr"] / 1e3,
                 100 * m["dead"], speed, "applied" if applied else "unchanged")
        return out

    # old names (fast/default/high_resolution) still work through apply_speed
    def apply_sdd_preset(self, name, **kw) -> dict:
        """Deprecated alias of apply_speed()."""
        return self.apply_speed(name, **kw)

    @property
    def config_cache(self) -> dict[str, str]:
        """Settings last written or read back (host-side copy)."""
        with self._lock:
            return dict(self._config_cache)

    # ===================================================================================
    # public API - MCA control and spectra
    # ===================================================================================
    def enable_mca(self):
        """0xF0 0x02 - start acquiring on the DP5 (does not start host polling)."""
        self._call(self._w_ack, P.REQ_ENABLE_MCA)

    def disable_mca(self):
        """0xF0 0x03 - stop acquiring on the DP5."""
        self._call(self._w_ack, P.REQ_DISABLE_MCA)

    def clear(self):
        """Clear spectrum, counters and timers on the DP5 (read-and-clear, discarded) and the
        host-side accumulation. Acquisition continues if running."""
        def job():
            self._w_read_spectrum(clear=True)
            self._w_reset_accum()
        self._call(job)

    def read_spectrum(self, clear: bool = False) -> Spectrum:
        """One-shot spectrum + status read (does not touch the polling state)."""
        return self._call(self._w_read_spectrum, clear)

    def autoset_fast_threshold(self, settle: float = 1.0) -> float | None:
        """0xF0 0x06 - let the DP5 set THFA from the noise (run with no source / beam off).

        Waits ``settle`` s, reads the threshold the DP5 found and writes it back as an ordinary
        THFA setting, which ends the autoset. Without the write-back the fast channel stayed
        almost blind afterwards on the bench (FW 6.10.02: ICR 1 kcps instead of 46 kcps until
        THFA was written). Returns the new THFA (``settle=0`` -> just send the command, None).
        """
        self._call(self._w_ack, P.REQ_AUTOSET_FAST_THRESHOLD)
        if not settle:
            return None
        time.sleep(settle)
        v = self.get("THFA")
        try:
            self.set("THFA", v)
            return float(v)
        except (TypeError, ValueError):
            return None

    def auto_threshold(self, speeds="current", max_noise_cps: float = 50.0, slow_start: float | None = None,
                       slow_step: float = 0.25, slow_limit: float = 10.0, seconds: float = 0.5) -> dict:
        """Set the fast and slow thresholds from the noise - run with the BEAM OFF (no X-rays).

        For each speed: (1) the DP5's own fast-threshold autoset (THFA, written back so it
        sticks), then (2) THSL is raised from ``slow_start`` (default: the preset's THSL) in
        ``slow_step`` % steps until noise alone triggers < ``max_noise_cps`` on the slow
        channel (each step measured for ``seconds``). The DP5 has no slow-threshold autoset.

        speeds   "current" (the speed in use, or just the present settings if no speed was
                 applied), "all" (1-5), or a list. Afterwards the original shaping is put back:
                 the speed in use is re-applied (with its new thresholds), or, if the device was
                 not on a preset, its previous shaping settings and gain are restored.
        Results are remembered in ``self.auto_thresholds[speed]`` and re-applied by every later
        ``apply_speed()`` / ``auto_speed()``; ``clear_auto_thresholds()`` forgets them.
        A running acquisition is paused and restarted (cleared) afterwards.
        Returns {speed or None: {"THFA": float, "THSL": float, "noise_cps": float}}.
        """
        from .presets import SHAPING_KEYS, SPEEDS, resolve
        cur = self.current_speed()
        before = self.read_config(list(SHAPING_KEYS) + ["GAIN"]) if cur is None else None
        if speeds == "current":
            todo = [cur]
        elif speeds == "all":
            todo = list(SPEEDS)
        else:
            todo = [resolve(x) for x in (speeds if isinstance(speeds, (list, tuple)) else [speeds])]
        was_running, mode = self.is_running, self.mode
        if was_running:
            self.stop()
        out = {}
        try:
            for n in todo:
                if n is not None and (n != cur or len(todo) > 1):
                    self.apply_speed(n, clear=False, verify=False)
                thfa = self.autoset_fast_threshold()
                th = float(self.get("THSL")) if slow_start is None else float(slow_start)
                while True:
                    self.set("THSL", f"{th:.3f}")
                    r = self.measure_dead_time(seconds)
                    log.info("speed %s THSL %.3f %%: noise %.0f cps", n, th, r["ocr"])
                    if r["ocr"] < max_noise_cps or th + slow_step > slow_limit + 1e-9:
                        break
                    th += slow_step
                if r["ocr"] >= max_noise_cps:
                    log.warning("speed %s: THSL reached %.2f %% with %.0f cps of noise still counted",
                                n, th, r["ocr"])
                res = {"THFA": thfa if thfa is not None else float(self.get("THFA")), "THSL": round(th, 3),
                       "noise_cps": r["ocr"]}
                out[n] = res
                if n is not None:
                    self.auto_thresholds[n] = {"THFA": f"{res['THFA']:.2f}", "THSL": f"{th:.3f}"}
            if cur is not None and todo[-1] != cur:
                self.apply_speed(cur, clear=False, verify=False)
            elif cur is None and todo != [None]:
                self.write_config(before)                     # back to the non-preset shaping
        finally:
            if was_running:
                self.start(mode=mode)
        return out

    def clear_auto_thresholds(self):
        """Forget the thresholds measured by auto_threshold(); speeds use their preset values again."""
        self.auto_thresholds.clear()

    def autoset_input_offset(self):
        """0xF0 0x05 - let the DP5 set the input offset (INOF)."""
        self._call(self._w_ack, P.REQ_AUTOSET_INPUT_OFFSET)

    def clear_gp_counter(self):
        """0xF0 0x10 - reset the general-purpose counter (status bytes 8-11)."""
        self._call(self._w_ack, P.REQ_CLEAR_GP_COUNTER)

    # ---- continuous acquisition ------------------------------------------------------
    def start(self, mode: str | None = None, interval: float | None = None, clear: bool = True):
        """Start acquiring and polling.

        mode      "mca" (cumulative reads) or "delta" (read-and-clear each poll).
        interval  poll period in seconds.
        clear     clear the DP5 spectrum/counters and host accumulation first.
        """
        if mode is not None:
            self.set_mode(mode)
        if interval is not None:
            self.set_poll_interval(interval)

        def job():
            self._w_ack(P.REQ_DISABLE_MCA)
            if clear:
                self._w_read_spectrum(clear=True)
                self._w_reset_accum()
            self._w_ack(P.REQ_ENABLE_MCA)
            with self._lock:
                self._running = True
                self._next_poll = time.monotonic() + self._poll_interval
                m = self._mode
            return m
        m = self._call(job)
        self._emit("started", m)

    def stop(self):
        """Stop polling and disable the MCA. Spectra stay available via latest()/accumulated()."""
        def job():
            with self._lock:
                self._running = False
            self._w_ack(P.REQ_DISABLE_MCA)
        self._call(job)
        self._emit("stopped", "user")

    def set_mode(self, mode: str):
        """Switch between "mca" and "delta" (allowed while running; resets the host sum).

        Switching to delta while acquiring discards the first read-and-clear, which would
        otherwise contain everything accumulated in MCA mode."""
        mode = mode.lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            if mode != self._mode:
                self._mode = mode
                self._accum = None
                self._prev_status = None
                # Switching to delta while acquiring: the first read-and-clear returns everything
                # accumulated so far in MCA mode -> discard it so delta sums start fresh.
                self._delta_prime = (mode == "delta" and self._running)

    def set_poll_interval(self, seconds: float):
        """Set the poll period (min 0.05 s); takes effect from the next poll."""
        with self._lock:
            self._poll_interval = max(0.05, float(seconds))

    @property
    def mode(self) -> str:
        """Current acquisition mode, "mca" or "delta"."""
        with self._lock:
            return self._mode

    @property
    def poll_interval(self) -> float:
        """Seconds between spectrum reads while acquiring."""
        with self._lock:
            return self._poll_interval

    @property
    def is_running(self) -> bool:
        """True while an acquisition started with start() is being polled."""
        with self._lock:
            return self._running

    def latest(self) -> Spectrum | None:
        """Last spectrum read (cumulative in MCA mode, last interval in Delta mode)."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def accumulated(self) -> Spectrum | None:
        """Total since start/clear (MCA: same as latest; Delta: host-side sum of intervals)."""
        with self._lock:
            return self._accum.copy() if self._accum is not None else None

    def interval_rates(self) -> tuple[float, float]:
        """(input cps, output cps) over the most recent poll interval."""
        with self._lock:
            sp = self._latest
            if sp is None:
                return 0.0, 0.0
            if sp.mode == "delta":
                return sp.input_rate, sp.output_rate
            return getattr(self, "_interval_rates", (sp.input_rate, sp.output_rate))

    def wait_for_spectra(self, n: int = 1, timeout: float = 30.0) -> Spectrum | None:
        """Block until ``n`` more polls have completed; returns accumulated()."""
        with self._lock:
            target = self._seq + n
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._seq >= target or not self._running:
                    break
            time.sleep(0.02)
        return self.accumulated()

    def save_mca(self, path: str, spectrum: Spectrum | None = None, description: str = ""):
        """Save ``spectrum`` (default: accumulated()) with the current configuration as .mca."""
        sp = spectrum or self.accumulated()
        if sp is None:
            raise P.DP5Error("no spectrum to save")
        save_mca(path, sp, self.config_cache, description)
