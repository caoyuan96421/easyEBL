"""Pulse-processor (MCA) part of the HAL: the X-ray detector electronics.

Applications (the EDS software) use one interface, :class:`PulseProcessor`,
whatever processes the detector signal:

* :class:`DP5PulseProcessor` – the Amptek DP5 over USB (driver in
  :mod:`easyebl.hal.dp5`), or the driver's mock device (``mock=True``);
* :class:`SimulatedPulseProcessor` – a software digital pulse processor fed
  directly with the photons of the HAL's EDS phantom (simulator backend, no
  analogue signal), with dead time, pile-up rejection, energy resolution,
  Si escape peaks and an LLD, timed on the stream's sample clock.

Acquisition model (what the DP5 does and every implementation follows):

* ``clear()`` resets the spectrum, the counters and the timers;
* ``enable()`` starts counting; with a preset real time
  (:meth:`set_preset_real_time`) the processor **stops by itself** exactly
  when it is reached and reports ``preset_done`` – this is what makes per-pixel
  EDS mapping exact without list mode;
* ``read(clear=True)`` reads and clears atomically.

Ownership: the EDS acquisition and, say, a settings dialog may share one
processor; :meth:`session` gives one owner exclusive use for a while.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass, field
import threading
import time
from typing import Callable

import numpy as np

from .backend import HALError

__all__ = [
    "MCAError",
    "MCABusyError",
    "MCAReading",
    "PulseProcessor",
    "DP5PulseProcessor",
    "SimulatedPulseProcessor",
    "create_pulse_processor",
]


class MCAError(HALError):
    """Pulse-processor error (communication, configuration)."""


class MCABusyError(MCAError):
    """The pulse processor is in use by another owner."""


@dataclass
class MCAReading:
    """Spectrum (optional) + counters of one read.

    ``real_time`` is the time the MCA was enabled (s); ``fast_count`` counts
    every detected photon (input), ``slow_count`` the ones in the spectrum
    (output, after pile-up rejection and the LLD).
    """

    counts: np.ndarray | None
    real_time: float = 0.0
    acc_time: float = 0.0
    fast_count: int = 0
    slow_count: int = 0
    mca_enabled: bool = False
    preset_done: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def icr(self) -> float:
        """Input count rate (cps)."""
        return self.fast_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def ocr(self) -> float:
        """Output count rate (cps)."""
        return self.slow_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def dead_fraction(self) -> float:
        return max(0.0, 1.0 - self.slow_count / self.fast_count) if self.fast_count > 0 else 0.0

    @property
    def live_time(self) -> float:
        """Real time × OCR/ICR (≤ real time: a fast channel that misses pulses – ICR < OCR, e.g. a
        fast threshold set too high – must not make the live time longer than the real time)."""
        if self.fast_count <= 0:
            return self.real_time
        return self.real_time * min(1.0, self.slow_count / self.fast_count)

    @property
    def total(self) -> int:
        return int(self.counts.sum()) if self.counts is not None else 0


class PulseProcessor(ABC):
    """Abstract MCA / digital pulse processor (see module docstring)."""

    name = "MCA"

    def __init__(self):
        self._own_lock = threading.RLock()
        self._owner: str | None = None
        self._owner_count = 0
        self._preset: float | None = None

    # ------------------------------------------------------------- lifecycle
    @abstractmethod
    def open(self) -> "PulseProcessor": ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @abstractmethod
    def info(self) -> dict:
        """Model, serial number, firmware … (for logs and file headers)."""

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    def device(self):
        """The vendor driver object (e.g. :class:`~easyebl.hal.dp5.DP5`), or None."""
        return None

    # ---------------------------------------------------------- configuration
    @abstractmethod
    def configure(self, settings: dict) -> None:
        """Write processor settings (DP5 command names, e.g. ``{"MCAC": 2048}``)."""

    @abstractmethod
    def get_settings(self, names) -> dict: ...

    def set_preset_real_time(self, seconds: float | None) -> None:
        """Stop automatically after *seconds* of real time (None: no preset).  1 ms resolution."""
        if seconds is not None:
            seconds = round(float(seconds), 3)
            if not seconds > 0:
                raise ValueError("preset real time must be > 0")
        if seconds == self._preset:
            return
        self._write_preset(seconds)
        self._preset = seconds

    @property
    def preset_real_time(self) -> float | None:
        return self._preset

    @abstractmethod
    def _write_preset(self, seconds: float | None) -> None: ...

    # -------------------------------------------------------------- counting
    @abstractmethod
    def enable(self) -> None: ...

    @abstractmethod
    def disable(self) -> None: ...

    @abstractmethod
    def read(self, clear: bool = False) -> MCAReading: ...

    @abstractmethod
    def status(self) -> MCAReading:
        """Counters and flags without the spectrum (fast)."""

    def clear(self) -> None:
        self.read(clear=True)

    def wait_done(self, timeout: float, poll_s: float = 0.002, expected_s: float | None = None) -> MCAReading:
        """Wait until the MCA stopped (preset reached).  Returns the last status.

        A DP5 that reaches PRER stops counting and sets "preset real time done" but keeps
        "MCA enabled" set (FW 6.10.02); the MCA is then disabled here, so a following
        read-and-clear does not restart counting."""
        deadline = time.monotonic() + timeout
        if expected_s:
            time.sleep(max(0.0, expected_s - 0.004))
        while True:
            st = self.status()
            if st.preset_done or not st.mca_enabled:
                if st.mca_enabled:
                    self.disable()
                return st
            if time.monotonic() > deadline:
                raise MCAError(f"{self.name}: preset not reached within {timeout:.1f} s")
            time.sleep(poll_s)

    def acquire(
        self,
        real_time_s: float,
        on_update: Callable[[MCAReading], None] | None = None,
        update_s: float = 0.25,
        cancel: threading.Event | None = None,
        timeout_extra_s: float = 5.0,
    ) -> MCAReading:
        """Clear, count for *real_time_s* (preset), return the spectrum.  Calls *on_update* with
        intermediate spectra; *cancel* stops early (the result covers the time counted)."""
        self.set_preset_real_time(real_time_s)
        self.clear()
        self.enable()
        deadline = time.monotonic() + real_time_s + timeout_extra_s
        t_next = time.monotonic() + update_s
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self.disable()
                    break
                time.sleep(min(0.01, max(0.0, t_next - time.monotonic())))
                st = self.status()
                if st.preset_done or not st.mca_enabled:
                    self.disable()
                    break
                if on_update is not None and time.monotonic() >= t_next:
                    on_update(self.read())
                    t_next = time.monotonic() + update_s
                if time.monotonic() > deadline:
                    self.disable()
                    raise MCAError(f"{self.name}: acquisition did not stop at its preset")
        finally:
            pass
        r = self.read()
        if on_update is not None:
            on_update(r)
        return r

    # -------------------------------------------------------------- shaping
    def speeds(self) -> dict:
        """Shaping presets {n: description} (1 = fastest … 5 = best resolution)."""
        from .dp5.presets import SPEEDS, pulse_width_us

        return {k: f"{k}: peaking {float(v['config']['TPEA']):g} µs, pulse ≈ {pulse_width_us(k):.1f} µs"
                for k, v in SPEEDS.items()}

    def apply_speed(self, speed: int) -> None:
        from .dp5.presets import speed_config

        self.configure({k: v for k, v in speed_config(speed).items() if k in ("TPEA", "TFLA")})

    def current_speed(self) -> int | None:
        from .dp5.presets import SPEEDS

        try:
            tpea = float(self.get_settings(["TPEA"])["TPEA"])
        except (KeyError, ValueError, MCAError):
            return None
        for k, v in SPEEDS.items():
            if abs(float(v["config"]["TPEA"]) - tpea) < 0.05:
                return k
        return None

    # -------------------------------------------------------------- sharing
    @contextlib.contextmanager
    def session(self, owner: str):
        """Exclusive use by *owner* (re-entrant for the same owner)."""
        with self._own_lock:
            if self._owner is not None and self._owner != owner:
                raise MCABusyError(f"{self.name} is in use by '{self._owner}'")
            self._owner = owner
            self._owner_count += 1
        try:
            yield self
        finally:
            with self._own_lock:
                self._owner_count -= 1
                if self._owner_count == 0:
                    self._owner = None

    @property
    def owner(self) -> str | None:
        return self._owner


# =================================================================== DP5
class DP5PulseProcessor(PulseProcessor):
    """Amptek DP5 through the vendored driver.  ``mock=True`` uses the driver's simulated DP5."""

    name = "DP5"

    def __init__(self, device=None, mock: bool = False, mock_cps: float = 20e3, call_timeout: float = 10.0):
        super().__init__()
        self._dev = device
        self._own_dev = device is None
        self._mock, self._mock_cps = mock, mock_cps
        self._call_timeout = call_timeout
        self._channels = 0
        self._info: dict = {}

    def open(self) -> "DP5PulseProcessor":
        from .dp5 import DP5

        try:
            if self._dev is None:
                transport = None
                if self._mock:
                    from .dp5.mock import MockTransport

                    transport = MockTransport(cps=self._mock_cps, seed=1)
                self._dev = DP5(transport, call_timeout=self._call_timeout)
            if not self._dev.is_open:
                self._dev.open()
            st = self._dev.status()
            cfg = self._dev.read_config(["MCAC", "PRER", "PRET", "GAIN", "TPEA"])
        except Exception as e:  # noqa: BLE001
            raise MCAError(f"cannot open the DP5: {e}") from e
        self._channels = int(float(cfg.get("MCAC", "1024")))
        p = cfg.get("PRER", "OFF")
        self._preset = None if p in ("OFF", "", "OF") else round(float(p), 3)
        self._info = {"model": st.device_name, "serial": st.serial_number, "firmware": st.firmware,
                      "fpga": st.fpga, "board": st.board_revision, "mock": bool(self._mock)}
        return self

    def close(self) -> None:
        if self._dev is not None and self._own_dev:
            try:
                self._dev.close()
            finally:
                self._dev = None

    @property
    def is_open(self) -> bool:
        return self._dev is not None and self._dev.is_open

    @property
    def device(self):
        return self._dev

    def info(self) -> dict:
        return dict(self._info)

    @property
    def channels(self) -> int:
        return self._channels

    def _stop_polling(self) -> None:
        if self._dev.is_running:  # the driver's own acquisition loop (e.g. its GUI) – not ours
            self._dev.stop()

    def configure(self, settings: dict) -> None:
        self._require()
        self._stop_polling()
        try:
            self._dev.write_config(settings)
        except Exception as e:  # noqa: BLE001
            raise MCAError(f"DP5 configuration failed: {e}") from e
        if "MCAC" in settings:
            self._channels = int(float(self._dev.get("MCAC")))
        if "PRER" in settings:
            p = str(settings["PRER"])
            self._preset = None if p.upper().startswith("OF") else round(float(p), 3)

    def get_settings(self, names) -> dict:
        self._require()
        return self._dev.read_config(list(names))

    def apply_speed(self, speed: int) -> None:
        self._require()
        self._stop_polling()
        try:
            self._dev.apply_speed(speed)
        except Exception as e:  # noqa: BLE001
            raise MCAError(f"DP5 speed preset failed: {e}") from e

    def auto_threshold(self) -> dict:
        """Fast/slow thresholds from the noise – beam OFF (blanked), no X-rays."""
        self._require()
        self._stop_polling()
        return self._dev.auto_threshold()

    def _write_preset(self, seconds):
        self._require()
        self._stop_polling()
        self._dev.write_config({"PRER": "OFF" if seconds is None else f"{seconds:.3f}"})

    def enable(self) -> None:
        self._require()
        self._stop_polling()
        self._dev.enable_mca()

    def disable(self) -> None:
        self._require()
        self._dev.disable_mca()

    def read(self, clear: bool = False) -> MCAReading:
        self._require()
        sp = self._dev.read_spectrum(clear=clear)
        st = sp.status
        return MCAReading(
            counts=np.asarray(sp.counts, dtype=np.int64),
            real_time=float(sp.real_time),
            acc_time=float(sp.acc_time),
            fast_count=int(sp.fast_count),
            slow_count=int(sp.slow_count),
            mca_enabled=bool(st.mca_enabled) if st else False,
            preset_done=bool(st.preset_real_time_done) if st else False,
        )

    def status(self) -> MCAReading:
        self._require()
        st = self._dev.status()
        return MCAReading(None, float(st.real_time), float(st.accumulation_time), int(st.fast_count),
                          int(st.slow_count), bool(st.mca_enabled), bool(st.preset_real_time_done))

    def _require(self):
        if not self.is_open:
            raise MCAError("DP5 not open")


# =================================================================== simulator
@dataclass
class SimMCAParams:
    """Detector + processor physics of :class:`SimulatedPulseProcessor`."""

    channels: int = 2048
    kev_per_channel: float = 0.0102  # the 'true' calibration the user must find …
    offset_kev: float = -0.015  # … with the calibration dialog
    fwhm_mn_ev: float = 132.0
    fano: float = 0.115
    tpea_us: float = 9.6  # peaking time (DP5 speed 3); dead time ≈ 1.1·(2·TPEA + TFLA)
    tfla_us: float = 0.2
    fast_resolving_us: float = 0.3  # photons closer than this are one fast trigger (true pile-up)
    lld_kev: float = 0.12
    escape_fraction: float | None = None  # Si Kα escape probability (None = physical, Reed & Ware)
    seed: int | None = 0


class SimulatedPulseProcessor(PulseProcessor):
    """Software pulse processor fed with photons from an HAL photon source (the EDS phantom).

    The *source* must offer ``add_photon_listener(fn)`` / ``remove_photon_listener(fn)``;
    ``fn(t0_s, dt_s, sample_index, energies_kev, n_samples)`` is called for every stream chunk.
    Time is the stream's sample clock: the processor only "sees time pass" while
    a stream runs (exactly like the real DP5, which only sees photons while the
    AO3 signal plays).
    """

    name = "Simulated DPP"

    def __init__(self, source=None, params: SimMCAParams | None = None):
        super().__init__()
        self.params = params or SimMCAParams()
        self._source = None
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(self.params.seed)
        self._open = False
        self._settings = {"MCAC": str(self.params.channels), "TPEA": f"{self.params.tpea_us:.3f}",
                          "TFLA": f"{self.params.tfla_us:.3f}", "PRER": "OFF", "GAIN": "1.000"}
        self._reset_counters()
        self._last_trig = -1.0
        self._pending: tuple | None = None  # last trigger of a chunk, decided with the next chunk
        self.clock_s = 0.0
        if source is not None:
            self.attach(source)

    # ---------------------------------------------------------- wiring
    def attach(self, source) -> None:
        """Connect to a photon source (an :class:`~easyebl.hal.eds_phantom.EDSPhantomSpecimen`)."""
        if self._source is not None:
            self._source.remove_photon_listener(self._on_photons)
        self._source = source
        if source is not None:
            source.add_photon_listener(self._on_photons)

    @classmethod
    def for_manager(cls, manager, params: SimMCAParams | None = None) -> "SimulatedPulseProcessor":
        """A simulated processor listening to the photons of *manager*'s EDS phantom.

        It listens on the backend (which keeps listeners across phantom rebuilds);
        photons arrive whenever a stream runs with ``phantom_kind == "eds"``.
        """
        return cls(manager.backend, params)

    def _reset_counters(self):
        self._counts = np.zeros(int(self.params.channels), np.int64)
        self._fast = 0
        self._slow = 0
        self._real = 0.0
        self._enabled = False
        self._done = False

    # ---------------------------------------------------------- lifecycle
    def open(self):
        self._open = True
        return self

    def close(self):
        self._open = False
        self.attach(None) if self._source is not None else None

    @property
    def is_open(self) -> bool:
        return self._open

    def info(self) -> dict:
        p = self.params
        return {"model": "simulated DPP", "serial": 0, "firmware": "sim", "fwhm_mn_ev": p.fwhm_mn_ev,
                "true_kev_per_channel": p.kev_per_channel, "true_offset_kev": p.offset_kev}

    @property
    def channels(self) -> int:
        return int(self.params.channels)

    def configure(self, settings: dict) -> None:
        with self._lock:
            for k, v in settings.items():
                k = str(k).upper()
                self._settings[k] = str(v)
                if k == "MCAC":
                    n = int(float(v))
                    if n not in (256, 512, 1024, 2048, 4096, 8192):
                        raise MCAError("MCAC must be 256…8192 (power of two)")
                    # same energy range over the new channel count (like changing MCAC on the DP5)
                    self.params.kev_per_channel *= self.params.channels / n
                    self.params.channels = n
                    self._counts = np.zeros(n, np.int64)
                elif k == "TPEA":
                    self.params.tpea_us = float(v)
                elif k == "TFLA":
                    self.params.tfla_us = float(v)
                elif k == "PRER":
                    s = str(v).upper()
                    self._preset = None if s.startswith("OF") else round(float(v), 3)

    def get_settings(self, names) -> dict:
        with self._lock:
            return {n: self._settings.get(str(n).upper(), "OFF") for n in names}

    def _write_preset(self, seconds):
        with self._lock:
            self._settings["PRER"] = "OFF" if seconds is None else f"{seconds:.3f}"

    # ---------------------------------------------------------- counting
    def enable(self) -> None:
        with self._lock:
            if self._preset is not None and self._real >= self._preset:
                self._done = True
                return
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    def read(self, clear: bool = False) -> MCAReading:
        with self._lock:
            r = MCAReading(self._counts.copy(), self._real, self._real, self._fast, self._slow,
                           self._enabled, self._done)
            if clear:
                en = self._enabled
                self._reset_counters()
                self._enabled = en
        return r

    def status(self) -> MCAReading:
        with self._lock:
            return MCAReading(None, self._real, self._real, self._fast, self._slow, self._enabled, self._done)

    # ---------------------------------------------------------- physics
    @property
    def dead_time_s(self) -> float:
        p = self.params
        return 1.1 * (2 * p.tpea_us + p.tfla_us) * 1e-6

    def _on_photons(self, t0: float, dt: float, idx: np.ndarray, energies: np.ndarray, n_samples: int) -> None:
        """Photon source callback (stream thread): one chunk of the detector signal."""
        with self._lock:
            t_end_chunk = t0 + n_samples * dt
            self.clock_s = t_end_chunk
            if not self._enabled:
                self._pending = None
                self._last_trig = -1.0
                return
            # counting window inside this chunk: [t0, t_stop)
            t_stop = t_end_chunk
            if self._preset is not None:
                t_stop = min(t_stop, t0 + max(0.0, self._preset - self._real))
            times = t0 + idx * dt + self._rng.uniform(0.0, dt, idx.size)
            sel = times < t_stop
            times, e = times[sel], energies[sel]
            order = np.argsort(times, kind="stable")
            times, e = times[order], e[order]
            self._process(times, e, final=t_stop < t_end_chunk)
            self._real += t_stop - t0
            if t_stop < t_end_chunk:  # preset reached inside this chunk: stop like the DP5
                self._enabled = False
                self._done = True

    def _process(self, times: np.ndarray, e: np.ndarray, final: bool) -> None:
        p = self.params
        # fast channel: photons within the fast resolving time merge (true pile-up: energies add)
        if self._pending is not None:
            times = np.concatenate([[self._pending[0]], times])
            e = np.concatenate([[self._pending[1]], e])
            self._pending = None
        if times.size == 0:
            return
        res = p.fast_resolving_us * 1e-6
        new = np.ones(times.size, bool)
        new[1:] = np.diff(times) > res
        grp = np.cumsum(new) - 1
        trig_t = times[new]
        trig_e = np.bincount(grp, weights=e)
        # the last trigger may still pile up with the next chunk: decide it then
        if not final:
            self._pending = (float(trig_t[-1]), float(trig_e[-1]))
            trig_t, trig_e = trig_t[:-1], trig_e[:-1]
        if trig_t.size == 0:
            return
        self._fast += int(trig_t.size)
        # slow channel: pile-up rejection – no other trigger within the shaping time on either side
        tau = self.dead_time_s / 1.1 * 0.5 + 1e-9  # half the pulse length each side
        prev = np.concatenate([[self._last_trig], trig_t[:-1]])
        nxt = np.concatenate([trig_t[1:], [np.inf if final else (self._pending[0] if self._pending else np.inf)]])
        ok = ((trig_t - prev) > tau) & ((nxt - trig_t) > tau)
        self._last_trig = float(trig_t[-1])
        ee = trig_e[ok]
        if ee.size == 0:
            return
        # Si escape peaks
        if p.escape_fraction is None:  # physical (Reed & Ware), energy dependent
            from ..xray.generation import escape_fraction

            pe = escape_fraction(ee)
        else:
            pe = np.where(ee > 1.84, p.escape_fraction, 0.0)
        esc = self._rng.random(ee.size) < pe
        ee = np.where(esc, ee - 1.74, ee)
        # resolution: electronic noise ⊕ Fano
        eps = 3.85e-3
        k = (2.3548 ** 2) * p.fano * eps
        noise2 = max((p.fwhm_mn_ev / 1e3) ** 2 - k * 5.895, 1e-6)
        sig = np.sqrt(noise2 + k * ee) / 2.3548
        ee = ee + self._rng.normal(0.0, 1.0, ee.size) * sig
        ee = ee[ee > p.lld_kev]
        ch = np.floor((ee - p.offset_kev) / p.kev_per_channel).astype(np.int64)
        ch = ch[(ch >= 0) & (ch < self._counts.size)]
        np.add.at(self._counts, ch, 1)
        self._slow += int(ch.size)

    def true_channel(self, energy_kev: float) -> float:
        """Channel where *energy_kev* lands (centre) – for tests."""
        p = self.params
        return (energy_kev - p.offset_kev) / p.kev_per_channel - 0.5


def create_pulse_processor(kind: str = "dp5", manager=None, **kw) -> PulseProcessor:
    """``"dp5"`` (USB), ``"mock"`` (DP5 driver's simulated device) or ``"sim"`` (fed by *manager*'s EDS phantom)."""
    kind = kind.lower()
    if kind == "dp5":
        return DP5PulseProcessor(**kw)
    if kind == "mock":
        return DP5PulseProcessor(mock=True, **kw)
    if kind == "sim":
        if manager is None:
            return SimulatedPulseProcessor(None, **kw)
        return SimulatedPulseProcessor.for_manager(manager, **kw)
    raise ValueError(f"unknown pulse processor '{kind}' (dp5, mock, sim)")
