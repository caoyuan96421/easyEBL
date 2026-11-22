"""Pulse-processor (MCA) part of the HAL: the X-ray detector electronics.

Applications (the EDS software) use one interface, :class:`PulseProcessor`,
whatever processes the detector signal:

* :class:`DP5PulseProcessor` – the Amptek DP5 over USB (driver in
  :mod:`easyebl.hal.dp5`), or the driver's mock device (``mock=True``).

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
        """Fast threshold from the noise – beam OFF (blanked), no X-rays.  The slow threshold
        is not touched (it follows the EDS minimum energy)."""
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


def create_pulse_processor(kind: str = "dp5", manager=None, **kw) -> PulseProcessor:
    """``"dp5"`` (USB) or ``"mock"`` (the DP5 driver's mock device).  *manager* is accepted and ignored."""
    kind = kind.lower()
    if kind == "dp5":
        return DP5PulseProcessor(**kw)
    if kind == "mock":
        return DP5PulseProcessor(mock=True, **kw)
    raise ValueError(f"unknown pulse processor '{kind}' (dp5, mock)")
