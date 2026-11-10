"""The imaging engine: SEM scanning in its own thread, controlled thread-safely.

    engine = ImagingEngine(manager, calibration)       # starts the scan thread (idle)
    engine.run()                                         # live scanning
    engine.set_speed("Slow 1"); engine.shift_beam(2, 0)  # take effect at the next line
    f = engine.wait_frame(complete=True, timeout=5)      # latest full frame
    fut = engine.capture(nx=2560, ny=1920, dwell_s=40e-6)
    photo = fut.result(timeout=600)                      # integrated photo, scan frozen
    engine.close()

Threading model
---------------
* **One scan thread** owns the DAQ handle.  It runs one hardware stream per
  scanning session; all waveform generation and AI processing happen there.
* **Commands** (``run``, ``freeze``, ``update`` …) may be called from any
  thread.  They validate their arguments in the caller's thread (raising
  ``ValueError`` there, without disturbing the stream), store the new desired
  state under a lock and return immediately.  The scan thread picks the
  change up at the next line boundary (≈ ``latency_s`` later at the beam).
* **Outputs** are published, never pushed: the latest :class:`Frame` sits in
  a :class:`FrameSlot` (poll or wait), events go to bounded subscriber
  queues.  No consumer can stall the scan thread or grow its memory.

Robustness
----------
* Lost DAQ connection (USB unplugged, power cycle): the HAL reports
  :class:`DeviceLostError` and reconnects by itself; the engine shows
  ``DISCONNECTED``, keeps the requested mode and settings, and resumes
  scanning as soon as the device is back.  A capture in progress fails (its
  frames would be incomplete); a single frame is rescanned.
* The HAL guarantees blank + park after every stream (normal, abort, error).
* Driver underflow/overflow: the engine doubles ``latency_s`` (up to
  ``max_latency_s``) and restarts the stream automatically, at most
  ``max_recoveries`` times in a row.
* Every wait has a timeout; :meth:`close` aborts and joins the thread.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field, replace
import itertools
import math
import sys
import threading
import time
import traceback

import numpy as np

from ...core.calibration import FieldCalibration, OutOfFieldError
from ...hal.backend import (
    DeviceBusyError,
    DeviceLostError,
    HALError,
    StreamMonitor,
    StreamOverflowError,
    StreamUnderflowError,
)
from ..raster import RasterConfig, RasterPlan
from .assembler import CaptureAccumulator, FrameRecord, ImageAccumulator, StreamAssembler
from .frames import EngineState, EventBus, Frame, FrameSlot
from .program import CaptureRequest, Directive, StreamProgram
from .settings import EngineOptions, ScanSpeed, speed_by_name
from .source import FrameSource

__all__ = ["ImagingEngine", "EngineStatus", "default_scan_config", "SATURATION_WARN"]

#: Warn when more than this fraction of a frame's pixels sit at the video input limit.
SATURATION_WARN = 0.02


def _boost_thread_priority() -> None:
    """Raise the calling thread's OS priority (Windows) – the scan thread feeds the DAQ."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
    except Exception:  # noqa: BLE001
        pass


def default_scan_config(calibration: FieldCalibration, rate: float, channels=("video",)) -> RasterConfig:
    """640×480 at 80 % of the field (room for beam shift), Fast 1."""
    return RasterConfig(
        nx=640,
        ny=480,
        width_um=0.8 * calibration.field_size_um,
        rate=rate,
        dwell_s=1e-6,
        preline_s=20e-6,
        flyback_s=40e-6,
        frame_settle_s=200e-6,
        ai_channels=tuple(channels),
    )


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState
    mode: str
    rate: float
    latency_s: float  # output buffer depth of the current/last stream
    frames_completed: int  # in the current/last stream
    fps_hw: float  # frame rate on the sample clock (exact, jitter-free)
    fps_display: float  # rate at which complete frames were published (wall clock)
    frame_time_s: float  # theoretical time of one frame at the current settings
    rows_done: int
    margin_s: float | None  # smallest output backlog seen (≈ distance to an underflow)
    ai_backlog_max: int
    underruns: int
    recoveries: int
    streams: int  # hardware streams started since the engine was created
    change_latency_s: float | None  # last settings change → first new line acquired
    start_latency_s: float | None  # last start command → first sample acquired
    stalled: bool  # running but no driver progress for > 2 s
    last_error: str | None
    owner_busy: str | None = None
    connected: bool = True  # DAQ reachable
    disconnects: int = 0  # connection losses seen by the engine
    saturation: float = 0.0  # fraction of pixels at the video input limit (last full frame)

    def summary(self) -> str:
        parts = [self.state.value, f"{self.rate / 1e6:g} MS/s", f"buf {self.latency_s * 1e3:.0f} ms"]
        if not self.connected:
            parts.append("DAQ DISCONNECTED")
        if self.saturation > SATURATION_WARN:
            parts.append(f"VIDEO SATURATED {self.saturation:.0%}")
        if self.fps_hw:
            parts.append(f"{self.fps_hw:.2f} fps")
        if self.margin_s is not None:
            parts.append(f"margin {self.margin_s * 1e3:.0f} ms")
        if self.underruns:
            parts.append(f"underruns {self.underruns}")
        if self.last_error and self.state == EngineState.ERROR:
            parts.append(f"error: {self.last_error}")
        return ", ".join(parts)


@dataclass
class _Run:
    """Per-stream bookkeeping (touched only by the scan thread)."""

    rate: float
    acc: ImageAccumulator
    t_start: float = field(default_factory=time.perf_counter)
    kind: str = "live"
    rec: FrameRecord | None = None
    rows_done: int = 0
    scan_row: int | None = None
    frames_completed: int = 0
    t_pub: float = 0.0
    last_hw_end: float | None = None
    fps_hw: float = 0.0
    pub_times: list = field(default_factory=list)
    saturation: float = 0.0  # fraction of pixels at the AI range limit (last complete frame)
    sat_warned: bool = False
    captures: dict = field(default_factory=dict)  # id -> (request, CaptureAccumulator)


class ImagingEngine:
    """Thread-safe SEM imaging controller (see module docstring)."""

    def __init__(
        self,
        manager,
        calibration: FieldCalibration | None = None,
        config: RasterConfig | None = None,
        options: EngineOptions | None = None,
        owner: str = "sem",
        start_thread: bool = True,
    ):
        # accept a DAQManager or a DAQHandle
        self.handle = manager.get_handle(owner) if hasattr(manager, "get_handle") else manager
        self.daq_config = self.handle.config
        self._cal = calibration or FieldCalibration(100.0, self.daq_config.ao_limit_v)
        opts = options or EngineOptions()
        opts.validate()
        self._opts = opts
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        chans = tuple(self.daq_config.ai_channels)[:1]
        rate = self._resolve_rate(opts, len(chans))
        cfg = config or default_scan_config(self._cal, rate, chans)
        self._cfg = self._normalize(cfg.with_(rate=rate), self._cal)
        self._gen = 1
        self._t_change: float | None = None
        self._mode = "idle"
        self._mode_gen = 0
        self._stop = False
        self._freeze = False
        self._stream_gen = 0
        self._capture: CaptureRequest | None = None
        self._cap_ids = itertools.count(1)
        self._latency = opts.latency_s
        self._state = EngineState.IDLE
        self._t_cmd: float | None = None
        # statistics
        self._underruns = 0
        self._disconnects = 0
        self._recoveries = 0
        self._streams = 0
        self._change_latency: float | None = None
        self._start_latency: float | None = None
        self._last_error: str | None = None
        self._monitor = StreamMonitor()
        self._run: _Run | None = None
        self._seq = itertools.count(1)
        self._closing = False
        self.frames = FrameSlot()
        self.events = EventBus()
        self._thread = threading.Thread(target=self._main, name="easyebl-imaging", daemon=True)
        if start_thread:
            self._thread.start()

    # ================================================================ helpers
    def _resolve_rate(self, opts: EngineOptions, n_ai: int) -> float:
        best = self.daq_config.max_scan_rate(n_ai)
        if opts.rate is None:
            return best
        if opts.rate > best * (1 + 1e-9):
            raise ValueError(f"rate {opts.rate:g} S/s exceeds the device limit {best:g} S/s")
        return float(opts.rate)

    def _normalize(self, cfg: RasterConfig, cal: FieldCalibration) -> RasterConfig:
        """Validate *cfg* (raises ValueError / OutOfFieldError) and fix engine-owned fields."""
        cfg = cfg.with_(frames=1)
        unknown = [c for c in cfg.ai_channels if c not in self.daq_config.ai_channels]
        if unknown or not cfg.ai_channels:
            raise ValueError(f"unknown AI channel(s) {unknown}; configured: {list(self.daq_config.ai_channels)}")
        if cfg.rate > self.daq_config.max_ai_rate(len(cfg.ai_channels)) * (1 + 1e-9):
            raise ValueError(
                f"{len(cfg.ai_channels)} AI channels at {cfg.rate:g} S/s exceed the device limit; lower the rate"
            )
        # geometry fits the field and timing is consistent – checked on the exact samples
        # the stream will emit, with no tolerance (DAQmx rejects anything beyond its range)
        FrameSource(RasterPlan(cfg, cal)).check(min(cal.v_full_scale, self.daq_config.ao_limit_v))
        return cfg

    def _check_open(self) -> None:
        if self._closing:
            raise RuntimeError("imaging engine is closed")

    def _set_state(self, st: EngineState, msg: str = "") -> None:
        with self._lock:
            if self._state == st:
                return
            self._state = st
        self.events.emit("state", msg or st.value, state=st.value)

    def _wake(self) -> None:
        self._cv.notify_all()

    # ================================================================ queries
    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    @property
    def config(self) -> RasterConfig:
        with self._lock:
            return self._cfg

    @property
    def options(self) -> EngineOptions:
        with self._lock:
            return self._opts

    @property
    def calibration(self) -> FieldCalibration:
        with self._lock:
            return self._cal

    @property
    def rate(self) -> float:
        with self._lock:
            return self._cfg.rate

    @property
    def is_scanning(self) -> bool:
        return self.state in (
            EngineState.STARTING,
            EngineState.LIVE,
            EngineState.FREEZING,
            EngineState.SINGLE,
            EngineState.CAPTURE,
        )

    def latest(self) -> Frame | None:
        return self.frames.latest()

    def wait_frame(self, after_seq: int | None = None, timeout: float | None = 5.0, complete: bool = False):
        """Next published frame after *after_seq* (default: the current one)."""
        if after_seq is None:
            after_seq = self.frames.seq
        return self.frames.wait(after_seq, timeout, complete)

    def subscribe(self, maxlen: int = 500):
        return self.events.subscribe(maxlen)

    def frame_time_s(self, cfg: RasterConfig | None = None) -> float:
        cfg = cfg or self.config
        return RasterPlan(cfg, self.calibration).frame_time_s

    def status(self) -> EngineStatus:
        with self._lock:
            run, mon = self._run, self._monitor
            st, mode = self._state, self._mode
            cfg = self._cfg
        fps_disp = 0.0
        if run is not None and len(run.pub_times) > 2:
            ts = run.pub_times
            fps_disp = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)
        now = time.perf_counter()
        stalled = bool(mon.running and now - mon.t_update > 2.0)
        return EngineStatus(
            state=st,
            mode=mode,
            rate=cfg.rate,
            latency_s=(mon.latency_s or self._latency),
            frames_completed=run.frames_completed if run else 0,
            fps_hw=run.fps_hw if run else 0.0,
            fps_display=fps_disp,
            frame_time_s=self.frame_time_s(cfg),
            rows_done=run.rows_done if run else 0,
            margin_s=mon.margin_s,
            ai_backlog_max=mon.ai_backlog_max,
            underruns=self._underruns,
            recoveries=self._recoveries,
            streams=self._streams,
            change_latency_s=self._change_latency,
            start_latency_s=self._start_latency,
            stalled=stalled,
            last_error=self._last_error,
            owner_busy=self.handle.manager.active_owner,
            connected=self.handle.manager.connected,
            disconnects=self._disconnects,
            saturation=run.saturation if run else 0.0,
        )

    @property
    def monitor(self) -> StreamMonitor:
        return self._monitor

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait until no stream is running (after stop/freeze); True if it ended in time."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                idle = self._mode == "idle" and self._state not in (
                    EngineState.STARTING, EngineState.LIVE, EngineState.FREEZING,
                    EngineState.SINGLE, EngineState.CAPTURE,
                )
            if idle and not self.handle.is_active:
                return True
            time.sleep(0.01)
        return False

    # =============================================================== commands
    def run(self) -> None:
        """Start (or continue) live scanning."""
        with self._lock:
            self._check_open()
            if self._mode == "live" and not self._stop and not self._freeze:
                return
            self._mode = "live"
            self._mode_gen += 1
            self._stop = self._freeze = False
            self._t_cmd = time.perf_counter()
            self._wake()
        self.events.emit("info", "run")

    def single(self) -> None:
        """Scan one frame with the current settings, then freeze."""
        with self._lock:
            self._check_open()
            self._mode = "single"
            self._mode_gen += 1
            self._stop = self._freeze = False
            self._t_cmd = time.perf_counter()
            self._wake()
        self.events.emit("info", "single frame")

    def freeze(self, immediate: bool = False) -> None:
        """Stop scanning at the end of the current frame (again / *immediate*: right now)."""
        with self._lock:
            if self._mode == "idle":
                return
            if immediate or self._freeze or self._mode != "live":
                pass  # fall through to stop()
            else:
                self._freeze = True
                self._wake()
                st = EngineState.FREEZING
                self._state = st
                self.events.emit("state", "freezing at end of frame", state=st.value)
                return
        self.stop()

    def toggle_run_freeze(self) -> None:
        if self.state in (EngineState.LIVE, EngineState.STARTING):
            self.freeze()
        elif self.state == EngineState.FREEZING:
            self.freeze(immediate=True)
        elif self.state in (EngineState.SINGLE, EngineState.CAPTURE, EngineState.DISCONNECTED):
            self.stop()  # also cancels the automatic resume after a lost connection
        else:
            self.run()

    def stop(self) -> None:
        """Stop scanning immediately (the HAL blanks and parks the beam)."""
        with self._lock:
            self._mode = "idle"
            self._mode_gen += 1
            self._stop = True
            self._freeze = False
            self._wake()
        self.handle.abort()
        self.events.emit("info", "stop")

    def update(self, **changes) -> RasterConfig:
        """Change scan settings (RasterConfig fields).  Validated here; applied at the next line."""
        changes.pop("rate", None)  # the sample clock is an engine option
        changes.pop("frames", None)
        with self._lock:
            self._check_open()
            new = self._normalize(self._cfg.with_(**changes), self._cal)
            if new == self._cfg:
                return new
            if tuple(new.ai_channels) != tuple(self._cfg.ai_channels):
                self._stream_gen += 1  # AI task layout changes: restart the stream
            self._cfg = new
            self._gen += 1
            self._t_change = time.perf_counter()
            self._wake()
        self.events.emit("settings", "settings changed", changes=_jsonable(changes))
        return new

    def set_speed(self, speed) -> RasterConfig:
        """Scan speed: a preset name ("TV", "Slow 2" …), a :class:`ScanSpeed` or a dwell in s."""
        if isinstance(speed, str):
            speed = speed_by_name(speed)
        dwell = speed.dwell_s if isinstance(speed, ScanSpeed) else float(speed)
        return self.update(dwell_s=dwell)

    def set_resolution(self, nx: int, ny: int) -> RasterConfig:
        return self.update(nx=int(nx), ny=int(ny), height_um=None)

    def set_scan_width(self, width_um: float) -> RasterConfig:
        """Scan field width (µm); clamped so the scan fits the deflection field."""
        cfg = self.config
        lo, hi = 1e-6 * self.calibration.field_size_um, float(width_um)
        if self._fits(cfg.with_(width_um=hi)):
            return self.update(width_um=hi)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._fits(cfg.with_(width_um=mid)):
                lo = mid
            else:
                hi = mid
        return self.update(width_um=lo)

    def _fits(self, cfg: RasterConfig) -> bool:
        try:
            cal = self.calibration
            src = FrameSource(RasterPlan(cfg.with_(frames=1), cal))
            # keep a hair of margin when clamping to the edge of the field
            src.check(min(cal.v_full_scale, self.daq_config.ao_limit_v) * (1 - 1e-9))
            return True
        except (ValueError, OutOfFieldError):
            return False

    def clamp_center(self, center_um) -> tuple:
        """Closest beam position to *center_um* (along the move) that keeps the scan in the field."""
        cfg = self.config
        c0 = np.asarray(cfg.center_um, float)
        c1 = np.asarray(center_um, float)
        if self._fits(cfg.with_(center_um=tuple(c1))):
            return (float(c1[0]), float(c1[1]))
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._fits(cfg.with_(center_um=tuple(c0 + (c1 - c0) * mid))):
                lo = mid
            else:
                hi = mid
        c = c0 + (c1 - c0) * lo
        return (float(c[0]), float(c[1]))

    def set_beam_shift(self, x_um: float, y_um: float) -> RasterConfig:
        """Absolute beam shift (scan centre, µm); clamped to the field."""
        return self.update(center_um=self.clamp_center((x_um, y_um)))

    def shift_beam(self, dx_um: float, dy_um: float) -> RasterConfig:
        cx, cy = self.config.center_um
        return self.set_beam_shift(cx + dx_um, cy + dy_um)

    def reset_beam_shift(self) -> RasterConfig:
        return self.set_beam_shift(0.0, 0.0)

    def set_rotation_deg(self, deg: float) -> RasterConfig:
        return self.update(rotation_rad=math.radians(deg))

    def set_calibration(self, calibration: FieldCalibration) -> None:
        """New µm ↔ V calibration (e.g. the SEM magnification changed)."""
        with self._lock:
            cfg = self._normalize(self._cfg, calibration)
            self._cal = calibration
            self._cfg = cfg
            self._gen += 1
            self._t_change = time.perf_counter()
            self._wake()
        self.events.emit("settings", f"calibration: field {calibration.field_size_um:g} µm")

    def set_options(self, **changes) -> EngineOptions:
        """Change engine/hardware options (see :class:`EngineOptions`).

        Sample rate, buffer latency and AI range restart the stream (≈ 0.3 s);
        averaging and the others apply immediately.
        """
        with self._lock:
            self._check_open()
            new = self._opts.with_(**changes)
            restart = False
            rate = self._resolve_rate(new, len(self._cfg.ai_channels))
            if rate != self._cfg.rate:
                self._cfg = self._normalize(self._cfg.with_(rate=rate), self._cal)
                self._gen += 1
                restart = True
            if new.latency_s != self._opts.latency_s:
                self._latency = new.latency_s
                restart = True
            if new.ai_range_v != self._opts.ai_range_v:
                restart = True
            self._opts = new
            if restart:
                self._stream_gen += 1
                self._t_cmd = time.perf_counter()
            self._wake()
        self.events.emit("settings", "options changed" + (" (stream restart)" if restart else ""),
                         changes=_jsonable(changes))
        return new

    def set_phantom(self, output: str | None, noise_v: float | None = None) -> dict:
        """Switch the phantom test chart (HAL dummy-image source) on an aux output, or off (None).

        Changing the routing pauses scanning, switches the HAL and resumes live
        scanning if it was running: the AO task gains or loses a channel, so the
        fastest sample clock changes (USB-6363: 2.0 ↔ 1.54 MS/s) and the clock
        is re-derived.  A noise-only change applies on the fly.  Returns the new
        routing ``{"output", "physical", "noise_v"}``.
        """
        h = self.handle
        if output == h.phantom["output"]:
            h.set_phantom(output, noise_v)
            self.events.emit("settings", f"phantom noise {h.phantom['noise_v']:g} V")
            return h.phantom
        resume = self._pause_for_hardware_change("phantom")
        try:
            h.set_phantom(output, noise_v)
        except Exception:
            if resume:
                self.run()
            raise
        self._rederive_rate()
        ph = h.phantom
        state = (f"phantom on {ph['output']} ({ph['physical'] or 'dummy – no effect'})" if ph["output"]
                 else "phantom off")
        self.events.emit("settings", f"{state}; sample clock {self.rate / 1e6:.4g} MS/s", phantom=ph)
        if resume:
            self.run()
        return ph

    def _pause_for_hardware_change(self, what: str) -> bool:
        """Stop scanning so the HAL may change the stream layout; True if live should resume."""
        with self._lock:
            self._check_open()
            resume = self._mode == "live"
        if self.is_scanning or self.state == EngineState.DISCONNECTED:
            self.stop()
            if not self.wait_idle(5.0):
                raise RuntimeError(f"scanning did not stop – {what} not changed")
        return resume

    def _rederive_rate(self) -> None:
        """The fastest clock depends on the stream layout: re-derive it after a HAL change."""
        with self._lock:
            n_ai = len(self._cfg.ai_channels)
            best = self.daq_config.max_scan_rate(n_ai)
            if self._opts.rate is not None and self._opts.rate > best * (1 + 1e-9):
                self._opts = self._opts.with_(rate=None)  # explicit clock no longer possible: fastest
            rate = self._resolve_rate(self._opts, n_ai)
            if rate != self._cfg.rate:
                self._cfg = self._normalize(self._cfg.with_(rate=rate), self._cal)
                self._gen += 1

    def set_ai_terminal(self, mode: str) -> str:
        """Video input mode (``"DIFF"``, ``"RSE"``, …).  Pauses scanning, switches, resumes."""
        mode = str(mode).upper()
        if mode == self.handle.ai_terminal:
            return mode
        resume = self._pause_for_hardware_change("AI input mode")
        try:
            self.handle.set_ai_terminal(mode)
        finally:
            if resume:
                self.run()
        self.events.emit("settings", f"video input mode {mode}", ai_terminal=mode)
        return mode

    def capture(
        self,
        frames: int = 1,
        config: RasterConfig | None = None,
        speed=None,
        **changes,
    ) -> concurrent.futures.Future:
        """Photo scan: *frames* frames integrated with the given settings, then freeze.

        Returns a ``Future`` resolving to the integrated :class:`Frame` (or
        raising ``RuntimeError`` if the capture was stopped).  Overrides such
        as ``nx=2560, ny=1920`` or ``speed="Photo 1"`` apply to the capture
        only; the live settings are kept.
        """
        if speed is not None:
            sp = speed_by_name(speed) if isinstance(speed, str) else speed
            changes["dwell_s"] = sp.dwell_s if isinstance(sp, ScanSpeed) else float(sp)
        if "nx" in changes or "ny" in changes:
            changes.setdefault("height_um", None)
        changes.pop("rate", None)
        frames = int(frames)
        if not 1 <= frames <= 1024:
            raise ValueError("frames must be 1…1024")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._check_open()
            base = config.with_(rate=self._cfg.rate) if config is not None else self._cfg
            cfg = self._normalize(base.with_(**changes), self._cal)
            old = self._capture
            req = CaptureRequest(cfg, frames, next(self._cap_ids), fut, time.perf_counter())
            self._capture = req
            self._mode = "capture"
            self._mode_gen += 1
            self._stop = self._freeze = False
            self._t_cmd = time.perf_counter()
            self._wake()
        if old is not None and not old.future.done():
            old.future.set_exception(RuntimeError("capture superseded by a new capture"))
        self.events.emit("info", f"capture {cfg.nx}×{cfg.ny}, {cfg.actual_dwell_s * 1e6:g} µs, {frames} frame(s)")
        return fut

    def directive(self) -> Directive:
        """Snapshot of the desired state (called by the stream program, any thread)."""
        with self._lock:
            return Directive(
                gen=self._gen,
                cfg=self._cfg,
                mode=self._mode,
                mode_gen=self._mode_gen,
                stop=self._stop,
                freeze=self._freeze,
                stream_gen=self._stream_gen,
                capture=self._capture,
                restart_on_change=self._opts.restart_on_change,
                chunk_s=self._opts.chunk_s,
                t_change=self._t_change,
                calibration=self._cal,
            )

    # ============================================================== lifecycle
    def close(self, timeout: float = 10.0) -> None:
        """Stop scanning, end the scan thread (joined with *timeout*) and release the handle."""
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._mode = "idle"
            self._stop = True
            self._mode_gen += 1
            self._wake()
        self.handle.abort()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        if self._thread.is_alive():
            self.events.emit("error", "scan thread did not stop in time")
        else:
            self.handle.close()
        self._set_state(EngineState.CLOSED)

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def __enter__(self) -> "ImagingEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ============================================================ scan thread
    def _main(self) -> None:
        _boost_thread_priority()
        gil = self._opts.gil_switch_interval_s
        if gil and sys.getswitchinterval() > gil:
            sys.setswitchinterval(gil)  # the scan thread gets the GIL within ~1 ms of asking
        failures = 0
        while True:
            with self._cv:
                while not self._closing and self._mode == "idle":
                    self._cv.wait(0.5)
                if self._closing:
                    break
            try:
                good = self._stream_once()
                failures = 0 if good else failures
            except DeviceLostError as e:
                failures = 0  # not the engine's fault; wait for the device instead of giving up
                self._wait_for_device(e)
            except (StreamUnderflowError, StreamOverflowError) as e:
                self._underruns += 1
                failures += 1
                self._last_error = str(e)
                with self._lock:
                    active = self._mode != "idle" and not self._closing
                    can = self._opts.auto_recover and failures <= self._opts.max_recoveries
                    if active and can:
                        self._latency = min(self._opts.max_latency_s, self._latency * 2)
                        self._recoveries += 1
                        lat = self._latency
                if active and can:
                    self.events.emit(
                        "warning", f"{type(e).__name__}: restarting with {lat * 1e3:.0f} ms buffer", error=str(e)
                    )
                    continue
                self._fail(e)
            except DeviceBusyError as e:
                self._fail(e, f"device busy: {e}")
            except Exception as e:  # noqa: BLE001 - never let the scan thread die
                failures += 1
                self._fail(e, tb=traceback.format_exc())
        self._finish_captures("engine closed")

    def _wait_for_device(self, e: Exception) -> None:
        """DAQ connection lost: keep the requested mode, wait for the HAL to reconnect."""
        self._disconnects += 1
        self._last_error = f"DAQ connection lost: {e}"
        with self._lock:
            if self._mode == "capture":  # its frames are incomplete: the capture fails, scanning stops
                self._mode = "idle"
                self._mode_gen += 1
        self._finish_captures("DAQ connection lost")
        self._set_state(EngineState.DISCONNECTED, "DAQ connection lost – waiting for the device")
        self.events.emit("warning", "DAQ connection lost – scanning resumes when the device is back", error=str(e))
        mgr = self.handle.manager
        while True:
            with self._lock:
                if self._closing or self._mode == "idle":
                    break
            if mgr.wait_connected(0.25):
                self.events.emit("info", "DAQ connection restored – resuming")
                return
        if not self._closing:
            self._set_state(EngineState.FROZEN if self.frames.latest() is not None else EngineState.IDLE)

    def _fail(self, e: Exception, msg: str | None = None, tb: str | None = None) -> None:
        self._last_error = msg or f"{type(e).__name__}: {e}"
        with self._lock:
            self._mode = "idle"
            self._mode_gen += 1
        self._finish_captures(self._last_error)
        self._set_state(EngineState.ERROR, self._last_error)
        self.events.emit("error", self._last_error, traceback=tb or "")

    def _finish_captures(self, why: str) -> None:
        with self._lock:
            req = self._capture
            self._capture = None
        if req is not None and not req.future.done():
            req.future.set_exception(RuntimeError(f"capture not completed: {why}"))

    def _stream_once(self) -> bool:
        """One hardware stream; returns True if it delivered at least one full frame."""
        with self._lock:
            opts, cfg, cal = self._opts, self._cfg, self._cal
            stream_gen, latency = self._stream_gen, self._latency
            t_cmd = self._t_cmd
        rate = cfg.rate
        chans = tuple(cfg.ai_channels)
        self._set_state(EngineState.STARTING, "starting stream")
        run = _Run(rate=rate, acc=self._carry_accumulator(chans))
        mon = StreamMonitor()
        asm = StreamAssembler(len(chans), self._on_line, self._on_frame)
        dcfg = self.daq_config
        prog = StreamProgram(self, asm, cal, rate, (dcfg.park_x_v, dcfg.park_y_v), stream_gen)
        with self._lock:
            self._run, self._monitor = run, mon
        self._streams += 1
        try:
            self.handle.run_stream(
                rate,
                prog.chunks(),
                ai_channels=chans,
                on_ai=asm.feed,
                timeout=opts.device_timeout_s,
                latency_s=latency,
                monitor=mon,
                ai_range_v=opts.ai_range_v,
            )
        finally:
            if asm.t_first_sample is not None and t_cmd is not None:
                self._start_latency = asm.t_first_sample - t_cmd
        reason = prog.end_reason
        with self._lock:
            if reason in ("frozen", "single_done", "capture_done") and self._mode_gen == prog.last_mode_gen:
                self._mode = "idle"  # the program finished what was asked (no newer command)
            idle = self._mode == "idle"
            capture_pending = self._mode == "capture" and self._capture is not None
        if not capture_pending:
            self._finish_captures("stopped" if reason != "capture_done" else "incomplete data")
        if idle:
            self._set_state(EngineState.FROZEN if self.frames.latest() is not None else EngineState.IDLE)
        return run.frames_completed > 0 or reason == "restart"

    def _carry_accumulator(self, chans) -> ImageAccumulator:
        """Keep the display composite across stream restarts (same geometry continues)."""
        old = self._run.acc if self._run is not None else None
        if old is not None and tuple(old.channels) == tuple(chans):
            return old
        return ImageAccumulator(chans)

    # ---- assembler callbacks (scan thread)
    def _on_line(self, rec: FrameRecord, j: int, row: int, vals: np.ndarray) -> None:
        run = self._run
        cfg = rec.source.cfg
        now = time.perf_counter()
        if rec is not run.rec:
            run.rec = rec
            run.rows_done = 0
            run.acc.ensure(cfg)
            # only changes made while this stream ran measure the change latency
            if rec.t_request is not None and rec.kind == "live" and rec.t_request >= run.t_start:
                lat = now - rec.t_request
                if self._change_latency is None or rec.gen != getattr(self, "_lat_gen", None):
                    self._change_latency = lat
                    self._lat_gen = rec.gen
            run.kind = rec.kind
            with self._lock:
                freezing = self._freeze
            st = {
                "live": EngineState.FREEZING if freezing else EngineState.LIVE,
                "single": EngineState.SINGLE,
                "capture": EngineState.CAPTURE,
            }[rec.kind]
            self._set_state(st)
        n_avg = self._opts.averaging if rec.kind == "live" else 1
        run.acc.add_row(row, vals, n_avg)
        if rec.capture_id is not None:
            ca = self._capture_acc(run, rec)
            if ca is not None:
                ca.add_row(row, vals)
        run.rows_done = j + 1
        run.scan_row = row
        if now - run.t_pub >= 1.0 / self._opts.publish_hz:
            run.t_pub = now
            self._publish(run, rec, complete=False)

    def _capture_acc(self, run: _Run, rec: FrameRecord):
        if rec.capture_id in run.captures:
            return run.captures[rec.capture_id][1]
        with self._lock:
            req = self._capture
        if req is None or req.id != rec.capture_id:
            return None
        cfg = rec.source.cfg
        ca = CaptureAccumulator(tuple(cfg.ai_channels), cfg.ny, cfg.nx, req.frames)
        run.captures[rec.capture_id] = (req, ca)
        return ca

    def _on_frame(self, rec: FrameRecord) -> None:
        run = self._run
        if rec.cut:
            self._publish(run, rec, complete=False)
            return
        run.frames_completed += 1
        t_end = rec.end / run.rate  # sample-clock time: exact, independent of USB/OS jitter
        if run.last_hw_end is not None:
            dt = t_end - run.last_hw_end
            if dt > 0:
                inst = 1.0 / dt
                run.fps_hw = inst if not run.fps_hw else 0.7 * run.fps_hw + 0.3 * inst
        run.last_hw_end = t_end
        f = self._publish(run, rec, complete=True)
        self._check_saturation(run, f)
        run.pub_times.append(time.perf_counter())
        del run.pub_times[:-30]
        if rec.capture_id is not None and rec.capture_id in run.captures:
            req, ca = run.captures[rec.capture_id]
            ca.frames_done += 1
            if ca.frames_done >= req.frames:
                del run.captures[rec.capture_id]
                photo = Frame(
                    images=ca.images(),
                    seq=next(self._seq),
                    frame_index=run.frames_completed - 1,
                    complete=True,
                    rows_done=rec.source.ny,
                    scan_row=None,
                    config=rec.source.cfg,
                    calibration=f.calibration,
                    t_sample=t_end,
                    frames_averaged=ca.frames_done,
                    kind="capture",
                )
                self.frames.publish(photo)
                with self._lock:
                    if self._capture is req:
                        self._capture = None
                if not req.future.done():
                    req.future.set_result(photo)
                self.events.emit("frame", "capture done", seq=photo.seq)

    def _check_saturation(self, run: _Run, f: Frame) -> None:
        """Pixels pinned at the AI range limit mean clipped video – usually wiring or range."""
        rng = self._opts.ai_range_v or self.daq_config.ai_range_v
        img = f.image
        with np.errstate(invalid="ignore"):
            frac = float(np.mean(np.abs(img) >= 0.98 * rng))
        run.saturation = frac
        if frac > SATURATION_WARN and not run.sat_warned:
            run.sat_warned = True
            mode = self.daq_config.ai_terminal
            hint = (f"In {mode} mode both AI+ and AI− must be connected – an unconnected AI− floats and drifts "
                    "into the rail. " if mode == "DIFF" else "")
            self.events.emit(
                "warning",
                f"video input saturated: {frac:.0%} of pixels at the ±{rng:g} V limit. {hint}"
                "Check the video wiring, or change the input mode / range (Advanced… → Hardware).",
                saturation=frac,
            )
        elif frac < SATURATION_WARN / 4:
            run.sat_warned = False

    def _publish(self, run: _Run, rec: FrameRecord, complete: bool) -> Frame:
        cfg = rec.source.cfg
        f = Frame(
            images=run.acc.snapshot(),
            seq=next(self._seq),
            frame_index=run.frames_completed - (1 if complete else 0),
            complete=complete,
            rows_done=run.rows_done,
            scan_row=None if complete else run.scan_row,
            config=cfg,
            calibration=rec.source.plan.cal,
            t_sample=rec.end / run.rate if complete else 0.0,
            frames_averaged=max(1, run.acc.frames_averaged()),
            kind=rec.kind,
        )
        self.frames.publish(f)
        return f


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out
