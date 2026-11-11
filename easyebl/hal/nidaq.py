"""NI-DAQmx implementation of :class:`DAQBackend` (X-series: USB-6341, USB-6363, …).

.. warning::
   Run the HAL test-pattern GUI with the beam blanked/off before first use on a
   new setup and verify every signal on an oscilloscope.

Timing model
------------
* The AO task (X, Y) owns the sample clock (``/DevN/ao/SampleClock``).
* The DO task (blanker + timed marker lines, all on port0) and the AI task
  use that clock as their sample clock, so output sample *k*, DO sample *k*
  and AI sample *k* share one clock edge.
* AO and DO stream with regeneration disabled into an explicitly sized
  buffer.  By default it holds ≥ ``min_buffer_s`` of data (robust, for
  lithography).  A stream may instead ask for ``latency_s`` (e.g. 0.1 s for
  live imaging), which bounds how far the host runs ahead of the beam.
  Input is re-chunked to ⅛ of the buffer, so the buffer stays ≥ ⅞ full and
  AI is read every ⅛ buffer – a steady, low-jitter data flow.  The AI buffer
  (≥ 4 s) is drained into one preallocated array between writes.
* Nothing is queued beyond the buffer size before the clock starts, so a
  stream can never block on its own prefill.
* Every stream ends with a clocked safe tail: blank while holding the last
  position (``tail_hold_s``, covers blanker latency), then park.  An abort
  plays the same tail instead of freezing the lines.  On errors the clock
  master (AO) is stopped first, the tasks are closed, then the static safe
  state is written blank-first.
* Driver underflow/overflow errors surface as :class:`StreamUnderflowError`
  / :class:`StreamOverflowError` *after* the safe state has been restored.

Remaining hardware risk: a buffer underflow (host too slow) stops AO with
the blank line frozen in its last state until the error surfaces.  Keep
callbacks light; a hardware watchdog on the blanker is recommended.

Port0 is driven as a whole by the DO task (bit *n* = ``port0/line n``), so
reserve port0 for hardware-timed lines; put relays etc. on port1/port2/PFI.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Iterable, Sequence
import warnings

import numpy as np

from ..core.waveform import OutputChunk, concat_chunks, rechunk
from .backend import (
    AIConsumer,
    BackendCapabilities,
    DAQBackend,
    DeviceLostError,
    HALError,
    StreamMonitor,
    StreamOverflowError,
    StreamResult,
    StreamUnderflowError,
)
from .config import DAQConfig

__all__ = ["NIDAQBackend"]

#: DAQmx error codes: output ran dry (host too slow) …
_UNDERFLOW_CODES = {-200016, -200018, -200019, -200290, -200621}
#: … and input overwritten before it was read.
_OVERFLOW_CODES = {-200279, -200361, -200010}


#: … and the device went away (USB unplugged, powered off, bus reset).
_LOST_CODES = {-201003, -50405, -88705, -88709, -88710, -201401, -50550}
_LOST_TEXT = ("no longer present", "cannot be accessed", "not present", "device has been removed",
              "device not found", "disconnected")


def _is_lost(e: BaseException) -> bool:
    if isinstance(e, DeviceLostError):
        return True
    code = getattr(e, "error_code", None)
    if code in _LOST_CODES:
        return True
    text = str(e).lower()
    return code is not None and any(t in text for t in _LOST_TEXT)


def _classify_daq_error(e: Exception) -> HALError | None:
    code = getattr(e, "error_code", None)
    if _is_lost(e):
        return DeviceLostError(f"device lost (DAQmx {code}): {e}")
    if code in _UNDERFLOW_CODES:
        return StreamUnderflowError(f"output underflow (DAQmx {code}): host could not keep up – {e}")
    if code in _OVERFLOW_CODES:
        return StreamOverflowError(f"AI overflow (DAQmx {code}): samples lost – {e}")
    return None


def _line_bit(line: str) -> int:
    # "port0/line3" -> 3
    try:
        return int(line.split("line")[1])
    except (IndexError, ValueError) as e:
        raise HALError(f"cannot parse DO line '{line}'") from e


def _port_of(line: str) -> str:
    return line.split("/")[0]


class NIDAQBackend(DAQBackend):
    def __init__(self, config: DAQConfig, process_lock: bool = True):
        super().__init__(config)
        self._nidaqmx = None
        from .devlock import DeviceLock

        # one process per device (see easyebl.hal.devlock); inside the process the manager arbitrates
        self._devlock = DeviceLock(f"NI-{config.device}") if process_lock else None

    # -------------------------------------------------------------- lifecycle
    def open(self) -> None:
        try:
            import nidaqmx  # noqa: F401
            import nidaqmx.system
        except ImportError as e:  # pragma: no cover - depends on install
            raise HALError("nidaqmx package not installed (pip install nidaqmx)") from e
        self._nidaqmx = nidaqmx
        if self._devlock is not None:
            self._devlock.acquire()  # DeviceBusyError if another easyEBL process drives this device
        names = [d.name for d in nidaqmx.system.System.local().devices]
        if self.config.device not in names:
            raise DeviceLostError(f"device '{self.config.device}' not found; available: {names}")
        self._open = True
        try:
            self.safe_state()
        except Exception as e:  # noqa: BLE001
            self._open = False
            mapped = _classify_daq_error(e)
            raise (mapped or e) from e

    def close(self) -> None:
        if self._open:
            try:
                self.safe_state()
            except Exception as e:  # noqa: BLE001 - a lost device cannot be blanked; closing still ends
                if not _is_lost(e):
                    raise
            finally:
                self._open = False
        if self._devlock is not None:
            self._devlock.release()

    # ------------------------------------------------------- connection health
    def is_device_lost(self, exc: BaseException) -> bool:
        return _is_lost(exc)

    def probe(self) -> bool:
        """Device listed *and* answering: one AI sample on the video input (outputs untouched).

        A USB device that has been unplugged stays listed by DAQmx, so a real
        access is needed; reading one AI sample does not disturb AO, DO or the
        external-control lines.
        """
        try:
            import nidaqmx
            import nidaqmx.system
            from nidaqmx.constants import TerminalConfiguration

            names = [d.name for d in nidaqmx.system.System.local().devices]
            if self.config.device not in names:
                return False
            ch = next(iter(self.config.ai_channels.values()), "ai0")
            term = getattr(TerminalConfiguration, self.config.ai_terminal, TerminalConfiguration.DIFF)
            with nidaqmx.Task() as t:
                t.ai_channels.add_ai_voltage_chan(self._phys(ch), terminal_config=term,
                                                  min_val=-self.config.ai_range_v, max_val=self.config.ai_range_v)
                t.read(timeout=1.0)
            return True
        except Exception:  # noqa: BLE001 - any failure means "not usable right now"
            return False

    def recover(self) -> None:
        """After the device came back: reset it (clears stale driver state), then open."""
        self._open = False
        try:
            import nidaqmx.system

            nidaqmx.system.Device(self.config.device).reset_device()
        except Exception:  # noqa: BLE001 - reset is best effort; open() reports real problems
            pass
        self.open()

    def capabilities(self) -> BackendCapabilities:
        c = self.config
        return BackendCapabilities(
            name=f"NI-DAQmx {c.device}",
            max_ao_rate=c.max_ao_rate,
            max_ai_aggregate_rate=c.max_ai_aggregate_rate,
            ai_channels=tuple(c.ai_channels),
            timed_lines=tuple(c.timed_lines),
            static_lines=tuple(c.static_lines),
            simulated=False,
            external_control=bool(c.external_control_lines),
        )

    # ----------------------------------------------------------------- static
    def _phys(self, t: str) -> str:
        return self.config.physical(t)

    def _write_blank_static(self, blank: bool) -> None:
        from nidaqmx.constants import LineGrouping

        with self._nidaqmx.Task() as t:
            t.do_channels.add_do_chan(
                self._phys(self.config.blank_line), line_grouping=LineGrouping.CHAN_PER_LINE
            )
            t.write(self.blank_level(blank), auto_start=True)

    def _write_xy_static(self, x_v: float, y_v: float) -> None:
        lim = self.config.ao_limit_v
        with self._nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(self._phys(self.config.ao_x), min_val=-lim, max_val=lim)
            t.ao_channels.add_ao_voltage_chan(self._phys(self.config.ao_y), min_val=-lim, max_val=lim)
            t.write([float(x_v), float(y_v)], auto_start=True)

    def write_static(self, x_v: float, y_v: float, blank: bool) -> None:
        # Blank *before* moving and move *before* unblanking, so the beam never
        # travels unblanked between the two (slow, software-timed) writes.
        if blank:
            self._write_blank_static(True)
            self._write_xy_static(x_v, y_v)
        else:
            self._write_xy_static(x_v, y_v)
            self._write_blank_static(False)

    def _write_aux(self, physical: str, volts: float) -> None:
        lim = self.config.ao_limit_v
        with self._nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(self._phys(physical), min_val=-lim, max_val=lim)
            t.write(float(volts), auto_start=True)

    def set_static_line(self, name: str, value: bool) -> None:
        nidaqmx = self._nidaqmx
        from nidaqmx.constants import LineGrouping

        line = self.config.static_lines[name]
        with nidaqmx.Task() as t:
            t.do_channels.add_do_chan(self._phys(line), line_grouping=LineGrouping.CHAN_PER_LINE)
            t.write(self.static_level(name, value), auto_start=True)

    def set_external_control(self, on: bool) -> None:
        from nidaqmx.constants import LineGrouping

        levels = self.external_levels(on)
        if not levels:
            raise HALError("no external-control lines configured")
        with self._nidaqmx.Task() as t:
            for wire in levels:
                t.do_channels.add_do_chan(self._phys(wire), line_grouping=LineGrouping.CHAN_PER_LINE)
            # all wires in one task, one write: they switch together, each with its own polarity
            vals = list(levels.values())
            t.write(vals if len(vals) > 1 else vals[0], auto_start=True)

    # ----------------------------------------------------------------- stream
    def _do_words(self, c: OutputChunk) -> np.ndarray:
        cfg = self.config
        words = np.zeros(len(c), dtype=np.uint32)
        lvl = c.blank if cfg.blank_active_high else ~c.blank
        words |= lvl.astype(np.uint32) << _line_bit(cfg.blank_line)
        for name, arr in c.markers.items():
            words |= arr.astype(np.uint32) << _line_bit(cfg.timed_lines[name])
        return words

    def _safe_tail(self, last_xy, n_park: int, rate: float) -> OutputChunk:
        """Blank while holding the last position, then park (still blanked)."""
        cfg = self.config
        n_hold = max(1, int(round(cfg.tail_hold_s * rate)))
        mk = {k: False for k in cfg.timed_lines}
        hold = OutputChunk.constant(n_hold, last_xy[0], last_xy[1], True, mk)
        park = OutputChunk.constant(max(1, n_park), cfg.park_x_v, cfg.park_y_v, True, mk)
        return concat_chunks([hold, park])

    def buffer_samples(self, rate: float, latency_s: float | None) -> int:
        """Output buffer depth (samples) used for a stream at *rate* with *latency_s*."""
        cfg = self.config
        if latency_s is None:
            return int(max(cfg.buffer_samples, cfg.min_buffer_s * rate))
        return int(max(cfg.min_latency_samples, round(latency_s * rate)))

    def _set_usb_xfer(self, chans, attr: str) -> None:
        n = self.config.usb_xfer_req_size
        if not n:
            return
        for ch in chans:
            try:
                setattr(ch, attr, int(n))
            except Exception as e:  # noqa: BLE001 - property not supported on this device
                warnings.warn(f"cannot set {attr}={n}: {e}", RuntimeWarning, stacklevel=4)
                return

    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        abort: threading.Event | None = None,
        progress: Callable[[int], None] | None = None,
        latency_s: float | None = None,
        monitor: StreamMonitor | None = None,
        ai_range_v: float | None = None,
    ) -> StreamResult:
        try:
            return self._run_stream(
                rate, chunks, ai_channels, on_ai, abort, progress, latency_s, monitor, ai_range_v
            )
        except HALError:
            raise
        except Exception as e:  # noqa: BLE001 - classify driver errors
            mapped = _classify_daq_error(e)
            if mapped is None:
                raise
            raise mapped from e

    def _run_stream(self, rate, chunks, ai_channels, on_ai, abort, progress, latency_s, monitor, ai_range_v):
        nidaqmx = self._nidaqmx
        from nidaqmx.constants import AcquisitionType, LineGrouping, RegenerationMode, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalSingleChannelWriter

        cfg = self.config
        dev = cfg.device
        phantom = self.phantom_source()
        eds = self.eds_phantom()  # photons (+ SDD preamp signal when sdd_physical)
        sdd_phys = cfg.sdd_physical if eds is not None else None
        buf = self.buffer_samples(rate, latency_s)
        # write granularity: small pieces keep the buffer topped up (≥ 7/8 full), so the
        # latency stays ≈ buf / rate while the margin against host stalls stays large,
        # and AI arrives in small, regular blocks (steady frame delivery)
        piece = max(1024, buf // 8)
        lim = cfg.ao_limit_v
        ai_lim = cfg.ai_range_v if ai_range_v is None else float(ai_range_v)
        n_ai = len(ai_channels)
        clock = f"/{dev}/ao/SampleClock"
        term = getattr(TerminalConfiguration, cfg.ai_terminal, TerminalConfiguration.DIFF)
        n_tail_park = max(1000, int(0.01 * rate))
        n_hold = max(1, int(round(cfg.tail_hold_s * rate)))
        write_timeout = max(2.0, 4.0 * piece / rate)
        ai_buf_n = int(max(4 * buf, rate * 4))
        mon = monitor if monitor is not None else StreamMonitor()
        mon.rate, mon.buffer_samples, mon.piece_samples = rate, buf, piece
        mon.ai_buffer_samples = ai_buf_n if n_ai else 0
        mon.written = mon.generated = mon.ai_read = mon.writes = mon.ai_backlog_max = 0
        mon.out_fill_min = None
        mon.running = False

        tasks: list = []  # creation order: ao, do, ai
        ao = do = ai = None
        st = {"written": 0, "queued": 0, "ai_read": 0, "started": False, "n_real": 0}
        aborted = False
        collected = []
        info: dict = {"buffer_samples": buf, "piece_samples": piece}
        last_xy = (cfg.park_x_v, cfg.park_y_v)
        t0 = time.perf_counter()
        mon.t_start = mon.t_update = t0
        it = rechunk(chunks, piece)

        try:
            ao = nidaqmx.Task("easyebl_ao")
            tasks.append(ao)
            do = nidaqmx.Task("easyebl_do")
            tasks.append(do)
            if n_ai:
                ai = nidaqmx.Task("easyebl_ai")
                tasks.append(ai)

            ao.ao_channels.add_ao_voltage_chan(self._phys(cfg.ao_x), min_val=-lim, max_val=lim)
            ao.ao_channels.add_ao_voltage_chan(self._phys(cfg.ao_y), min_val=-lim, max_val=lim)
            if phantom is not None:  # synthetic video on an aux output, same clock as X/Y
                ao.ao_channels.add_ao_voltage_chan(self._phys(cfg.phantom_physical), min_val=-lim, max_val=lim)
            if sdd_phys:  # simulated SDD preamp output on the other aux output
                ao.ao_channels.add_ao_voltage_chan(self._phys(sdd_phys), min_val=-lim, max_val=lim)
            ao.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buf)
            ao.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            ao.out_stream.output_buf_size = buf
            self._set_usb_xfer(ao.ao_channels, "ao_usb_xfer_req_size")
            actual = float(ao.timing.samp_clk_rate)
            info["actual_rate"] = actual
            if abs(actual - rate) > 1e-6 * rate:
                warnings.warn(
                    f"sample clock coerced from {rate:g} to {actual:g} S/s "
                    f"({(actual / rate - 1) * 100:+.4f}% timing/dose error) – choose a rate dividing 100 MHz",
                    RuntimeWarning,
                    stacklevel=4,
                )

            port = _port_of(cfg.blank_line)
            do.do_channels.add_do_chan(f"{dev}/{port}", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
            do.timing.cfg_samp_clk_timing(
                rate, source=clock, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buf
            )
            do.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            do.out_stream.output_buf_size = buf
            self._set_usb_xfer(do.do_channels, "do_usb_xfer_req_size")

            reader = None
            ai_flat = None
            ai_block = 0
            if ai is not None:
                for name in ai_channels:
                    ai.ai_channels.add_ai_voltage_chan(
                        self._phys(cfg.ai_channels[name]),
                        terminal_config=term,
                        min_val=-ai_lim,
                        max_val=ai_lim,
                    )
                ai.timing.cfg_samp_clk_timing(
                    rate, source=clock, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=ai_buf_n
                )
                self._set_usb_xfer(ai.ai_channels, "ai_usb_xfer_req_size")
                reader = AnalogMultiChannelReader(ai.in_stream)
                # one preallocated read buffer, reused for every read (no allocation churn)
                ai_block = max(2 * piece, 65536)
                ai_flat = np.empty(n_ai * ai_block, dtype=np.float64)

            ao_w = AnalogMultiChannelWriter(ao.out_stream, auto_start=False)
            do_w = DigitalSingleChannelWriter(do.out_stream, auto_start=False)

            def start_tasks():
                for t in (ai, do, ao):  # slaves first: they wait for the AO sample clock
                    if t is not None:
                        t.start()
                st["started"] = True
                mon.running = True

            def write(c: OutputChunk):
                nonlocal last_xy
                n = len(c)
                if not n:
                    return
                # never queue more than the buffer holds before the clock runs: a blocking
                # write into a full, stopped buffer could only end in a timeout
                if not st["started"] and st["queued"] + n > buf:
                    start_tasks()
                rows = [c.x, c.y]
                if phantom is not None:
                    rows.append(phantom.output(c.x, c.y, c.blank))
                if sdd_phys:
                    rows.append(eds.sdd_output(c.x, c.y, c.blank, rate))
                elif eds is not None:
                    eds.photons(c.x, c.y, c.blank, rate)  # simulated pulse processor only
                xy = np.vstack(rows)
                # the manager has rejected anything beyond ±limit (1e-9 relative tolerance); clip the
                # rounding residue so DAQmx (which checks its range exactly) never sees it
                np.clip(xy, -lim, lim, out=xy)
                ao_w.write_many_sample(xy, timeout=write_timeout)
                do_w.write_many_sample_port_uint32(self._do_words(c), timeout=write_timeout)
                st["queued"] += n
                last_xy = (float(c.x[-1]), float(c.y[-1]))
                mon.writes += 1

            def deliver(block: np.ndarray):
                keep = max(0, min(block.shape[1], st["n_real"] - st["ai_read"]))
                if keep:
                    part = block[:, :keep]
                    if on_ai is not None:
                        on_ai(part)  # valid during the call only (the buffer is reused)
                    else:
                        collected.append(part.copy())
                st["ai_read"] += block.shape[1]
                mon.ai_read = st["ai_read"]

            def drain():
                if reader is None or not st["started"]:
                    return
                avail = int(ai.in_stream.avail_samp_per_chan)
                if avail > mon.ai_backlog_max:
                    mon.ai_backlog_max = avail
                while avail > 0:
                    n = min(avail, ai_block)
                    view = ai_flat[: n_ai * n].reshape(n_ai, n)
                    got = reader.read_many_sample(view, number_of_samples_per_channel=n, timeout=1.0)
                    got = n if got is None else int(got)
                    deliver(view[:, :got])
                    avail -= n

            def poll():
                if not st["started"]:
                    return
                gen = int(ao.out_stream.total_samp_per_chan_generated)
                mon.generated = gen
                fill = st["queued"] - gen
                if mon.out_fill_min is None or fill < mon.out_fill_min:
                    mon.out_fill_min = fill
                mon.t_update = time.perf_counter()

            # ---- prefill (all but one piece of the buffer) before starting the clock
            exhausted = False
            while st["queued"] + piece <= buf:
                try:
                    c = next(it)
                except StopIteration:
                    exhausted = True
                    break
                write(c)
                st["written"] += len(c)
                st["n_real"] = st["written"]
            mon.written = st["written"]
            if exhausted:
                write(self._safe_tail(last_xy, n_tail_park, rate))
            if not st["started"]:
                start_tasks()

            # ---- stream
            while not exhausted:
                if abort is not None and abort.is_set():
                    aborted = True
                    # controlled stop: blank at the current position and park, played by the clock
                    write(self._safe_tail(last_xy, n_tail_park, rate))
                    break
                try:
                    c = next(it)
                except StopIteration:
                    exhausted = True
                    write(self._safe_tail(last_xy, n_tail_park, rate))
                    break
                write(c)
                st["written"] += len(c)
                st["n_real"] = st["written"]
                mon.written = st["written"]
                poll()
                drain()
                if progress is not None:
                    progress(st["written"])

            # ---- wait until the real data (or, on abort, the safe tail) has been generated
            n_real = st["n_real"]
            target = n_real + (n_hold + n_tail_park // 2 if aborted else 0)
            timeout = cfg.abort_drain_timeout_s if aborted else cfg.write_timeout_s + buf / rate
            deadline = time.perf_counter() + timeout
            while True:
                drain()
                gen = int(ao.out_stream.total_samp_per_chan_generated)
                mon.generated = gen
                mon.t_update = time.perf_counter()
                if gen >= target and (aborted or reader is None or st["ai_read"] >= n_real):
                    break
                if time.perf_counter() > deadline:
                    raise HALError(
                        f"timeout waiting for stream completion (generated {gen}/{target}, "
                        f"AI {st['ai_read']}/{n_real})"
                    )
                time.sleep(0.002)
            if progress is not None:
                progress(st["written"])
        finally:
            mon.running = False
            # stop the clock master first so DO/AI freeze together, then release everything;
            # safe_state() (blank first, then park) runs only after the tasks are closed.
            for t in (ao, do, ai):
                if t is not None:
                    try:
                        t.stop()
                    except Exception:
                        pass
            for t in tasks:
                try:
                    t.close()
                except Exception:
                    pass
            try:
                self.safe_state()
            except Exception as e:  # noqa: BLE001
                # with the device gone the stream error (already in flight, if any) tells the story
                if not _is_lost(e):
                    raise
                if sys.exc_info()[1] is None:
                    raise DeviceLostError(f"device lost while finishing the stream: {e}") from e

        ai_all = None
        if n_ai and on_ai is None:
            ai_all = np.concatenate(collected, axis=1) if collected else np.zeros((n_ai, 0))
        info["out_fill_min"] = mon.out_fill_min
        info["ai_backlog_max"] = mon.ai_backlog_max
        return StreamResult(
            samples_written=st["written"],
            rate=rate,
            aborted=aborted,
            ai=ai_all,
            ai_channels=tuple(ai_channels),
            elapsed_s=time.perf_counter() - t0,
            info=info,
        )
