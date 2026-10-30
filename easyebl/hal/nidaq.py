"""NI-DAQmx implementation of :class:`DAQBackend` (X-series, e.g. USB-6341).

.. warning::
   Written against the ``nidaqmx`` Python API (1.x) but **not yet validated on
   hardware**.  Run the HAL test-pattern GUI with the beam blanked/off before
   first use and verify every signal on an oscilloscope.

Timing model
------------
* The AO task (X, Y) owns the sample clock (``/DevN/ao/SampleClock``).
* The DO task (blanker + timed marker lines, all on port0) and the AI task
  use that clock as their sample clock, so output sample *k*, DO sample *k*
  and AI sample *k* share one clock edge.
* AO and DO stream with regeneration disabled into an explicitly sized
  buffer (≥ ``min_buffer_s`` of data); input is re-chunked to ¼ buffer.  The
  AI buffer is drained between writes.
* Every stream ends with a clocked safe tail: blank while holding the last
  position (``tail_hold_s``, covers blanker latency), then park.  An abort
  plays the same tail instead of freezing the lines.  On errors the clock
  master (AO) is stopped first, the tasks are closed, then the static safe
  state is written blank-first.

Remaining hardware risk: a buffer underflow (host too slow) stops AO with
the blank line frozen in its last state until the error surfaces.  Keep
callbacks light; a hardware watchdog on the blanker is recommended.

Port0 is driven as a whole by the DO task (bit *n* = ``port0/line n``), so
reserve port0 for hardware-timed lines; put relays etc. on port1/port2.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Sequence
import warnings

import numpy as np

from ..core.waveform import OutputChunk, concat_chunks, rechunk
from .backend import AIConsumer, BackendCapabilities, DAQBackend, HALError, StreamResult
from .config import DAQConfig

__all__ = ["NIDAQBackend"]


def _line_bit(line: str) -> int:
    # "port0/line3" -> 3
    try:
        return int(line.split("line")[1])
    except (IndexError, ValueError) as e:
        raise HALError(f"cannot parse DO line '{line}'") from e


def _port_of(line: str) -> str:
    return line.split("/")[0]


class NIDAQBackend(DAQBackend):
    def __init__(self, config: DAQConfig):
        super().__init__(config)
        self._nidaqmx = None

    # -------------------------------------------------------------- lifecycle
    def open(self) -> None:
        try:
            import nidaqmx  # noqa: F401
            import nidaqmx.system
        except ImportError as e:  # pragma: no cover - depends on install
            raise HALError("nidaqmx package not installed (pip install nidaqmx)") from e
        self._nidaqmx = nidaqmx
        names = [d.name for d in nidaqmx.system.System.local().devices]
        if self.config.device not in names:
            raise HALError(f"device '{self.config.device}' not found; available: {names}")
        self._open = True
        self.safe_state()

    def close(self) -> None:
        if self._open:
            try:
                self.safe_state()
            finally:
                self._open = False

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

    def set_static_line(self, name: str, value: bool) -> None:
        nidaqmx = self._nidaqmx
        from nidaqmx.constants import LineGrouping

        line = self.config.static_lines[name]
        with nidaqmx.Task() as t:
            t.do_channels.add_do_chan(self._phys(line), line_grouping=LineGrouping.CHAN_PER_LINE)
            t.write(self.static_level(name, value), auto_start=True)

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

    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        abort: threading.Event | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> StreamResult:
        nidaqmx = self._nidaqmx
        from nidaqmx.constants import AcquisitionType, LineGrouping, RegenerationMode, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalSingleChannelWriter

        cfg = self.config
        dev = cfg.device
        buf = int(max(cfg.buffer_samples, cfg.min_buffer_s * rate))
        piece = max(1000, buf // 4)  # write granularity: keeps ≥ 3/4 buffer of margin
        lim = cfg.ao_limit_v
        n_ai = len(ai_channels)
        clock = f"/{dev}/ao/SampleClock"
        term = getattr(TerminalConfiguration, cfg.ai_terminal, TerminalConfiguration.DIFF)
        n_tail_park = max(1000, int(0.01 * rate))

        tasks: list = []  # creation order: ao, do, ai
        ao = do = ai = None
        written = 0
        ai_read = 0
        aborted = False
        collected = []
        info: dict = {}
        last_xy = (cfg.park_x_v, cfg.park_y_v)
        t0 = time.perf_counter()
        it = rechunk(chunks, piece)

        def deliver(block: np.ndarray, n_real: int):
            nonlocal ai_read
            keep = max(0, min(block.shape[1], n_real - ai_read))
            if keep:
                part = block[:, :keep]
                if on_ai is not None:
                    on_ai(part)
                else:
                    collected.append(part.copy())
            ai_read += block.shape[1]

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
            ao.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buf)
            ao.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            ao.out_stream.output_buf_size = buf
            actual = float(ao.timing.samp_clk_rate)
            info["actual_rate"] = actual
            if abs(actual - rate) > 1e-6 * rate:
                warnings.warn(
                    f"sample clock coerced from {rate:g} to {actual:g} S/s "
                    f"({(actual / rate - 1) * 100:+.4f}% timing/dose error) – choose a rate dividing 100 MHz",
                    RuntimeWarning,
                    stacklevel=3,
                )

            port = _port_of(cfg.blank_line)
            do.do_channels.add_do_chan(f"{dev}/{port}", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
            do.timing.cfg_samp_clk_timing(
                rate, source=clock, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buf
            )
            do.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
            do.out_stream.output_buf_size = buf

            reader = None
            if ai is not None:
                for name in ai_channels:
                    ai.ai_channels.add_ai_voltage_chan(
                        self._phys(cfg.ai_channels[name]),
                        terminal_config=term,
                        min_val=-cfg.ai_range_v,
                        max_val=cfg.ai_range_v,
                    )
                ai.timing.cfg_samp_clk_timing(
                    rate,
                    source=clock,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=max(4 * buf, int(rate * 4)),
                )
                reader = AnalogMultiChannelReader(ai.in_stream)

            ao_w = AnalogMultiChannelWriter(ao.out_stream, auto_start=False)
            do_w = DigitalSingleChannelWriter(do.out_stream, auto_start=False)

            def write(c: OutputChunk):
                nonlocal last_xy
                ao_w.write_many_sample(np.vstack([c.x, c.y]), timeout=cfg.write_timeout_s)
                do_w.write_many_sample_port_uint32(self._do_words(c), timeout=cfg.write_timeout_s)
                if len(c):
                    last_xy = (float(c.x[-1]), float(c.y[-1]))

            # ---- prefill (half the buffer) before starting the clock
            n_real = 0
            exhausted = False
            while n_real < buf // 2:
                try:
                    c = next(it)
                except StopIteration:
                    exhausted = True
                    break
                write(c)
                n_real += len(c)
            written = n_real
            if exhausted:
                write(self._safe_tail(last_xy, n_tail_park, rate))

            for t in (ai, do, ao):  # slaves first: they wait for the AO sample clock
                if t is not None:
                    t.start()

            def drain():
                if reader is None:
                    return
                avail = ai.in_stream.avail_samp_per_chan
                if avail:
                    block = np.empty((n_ai, avail))
                    reader.read_many_sample(block, number_of_samples_per_channel=avail, timeout=1.0)
                    deliver(block, n_real)

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
                written += len(c)
                n_real = written
                drain()
                if progress is not None:
                    progress(written)

            # ---- wait until the real data (or, on abort, the safe tail) has been generated
            n_hold = max(1, int(round(cfg.tail_hold_s * rate)))
            target = n_real + (n_hold + n_tail_park // 2 if aborted else 0)
            timeout = cfg.abort_drain_timeout_s if aborted else cfg.write_timeout_s + written / rate
            deadline = time.perf_counter() + timeout
            while True:
                drain()
                gen = ao.out_stream.total_samp_per_chan_generated
                if gen >= target and (aborted or reader is None or ai_read >= n_real):
                    break
                if time.perf_counter() > deadline:
                    raise HALError("timeout waiting for stream completion")
                time.sleep(0.002)
            if progress is not None:
                progress(written)
        finally:
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
            self.safe_state()

        ai_all = None
        if n_ai and on_ai is None:
            ai_all = np.concatenate(collected, axis=1) if collected else np.zeros((n_ai, 0))
        return StreamResult(
            samples_written=written,
            rate=rate,
            aborted=aborted,
            ai=ai_all,
            ai_channels=tuple(ai_channels),
            elapsed_s=time.perf_counter() - t0,
            info=info,
        )
