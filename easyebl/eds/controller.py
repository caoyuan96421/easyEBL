"""EDS acquisition backend: spectra, electron images, line scans and spectrum maps.

No Qt here.  The GUI (or a script, or a test) calls the controller; every
acquisition runs as an :class:`EDSJob` in a worker thread and reports through
an event bus (``events.subscribe()``) and the job's ``partial`` result.

Sharing the DAQ with SEM imaging
--------------------------------
The controller has its own HAL handle (owner ``"eds"``).  Every job that moves
the beam runs inside ``handle.exclusive(...)``: the live SEM image (a
preemptible handle) pauses, other modules are refused, and the SEM scan
source is switched to external for the job and restored afterwards.  When the
job ends the SEM image resumes by itself.  Jobs on the pulse processor take
its ``session("eds")``, so two EDS front-ends cannot interleave either.

Acquisition model (DP5: no list mode)
-------------------------------------
Each pixel is a separate preset-real-time acquisition while the beam is held
on it (see :mod:`easyebl.eds.beam`): move → wait for the beam to be there (+
settle) → enable the MCA (it stops itself after the preset) → move to the next
pixel while the spectrum is read out and cleared.  The SE signal is averaged
over the same samples.  Per-pixel overhead ≈ the output latency (≈ 13 ms at
the minimum buffer) plus a few USB transactions.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import dataclass, field, replace
import itertools
import threading
import time
import traceback

import numpy as np

from ..core.calibration import FieldCalibration
from ..core.waveform import OutputChunk
from ..hal.backend import StreamMonitor
from ..hal.mca import MCAReading, PulseProcessor
from ..imaging.engine.frames import EventBus
from .beam import AIWindows, BeamProgram, grid_points, raster_pattern
from .spectrum import ElectronImage, EnergyCalibration, Spectrum, SpectrumMap

__all__ = ["EDSOptions", "EDSJob", "EDSController", "JobCancelled"]


class JobCancelled(RuntimeError):
    pass


@dataclass
class EDSOptions:
    rate: float | None = None  # stream sample clock (S/s); None = the device maximum for the layout
    max_rate: float = 1.25e6  # …but not above this (plenty for SE averaging; light on USB)
    # Output buffer.  Deep, like the DP5 simulator scripts (1 s buffer, 50 ms writes): the EDS
    # phantom generates its SDD waveform in Python, and a shallow buffer underflows on Windows.
    # Maps do not wait for it – their pixels are scheduled ahead on the sample clock.
    latency_s: float = 0.4
    chunk_n: int = 8192
    pixel_slack_s: float = 0.015  # map slot = settle + dwell + slack (DP5 read-out and clear)
    enable_guard_s: float = 0.003  # latest MCA enable: slot end − dwell − guard
    settle_s: float = 100e-6  # after a move before counting starts
    row_settle_s: float = 1e-3  # after the jump to the next row
    video_channel: str | None = "video"  # AI channel co-acquired as the SE signal (None: none)
    ai_delay_samples: int = 1  # AI sample k+d belongs to output sample k
    external_control: bool = True  # switch the SEM to external scan for each job
    lease_timeout_s: float = 15.0
    update_s: float = 0.25  # live spectrum / progress event interval
    area_points: int = 48  # area spectra: grid of area_points² positions (one sample each)
    flyback_s: float = 200e-6  # electron image: blanked return at the start of each row
    cube_dtype: str = "uint16"


@dataclass
class EDSJob:
    """A running or finished acquisition."""

    kind: str
    description: str
    id: int
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress: float = 0.0
    message: str = ""
    partial: object = None
    t_start: float = field(default_factory=time.time)

    def cancel(self) -> None:
        """Stop early.  Spectra/maps keep what was acquired so far (``result()`` returns it)."""
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None):
        return self.future.result(timeout)

    def exception(self, timeout: float | None = None):
        return self.future.exception(timeout)


class _Beam:
    """The beam during one job: a HAL stream driven by a :class:`BeamProgram`."""

    def __init__(self, ctrl: "EDSController", rate: float, start_v, job: EDSJob, video: bool):
        self.ctrl, self.rate, self.job = ctrl, rate, job
        self.prog = BeamProgram(rate, start=start_v, chunk_n=ctrl.options.chunk_n, blank=True)
        self.mon = StreamMonitor()
        self.ai = AIWindows(1) if video else None
        self.error: BaseException | None = None
        self.result = None
        self._thread = threading.Thread(target=self._run, name="easyebl-eds-stream", daemon=True)

    def _run(self):
        o = self.ctrl.options
        try:
            chans = [o.video_channel] if self.ai is not None else []
            self.result = self.ctrl.handle.run_stream(
                self.rate, self.prog.chunks(), ai_channels=chans, on_ai=self.ai.feed if self.ai else None,
                timeout=2.0, latency_s=o.latency_s, monitor=self.mon)
        except BaseException as e:  # noqa: BLE001 - reported to the job thread
            self.error = e
        finally:
            self.prog.stop()

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self.mon.running or self.mon.generated <= 0:
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError("the EDS beam stream did not start")
            time.sleep(0.001)

    def check(self):
        if self.error is not None:
            raise self.error
        if not self._thread.is_alive() and self._thread.ident is not None and self.error is None:
            raise RuntimeError("the EDS beam stream ended unexpectedly")
        if self.job.cancelled:
            raise JobCancelled()

    def now_index(self) -> int:
        """Best estimate of the sample the device is generating now (between monitor polls)."""
        m = self.mon
        est = m.generated + (time.perf_counter() - m.t_update) * self.rate if m.running else m.generated
        return int(min(est, m.written)) if m.written else int(est)

    def wait_index(self, index: int, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while True:
            now = self.now_index()
            if now >= index:
                return now
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError(f"beam stream stalled (at {now} < {index})")
            time.sleep(min(max((index - now) / self.rate - 0.0005, 0.0002), 0.005))

    def wait_generated(self, index: int, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while self.mon.generated < index:
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError(f"beam stream stalled (generated {self.mon.generated} < {index})")
            lag = (index - self.mon.generated) / self.rate
            time.sleep(min(max(lag - 0.0005, 0.0002), 0.005))

    def issue(self, x_v, y_v, blank=False):
        return self.prog.move(x_v, y_v, blank)

    def arrive(self, ticket, settle_s: float) -> int:
        """Wait until the beam has been at the commanded position for *settle_s*; returns that sample index."""
        idx = ticket.wait(5.0) + int(round(settle_s * self.rate))
        self.wait_generated(idx)
        return idx

    def goto(self, x_v, y_v, settle_s: float) -> int:
        return self.arrive(self.issue(x_v, y_v), settle_s)

    def stop(self):
        self.prog.stop()
        self._thread.join(15.0)


class EDSController:
    """Thread-safe EDS backend over a HAL handle and a :class:`PulseProcessor`."""

    def __init__(self, manager, mca: PulseProcessor, calibration: FieldCalibration | None = None,
                 energy_calibration: EnergyCalibration | None = None, beam_kv: float = 20.0,
                 takeoff_deg: float = 35.0, options: EDSOptions | None = None, owner: str = "eds"):
        self.handle = manager.get_handle(owner) if hasattr(manager, "get_handle") else manager
        self.mca = mca
        self.daq_config = self.handle.config
        self._cal = calibration or FieldCalibration(100.0, self.daq_config.ao_limit_v)
        self._ecal = energy_calibration or EnergyCalibration(0.0, 0.01)
        self.beam_kv = float(beam_kv)
        self.takeoff_deg = float(takeoff_deg)
        self.options = options or EDSOptions()
        self.events = EventBus()
        self._lock = threading.RLock()
        self._job: EDSJob | None = None
        self._ids = itertools.count(1)
        self._closed = False
        self.last_beam_v: tuple | None = None

    # ------------------------------------------------------------ settings
    @property
    def calibration(self) -> FieldCalibration:
        return self._cal

    def set_calibration(self, cal: FieldCalibration) -> None:
        with self._lock:
            self._cal = cal
        self.events.emit("settings", f"field {cal.field_size_um:g} µm")

    @property
    def energy_calibration(self) -> EnergyCalibration:
        return self._ecal

    def set_energy_calibration(self, cal: EnergyCalibration) -> None:
        with self._lock:
            self._ecal = cal
        self.events.emit("settings", f"energy calibration {cal.offset_kev * 1e3:.1f} eV + "
                                     f"{cal.gain_kev * 1e3:.4f} eV/ch", calibration=cal.to_dict())

    @property
    def rate(self) -> float:
        o = self.options
        best = self.daq_config.max_scan_rate(1 if o.video_channel else 0)
        r = min(best, o.max_rate) if o.rate is None else min(o.rate, best)
        return 100e6 / int(-(-100e6 // r))  # a divisor of the 100 MHz timebase

    @property
    def job(self) -> EDSJob | None:
        return self._job

    @property
    def busy(self) -> bool:
        j = self._job
        return j is not None and not j.done()

    def full_field_um(self) -> tuple:
        """(cx, cy, w, h) of the whole deflection field."""
        f = self._cal.field_size_um
        return (0.0, 0.0, f, f)

    # ------------------------------------------------------------ job plumbing
    def _start(self, kind: str, description: str, fn, *args) -> EDSJob:
        with self._lock:
            if self._closed:
                raise RuntimeError("EDS controller closed")
            if self.busy:
                raise RuntimeError(f"EDS busy: {self._job.description}")
            job = EDSJob(kind, description, next(self._ids))
            self._job = job
        t = threading.Thread(target=self._run_job, args=(job, fn, args), name=f"easyebl-eds-{kind}", daemon=True)
        t.start()
        return job

    def _run_job(self, job: EDSJob, fn, args):
        try:
            self.events.emit("job_started", job.description, job=job.id, job_kind=job.kind)
            with self.mca.session(self.handle.owner):
                res = fn(job, *args)
        except JobCancelled:
            res = job.partial
            job.message = "cancelled"
            job.future.set_result(res)
            self.events.emit("job_done", f"{job.description}: cancelled", job=job.id, job_kind=job.kind, cancelled=True)
            return
        except BaseException as e:  # noqa: BLE001
            job.message = f"{type(e).__name__}: {e}"
            job.future.set_exception(e)
            self.events.emit("error", f"{job.description}: {job.message}", job=job.id, job_kind=job.kind,
                             traceback=traceback.format_exc())
            self.events.emit("job_done", f"{job.description}: failed", job=job.id, job_kind=job.kind, failed=True)
            return
        job.progress = 1.0
        job.message = "done"
        job.partial = res
        job.future.set_result(res)
        self.events.emit("job_done", f"{job.description}: done", job=job.id, job_kind=job.kind)

    def _progress(self, job: EDSJob, frac: float, msg: str, kind: str = "progress", **data):
        job.progress, job.message = float(frac), msg
        self.events.emit(kind, msg, job=job.id, progress=float(frac), **data)

    @contextlib.contextmanager
    def _beam(self, job: EDSJob, reason: str, start_v=(0.0, 0.0), video: bool | None = None):
        o = self.options
        video = (o.video_channel is not None) if video is None else video
        with self.handle.exclusive(reason, timeout=o.lease_timeout_s, external_control=o.external_control):
            beam = _Beam(self, self.rate, start_v, job, video)
            try:
                beam.start()
                yield beam
            finally:
                beam.stop()
                if beam.error is not None and not job.cancelled and not isinstance(beam.error, JobCancelled):
                    raise beam.error

    def cancel(self) -> None:
        j = self._job
        if j is not None:
            j.cancel()

    def close(self, timeout: float = 15.0) -> None:
        with self._lock:
            self._closed = True
            j = self._job
        if j is not None and not j.done():
            j.cancel()
            try:
                j.future.exception(timeout)
            except Exception:  # noqa: BLE001
                pass
        self.handle.close()

    # ------------------------------------------------------------ conversions
    def _to_volts(self, points_um) -> np.ndarray:
        lim = min(self._cal.v_full_scale, self.daq_config.ao_limit_v)
        return self._cal.um_to_volts(np.atleast_2d(np.asarray(points_um, float)), check=True, limit_v=lim)

    def _spectrum(self, r: MCAReading, title: str, **meta) -> Spectrum:
        return Spectrum(r.counts.astype(np.int64), self._ecal, r.real_time, r.live_time, r.fast_count, r.slow_count,
                        self.beam_kv, self.takeoff_deg, title=title, metadata=meta)

    def cube_channels(self) -> int:
        n = int(self.mca.channels)
        k = int(np.ceil(self._ecal.channel(self.beam_kv + 0.5))) + 1
        return max(16, min(n, k))

    # ============================================================== spectra
    def acquire_spectrum(self, real_time_s: float, at_um=None, region_um=None, shape: str = "rect",
                         live_time: bool = False, title: str = "") -> EDSJob:
        """One spectrum: spot at *at_um* (x, y), averaged over *region_um* (cx, cy, w, h;
        ``shape`` rect | ellipse), or – both None – with the beam where the SEM put it
        (spot mode, no DAQ involvement).  ``live_time=True`` counts until the live
        time reaches *real_time_s*."""
        if not real_time_s > 0:
            raise ValueError("acquisition time must be > 0")
        if at_um is not None and region_um is not None:
            raise ValueError("give a point or a region, not both")
        if at_um is not None:
            volts = self._to_volts([at_um])[0]
            desc = f"spectrum at ({at_um[0]:.3g}, {at_um[1]:.3g}) µm"
        elif region_um is not None:
            volts = self._area_pattern(region_um, shape)
            desc = f"{shape} spectrum {region_um[2]:.3g}×{region_um[3]:.3g} µm"
        else:
            volts = None
            desc = "spectrum (beam positioned by the SEM)"
        return self._start("spectrum", title or desc, self._job_spectrum, float(real_time_s), volts,
                           bool(live_time), title or desc,
                           {"at_um": list(at_um) if at_um is not None else None,
                            "region_um": list(region_um) if region_um is not None else None, "shape": shape})

    def _area_pattern(self, region_um, shape: str) -> np.ndarray:
        cx, cy, w, h = map(float, region_um)
        n = int(self.options.area_points)
        pts = grid_points((cx, cy), w, h, n, n)
        if shape == "ellipse":
            u = (pts[:, 0] - cx) / (w / 2)
            v = (pts[:, 1] - cy) / (h / 2)
            pts = pts[u * u + v * v <= 1.0]
        elif shape != "rect":
            raise ValueError("shape must be 'rect' or 'ellipse'")
        volts = self._to_volts(pts)
        # serpentine order (short jumps): sort rows, alternate direction
        rows = np.unique(np.round(pts[:, 1], 9))[::-1]
        order = []
        for j, y in enumerate(rows):
            idx = np.nonzero(np.round(pts[:, 1], 9) == y)[0]
            idx = idx[np.argsort(pts[idx, 0])]
            order.append(idx if j % 2 == 0 else idx[::-1])
        return volts[np.concatenate(order)]

    def _job_spectrum(self, job: EDSJob, t_s, volts, live, title, meta):
        mca = self.mca
        o = self.options
        mca.set_preset_real_time(None if live else t_s)
        if volts is None:
            return self._count(job, t_s, live, title, meta)
        start = tuple(volts[0]) if volts.ndim == 2 else tuple(volts)
        with self._beam(job, title, start_v=start) as beam:
            if volts.ndim == 2 and len(volts) > 1:
                t = beam.prog.pattern(volts[:, 0], volts[:, 1])
                i0 = beam.arrive(t, o.settle_s)
            else:
                i0 = beam.goto(volts[0], volts[1], o.settle_s) if volts.ndim == 1 else beam.goto(*volts[0], o.settle_s)
            self.last_beam_v = start
            sp = self._count(job, t_s, live, title, meta, beam=beam, i0=i0)
        return sp

    def _count(self, job, t_s, live, title, meta, beam=None, i0=None) -> Spectrum:
        mca = self.mca
        o = self.options
        mca.clear()
        mca.enable()
        t_next = time.monotonic() + o.update_s
        t_end = time.monotonic() + t_s * (20 if live else 1) + 10.0
        try:
            while True:
                if job.cancelled:
                    mca.disable()
                    break
                if beam is not None:
                    beam.check()
                time.sleep(0.005)
                st = mca.status()
                done = (not st.mca_enabled) or (live and st.live_time >= t_s)
                if done:
                    mca.disable()
                    break
                if time.monotonic() >= t_next:
                    r = mca.read()
                    sp = self._spectrum(r, title, **meta)
                    job.partial = sp
                    frac = min(1.0, (r.live_time if live else r.real_time) / t_s)
                    self._progress(job, frac, f"{title}: {r.real_time:.1f} s, {r.icr / 1e3:.1f} kcps",
                                   kind="spectrum", spectrum=sp)
                    t_next = time.monotonic() + o.update_s
                if time.monotonic() > t_end:
                    mca.disable()
                    raise TimeoutError("the pulse processor did not stop at its preset")
        except JobCancelled:
            mca.disable()
        r = mca.read()
        if beam is not None and beam.ai is not None and i0 is not None:
            d = o.ai_delay_samples
            stop = max(i0 + 1, beam.now_index())
            beam.ai.request("se", i0 + d, stop + d)
            try:
                beam.wait_index(stop + d + o.chunk_n, timeout=2.0)
                time.sleep(0.02)
                meta = dict(meta, se_v=float(beam.ai.result("se", timeout=2.0)[0]))
            except (TimeoutError, JobCancelled):
                pass
        sp = self._spectrum(r, title, **meta)
        job.partial = sp
        self._progress(job, 1.0, f"{title}: {r.real_time:.2f} s, {sp.total} counts", kind="spectrum", spectrum=sp)
        return sp

    # ============================================================== electron image
    def acquire_image(self, nx: int = 256, ny: int = 256, dwell_s: float = 4e-6, region_um=None) -> EDSJob:
        """SE image for navigation (one frame, raster, blanked flyback)."""
        region = tuple(region_um) if region_um is not None else self.full_field_um()
        if not (nx >= 2 and ny >= 2 and dwell_s > 0):
            raise ValueError("bad image size / dwell")
        if not self.options.video_channel:
            raise ValueError("no video channel configured")
        pts = grid_points(region[:2], region[2], region[3], nx, ny)
        volts = self._to_volts(pts).reshape(ny, nx, 2)
        return self._start("image", f"electron image {nx}×{ny}", self._job_image, nx, ny, float(dwell_s), region,
                           volts)

    def _job_image(self, job, nx, ny, dwell, region, volts):
        o = self.options
        rate = self.rate
        spp = max(1, int(round(dwell * rate)))
        n_fb = max(8, int(round(o.flyback_s * rate)))
        skip = min(spp - 1, int(round(o.settle_s * rate * 0.1)))  # first sample(s) of a pixel: settling
        row_n = n_fb + nx * spp
        d = o.ai_delay_samples
        tail = max(64, d + 16)
        ai_all = np.zeros(ny * row_n + tail, np.float32)
        img = np.full((ny, nx), np.nan)
        state = {"got": 0, "rows": 0}

        def chunks():
            for j in range(ny):
                if job.cancelled:
                    return
                x = np.concatenate([np.full(n_fb, volts[j, 0, 0]), np.repeat(volts[j, :, 0], spp)])
                y = np.concatenate([np.full(n_fb, volts[j, 0, 1]), np.repeat(volts[j, :, 1], spp)])
                b = np.zeros(row_n, bool)
                b[:n_fb] = True
                yield OutputChunk(x, y, b)
            yield OutputChunk.constant(tail, volts[-1, -1, 0], volts[-1, -1, 1], blank=True)

        def on_ai(block):
            g = state["got"]
            k = min(block.shape[1], ai_all.size - g)
            ai_all[g:g + k] = block[0, :k]
            state["got"] = g = g + k
            while state["rows"] < ny and g >= (state["rows"] + 1) * row_n + d:
                j = state["rows"]
                s0 = j * row_n + n_fb + d
                img[j] = ai_all[s0:s0 + nx * spp].reshape(nx, spp)[:, skip:].mean(axis=1)
                state["rows"] = j + 1

        with self.handle.exclusive("EDS electron image", timeout=o.lease_timeout_s,
                                   external_control=o.external_control):
            mon = StreamMonitor()
            finished = threading.Event()
            res = {}

            def run():
                try:
                    res["r"] = self.handle.run_stream(rate, chunks(), ai_channels=[o.video_channel], on_ai=on_ai,
                                                      timeout=2.0, latency_s=max(o.latency_s, 0.05), monitor=mon)
                except BaseException as e:  # noqa: BLE001
                    res["e"] = e
                finally:
                    finished.set()

            th = threading.Thread(target=run, name="easyebl-eds-image", daemon=True)
            th.start()
            last = time.monotonic()
            while not finished.wait(0.02):
                if job.cancelled:
                    self.handle.abort()
                if time.monotonic() - last > o.update_s:
                    last = time.monotonic()
                    part = ElectronImage(img.copy(), region, dwell, metadata={"rows": state["rows"]})
                    job.partial = part
                    self._progress(job, state["rows"] / ny, f"electron image row {state['rows']}/{ny}",
                                   kind="image", image=part)
            th.join(5)
            if "e" in res:
                raise res["e"]
        im = ElectronImage(img, region, dwell, metadata={"rate": rate, "samples_per_pixel": spp})
        job.partial = im
        if job.cancelled:
            raise JobCancelled()
        self._progress(job, 1.0, "electron image done", kind="image", image=im)
        return im

    # ============================================================== maps / line scans
    def acquire_map(self, nx: int, ny: int, dwell_s: float, region_um=None, title: str = "") -> EDSJob:
        """Spectrum image (step scan, preset real time *dwell_s* per pixel)."""
        region = tuple(region_um) if region_um is not None else self.full_field_um()
        if not (nx >= 1 and ny >= 1 and dwell_s >= 0.001):
            raise ValueError("map needs ≥1×1 pixels and a dwell ≥ 1 ms (DP5 preset resolution)")
        pts = grid_points(region[:2], region[2], region[3], nx, ny)
        volts = self._to_volts(pts)
        desc = title or f"map {nx}×{ny} @ {dwell_s * 1e3:g} ms"
        return self._start("map", desc, self._job_points, volts, (ny, nx), float(dwell_s), region, desc,
                           {"type": "map"})

    def acquire_linescan(self, p0_um, p1_um, n: int, dwell_s: float, title: str = "") -> EDSJob:
        """Spectra at *n* points from *p0_um* to *p1_um* (a 1×n map whose metadata holds the path)."""
        p0, p1 = np.asarray(p0_um, float), np.asarray(p1_um, float)
        t = (np.arange(n) + 0.5) / n
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        volts = self._to_volts(pts)
        L = float(np.hypot(*(p1 - p0)))
        region = (float((p0[0] + p1[0]) / 2), float((p0[1] + p1[1]) / 2), L, 0.0)
        desc = title or f"line scan {n} pts @ {dwell_s * 1e3:g} ms"
        return self._start("linescan", desc, self._job_points, volts, (1, n), float(dwell_s), region, desc,
                           {"type": "linescan", "p0_um": p0.tolist(), "p1_um": p1.tolist(),
                            "points_um": pts.tolist()})

    def _job_points(self, job, volts, shape, dwell, region, title, meta):
        """Step scan with pixels scheduled ahead on the sample clock.

        Every pixel gets a slot of settle + dwell + slack samples, queued in the beam
        program long before the beam gets there, so the deep output buffer costs nothing
        per pixel.  When the beam reaches a slot (estimated from the stream monitor) the
        MCA is enabled with a preset real time = dwell; it stops itself inside the slot, is
        read and cleared while the beam already moves on.  A slot the host reached too late
        is rescheduled at the end (and the slack grows), so no pixel is ever counted at the
        wrong position.
        """
        o = self.options
        mca = self.mca
        ny, nx = shape
        n = len(volts)
        nch = self.cube_channels()
        sm = SpectrumMap.empty(nx, ny, nch, self._ecal, region, dwell, dtype=np.dtype(o.cube_dtype),
                               beam_kv=self.beam_kv, takeoff_deg=self.takeoff_deg, metadata=dict(meta))
        sm.metadata.update(rate=0.0, overhead_s=0.0, rescheduled=0)
        job.partial = sm
        mca.set_preset_real_time(dwell)
        cap = np.iinfo(sm.cube.dtype).max
        t_job = time.monotonic()
        with self._beam(job, title, start_v=tuple(volts[0])) as beam:
            R = beam.rate
            sm.metadata["rate"] = R
            mca.clear()
            dwell_n = int(round(dwell * R))
            settle_n = int(round(o.settle_s * R))
            row_n = int(round(o.row_settle_s * R))
            guard_n = int(round(o.enable_guard_s * R))
            slack = o.pixel_slack_s
            slot_n = row_n + dwell_n + int(round(slack * R))
            d = o.ai_delay_samples
            prog = beam.prog
            prog.schedule([(i, volts[i, 0], volts[i, 1]) for i in range(n)], slot_n)
            remaining = n
            late = 0
            k = 0
            t_last = time.monotonic()
            try:
                while remaining:
                    beam.check()
                    key, start, length = prog.slot(k, timeout=10.0)
                    k += 1
                    row, col = divmod(key, nx)
                    if sm.done[row, col]:
                        continue
                    t_en = start + (row_n if col == 0 or key == 0 else settle_n)
                    now = beam.wait_index(t_en, timeout=10.0)
                    if now > start + length - dwell_n - guard_n:
                        # too late for this slot: redo the pixel at the end, with more slack
                        late += 1
                        slack = min(slack * 1.25 + 0.002, 0.25)
                        slot_n = row_n + dwell_n + int(round(slack * R))
                        prog.schedule([(key, volts[key, 0], volts[key, 1])], slot_n)
                        continue
                    mca.enable()
                    if beam.ai is not None:
                        beam.ai.request(key, t_en + d, t_en + d + max(1, dwell_n))
                    mca.wait_done(timeout=dwell + 5.0, expected_s=dwell)
                    r = mca.read(clear=True)
                    c = r.counts[:nch]
                    sm.cube[row, col] = np.minimum(c, cap)
                    sm.real_time[row, col] = r.real_time
                    sm.live_time[row, col] = r.live_time
                    sm.fast_count[row, col] = r.fast_count
                    sm.done[row, col] = True
                    remaining -= 1
                    if beam.ai is not None:
                        for kk, v in beam.ai.ready().items():
                            rr, cc = divmod(kk, nx)
                            sm.se[rr, cc] = v[0]
                    now_t = time.monotonic()
                    if now_t - t_last >= o.update_s or remaining == 0:
                        t_last = now_t
                        nd = n - remaining
                        per = (now_t - t_job) / nd
                        sm.metadata.update(overhead_s=max(0.0, per - dwell), rescheduled=late,
                                           slot_s=slot_n / R)
                        self._progress(job, nd / n,
                                       f"{title}: {nd}/{n} px, {per * 1e3:.0f} ms/px, "
                                       f"{remaining * per:.0f} s left" + (f", {late} redone" if late else ""),
                                       kind="pixel", map=sm, index=key)
            finally:
                mca.disable()
                if beam.ai is not None:  # SE of the last pixels
                    try:
                        beam.wait_index(beam.now_index() + int(0.05 * R) + d, timeout=2.0)
                        time.sleep(0.05)
                    except Exception:  # noqa: BLE001
                        pass
                    for kk, v in beam.ai.ready().items():
                        rr, cc = divmod(kk, nx)
                        sm.se[rr, cc] = v[0]
                sm.metadata["elapsed_s"] = time.monotonic() - t_job
        return sm
