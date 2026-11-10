"""Robustness and performance of the imaging engine on the real NI DAQ.

Every test checks, besides its own subject, that the beam ends blanked and
parked (fixture teardown) and that the scan thread ends on close().
"""
from __future__ import annotations

import gc
import random
import statistics
import sys
import threading
import time
import tracemalloc

import numpy as np
import pytest

from easyebl.core.waveform import OutputChunk
from easyebl.hal import DeviceBusyError
from easyebl.imaging.engine import SCAN_SPEEDS, EngineState

from .conftest import rss_bytes


# ----------------------------------------------------------------- helpers
def wait_state(eng, states, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if eng.state in states:
            return True
        time.sleep(0.005)
    return False


class BufferWatch:
    """Samples the stream monitor every ~2 ms: output backlog and AI backlog extremes."""

    def __init__(self, eng):
        self.eng = eng
        self.max_fill = 0
        self.max_buffer = 0
        self.max_ai_backlog = 0
        self.ai_buffer = 0
        self.samples = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            m = self.eng.monitor
            if m.running:
                fill = m.written - m.generated
                self.max_fill = max(self.max_fill, fill)
                self.max_buffer = max(self.max_buffer, m.buffer_samples)
                self.max_ai_backlog = max(self.max_ai_backlog, m.ai_backlog_max)
                self.ai_buffer = max(self.ai_buffer, m.ai_buffer_samples)
                self.samples += 1
            time.sleep(0.002)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(2)

    def check(self, report, label=""):
        report(f"  {label}buffer watch: max output backlog {self.max_fill} / buffer {self.max_buffer} samples, "
               f"max AI backlog {self.max_ai_backlog} / {self.ai_buffer} ({self.samples} polls)")
        # the host never queues more than the buffer it asked for (+1 piece in flight)…
        assert self.max_fill <= self.max_buffer * 1.13 + 2048
        # …and AI never comes anywhere near overflowing its buffer
        assert self.max_ai_backlog < 0.5 * max(self.ai_buffer, 1)


def collect_frames(eng, seconds, complete=True):
    """Complete frames for *seconds*; returns (frames, wall times)."""
    out, ts = [], []
    seq = eng.frames.seq
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        f = eng.wait_frame(seq, timeout=max(2.0, 3 * eng.frame_time_s()), complete=complete)
        assert f is not None, f"no frame for {max(2.0, 3 * eng.frame_time_s()):.1f} s – stream stalled? {eng.status()}"
        seq = f.seq
        out.append(f)
        ts.append(time.perf_counter())
    return out, ts


# ----------------------------------------------------------------- tests
def test_static_safe_state(hw_manager, report):
    h = hw_manager.get_handle("t")
    h.park(0.5, -0.5, blank=True)
    h.park()
    assert hw_manager.beam_state.blank
    report(f"device {hw_manager.config.device}: open, park OK")


def test_live_at_max_rate_is_steady(hw_engine, report):
    eng = hw_engine(latency_s=0.1)
    assert eng.rate == 2e6
    eng.update(nx=640, ny=480, dwell_s=0.5e-6)  # TV speed: fastest scan
    eng.run()
    with BufferWatch(eng) as bw:
        frames, ts = collect_frames(eng, 6.0)
    st = eng.status()
    theory = 1 / eng.frame_time_s()
    wall = (len(ts) - 1) / (ts[-1] - ts[0])
    jitter = np.std(np.diff(ts[2:])) * 1e3 if len(ts) > 4 else float("nan")
    report(f"live 640x480 TV @2 MS/s: {len(frames)} frames, theory {theory:.2f} fps, sample-clock "
           f"{st.fps_hw:.3f} fps, delivered {wall:.2f} fps (jitter {jitter:.1f} ms), "
           f"margin {st.margin_s * 1e3 if st.margin_s else float('nan'):.0f} ms, underruns {st.underruns}")
    bw.check(report)
    assert st.underruns == 0 and st.streams == 1
    assert st.fps_hw == pytest.approx(theory, rel=1e-6)
    assert wall == pytest.approx(theory, rel=0.05)
    assert all(not np.isnan(f.image).any() for f in frames[1:])
    hw_t = np.array([f.t_sample for f in frames])
    assert np.allclose(np.diff(hw_t), 1 / theory, rtol=1e-6), "frames lost or duplicated"


@pytest.mark.parametrize("latency_ms", [20, 30, 50, 100, 250])
def test_change_latency_follows_buffer(hw_engine, report, latency_ms):
    """Change latency ≈ buffer depth.  Below ~50 ms USB + OS scheduling can starve the
    buffer: then the engine must recover by itself (larger buffer, still live)."""
    eng = hw_engine(latency_s=latency_ms / 1e3)
    eng.update(nx=512, ny=384, dwell_s=1e-6)
    eng.run()
    collect_frames(eng, 1.0)
    lat = []
    for k in range(8):
        eng.shift_beam(0.5 if k % 2 else -0.5, 0)
        t0 = time.perf_counter()
        while eng.status().change_latency_s is None or eng._lat_gen != eng.directive().gen:
            time.sleep(0.001)
            assert time.perf_counter() - t0 < 5
        lat.append(eng.status().change_latency_s)
        time.sleep(0.15)
    st = eng.status()
    med = statistics.median(lat)
    report(f"buffer {latency_ms:4d} ms: change → first new line median {med * 1e3:6.1f} ms, "
           f"max {max(lat) * 1e3:6.1f} ms, margin {st.margin_s * 1e3 if st.margin_s else float('nan'):.0f} ms, "
           f"underruns {st.underruns}")
    assert st.state == EngineState.LIVE
    if latency_ms >= 50:
        assert st.underruns == 0
        assert med < latency_ms / 1e3 + 0.1
    else:
        assert st.recoveries == st.underruns


def test_start_stop_cycles(hw_engine, report):
    eng = hw_engine(latency_s=0.05)
    eng.update(nx=256, ny=192, dwell_s=1e-6)
    starts, stops = [], []
    for _ in range(25):
        t0 = time.perf_counter()
        eng.run()
        f = eng.wait_frame(complete=False, timeout=5)
        assert f is not None
        starts.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        eng.stop()
        assert eng.wait_idle(3.0), eng.status()
        stops.append(time.perf_counter() - t1)
    report(f"25 start/stop cycles: start → first image median {statistics.median(starts) * 1e3:.0f} ms "
           f"(max {max(starts) * 1e3:.0f}), stop → idle median {statistics.median(stops) * 1e3:.0f} ms "
           f"(max {max(stops) * 1e3:.0f})")
    assert max(stops) < 1.0
    assert eng.status().underruns == 0


def test_abort_at_random_times(hw_engine, hw_manager, report):
    eng = hw_engine(latency_s=0.05)
    rnd = random.Random(1)
    for _ in range(40):
        eng.run()
        time.sleep(rnd.uniform(0, 0.35))
        eng.stop()
        assert eng.wait_idle(3.0), eng.status()
        assert hw_manager.beam_state.blank
    st = eng.status()
    report(f"40 random aborts: streams {st.streams}, errors {st.last_error!r}")
    assert st.state in (EngineState.IDLE, EngineState.FROZEN)
    eng.run()
    assert eng.wait_frame(complete=True, timeout=10) is not None


def test_settings_storm_keeps_one_stream(hw_engine, report):
    eng = hw_engine(latency_s=0.1)
    eng.update(nx=320, ny=240, dwell_s=1e-6)
    eng.run()
    rnd = random.Random(7)
    t_end = time.monotonic() + 20
    n_changes = 0
    last_seq, gaps = eng.frames.seq, []
    t_last = time.monotonic()
    with BufferWatch(eng) as bw:
        while time.monotonic() < t_end:
            op = rnd.randrange(6)
            try:
                if op == 0:
                    eng.set_speed(rnd.choice(SCAN_SPEEDS[:4]))
                elif op == 1:
                    eng.set_resolution(*rnd.choice([(320, 240), (256, 256), (640, 480), (128, 96)]))
                elif op == 2:
                    eng.shift_beam(rnd.uniform(-3, 3), rnd.uniform(-3, 3))
                elif op == 3:
                    eng.set_rotation_deg(rnd.uniform(-20, 20))
                elif op == 4:
                    eng.update(bidirectional=rnd.random() < 0.5)
                else:
                    eng.set_options(averaging=rnd.randint(1, 4))
                n_changes += 1
            except ValueError:
                pass
            time.sleep(rnd.uniform(0.02, 0.1))
            seq = eng.frames.seq
            now = time.monotonic()
            if seq != last_seq:
                gaps.append(now - t_last)
                last_seq, t_last = seq, now
    st = eng.status()
    report(f"settings storm: {n_changes} changes in 20 s, streams {st.streams}, underruns {st.underruns}, "
           f"max gap between published images {max(gaps) * 1e3:.0f} ms")
    bw.check(report)
    assert st.streams == 1 and st.underruns == 0 and st.state == EngineState.LIVE
    assert max(gaps) < 1.0


def _hold_gil(seconds: float):
    """Hold the GIL for *seconds* (pure-Python loop with a huge switch interval)."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(seconds + 1)
    try:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            pass
    finally:
        sys.setswitchinterval(old)


def test_survives_host_stalls_shorter_than_buffer(hw_engine, report):
    eng = hw_engine(latency_s=0.15, auto_recover=False)
    eng.update(nx=640, ny=480, dwell_s=0.5e-6)
    eng.run()
    collect_frames(eng, 1.0)
    for _ in range(10):
        _hold_gil(0.05)  # 50 ms stall of every Python thread
        time.sleep(0.15)
    frames, _ = collect_frames(eng, 1.0)
    st = eng.status()
    report(f"10 × 50 ms GIL stalls with 150 ms buffer: underruns {st.underruns}, "
           f"margin {st.margin_s * 1e3 if st.margin_s else float('nan'):.0f} ms, state {st.state.value}")
    assert st.underruns == 0 and st.state == EngineState.LIVE


def test_recovers_from_stall_longer_than_buffer(hw_engine, hw_manager, report):
    eng = hw_engine(latency_s=0.03, max_latency_s=1.0)
    eng.update(nx=640, ny=480, dwell_s=0.5e-6)
    sub = eng.subscribe()
    eng.run()
    collect_frames(eng, 0.5)
    _hold_gil(0.3)  # 10× the buffer: the device must underflow
    time.sleep(0.2)
    frames, _ = collect_frames(eng, 2.0)  # …and the engine must be streaming again
    st = eng.status()
    msgs = [str(e) for e in sub.drain() if e.kind in ("warning", "error")]
    report(f"300 ms stall with 30 ms buffer: underruns {st.underruns}, recoveries {st.recoveries}, "
           f"buffer now {eng._latency * 1e3:.0f} ms, state {st.state.value}; {msgs[:2]}")
    assert st.state == EngineState.LIVE and len(frames) > 2
    if st.underruns:
        assert st.recoveries == st.underruns and eng._latency > 0.03


def test_large_capture_bigger_than_buffer(hw_engine, report):
    eng = hw_engine(latency_s=0.1)
    with BufferWatch(eng) as bw:
        t0 = time.perf_counter()
        fut = eng.capture(frames=1, nx=2560, ny=1920, dwell_s=0.5e-6)
        photo = fut.result(timeout=60)
        dt = time.perf_counter() - t0
    report(f"capture 2560x1920 TV: {dt:.2f} s (frame {eng.frame_time_s(photo.config):.2f} s), "
           f"NaN {np.isnan(photo.image).mean() * 100:.2f} %")
    bw.check(report)
    assert photo.image.shape == (1920, 2560) and not np.isnan(photo.image).any()
    assert wait_state(eng, {EngineState.FROZEN})


def test_slow_photo_scan_abort_midway(hw_engine, report):
    eng = hw_engine(latency_s=0.1)
    fut = eng.capture(frames=1, nx=1280, ny=960, dwell_s=20e-6)  # ≈ 25 s
    assert wait_state(eng, {EngineState.CAPTURE}, 5)
    time.sleep(1.0)
    t0 = time.perf_counter()
    eng.stop()
    with pytest.raises(RuntimeError):
        fut.result(timeout=5)
    dt = time.perf_counter() - t0
    report(f"abort of a 25 s photo scan: capture cancelled after {dt * 1e3:.0f} ms")
    assert eng.wait_idle(2.0) and dt < 1.0


def test_concurrent_access_is_refused_cleanly(hw_engine, hw_manager, report):
    eng = hw_engine()
    eng.update(nx=256, ny=192)
    eng.run()
    collect_frames(eng, 0.5)
    other = hw_manager.get_handle("litho")
    with pytest.raises(DeviceBusyError):
        other.run_stream(1e6, [OutputChunk.constant(1000, 0, 0, True)])
    frames, _ = collect_frames(eng, 0.5)
    assert frames and eng.status().streams == 1
    report("second handle refused while live; engine unaffected")


def test_close_while_live(hw_engine, hw_manager, report):
    eng = hw_engine()
    eng.update(nx=1280, ny=960, dwell_s=5e-6)
    eng.run()
    assert wait_state(eng, {EngineState.LIVE})
    time.sleep(0.5)
    t0 = time.perf_counter()
    eng.close()
    dt = time.perf_counter() - t0
    report(f"close() while live: {dt * 1e3:.0f} ms")
    assert dt < 2.0 and not eng.thread_alive and not hw_manager.is_busy


def test_soak_memory_and_steadiness(hw_engine, report, pytestconfig):
    duration = pytestconfig.getoption("--hw-soak")
    eng = hw_engine(latency_s=0.1)
    eng.update(nx=640, ny=480, dwell_s=0.5e-6)
    eng.run()
    collect_frames(eng, 3.0)  # warm-up: buffers, caches, driver allocations
    gc.collect()
    rss0 = rss_bytes()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    rnd = random.Random(3)
    t_end = time.monotonic() + duration
    n_frames = 0
    seq = eng.frames.seq
    next_change = time.monotonic() + 1
    with BufferWatch(eng) as bw:
        while time.monotonic() < t_end:
            f = eng.wait_frame(seq, timeout=3, complete=True)
            assert f is not None, eng.status()
            seq = f.seq
            n_frames += 1
            if time.monotonic() > next_change:
                next_change += 1
                eng.set_speed(rnd.choice(SCAN_SPEEDS[:3]))
                eng.shift_beam(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    gc.collect()
    rss1 = rss_bytes()
    py_growth = sum(s.size_diff for s in snap1.compare_to(snap0, "filename"))
    st = eng.status()
    report(f"soak {duration:.0f} s: {n_frames} frames, streams {st.streams}, underruns {st.underruns}, "
           f"private memory {rss0 / 1e6:.0f} → {rss1 / 1e6:.0f} MB, Python heap growth {py_growth / 1e6:.2f} MB")
    bw.check(report)
    assert st.underruns == 0 and st.streams == 1
    assert py_growth < 5e6
    assert rss1 - rss0 < 50e6


def test_loopback_image_content(hw_engine, report, pytestconfig):
    """AO0 → AI0: every image row must reproduce the X deflection voltage of its pixels."""
    if not pytestconfig.getoption("--hw-loopback"):
        pytest.skip("needs AO0 wired to AI0 (--hw-loopback)")
    eng = hw_engine(latency_s=0.05)
    cal = eng.calibration
    for bidir in (False, True):
        eng.update(nx=200, ny=50, width_um=60, dwell_s=1e-6, bidirectional=bidir, center_um=(5.0, 0.0))
        eng.run()
        f = None
        for _ in range(3):
            f = eng.wait_frame(complete=True, timeout=10)
        eng.stop()
        eng.wait_idle()
        xs_um = 5.0 + (np.arange(200) + 0.5) * 60 / 200 - 30
        xv = xs_um * cal.volts_per_um
        img = f.image
        g, o = np.polyfit(xv, img.mean(axis=0), 1)
        resid = img - (g * xv + o)
        worst = np.abs(resid).max() / (xv.max() - xv.min())
        rows_fwd = img[0::2] if bidir else img
        rows_bwd = img[1::2] if bidir else img
        dshift = float(np.mean(rows_fwd) - np.mean(rows_bwd))
        report(f"loopback bidir={bidir}: gain {g:.4f}, offset {o * 1e3:.1f} mV, worst residual "
               f"{worst * 100:.2f} % of span, fwd-bwd {dshift * 1e3:.2f} mV")
        assert 0.9 < g < 1.1 and worst < 0.02


def test_external_control_switch(hw_engine, hw_manager, report, pytestconfig):
    """HAL external control on the real wires: refused while scanning, internal again on close."""
    if not pytestconfig.getoption("--hw-external"):
        pytest.skip("switches the SEM scan source: pass --hw-external")
    wires = hw_manager.config.external_control_map()
    assert hw_manager.external_control is False  # opening selected internal scan
    h = hw_manager.get_handle("external_scan")
    eng = hw_engine()
    eng.update(nx=256, ny=192)
    eng.run()
    collect_frames(eng, 0.3)
    with pytest.raises(DeviceBusyError):
        h.set_external_control(True)  # never switched under a running scan
    eng.stop()
    assert eng.wait_idle(3)
    t0 = time.perf_counter()
    h.set_external_control(True)
    dt = time.perf_counter() - t0
    assert hw_manager.external_control is True and hw_manager.beam_state.blank
    eng.run()
    frames, _ = collect_frames(eng, 1.0)
    eng.stop()
    assert eng.wait_idle(3)
    h.close()  # → internal scan
    assert hw_manager.external_control is False
    report(f"external control {wires}: switch {dt * 1e3:.0f} ms, {len(frames)} frames under external scan, "
           "back to internal on close")


def test_phantom_three_channel_stream_runs(hw_config, report):
    """AO2 as the phantom output: a 3-channel AO stream at the 3-channel clock limit (no wire needed)."""
    from dataclasses import replace

    from easyebl.core import FieldCalibration
    from easyebl.hal import DAQManager, create_backend
    from easyebl.imaging.engine import EngineOptions, ImagingEngine

    cfg = replace(hw_config, phantom_output="aux0")
    mgr = DAQManager(create_backend("nidaqmx", cfg)).open()
    eng = ImagingEngine(mgr, FieldCalibration(100.0, cfg.ao_limit_v), options=EngineOptions(latency_s=0.1))
    try:
        eng.update(nx=512, ny=384, dwell_s=4 / eng.rate)
        eng.run()
        frames, ts = collect_frames(eng, 3.0)
        st = eng.status()
        report(f"phantom stream: clock {eng.rate / 1e6:.4f} MS/s with AO0+AO1+AO2, {len(frames)} frames, "
               f"{st.fps_hw:.3f} fps, underruns {st.underruns}")
        assert eng.rate == pytest.approx(100e6 / 65) and st.underruns == 0 and len(frames) >= 3
    finally:
        eng.close()
        mgr.close()


def test_phantom_loopback_image(hw_config, report, pytestconfig):
    """AO2 → AI0: the image must reproduce the phantom chart (fidelity, registration, zoom, noise)."""
    if not pytestconfig.getoption("--hw-phantom"):
        pytest.skip("needs AO2 wired to AI0 (--hw-phantom)")
    from dataclasses import replace

    from easyebl.core import FieldCalibration
    from easyebl.hal import DAQManager, create_backend
    from easyebl.imaging.engine import EngineOptions, ImagingEngine
    from easyebl.imaging.phantom_check import fit_to_expected, pair_noise, subpixel_shift

    cfg = replace(hw_config, phantom_output="aux0", phantom_noise_v=0.2)
    be = create_backend("nidaqmx", cfg)
    mgr = DAQManager(be).open()
    cal = FieldCalibration(100.0, cfg.ao_limit_v)
    eng = ImagingEngine(mgr, cal, options=EngineOptions(latency_s=0.1))
    ph = be.phantom_source()
    T = 1 / eng.rate

    def frame(n=3):
        f = None
        for _ in range(n):
            f = eng.wait_frame(None if f is None else f.seq, complete=True, timeout=60)
        return f

    try:
        eng.update(nx=400, ny=300, width_um=60.0, dwell_s=16 * T, bidirectional=True)
        eng.run()
        f = frame()
        exp = ph.expected_image(f.config, cal)
        fit = fit_to_expected(f.image, exp)
        dx, dy = subpixel_shift(f.image, exp)
        dfb = subpixel_shift(f.image[0::2], exp[0::2])[0] - subpixel_shift(f.image[1::2], exp[1::2])[0]
        report(f"phantom loopback: {fit}; shift ({dx:+.3f}, {dy:+.3f}) px, fwd−bwd {dfb:+.3f} px")
        assert fit.r2 > 0.9 and 0.9 < fit.gain < 1.1
        assert abs(dx) < 0.5 and abs(dy) < 0.5 and abs(dfb) < 0.5
        noise = []
        for spp in (1, 4, 16):
            eng.update(nx=256, ny=256, width_um=40.0, dwell_s=spp * T, bidirectional=False)
            a = frame()
            b = eng.wait_frame(a.seq, complete=True, timeout=60)
            noise.append((spp, pair_noise(a.image, b.image)))
        report("phantom noise vs samples/px: " + ", ".join(f"{s}: {n * 1e3:.1f} mV" for s, n in noise))
        for spp, n in noise:
            assert n == pytest.approx(0.2 * fit.gain / np.sqrt(spp), rel=0.25)
    finally:
        eng.close()
        mgr.close()
