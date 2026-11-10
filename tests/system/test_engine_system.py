"""ImagingEngine end-to-end on the simulator: threading, commands, robustness."""
import concurrent.futures
import random
import threading
import time
import tracemalloc

import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.core.waveform import OutputChunk
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend, StreamUnderflowError
from easyebl.imaging import RasterImage
from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine

pytestmark = pytest.mark.system

CAL = FieldCalibration(100.0, 10.0)


@pytest.fixture
def make_engine():
    made = []

    def _make(params=None, options=None, realtime=False, backend_cls=SimulatedBackend, **cfg_kw):
        p = params or SimParams()
        p.realtime = realtime
        be = backend_cls(DAQConfig(max_ao_rate=2e6, max_ai_aggregate_rate=2e6), p)
        mgr = DAQManager(be).open()
        eng = ImagingEngine(mgr, CAL, options=options or EngineOptions(rate=500e3))
        eng.update(nx=64, ny=48, dwell_s=4e-6, **cfg_kw)
        made.append((eng, mgr))
        return eng, mgr

    yield _make
    for eng, mgr in made:
        eng.close()
        assert not eng.thread_alive, "scan thread must end on close()"
        mgr.close()


def wait_state(eng, states, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if eng.state in states:
            return True
        time.sleep(0.005)
    return False


def truth(sim, frame):
    img = RasterImage({"video": frame.image}, frame.config)
    cols, rows = np.meshgrid(np.arange(frame.config.nx), np.arange(frame.config.ny))
    xy = img.pixel_to_field(cols, rows)
    return sim.sample.intensity(xy[..., 0], xy[..., 1]) * sim.params.video_gain_v


def test_live_frames_are_exact(make_engine, quiet_params):
    eng, mgr = make_engine(quiet_params)
    eng.run()
    f = eng.wait_frame(complete=True, timeout=10)
    f = eng.wait_frame(f.seq, complete=True, timeout=10)  # a steady-state frame
    assert f.complete and f.image.shape == (48, 64) and f.image.dtype == np.float32
    np.testing.assert_allclose(f.image, truth(mgr.backend, f), atol=1e-5)
    assert eng.status().streams == 1


def test_settings_change_without_restarting_hardware(make_engine, quiet_params):
    eng, mgr = make_engine(quiet_params)
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    eng.update(nx=40, ny=30, width_um=60.0, center_um=(5.0, -3.0))
    eng.set_speed(2e-6)  # the 20 µs ramp lead-in now overscans 10 px: still inside the field
    deadline = time.monotonic() + 10
    while True:
        f = eng.wait_frame(complete=True, timeout=10)
        if f.config.nx == 40 and f.config.dwell_s == 2e-6 and f.config.center_um == (5.0, -3.0):
            f = eng.wait_frame(f.seq, complete=True, timeout=10)
            break
        assert time.monotonic() < deadline
    np.testing.assert_allclose(f.image, truth(mgr.backend, f), atol=1e-5)
    st = eng.status()
    assert st.streams == 1, "settings changes must not restart the hardware stream"
    assert st.change_latency_s is not None and st.change_latency_s < 2.0


def test_invalid_settings_raise_in_caller_and_keep_scanning(make_engine):
    eng, _ = make_engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    before = eng.config
    with pytest.raises(ValueError):
        eng.update(nx=0)
    with pytest.raises(ValueError):
        eng.update(width_um=1e6)  # does not fit the field
    with pytest.raises(ValueError):
        eng.update(ai_channels=("nope",))
    assert eng.config == before and eng.state == EngineState.LIVE
    seq = eng.frames.seq
    assert eng.wait_frame(seq, complete=True, timeout=10) is not None


def test_beam_shift_is_clamped_to_the_field(make_engine):
    eng, _ = make_engine()
    c = eng.set_beam_shift(1e4, 0)
    assert 0 < c.center_um[0] < 50
    eng.update()  # still valid
    w = eng.set_scan_width(1e4)
    assert w.width_um <= 100.0
    eng.reset_beam_shift()
    assert eng.config.center_um == (0.0, 0.0)


def test_freeze_finishes_the_frame_and_parks(make_engine):
    eng, mgr = make_engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    eng.freeze()
    assert wait_state(eng, {EngineState.FROZEN}), eng.state
    f = eng.frames.latest_complete()
    assert not np.isnan(f.image).any()
    assert mgr.beam_state.blank and not mgr.is_busy
    assert eng.wait_idle(2.0)


def test_freeze_twice_stops_immediately(make_engine):
    eng, mgr = make_engine(realtime=True)
    eng.update(nx=256, ny=256, dwell_s=20e-6)  # 1.3+ s frames
    eng.run()
    assert wait_state(eng, {EngineState.LIVE})
    eng.freeze()
    t0 = time.monotonic()
    eng.freeze()
    assert wait_state(eng, {EngineState.FROZEN, EngineState.IDLE}, 2.0)
    assert time.monotonic() - t0 < 1.0 and mgr.beam_state.blank


def test_single_frame_then_frozen(make_engine):
    eng, _ = make_engine()
    eng.single()
    assert wait_state(eng, {EngineState.FROZEN})
    f = eng.frames.latest_complete()
    assert f.kind == "single" and not np.isnan(f.image).any()
    n = eng.status().frames_completed
    assert n == 1


def test_capture_integrates_and_keeps_live_settings(make_engine, quiet_params):
    eng, mgr = make_engine(quiet_params)
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    live_cfg = eng.config
    fut = eng.capture(frames=3, nx=80, ny=60, speed=8e-6)
    photo = fut.result(timeout=30)
    assert photo.kind == "capture" and photo.frames_averaged == 3 and photo.image.shape == (60, 80)
    np.testing.assert_allclose(photo.image, truth(mgr.backend, photo), atol=1e-5)
    assert eng.config == live_cfg
    assert wait_state(eng, {EngineState.FROZEN})


def test_stop_cancels_capture(make_engine):
    eng, _ = make_engine(realtime=True)
    fut = eng.capture(frames=50, nx=256, ny=256, speed=20e-6)
    assert wait_state(eng, {EngineState.CAPTURE})
    eng.stop()
    with pytest.raises(RuntimeError):
        fut.result(timeout=5)
    assert wait_state(eng, {EngineState.FROZEN, EngineState.IDLE})


def test_rate_change_restarts_stream_and_averaging(make_engine):
    eng, _ = make_engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    eng.set_options(rate=250e3, averaging=4)
    deadline = time.monotonic() + 10
    while eng.status().streams < 2 or eng.wait_frame(complete=True, timeout=5).config.rate != 250e3:
        assert time.monotonic() < deadline
    assert eng.rate == 250e3 and eng.state == EngineState.LIVE


def test_device_busy_is_reported_and_recoverable(make_engine):
    eng, mgr = make_engine()
    other = mgr.get_handle("litho")
    release = threading.Event()

    def hog():
        def chunks():
            while not release.is_set():
                yield OutputChunk.constant(1000, 0, 0, True)
                time.sleep(0.001)

        other.run_stream(500e3, chunks())

    t = threading.Thread(target=hog)
    t.start()
    try:
        time.sleep(0.05)
        eng.set_options(device_timeout_s=0.0)
        eng.run()
        assert wait_state(eng, {EngineState.ERROR})
        assert "busy" in eng.status().last_error
    finally:
        release.set()
        t.join(5)
    eng.run()
    assert eng.wait_frame(complete=True, timeout=10) is not None


class FlakyBackend(SimulatedBackend):
    """Raises an underflow on the first stream, after some data."""

    fails = 1

    def run_stream(self, *a, **kw):
        if FlakyBackend.fails > 0:
            FlakyBackend.fails -= 1
            chunks = a[1]
            next(iter(chunks))
            raise StreamUnderflowError("simulated underflow")
        return super().run_stream(*a, **kw)


def test_underflow_recovers_with_larger_buffer(make_engine):
    FlakyBackend.fails = 1
    eng, mgr = make_engine(backend_cls=FlakyBackend, options=EngineOptions(rate=500e3, latency_s=0.05))
    sub = eng.subscribe()
    eng.run()
    assert eng.wait_frame(complete=True, timeout=10) is not None
    st = eng.status()
    assert st.underruns == 1 and st.recoveries == 1 and st.streams == 2
    assert eng.directive() and eng._latency == pytest.approx(0.1)
    assert any(e.kind == "warning" for e in sub.drain())
    assert mgr.beam_state.blank or mgr.is_busy


def test_concurrent_command_storm(make_engine):
    """Many threads issue random commands while readers wait for frames: no deadlock, no crash."""
    eng, mgr = make_engine()
    stop = threading.Event()
    errors = []
    frames_seen = [0]

    def commander(seed):
        rnd = random.Random(seed)
        while not stop.is_set():
            op = rnd.randrange(10)
            try:
                if op == 0:
                    eng.run()
                elif op == 1:
                    eng.freeze()
                elif op == 2:
                    eng.single()
                elif op == 3:
                    eng.set_speed(rnd.choice([2e-6, 4e-6, 6e-6]))
                elif op == 4:
                    eng.shift_beam(rnd.uniform(-5, 5), rnd.uniform(-5, 5))
                elif op == 5:
                    eng.set_resolution(rnd.choice([32, 48, 64]), rnd.choice([24, 48]))
                elif op == 6:
                    eng.capture(frames=1, nx=32, ny=24)
                elif op == 7:
                    eng.stop()
                elif op == 8:
                    eng.set_options(averaging=rnd.randint(1, 4))
                else:
                    eng.status()
            except (ValueError,):
                pass
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            time.sleep(rnd.uniform(0, 0.01))

    def reader():
        seq = 0
        while not stop.is_set():
            f = eng.wait_frame(seq, timeout=0.2)
            if f is not None:
                seq = f.seq
                frames_seen[0] += 1
                assert np.isfinite(f.image[: max(1, f.rows_done)]).any() or not f.complete

    threads = [threading.Thread(target=commander, args=(k,)) for k in range(4)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    time.sleep(4.0)
    stop.set()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    assert not errors, errors[:3]
    assert eng.thread_alive and frames_seen[0] > 0
    eng.stop()
    assert eng.wait_idle(5.0)
    assert mgr.beam_state.blank and not mgr.is_busy
    # engine still works after the storm
    eng.run()
    assert eng.wait_frame(complete=True, timeout=10) is not None


def test_close_while_live_is_prompt(make_engine):
    eng, mgr = make_engine(realtime=True)
    eng.update(nx=512, ny=512, dwell_s=20e-6)
    eng.run()
    assert wait_state(eng, {EngineState.LIVE})
    t0 = time.monotonic()
    eng.close()
    assert time.monotonic() - t0 < 2.0 and not eng.thread_alive
    assert mgr.beam_state.blank and not mgr.is_busy
    with pytest.raises(RuntimeError):
        eng.run()


def test_fps_is_steady_on_the_sample_clock(make_engine):
    eng, _ = make_engine(realtime=True)
    eng.update(nx=64, ny=48, dwell_s=4e-6)
    eng.run()
    times = []
    seq = 0
    while len(times) < 12:
        f = eng.wait_frame(seq, complete=True, timeout=5)
        seq = f.seq
        times.append(f.t_sample)
    dt = np.diff(times[2:])
    assert np.allclose(dt, eng.frame_time_s(), rtol=1e-9), dt  # exact on the sample clock
    st = eng.status()
    assert st.fps_hw == pytest.approx(1 / eng.frame_time_s(), rel=1e-6)


def test_memory_is_bounded_over_many_frames(make_engine):
    eng, _ = make_engine()
    eng.update(nx=128, ny=96, dwell_s=2e-6)
    eng.run()
    for _ in range(10):
        eng.wait_frame(complete=True, timeout=10)
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    seq = eng.frames.seq
    for _ in range(150):
        f = eng.wait_frame(seq, complete=True, timeout=10)
        seq = f.seq
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(s.size_diff for s in snap.compare_to(base, "filename"))
    assert growth < 5e6, f"memory grew by {growth / 1e6:.1f} MB over 150 frames"


def test_capture_future_is_a_real_future(make_engine):
    eng, _ = make_engine()
    fut = eng.capture(frames=1)
    assert isinstance(fut, concurrent.futures.Future)
    assert fut.result(timeout=20).kind == "capture"
