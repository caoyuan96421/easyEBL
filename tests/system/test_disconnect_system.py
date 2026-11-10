"""Sudden loss of the DAQ (USB unplugged / power cycle) and automatic recovery – simulator.

Covers the HAL (fail fast, safe state, reconnect, re-applying external
control and static lines) and the imaging engine (DISCONNECTED state,
automatic resume, captures, commands while disconnected, repeated loss,
closing while disconnected, no hangs, no leaks).
"""
import random
import threading
import time

import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.core.waveform import OutputChunk
from easyebl.hal import (
    DAQConfig,
    DAQManager,
    DeviceLostError,
    SimParams,
    SimulatedBackend,
)
from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine

pytestmark = pytest.mark.system


def wait_for(cond, timeout=10.0, step=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def hal():
    made = []

    def _make(realtime=False, interval=0.05, cfg=None, **kw):
        be = SimulatedBackend(cfg or DAQConfig.usb6363(), SimParams(realtime=realtime, v_full_scale=5.0))
        mgr = DAQManager(be, reconnect_interval_s=interval, **kw)
        made.append(mgr)
        return mgr, be

    yield _make
    for m in made:
        m.close()


@pytest.fixture
def engine(hal):
    made = []

    def _make(realtime=True, **opts):
        mgr, be = hal(realtime=realtime)
        mgr.open()
        o = dict(rate=500e3, latency_s=0.05)
        o.update(opts)
        eng = ImagingEngine(mgr, FieldCalibration(100, 5.0), options=EngineOptions(**o))
        eng.update(nx=64, ny=48, dwell_s=4e-6)
        made.append(eng)
        return eng, mgr, be

    yield _make
    for e in made:
        e.close()
        assert not e.thread_alive


def stream_chunks(n=200, size=2000, stop=None):
    for _ in range(n):
        if stop is not None and stop.is_set():
            return
        yield OutputChunk.constant(size, 0.1, 0.1, False)


# =========================================================================== HAL
class TestHAL:
    def test_static_call_on_lost_device_fails_fast_and_marks_lost(self, hal):
        mgr, be = hal(auto_reconnect=False)
        mgr.open()
        events = []
        mgr.add_listener(lambda e, d: events.append(e))
        h = mgr.get_handle("x")
        be.unplug()
        with pytest.raises(DeviceLostError):
            h.park(1.0, 1.0)
        assert not mgr.connected and "device_lost" in events
        b = mgr.beam_state
        assert np.isnan(b.x_v) and not b.blank  # unknown – never claimed safe
        t0 = time.perf_counter()
        for call in (lambda: h.park(), lambda: h.set_blank(True),
                     lambda: h.run_stream(100e3, stream_chunks(1)), lambda: h.set_external_control(True)):
            with pytest.raises(DeviceLostError):
                call()
        assert time.perf_counter() - t0 < 0.1, "operations on a lost device must fail immediately"
        assert mgr.disconnects == 1

    def test_stream_interrupted_then_auto_reconnect(self, hal):
        mgr, be = hal()
        mgr.open()
        events = []
        mgr.add_listener(lambda e, d: events.append(e))
        h = mgr.get_handle("x")
        be.unplug(after_samples=50_000)
        with pytest.raises(DeviceLostError):
            h.run_stream(100e3, stream_chunks(), ai_channels=["video"])
        assert not mgr.connected and not mgr.is_busy and mgr.active_owner is None
        time.sleep(0.2)
        assert not mgr.connected  # still unplugged: watchdog keeps probing
        be.replug()
        assert mgr.wait_connected(2.0)
        assert events.count("device_lost") == 1 and "device_restored" in events
        assert mgr.beam_state.blank and mgr.reconnects == 1
        r = h.run_stream(100e3, stream_chunks(5), ai_channels=["video"])  # works again
        assert r.samples_written == 10_000

    def test_external_control_restored_after_reconnect(self, hal):
        mgr, be = hal()
        mgr.open()
        sem = mgr.get_handle("sem")
        sem.set_external_control(True)
        assert be.external_control
        be.unplug()
        with pytest.raises(DeviceLostError):
            sem.park()
        assert mgr.external_control is None  # unknown while the device is away
        be.replug()  # power-cycled device comes back in internal scan …
        assert mgr.wait_connected(2.0)
        assert mgr.external_control is True and be.external_control  # … and the HAL re-applies the request
        assert be.external_wires == {"PFI6": True, "PFI7": True}

    def test_external_released_while_lost_stays_internal(self, hal):
        mgr, be = hal()
        mgr.open()
        sem = mgr.get_handle("sem")
        sem.set_external_control(True)
        be.unplug()
        with pytest.raises(DeviceLostError):
            sem.park()
        sem.close()  # must not raise although the device is gone
        be.replug()
        assert mgr.wait_connected(2.0)
        assert mgr.external_control is False and not be.external_control
        assert "external_control" not in mgr.claims()

    def test_static_lines_reapplied(self, hal):
        cfg = DAQConfig.usb6363()
        cfg.static_lines = {"aux": "port1/line3"}
        mgr, be = hal(cfg=cfg)
        mgr.open()
        h = mgr.get_handle("x")
        h.claim("aux")
        h.set_line("aux", True)
        be.unplug()
        with pytest.raises(DeviceLostError):
            h.park()
        be.replug()
        assert be.static_lines["aux"] is False  # power cycled
        assert mgr.wait_connected(2.0)
        assert be.static_lines["aux"] is True

    def test_open_while_missing(self, hal):
        mgr, be = hal()
        be.unplug()
        with pytest.raises(DeviceLostError):
            mgr.open()
        assert not mgr.is_open
        mgr.open(allow_missing=True)
        assert mgr.is_open and not mgr.connected and mgr.reconnecting
        be.replug()
        assert mgr.wait_connected(2.0) and mgr.external_control is False and mgr.beam_state.blank

    def test_slow_failing_calls_do_not_hang(self, hal):
        """A dead USB link may make each call time out slowly: close() still completes."""
        mgr, be = hal()
        mgr.open()
        h = mgr.get_handle("x")
        be.unplug(fail_delay_s=0.3)
        with pytest.raises(DeviceLostError):
            h.park()
        t0 = time.perf_counter()
        mgr.close()
        assert time.perf_counter() - t0 < 2.0 and not mgr.reconnecting

    def test_close_while_reconnecting(self, hal):
        mgr, be = hal()
        mgr.open()
        be.unplug()
        with pytest.raises(DeviceLostError):
            mgr.get_handle("x").park()
        assert mgr.reconnecting
        mgr.close()
        assert not mgr.reconnecting and not mgr.is_open

    def test_flapping_connection(self, hal):
        """Plug/unplug many times quickly: the HAL ends up consistent and usable."""
        mgr, be = hal(interval=0.01)
        mgr.open()
        h = mgr.get_handle("x")
        h.set_external_control(True)
        rnd = random.Random(5)
        for _ in range(40):
            be.unplug()
            try:
                h.park()
            except DeviceLostError:
                pass
            time.sleep(rnd.uniform(0, 0.02))
            be.replug()
            time.sleep(rnd.uniform(0, 0.03))
        assert mgr.wait_connected(2.0)
        assert wait_for(lambda: mgr.external_control is True, 2.0)
        h.park()
        assert mgr.beam_state.blank
        assert mgr.reconnects >= 1 and mgr.disconnects >= 1


# ======================================================================== engine
def test_engine_resumes_live_after_unplug(engine):
    eng, mgr, be = engine()
    sub = eng.subscribe()
    eng.run()
    f0 = eng.wait_frame(complete=True, timeout=10)
    be.unplug()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    st = eng.status()
    assert not st.connected and st.disconnects == 1
    assert eng.latest() is not None  # the last image stays on screen
    time.sleep(0.3)
    assert eng.state == EngineState.DISCONNECTED and eng.thread_alive
    be.replug()
    assert wait_for(lambda: eng.state == EngineState.LIVE, 5)
    f1 = eng.wait_frame(eng.frames.seq, complete=True, timeout=10)
    assert f1.seq > f0.seq and not np.isnan(f1.image).any()
    msgs = [e.message for e in sub.drain()]
    assert any("lost" in m for m in msgs) and any("restored" in m for m in msgs)


def test_engine_mid_stream_unplug_at_random_points(engine):
    eng, mgr, be = engine(realtime=False)
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    rnd = random.Random(11)
    for k in range(15):
        be.unplug(after_samples=rnd.randint(0, 40_000))
        assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5), eng.status()
        time.sleep(rnd.uniform(0, 0.1))
        be.replug()
        assert wait_for(lambda: eng.state == EngineState.LIVE, 5), eng.status()
        assert eng.wait_frame(eng.frames.seq, complete=True, timeout=10) is not None
    st = eng.status()
    assert st.disconnects == 15 and st.streams >= 16 and st.state == EngineState.LIVE


def test_engine_settings_while_disconnected_apply_after(engine):
    eng, mgr, be = engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    be.unplug()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    eng.set_resolution(48, 36)
    eng.shift_beam(2.0, 0.0)
    be.replug()
    assert wait_for(lambda: (eng.latest() is not None and eng.latest().complete
                             and eng.latest().config.nx == 48), 10)
    f = eng.latest()
    assert f.config.center_um == (2.0, 0.0)


def test_engine_stop_while_disconnected(engine):
    eng, mgr, be = engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    be.unplug()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    eng.toggle_run_freeze()  # the "Stop" button
    assert wait_for(lambda: eng.state in (EngineState.FROZEN, EngineState.IDLE), 2)
    be.replug()
    assert mgr.wait_connected(2)
    time.sleep(0.2)
    assert eng.state in (EngineState.FROZEN, EngineState.IDLE)  # no resume after stop
    eng.run()
    assert eng.wait_frame(eng.frames.seq, complete=True, timeout=10) is not None


def test_engine_run_while_disconnected_starts_when_back(engine):
    eng, mgr, be = engine()
    be.unplug()
    with pytest.raises(DeviceLostError):
        mgr.get_handle("probe").park()
    eng.run()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    be.replug()
    assert eng.wait_frame(complete=True, timeout=10) is not None
    assert eng.state == EngineState.LIVE


def test_capture_fails_cleanly_on_unplug(engine):
    eng, mgr, be = engine()
    fut = eng.capture(frames=20, nx=128, ny=96, speed=8e-6)
    assert wait_for(lambda: eng.state == EngineState.CAPTURE, 5)
    be.unplug()
    with pytest.raises(RuntimeError, match="lost"):
        fut.result(timeout=5)
    be.replug()
    assert mgr.wait_connected(2)
    assert wait_for(lambda: eng.state in (EngineState.FROZEN, EngineState.IDLE), 5)
    photo = eng.capture(frames=1, nx=64, ny=48).result(timeout=20)  # works again
    assert photo.kind == "capture"


def test_single_frame_rescanned_after_unplug(engine):
    eng, mgr, be = engine()
    eng.update(nx=128, ny=128, dwell_s=8e-6)  # ≈ 0.15 s frame
    eng.single()
    assert wait_for(lambda: eng.state == EngineState.SINGLE, 5)
    be.unplug()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    be.replug()
    assert wait_for(lambda: eng.state == EngineState.FROZEN, 10)
    f = eng.frames.latest_complete()
    assert f is not None and f.kind == "single" and not np.isnan(f.image).any()


def test_engine_close_while_disconnected(engine):
    eng, mgr, be = engine()
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    be.unplug(fail_delay_s=0.2)
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    t0 = time.perf_counter()
    eng.close()
    assert time.perf_counter() - t0 < 2.0 and not eng.thread_alive


def test_engine_with_external_control_survives_power_cycle(engine):
    eng, mgr, be = engine()
    ext = mgr.get_handle("external_scan")
    ext.set_external_control(True)
    eng.run()
    eng.wait_frame(complete=True, timeout=10)
    be.unplug()
    assert wait_for(lambda: eng.state == EngineState.DISCONNECTED, 5)
    be.replug()
    assert wait_for(lambda: eng.state == EngineState.LIVE, 5)
    assert mgr.external_control is True and be.external_control  # SEM is on the DAQ scan again
    eng.close()
    ext.close()
    assert not be.external_control


def test_command_storm_with_random_unplugs(engine):
    """Commands from several threads while the device drops out repeatedly: no deadlock, no crash."""
    eng, mgr, be = engine(realtime=False)
    stop = threading.Event()
    errors = []

    def commander(seed):
        rnd = random.Random(seed)
        while not stop.is_set():
            try:
                op = rnd.randrange(6)
                if op == 0:
                    eng.run()
                elif op == 1:
                    eng.freeze()
                elif op == 2:
                    eng.set_speed(rnd.choice([2e-6, 4e-6]))
                elif op == 3:
                    eng.shift_beam(rnd.uniform(-2, 2), rnd.uniform(-2, 2))
                elif op == 4:
                    eng.capture(frames=1, nx=32, ny=24)
                else:
                    eng.status()
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            time.sleep(rnd.uniform(0, 0.01))

    def unplugger():
        rnd = random.Random(99)
        while not stop.is_set():
            time.sleep(rnd.uniform(0.05, 0.2))
            be.unplug(after_samples=rnd.choice([None, rnd.randint(0, 20000)]))
            time.sleep(rnd.uniform(0.0, 0.15))
            be.replug()

    threads = [threading.Thread(target=commander, args=(k,)) for k in range(3)] + [threading.Thread(target=unplugger)]
    for t in threads:
        t.start()
    time.sleep(5.0)
    stop.set()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    be.replug()
    assert not errors, errors[:3]
    assert eng.thread_alive and mgr.wait_connected(3)
    eng.run()
    assert eng.wait_frame(eng.frames.seq, complete=True, timeout=10) is not None
    eng.stop()
    assert eng.wait_idle(5) and mgr.beam_state.blank
