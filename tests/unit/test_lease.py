"""Exclusive DAQ lease, preemption of the live SEM image, and the cross-process device lock."""
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from easyebl.core import FieldCalibration, OutputChunk
from easyebl.hal import (
    DAQConfig,
    DAQManager,
    DeviceBusyError,
    LeaseHeldError,
    SimParams,
    SimulatedBackend,
)
from easyebl.hal.devlock import DeviceLock
from easyebl.imaging.engine.core import ImagingEngine
from easyebl.imaging.engine.frames import EngineState
from easyebl.imaging.engine.settings import EngineOptions


def _mgr(cfg=None):
    cfg = cfg or DAQConfig.usb6363()
    return DAQManager(SimulatedBackend(cfg, SimParams(realtime=True))).open()


def _hold(n=20_000):
    return iter([OutputChunk.constant(n, 0.0, 0.0, blank=False)])


def test_lease_refuses_other_handles_and_is_reentrant():
    mgr = _mgr()
    try:
        a, b = mgr.get_handle("eds"), mgr.get_handle("litho")
        events = []
        mgr.add_listener(lambda ev, d: events.append((ev, d)) if ev == "lease" else None)
        with a.exclusive("EDS map") as lease:
            assert lease.owner == "eds" and a.holds_lease and not b.holds_lease
            with pytest.raises(LeaseHeldError):
                b.run_stream(1e5, _hold(1000))
            with pytest.raises(LeaseHeldError):
                b.park(0.0, 0.0)
            with pytest.raises(LeaseHeldError):
                b.set_ai_terminal("DIFF")
            with a.exclusive("nested"):
                a.run_stream(1e5, _hold(1000))  # the holder streams normally
            assert a.holds_lease
            a.park(1.0, 1.0)
            # a second lease request times out
            with pytest.raises(LeaseHeldError):
                with b.exclusive("other", timeout=0.05):
                    pass
        assert mgr.lease is None
        b.run_stream(1e5, _hold(1000))
        assert [d["active"] for _, d in events] == [True, False]
    finally:
        mgr.close()


def test_lease_waits_for_a_non_preemptible_stream():
    mgr = _mgr()
    try:
        litho, eds = mgr.get_handle("litho"), mgr.get_handle("eds")
        t = threading.Thread(target=lambda: litho.run_stream(1e5, _hold(30_000)))  # 0.3 s exposure
        t.start()
        time.sleep(0.05)
        t0 = time.monotonic()
        with eds.exclusive("spectrum", timeout=5.0):
            waited = time.monotonic() - t0
            assert not litho.is_active
        t.join(5)
        assert waited > 0.15  # the exposure was not interrupted
        # too short a timeout: the lease is refused and nothing is left reserved
        t = threading.Thread(target=lambda: litho.run_stream(1e5, _hold(30_000)))
        t.start()
        time.sleep(0.05)
        with pytest.raises(LeaseHeldError):
            with eds.exclusive("spectrum", timeout=0.05):
                pass
        assert mgr.lease is None
        t.join(5)
    finally:
        mgr.close()


def test_lease_switches_external_control_and_restores_it():
    mgr = _mgr()
    try:
        sem, eds = mgr.get_handle("sem"), mgr.get_handle("eds")
        assert mgr.external_control is False
        with eds.exclusive("map", external_control=True):
            assert mgr.external_control is True
        assert mgr.external_control is False
        sem.set_external_control(True)  # SEM already in external scan: the lease leaves it on
        with eds.exclusive("map", external_control=True):
            assert mgr.external_control is True
        assert mgr.external_control is True
    finally:
        mgr.close()


def test_live_imaging_pauses_and_resumes():
    mgr = _mgr()
    eng = ImagingEngine(mgr, FieldCalibration(100.0, mgr.config.ao_limit_v), options=EngineOptions(rate=250e3, latency_s=0.02))
    try:
        states = []
        sub = eng.subscribe()
        eng.run()
        f0 = eng.wait_frame(timeout=10)
        assert f0 is not None
        eds = mgr.get_handle("eds")
        with eds.exclusive("EDS spectrum", timeout=5.0):
            deadline = time.monotonic() + 5
            while eng.state != EngineState.PAUSED and time.monotonic() < deadline:
                time.sleep(0.01)
            assert eng.state == EngineState.PAUSED
            assert not eng.handle.is_active
            eds.run_stream(250e3, _hold(5000))
        deadline = time.monotonic() + 10
        while eng.state != EngineState.LIVE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert eng.state == EngineState.LIVE
        f1 = eng.wait_frame(after_seq=f0.seq, timeout=10)
        assert f1 is not None and f1.seq > f0.seq
        for ev in sub.drain() if hasattr(sub, "drain") else []:
            states.append(ev)
        # stop while paused: the engine goes idle and does not resume
        with eds.exclusive("again", timeout=5.0):
            deadline = time.monotonic() + 5
            while eng.state != EngineState.PAUSED and time.monotonic() < deadline:
                time.sleep(0.01)
            eng.stop()
            time.sleep(0.3)
            assert eng.state in (EngineState.FROZEN, EngineState.IDLE)
        time.sleep(0.3)
        assert eng.state in (EngineState.FROZEN, EngineState.IDLE)
    finally:
        eng.close()
        mgr.close()


def test_capture_is_not_interrupted():
    mgr = _mgr()
    eng = ImagingEngine(mgr, FieldCalibration(100.0, mgr.config.ao_limit_v), options=EngineOptions(rate=250e3, latency_s=0.02))
    try:
        fut = eng.capture(nx=128, ny=96, dwell_s=8e-6)  # ≈ 0.1 s of scanning
        time.sleep(0.02)
        eds = mgr.get_handle("eds")
        with eds.exclusive("spectrum", timeout=20.0):
            assert fut.done() and fut.exception() is None
    finally:
        eng.close()
        mgr.close()


def test_device_lock_across_processes(tmp_path):
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {repr(str(__import__('pathlib').Path(__file__).resolve().parents[2]))})
        from easyebl.hal.devlock import DeviceLock
        DeviceLock("NI-Dev9", directory={repr(str(tmp_path))}).acquire()
        import os
        print("held", os.getpid(), flush=True)
        time.sleep(30)
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        word, pid = p.stdout.readline().split()
        assert word == "held"  # the holder's own PID: a venv python.exe is a launcher (p.pid is not it)
        lk = DeviceLock("NI-Dev9", directory=str(tmp_path))
        with pytest.raises(DeviceBusyError, match=pid):
            lk.acquire()
        os.kill(int(pid), signal.SIGTERM)  # the holder itself (TerminateProcess on Windows)
        p.kill()
        p.wait(10)
        t0 = time.monotonic()
        while True:  # released by the OS when the holder dies
            try:
                lk.acquire()
                break
            except DeviceBusyError:
                assert time.monotonic() - t0 < 10, "the lock was not released after the holder died"
                time.sleep(0.05)
        other = DeviceLock("NI-Dev9", directory=str(tmp_path))
        other.acquire()  # same process: shared
        other.release()
        lk.release()
    finally:
        if p.poll() is None:
            p.kill()
