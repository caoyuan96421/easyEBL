import threading
import time

import numpy as np
import pytest

from easyebl.core import OutputChunk
from easyebl.hal import (
    DAQConfig,
    DeviceBusyError,
    HALError,
    ResourceConflictError,
    VoltageRangeError,
    create_backend,
)


def chunks(n_chunks=3, n=1000, v=1.0, markers=None):
    for _ in range(n_chunks):
        yield OutputChunk(np.full(n, v), np.zeros(n), np.zeros(n, bool), dict(markers or {}))


def slow_chunks(started: threading.Event, release: threading.Event, n=100):
    started.set()
    for _ in range(10_000):
        if release.is_set():
            return
        time.sleep(0.001)
        yield OutputChunk(np.zeros(n), np.zeros(n), np.zeros(n, bool))


def run_in_thread(fn):
    out = {}

    def target():
        try:
            out["result"] = fn()
        except Exception as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=target)
    t.start()
    return t, out


class TestConfig:
    def test_validate(self):
        DAQConfig().validate()
        with pytest.raises(ValueError):
            DAQConfig(timed_lines={"sync": "port1/line0"}).validate()  # wrong port
        with pytest.raises(ValueError):
            DAQConfig(timed_lines={"sync": "port0/line0"}).validate()  # collides with blank
        with pytest.raises(ValueError):
            DAQConfig(park_x_v=20).validate()

    def test_json_roundtrip(self):
        c = DAQConfig(device="Dev3", timed_lines={"sync": "port0/line2"})
        assert DAQConfig.from_json(c.to_json()) == c

    def test_factory(self):
        assert create_backend("sim").capabilities().simulated
        with pytest.raises(ValueError):
            create_backend("nope")


class TestHandles:
    def test_stream_and_safe_state(self, manager):
        h = manager.get_handle("a")
        r = h.run_stream(100e3, chunks(), ai_channels=["video"])
        assert r.samples_written == 3000 and r.ai.shape == (1, 3000) and not r.aborted
        st = manager.beam_state
        assert st.blank and st.x_v == manager.config.park_x_v
        assert not manager.is_busy and manager.active_owner is None

    def test_on_ai_consumer_receives_all_samples(self, manager):
        h = manager.get_handle("a")
        got = []
        r = h.run_stream(100e3, chunks(), ai_channels=["video"], on_ai=got.append)
        assert r.ai is None and sum(b.shape[1] for b in got) == 3000

    def test_busy_and_timeout(self, manager):
        a, b = manager.get_handle("a"), manager.get_handle("b")
        started, release = threading.Event(), threading.Event()
        t, out = run_in_thread(lambda: a.run_stream(100e3, slow_chunks(started, release)))
        assert started.wait(2)
        time.sleep(0.02)
        assert manager.active_owner == "a" and a.is_active
        with pytest.raises(DeviceBusyError):
            b.run_stream(100e3, chunks())
        with pytest.raises(DeviceBusyError):
            b.park(0, 0)
        with pytest.raises(DeviceBusyError):
            b.run_stream(100e3, chunks(), timeout=0.05)
        # waiting caller succeeds once the first stream ends
        t2, out2 = run_in_thread(lambda: b.run_stream(100e3, chunks(), timeout=5))
        time.sleep(0.05)
        release.set()
        t.join(5)
        t2.join(5)
        assert "error" not in out and "error" not in out2
        assert out2["result"].samples_written == 3000

    def test_abort_own_stream(self, manager):
        a = manager.get_handle("a")
        started, release = threading.Event(), threading.Event()
        t, out = run_in_thread(lambda: a.run_stream(100e3, slow_chunks(started, release)))
        assert started.wait(2)
        time.sleep(0.02)
        assert manager.get_handle("other").abort() is False
        assert a.abort() is True
        t.join(5)
        assert out["result"].aborted
        assert manager.beam_state.blank

    def test_abort_all(self, manager):
        a = manager.get_handle("a")
        started, release = threading.Event(), threading.Event()
        t, out = run_in_thread(lambda: a.run_stream(100e3, slow_chunks(started, release)))
        assert started.wait(2)
        time.sleep(0.02)
        manager.abort_all()
        t.join(5)
        assert out["result"].aborted

    def test_claims(self, manager):
        a, b = manager.get_handle("a"), manager.get_handle("b")
        with pytest.raises(ResourceConflictError):
            a.set_line("scan_relay", True)  # not claimed
        a.claim("scan_relay", "stage")
        assert a.claimed == {"scan_relay", "stage"}
        with pytest.raises(ResourceConflictError):
            b.claim("stage")
        with pytest.raises(ResourceConflictError):
            b.set_line("scan_relay", True)
        a.set_line("scan_relay", True)
        assert manager.line_state == {"scan_relay": True}
        assert manager.backend.static_lines["scan_relay"] is True
        assert manager.beam_state.blank  # switching a line blanks first
        with pytest.raises(HALError):
            a.set_line("nonexistent", True)
        a.release("stage")
        b.claim("stage")
        a.close()
        assert "scan_relay" not in manager.claims()
        with pytest.raises(HALError):
            a.park()

    def test_static_limits(self, manager):
        h = manager.get_handle("a")
        h.park(1.0, -2.0, blank=False)
        st = manager.beam_state
        assert (st.x_v, st.y_v, st.blank) == (1.0, -2.0, False)
        h.set_blank(True)
        assert manager.beam_state.blank and manager.beam_state.x_v == 1.0
        with pytest.raises(VoltageRangeError):
            h.park(10.5, 0)

    def test_stream_validation(self, manager):
        h = manager.get_handle("a")
        with pytest.raises(HALError):
            h.run_stream(5e6, chunks())
        with pytest.raises(HALError):
            h.run_stream(100e3, chunks(), ai_channels=["nope"])
        with pytest.raises(HALError):
            h.run_stream(600e3, chunks(), ai_channels=["video"])  # AI aggregate
        with pytest.raises(VoltageRangeError):
            h.run_stream(100e3, chunks(v=11.0))
        assert manager.beam_state.blank and not manager.is_busy
        with pytest.raises(HALError):
            h.run_stream(100e3, chunks(markers={"sync": np.zeros(1000, bool)}))

    def test_generator_exception_restores_safe_state(self, manager):
        h = manager.get_handle("a")

        def bad():
            yield OutputChunk(np.ones(10), np.ones(10), np.zeros(10, bool))
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            h.run_stream(100e3, bad())
        assert manager.beam_state.blank and not manager.is_busy
        h.run_stream(100e3, chunks())  # device usable again

    def test_reentrant_call_rejected(self, manager):
        h = manager.get_handle("a")

        def nested():
            yield OutputChunk(np.zeros(5), np.zeros(5), np.ones(5, bool))
            h.run_stream(100e3, chunks())

        with pytest.raises(DeviceBusyError):
            h.run_stream(100e3, nested())

    def test_listeners(self, manager):
        events = []
        manager.add_listener(lambda e, d: events.append(e))
        manager.add_listener(lambda e, d: 1 / 0)  # faulty listener must not break anything
        h = manager.get_handle("x")
        h.run_stream(100e3, chunks(1))
        h.claim("scan_relay")
        assert {"handle_created", "stream_started", "stream_finished", "claim"} <= set(events)
        assert [x.owner for x in manager.handles()].count("x") == 1

    def test_markers_allowed_when_configured(self, make_manager):
        m = make_manager(config=DAQConfig(timed_lines={"sync": "port0/line1"}), record=True)
        h = m.get_handle("a")
        h.run_stream(100e3, chunks(1, markers={"sync": np.ones(1000, bool)}))
        assert m.backend.recorded[0].markers["sync"].all()
