"""Scan monitor: non-blocking hand-off, blanking, binning over ±10 V, retention."""
import threading
import time

import numpy as np
import pytest

from easyebl.core import OutputChunk
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend
from easyebl.hal.scanmonitor import PersistenceAccumulator, ScanMonitorWorker, ScanTap


def chunk(n, x=0.0, y=0.0, blank=False):
    return np.full(n, x), np.full(n, y), np.full(n, blank)


# ------------------------------------------------------------------ hand-off
def test_tap_disabled_costs_nothing():
    tap = ScanTap(16)
    assert tap.push(*chunk(10)) == 0 and tap.pending == 0 and tap.pushed == 0


def test_tap_round_trip_with_wrap():
    tap = ScanTap(10)
    tap.enabled = True
    tap.push(np.arange(7.0), -np.arange(7.0), np.zeros(7, bool))
    assert len(tap.pop()[0]) == 7
    tap.push(np.arange(6.0) + 100, np.arange(6.0), np.arange(6) % 2 == 0)  # wraps around the end
    x, y, b = tap.pop()
    np.testing.assert_array_equal(x, np.arange(6.0) + 100)
    np.testing.assert_array_equal(b, np.arange(6) % 2 == 0)
    assert tap.dropped == 0 and tap.pending == 0


def test_tap_decimates_then_drops_when_full_and_never_blocks():
    tap = ScanTap(100)
    tap.enabled = True
    assert tap.push(*chunk(60)) == 60
    kept = tap.push(np.arange(200.0), np.zeros(200), np.zeros(200, bool))  # 40 free: every 5th sample
    assert kept == 40 and tap.pending == 100 and tap.dropped == 160
    assert tap.push(*chunk(50)) == 0 and tap.dropped == 210  # full: dropped, not waited for
    x, _y, _b = tap.pop()
    np.testing.assert_array_equal(x[60:], np.arange(0.0, 200.0, 5.0))


def test_push_never_waits_for_a_slow_consumer():
    """The writer thread runs at full speed while the consumer is stalled."""
    tap = ScanTap(4096)
    tap.enabled = True
    x, y, b = chunk(1000, 1.0, 1.0)
    t0 = time.perf_counter()
    for _ in range(2000):  # 2 M samples into a 4 k ring, nobody reading
        tap.push(x, y, b)
    dt = time.perf_counter() - t0
    assert tap.pending == 4096 and tap.dropped == 2000 * 1000 - 4096
    assert dt < 1.0


def test_concurrent_producer_consumer_lose_nothing_when_not_full():
    tap = ScanTap(1 << 16)
    tap.enabled = True
    got = []
    done = threading.Event()

    def consume():
        while not done.is_set() or tap.pending:
            got.append(tap.pop()[0])
            time.sleep(0.0005)

    th = threading.Thread(target=consume)
    th.start()
    for k in range(400):
        tap.push(np.arange(k * 100, (k + 1) * 100, dtype=float), np.zeros(100), np.zeros(100, bool))
        if k % 20 == 0:
            time.sleep(0.001)
    done.set()
    th.join(5)
    x = np.concatenate(got)
    assert tap.dropped == 0
    np.testing.assert_array_equal(x, np.arange(40000, dtype=float))  # order kept, nothing lost


# ------------------------------------------------------------------ accumulation
def test_blanked_samples_are_not_plotted():
    acc = PersistenceAccumulator(resolution=8, retention_s=0)
    x = np.array([-9.0, 9.0, 9.0, -9.0])
    y = np.array([9.0, 9.0, -9.0, -9.0])
    blank = np.array([False, True, False, True])
    assert acc.add(x, y, blank, now=0.0) == 2
    img = acc.image(now=0.0)
    assert img.sum() == 2
    assert img[0, 0] == 1  # (−9, +9): top left
    assert img[7, 7] == 1  # (+9, −9): bottom right
    assert img[0, 7] == 0 and img[7, 0] == 0  # the blanked corners stay dark
    assert acc.add(*chunk(100, blank=True), now=0.0) == 0 and acc.image(now=0.0).sum() == 2


def test_full_field_and_edges():
    acc = PersistenceAccumulator(resolution=4, retention_s=0)
    acc.add(np.array([-10.0, 10.0, 0.0, 12.0]), np.array([-10.0, 10.0, 0.0, 0.0]), np.zeros(4, bool), now=0)
    img = acc.image(now=0)
    assert img[3, 0] == 1 and img[0, 3] == 1  # ±10 V land on the outermost pixels
    assert img[1, 2] == 1  # 0 V: the upper-right of the four centre pixels
    assert img[1, 3] == 1  # out of range is clipped to the edge
    assert acc.image().shape == (4, 4)


def test_retention_expires_old_samples():
    acc = PersistenceAccumulator(resolution=16, retention_s=1.0, fade=False, n_buckets=10)
    acc.add(*chunk(5, -5, -5), now=0.0)
    acc.add(*chunk(3, 5, 5), now=0.8)
    assert acc.image(now=0.9).sum() == 8
    img = acc.image(now=1.2)  # the first samples are older than 1 s
    assert img.sum() == 3
    assert acc.image(now=1.95).sum() == 0  # all gone
    acc.add(*chunk(2), now=2.0)
    assert acc.image(now=2.0).sum() == 2


def test_fade_weights_older_samples_less():
    acc = PersistenceAccumulator(resolution=16, retention_s=1.0, fade=True, n_buckets=10)
    acc.add(*chunk(10, -5, -5), now=0.0)
    acc.add(*chunk(10, 5, 5), now=0.6)
    flat = acc.image(now=0.6).ravel()
    i_old = acc.bin_index([-5], [-5])[0]
    i_new = acc.bin_index([5], [5])[0]
    assert 0 < flat[i_old] < flat[i_new] <= 10
    assert flat.max() == flat[i_new]


def test_infinite_retention_until_clear():
    acc = PersistenceAccumulator(resolution=8, retention_s=0)
    acc.add(*chunk(4), now=0.0)
    assert acc.image(now=1e6).sum() == 4
    acc.clear()
    assert acc.image().sum() == 0 and acc.total == 0


def test_render_is_8bit_with_every_hit_visible():
    acc = PersistenceAccumulator(resolution=8, retention_s=0)
    acc.add(np.r_[np.full(10000, -9.0), 9.0], np.r_[np.full(10000, -9.0), 9.0], np.zeros(10001, bool), now=0)
    f = acc.render(now=0)
    assert f.dtype == np.uint8 and f.max() == 255
    assert f[0, 7] >= 1 and (f > 0).sum() == 2


# ------------------------------------------------------------------ worker + backend
def test_worker_step_applies_settings():
    tap = ScanTap(1000)
    tap.enabled = True
    w = ScanMonitorWorker(tap, lambda f, s: None, resolution=32, retention_s=0)
    tap.push(*chunk(10, 1, 1))
    tap.push(*chunk(10, 2, 2, blank=True))
    frame, st, n = w.step(now=0)
    assert frame.shape == (32, 32) and n == 20 and st.unblanked == 0.5 and st.total == 10
    w.configure(resolution=64)
    frame, st, _ = w.step(now=0)
    assert frame.shape == (64, 64)
    w.clear()
    tap.push(*chunk(3))
    frame, st, _ = w.step(now=0)
    assert st.total == 3


def test_worker_thread_delivers_frames():
    tap = ScanTap(1 << 14)
    frames = []
    w = ScanMonitorWorker(tap, lambda f, s: frames.append((f, s)), resolution=16, retention_s=5, interval_s=0.02)
    w.start()
    try:
        assert tap.enabled
        tap.push(*chunk(100, 3, 3))
        deadline = time.time() + 3
        while time.time() < deadline and not any(s.total for _f, s in frames):
            time.sleep(0.01)
        assert any(s.total == 100 for _f, s in frames)
    finally:
        w.stop()
    assert not tap.enabled and not w.running


def test_stream_feeds_the_tap_exactly_as_written():
    be = SimulatedBackend(DAQConfig(), SimParams(realtime=False))
    tap = ScanTap(1 << 16)
    tap.enabled = True
    be.output_tap = tap
    mgr = DAQManager(be).open()
    x = np.linspace(-8, 8, 5000)
    blank = (np.arange(5000) % 3) == 0
    mgr.get_handle("t").run_stream(100e3, [OutputChunk(x, -x, blank)])
    tx, ty, tb = tap.pop()
    np.testing.assert_allclose(tx, x.astype(np.float32))
    np.testing.assert_allclose(ty, (-x).astype(np.float32))
    np.testing.assert_array_equal(tb, blank)
    mgr.close()


def test_a_broken_tap_never_disturbs_the_stream():
    class Broken:
        def push(self, *a):
            raise RuntimeError("boom")

    be = SimulatedBackend(DAQConfig(), SimParams(realtime=False))
    be.output_tap = Broken()
    mgr = DAQManager(be).open()
    r = mgr.get_handle("t").run_stream(100e3, [OutputChunk.constant(1000, 0, 0, False)])
    assert r.samples_written == 1000
    mgr.close()


def test_nidaq_writer_feeds_the_tap(monkeypatch):
    from easyebl.hal.nidaq import NIDAQBackend

    from . import fake_nidaqmx

    fake = fake_nidaqmx.install(monkeypatch)
    be = NIDAQBackend(DAQConfig(buffer_samples=5000))
    tap = ScanTap(1 << 16)
    tap.enabled = True
    be.output_tap = tap
    mgr = DAQManager(be).open()
    x = np.linspace(-1, 1, 3000)
    mgr.get_handle("t").run_stream(200e3, [OutputChunk(x, -x, np.zeros(3000, bool))] * 3)
    tx, _ty, tb = tap.pop()
    n = len(fake.ao_x)
    assert len(tx) == n  # every sample written to the device, safe tail included
    np.testing.assert_allclose(tx, np.asarray(fake.ao_x, dtype=np.float32))
    assert not tb[:9000].any() and tb[9000:].all()  # the safe tail is blanked
    mgr.close()


@pytest.mark.parametrize("bad", [0, 1])
def test_bad_sizes(bad):
    with pytest.raises(ValueError):
        ScanTap(bad - 1) if bad == 0 else PersistenceAccumulator(resolution=bad)


@pytest.mark.parametrize("ix", [False, True])
@pytest.mark.parametrize("iy", [False, True])
@pytest.mark.parametrize("sw", [False, True])
def test_orientation_matches_the_image_coordinates(ix, iy, sw):
    """Invert / swap of the outputs are undone: a field point lands where the SEM image shows it."""
    from easyebl.core.calibration import FieldCalibration

    cal = FieldCalibration(100.0, 10.0, invert_x=ix, invert_y=iy, swap_xy=sw)
    v = cal.um_to_volts(np.array([[30.0, 20.0]]))  # upper right of the field: +X, +Y
    acc = PersistenceAccumulator(resolution=10, retention_s=0, invert_x=ix, invert_y=iy, swap_xy=sw)
    acc.add(v[:, 0], v[:, 1], np.zeros(1, bool), now=0)
    img = acc.image(now=0)
    # field (+30 µm, +20 µm) of a 100 µm field = (+6 V, +4 V) → column 8, row 10-1-7 = 2
    assert img[2, 8] == 1 and img.sum() == 1


def test_worker_orientation_change_clears_and_rebins():
    tap = ScanTap(100)
    tap.enabled = True
    w = ScanMonitorWorker(tap, lambda f, s: None, resolution=10, retention_s=0)
    tap.push(np.array([6.0]), np.array([4.0]), np.zeros(1, bool))
    w.step(now=0)
    assert w.acc.image(now=0)[2, 8] == 1
    w.configure(orientation=(True, False, True))
    tap.push(np.array([-4.0]), np.array([6.0]), np.zeros(1, bool))  # field (+6, +4) after invert X + swap
    w.step(now=0)
    img = w.acc.image(now=0)
    assert w.acc.orientation == (True, False, True) and img[2, 8] == 1 and img.sum() == 1


# ------------------------------------------------------------------ current beam position
def test_position_follows_static_writes():
    tap = ScanTap(100)
    assert tap.position() is None
    tap.set_static(1.0, -2.0, True)
    assert tap.position() == (1.0, -2.0, True)
    tap.set_static(3.0, 4.0, False)
    assert tap.position() == (3.0, 4.0, False)


def test_position_follows_the_generated_samples_not_the_writer():
    tap = ScanTap(10000)
    tap.enabled = True
    tap.set_static(-9.0, -9.0, True)
    tap.begin_stream()
    x = np.arange(1000, dtype=float) / 100  # sample k at x = k / 100
    blank = np.arange(1000) >= 500
    tap.push(x, -x, blank)  # written far ahead of the beam
    tap.push(x + 0.5, -x, np.ones(1000, bool))
    assert tap.position() == (-9.0, -9.0, True)  # nothing played yet: still where it was
    tap.set_generated(1)
    assert tap.position()[0] == pytest.approx(0.0)
    tap.set_generated(400)  # beam at sample 399: on, marks every ≤ 63 samples
    px, py, pb = tap.position()
    assert 3.99 - 0.64 <= px <= 3.99 and py == pytest.approx(-px) and not pb
    tap.set_generated(1000)  # last sample of the first chunk: blanked
    assert tap.position() == pytest.approx((9.99, -9.99, True))
    tap.set_generated(2000)
    assert tap.position()[0] == pytest.approx(10.49) and tap.position()[2]
    tap.set_static(0.0, 0.0, True)  # the stream ended: parked
    assert tap.position() == (0.0, 0.0, True)


def test_spot_in_image_coordinates():
    tap = ScanTap(100)
    w = ScanMonitorWorker(tap, lambda f, s: None, resolution=8, orientation=(True, False, True))
    tap.set_static(-4.0, 6.0, False)  # outputs for field (+6, +4) with invert X + swap
    _f, st, _n = w.step(now=0)
    assert st.spot == pytest.approx((6.0, 4.0, False))


def test_backend_reports_the_beam_position():
    be = SimulatedBackend(DAQConfig(), SimParams(realtime=False))
    mgr = DAQManager(be).open()
    tap = ScanTap(1 << 16)
    be.output_tap = tap  # attached after open: it learns the parked state at once
    assert tap.position() == (mgr.config.park_x_v, mgr.config.park_y_v, True)
    h = mgr.get_handle("t")
    h.park(2.0, 3.0, blank=False)
    assert tap.position() == (2.0, 3.0, False)
    seen = []
    tap.enabled = True

    def chunks():
        for k in range(5):
            yield OutputChunk.constant(1000, k, -k, blank=(k % 2 == 1))
            seen.append(tap.position())

    h.run_stream(100e3, chunks())
    # each chunk is generated before the next is asked for: the spot is on the chunk just played
    assert seen == [(float(k), float(-k), k % 2 == 1) for k in range(5)]
    assert tap.position() == (mgr.config.park_x_v, mgr.config.park_y_v, True)  # parked afterwards
    mgr.close()


def test_nidaq_reports_generated_samples(monkeypatch):
    from easyebl.hal.nidaq import NIDAQBackend

    from . import fake_nidaqmx

    fake_nidaqmx.install(monkeypatch)
    be = NIDAQBackend(DAQConfig(buffer_samples=5000))
    calls = []

    class Spy(ScanTap):
        def set_generated(self, n):
            calls.append(n)
            super().set_generated(n)

    tap = Spy(1 << 16)
    tap.enabled = True
    be.output_tap = tap
    mgr = DAQManager(be).open()
    assert tap.position()[2]  # parked, blanked
    mgr.get_handle("t").run_stream(200e3, [OutputChunk.constant(3000, 1.0, 1.0, False)] * 3)
    assert calls and calls == sorted(calls) and calls[-1] >= 9000
    assert tap.position() == (mgr.config.park_x_v, mgr.config.park_y_v, True)
    mgr.close()
