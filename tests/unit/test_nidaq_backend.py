"""NIDAQBackend control flow against an in-memory fake of the nidaqmx API."""
import numpy as np
import pytest

from easyebl.core import OutputChunk
from easyebl.hal import DAQConfig, DAQManager, HALError
from easyebl.hal.nidaq import NIDAQBackend, _line_bit, _port_of

from . import fake_nidaqmx


@pytest.fixture
def fake(monkeypatch):
    return fake_nidaqmx.install(monkeypatch)


def make(fake, **cfg):
    be = NIDAQBackend(DAQConfig(buffer_samples=5000, **cfg))
    return DAQManager(be).open(), be


def stream(n_chunks=7, n=3000, sync=False):
    for k in range(n_chunks):
        x = np.linspace(-1, 1, n) + k
        mk = {"sync": (np.arange(n) % 7) == 0} if sync else {}
        yield OutputChunk(x, -x, (np.arange(n) % 5) == 0, mk)


def test_helpers():
    assert _line_bit("port0/line7") == 7 and _port_of("port0/line7") == "port0"
    with pytest.raises(HALError):
        _line_bit("port0")


def test_open_checks_device(fake):
    fake.name = "Dev9"
    with pytest.raises(HALError, match="not found"):
        NIDAQBackend(DAQConfig()).open()


def test_stream_alignment_and_cleanup(fake):
    mgr, be = make(fake, timed_lines={"sync": "port0/line3"})
    h = mgr.get_handle("t")
    total = 7 * 3000
    rate = 200e3
    r = h.run_stream(rate, stream(sync=True), ai_channels=["video"])
    assert r.samples_written == total and not r.aborted
    # AI loopback == AO X, sample-for-sample, padding trimmed
    np.testing.assert_allclose(r.ai[0], fake.ao_x[:total])
    # safe tail: hold the last position blanked (blanker latency), then park
    n_hold = round(mgr.config.tail_hold_s * rate)
    last = fake.ao_x[total - 1]
    np.testing.assert_allclose(fake.ao_x[total : total + n_hold], last)
    assert np.all(fake.ao_x[total + n_hold :] == mgr.config.park_x_v) and len(fake.ao_x) > total + n_hold
    assert np.all(fake.do_words[total:] & 1)  # tail blanked from its first sample
    # DO words: blank on bit 0, marker on bit 3
    w = fake.do_words[:total]
    blank = np.concatenate([(np.arange(3000) % 5) == 0] * 7)
    sync = np.concatenate([(np.arange(3000) % 7) == 0] * 7)
    np.testing.assert_array_equal((w & 1).astype(bool), blank)
    np.testing.assert_array_equal(((w >> 3) & 1).astype(bool), sync)
    # AI and DO slaved to the AO sample clock, regeneration disabled, explicit buffer ≥ 1 s
    tasks = {t.name: t for t in fake.tasks if t.name.startswith("easyebl")}
    assert tasks["easyebl_ai"].timing_cfg["source"] == "/Dev1/ao/SampleClock"
    assert tasks["easyebl_do"].timing_cfg["source"] == "/Dev1/ao/SampleClock"
    assert tasks["easyebl_ao"].out_stream.regen_mode.name == "DONT_ALLOW_REGENERATION"
    assert tasks["easyebl_ao"].out_stream.output_buf_size >= rate
    # start order: slaves before the clock master; stop order: clock master first; all closed
    starts = [n for ev, n in fake.log if ev == "start"]
    stops = [n for ev, n in fake.log if ev == "stop"]
    assert starts[-1] == "easyebl_ao" and stops[0] == "easyebl_ao"
    assert {n for ev, n in fake.log if ev == "close"} >= {"easyebl_ao", "easyebl_do", "easyebl_ai"}
    # static safe state afterwards: blank first, then park
    statics = [ev for ev, _ in fake.log if ev.startswith("static")]
    assert statics[-2:] == ["static_do", "static_ao"]
    assert fake.static_ao[-1] == [mgr.config.park_x_v, mgr.config.park_y_v]
    assert fake.static_do[-1] is True  # blank_active_high
    assert r.info["actual_rate"] == pytest.approx(rate)


def test_static_order(fake):
    mgr, be = make(fake)
    h = mgr.get_handle("t")
    fake.log.clear()
    h.park(1.0, 2.0, blank=False)  # move, then unblank
    h.park(0.0, 0.0, blank=True)  # blank, then move
    statics = [ev for ev, _ in fake.log if ev.startswith("static")]
    assert statics == ["static_ao", "static_do", "static_do", "static_ao"]


def test_large_chunks_are_rechunked(fake):
    mgr, be = make(fake)
    n = 250_000  # larger than the 100 kS/s × 1 s buffer
    mgr.get_handle("t").run_stream(100e3, [OutputChunk(np.zeros(n), np.zeros(n), np.zeros(n, bool))])
    sizes = [v for ev, v in fake.log if ev == "ao_write"]
    assert max(sizes) <= 100_000 // 4 and sum(sizes) > n


def test_rate_coercion_warns(fake):
    mgr, be = make(fake)
    with pytest.warns(RuntimeWarning, match="coerced"):
        r = mgr.get_handle("t").run_stream(840e3, stream(1, 100))
    assert r.info["actual_rate"] == pytest.approx(100e6 / 119)


def test_active_low_blanker(fake):
    mgr, be = make(fake, blank_active_high=False)
    mgr.get_handle("t").run_stream(100e3, stream(1, 100))
    w = fake.do_words[:100]
    np.testing.assert_array_equal((w & 1).astype(bool), ~((np.arange(100) % 5) == 0))
    assert fake.static_do[-1] is False


def test_short_stream_and_consumer(fake):
    mgr, be = make(fake)
    got = []
    r = mgr.get_handle("t").run_stream(100e3, stream(1, 10), ai_channels=["video"], on_ai=got.append)
    assert r.samples_written == 10
    np.testing.assert_allclose(np.concatenate(got, axis=1)[0], fake.ao_x[:10])


def test_abort_plays_safe_tail(fake):
    mgr, be = make(fake)
    h = mgr.get_handle("t")

    def gen():
        for k in range(100):
            if k == 30:
                h.abort()
            yield OutputChunk(np.full(3000, 2.0), np.zeros(3000), np.zeros(3000, bool))

    r = h.run_stream(100e3, gen())
    assert r.aborted and r.samples_written < 100 * 3000
    # the last generated samples are the clocked safe tail: blanked, and parked
    assert np.all(fake.do_words[r.samples_written :] & 1)
    assert fake.ao_x[-1] == mgr.config.park_x_v
    assert ("stop", "easyebl_ao") in fake.log


def test_task_creation_failure_cleans_up(fake, monkeypatch):
    mgr, be = make(fake)
    import nidaqmx

    orig = nidaqmx.Task

    def flaky(name=""):
        if name == "easyebl_do":
            raise RuntimeError("resource reserved")
        return orig(name)

    monkeypatch.setattr(nidaqmx, "Task", flaky)
    with pytest.raises(RuntimeError):
        mgr.get_handle("t").run_stream(100e3, stream(1, 100))
    assert ("close", "easyebl_ao") in fake.log
    assert mgr.beam_state.blank


def test_usb6363_preset_live_imaging(fake):
    """Live raster on the USB-6363 preset: P0.23 active-low blanker, PFI relay lines,
    frames streamed back-to-back (the fake loops AI back from AO X)."""
    from dataclasses import replace

    from easyebl.core import FieldCalibration
    from easyebl.imaging import RasterConfig, RasterImager

    cfg = replace(DAQConfig.usb6363(), buffer_samples=5000)
    mgr = DAQManager(NIDAQBackend(cfg)).open()
    h = mgr.get_handle("img")
    h.set_external_control(True)
    assert mgr.external_control is True
    cal = FieldCalibration(100.0, cfg.ao_limit_v)
    rc = RasterConfig(nx=16, ny=8, width_um=40, rate=1e6, dwell_s=4e-6, ai_latency_samples=0.0)
    frames = []
    n = RasterImager(h, cal).live(rc, lambda k, img: frames.append(img.image()), max_frames=3)
    assert n == 3
    # AI == AO X, so every row is the X staircase in volts
    xs = (np.arange(16) + 0.5) * 40 / 16 - 20
    for img in frames:
        np.testing.assert_allclose(img, np.tile(xs * cal.volts_per_um, (8, 1)), atol=1e-9)
    # blanker on bit 23, active low: unblanked pixel samples drive the bit high
    w = fake.do_words
    assert set(np.unique(w)) <= {0, 1 << 23} and (w == (1 << 23)).any()
    assert mgr.beam_state.blank


# ------------------------------------------------------------------ low-latency streaming
def test_latency_sets_buffer_and_never_overfills_before_start(fake):
    from easyebl.hal import StreamMonitor

    mgr, be = make(fake)
    mon = StreamMonitor()
    rate = 500e3
    n = 300_000
    r = mgr.get_handle("t").run_stream(rate, [OutputChunk(np.zeros(n), np.zeros(n), np.zeros(n, bool))],
                                       ai_channels=["video"], latency_s=0.05, monitor=mon)
    buf = int(0.05 * rate)
    assert r.info["buffer_samples"] == buf and r.info["piece_samples"] == buf // 8
    sizes = [v for ev, v in fake.log if ev == "ao_write"]
    assert max(sizes[:-1]) <= buf // 8  # data pieces; the last write is the safe tail
    assert fake.max_prestart_queue <= buf
    assert mon.buffer_samples == buf and mon.written == n and mon.ai_read >= n and not mon.running
    assert mon.latency_s == pytest.approx(0.05)


def test_latency_floor(fake):
    mgr, be = make(fake)
    assert be.buffer_samples(1e6, 0.001) == mgr.config.min_latency_samples
    assert be.buffer_samples(1e6, None) == 1_000_000  # conservative default: ≥ min_buffer_s


def test_short_stream_with_tail_larger_than_free_buffer(fake):
    """Prefill + safe tail exceed the buffer: the clock must start before the tail is queued."""
    mgr, be = make(fake)
    n = 15_000  # tail at 500 kS/s is 10 ms = 5 000 samples; buffer 16 384
    r = mgr.get_handle("t").run_stream(500e3, [OutputChunk(np.zeros(n), np.zeros(n), np.zeros(n, bool))],
                                       latency_s=0.001)
    assert r.samples_written == n and fake.max_prestart_queue <= mgr.config.min_latency_samples


@pytest.mark.parametrize("code,exc", [(-200290, "StreamUnderflowError"), (-200621, "StreamUnderflowError"),
                                      (-200279, "StreamOverflowError")])
def test_driver_errors_are_classified_and_cleaned_up(fake, code, exc):
    import easyebl.hal as hal

    mgr, be = make(fake)
    fake.fail_on_write = (5, code)
    with pytest.raises(getattr(hal, exc)):
        mgr.get_handle("t").run_stream(100e3, stream(20, 3000), ai_channels=["video"], latency_s=0.05)
    assert ("stop", "easyebl_ao") in fake.log and ("close", "easyebl_ai") in fake.log
    assert mgr.beam_state.blank and not mgr.is_busy
    statics = [ev for ev, _ in fake.log if ev.startswith("static")]
    assert statics[-2:] == ["static_do", "static_ao"]


def test_unknown_driver_error_passes_through(fake):
    import nidaqmx.errors

    mgr, be = make(fake)
    fake.fail_on_write = (2, -123)
    with pytest.raises(nidaqmx.errors.DaqError):
        mgr.get_handle("t").run_stream(100e3, stream(5, 3000))
    assert mgr.beam_state.blank


def test_ai_blocks_are_reused_consumer_must_copy(fake):
    mgr, be = make(fake)
    ids, got = set(), []

    def consume(block):
        ids.add(id(block.base) if block.base is not None else id(block))
        got.append(block.copy())

    r = mgr.get_handle("t").run_stream(100e3, stream(10, 3000), ai_channels=["video"], on_ai=consume,
                                       latency_s=0.05)
    np.testing.assert_allclose(np.concatenate(got, axis=1)[0], fake.ao_x[: r.samples_written])


def test_engine_on_usb6363_preset(fake):
    """The threaded engine on the NI backend: loopback image == X staircase; changes keep the stream."""
    import time
    from dataclasses import replace

    from easyebl.core import FieldCalibration
    from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine

    cfg = replace(DAQConfig.usb6363())
    mgr = DAQManager(NIDAQBackend(cfg)).open()
    cal = FieldCalibration(100.0, cfg.ao_limit_v)
    eng = ImagingEngine(mgr, cal, options=EngineOptions(latency_s=0.02))
    try:
        assert eng.rate == 2e6  # fastest clock of the 6363 with one video channel
        eng.update(nx=32, ny=16, width_um=40, dwell_s=2e-6, ai_latency_samples=0.0)
        eng.run()
        f = eng.wait_frame(complete=True, timeout=10)
        f = eng.wait_frame(f.seq, complete=True, timeout=10)
        xs = (np.arange(32) + 0.5) * 40 / 32 - 20
        np.testing.assert_allclose(f.image, np.tile(xs * cal.volts_per_um, (16, 1)), atol=1e-5)
        eng.update(nx=20, center_um=(4.0, 0.0))
        deadline = time.monotonic() + 10
        while f.config.nx != 20:
            f = eng.wait_frame(f.seq, complete=True, timeout=10)
            assert time.monotonic() < deadline
        f = eng.wait_frame(f.seq, complete=True, timeout=10)
        xs = 4.0 + (np.arange(20) + 0.5) * 40 / 20 - 20
        np.testing.assert_allclose(f.image, np.tile(xs * cal.volts_per_um, (f.config.ny, 1)), atol=1e-5)
        st = eng.status()
        assert st.streams == 1 and st.underruns == 0
        eng.freeze()
        deadline = time.monotonic() + 10
        while eng.state != EngineState.FROZEN:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert mgr.beam_state.blank
        w = fake.do_words
        assert set(np.unique(w)) <= {0, 1 << 23}  # P0.23 only, active low
    finally:
        eng.close()
        mgr.close()


def test_external_control_wires_and_polarity(fake):
    """One HAL function drives several wires, each with its own polarity, in a single write."""
    cfg = DAQConfig.usb6363()
    cfg.external_control_lines = {"PFI6": True, "PFI7": False}
    mgr = DAQManager(NIDAQBackend(cfg)).open()
    ext_tasks = lambda: [t for t in fake.tasks if [p for _, p in t.channels] == ["Dev1/PFI6", "Dev1/PFI7"]]
    # opening the device selects the SEM's internal scan
    assert mgr.external_control is False and len(ext_tasks()) == 1
    assert fake.static_do[-1] == [False, True]
    h = mgr.get_handle("sem")
    n_log = len(fake.log)
    h.set_external_control(True)
    assert fake.static_do[-1] == [True, False] and len(ext_tasks()) == 2
    # the beam was blanked (active-low blanker: level False) before the switch
    assert ("static_do", False) in fake.log[n_log:-1]
    k = len(fake.static_do)
    mgr.close()  # closing returns the scan to the SEM (then the backend blanks once more)
    assert [False, True] in fake.static_do[k:] and mgr.external_control is False


# ------------------------------------------------------------------ lost USB connection
def _wait(cond, timeout=5.0):
    import time

    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_unplug_mid_stream_is_device_lost_and_cleans_up(fake):
    from easyebl.hal import DeviceLostError

    mgr = DAQManager(NIDAQBackend(DAQConfig.usb6363()), auto_reconnect=False).open()
    h = mgr.get_handle("t")
    fake.unplug_after_writes = fake.ao_writes + 6
    with pytest.raises(DeviceLostError):
        h.run_stream(500e3, (OutputChunk.constant(3000, 0.5, -0.5, False) for _ in range(40)),
                     ai_channels=["video"], latency_s=0.05)
    # every task was stopped and closed even though the driver kept failing
    names = {"easyebl_ao", "easyebl_do", "easyebl_ai"}
    assert names <= {n for ev, n in fake.log if ev == "close"}
    assert not mgr.connected and not mgr.is_busy and mgr.active_owner is None
    assert not mgr.backend.probe()
    with pytest.raises(DeviceLostError):
        h.park()
    mgr.close()  # completes although the device is gone


def test_reconnect_resets_device_and_restores_external_control(fake):
    from easyebl.hal import DeviceLostError

    mgr = DAQManager(NIDAQBackend(DAQConfig.usb6363()), reconnect_interval_s=0.01).open()
    h = mgr.get_handle("sem")
    h.set_external_control(True)
    fake.present = False
    with pytest.raises(DeviceLostError):
        h.park()
    assert mgr.reconnecting and fake.resets == 0
    fake.present = True
    assert mgr.wait_connected(3)
    assert fake.resets == 1  # stale driver state cleared after the reconnect
    assert mgr.external_control is True and fake.static_do[-1] == [True, True]
    r = h.run_stream(500e3, (OutputChunk.constant(3000, 0.5, -0.5, False) for _ in range(3)),
                     ai_channels=["video"], latency_s=0.05)
    assert r.samples_written == 9000
    mgr.close()


def test_lost_codes_vs_other_errors(fake):
    from easyebl.hal.nidaq import _is_lost

    Err = fake_nidaqmx.DaqError
    assert _is_lost(Err("x", -201003)) and _is_lost(Err("x", -50405))
    assert _is_lost(Err("Device is no longer present in the system", -12345))
    assert not _is_lost(Err("buffer underflow", -200290)) and not _is_lost(RuntimeError("no longer present"))
    # an unknown driver error while the device still answers is *not* a lost device
    mgr, be = make(fake)
    fake.fail_on_write = (2, -99999)
    with pytest.raises(fake_nidaqmx.DaqError):
        mgr.get_handle("t").run_stream(100e3, stream(5, 3000))
    assert mgr.connected


def test_open_with_missing_device_waits_for_it(fake):
    fake.present = False
    fake.name = "Dev1"
    mgr = DAQManager(NIDAQBackend(DAQConfig.usb6363()), reconnect_interval_s=0.01)
    mgr.open(allow_missing=True)
    assert not mgr.connected
    fake.present = True
    assert mgr.wait_connected(3) and mgr.external_control is False
    mgr.close()


# ------------------------------------------------------------------ phantom on aux0 = AO2
def test_phantom_stream_on_ao2_and_aux_static(fake):
    """The phantom adds AO2 to the hardware-timed AO task; AI looped back from AO2 gives the chart."""
    import time

    from easyebl.core import FieldCalibration
    from easyebl.imaging.engine import EngineOptions, ImagingEngine

    cfg = DAQConfig.usb6363()
    cfg.phantom_output = "aux0"
    mgr = DAQManager(NIDAQBackend(cfg)).open()
    cal = FieldCalibration(100.0, cfg.ao_limit_v)
    fake.loopback_row = 2  # the wire AO2 -> AI0
    eng = ImagingEngine(mgr, cal, options=EngineOptions(latency_s=0.02))
    try:
        assert eng.rate == pytest.approx(100e6 / 65)  # 3 AO channels on the 6363
        eng.update(nx=64, ny=48, width_um=50, dwell_s=4e-6, ai_latency_samples=0.0, center_um=(5.0, -5.0))
        eng.run()
        f = eng.wait_frame(complete=True, timeout=10)
        f = eng.wait_frame(f.seq, complete=True, timeout=10)
        assert fake.stream_ao_channels == ["Dev1/ao0", "Dev1/ao1", "Dev1/ao2"]
        expected = mgr.backend.phantom_source().expected_image(f.config, cal)
        np.testing.assert_allclose(f.image, expected, atol=1e-6)
        eng.stop()
        assert eng.wait_idle(5)
        t0 = time.monotonic()
        while mgr.is_busy and time.monotonic() - t0 < 2:
            time.sleep(0.01)
        h = mgr.get_handle("aux")
        h.set_aux_output("aux1", 1.5)  # static write on AO3
        assert fake.static_aux[-1] == ("Dev1/ao3", 1.5)
    finally:
        eng.close()
        mgr.close()
    assert fake.static_aux[-1] == ("Dev1/ao3", 0.0)  # back to 0 V on close
