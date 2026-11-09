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
    h.claim("relay_pfi6", "relay_pfi7")
    h.set_line("relay_pfi6", True)
    h.set_line("relay_pfi7", True)
    assert mgr.line_state == {"relay_pfi6": True, "relay_pfi7": True}
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
