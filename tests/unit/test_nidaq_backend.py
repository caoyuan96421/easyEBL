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
    r = h.run_stream(200e3, stream(sync=True), ai_channels=["video"])
    assert r.samples_written == total and not r.aborted
    # AI loopback == AO X, sample-for-sample, padding trimmed
    np.testing.assert_allclose(r.ai[0], fake.ao_x[:total])
    # tail of safe samples appended after the real data
    assert len(fake.ao_x) > total and np.all(fake.ao_x[total:] == mgr.config.park_x_v)
    # DO words: blank on bit 0, marker on bit 3
    w = fake.do_words[:total]
    blank = np.concatenate([(np.arange(3000) % 5) == 0] * 7)
    sync = np.concatenate([(np.arange(3000) % 7) == 0] * 7)
    np.testing.assert_array_equal((w & 1).astype(bool), blank)
    np.testing.assert_array_equal(((w >> 3) & 1).astype(bool), sync)
    assert np.all(fake.do_words[total:] & 1)  # tail blanked
    # AI and DO slaved to the AO sample clock, regeneration disabled
    tasks = {t.name: t for t in fake.tasks if t.name.startswith("easyebl")}
    assert tasks["easyebl_ai"].timing_cfg["source"] == "/Dev1/ao/SampleClock"
    assert tasks["easyebl_do"].timing_cfg["source"] == "/Dev1/ao/SampleClock"
    assert tasks["easyebl_ao"].out_stream.regen_mode.name == "DONT_ALLOW_REGENERATION"
    # start order: slaves before the clock master; everything closed
    starts = [n for ev, n in fake.log if ev == "start"]
    assert starts.index("easyebl_ao") == len(starts) - 1
    assert {n for ev, n in fake.log if ev == "close"} >= {"easyebl_ao", "easyebl_do", "easyebl_ai"}
    # safe state written statically afterwards
    assert fake.static_ao[-1] == [mgr.config.park_x_v, mgr.config.park_y_v]
    assert fake.static_do[-1] is True  # blank_active_high


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


def test_abort(fake):
    mgr, be = make(fake)
    h = mgr.get_handle("t")

    def gen():
        for k in range(100):
            if k == 3:
                h.abort()
            yield OutputChunk(np.zeros(3000), np.zeros(3000), np.zeros(3000, bool))

    r = h.run_stream(100e3, gen())
    assert r.aborted and r.samples_written < 100 * 3000
    assert ("stop", "easyebl_ao") in fake.log
