"""Pulse-processor HAL: the DP5 adapter (driver mock) and the simulated DPP fed by the EDS phantom."""
import threading

import numpy as np
import pytest

from easyebl.core import OutputChunk
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend
from easyebl.hal.mca import (
    DP5PulseProcessor,
    MCABusyError,
    MCAReading,
    SimMCAParams,
    SimulatedPulseProcessor,
    create_pulse_processor,
)

FS = 1.25e6


# ------------------------------------------------------------------ DP5 (mock transport)
@pytest.fixture
def dp5():
    p = DP5PulseProcessor(mock=True, mock_cps=20e3).open()
    yield p
    p.close()


def test_dp5_open_info_and_config(dp5):
    assert dp5.is_open and dp5.channels > 0
    info = dp5.info()
    assert info["mock"] is True and info["serial"]
    dp5.configure({"MCAC": 2048})
    assert dp5.channels == 2048
    assert dp5.get_settings(["MCAC"])["MCAC"].startswith("2048")


def test_dp5_preset_real_time_stops_the_mca(dp5):
    dp5.set_preset_real_time(0.05)
    assert dp5.preset_real_time == 0.05
    dp5.clear()
    dp5.enable()
    st = dp5.wait_done(timeout=3.0, expected_s=0.05)
    assert not st.mca_enabled and st.preset_done
    r = dp5.read(clear=True)
    assert r.real_time == pytest.approx(0.05, abs=2e-3)
    assert r.counts.size == dp5.channels and r.total > 0
    after = dp5.status()
    assert after.real_time == 0 and not after.preset_done


def test_dp5_acquire_with_updates_and_cancel(dp5):
    seen = []
    r = dp5.acquire(0.2, on_update=seen.append, update_s=0.05)
    assert r.real_time == pytest.approx(0.2, abs=5e-3) and seen and seen[-1] is r
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    r = dp5.acquire(5.0, cancel=cancel)
    assert r.real_time < 2.0


def test_session_is_exclusive(dp5):
    with dp5.session("eds"):
        with dp5.session("eds"):  # re-entrant
            assert dp5.owner == "eds"
        with pytest.raises(MCABusyError):
            with dp5.session("other"):
                pass
    assert dp5.owner is None


def test_reading_rates():
    r = MCAReading(np.zeros(4, np.int64), real_time=2.0, acc_time=1.9, fast_count=2000, slow_count=1500,
                   mca_enabled=False, preset_done=True)
    assert r.icr == 1000 and r.ocr == 750 and r.dead_fraction == pytest.approx(0.25)
    assert r.live_time == pytest.approx(1.5)


# ------------------------------------------------------------------ simulated DPP
class FakeSource:
    """Photon source: Mn Kα (5.895 keV) at a fixed rate on a virtual sample clock."""

    def __init__(self, cps, seed=0, energy=5.895):
        self.cps, self.e, self.t = cps, energy, 0.0
        self.rng = np.random.default_rng(seed)
        self.listeners = []

    def add_photon_listener(self, fn):
        self.listeners.append(fn)

    def remove_photon_listener(self, fn):
        self.listeners.remove(fn)

    def run(self, seconds, chunk=65536):
        n_total = int(round(seconds * FS))
        while n_total > 0:
            n = min(chunk, n_total)
            k = self.rng.poisson(self.cps / FS, n)
            idx = np.repeat(np.arange(n), k)
            e = np.full(idx.size, self.e)
            for fn in list(self.listeners):
                fn(self.t, 1 / FS, idx, e, n)
            self.t += n / FS
            n_total -= n


def test_sim_dpp_preset_resolution_and_calibration():
    src = FakeSource(5e3)
    p = SimMCAParams(fwhm_mn_ev=130.0, escape_fraction=0.0)
    dpp = SimulatedPulseProcessor(src, p).open()
    dpp.set_preset_real_time(1.0)
    dpp.clear()
    dpp.enable()
    src.run(1.5)
    r = dpp.read()
    assert r.preset_done and not r.mca_enabled
    assert r.real_time == pytest.approx(1.0, abs=1e-6)  # stops inside a chunk, like the DP5
    assert r.fast_count == pytest.approx(5e3, rel=0.05)
    ch = np.arange(r.counts.size)
    c = np.where(np.abs(ch - dpp.true_channel(5.895)) < 40, r.counts, 0)  # the peak only (no sum peak)
    mu = (c * ch).sum() / c.sum()
    assert mu == pytest.approx(dpp.true_channel(5.895), abs=0.3)
    sigma = np.sqrt((c * (ch - mu) ** 2).sum() / c.sum()) * p.kev_per_channel
    assert sigma * 2.3548 == pytest.approx(0.130, rel=0.06)
    # no time passes without a stream
    dpp.clear()
    dpp.enable()
    assert dpp.status().real_time == 0
    dpp.close()
    assert src.listeners == []


def test_sim_dpp_dead_time_and_pileup():
    src = FakeSource(100e3, seed=2)
    dpp = SimulatedPulseProcessor(src, SimMCAParams(escape_fraction=0.0)).open()
    dpp.set_preset_real_time(0.5)
    dpp.clear()
    dpp.enable()
    src.run(0.6)
    r = dpp.read()
    tau = dpp.dead_time_s / 1.1 * 0.5
    expected_ocr = r.icr * np.exp(-2 * r.icr * tau)
    assert r.ocr == pytest.approx(expected_ocr, rel=0.05)
    assert 0.3 < r.dead_fraction < 0.9
    # photons closer than the fast resolving time merge into one pulse: a sum peak at 2 × 5.895 keV
    k = int(round(dpp.true_channel(2 * 5.895)))
    frac = r.counts[k - 30:k + 31].sum() / r.total
    assert 0.3 < frac / (r.icr * dpp.params.fast_resolving_us * 1e-6) < 2.0


def test_sim_dpp_through_the_hal_stream():
    cfg = DAQConfig.usb6363()
    cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output = "aux0", "eds", "aux1"
    mgr = DAQManager(SimulatedBackend(cfg, SimParams())).open()
    try:
        dpp = create_pulse_processor("sim", manager=mgr).open()
        dpp.set_preset_real_time(0.2)
        dpp.clear()
        dpp.enable()
        h = mgr.get_handle("eds")
        h.run_stream(FS, iter([OutputChunk.constant(65536, 0.0, 0.0, blank=False) for _ in range(5)]),
                     ai_channels=[])
        r = dpp.read()
        assert r.preset_done and r.real_time == pytest.approx(0.2, abs=1e-6)
        rate_cu = mgr.backend.eds_phantom().phase_rates[2]
        assert r.icr == pytest.approx(rate_cu, rel=0.1)
        # Cu Kα in its channel
        k = int(round(dpp.true_channel(8.047)))
        assert r.counts[k - 10:k + 11].sum() > 20 * r.counts[k + 40:k + 61].sum() / 1 + 10
        # the processor survives a phantom rebuild (listeners live on the backend)
        h.set_phantom(eds={"beam_kv": 25.0})
        dpp.clear()
        dpp.enable()
        h.run_stream(FS, iter([OutputChunk.constant(65536, 0.0, 0.0, blank=False)]), ai_channels=[])
        assert dpp.read().fast_count > 0
    finally:
        mgr.close()
