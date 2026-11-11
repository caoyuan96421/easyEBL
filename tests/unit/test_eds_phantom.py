"""EDS phantom (image + reset-type SDD preamp signal), its HAL wiring and the X-ray model."""
import numpy as np
import pytest

from easyebl import xray
from easyebl.core import OutputChunk
from easyebl.hal import DAQConfig, DAQManager, DeviceBusyError, ResourceConflictError, SimParams, SimulatedBackend
from easyebl.hal.eds_phantom import EDSPhantomSpecimen, ResetPreamp, default_phases
from easyebl.xray import generation as gen

FS = 1.25e6


@pytest.fixture(scope="module")
def phantom():
    return EDSPhantomSpecimen(n=512, seed=1)


def eds_config(**kw):
    cfg = DAQConfig.usb6363()
    cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output = "aux0", "eds", "aux1"
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------- X-ray data / model
def test_xray_lines_and_edges():
    assert xray.z_of("fe") == 26 and xray.symbol(29) == "Cu"
    ka = xray.main_line("Cu", "K")
    assert ka.name == "Ka1" and abs(ka.energy - 8.047) < 0.01
    assert abs(xray.edge_energy("Cu", "K") - 8.979) < 0.01
    assert abs(xray.main_line("Au", "M").energy - 2.12) < 0.02
    assert [ln.element for ln in xray.lines_near(6.40, tol=0.03)][:1] == ["Fe"]
    assert xray.lines("Au", "L", beam_kv=10) == []  # Au L3 edge (11.9 keV) not excited
    # µ/ρ jumps at the Fe K edge
    assert xray.mac(7.2, "Fe") > 5 * xray.mac(7.0, "Fe")


def test_generation_model_sanity():
    cond = gen.Conditions(beam_kv=20)
    cu = dict((ln.name, i) for ln, i in gen.characteristic({"Cu": 1.0}, cond) if ln.family == "K")
    assert 5e-5 < cu["Ka1"] < 2e-4  # ≈ 1e-4 photons / electron / sr
    # a line can only be excited above its edge
    assert gen.family_intensity("Cu", "K", {"Cu": 1}, gen.Conditions(beam_kv=8.5)) == 0
    # more beam energy → more Cu K
    i15 = gen.family_intensity("Cu", "K", {"Cu": 1}, gen.Conditions(beam_kv=15))
    i25 = gen.family_intensity("Cu", "K", {"Cu": 1}, gen.Conditions(beam_kv=25))
    assert i25 > 2 * i15
    # matrix absorption: Al Kα in a Cu matrix is weaker per unit mass than in pure Al
    f_pure = gen.family_intensity("Al", "K", {"Al": 1}, cond)
    f_cu = gen.family_intensity("Al", "K", {"Al": 0.5, "Cu": 0.5}, cond) / 0.5
    assert f_cu < 0.9 * f_pure
    # detector: the window kills C K far more than Fe K; FWHM grows with energy
    det = gen.Detector()
    assert det.efficiency(0.277) < 0.6 * det.efficiency(6.4)
    assert det.fwhm_kev(5.895) == pytest.approx(0.130, rel=1e-6)
    assert det.fwhm_kev(1.0) < det.fwhm_kev(8.0)
    # Kramers continuum falls to zero at the beam energy
    c = gen.continuum(np.array([2.0, 10.0, 19.99, 25.0]), {"Cu": 1}, cond)
    assert c[0] > c[1] > c[2] > 0 and c[3] == 0


# ---------------------------------------------------------------- phantom
def test_phantom_phases_and_image(phantom):
    assert phantom.labels.shape == (512, 512)
    present = set(np.unique(phantom.labels))
    assert present == set(range(len(default_phases())))
    assert phantom.phase_at(0.0, 0.0) == 2  # Cu grid crosses at the centre
    assert phantom.phase_at(9.0, 0.0) == 0  # outside the field → substrate
    assert phantom.image.shape == phantom.labels.shape and 0 <= phantom.image.min() <= phantom.image.max() <= 1
    # brightest phase emits sdd_cps; every phase emits something
    assert phantom.phase_rates.max() == pytest.approx(phantom.cps)
    assert (phantom.phase_rates > 0).all()


def test_photons_follow_the_beam(phantom):
    ph = EDSPhantomSpecimen(n=512, seed=3, cps=50e3)
    n = 400_000
    got = []
    ph.add_photon_listener(lambda *a: got.append(a))
    # blanked: nothing
    idx, e = ph.photons(np.zeros(n), np.zeros(n), np.ones(n, bool), FS)
    assert idx.size == 0 and got[-1][4] == n
    # on the Cu grid: Poisson rate and Cu lines
    idx, e = ph.photons(np.zeros(n), np.zeros(n), np.zeros(n, bool), FS)
    exp = ph.phase_rates[2] * n / FS
    assert abs(idx.size - exp) < 5 * np.sqrt(exp)
    assert np.mean(np.abs(e - 8.04) < 0.02) > 0.1  # Cu Kα
    assert np.mean(np.abs(e - 0.93) < 0.03) > 0.1  # Cu L
    assert np.all((e > 0) & (e < ph.beam_kv))
    # listeners get stream-clock times
    assert got[-1][0] == pytest.approx(n / FS) and got[-1][1] == pytest.approx(1 / FS)
    # an Al pad: Al Kα, no Cu Kα
    yx = np.argwhere(ph.labels == 1)[0]
    L = ph.extent_v
    xv = -L + (yx[1] + 0.5) * 2 * L / ph.labels.shape[1]
    yv = L - (yx[0] + 0.5) * 2 * L / ph.labels.shape[0]
    assert ph.phase_at(xv, yv) == 1
    idx, e = ph.photons(np.full(n, xv), np.full(n, yv), np.zeros(n, bool), FS)
    assert np.mean(np.abs(e - 1.487) < 0.02) > 0.5 and np.mean(np.abs(e - 8.04) < 0.05) < 2e-3  # continuum only
    ph.set_cps(0)
    assert ph.photons(np.zeros(1000), np.zeros(1000), np.zeros(1000, bool), FS)[0].size == 0


def test_reset_preamp_steps_and_resets():
    pa = ResetPreamp(mv_per_kev=6.0, leak_v_per_s=0.0, noise_mv=0.0, rise_s=1 / FS, seed=0)
    n = 1000
    w = pa.waveform(n, FS, np.array([100, 600]), np.array([10.0, 5.0]))
    assert w[0] == pytest.approx(-4.5)
    assert w[300] - w[50] == pytest.approx(0.060, abs=1e-9)  # 10 keV × 6 mV/keV
    assert w[900] - w[300] == pytest.approx(0.030, abs=1e-9)
    # continuity across chunks + a reset when the ramp hits the top
    pa = ResetPreamp(mv_per_kev=6.0, leak_v_per_s=1e5, noise_mv=0.0, seed=0)  # 9 V in 90 µs
    a = pa.waveform(100, FS, np.zeros(0, int), np.zeros(0))
    b = pa.waveform(100, FS, np.zeros(0, int), np.zeros(0))
    assert b[0] - a[-1] == pytest.approx(1e5 / FS, rel=1e-6)
    w = np.concatenate([a, b] + [pa.waveform(100, FS, np.zeros(0, int), np.zeros(0)) for _ in range(3)])
    assert pa.n_resets >= 1 and w.max() < 4.5 + 1e-9 and w.min() >= -4.5 - 1e-9
    assert np.min(np.diff(w)) < -1.0  # the fast reset edge


def test_sdd_output_is_a_staircase(phantom):
    ph = EDSPhantomSpecimen(n=256, seed=5, cps=20e3, noise_mv=0.0, leak_v_per_s=0.0)
    n = 200_000
    w = ph.sdd_output(np.zeros(n), np.zeros(n), np.zeros(n, bool), FS)
    steps = np.diff(w)
    big = steps[steps > 0.5e-3] / 6e-3  # keV per step (one-sample rise spread over rise_s samples)
    assert ph.preamp.n_photons > 1000
    assert w.min() >= -4.5 - 1e-9 and w.max() <= 4.5 + 1e-9
    assert big.size > 0


# ---------------------------------------------------------------- HAL wiring
def test_config_channels_and_rate():
    cfg = eds_config()
    assert cfg.phantom_physical == "ao2" and cfg.sdd_physical == "ao3"
    assert cfg.phantom_outputs == {"aux0", "aux1"}
    assert cfg.stream_ao_channels == 4 and cfg.max_scan_rate() == pytest.approx(1.25e6)
    cfg.phantom_kind = "chart"
    assert cfg.sdd_physical is None and cfg.stream_ao_channels == 3
    with pytest.raises(ValueError):
        eds_config(sdd_output="aux0")
    with pytest.raises(ValueError):
        eds_config(phantom_kind="bogus")


def test_manager_routes_eds_phantom_and_protects_outputs():
    mgr = DAQManager(SimulatedBackend(DAQConfig.usb6363(), SimParams())).open()
    try:
        h = mgr.get_handle("app")
        h.set_phantom("aux0", kind="eds", sdd_output="aux1", eds={"cps": 5e3})
        ph = h.phantom
        assert (ph["kind"], ph["physical"], ph["sdd_physical"], ph["eds"]["cps"]) == ("eds", "ao2", "ao3", 5e3)
        assert mgr.backend.eds_phantom().cps == 5e3
        with pytest.raises(ResourceConflictError):
            h.set_aux_output("aux1", 1.0)  # the SDD signal owns AO3
        # the rate may change live; the model is rebuilt only for model parameters
        obj = mgr.backend.eds_phantom()
        h.set_phantom(eds={"cps": 8e3})
        assert mgr.backend.eds_phantom() is obj and obj.cps == 8e3
        h.set_phantom(eds={"beam_kv": 15.0})
        assert mgr.backend.eds_phantom() is not obj and mgr.backend.eds_phantom().beam_kv == 15.0
        # back to the chart: AO3 is free again
        h.set_phantom(kind="chart")
        assert mgr.backend.eds_phantom() is None and h.phantom["sdd_physical"] is None
        h.set_aux_output("aux1", 1.0)
        with pytest.raises(ResourceConflictError):
            h.set_phantom(kind="eds")  # aux1 is held by this handle's static value
    finally:
        mgr.close()


def test_stream_feeds_photon_listeners():
    mgr = DAQManager(SimulatedBackend(eds_config(), SimParams())).open()
    try:
        got = []
        mgr.backend.add_photon_listener(lambda t0, dt, idx, e, n: got.append((t0, n, idx.size)))
        h = mgr.get_handle("eds")
        chunks = [OutputChunk.constant(50_000, 0.0, 0.0, blank=False) for _ in range(4)]
        res = h.run_stream(FS, iter(chunks), ai_channels=["video"])
        assert sum(n for _, n, _ in got) >= 200_000
        assert sum(k for *_, k in got) > 0.5 * mgr.backend.eds_phantom().phase_rates[2] * 200_000 / FS
        # times are contiguous on the stream clock
        for (t0, n, _), (t1, *_r) in zip(got, got[1:]):
            assert t1 == pytest.approx(t0 + n / FS)
        assert res.ai.shape[0] == 1
        # the phantom image reaches the video input (AO2 → AI0 loopback in the simulator)
        assert res.ai[0, -1000:].mean() > 1.0
        with pytest.raises(DeviceBusyError):
            # routing cannot change while a stream runs
            import threading

            ev = threading.Event()

            def slow():
                for c in chunks * 50:
                    ev.set()
                    yield c

            t = threading.Thread(target=lambda: h.run_stream(FS, slow(), ai_channels=[]))
            mgr.backend.params.realtime = True
            t.start()
            ev.wait(5)
            try:
                mgr.get_handle("other").set_phantom(kind="chart")
            finally:
                h.abort()
                t.join(10)
    finally:
        mgr.close()
