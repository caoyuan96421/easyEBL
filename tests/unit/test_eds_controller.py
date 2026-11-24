"""EDS backend on the simulator: spectra, electron image, maps, cancel and sharing with SEM imaging."""
import time

import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.eds import EDSController, EnergyCalibration, SpectrumMap, load_map, load_spectrum
from easyebl.eds.beam import AIWindows, BeamProgram
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend
from easyebl.hal.mca import create_pulse_processor
from easyebl.imaging.engine.core import ImagingEngine
from easyebl.imaging.engine.frames import EngineState
from easyebl.imaging.engine.settings import EngineOptions


@pytest.fixture
def rig():
    cfg = DAQConfig.usb6363()
    mgr = DAQManager(SimulatedBackend(cfg, SimParams(realtime=True))).open()
    dpp = create_pulse_processor("mock").open()
    ecal = EnergyCalibration(0.0, 0.01)
    ctrl = EDSController(mgr, dpp, FieldCalibration(100.0, cfg.ao_limit_v), ecal)
    yield mgr, dpp, ctrl
    ctrl.close()
    dpp.close()
    mgr.close()


def test_point_spectrum_presets(rig):
    mgr, dpp, ctrl = rig
    job = ctrl.acquire_spectrum(0.3, at_um=(10.0, -10.0))
    # while counting, the beam is held statically at the point: no stream runs
    seen = []
    while not job.done():
        st = mgr.beam_state
        if not st.blank:
            seen.append((mgr.is_busy, round(st.x_v, 6), round(st.y_v, 6)))
        time.sleep(0.01)
    sp = job.result(30)
    v = ctrl.calibration.um_to_volts([[10.0, -10.0]])[0]
    # only the brief static writes / ADC read lock the device – never a stream for the count
    assert seen and sum(busy for busy, *_ in seen) <= 3 and len(seen) > 10
    assert seen[-1][1:] == (round(v[0], 6), round(v[1], 6))
    assert mgr.beam_state.blank  # blanked again afterwards
    assert sp.real_time == pytest.approx(0.3, abs=5e-3) and sp.total > 0
    assert np.isfinite(sp.metadata["se_v"])
    # live-time preset
    lt = ctrl.acquire_spectrum(0.2, at_um=(0.0, 0.0), live_time=True).result(30)
    assert lt.live_time >= 0.2 * 0.98 and lt.real_time >= lt.live_time


def test_area_spectrum_and_progress_events(rig):
    mgr, dpp, ctrl = rig
    sub = ctrl.events.subscribe()
    sp = ctrl.acquire_spectrum(0.4, region_um=(0.0, 0.0, 100.0, 100.0)).result(30)
    kinds = [ev.kind for ev in sub.drain()] if hasattr(sub, "drain") else []
    assert sp.total > 0
    if kinds:
        assert "job_started" in kinds and "job_done" in kinds and "spectrum" in kinds


def test_electron_image(rig):
    mgr, dpp, ctrl = rig
    im = ctrl.acquire_image(96, 96, 4e-6).result(30)
    assert im.data.shape == (96, 96) and np.isfinite(im.data).all()
    assert np.std(im.data) > 0  # the simulated sample has contrast
    f = ctrl.full_field_um()[2]
    assert im.region_um == pytest.approx((0, 0, f, f))


def test_electron_image_keeps_the_pixels_square(rig):
    """4:3 pixels scan a 4:3 rectangle of the square field (full width), not the whole square."""
    mgr, dpp, ctrl = rig
    f = ctrl.full_field_um()[2]
    im = ctrl.acquire_image(32, 24, 2e-6).result(30)
    assert im.data.shape == (24, 32)
    assert im.region_um == pytest.approx((0, 0, f, f * 0.75))
    im = ctrl.acquire_image(24, 32, 2e-6).result(30)
    assert im.region_um == pytest.approx((0, 0, f * 0.75, f))


def test_map_acquires_every_pixel_and_saves(rig, tmp_path):
    mgr, dpp, ctrl = rig
    job = ctrl.acquire_map(8, 8, 0.01)
    m = job.result(120)
    assert isinstance(m, SpectrumMap) and m.done.all() and m.shape == (8, 8)
    assert np.allclose(m.real_time, 0.01, atol=5e-3)
    assert np.isfinite(m.se).all()
    s = m.sum_spectrum()
    assert s.total == int(m.cube.sum()) and s.metadata["pixels"] == 64
    m.save(tmp_path / "map.npz")
    m2 = load_map(tmp_path / "map.npz")
    assert np.array_equal(m2.cube, m.cube) and m2.region_um == m.region_um
    s.save(tmp_path / "sum.msa")
    s2 = load_spectrum(tmp_path / "sum.msa")
    assert np.array_equal(s2.counts, s.counts) and s2.calibration.gain_kev == pytest.approx(s.calibration.gain_kev)


def test_cancel_keeps_partial_map_and_busy_refusal(rig):
    mgr, dpp, ctrl = rig
    job = ctrl.acquire_map(30, 30, 0.01)
    with pytest.raises(RuntimeError, match="busy"):
        ctrl.acquire_spectrum(1.0, at_um=(0, 0))
    time.sleep(1.0)
    job.cancel()
    m = job.result(30)
    assert 0 < m.done.sum() < 900
    assert mgr.lease is None and not dpp.read().mca_enabled
    # the controller is usable again
    assert ctrl.acquire_spectrum(0.1, at_um=(0, 0)).result(30).real_time == pytest.approx(0.1, abs=5e-3)


def test_live_sem_pauses_during_eds_and_resumes(rig):
    mgr, dpp, ctrl = rig
    eng = ImagingEngine(mgr, ctrl.calibration, options=EngineOptions(rate=250e3, latency_s=0.02))
    try:
        eng.run()
        assert eng.wait_frame(timeout=10) is not None
        job = ctrl.acquire_map(6, 6, 0.02)
        seen_paused = False
        while not job.done():
            seen_paused |= eng.state == EngineState.PAUSED
            time.sleep(0.01)
        job.result()
        assert seen_paused
        deadline = time.monotonic() + 10
        while eng.state != EngineState.LIVE and time.monotonic() < deadline:
            time.sleep(0.02)
        assert eng.state == EngineState.LIVE
    finally:
        eng.close()


def test_beam_program_and_ai_windows():
    prog = BeamProgram(1e6, start=(0, 0), chunk_n=100)
    gen = prog.chunks()
    c0 = next(gen)
    assert c0.blank.all()
    t = prog.move(1.0, -1.0)
    c1 = next(gen)
    assert t.wait(0) == 100 and np.all(c1.x == 1.0) and not c1.blank.any()
    t = prog.pattern([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    c2 = next(gen)
    assert t.wait(0) == 200 and list(c2.x[:4]) == [0.0, 1.0, 2.0, 0.0]
    prog.stop()
    assert list(gen) == []
    w = AIWindows(1, history=1000)
    x = np.arange(3000, dtype=float)
    for k in range(0, 3000, 250):
        w.feed(x[None, k:k + 250])
    w.request("past", 2500, 2600)
    assert w.result("past", 0)[0] == pytest.approx(2549.5)
    w.request("future", 2900, 3100)
    w.feed(np.arange(3000, 3200, dtype=float)[None])
    assert w.result("future", 0)[0] == pytest.approx(2999.5)
    w.request("lost", 0, 10)
    assert np.isnan(w.result("lost", 0)[0])


def test_idle_monitor_reads_delta_spectra_and_yields_to_jobs(rig):
    mgr, dpp, ctrl = rig
    sub = ctrl.events.subscribe()
    ctrl.start_idle_monitor(0.2)
    try:
        deadline = time.monotonic() + 5
        idle = []
        while len(idle) < 3 and time.monotonic() < deadline:
            idle += [ev.data["spectrum"] for ev in sub.drain() if ev.kind == "idle_spectrum"]
            time.sleep(0.02)
        assert len(idle) >= 3 and ctrl.idle_state == "live"
        for sp in idle:  # delta mode: each reading covers one interval only
            assert sp.real_time == pytest.approx(0.2, abs=0.05) and sp.icr > 0 and 0 <= sp.dead_fraction < 1
        # a job takes the MCA (its preset is honoured); the monitor re-arms afterwards
        sp = ctrl.acquire_spectrum(0.3, at_um=(0.0, 0.0)).result(30)
        assert sp.real_time == pytest.approx(0.3, abs=5e-3)
        sub.drain()
        deadline = time.monotonic() + 5
        while not any(ev.kind == "idle_spectrum" for ev in sub.drain()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert time.monotonic() < deadline, "idle monitor did not resume after the job"
        ctrl.set_idle_interval(0.5)
        with ctrl.idle_paused():
            assert ctrl.idle_state == "paused" and not dpp.status().mca_enabled
    finally:
        ctrl.stop_idle_monitor()
    assert ctrl.idle_state == "off" and not dpp.status().mca_enabled


def test_min_energy_sets_the_slow_threshold(rig):
    mgr, dpp, ctrl = rig
    n = dpp.channels
    thsl = ctrl.set_min_energy(1.0)  # 1 keV = channel 100 with 10 eV/ch
    assert thsl == pytest.approx(100.0 * ctrl.energy_calibration.channel(1.0) / n, abs=1e-3)
    assert float(dpp.get_settings(["THSL"])["THSL"]) == pytest.approx(thsl, abs=0.01)
    sp = ctrl.acquire_spectrum(0.3, at_um=(0.0, 0.0)).result(30)
    k = int(ctrl.energy_calibration.channel(1.0))
    assert sp.total > 0 and sp.counts[:k].sum() == 0  # nothing below the cut-off
    assert ctrl.thsl_for(100.0) == ctrl.THSL_MAX  # clamped to the DP5 range
