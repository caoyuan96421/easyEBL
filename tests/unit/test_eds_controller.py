"""EDS backend on the simulator: spectra, electron image, maps, cancel and sharing with SEM imaging."""
import time

import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.eds import EDSController, EDSOptions, EnergyCalibration, Spectrum, SpectrumMap, load_map, load_spectrum
from easyebl.eds.beam import AIWindows, BeamProgram, grid_points
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend
from easyebl.hal.mca import SimMCAParams, SimulatedPulseProcessor
from easyebl.imaging.engine.core import ImagingEngine
from easyebl.imaging.engine.frames import EngineState
from easyebl.imaging.engine.settings import EngineOptions


@pytest.fixture
def rig():
    cfg = DAQConfig.usb6363()
    cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output = "aux0", "eds", "aux1"
    cfg.eds_phantom = {"cps": 30e3, "n": 512}
    mgr = DAQManager(SimulatedBackend(cfg, SimParams(realtime=True))).open()
    p = SimMCAParams()
    dpp = SimulatedPulseProcessor.for_manager(mgr, p).open()
    ecal = EnergyCalibration(p.offset_kev + p.kev_per_channel / 2, p.kev_per_channel)
    ctrl = EDSController(mgr, dpp, FieldCalibration(100.0, cfg.ao_limit_v), ecal)
    yield mgr, dpp, ctrl
    ctrl.close()
    mgr.close()


_BG = {8.04: (9.2, 9.36)}  # line-free background windows (default: 3.6–3.76 keV)


def _peak(sp: Spectrum, e: float, w: float = 0.08) -> float:
    """Counts in a window on the line / counts in an equally wide line-free window."""
    lo, hi = _BG.get(e, (3.6, 3.76))
    return sp.counts_in(e - w, e + w) / max(1, sp.counts_in(lo, hi))


def _um_of_phase(mgr, ctrl, phase: int) -> tuple:
    ph = mgr.backend.eds_phantom()
    pts = grid_points((0, 0), 100.0, 100.0, 200, 200)
    v = ctrl._to_volts(pts)
    lab = ph.phase_at(v[:, 0], v[:, 1])
    # a point well inside the phase (all 4 neighbours at ±1 µm agree)
    for i in np.nonzero(lab == phase)[0]:
        x, y = pts[i]
        if max(abs(x), abs(y)) > 45:
            continue
        nb = ctrl._to_volts([[x + 1.5, y], [x - 1.5, y], [x, y + 1.5], [x, y - 1.5]])
        if np.all(ph.phase_at(nb[:, 0], nb[:, 1]) == phase):
            return float(x), float(y)
    raise AssertionError("phase not found")


def test_point_spectra_identify_phases(rig):
    mgr, dpp, ctrl = rig
    cu = ctrl.acquire_spectrum(0.4, at_um=_um_of_phase(mgr, ctrl, 2)).result(30)
    al = ctrl.acquire_spectrum(0.4, at_um=_um_of_phase(mgr, ctrl, 1)).result(30)
    assert cu.real_time == pytest.approx(0.4, abs=1e-6) and cu.icr > 5e3
    assert _peak(cu, 8.04) > 10 and _peak(cu, 1.487) < 2
    assert _peak(al, 1.487) > 10 and al.counts_in(7.96, 8.12) < 0.01 * al.counts_in(1.41, 1.57)
    assert cu.metadata["se_v"] > al.metadata["se_v"]  # Cu is brighter in SE
    # live-time preset
    lt = ctrl.acquire_spectrum(0.2, at_um=(0.0, 0.0), live_time=True).result(30)
    assert lt.live_time >= 0.2 and lt.real_time > lt.live_time


def test_area_spectrum_mixes_phases_and_progress_events(rig):
    mgr, dpp, ctrl = rig
    sub = ctrl.events.subscribe()
    sp = ctrl.acquire_spectrum(0.6, region_um=(0.0, 0.0, 100.0, 100.0)).result(30)
    kinds = [ev.kind for ev in sub.drain()] if hasattr(sub, "drain") else []
    assert sp.total > 1000
    for e in (1.74, 1.487, 8.04):  # Si, Al, Cu all present
        assert _peak(sp, e) > 3
    if kinds:
        assert "job_started" in kinds and "job_done" in kinds and "spectrum" in kinds


def test_electron_image_matches_the_phantom(rig):
    mgr, dpp, ctrl = rig
    im = ctrl.acquire_image(96, 96, 4e-6).result(30)
    assert im.data.shape == (96, 96) and np.isfinite(im.data).all()
    ph = mgr.backend.eds_phantom()
    pts = grid_points((0, 0), 100.0, 100.0, 96, 96)
    v = ctrl._to_volts(pts)
    exp = ph.intensity_v(v[:, 0], v[:, 1]).reshape(96, 96)
    r = np.corrcoef(exp.ravel(), im.data.ravel())[0, 1]
    assert r > 0.9


def test_map_matches_phases_and_saves(rig, tmp_path):
    mgr, dpp, ctrl = rig
    job = ctrl.acquire_map(12, 12, 0.01)
    m = job.result(120)
    assert isinstance(m, SpectrumMap) and m.done.all() and m.shape == (12, 12)
    assert np.allclose(m.real_time, 0.01, atol=1e-6)
    ph = mgr.backend.eds_phantom()
    v = ctrl._to_volts(grid_points((0, 0), 100.0, 100.0, 12, 12))
    lab = ph.phase_at(v[:, 0], v[:, 1]).reshape(12, 12)
    al = m.window_image(1.40, 1.57)
    si = m.window_image(1.66, 1.82)
    assert al[lab == 1].mean() > 4 * al[lab == 0].mean()
    assert si[lab == 0].mean() > 4 * si[lab == 1].mean()
    assert np.isfinite(m.se).all()
    se_al = m.se[lab == 1].mean()
    se_si = m.se[lab == 0].mean()
    assert se_al > se_si  # Al pads are brighter than the Si substrate
    s = m.sum_spectrum()
    assert s.total == int(m.cube.sum()) and s.metadata["pixels"] == 144
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
    assert ctrl.acquire_spectrum(0.1, at_um=(0, 0)).result(30).real_time == pytest.approx(0.1, abs=1e-6)


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
