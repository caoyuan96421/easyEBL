"""Microscope HAL (Hitachi S-3000 over RS-232) on the simulator: shared link, SEM and stage modules."""
import concurrent.futures
import threading
import time

import pytest

from easyebl.core import FieldCalibration
from easyebl.microscope import (MAGNIFICATIONS, POLL, SEM_SET, MicroscopeLink, NotAvailable, SEMControl,
                                StageControl, StubMicroscope)
from easyebl.microscope.coupling import FieldFollower
from easyebl.microscope.s3000 import MotionError


@pytest.fixture
def mic():
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    sem = SEMControl(link, status_poll_s=0)
    stage = StageControl(link, polling=False)
    sem.refresh().result(10)
    stage.refresh().result(10)
    yield link, sem, stage
    link.close()


def _block(link, seconds, priority=POLL):
    """Occupy the port for a while (a slow transaction in progress)."""
    started = threading.Event()

    def fn(_s):
        started.set()
        time.sleep(seconds)

    f = link.submit(fn, priority, "busy")
    assert started.wait(5)
    return f


def test_readbacks_and_settings(mic):
    link, sem, stage = mic
    assert sem.magnification() == 1000 and sem.accelerating_voltage_kv() == 20.0
    assert sem.working_distance_mm() == 15 and sem.connected
    sem.set_magnification("2.5k").result(10)
    assert sem.magnification() == 2500 and link.session.simulator.sem["MAG"] == "2.5k"
    sem.set_hv(15).result(10)
    assert sem.accelerating_voltage_kv() == 15.0
    sem.hv_off().result(10)
    assert sem.accelerating_voltage_kv() is None and sem.state["hv_on"] is False
    sem.set_beam("C", 42).result(10)
    sem.set_focus("F", 1000).result(10)
    sem.stigmate(1, -1).result(10)
    assert sem.state["beam_coarse"] == 42 and sem.state["focus_fine"] == 1000
    assert sem.state["stig_x"] == 2048 + sem.stig_step and sem.state["stig_y"] == 2048 - sem.stig_step
    with pytest.raises(ValueError):
        sem.set_magnification(1234)  # not in the instrument's list: refused before sending
    with pytest.raises(ValueError):
        sem.set_beam("C", 100)
    snap = sem.snapshot()
    assert snap["mag"] == 2500 and "hv_setpoint_kv" in snap


def test_sem_settings_go_before_polling(mic):
    link, sem, stage = mic
    busy = _block(link, 0.3)
    polls = [link.submit(lambda s: s.sem.read("MAG"), POLL, f"poll {i}") for i in range(4)]
    stage.refresh(priority=POLL)
    sem.set_contrast(1000).result(10)
    for p in polls:
        p.result(10)
    busy.result(5)
    order = [w for _, w in link.executed]
    i_busy, i_set = order.index("busy"), order.index("CONT 1000")
    assert i_set == i_busy + 1  # the setting jumped every queued poll
    assert all(order.index(f"poll {i}") > i_set for i in range(4))


def test_repeated_settings_coalesce(mic):
    link, sem, stage = mic
    busy = _block(link, 0.3, priority=SEM_SET)
    futs = [sem.step_magnification(+1) for _ in range(6)]
    futs[-1].result(10)
    busy.result(5)
    sent = [t for t in link.session.simulator.received if t.startswith("MAG")]
    assert len(sent) == 1  # six wheel steps, one transaction with the final value
    assert sem.magnification() == MAGNIFICATIONS[MAGNIFICATIONS.index(1000) + 6]


def test_stage_poll_is_never_faster_than_20_s(mic):
    link, sem, stage = mic
    assert stage.set_poll_interval(5) == 20.0 and stage.poll_interval_s == 20.0
    assert stage.set_poll_interval(45) == 45.0
    s2 = StageControl(link, poll_s=1, polling=False)
    assert s2.poll_interval_s == StageControl.MIN_POLL_S == 20.0
    s2.close()


def test_stage_polls_in_the_background(monkeypatch):
    monkeypatch.setattr(StageControl, "MIN_POLL_S", 0.3)  # fast for the test only
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    try:
        stage = StageControl(link, poll_s=0.3)
        t0 = time.monotonic()
        while sum(1 for _, w in link.executed if w.startswith("stage read")) < 6 and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        reads = [(p, w) for p, w in link.executed if w.startswith("stage read")]
        assert len(reads) >= 6 and all(p == POLL for p, _ in reads)
        assert stage.positions["X"] == pytest.approx(0x17077D / 25.6)
        assert stage.positions["Z"] == pytest.approx(0x051BEB / 22.32)  # ≈ 15 mm = WD
    finally:
        link.close()


def test_stage_moves_relative_and_absolute(mic):
    link, sem, stage = mic
    x0, y0 = stage.positions["X"], stage.positions["Y"]
    pos = stage.move({"X": 250.0}, relative=True).result(30)
    assert pos["X"] == pytest.approx(x0 + 250.0, abs=0.05) and pos["Y"] == pytest.approx(y0, abs=0.05)
    pos = stage.move({"X": 50000.0, "Y": 30000.0}).result(60)
    assert pos["X"] == pytest.approx(50000.0, abs=0.05) and pos["Y"] == pytest.approx(30000.0, abs=0.05)
    pos = stage.move({"T": 2.0}, relative=True).result(30)
    assert pos["T"] == pytest.approx(2.0 + 0x62 / 3528, abs=1e-3)
    assert link.session.simulator.sem["STAGE"] == "ON"  # #STAGE ON was sent automatically
    with pytest.raises(ValueError):
        stage.move({"Q": 1})
    with pytest.raises(ValueError):
        stage.move({"X": 0.0}, relative=True)
    with pytest.raises(ValueError, match="below the stage zero"):
        stage.move({"X": -1e9}, relative=True).result(30)


@pytest.mark.parametrize("delay", [0.0, 0.03, 0.4])
def test_stop_halts_at_any_moment(mic, delay):
    link, sem, stage = mic
    x0 = stage.positions["X"]
    m = stage.move({"X": 20000.0}, relative=True)  # ~10 s at the simulated speed
    time.sleep(delay)
    stage.stop().result(5)
    with pytest.raises((MotionError, concurrent.futures.CancelledError)):
        m.result(15)
    stage.refresh(["X"]).result(10)
    assert stage.positions["X"] - x0 < 2000.0
    assert not stage.moving


def test_stop_is_not_queued_behind_sem_work(mic):
    link, sem, stage = mic
    m = stage.move({"X": 20000.0}, relative=True)
    t0 = time.monotonic()
    while not stage.moving and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    busy = _block(link, 2.0, priority=SEM_SET)  # a long SEM transaction holds the port
    t = time.monotonic()
    stage.stop().result(5)
    assert time.monotonic() - t < 1.5  # S0 went out while the port was busy
    with pytest.raises(MotionError):
        m.result(15)
    busy.result(5)


def test_timeout_makes_the_link_uncertain_until_resync(mic):
    link, sem, stage = mic
    link.session.sem.timeouts.default = 0.3
    link.session.simulator.faults.drop_next_reply = 1
    with pytest.raises(Exception):
        sem.set_contrast(500).result(10)
    assert link.state == "uncertain"
    with pytest.raises(Exception):  # refused until checked
        sem.set_contrast(600).result(10)
    link.resync()
    assert link.state == "connected"
    sem.set_contrast(700).result(10)
    assert sem.state["contrast"] == 700


def test_stub_microscope():
    s = StubMicroscope()
    assert s.magnification() is None and not s.connected
    with pytest.raises(NotAvailable):
        s.step_magnification(1)


class _Target:
    def __init__(self, field):
        self.calibration = FieldCalibration(field, 5.0)

    def set_calibration(self, cal):
        self.calibration = cal


def test_field_follows_magnification(mic):
    link, sem, stage = mic
    eng, eds = _Target(100.0), _Target(80.0)
    events = link.events.subscribe()
    f = FieldFollower(sem, [eng, eds])
    assert f.k_um == pytest.approx(100.0 * 1000)  # adopted from the field at the current ×1k
    sem.set_magnification(2000).result(10)
    assert eng.calibration.field_size_um == pytest.approx(50.0)
    assert eds.calibration.field_size_um == pytest.approx(50.0)
    # the field is calibrated by hand at ×2k: K follows
    eng.calibration = FieldCalibration(60.0, 5.0)
    f.adopt()
    assert f.k_um == pytest.approx(120000.0) and eds.calibration.field_size_um == pytest.approx(60.0)
    sem.set_magnification(500).result(10)
    assert eng.calibration.field_size_um == pytest.approx(240.0)
    assert any("field" in ev.message for ev in events.drain() if ev.kind == "info")
    f.close()


def test_beam_presets(mic, tmp_path):
    from easyebl.microscope import PRESET_KEYS
    from easyebl.microscope.settings import MicroscopeSettings

    link, sem, stage = mic
    sim = link.session.simulator
    sim.sem.update({"BEAM C": "33", "BEAM F": "44", "STG X": "2100", "STG Y": "1990", "CONT": "1500", "BRT": "1700"})
    got = sem.read_preset().result(10)  # read fresh from the SEM, not from the cache
    assert got == {"beam_coarse": 33, "beam_fine": 44, "stig_x": 2100, "stig_y": 1990,
                   "contrast": 1500, "brightness": 1700}
    preset = {"name": "low current", "beam_coarse": 60, "beam_fine": 10, "stig_x": 2000, "stig_y": 2050,
              "contrast": 1800, "brightness": 1900}
    n0 = len(sim.received)
    assert sem.apply_preset(preset).result(10)["beam_coarse"] == 60
    sent = sim.received[n0:]
    assert sent == ["BEAM C 60", "BEAM F 10", "STG X 2000", "STG Y 2050", "CONT 1800", "BRT 1900"]
    assert {k: sem.state[k] for k in PRESET_KEYS} == {k: preset[k] for k in PRESET_KEYS}
    # a bad or incomplete preset is refused before anything is sent
    n0 = len(sim.received)
    with pytest.raises(ValueError):
        sem.apply_preset({**preset, "contrast": 5000})
    with pytest.raises(ValueError):
        sem.apply_preset({"beam_coarse": 10})
    assert len(sim.received) == n0
    # stored with the microscope settings
    s = MicroscopeSettings(path=str(tmp_path / "m.json"))
    s.beam_presets.append(preset)
    s.save()
    assert MicroscopeSettings.load(str(tmp_path / "m.json")).beam_presets == [preset]
