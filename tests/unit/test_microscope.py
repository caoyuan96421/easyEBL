"""Microscope HAL (Hitachi S-3000 over RS-232) on the simulator: shared link, SEM and stage modules."""
import concurrent.futures
import threading
import time

import pytest

from easyebl.core import FieldCalibration
from easyebl.microscope import (MAGNIFICATIONS, POLL, SEM_SET, MicroscopeLink, NotAvailable, SEMControl,
                                StageControl, StubMicroscope)
from easyebl.microscope.coupling import FieldScale
from easyebl.microscope.s3000 import MotionError


@pytest.fixture
def mic():
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    sem = SEMControl(link, polling=False)
    stage = StageControl(link, polling=False, hand_control=False)
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


def test_poller_reads_each_level_half_as_often_and_none_beyond_the_floor():
    """Order: the largest age / 2 ** (level - 1) first, whatever the read time; overdue before all."""
    from easyebl.microscope.poll import PollTask

    l1, l3 = PollTask("l1", None, level=1, cost=5.0), PollTask("l3", None, level=3, cost=0.01)
    l1.last = l3.last = 100.0
    assert l1.score(101.0) == pytest.approx(1.0) and l3.score(101.0) == pytest.approx(0.25)
    assert l1.score(101.0) > l3.score(101.0)  # same age: the important one, although it is slow
    assert l3.score(104.1) > l1.score(101.0)  # level 3 goes when 4x older
    assert PollTask("new", None).score(0.0) == float("inf")
    late = PollTask("late", None, level=9)
    late.last = 84.0
    assert late.score(100.0, 2.0, 15.0) > l1.score(100.0 + 1e3, 2.0, 15.0 * 1e9)  # overdue beats any


def test_poller_schedule_on_the_simulator(monkeypatch):
    """Every SEM item and stage controller is a task; costs are measured; nothing waits past the floor."""
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    try:
        sem = SEMControl(link, refresh_on_start=False)
        stage = StageControl(link)
        poller = link.poller
        names = {t.name for t in poller.tasks()}
        assert {f"SEM {i}" for i in ("MAG", "HV", "WD", "STG X")} <= names
        assert {"stage X/Y", "stage Z/T", "stage R"} <= names
        lv = {x.name: x.level for x in poller.tasks()}
        assert lv["SEM MAG"] == lv["SEM HV"] == 1 and lv["SEM FOCUS F"] == lv["SEM STG X"] == lv["SEM WD"] == 2
        assert lv["stage X/Y"] == lv["stage Z/T"] == 3 and lv["stage R"] == lv["SEM CONT"] == 4
        t0 = time.monotonic()
        while (any(t.reads < 2 for t in poller.tasks()) and time.monotonic() - t0 < 60):
            time.sleep(0.1)
        assert all(t.reads >= 2 for t in poller.tasks())
        assert all(t.cost > 0 and t.period for t in poller.tasks())  # measured, not the guesses
        reads = [(p, w) for p, w in link.executed if w.startswith(("SEM read", "stage read"))]
        assert reads and all(p == POLL for p, _ in reads)
        assert sem.magnification() is not None and stage.positions["X"] == pytest.approx(0x17077D / 25.6)
        assert stage.poll_period_s() is not None
        # a setting waits for one read at most
        last = link.executed[-1]  # the log keeps the last 200: find the place by identity
        sem.set_magnification(2000).result(10)
        log = list(link.executed)
        k = next(i for i in range(len(log) - 1, -1, -1) if log[i] is last)
        assert [p for p, _ in log[k + 1:]].index(SEM_SET) <= 1
        # switched off: no more reads of that module
        sem.set_polling(False)
        stage.set_polling(False)
        time.sleep(1.5)  # the read in progress finishes
        n = poller.reads
        time.sleep(1.0)
        assert poller.reads == n
        stage.set_polling(True)
        t0 = time.monotonic()
        while poller.reads < n + 2 and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        assert poller.reads >= n + 2 and all(w.startswith("stage read") for _, w in link.executed[-2:])
    finally:
        link.close()


def test_poller_never_lets_a_low_level_wait_past_the_floor(monkeypatch):
    """A task of a very low level is still read every MAX_PERIOD_S."""
    from easyebl.microscope.poll import Poller

    monkeypatch.setattr(Poller, "MAX_PERIOD_S", 1.0)
    link = MicroscopeLink.simulated(command_delay_s=0.0)
    try:
        poller = link.poller
        poller.add("top", lambda s: time.sleep(0.01), level=1)
        slow = poller.add("low", lambda s: time.sleep(0.05), level=30)  # by level alone: ~never
        time.sleep(4.0)
        assert slow.reads >= 3 and slow.period == pytest.approx(1.0, abs=0.3)
        assert poller.task("top").reads > 20 * slow.reads
    finally:
        link.close()


def test_old_settings_keep_the_sem_read_switch(tmp_path):
    import json
    from easyebl.microscope.settings import MicroscopeSettings
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"port": "COM9", "stage_poll_s": 30.0, "sem_status_poll_s": 10.0}))
    s = MicroscopeSettings.load(str(f))
    assert s.port == "COM9" and s.sem_polling is True and s.stage_polling is True
    f.write_text(json.dumps({"sem_status_poll_s": 0}))
    assert MicroscopeSettings.load(str(f)).sem_polling is False


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


def test_hand_control_release_and_engage(mic):
    link, sem, stage = mic
    sim = link.session.simulator
    m = stage.move({"X": 20000.0}, relative=True)  # still running when released
    t0 = time.monotonic()
    while not stage.moving and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    stage.release().result(15)
    with pytest.raises((MotionError, concurrent.futures.CancelledError)):
        m.result(15)
    assert stage.hand_control and not stage.moving and sim.sem["STAGE"] == "OFF"
    assert not any("S2" in line for line in sim.received)  # the SEM PC decides about motor power
    # positions are still read; #STAGE OFF follows each read (the trackball keeps working)
    sim.axes[(0, 0)].position += 2560  # moved with the trackball: +100 µm
    x = stage.positions["X"]
    stage.refresh(["X"]).result(15)
    assert stage.positions["X"] == pytest.approx(x + 100.0, abs=0.05) and sim.sem["STAGE"] == "OFF"
    with pytest.raises(RuntimeError, match="hand control"):
        stage.move({"X": 10.0}, relative=True)
    stage.engage().result(10)
    assert not stage.hand_control and sim.sem["STAGE"] == "ON"
    stage.move({"X": 10.0}, relative=True).result(30)


def test_stage_starts_under_hand_control():
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    try:
        stage = StageControl(link, polling=False)
        assert stage.hand_control
        stage.refresh().result(10)
        assert link.session.simulator.sem["STAGE"] == "OFF" and stage.positions["X"] is not None
        with pytest.raises(RuntimeError):
            stage.move({"X": 1.0}, relative=True)
    finally:
        link.close()


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


def test_global_field_calibration_follows_the_magnification(mic, tmp_path):
    from easyebl.microscope.settings import MicroscopeSettings

    link, sem, stage = mic
    eng, eds = _Target(100.0), _Target(80.0)
    f = FieldScale(sem, [eng, eds])  # in memory: K keeps the present field at the current ×1k
    assert f.k_um == pytest.approx(100.0 * 1000) and f.from_microscope
    assert eds.calibration.field_size_um == pytest.approx(100.0)  # every target on K / magnification
    sem.set_magnification(2000).result(10)
    assert eng.calibration.field_size_um == pytest.approx(50.0)
    assert eds.calibration.field_size_um == pytest.approx(50.0)
    assert f.um_per_volt() == pytest.approx(50.0 / 10.0)  # ±5 V full scale
    # calibrated with a line: measured 10 µm, really 12 µm -> K × 1.2, at every magnification
    assert f.calibrate(10.0, 12.0) == pytest.approx(1.2)
    assert f.k_um == pytest.approx(120000.0) and eng.calibration.field_size_um == pytest.approx(60.0)
    sem.set_magnification(500).result(10)
    assert eng.calibration.field_size_um == pytest.approx(240.0)
    with pytest.raises(ValueError):
        f.calibrate(0.0, 1.0)
    f.close()
    # with settings: K and the manual magnification persist; no microscope -> the manual magnification
    st = MicroscopeSettings(path=str(tmp_path / "m.json"))
    st.field_at_1x_um = 50000.0
    g = FieldScale(None, [eng], st)
    assert not g.from_microscope and g.magnification == 1000 and eng.calibration.field_size_um == pytest.approx(50.0)
    g.set_manual_magnification(250)
    assert eng.calibration.field_size_um == pytest.approx(200.0)
    g.set_k(80000.0)
    again = MicroscopeSettings.load(st.path)
    assert again.field_at_1x_um == 80000.0 and again.manual_magnification == 250
    g.close()


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
    sent = [c for c in sim.received[n0:] if not c.startswith("R ")]  # (the stigmator is read back)
    # each stigmator value is written twice: a single one-step write drifts on the S-3000
    assert sent == ["BEAM C 60", "BEAM F 10", "STG X 2000", "STG X 2000", "STG Y 2050", "STG Y 2050",
                    "CONT 1800", "BRT 1900"]
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
