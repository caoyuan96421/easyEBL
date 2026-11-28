"""Microscope HAL (Hitachi S-3000 over RS-232) on the simulator: shared link, SEM and stage modules."""
import concurrent.futures
import threading
import time

import pytest

from easyebl.core import FieldCalibration
from easyebl.microscope import (MAGNIFICATIONS, POLL, SEM_SET, MicroscopeLink, NotAvailable, SEMControl,
                                StageBusy, StageControl, StubMicroscope)
from easyebl.microscope.coupling import FieldScale
from easyebl.microscope.s3000 import MotionError


@pytest.fixture
def mic():
    # no late X / Y correction here (the moves end on target): test_move_waits_until_* models it
    link = MicroscopeLink.simulated(command_delay_s=0.01, settle_error_um=0.0)
    sem = SEMControl(link, polling=False)
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
    assert sem.working_distance_mm() == pytest.approx(15, abs=0.01) and sem.connected
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


def test_hv_on_from_off_starts_at_30_kv(mic):
    link, sem, _stage = mic
    sem.hv_off().result(10)
    sem.hv_on().result(10)  # was off: 30 kV (the setpoint was 20 kV)
    assert sem.accelerating_voltage_kv() == 30.0 and sem.state["hv_setpoint_kv"] == 30.0
    sem.set_hv(15).result(10)
    sem.hv_on().result(10)  # already on: the voltage stays
    assert sem.accelerating_voltage_kv() == 15.0
    sem.hv_off().result(10)
    sem.hv_on(kv=None).result(10)  # keep the SEM's setpoint
    assert sem.accelerating_voltage_kv() == 15.0


def test_wd_from_focus_table():
    """WD is the objective focus read through the S-3000's table (measured 2026-11-23)."""
    from easyebl.microscope.s3000.focus_wd import (FINE_WEIGHT, WD_TABLE, effective_focus, focus_for_wd,
                                                   wd_from_focus)

    for wd, (c, f) in WD_TABLE.items():  # WD n -> its own table focus
        assert wd_from_focus(c, f) == pytest.approx(wd, abs=1e-6)
    # the SEM's readback steps (bisected on the instrument, fine 2048) fall at n + 0.5
    for c_step, n in ((15654, 4), (12100, 10), (10543, 15), (9509, 20), (8194, 30), (7375, 40), (6424, 59)):
        assert wd_from_focus(c_step + 0.5, 2048) == pytest.approx(n + 0.5, abs=0.01)
    # fine focus: 1 count = FINE_WEIGHT coarse counts; the 15 -> 16 step moves with it
    assert effective_focus(10000, 2048 + 100) == pytest.approx(10000 + 100 * FINE_WEIGHT)
    assert wd_from_focus(10711, 0) == pytest.approx(15.5, abs=0.01)
    assert wd_from_focus(10377, 4095) == pytest.approx(15.5, abs=0.01)
    # monotone, held to the SEM's 3-60 mm, and invertible
    w = [wd_from_focus(c, 2048) for c in range(5174, 16384, 97)]
    assert all(b <= a for a, b in zip(w, w[1:])) and min(w) >= 3 and max(w) <= 60
    for wd in (4.0, 7.3, 11.137, 15.0, 33.3, 59.9):
        c, f = focus_for_wd(wd)
        assert 0 <= f <= 4095 and wd_from_focus(c, f) == pytest.approx(wd, abs=0.002)


def test_sem_computes_wd_from_the_focus(mic):
    link, sem, stage = mic
    sim = link.session.simulator
    from easyebl.microscope.s3000.focus_wd import focus_for_wd

    n0 = len(sim.received)
    sem.set_working_distance(22.0).result(10)  # through the focus: FOCUS C and F of the table
    assert sim.sem["FOCUS C"] == "9262" and sem.state["focus_coarse"] == 9262
    assert sem.working_distance_mm() == pytest.approx(22, abs=0.01)
    sem.set_working_distance(15.437).result(10)  # any value, not whole mm
    c, f = focus_for_wd(15.437)
    assert (sem.state["focus_coarse"], sem.state["focus_fine"]) == (c, f)
    assert sem.working_distance_mm() == pytest.approx(15.437, abs=0.002)
    sent = sim.received[n0:]
    assert not any(ln.startswith("WD") or ln == "R WD" for ln in sent)  # never WD n, never R WD
    assert f"FOCUS C {c}" in sent and f"FOCUS F {f}" in sent
    with pytest.raises(ValueError):
        sem.set_working_distance(2.0)
    sem.set_focus("C", 9262).result(10)
    sem.set_focus("F", 2048 + 1000).result(10)  # fine focus: a fraction of a mm
    assert 21.4 < sem.working_distance_mm() < 21.6
    sem.set_focus("C", 100).result(10)  # the SEM holds the coarse focus at its minimum
    assert sem.state["focus_coarse"] == 5174 and sem.working_distance_mm() == 60.0
    sem.refresh().result(10)
    assert "wd_readback_mm" not in sem.state and "R WD" not in sim.received  # WD is never read
    assert "SEM WD" not in {t.name for t in link.poller.tasks()}


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
        assert {f"SEM {i}" for i in ("MAG", "HV", "FOCUS C", "STG X")} <= names and "SEM WD" not in names
        assert {"stage X/Y", "stage Z/T", "stage R"} <= names
        lv = {x.name: x.level for x in poller.tasks()}
        assert lv["SEM MAG"] == lv["SEM HV"] == 1 and lv["SEM FOCUS F"] == lv["SEM STG X"] == lv["SEM BEAM C"] == 2
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
    assert link.session.simulator.sem["STAGE"] == "OFF"  # the SEM PC has the stage again after each move
    with pytest.raises(ValueError):
        stage.move({"Q": 1})
    with pytest.raises(ValueError):
        stage.move({"X": 0.0}, relative=True)
    with pytest.raises(ValueError, match="below the stage zero"):
        stage.move({"X": -1e9}, relative=True).result(30)


def test_xy_moves_end_with_a_one_way_approach(mic):
    """X / Y end travelling in + by the backlash distance: a move in − overshoots and comes back."""
    link, sem, stage = mic
    assert stage.backlash_um == 50.0
    sent = []
    orig = link.session.stage.move

    def spy(pulses, **kw):  # the legs, in µm, as sent to the controllers
        from easyebl.microscope.s3000 import from_pulses
        sent.append({a: from_pulses(a, p) for a, p in pulses.items()})
        return orig(pulses, **kw)

    link.session.stage.move = spy
    x0, y0 = stage.positions["X"], stage.positions["Y"]
    pos = stage.move({"X": x0 + 200.0}).result(30)  # + by more than 50: one move
    assert sent == [{"X": x0 + 200.0}] and pos["X"] == pytest.approx(x0 + 200.0, abs=0.05)
    sent.clear()
    pos = stage.move({"X": -150.0, "Y": 10.0}, relative=True).result(60)  # − in X, + by only 10 in Y
    assert sent == [{"X": x0 + 50.0 - 50.0, "Y": y0 + 10.0 - 50.0}, {"X": x0 + 50.0, "Y": y0 + 10.0}]
    assert pos["X"] == pytest.approx(x0 + 50.0, abs=0.05) and pos["Y"] == pytest.approx(y0 + 10.0, abs=0.05)
    sent.clear()
    stage.move({"Z": 10.0}, relative=True).result(30)  # Z: direct
    assert len(sent) == 1
    sent.clear()
    x1 = stage.positions["X"]
    stage.move({"X": -100.0}, relative=True, approach=False).result(30)
    assert sent == [{"X": pytest.approx(x1 - 100.0, abs=0.05)}]


def _settling_mic(error_um=3.0, settle_s=1.5, **stage_kw):
    link = MicroscopeLink.simulated(command_delay_s=0.01, settle_error_um=error_um, settle_time_s=settle_s)
    stage = StageControl(link, polling=False, **stage_kw)
    stage.refresh().result(10)
    return link, stage


def test_move_waits_until_xy_read_within_the_tolerance_of_the_setpoint():
    """The controllers report STOPPED 3 µm short; the move waits for the encoder correction."""
    link, stage = _settling_mic()
    try:
        assert stage.settle_tol["X"] == 0.5 and stage.settle_tol["R"] == 0.01
        x0 = stage.positions["X"]
        offs = []
        fut = stage.move({"X": x0 + 200.0})
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < 30:
            if stage.settling:
                offs.append(stage.settling["X"])
            time.sleep(0.02)
        pos = fut.result(1)
        assert time.monotonic() - t0 > 0.7  # STOPPED came almost at once; the correction took ~1 s
        assert offs and min(offs) < -0.5  # it was seen short of the target meanwhile
        assert abs(pos["X"] - (x0 + 200.0)) <= 0.5 and stage.settling is None
        # the backlash pre-leg does not wait: a move in − has the same single settling phase at the end
        pos = stage.move({"X": x0 + 100.0}).result(30)
        assert abs(pos["X"] - (x0 + 100.0)) <= 0.5
    finally:
        link.close()


def test_move_fails_when_xy_do_not_settle_in_time():
    link, stage = _settling_mic(error_um=5.0, settle_s=20.0, settle_timeout_s=0.8)
    try:
        with pytest.raises(MotionError, match="did not settle"):
            stage.move({"Y": stage.positions["Y"] + 100.0}).result(30)
        assert not stage.busy and stage.settling is None
    finally:
        link.close()


def test_stop_ends_the_settling_wait():
    link, stage = _settling_mic(error_um=5.0, settle_s=20.0)
    try:
        fut = stage.move({"X": stage.positions["X"] + 100.0})
        t0 = time.monotonic()
        while not stage.settling and time.monotonic() - t0 < 10:
            time.sleep(0.02)
        assert stage.settling
        stage.stop()
        with pytest.raises(concurrent.futures.CancelledError):
            fut.result(10)
        assert time.monotonic() - t0 < 5 and not stage.busy
    finally:
        link.close()


def test_zrt_off_the_setpoint_fail_at_once():
    """Z / R / T do not correct: a lasting error (lost steps) fails without waiting out the timeout."""
    link, stage = _settling_mic(error_um=0.0, settle_timeout_s=30.0)
    try:
        sim = link.session.simulator
        pos = stage.move({"T": 1.0}, relative=True).result(30)  # on target: fine
        assert pos["T"] == pytest.approx(1.0 + 0x62 / 3528, abs=0.01)
        sim.options.final_error_pulses[(2, 1)] = 100  # T: 100 / 3528 ° ≈ 0.028° off
        t0 = time.monotonic()
        with pytest.raises(MotionError, match="did not reach the setpoint: T"):
            stage.move({"T": 1.0}, relative=True).result(30)
        assert time.monotonic() - t0 < 5
        sim.options.final_error_pulses[(2, 1)] = 10  # 0.003°: within 0.01°
        stage.move({"T": 1.0}, relative=True).result(30)
    finally:
        link.close()


def _stage_lines(sim, n0=0):
    """The #STAGE / S2 / move lines the simulator received since index *n0*."""
    return [ln for ln in sim.received[n0:] if ln.startswith("#STAGE") or "S2" in ln or "C0" in ln or "M0" in ln]


def test_every_move_is_wrapped_in_stage_on_and_off(mic):
    """#STAGE ON right before a move, S2 (motors free) and #STAGE OFF right after – also for reads."""
    link, sem, stage = mic
    sim = link.session.simulator
    assert sim.sem["STAGE"] == "OFF"  # reads left the relay off
    n0 = len(sim.received)
    stage.move({"X": 300.0}, relative=True).result(30)
    lines = _stage_lines(sim, n0)
    assert lines[0] == "#STAGE ON" and lines[-1] == "#STAGE OFF" and lines.count("#STAGE ON") == 1
    assert any("S2" in ln for ln in lines) and "S2" in lines[-2]
    assert sim.sem["STAGE"] == "OFF" and not stage.busy
    # a read: ON, read, OFF
    n0 = len(sim.received)
    stage.refresh(["X"]).result(10)
    assert [ln for ln in sim.received[n0:] if ln.startswith("#STAGE")] == ["#STAGE ON", "#STAGE OFF"]
    # a failed move ends with #STAGE OFF too
    n0 = len(sim.received)
    with pytest.raises(ValueError, match="below the stage zero"):
        stage.move({"X": -1e9}, relative=True).result(30)
    assert _stage_lines(sim, n0)[-1] == "#STAGE OFF" and not stage.busy


def test_stage_off_already_off_is_not_an_error():
    """#STAGE OFF while the relay is already off: the S-3000 answers E6, which is ignored."""
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    try:
        stage = StageControl(link, polling=False)
        link.call(lambda s: stage._stage_off(s))
        assert link.session.simulator.sem["STAGE"] == "OFF"
    finally:
        link.close()


def test_a_move_holds_the_port_and_refuses_a_second_move(mic):
    """During a move: a second move is refused, SEM settings wait in line and go out after it,
    the background reads pause; STOP still gets through."""
    link, sem, stage = mic
    sim = link.session.simulator
    stage.set_polling(True)
    sem.set_polling(True)
    time.sleep(0.5)
    m = stage.move({"X": 4000.0}, relative=True)
    t0 = time.monotonic()
    while not stage.moving and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert stage.busy and link.exclusive
    with pytest.raises(StageBusy):
        stage.move({"Y": 10.0}, relative=True)
    n0 = len(sim.received)
    f = sem.set_magnification(MAGNIFICATIONS[5])
    time.sleep(0.6)
    assert not f.done() and stage.moving  # queued behind the move
    during = sim.received[n0:]
    assert not any(ln.startswith("R ") or ln.startswith("MAG") for ln in during)  # no SEM traffic, no polls
    pos = m.result(60)
    f.result(10)
    after = sim.received[n0:]
    i_off = after.index("#STAGE OFF")
    assert any(ln.startswith("MAG") for ln in after[i_off:])  # the setting went out after the move
    assert pos["X"] is not None and not stage.busy and link.exclusive is None
    # STOP during a move: halted, then S2 and #STAGE OFF
    m = stage.move({"X": 20000.0}, relative=True)
    t0 = time.monotonic()
    while not stage.moving and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    n0 = len(sim.received)
    assert stage.stop().result(10) is True
    with pytest.raises((MotionError, concurrent.futures.CancelledError)):
        m.result(30)
    lines = sim.received[n0:]
    assert any("S0" in ln for ln in lines) and _stage_lines(sim, n0)[-1] == "#STAGE OFF"
    assert sim.sem["STAGE"] == "OFF" and not stage.busy


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
    queued = sem.set_contrast(321)  # waits behind the move (the move job holds the port)
    time.sleep(0.2)
    assert link.busy_with.startswith("stage move") and not queued.done()
    t = time.monotonic()
    stage.stop().result(5)
    assert time.monotonic() - t < 1.5  # S0 went out while the move job held the port
    with pytest.raises(MotionError):
        m.result(15)
    queued.result(10)  # and the queued setting went out after it


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
    # more than one stigmator step from the value read: written once (a single one-step write
    # drifts on the S-3000, those are written twice)
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


def test_focus_is_confirmed_by_readback_and_stigmator_written_once_for_large_steps(mic):
    """One write + one readback per setting; the stigmator is written twice only for a one-step
    change (the S-3000 misapplies a single one-step write, measured 2026-11-25)."""
    link, sem, _stage = mic
    sim = link.session.simulator
    n0 = len(sim.received)
    assert sem.set_focus("F", 1500).result(10) == 1500
    assert sim.received[n0:] == ["FOCUS F 1500", "R FOCUS F"]
    n0 = len(sim.received)
    assert sem.set_focus("C", 10000).result(10) == 10000
    assert sim.received[n0:] == ["FOCUS C 10000", "R FOCUS C"]
    n0 = len(sim.received)
    assert sem.set_stigmator("X", 2048 + 256).result(10) == 2048 + 256
    assert sim.received[n0:] == ["STG X 2304", "R STG X"]
    n0 = len(sim.received)
    sem.set_stigmator("X", 2304 + 16).result(10)
    assert sim.received[n0:] == ["STG X 2320", "STG X 2320", "R STG X"]


def test_a_focus_readback_that_disagrees_is_written_again_then_reported(mic, monkeypatch):
    link, sem, _stage = mic
    real = link.session.sem.read
    monkeypatch.setattr(link.session.sem, "read", lambda item: "1234" if item == "FOCUS F" else real(item))
    with pytest.raises(RuntimeError, match="reads back 1234"):
        sem.set_focus("F", 1600).result(10)
    assert sem.state["focus_fine"] == 1234  # what the SEM says


def test_holding_the_poller_stops_background_reads():
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    try:
        sem = SEMControl(link, polling=True)
        time.sleep(0.3)
        with sem.hold_polling():
            time.sleep(0.1)  # a read in progress ends
            n = link.poller.reads
            time.sleep(0.4)
            assert link.poller.reads == n and link.poller.is_held
        time.sleep(0.4)
        assert link.poller.reads > n and not link.poller.is_held
    finally:
        link.close()
