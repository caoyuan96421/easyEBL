"""Hitachi S-3000 over RS-232: communication and settings (skips when the SEM does not answer).

Never touched: the accelerating voltage, HV on/off, the automatic routines (AFC / ASF /
ABC / AGA) and the stage's Z / R / T.  Each test puts back what it changed, and the
``microscope`` fixture checks at the end that every setting and the X / Y position are
as they were.
"""
import time

import pytest

from easyebl.microscope import MAGNIFICATIONS, POLL, SEM_SET
from easyebl.microscope.sem import ALL_ITEMS

from .conftest import NO_STAGE

pytestmark = pytest.mark.hardware


def _read(stage, axes=("X", "Y", "Z", "R", "T")) -> dict:
    """Positions read now from the stage controllers (µm / degrees)."""
    stage.refresh(list(axes)).result(30)
    return dict(stage.positions)


def _nudge(value: int, lo: int, hi: int, step: int) -> int:
    """A different value within the range (up if there is room, else down)."""
    return value + step if value + step <= hi else value - step


def test_every_readback_parses(microscope):
    link, sem, stage = microscope
    got = sem.refresh().result(30)
    assert link.state == "connected"
    for key in ("mag", "focus_coarse", "focus_fine", "stig_x", "stig_y", "beam_coarse", "beam_fine",
                "contrast", "brightness"):
        assert isinstance(got.get(key), int), (key, got.get(key))
    assert 3.0 <= sem.working_distance_mm() <= 60.0  # computed from the focus (WD is not read)
    assert got["mag"] in MAGNIFICATIONS
    assert isinstance(got.get("hv_on"), bool) and isinstance(got.get("running"), bool)
    assert len(ALL_ITEMS) == len({i for i in ALL_ITEMS})


@pytest.mark.parametrize("key,setter,lo,hi,step", [
    ("contrast", lambda s, v: s.set_contrast(v), 0, 4095, 16),
    ("brightness", lambda s, v: s.set_brightness(v), 0, 4095, 16),
    ("stig_x", lambda s, v: s.set_stigmator("X", v), 0, 4095, 16),
    ("stig_y", lambda s, v: s.set_stigmator("Y", v), 0, 4095, 16),
    ("focus_fine", lambda s, v: s.set_focus("F", v), 0, 4095, 8),
    ("focus_coarse", lambda s, v: s.set_focus("C", v), 0, 16383, 1),
    ("beam_coarse", lambda s, v: s.set_beam("C", v), 0, 99, 1),
    ("beam_fine", lambda s, v: s.set_beam("F", v), 0, 99, 1),
])
def test_setting_reads_back_and_is_restored(microscope, key, setter, lo, hi, step):
    link, sem, stage = microscope
    item = {"contrast": "CONT", "brightness": "BRT", "stig_x": "STG X", "stig_y": "STG Y",
            "focus_fine": "FOCUS F", "focus_coarse": "FOCUS C", "beam_coarse": "BEAM C",
            "beam_fine": "BEAM F"}[key]
    v0 = sem.refresh((item,)).result(10)[key]
    new = _nudge(v0, lo, hi, step)
    try:
        setter(sem, new).result(10)
        assert sem.refresh((item,)).result(10)[key] == new  # what the SEM reports, not the cache
    finally:
        setter(sem, v0).result(10)
    assert sem.refresh((item,)).result(10)[key] == v0


def test_magnification_step_and_restore(microscope):
    link, sem, stage = microscope
    m0 = sem.refresh(("MAG",)).result(10)["mag"]
    i = MAGNIFICATIONS.index(m0)
    new = MAGNIFICATIONS[i + 1] if i + 1 < len(MAGNIFICATIONS) else MAGNIFICATIONS[i - 1]
    try:
        assert sem.set_magnification(new).result(10) == new
        assert sem.refresh(("MAG",)).result(10)["mag"] == new
    finally:
        sem.set_magnification(m0).result(10)
    assert sem.refresh(("MAG",)).result(10)["mag"] == m0


def test_repeated_magnification_steps_coalesce(microscope):
    """Wheel steps sent while the port is busy go out as one MAG with the final value."""
    link, sem, stage = microscope
    m0 = sem.refresh(("MAG",)).result(10)["mag"]
    i = MAGNIFICATIONS.index(m0)
    steps = 3 if i + 3 < len(MAGNIFICATIONS) else -3
    n0 = len(link.executed)
    busy = link.submit(lambda s: s.sem.read("WD"), SEM_SET, "busy read")
    futs = [sem.step_magnification(1 if steps > 0 else -1) for _ in range(abs(steps))]
    try:
        futs[-1].result(15)
        busy.result(10)
        mags = [w for _, w in link.executed[n0:] if w.startswith("MAG")]
        assert len(mags) <= 2  # one transaction (two if the first left before the others arrived)
        assert sem.refresh(("MAG",)).result(10)["mag"] == MAGNIFICATIONS[i + steps]
    finally:
        sem.set_magnification(m0).result(10)


def test_working_distance_and_restore(microscope):
    link, sem, stage = microscope
    sem.refresh(("FOCUS C", "FOCUS F")).result(10)
    c0, f0 = sem.state["focus_coarse"], sem.state["focus_fine"]
    wd0 = sem.working_distance_mm()
    new = wd0 + 0.5 if wd0 < 59 else wd0 - 0.5
    try:
        sem.set_working_distance(new).result(10)  # FOCUS C / F from the table
        sem.refresh(("FOCUS C", "FOCUS F")).result(10)
        assert sem.working_distance_mm() == pytest.approx(new, abs=0.01)
    finally:
        sem.set_focus("C", c0).result(10)
        sem.set_focus("F", f0).result(10)
    sem.refresh(("FOCUS C", "FOCUS F")).result(10)
    assert (sem.state["focus_coarse"], sem.state["focus_fine"]) == (c0, f0)


def test_freeze_and_run(microscope):
    link, sem, stage = microscope
    run0 = sem.refresh(("RUN",)).result(10)["running"]
    try:
        sem.set_running(not run0).result(10)
        assert sem.refresh(("RUN",)).result(10)["running"] is (not run0)
    finally:
        sem.set_running(run0).result(10)
    assert sem.refresh(("RUN",)).result(10)["running"] is run0


def test_detector_selection_is_accepted(microscope):
    """The current detector is selected again (other detectors may not be installed)."""
    link, sem, stage = microscope
    det = sem.refresh(("INPUTS AF",)).result(10)["detector"]
    sem.set_detector(det).result(10)
    assert sem.refresh(("INPUTS AF",)).result(10)["detector"] == det


def test_beam_preset_roundtrip(microscope):
    link, sem, stage = microscope
    cur = sem.read_preset().result(15)
    changed = dict(cur, contrast=_nudge(cur["contrast"], 0, 4095, 16), beam_fine=_nudge(cur["beam_fine"], 0, 99, 1))
    try:
        sem.apply_preset(changed).result(20)
        assert sem.read_preset().result(15) == changed
    finally:
        sem.apply_preset(cur).result(20)
    assert sem.read_preset().result(15) == cur


def test_settings_go_before_queued_polls(microscope):
    link, sem, stage = microscope
    c0 = sem.refresh(("CONT",)).result(10)["contrast"]
    n0 = len(link.executed)
    polls = [link.submit(lambda s: s.sem.read("MAG"), POLL, f"hw poll {i}") for i in range(3)]
    try:
        sem.set_contrast(_nudge(c0, 0, 4095, 16)).result(15)
        for p in polls:
            p.result(15)
        order = [w for _, w in link.executed[n0:]]
        i_set = next(i for i, w in enumerate(order) if w.startswith("CONT"))
        assert i_set <= 1  # at most the poll already on the wire went first
    finally:
        sem.set_contrast(c0).result(10)


def test_stage_positions_read(microscope):
    link, sem, stage = microscope
    t = time.time()
    pos = _read(stage)
    assert link.state == "connected"
    for a in ("X", "Y", "Z", "R", "T"):
        assert pos.get(a) is not None and stage.t_read.get(a, 0) >= t - 1, a
    assert pos["X"] >= 0 and pos["Y"] >= 0


@pytest.mark.skipif(NO_STAGE, reason="stage moves switched off (EASYEBL_HW_NO_STAGE)")
@pytest.mark.parametrize("axis", ["X", "Y"])
def test_small_stage_move_and_back(microscope, axis):
    link, sem, stage = microscope
    p0 = _read(stage, [axis])[axis]
    try:
        pos = stage.move({axis: 20.0}, relative=True).result(60)
        assert pos[axis] == pytest.approx(p0 + 20.0, abs=1.0)
    finally:
        stage.acknowledge()
        stage.move({axis: p0}).result(60)
    assert _read(stage, [axis])[axis] == pytest.approx(p0, abs=1.0)


@pytest.mark.skipif(NO_STAGE, reason="stage moves switched off (EASYEBL_HW_NO_STAGE)")
def test_stop_halts_a_move(microscope):
    from easyebl.microscope.s3000 import MotionError

    link, sem, stage = microscope
    # the SEM relays each stage line in ~0.8 s, so S0 reaches the controller ~1.6 s after C0
    # (C0's acknowledgement first): a 500 µm move is over by then; 2 mm is still under way
    dist = 2000.0
    x0 = _read(stage, ["X"])["X"]
    m = stage.move({"X": dist}, relative=True)
    try:
        t0 = time.monotonic()
        while not stage.moving and time.monotonic() - t0 < 10:
            time.sleep(0.02)
        assert stage.moving, "the move did not start"
        stage.stop().result(10)
        assert link.state == "connected"  # every S0 acknowledged
        with pytest.raises(MotionError, match="halted"):
            m.result(60)
        x = _read(stage, ["X"])["X"]
        assert x0 - 1.0 <= x < x0 + dist - 1.0  # stopped part-way
    finally:
        stage.acknowledge()
        stage.move({"X": x0}).result(60)
    assert _read(stage, ["X"])["X"] == pytest.approx(x0, abs=1.0)


@pytest.mark.skipif(NO_STAGE, reason="stage moves switched off (EASYEBL_HW_NO_STAGE)")
def test_a_move_is_exclusive_and_hands_the_stage_back(microscope):
    """#STAGE ON … S2 … #STAGE OFF around the move; a SEM read sent meanwhile waits for it;
    a second move is refused."""
    from easyebl.microscope import StageBusy

    link, sem, stage = microscope
    x0 = _read(stage, ["X"])["X"]
    n0 = len(link.log_entries())
    m = stage.move({"X": 300.0}, relative=True)
    try:
        t0 = time.monotonic()
        while not stage.busy and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        with pytest.raises(StageBusy):
            stage.move({"Y": 10.0}, relative=True)
        r = sem.refresh(("MAG",))
        m.result(90)
        t_move = time.monotonic()
        r.result(30)
        assert not stage.busy and link.exclusive is None and link.state == "connected"
        tx = [e.format() for e in link.log_entries()[n0:]]
        text = "\n".join(tx)
        i_on, i_s2, i_off = text.find("#STAGE ON"), text.find("S2"), text.rfind("#STAGE OFF")
        assert -1 < i_on < i_s2 < i_off, text[-2000:]
        assert text.rfind("R MAG") > i_off  # the SEM read went out after the move
        assert time.monotonic() - t_move < 30
    finally:
        stage.acknowledge()
        stage.move({"X": x0}).result(90)
    assert _read(stage, ["X"])["X"] == pytest.approx(x0, abs=1.0)
