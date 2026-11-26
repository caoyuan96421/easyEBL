"""Lithography window (offscreen Qt): tabs, chips, step editor, schedule file, simulated runs, beam and field
calibration windows, SEM mark picking, the program's backend."""
import math
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("gdstk")
pytestmark = pytest.mark.gui

from PySide6 import QtWidgets  # noqa: E402

from easyebl.app.config import SystemConfig  # noqa: E402
from easyebl.litho.beamcal import BeamCalConfig, BeamCalibration  # noqa: E402
from easyebl.litho.coords import S3000_AXES  # noqa: E402
from easyebl.litho.demo import CUP_POSITION, GRID_POSITION  # noqa: E402
from easyebl.litho.fieldcal import Condition, FieldCalConfig  # noqa: E402
from easyebl.litho.height import ChipHeight  # noqa: E402
from easyebl.litho.runner import MarkRequest, SimulatedHardware  # noqa: E402
from easyebl.litho.schedule import Chip, ColumnSetting, Exposure, FieldCorrection, Schedule  # noqa: E402
from easyebl.litho.settings import LithoSettings, StagePosition  # noqa: E402

ORIGIN = (5000.0, 5000.0)
H = 4000.0  # the fixture's calibration positions: in focus at Z = 15000 − 4000 at WD 15 mm


def _rot(deg, s=1.0):
    t = math.radians(deg)
    return s * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


@pytest.fixture
def demo(tmp_path):
    from easyebl.litho.demo import make_demo_files

    return make_demo_files(str(tmp_path / "demo"))


@pytest.fixture
def win(qtbot, tmp_path):
    from easyebl.litho.module import LithoWindow

    hw = SimulatedHardware(_rot(0.3, 1.001), stage_noise_um=0.3, pivot=ORIGIN, start=ORIGIN,
                           beam_linear=[[1.002, 0.0], [0.0, 0.999]], beam_k=1e-7, grid_at=((2000.0, 2000.0), 60.0),
                           start_z=11100.0)
    settings = LithoSettings.load(str(tmp_path / "litho.json"))
    settings.add_position(GRID_POSITION, 2000.0, 2000.0, H)  # the demo schedules' Beam / Field calibration
    settings.add_position(CUP_POSITION, 4700.0, 5000.0, H)
    w = LithoWindow(hw, settings)
    w.ask_on_close = False
    qtbot.addWidget(w)
    w.show()
    return w


def _waits(qtbot, get, timeout=30000):
    qtbot.waitUntil(lambda: get() is not None and get().awaiting, timeout=timeout)
    return get()


def _calibration_waits(win, qtbot, timeout=30000):
    return _waits(qtbot, lambda: win.fieldcal_window, timeout)


def _fast(schedule):
    """The fixed steps of a demo schedule, quick on the simulator."""
    schedule[1].config.pixels = 800
    schedule[2].config.settle_s = 0.0


def test_window_open_edit_save_run(win, tmp_path, demo, qtbot):
    tab = win.tab_schedule
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Stage Positions", "Chips",
                                                                      "Exposure Schedule"]
    assert win.axes == S3000_AXES and win.view.flipped == (True, False)
    assert isinstance(win.schedule[0], ColumnSetting) and tab.list.count() == 3 and not tab.b_del.isEnabled()
    assert "WD 15 mm" in tab.list.item(0).text() and tab.list.item(0).font().bold()
    tab.select(2)
    assert not tab.b_del.isEnabled() and not tab.b_up.isEnabled() and "(fixed step)" in tab.editor.title.text()

    assert win.open_schedule(demo["schedule"])
    assert tab.list.count() == 4 and "(9 fields)" in tab.list.item(3).text()
    assert win.tab_chips.table.rowCount() == 1 and win.tab_chips.table.item(0, 0).text() == "Demo chip"
    x, y = win.view._patterns.getData()
    assert np.nanmin(x) == pytest.approx(4760.0, abs=0.5) and np.nanmax(x) == pytest.approx(5040.0, abs=0.5)
    assert np.nanmax(y) == pytest.approx(5040.0, abs=0.5)  # the layout in SCS around the chip origin

    # the step editor
    tab.select(3)
    assert tab.b_del.isEnabled()
    win.replan()
    ed = tab.editor
    assert isinstance(ed.op, Exposure) and "9 field(s) of 100 µm" in ed.info.text()
    assert "Global Align in FIELD_R1C1" in ed.info.text() and "8 field(s) with a Local marker" in ed.info.text()
    boxes = {b.title().split(" (")[0]: b for b in ed.findChildren(QtWidgets.QGroupBox)}
    boxes["Local markers"].setChecked(False)
    assert not win.schedule[3].local_align and win.modified
    qtbot.waitUntil(lambda: isinstance(tab.editor.op, Exposure) and not tab.editor.op.local_align)
    win.schedule[3].local_align = True
    tab.editor.set_op(win.schedule[3])
    n = tab.editor.set_local_per_field(-45.0, 45.0)
    assert n == 9 and len(win.schedule[3].local_markers) == 9
    win.view.stepClicked.emit(3)
    assert tab.selected() == 3

    # stage positions persist with H = WD − Z; Go to (in focus) / XY; Replace with current
    pt = win.tab_positions
    pt.add("Spot", 100.0, 200.0, 3000.0)
    assert LithoSettings.load(win.settings.path).positions()["Spot"] == StagePosition("Spot", 100.0, 200.0, 3000.0)
    pt.table.selectRow(2)
    pt.b_go.click()
    qtbot.waitUntil(lambda: win.hw.calls and win.hw.calls[-1] == ("move", 100.0, 200.0, 12000.0))  # Z = 15000 − H
    pt.b_go_xy.click()
    qtbot.waitUntil(lambda: win.hw.calls[-1] == ("move", 100.0, 200.0))
    win.hw.z = 12500.0  # focused on it with Z
    pt.replace_selected(confirm=False)
    assert win.settings.positions()["Spot"] == StagePosition("Spot", 100.0, 200.0, 2500.0)
    assert LithoSettings.load(win.settings.path).positions()["Spot"].h == 2500.0 and pt.table.item(2, 3).text() == \
        "2500.0"
    # a position without a height: Go to (in focus) off, shown red; a second tab on the settings follows
    from easyebl.litho.module.tabs import StagePositionsTab

    other = StagePositionsTab(win.settings)
    qtbot.addWidget(other)
    pt.add("Old", 1.0, 2.0)
    assert other.table.rowCount() == 4 and other.table.item(3, 0).text() == "Old"
    other.table.selectRow(3)
    assert not other.b_go.isEnabled() and other.b_go_xy.isEnabled()
    other.delete_selected()
    assert pt.table.rowCount() == 3 and "Old" not in win.settings.positions()

    # save / reopen
    p = str(tmp_path / "edited.ebls.json")
    assert win.save_to(p) and not win.modified
    before = win.schedule.to_dict(str(tmp_path))
    assert win.open_schedule(p) and win.schedule.to_dict(str(tmp_path)) == before

    # run the demo: the WD, the Beam and the Field calibration (Accept each), then the Exposure with Global
    # Align + Local markers; the demo chip has a height: Z brings it into focus
    assert win.open_schedule(demo["schedule"])
    _fast(win.schedule)
    win.set_chip_height("Demo chip", ChipHeight("constant", 3900.0))
    win.hw.stage = win.hw.readback = np.array(ORIGIN)
    win.hw.stage_noise = 0.0
    win.hw.calls.clear()
    assert win.run(confirm=False)
    _waits(qtbot, lambda: win.beamcal_window).b_accept.click()
    _calibration_waits(win, qtbot).b_accept.click()
    assert win.wait(30)
    assert tab.states == ["done"] * 4, win.log_box.toPlainText()
    assert len(win.hw.exposures) == 9 and "9 of 9 field(s) exposed" in tab.list.item(3).toolTip()
    assert win.hw.calls[:2] == [("prepare",), ("wd", 15.0)] and ("focus", 10000) in win.hw.calls
    assert {c[3] for c in win.hw.calls if c[0] == "move" and len(c) == 4} == {11000.0, 11100.0}  # the chip: 15000 − 3900
    assert win.hw.calls[-1] == ("move", 2000.0, 2000.0, 11000.0)  # finished: parked at the calibration grid
    assert win.chips["Demo chip"].measured and win.l_global.text().startswith("Chip Demo chip: rot")
    win.replan()
    assert win.tab_chips.table.item(0, 3).text().startswith("rot")
    assert win.last_map is not None and "schedule finished" in win.log_box.toPlainText()
    win.reset_chip("Demo chip")
    assert "Demo chip" not in win.chips


def test_editor_types_chips_and_warnings(win, demo):
    s = Schedule([ColumnSetting(magnification=2000), Exposure(chip="A", path=Schedule.load(demo["schedule"])
                                                              .exposures[0].path), FieldCorrection()],
                 chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    assert [o.TYPE for o in win.schedule] == ["column", "beam_calibration", "field_correction", "exposure",
                                              "field_correction"]
    ws = win.validate()
    assert any("outside the field" in w for i, w in ws if i == 3)  # 80 µm fields in a 50 µm field
    assert "⚠" in win.tab_schedule.list.item(3).text()
    tab = win.tab_schedule
    tab.select(0)
    combo = tab.editor.findChildren(QtWidgets.QComboBox)[0]
    assert combo.itemText(0) == "Keep as is" and combo.itemData(1) == "default"
    combo.setCurrentIndex(1)
    assert win.schedule[0].preset == "default"
    wd = [sp for sp in tab.editor.findChildren(QtWidgets.QDoubleSpinBox) if sp.suffix() == " mm"]
    assert len(wd) == 1 and wd[0].value() == 15.0  # the schedule's WD: in the first Column setting
    wd[0].setValue(12.5)
    assert win.schedule.wd_mm == 12.5 and "WD 12.5 mm" in tab.list.item(0).text()
    win.hw.wd = 14.2371  # the column's WD now (from the focus)
    tab.editor.b_wd_current.click()
    assert win.schedule.wd_mm == 14.237 and wd[0].value() == 14.237 and "WD 14.237 mm" in tab.list.item(0).text()
    win.hw.wd = None  # not known: nothing changes
    assert not tab.editor.set_wd_to_current() and win.schedule.wd_mm == 14.237 and "cannot be read" in \
        tab.editor.msg.text()
    win.hw.wd = 15.0
    # the calibration settings are in the steps' editors
    tab.select(1)
    combos = tab.editor.findChildren(QtWidgets.QComboBox)
    pos = combos[0]
    assert pos.findData(GRID_POSITION) > 0 and "H 4000.0" in pos.itemText(pos.findData(GRID_POSITION))
    pos.setCurrentIndex(pos.findData(GRID_POSITION))
    assert win.schedule[1].position == GRID_POSITION
    px = [sp for sp in tab.editor.findChildren(QtWidgets.QSpinBox) if "px" in sp.suffix()][0]
    px.setValue(1000)
    assert win.schedule[1].config.pixels == 1000
    mf = [b for b in tab.editor.findChildren(QtWidgets.QCheckBox) if b.text().startswith("focus and stigmate")][0]
    assert mf.isChecked() and win.schedule[1].config.focus_magnification == 10000
    mf.setChecked(False)
    assert win.schedule[1].config.manual_focus is False
    mf.setChecked(True)
    tab.select(2)
    pos = tab.editor.findChildren(QtWidgets.QComboBox)[0]
    pos.setCurrentIndex(pos.findData(CUP_POSITION))
    assert win.schedule[2].config.position == CUP_POSITION
    keep = [b for b in tab.editor.findChildren(QtWidgets.QCheckBox) if b.text().startswith("keep the field")][0]
    assert not keep.isChecked()
    keep.setChecked(True)
    assert win.schedule[2].reuse and "in force" in tab.list.item(2).text()
    keep.setChecked(False)
    # Add after the fixed steps (a Field / Beam calibration set up like the fixed one)
    tab.select(1)
    tab.add_menu.actions()[1].trigger()  # Column setting
    tab.add_menu.actions()[2].trigger()  # Field calibration
    tab.add_menu.actions()[0].trigger()  # Exposure on the selected chip
    tab.add_menu.actions()[3].trigger()  # Beam calibration
    assert [o.TYPE for o in win.schedule] == ["column", "beam_calibration", "field_correction", "column",
                                              "field_correction", "exposure", "beam_calibration", "exposure",
                                              "field_correction"]
    assert win.schedule[5].chip == "A" and win.schedule[5].alignment_only
    assert win.schedule[4].position == CUP_POSITION and win.schedule[6].config.pixels == 1000
    tab.select(3)
    tab.delete()
    assert len(win.schedule) == 8
    for i in range(3):  # the fixed steps stay
        tab.select(i)
        tab.delete()
    assert len(win.schedule) == 8
    # chips: add, rename (the Exposures follow), move, delete only when unused
    ct = win.tab_chips
    name = ct.add_chip(origin=(100.0, 100.0))
    assert name == "Chip 1" and win.schedule.chip("Chip 1").origin == (100.0, 100.0)
    ct.table.item(0, 0).setText("B")
    assert win.schedule.exposures[0].chip == "B"
    ct.table.selectRow(0)
    ct.delete_selected()
    assert "used by" in ct.msg.text() and win.schedule.chip("B") is not None
    win.move_chip("B", 10.0, -5.0)
    assert win.schedule.chip("B").origin == (5010.0, 4995.0) and "moved chip B" in win.log_box.toPlainText()
    # an Exposure of a GDS file goes to the selected chip
    op = win.add_exposure_from_gds(next(e.path for e in win.schedule.exposures if e.path))
    assert op is not None and op.chip == "B" and "9 field(s)" in win.log_box.toPlainText()


def test_menu_bar(win):
    menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert menus == ["Schedule", "Step", "Calibration", "View", "Stage"]
    cal = [a.text() for a in win.menuBar().actions()[2].menu().actions() if a.text()]
    assert cal == ["Run the fixed steps only (column, beam and field calibration)",
                   "Forget the beam calibration of this condition", "Forget the field calibration"]
    assert not win.a_abort.isEnabled() and win.a_run.isEnabled()


def test_move_chip(win, demo, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import MoveFieldsDialog

    assert win.open_schedule(demo["schedule"])
    a = win.anchors("Demo chip")
    assert a["origin"][1] == ORIGIN and a["first"][1] == ORIGIN
    assert a["bbox"][1] == pytest.approx((4900.0, 4900.0))  # 3 × 3 fields, first at the top left in CCS; X mirrored
    assert a["marker"][1] == pytest.approx((5045.0, 4955.0))  # Global marker 1 = CCS (−45, −45)
    win.hw.stage = win.hw.readback = np.array([7000.0, 6000.0])
    d = MoveFieldsDialog(a, win.hw.stage_position(), click=(100.0, 200.0))
    qtbot.addWidget(d)
    assert d.from_buttons["origin"].isChecked() and d.to_buttons["click"].isChecked()
    assert d.value() == (-4900.0, -4800.0)

    def fake_exec(dlg):
        dlg.from_buttons["bbox"].setChecked(True)
        dlg.to_buttons["stage"].setChecked(True)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(MoveFieldsDialog, "exec", fake_exec)
    assert win.move_chip_dialog(click=(1.0, 2.0))
    assert win.anchors("Demo chip")["bbox"][1] == pytest.approx((7000.0, 6000.0))
    assert win.schedule.chips[0].origin == pytest.approx((7100.0, 6100.0))
    win.view.contextRequested.emit(5.0, 6.0, win.mapToGlobal(win.rect().center()))
    assert [x.text() for x in win._last_menu.actions() if x.text()][:2] == ["Move stage…", "Move the chip…"]
    win._last_menu.close()


def _calibration_schedule(win):
    s = Schedule()
    s[1].config = BeamCalConfig(pixels=800, position=GRID_POSITION)
    s[2].config = FieldCalConfig(position=CUP_POSITION, settle_s=0.0)
    win.set_schedule(s)
    return s


def test_the_fixed_steps_beam_then_field_calibration(win, qtbot):
    assert win.l_beam.text().startswith("Beam calibration: none for ×1000, WD 15.00 mm")
    _calibration_schedule(win)
    win.hw.stage_noise = 0.0
    assert win.run_calibration_steps()
    bw = _waits(qtbot, lambda: win.beamcal_window)
    assert bw.isVisible() and "X scale" in bw.numbers.text() and "+2" in bw.numbers.text()
    assert bw._img.image.shape == (800, 800)  # the full image, not binned
    assert win.hw.z == 15000.0 - H and win.beamcal() is None  # in focus through its height; not before Accept
    assert ("move", 2000.0, 2000.0, 15000.0 - H) in win.hw.calls
    bw.b_accept.click()
    cw = _calibration_waits(win, qtbot)  # the field calibration goes through it
    bc = win.beamcal()
    assert bc is not None and bc.sx == pytest.approx(1.002, abs=3e-5) and bc.sy == pytest.approx(0.999, abs=3e-5)
    assert len(LithoSettings.load(win.settings.path).beamcals) == 1  # saved
    assert cw.banner.isHidden()
    cw.b_accept.click()
    assert win.wait(30) and win.calibration.beamcal_id == bc.id
    assert win.tab_schedule.states == ["done"] * 3, win.log_box.toPlainText()
    win._update_status()
    assert win.l_beam.text().startswith("Beam calibration: sx") and win.l_cal.text().startswith("Field calibration: rot")
    assert win.l_wd.text().startswith("WD 15.000 mm")
    # now the schedule may run from a step after the fixed ones
    win.schedule.add(Exposure(chip="A", global_align=True, global_markers=[(0.0, 0.0), (10.0, 0.0)]))
    assert win.run_from_problem(3) is None
    win.hw.wd = 15.3  # another WD: neither applies, and no run from step 4
    win._update_status()
    assert "none for" in win.l_beam.text() and "not ×1000, WD 15.30 mm" in win.l_cal.text()
    assert "not at the schedule's WD" in win.run_from_problem(3) and not win.run(3, confirm=False)
    win.hw.wd = 15.0
    win.forget_beamcal()
    win._update_status()
    assert win.beamcal() is None and "another beam calibration" in win.l_cal.text()
    assert "no beam calibration" in win.run_from_problem(3) and not win.run(3, confirm=False)
    assert "cannot run from step 4" in win.log_box.toPlainText()
    # Discard, and Abort while it waits
    assert win.run_calibration_steps()
    _waits(qtbot, lambda: win.beamcal_window).b_discard.click()
    assert win.wait(30) and win.beamcal() is None and "discarded" in win.log_box.toPlainText()
    assert win.run_calibration_steps()
    _waits(qtbot, lambda: win.beamcal_window).b_accept.click()
    _calibration_waits(win, qtbot)
    win.abort()
    assert win.wait(30) and not win.fieldcal_window.awaiting


def test_field_calibration_mirror_mismatch_points_to_the_hardware_settings(win, qtbot):
    s = _calibration_schedule(win)
    s[1].reuse = True  # a stored beam calibration: no grid image
    win.settings.set_beamcal(BeamCalibration(condition=Condition(1000.0, None, 15.0)))
    win.hw.B = np.eye(2)  # this instrument's stage X runs along the image's x, the settings say mirrored
    win.hw.stage_noise = 0.0
    assert win.run_calibration_steps()
    cw = _calibration_waits(win, qtbot)
    assert not cw.banner.isHidden() and "Hardware → Configure" in cw.banner_text.text() and cw.b_fix.isHidden()
    cw.b_discard.click()
    assert win.wait(30) and win.calibration is None


def test_a_refused_calibration_can_be_used_anyway(win, qtbot):
    s = _calibration_schedule(win)
    s[1].reuse = True
    win.settings.set_beamcal(BeamCalibration(condition=Condition(1000.0, None, 15.0)))
    hw = win.hw
    real = hw.capture_image
    hw.capture_image = lambda w, px, frames=1: (np.random.default_rng(len(hw.calls)).normal(size=(px, px)),
                                                (w / px, w / px))  # nothing to match: the fit is refused
    hw.stage_noise = 0.0
    for use in (False, True):
        assert win.run_calibration_steps()
        cw = _calibration_waits(win, qtbot)
        assert cw.result.problem and cw.b_accept.isVisible() and cw.b_accept.text() == "Use anyway"
        assert "Use anyway to override" in cw.step.text()
        (cw.b_accept if use else cw.b_discard).click()
        assert win.wait(30)
        assert win.tab_schedule.states[2] == ("done" if use else "failed"), win.log_box.toPlainText()
    assert win.calibration is cw.result and "override" in win.log_box.toPlainText()
    # a good fit: Accept again
    hw.capture_image = real
    assert win.run_calibration_steps()
    cw = _calibration_waits(win, qtbot)
    assert cw.b_accept.text() == "Accept"
    cw.b_accept.click()
    assert win.wait(30)


def test_chip_heights_from_the_chips_tab(win, qtbot, demo):
    assert win.open_schedule(demo["schedule"])
    ct = win.tab_chips
    assert ct.table.item(0, 4).text() == "Z kept"
    # constant: focused on a feature with Z, Record here
    win.hw.z = 11500.0
    d = win.height_dialog("Demo chip", exec_=False)
    qtbot.addWidget(d)
    d.mode.setCurrentIndex(d.mode.findData("constant"))
    d.record_constant()
    h = d.value()
    assert h.mode == "constant" and h.value_um == 3500.0
    win.set_chip_height("Demo chip", h)
    assert ct.table.item(0, 4).text() == "H 3500.0 µm" and win.modified
    p = win._plan()[3]
    assert {f.z for f in p.fields} == {11500.0}
    # a map: three points at three stage positions (recorded in CCS)
    d = win.height_dialog("Demo chip", exec_=False)
    qtbot.addWidget(d)
    d.mode.setCurrentIndex(d.mode.findData("map"))
    for (x, y), z in (((5000.0, 5000.0), 11500.0), ((4900.0, 5000.0), 11510.0), ((5000.0, 5100.0), 11490.0)):
        win.hw.readback = np.array([x, y])
        win.hw.z = z
        d.record_point()
    h = d.value()
    assert h.mode == "map" and len(h.points) == 3 and h.points[1][:2] == pytest.approx((100.0, 0.0))  # X mirrored
    assert h.at((0.0, 0.0)) == pytest.approx(3500.0) and h.at((100.0, 0.0)) == pytest.approx(3490.0)
    win.set_chip_height("Demo chip", h)
    assert ct.table.item(0, 4).text().startswith("map 3 pts")
    zs = [f.z for f in win._plan()[3].fields]
    assert len(set(zs)) > 1 and min(zs) > 11000.0
    # two points are not a map: the plan warns and the exposure is refused
    win.set_chip_height("Demo chip", ChipHeight("map", points=h.points[:2]))
    assert any("at least 3" in w for i, w in win.validate() if i == 3)
    win.save_to(demo["schedule"])
    assert Schedule.load(demo["schedule"]).chips[0].height.points == [tuple(q) for q in h.points[:2]]


def test_stage_settings_dialog_shows_the_axes(win, monkeypatch):
    from easyebl.litho.module.dialogs import StageAxesDialog

    def fake_exec(dlg):
        assert "mirror X on" in dlg.findChildren(QtWidgets.QLabel)[0].text()
        dlg.tol.setValue(0.7)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(StageAxesDialog, "exec", fake_exec)
    win.edit_axes()
    s = LithoSettings.load(win.settings.path)
    assert s.stage_tolerance_um == 0.7 and win.axes == S3000_AXES


# ====================================================================== in the easyEBL program
@pytest.fixture
def ui(qtbot, tmp_path, monkeypatch):
    from easyebl.app.gui import EasyEBLApp

    qapp = QtWidgets.QApplication.instance()
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda *a: None)
    cfg = SystemConfig()
    cfg.path = str(tmp_path / "system.json")
    app = EasyEBLApp(cfg, eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01, use_tray=False,
                     confirm=lambda title, text: True)
    app.start()
    yield app
    for w in list(app.windows.values()):
        if hasattr(w, "ask_on_close"):
            w.ask_on_close = False
    if not app.closing:
        app.shutdown(ask=False)
    qapp.setQuitOnLastWindowClosed(True)


def test_litho_in_the_program_marks_are_clicked_in_the_sem_window(ui, qtbot):
    from easyebl.litho.module.hardware import BackendHardware

    assert "Lithography" in [a.text() for a in ui.menu.m_open.actions()]
    w = ui.open("litho")
    w.ask_on_close = False
    assert type(w).__name__ == "LithoWindow" and isinstance(w.hw, BackendHardware)
    assert w.axes == S3000_AXES  # from Hardware → Configure… (the default: mirror X)
    assert w.hw.field_um() == pytest.approx(ui.backend.engine.calibration.field_size_um)
    assert w.hw.field_at_1x_um() == pytest.approx(ui.backend.microscope.field_scale.k_um)
    assert w.hw.working_distance_mm() == pytest.approx(ui.backend.microscope.sem.working_distance_mm())
    # a calibration image of the whole field: narrowed to what the ±10 V range takes (line start, flyback)
    field = w.hw.field_um()
    img, (pxx, pxy) = w.hw.capture_image(field, 64)
    assert img.shape == (64, 64) and 0.6 * field < 64 * pxx < field and pxx == pytest.approx(pxy)
    img, (pxx, _) = w.hw.capture_image(0.5 * field, 64)
    assert 64 * pxx == pytest.approx(0.5 * field)  # fits: as asked

    req = MarkRequest("mark 1", (3.0, -2.0), (1003.0, 998.0), (3.0, -2.0), 10.0)
    out = {}
    threading.Thread(target=lambda: out.setdefault("xy", w.hw.measure_marks([req])), daemon=True).start()
    qtbot.waitUntil(lambda: "sem" in ui.windows and ui.windows["sem"].view.pick_mode, timeout=10000)
    sem = ui.windows["sem"]
    cfg = ui.backend.engine.config
    assert cfg.width_um == pytest.approx(10.0) and cfg.center_um == pytest.approx((3.0, -2.0))  # zoomed on the mark
    assert sem.view._pick_targets == [(3.0, -2.0)] and sem.view._pick_centre_cross and sem.pick_bar.isVisible()
    # each click centres the scan there; nothing is answered until Confirm
    eng = ui.backend.engine
    sem.view.pointPicked.emit(3.4, -1.7)
    assert eng.config.center_um == pytest.approx((3.4, -1.7)) and "1 click(s)" in sem.l_pick.text()
    sem.view.pointPicked.emit(3.5, -1.6)
    assert eng.config.center_um == pytest.approx((3.5, -1.6)) and "(+0.500, +0.400)" in sem.l_pick.text()
    sem.reset_pick()
    assert eng.config.center_um == pytest.approx((3.0, -2.0))
    sem.view.pointPicked.emit(1e4, 0.0)  # beyond the field: the centre stops at the edge
    assert eng.config.center_um[0] < 1e4 and "field edge" in sem.l_pick.text()
    sem.view.pointPicked.emit(3.45, -1.65)
    assert "xy" not in out and sem.view.pick_mode
    sem.b_pick_confirm.click()
    qtbot.waitUntil(lambda: "xy" in out, timeout=10000)
    assert out["xy"] == [pytest.approx((3.45, -1.65))] and not sem.view.pick_mode and not sem.pick_bar.isVisible()
    qtbot.waitUntil(lambda: not ui.backend.engine.is_scanning, timeout=10000)

    out.clear()
    threading.Thread(target=lambda: out.setdefault("xy", w.hw.measure_marks([req])), daemon=True).start()
    qtbot.waitUntil(lambda: sem.view.pick_mode, timeout=10000)
    sem.cancel_pick()
    qtbot.waitUntil(lambda: "xy" in out, timeout=10000)
    assert out["xy"] is None


def _program_fixed_steps(w, s, mag=1000.0):
    """The fixed steps of *s* on the program's simulators: step 1 sets the WD (15 mm), a stored (identity) beam
    calibration, the field calibration at a stage position in focus where the stage's Z is now."""
    mic = w.hw.bundle
    wd = s.wd_mm = 15.0
    mic.stage.refresh(("X", "Y", "Z")).result(30)
    z = float(mic.stage.positions["Z"])
    if "tex" not in w.settings.positions():
        w.settings.add_position("tex", 300.0, 300.0, 1000.0 * wd - z)
    s[1].reuse = True
    w.settings.set_beamcal(BeamCalibration(condition=Condition(mag, None, wd)))
    s[2].config = FieldCalConfig(position="tex", settle_s=0.1, image_px=256, pattern="cross")


def test_litho_runs_on_the_simulated_backend(ui, qtbot, demo):
    """Beam preset, magnification, stage moves and a real (simulated-DAQ) exposure through BackendHardware."""
    from easyebl.microscope.sem import PRESET_KEYS

    mic = ui.backend.microscope
    mic.settings.beam_presets.append(dict({k: 10 for k in PRESET_KEYS}, name="test"))
    w = ui.open("litho")
    w.ask_on_close = False
    gds = Schedule.load(demo["schedule"]).exposures[0].path
    s = Schedule([ColumnSetting(preset="test", magnification=900.0, current_pa=200.0),
                  Exposure(chip="A", path=gds, layers=[(1, 0)], dwell_us=0.5, step_nm=500.0)],
                 chips=[Chip("A", (300.0, 400.0))])
    _program_fixed_steps(w, s, mag=900.0)
    w.set_schedule(s)
    assert w.run(confirm=False)
    _calibration_waits(w, qtbot, timeout=180000).b_accept.click()
    qtbot.waitUntil(lambda: w.runner is None, timeout=180000)
    assert w.tab_schedule.states == ["done"] * 4, w.log_box.toPlainText()
    assert mic.sem.magnification() == 900
    assert w.hw.stage_position() == pytest.approx((300.0 - 200.0, 400.0 - 200.0), abs=1.0)  # the last field
    assert w.hw.move_dialogs >= 9 and w.hw.move_dialog is None
    # a move in progress (e.g. the user's) is waited for; Z moves too
    z0 = mic.stage.positions["Z"]
    fut = mic.stage.move({"X": 310.0})
    assert mic.stage.busy
    w.hw.move_stage(320.0, 400.0, z0 + 5.0)
    assert fut.done() and w.hw.stage_position() == pytest.approx((320.0, 400.0), abs=1.0)
    assert w.hw.stage_z() == pytest.approx(z0 + 5.0, abs=1.0)


def test_an_exposure_holds_the_daq_with_the_sem_on_external_scan(ui):
    """BackendHardware.expose: the pattern streams with the SEM on external scan (DAQ X / Y) and the DAQ
    reserved; afterwards the SEM is back on its internal scan and the lease is free – also on failure."""
    from easyebl.litho.compiler import VectorScanSettings
    from easyebl.litho.coords import FieldMap
    from easyebl.litho.geometry import Pattern, rectangle
    from easyebl.litho.runner import params_for_dwell

    w = ui.open("litho")
    w.ask_on_close = False
    h = w.hw.handle
    assert h.has_external_control
    seen = []
    real = h.run_stream

    def stream(*a, **kw):
        seen.append((h.external_control, h.holds_lease))
        return real(*a, **kw)

    h.run_stream = stream
    pat = Pattern([rectangle(-2.0, -2.0, 2.0, 2.0)])
    rep = w.hw.expose(pat, params_for_dwell(0.5e-6, 0.5), FieldMap(), VectorScanSettings())
    assert seen == [(True, True)] and not rep.aborted
    assert h.external_control is not True and not h.holds_lease and ui.backend.manager.lease is None

    def broken(*a, **kw):
        raise RuntimeError("stream failed")

    h.run_stream = broken
    with pytest.raises(RuntimeError, match="stream failed"):
        w.hw.expose(pat, params_for_dwell(0.5e-6, 0.5), FieldMap(), VectorScanSettings())
    assert h.external_control is not True and ui.backend.manager.lease is None


def test_field_calibration_in_the_program(ui, qtbot):
    """BackendHardware on the simulators: real stage moves, SEM captures and the simulated image following
    the stage – the calibration finds the simulators' hidden map (an identity beam calibration: as measured)."""
    from easyebl.app.backend import SIM_STAGE_TO_BEAM

    mic = ui.backend.microscope
    w = ui.open("litho")
    w.ask_on_close = False
    s = Schedule()
    _program_fixed_steps(w, s)
    w.set_schedule(s)
    assert w.run_calibration_steps()
    cw = _calibration_waits(w, qtbot, timeout=180000)
    assert cw.banner.isHidden(), cw.banner_text.text()
    cw.b_accept.click()
    qtbot.waitUntil(lambda: w.runner is None, timeout=30000)
    assert w.calibration is not None, w.log_box.toPlainText()
    assert np.allclose(w.calibration.stage_to_beam, np.asarray(SIM_STAGE_TO_BEAM), atol=5e-3)
    assert mic.stage.positions["X"] == pytest.approx(300.0, abs=0.5)


def test_sem_center_with_stage_in_the_program(ui, qtbot):
    """Right click → Center with stage: the stage moves by the click's offset through the configured mirror."""
    sem = ui.open("sem")
    st = ui.backend.microscope.stage
    assert sem.stage_mirror == (True, False) and sem._stage() is st
    st.refresh(("X", "Y")).result(30)
    x0, y0 = st.positions["X"], st.positions["Y"]
    c = sem.engine.config.center_um
    sem.center_with_stage(c[0] + 4.0, c[1] - 3.0)  # behind the modal stage dialog, which closes itself
    assert st.positions["X"] == pytest.approx(x0 - 4.0, abs=0.5) and st.positions["Y"] == pytest.approx(y0 - 3.0,
                                                                                                        abs=0.5)


def test_move_stage_dialog(win, demo, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import MoveStageDialog

    assert win.open_schedule(demo["schedule"])
    win.set_chip_height("Demo chip", ChipHeight("constant", 3900.0))
    click, markers, positions = win.stage_targets((4980.0, 5010.0))
    assert click == ("the clicked position (chip Demo chip)", (4980.0, 5010.0), 11100.0)  # on the chip: in focus
    assert win.stage_targets((100.0, 100.0))[0] == ("the clicked position", (100.0, 100.0), None)  # off: Z stays
    ms = markers["Demo chip"]
    assert len(ms) == 2 + 9 and ms[0][1] == pytest.approx((5045.0, 4955.0)) and ms[0][2] == 11100.0  # X mirrored
    assert [p[0] for p in positions] == [GRID_POSITION, CUP_POSITION] and positions[0][2] == 15000.0 - H
    d = win.move_stage_dialog(click=(4980.0, 5010.0), exec_=False)
    qtbot.addWidget(d)
    assert d.b_click.isChecked() and d.value() == (4980.0, 5010.0, 11100.0)
    d.marker.setCurrentIndex(1)  # → the marker option
    assert d.b_marker.isChecked() and d.value() == (pytest.approx(ms[1][1][0]), pytest.approx(ms[1][1][1]), 11100.0)
    d.position.setCurrentIndex(1)
    assert d.b_position.isChecked() and d.value() == (4700.0, 5000.0, 15000.0 - H)
    d.focus.setChecked(False)
    assert d.value() == (4700.0, 5000.0, None)
    # from the plot's context menu: the stage goes there
    picked = {}

    def fake_exec(dlg):
        dlg.b_position.setChecked(True)
        dlg.position.setCurrentIndex(0)
        picked["v"] = dlg.value()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(MoveStageDialog, "exec", fake_exec)
    win.view.contextRequested.emit(5.0, 6.0, win.mapToGlobal(win.rect().center()))
    win._last_menu.actions()[0].trigger()
    win._last_menu.close()
    assert picked["v"] == (2000.0, 2000.0, 15000.0 - H)
    qtbot.waitUntil(lambda: win.hw.calls and win.hw.calls[-1] == ("move", 2000.0, 2000.0, 15000.0 - H))


def test_focus_prompt_in_the_program(ui, qtbot):
    """Before a beam calibration: the SEM window scans live, a prompt that does not block it waits for Continue /
    Abort; the scan stops after.  A run starts with the SEM in RUN mode on Slow scan 1."""
    w = ui.open("litho")
    w.ask_on_close = False
    eng = ui.backend.engine
    for answer in (True, False):
        out = {}
        threading.Thread(target=lambda: out.setdefault("ok", w.hw.ask_focus("focus here")), daemon=True).start()
        qtbot.waitUntil(lambda: w.hw.focus_prompt is not None and w.hw.focus_prompt.isVisible(), timeout=10000)
        d = w.hw.focus_prompt
        assert not d.isModal() and "focus here" in d.text.text() and "sem" in ui.windows
        qtbot.waitUntil(lambda: eng.is_scanning, timeout=10000)
        (d.b_continue if answer else d.b_abort).click()
        qtbot.waitUntil(lambda: "ok" in out, timeout=10000)
        assert out["ok"] is answer and w.hw.focus_prompt is None
        qtbot.waitUntil(lambda: not eng.is_scanning, timeout=10000)
    sem = ui.backend.microscope.sem
    sem.set_running(False).result(10)
    sem.set_scan("F").result(10)
    w.hw.prepare_run()
    assert sem.state["running"] is True and sem.state["scan"] == "1"
    # already so: nothing is sent (the SEM refuses RUN ON when running, E2)
    sim = ui.backend.microscope.link.session.simulator
    n = len(sim.received)
    w.hw.prepare_run()
    assert not [x for x in sim.received[n:] if x.startswith(("#RUN", "#SCAN", "RUN ", "SCAN "))], sim.received[n:]
