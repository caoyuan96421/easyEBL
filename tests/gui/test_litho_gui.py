"""Lithography window (offscreen Qt): tabs, chips, step editor, schedule file, simulated runs, beam and field
calibration windows, field sizes, the calibration sample (Z_cal, WD1), chip heights (WD2), SEM mark picking, the
program's backend."""
import math
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("gdstk")
pytestmark = pytest.mark.gui

from PySide6 import QtCore, QtWidgets  # noqa: E402

from easyebl.app.config import SystemConfig  # noqa: E402
from easyebl.litho.beamcal import BeamCalConfig, BeamCalibration  # noqa: E402
from easyebl.litho.coords import S3000_AXES  # noqa: E402
from easyebl.litho.fieldcal import Condition  # noqa: E402
from easyebl.litho.height import ChipHeight  # noqa: E402
from easyebl.litho.runner import MarkRequest, SimulatedHardware  # noqa: E402
from easyebl.litho.schedule import Chip, ColumnSetting, Exposure, Schedule  # noqa: E402
from easyebl.litho.settings import FieldSize, LithoSettings, StagePosition  # noqa: E402

ORIGIN = (5000.0, 5000.0)
ZC = 11000.0  # the fixture's calibration sample: Z_cal …
WD1 = 11.0  # … and WD1 (the simulator's Z and WD); the demo's field size "250 µm" is ×500
CAL_NAME = "Calibration sample"


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
                           start_z=ZC, magnification=500, wd_mm=WD1)
    settings = LithoSettings.load(str(tmp_path / "litho.json"))
    settings.field_size = "250 µm"  # as selected in the Microscope window
    settings.replace_position(CAL_NAME, 2000.0, 2000.0, ZC)  # the grid
    settings.record_wd(WD1, ZC)  # WD1 of "250 µm"
    settings.replace_position("Faraday cup", 4700.0, 5000.0, ZC + 20.0)  # a texture there
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


def _stored_beamcal(win, mag=500.0, fs="250 µm"):
    win.settings.set_beamcal(BeamCalibration(condition=Condition(mag, None, WD1, fs)))


def test_window_open_edit_save_run(win, tmp_path, demo, qtbot):
    tab = win.tab_schedule
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Stage Positions", "Chips",
                                                                      "Exposure Schedule"]
    assert win.axes == S3000_AXES and win.view.flipped == (True, False)
    assert isinstance(win.schedule[0], ColumnSetting) and tab.list.count() == 2 and not tab.b_del.isEnabled()
    assert tab.list.item(0).font().bold()
    tab.select(1)
    assert not tab.b_del.isEnabled() and not tab.b_up.isEnabled() and "(fixed step)" in tab.editor.title.text()

    assert win.open_schedule(demo["schedule"])
    assert tab.list.count() == 3 and "(9 fields)" in tab.list.item(2).text() and "field 250 µm" in tab.list.item(0).text()
    assert win.tab_chips.table.rowCount() == 1 and win.tab_chips.table.item(0, 0).text() == "Demo chip"
    x, y = win.view._patterns.getData()
    assert np.nanmin(x) == pytest.approx(4760.0, abs=0.5) and np.nanmax(x) == pytest.approx(5040.0, abs=0.5)
    assert np.nanmax(y) == pytest.approx(5040.0, abs=0.5)  # the layout in SCS around the chip origin
    x, y = win.view._other.getData()  # every layer, gray: the alignment crosses (layer 2) too
    assert np.nanmin(x) == pytest.approx(4751.0, abs=0.5) and np.nanmax(x) == pytest.approx(5049.0, abs=0.5)

    # the step editor
    tab.select(2)
    assert tab.b_del.isEnabled()
    win.replan()
    ed = tab.editor
    assert isinstance(ed.op, Exposure) and "9 field(s) of 200 µm" in ed.info.text()
    assert "Global Align in FIELD_R1C1" in ed.info.text() and "8 field(s) with a Local marker" in ed.info.text()
    boxes = {b.title().split(" (")[0]: b for b in ed.findChildren(QtWidgets.QGroupBox)}
    boxes["Local markers"].setChecked(False)
    assert not win.schedule[2].local_align and win.modified
    qtbot.waitUntil(lambda: isinstance(tab.editor.op, Exposure) and not tab.editor.op.local_align)
    win.schedule[2].local_align = True
    tab.editor.set_op(win.schedule[2])
    n = tab.editor.set_local_per_field(-45.0, 45.0)
    assert n == 9 and len(win.schedule[2].local_markers) == 9
    tol = [sp for sp in tab.editor.findChildren(QtWidgets.QDoubleSpinBox) if "agree" in sp.toolTip()]
    assert len(tol) == 1 and tol[0].value() == 0.5
    tol[0].setValue(0.3)
    assert win.schedule[2].align_tolerance_um == 0.3
    win.view.stepClicked.emit(2)
    assert tab.selected() == 2

    # save / reopen
    p = str(tmp_path / "edited.ebls.json")
    assert win.save_to(p) and not win.modified
    before = win.schedule.to_dict(str(tmp_path))
    assert win.open_schedule(p) and win.schedule.to_dict(str(tmp_path)) == before

    # run the demo (a stored beam calibration): the field size, WD1; the Exposure with Global Align (confirmed) +
    # Local markers; the stage moves in X, Y only, the focus to the chip's WD2; parked at the calibration sample
    assert win.open_schedule(demo["schedule"])
    win.schedule[1].reuse = True
    _stored_beamcal(win)
    win.set_chip_height("Demo chip", ChipHeight("constant", wd_mm=11.3, z_um=ZC))
    win.hw.stage = win.hw.readback = np.array(ORIGIN)
    win.hw.stage_noise = 0.0
    win.hw.calls.clear()
    assert win.run(confirm=False)
    assert win.wait(30)
    assert tab.states == ["done"] * 3, win.log_box.toPlainText()
    assert len(win.hw.exposures) == 9 and "9 of 9 field(s) exposed" in tab.list.item(2).toolTip()
    calls = win.hw.calls
    assert calls[:3] == [("prepare",), ("mag", 500), ("wd", WD1)] and ("wd", 11.3) in calls
    assert all(len(c) == 3 for c in calls[:-2] if c[0] == "move") and win.hw.z == ZC  # Z stays
    assert calls[-2:] == [("move", 2000.0, 2000.0, ZC), ("wd", WD1)]  # parked: the calibration sample
    assert len(win.hw.measurements) == 4 + 8  # the Global Align markers twice, 8 Local markers
    assert win.chips["Demo chip"].measured and win.l_global.text().startswith("Chip Demo chip: rot")
    win.replan()
    assert win.tab_chips.table.item(0, 3).text().startswith("rot")
    assert win.last_map is not None and "schedule finished" in win.log_box.toPlainText()
    win.reset_chip("Demo chip")
    assert "Demo chip" not in win.chips


def test_stage_positions_dz_record_wd_and_replace(win, qtbot):
    from easyebl.litho.module.tabs import StagePositionsTab

    pt = win.tab_positions
    assert [pt.table.item(r, 0).text() for r in range(pt.table.rowCount())] == [CAL_NAME, "Faraday cup"]
    assert pt.table.item(0, 3).text() == "0 (reference)" and pt.table.item(1, 3).text() == "+20.0"
    assert "Calibration sample (250 µm): Z_cal 11000.0 µm, WD1 11.0000 mm" in pt.l_cal.text()
    pt.table.selectRow(0)
    assert not pt.b_del.isEnabled() and pt.b_replace.isEnabled()  # always kept
    # a position: ΔZ against the calibration sample; Go to XYZ: Z_cal + ΔZ and the focus to WD1
    win.hw.wd = 12.0
    pt.add("Spot", 100.0, 200.0, 30.0)
    assert LithoSettings.load(win.settings.path).positions()["Spot"] == StagePosition("Spot", 100.0, 200.0, 30.0)
    pt.table.selectRow(2)
    pt.b_go.click()
    qtbot.waitUntil(lambda: win.hw.calls[-2:] == [("move", 100.0, 200.0, ZC + 30.0), ("wd", WD1)])
    pt.b_go_xy.click()
    qtbot.waitUntil(lambda: win.hw.calls[-1] == ("move", 100.0, 200.0))
    win.hw.z = ZC + 50.0
    pt.replace_selected(confirm=False)
    assert win.settings.positions()["Spot"].dz == 50.0 and pt.table.item(2, 3).text() == "+50.0"
    # Record WD: WD1 and Z_cal from the column and the stage now; no other ΔZ changes
    win.hw.wd, win.hw.z = 11.05, ZC + 10.0
    assert pt.record_wd(confirm=False)
    cal = win.settings.calibration
    assert (cal.wd_mm, cal.z_um) == (11.05, ZC + 10.0) and win.settings.positions()["Spot"].dz == 50.0
    assert LithoSettings.load(win.settings.path).calibration.wd_mm == 11.05
    # Replace the calibration sample at another Z: the other ΔZ follow (their Z stays)
    pt.table.selectRow(0)
    win.hw.stage = win.hw.readback = np.array([2010.0, 2000.0])
    win.hw.z = ZC
    pt.replace_selected(confirm=False)
    assert cal.z_um == ZC and cal.x == 2010.0 and win.settings.positions()["Spot"].dz == 60.0
    assert win.settings.faraday.dz == 30.0 and cal.wd_mm == 11.05
    # a second tab on the settings follows; a position without ΔZ: Go to XYZ off; the fixed ones are not deleted
    other = StagePositionsTab(win.settings)
    qtbot.addWidget(other)
    pt.add("Old", 1.0, 2.0)
    assert other.table.rowCount() == 4 and other.table.item(3, 0).text() == "Old"
    other.table.selectRow(3)
    assert not other.b_go.isEnabled() and other.b_go_xy.isEnabled()
    other.delete_selected()
    assert pt.table.rowCount() == 3 and "Old" not in win.settings.positions()
    other.table.selectRow(1)
    other.delete_selected()
    assert pt.table.rowCount() == 3
    # no WD known: Record WD refuses
    win.hw.wd = None
    assert not pt.record_wd(confirm=False) and "not known" in pt.msg.text()


def test_editor_types_chips_and_warnings(win, demo):
    gds = Schedule.load(demo["schedule"]).exposures[0].path
    s = Schedule([ColumnSetting(field_size="100 µm"), Exposure(chip="A", path=gds)], chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    assert [o.TYPE for o in win.schedule] == ["column", "beam_calibration", "exposure"]
    ws = win.validate()
    assert any("outside the field" in w for i, w in ws if i == 2)  # 80 µm fields + crosses at ±49 in 83 µm
    assert any("the Microscope window is at field size 250 µm" in w for i, w in ws if i == 0)
    assert "⚠" in win.tab_schedule.list.item(2).text()
    tab = win.tab_schedule
    tab.select(0)
    fs = tab.editor.field_size_combo
    assert fs.currentData() == "100 µm" and fs.itemText(fs.findData("250 µm")) == "250 µm: ×500, WD 11 mm"
    fs.setCurrentIndex(fs.findData("250 µm"))
    assert win.schedule.field_size == "250 µm" and "field 250 µm" in tab.list.item(0).text()
    combo = [c for c in tab.editor.findChildren(QtWidgets.QComboBox) if c.count() and c.itemText(0) == "Keep as is"][0]
    assert combo.itemData(1) == "default"
    combo.setCurrentIndex(1)
    assert win.schedule[0].preset == "default"
    # the beam calibration settings are in its editor (always on the calibration sample)
    tab.select(1)
    assert not [c for c in tab.editor.findChildren(QtWidgets.QComboBox) if c.itemText(0).startswith("(choose")]
    px = [sp for sp in tab.editor.findChildren(QtWidgets.QSpinBox) if "px" in sp.suffix()][0]
    px.setValue(1000)
    assert win.schedule[1].config.pixels == 1000
    boxes = {b.text().split(",")[0]: b for b in tab.editor.findChildren(QtWidgets.QCheckBox)}
    mf = boxes["confirm the focus by hand first"]
    assert mf.isChecked() and win.schedule[1].config.focus_magnification == 10000
    mf.setChecked(False)
    assert win.schedule[1].config.manual_focus is False
    mf.setChecked(True)
    ed = tab.editor
    assert "none stored for field size 250 µm: the step measures" in ed.stored.text()
    boxes["use a stored calibration of the field size (else measure)"].setChecked(True)
    assert win.schedule[1].reuse and "stored one" in tab.list.item(1).text()
    win.replan()
    assert "(none stored: measures)" in tab.list.item(1).text()
    _stored_beamcal(win)  # stored with its field size: the step reuses it
    win.settings.set_beamcal(BeamCalibration(sx=1.01, condition=Condition(500.0, None, WD1, "500 µm")))  # not this
    win.replan()
    assert "(stored " in tab.list.item(1).text()
    assert "stored for field size 250 µm" in tab.editor.stored.text() and "used, nothing measured" in \
        tab.editor.stored.text()
    # Add after the fixed steps (a Beam calibration set up like the fixed one); a later Column setting: no field size
    tab.add_menu.actions()[1].trigger()  # Column setting
    assert not hasattr(tab.editor, "field_size_combo") or tab.editor.field_size_combo.parent() is None or \
        "The field size is the schedule's" in " ".join(lb.text() for lb in tab.editor.findChildren(QtWidgets.QLabel))
    tab.add_menu.actions()[0].trigger()  # Exposure on the selected chip
    tab.add_menu.actions()[2].trigger()  # Beam calibration
    assert [o.TYPE for o in win.schedule] == ["column", "beam_calibration", "column", "exposure", "beam_calibration",
                                              "exposure"]
    assert win.schedule[3].chip == "A" and win.schedule[3].alignment_only and win.schedule[4].config.pixels == 1000
    assert [a.text() for a in tab.add_menu.actions()] == ["Exposure", "Column setting", "Beam calibration"]
    tab.select(2)
    tab.delete()
    assert len(win.schedule) == 5
    for i in range(2):  # the fixed steps stay
        tab.select(i)
        tab.delete()
    assert len(win.schedule) == 5
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
    assert cal == ["Run the fixed steps only (column setting, beam calibration)", "Field calibration…",
                   "Field sizes…", "Forget the beam calibration of this field size",
                   "Forget the field calibration of this field size"]
    step = [a.text() for a in win.menuBar().actions()[1].menu().actions() if a.text()]
    assert "Add Field calibration" not in step and "Add Beam calibration" in step
    assert not win.a_abort.isEnabled() and win.a_run.isEnabled()
    assert win.l_fs.text() == "Field size 250 µm (Microscope), schedule ?"


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
    assert [x.text() for x in win._last_menu.actions() if x.text()][:2] == ["Move stage…", "Move pattern…"]
    win._last_menu.close()


def _beam_schedule(win):
    """The fixed steps at field size "100 µm" (×1200: 83 µm on the simulator), 800 px."""
    s = Schedule([ColumnSetting(field_size="100 µm")])
    s[1].config = BeamCalConfig(pixels=800)
    win.settings.field_size = "100 µm"
    assert win.settings.calibration.wd_mm is None  # WD1 is per field size: none for this one yet
    win.settings.record_wd(WD1, ZC)
    win.set_schedule(s)
    return s


def test_the_fixed_steps_and_the_beam_calibration(win, qtbot):
    assert win.l_beam.text().startswith("Beam calibration: none for 250 µm ×500")
    _beam_schedule(win)
    win.hw.stage_noise = 0.0
    assert win.run_calibration_steps()
    bw = _waits(qtbot, lambda: win.beamcal_window)
    assert bw.isVisible() and "X scale" in bw.numbers.text() and "+2" in bw.numbers.text()
    assert bw._img.image.shape == (800, 800)  # the full image, not binned
    assert win.hw.z == ZC and win.beamcal() is None  # on the calibration sample; not before Accept
    assert ("move", 2000.0, 2000.0, ZC) in win.hw.calls and ("focus", 10000) in win.hw.calls
    bw.b_accept.click()
    assert win.wait(30)
    bc = win.beamcal()
    assert bc is not None and bc.sx == pytest.approx(1.002, abs=3e-5) and bc.sy == pytest.approx(0.999, abs=3e-5)
    assert bc.condition.field_size == "100 µm" and len(LithoSettings.load(win.settings.path).beamcals) == 1  # saved
    assert win.tab_schedule.states == ["done"] * 2, win.log_box.toPlainText()
    win._update_status()
    assert win.l_beam.text().startswith("Beam calibration: sx") and win.l_wd.text() == "WD 11.000 mm (WD1 11)"
    # now the schedule may run from a step after the fixed ones
    win.schedule.add(Exposure(chip="A", global_align=True, global_markers=[(0.0, 0.0), (10.0, 0.0)]))
    win.schedule.add_chip("A", ORIGIN)
    assert win.run_problems(2) == []
    win.hw.z = ZC + 5.0  # Z moved: no run from step 3
    assert "not at the calibration sample's Z" in win.run_from_problem(2) and not win.run(2, confirm=False)
    win.hw.z = ZC
    win.hw.mag = 1000.0
    assert "not at the field size's ×1200" in win.run_from_problem(2)
    win.hw.mag = 1200.0
    win.forget_beamcal()
    assert win.beamcal() is None and "no beam calibration" in win.run_from_problem(2)
    assert not win.run(2, confirm=False) and "cannot run from step 3" in win.log_box.toPlainText()
    # Discard, and Abort while it waits
    assert win.run_calibration_steps()
    _waits(qtbot, lambda: win.beamcal_window).b_discard.click()
    assert win.wait(30) and win.beamcal() is None and "discarded" in win.log_box.toPlainText()
    assert win.run_calibration_steps()
    _waits(qtbot, lambda: win.beamcal_window)
    win.abort()
    assert win.wait(30) and not win.beamcal_window.awaiting


def test_the_focus_confirmed_by_hand_is_saved_as_wd1(win, qtbot):
    s = _beam_schedule(win)
    win.hw.refocus_wd = 11.08
    win.hw.stage_noise = 0.0
    assert win.run_calibration_steps()
    _waits(qtbot, lambda: win.beamcal_window).b_accept.click()
    assert win.wait(30)
    assert win.settings.calibration.wd_mm == 11.08 and LithoSettings.load(win.settings.path).calibration.wd_mm == 11.08
    assert "WD1 11.08 mm" in win.tab_positions.l_cal.text() or "WD1 11.0800 mm" in win.tab_positions.l_cal.text()
    assert s[1].config.manual_focus


def test_a_run_needs_the_microscope_at_the_schedules_field_size(win, demo):
    assert win.open_schedule(demo["schedule"])
    win.settings.field_size = "100 µm"
    assert not win.run(confirm=False) and "the Microscope window is at field size 100 µm" in win.log_box.toPlainText()
    win.settings.field_size = "500 µm"  # no WD1 recorded for it
    win.schedule.field_size = "500 µm"
    assert not win.run(confirm=False) and "no WD1" in win.log_box.toPlainText()
    win.settings.field_size = win.schedule.field_size = "250 µm"
    assert win.settings.calibration.wd_mm == WD1
    # a chip height recorded at another Z than Z_cal: not allowed to run
    win.set_chip_height("Demo chip", ChipHeight("constant", wd_mm=11.3, z_um=ZC + 40.0))
    assert any("chip Demo chip" in p and "not at the calibration sample's Z" in p for p in win.run_problems())
    assert not win.run(confirm=False)
    assert "⚠" in win.tab_chips.table.item(0, 4).toolTip() or win.tab_chips.table.item(0, 4).toolTip()


def test_the_field_calibration_dialog(win, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import FieldCalSetupDialog

    d = win.field_calibration_dialog(exec_=False)
    qtbot.addWidget(d)
    assert "Field size 250 µm" in d.findChildren(QtWidgets.QLabel)[0].text()
    assert not d.bb.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()  # no position chosen
    d.position.setCurrentIndex(d.position.findData("Faraday cup"))
    assert d.bb.button(QtWidgets.QDialogButtonBox.Ok).isEnabled() and d.value().position == "Faraday cup"

    def fake_exec(dlg):
        dlg.position.setCurrentIndex(dlg.position.findData("Faraday cup"))
        dlg.cfg.settle_s = 0.0
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(FieldCalSetupDialog, "exec", fake_exec)
    win.hw.stage_noise = 0.0
    _stored_beamcal(win)
    win.field_calibration_dialog()
    cw = _calibration_waits(win, qtbot)
    assert ("move", 4700.0, 5000.0, ZC + 20.0) in win.hw.calls  # Z_cal + ΔZ
    assert cw.banner.isHidden()
    cw.b_accept.click()
    assert win.wait(30) and win.calibration is not None
    assert win.settings.fieldcal_config.position == "Faraday cup"
    t = LithoSettings.load(win.settings.path)
    assert t.fieldcal("250 µm") is not None and t.fieldcal_config.position == "Faraday cup"  # kept across sessions
    win._update_status()
    assert win.l_cal.text().startswith("Field calibration: rot")
    # a wrong mirror setting: pointed to the hardware settings; Discard keeps the one in force
    keep = win.calibration
    win.hw.B = np.eye(2)
    assert win.run_field_calibration()
    cw = _calibration_waits(win, qtbot)
    assert not cw.banner.isHidden() and "Hardware → Configure" in cw.banner_text.text()
    cw.b_discard.click()
    assert win.wait(30) and win.calibration is keep
    # refused (nothing to match): Use anyway overrides
    win.hw.B = win.axes.matrix
    hw = win.hw
    hw.capture_image = lambda w, px, frames=1: (np.random.default_rng(len(hw.calls)).normal(size=(px, px)),
                                                (w / px, w / px))
    assert win.run_field_calibration()
    cw = _calibration_waits(win, qtbot)
    assert cw.result.problem and cw.b_accept.text() == "Use anyway"
    cw.b_accept.click()
    assert win.wait(30) and win.calibration is cw.result and "override" in win.log_box.toPlainText()
    win.forget_calibration()
    assert win.calibration is None and LithoSettings.load(win.settings.path).fieldcal("250 µm") is None
    # not at the field size's magnification: refused
    win.hw.mag = 1000.0
    assert win.run_field_calibration() and win.wait(30)
    assert "not field size 250 µm's ×500" in win.log_box.toPlainText()


def test_the_field_sizes_dialog(win, qtbot):
    from easyebl.litho.module.tabs import PositionsHub

    d = win.field_sizes_dialog(exec_=False)
    qtbot.addWidget(d)
    assert d.table.rowCount() == 3 and d.table.item(1, 3).text() == "200"  # 100 mm at ×1 / 500
    ok = d.bb.button(QtWidgets.QDialogButtonBox.Ok)
    d.table.item(0, 1).setText("1234")  # not a magnification the SEM accepts
    assert not ok.isEnabled() and "×1234" in d.msg.text()
    d.table.item(0, 1).setText("1000")
    assert ok.isEnabled() and d.table.item(0, 3).text() == "100"
    d.table.item(2, 0).setText("100 µm")
    assert not ok.isEnabled() and "same name" in d.msg.text()
    d.table.item(2, 0).setText("500 µm")
    seen = []
    PositionsHub.instance().changed.connect(lambda st: seen.append(st))
    win.set_field_sizes(d.value())
    assert seen and win.settings.field_sizes[0] == FieldSize("100 µm", 1000, 7.0)
    assert LithoSettings.load(win.settings.path).field_sizes[0].magnification == 1000


def test_chip_heights_from_the_chips_tab(win, qtbot, demo):
    assert win.open_schedule(demo["schedule"])
    ct = win.tab_chips
    assert ct.table.item(0, 4).text() == "focus kept"
    # constant: focused on a feature with the stage at Z_cal, Record here takes the WD
    win.hw.wd = 11.3
    d = win.height_dialog("Demo chip", exec_=False)
    qtbot.addWidget(d)
    assert "Z_cal (11000.0 µm)" in d.findChildren(QtWidgets.QLabel)[0].text()
    d.mode.setCurrentIndex(d.mode.findData("constant"))
    d.record_constant()
    h = d.value()
    assert h.mode == "constant" and h.wd_mm == 11.3 and h.z_um == ZC
    win.set_chip_height("Demo chip", h)
    assert ct.table.item(0, 4).text() == "WD2 11.300 mm at Z 11000 µm" and win.modified
    p = win._plan()[2]
    assert {f.wd for f in p.fields} == {11.3}
    # not at Z_cal: refused
    win.hw.z = ZC + 10.0
    d = win.height_dialog("Demo chip", exec_=False)
    qtbot.addWidget(d)
    d.mode.setCurrentIndex(d.mode.findData("map"))
    d.record_point()
    assert not d.value().points and "is at Z 11010.0 µm" in d.msg.text()
    win.hw.z = ZC
    # a map: three points at three stage positions (recorded in CCS), each with its WD
    for (x, y), wd in (((5000.0, 5000.0), 11.3), ((4900.0, 5000.0), 11.31), ((5000.0, 5100.0), 11.29)):
        win.hw.readback = np.array([x, y])
        win.hw.wd = wd
        d.record_point()
    h = d.value()
    assert h.mode == "map" and len(h.points) == 3 and h.points[1][:2] == pytest.approx((100.0, 0.0))  # X mirrored
    assert h.wd_at((0.0, 0.0)) == pytest.approx(11.3) and h.wd_at((100.0, 0.0)) == pytest.approx(11.31)
    win.set_chip_height("Demo chip", h)
    assert ct.table.item(0, 4).text().startswith("map 3 pts")
    wds = [f.wd for f in win._plan()[2].fields]
    assert len(set(wds)) > 1 and min(wds) > 11.0
    # Go to a point: X, Y at Z_cal, the focus to its WD
    d.table.selectRow(1)
    d._goto_point()
    qtbot.waitUntil(lambda: win.hw.calls[-1] == ("wd", 11.31))
    assert win.hw.calls[-2] == ("move", pytest.approx(4900.0), pytest.approx(5000.0), ZC)
    # two points are not a map: the plan warns and the exposure is refused
    win.set_chip_height("Demo chip", ChipHeight("map", points=h.points[:2], z_um=ZC))
    assert any("at least 3" in w for i, w in win.validate() if i == 2)
    win.save_to(demo["schedule"])
    assert Schedule.load(demo["schedule"]).chips[0].height.points == [tuple(q) for q in h.points[:2]]
    # more than 1 mm from WD1 (11 mm): warned, also in the dialog
    win.set_chip_height("Demo chip", ChipHeight("constant", wd_mm=12.5, z_um=ZC))
    assert any("WD2 12.500 mm more than 1 mm from WD1 11 mm" in w for i, w in win.validate() if i == 2)
    d = win.height_dialog("Demo chip", exec_=False)
    qtbot.addWidget(d)
    assert "from WD1 11 mm" in d.info.text()


def test_every_gds_layer_is_drawn_gray(win, tmp_path):
    """The layers the Exposure does not write are drawn gray – also shapes outside its fields (in the top cell
    itself, or in cells without an exposed layer)."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    f = lib.new_cell("F")
    f.add(gdstk.rectangle((-10, -10), (10, 10), layer=1))
    m = lib.new_cell("MARKS")  # no layer 1 in it: not a field of the Exposure
    m.add(gdstk.rectangle((300, 0), (310, 10), layer=2))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(f, (0, 0)), gdstk.Reference(m, (0, 0)))
    top.add(gdstk.rectangle((-200, -200), (-190, -190), layer=3))  # in the top cell itself
    path = str(tmp_path / "layers.gds")
    lib.write_gds(path)
    s = Schedule([ColumnSetting(field_size="250 µm"), Exposure(chip="A", path=path, layers=[(1, 0)])],
                 chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    win.replan()
    x, _ = win.view._patterns.getData()
    assert np.nanmin(x) == pytest.approx(4990.0) and np.nanmax(x) == pytest.approx(5010.0)  # layer 1: the field
    x, y = win.view._other.getData()
    assert np.nanmin(x) == pytest.approx(5000.0 - 310.0) and np.nanmax(x) == pytest.approx(5000.0 + 200.0)  # X mirrored
    assert np.nanmin(y) == pytest.approx(4800.0)
    s.exposures[0].layers = None  # every layer written: what is in no field (the top cell's own shapes) still gray
    win.replan()
    x, y = win.view._other.getData()
    assert np.nanmax(x) == pytest.approx(5000.0 + 200.0) and np.nanmin(y) == pytest.approx(4800.0)


def _dose_gds(tmp_path):
    """Two fields (cells A, B) of squares with dose factors as GDS property 7 (1.0, 1.5, 2.0) and
    property 3 holding text."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("TOP")
    for k, (name, x) in enumerate((("A", 0.0), ("B", 120.0))):
        c = lib.new_cell(name)
        for j, d in enumerate((1.0, 1.5, 2.0)):
            r = gdstk.rectangle((-30 + 20 * j, -5), (-20 + 20 * j, 5), layer=1)
            r.set_gds_property(7, f"{d:g}")
            r.set_gds_property(3, "no number")
            c.add(r)
        top.add(gdstk.Reference(c, (x, 0)))
    path = str(tmp_path / "dose.gds")
    lib.write_gds(path)
    return path


def test_the_toolbar_and_the_drawing_modes(win, tmp_path, qtbot):
    path = _dose_gds(tmp_path)
    s = Schedule([ColumnSetting(field_size="250 µm"), Exposure(chip="A", path=path, step_nm=500.0)],
                 chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    win.tab_schedule.select(0)
    win.replan()
    texts = [a.text() for a in win.tools.actions() if a.text()]
    assert texts == ["Zoom to pattern", "Move pattern", "Move stage", "Color by Field", "Color by Dose",
                     "Writing order", "Show dots"]
    assert win.tools.toolButtonStyle() == QtCore.Qt.ToolButtonTextUnderIcon
    v = win.view
    # by field: each field its colour
    win.a_color_field.setChecked(True)
    filled = [it for it in v._colored if it.xData is not None and len(it.xData)]
    assert v.color_mode == "field" and len(filled) == 2 and not len(v._patterns.xData or [])
    # by dose (exclusive with by field): the dose factors of the GDS property
    s.exposures[0].dose.mode = "property"
    s.exposures[0].dose.property_attr = 7
    win.a_color_dose.setChecked(True)
    assert not win.a_color_field.isChecked() and v.color_mode == "dose"
    assert v.dose_range == pytest.approx((1.0, 2.0)) and v.legend.isVisibleTo(v)
    filled = [it for it in v._colored if it.xData is not None and len(it.xData)]
    assert len(filled) == 3  # the lowest, middle and highest bins
    win.a_color_dose.setChecked(False)
    assert v.color_mode == "plain" and len(v._patterns.xData)
    # the writing order: the field centres in turn, numbered
    win.a_order.setChecked(True)
    assert len(v._order.xData) == 2 and len(v._order_labels) == 2
    # the exposure points: 3 squares of 10 µm at 0.5 µm per field
    win.a_dots.setChecked(True)
    n = len(v._dots.xData)
    assert n == pytest.approx(2 * 3 * 20 * 20, rel=0.15) and "exposure points" in v.dots_note
    win.a_dots.setChecked(False)
    win.a_order.setChecked(False)
    assert v._dots.xData is None or len(v._dots.xData) == 0
    assert not v._order_labels
    # Move pattern (was Move the chip): menu, toolbar and the plot's right-click menu
    assert win.a_move.text() == "Move pattern…"


def test_the_dose_factor_property_is_chosen(win, tmp_path):
    from easyebl.litho.gds import dose_properties

    path = _dose_gds(tmp_path)
    assert dose_properties(path) == {7: (6, 1.0, 2.0)}  # property 3 holds no number
    s = Schedule([Exposure(chip="A", path=path)], chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    ed = win.tab_schedule.editor
    ed.set_op(s.exposures[0])
    box = ed.dose_box
    items = [box.itemText(i) for i in range(box.count())]
    assert items[:3] == ["none (all 1)", "any property with a number", "property 7 (6 shapes, 1 … 2)"]
    box.setCurrentIndex(2)
    d = s.exposures[0].dose
    assert d.mode == "property" and d.property_attr == 7
    win.replan()
    box.setCurrentIndex(0)
    assert s.exposures[0].dose.mode == "none"


def _marks_gds(tmp_path):
    """F: layer 1 square ±10 µm (exposed); in the top cell itself a layer-2 mark: a 10 µm square at
    (−200 … −190, −200 … −190)."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    f = lib.new_cell("F")
    f.add(gdstk.rectangle((-10, -10), (10, 10), layer=1))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(f, (0, 0)))
    top.add(gdstk.rectangle((-200, -200), (-190, -190), layer=2))
    path = str(tmp_path / "marks.gds")
    lib.write_gds(path)
    return path


def test_markers_are_picked_on_polygon_corners_and_centres(win, tmp_path, qtbot):
    from easyebl.litho.module.view import snap_targets

    op = Exposure(chip="A", path=_marks_gds(tmp_path), layers=[(1, 0)], offset=(5.0, 0.0), global_align=True,
                  global_markers=[(0.0, 0.0), (1.0, 1.0)], local_align=True)
    s = Schedule([ColumnSetting(field_size="250 µm"), op], chips=[Chip("A", ORIGIN)])
    win.set_schedule(s)
    ccs, kinds = snap_targets(op, win.library)
    assert len(ccs) == 10 and kinds.count("centre") == 2  # 4 corners + the centre of each square (CCS = GDS + 5, 0)
    assert any(np.allclose(c, (-190.0, -195.0)) for c in ccs) and any(np.allclose(c, (5.0, 0.0)) for c in ccs)
    tab = win.tab_schedule
    tab.select(2)
    ed = tab.editor
    v = win.view
    g = s.chips[0].nominal(win.axes)

    def click(ccs_pt, free=False, off=(0.0, 0.0)):
        x, y = g.to_scs(np.asarray(ccs_pt, float)) + np.asarray(off)
        return v.pick_at(float(x), float(y), free=free)

    v.getPlotItem().getViewBox().setRange(xRange=(4700, 5300), yRange=(4700, 5300), padding=0)
    # one Global marker: the corner of the gray (layer 2, top-cell) square
    [b for b in ed.findChildren(QtWidgets.QPushButton) if b.text() == "Pick"][1].click()
    assert v.picking and "Global marker 2 of 2" in v.pick_banner.text()
    assert click((-185.0, -190.0), off=(0.3, 0.0))  # 0.3 µm off: snaps (well within 12 px)
    assert op.global_markers[1] == (-185.0, -190.0) and not v.picking and win.modified
    # all in turn: the exposed square's centre, then a corner
    ed = tab.editor
    ed.b_pick_global.click()
    assert click((5.0, 0.0)) and v.picking and "Global marker 2 of 2" in v.pick_banner.text()
    assert not click((100.0, 100.0))  # nothing near: nothing taken
    assert click((15.0, 10.0)) and not v.picking
    assert op.global_markers == [(5.0, 0.0), (15.0, 10.0)]
    # Local markers: each click adds one; Shift-click takes the point itself; Esc ends
    tab.editor.b_pick_local.click()
    assert click((-190.0, -195.0))  # the gray square's centre
    assert click((50.0, 60.0), free=True)
    assert "2 added" in v.pick_banner.text()
    v.pickCancelled.emit()
    assert not v.picking and op.local_markers == [(-190.0, -195.0), pytest.approx((50.0, 60.0))]
    assert tab.editor.local_table.rowCount() == 2
    # snapping is in screen pixels: zoomed far out a 30 µm miss still snaps, zoomed in it does not
    win.pick_markers("global", 0)
    v.getPlotItem().getViewBox().setRange(xRange=(4990, 5010), yRange=(4990, 5010), padding=0)
    assert not click((5.0, 0.0), off=(3.0, 0.0))
    v.getPlotItem().getViewBox().setRange(xRange=(0, 10000), yRange=(0, 10000), padding=0)
    assert click((5.0, 0.0), off=(3.0, 0.0)) and op.global_markers[0] == (5.0, 0.0)
    # hovering rings the target a click would take
    from PySide6 import QtCore

    win.pick_markers("local")
    vb = v.getPlotItem().getViewBox()
    x, y = g.to_scs(np.array([5.0, 0.0]))
    v._moved(vb.mapViewToScene(QtCore.QPointF(float(x), float(y))))
    assert np.allclose(v._snap.getData()[0], [x]) and np.allclose(v._snap.getData()[1], [y])
    v._moved(vb.mapViewToScene(QtCore.QPointF(2000.0, 2000.0)))
    assert len(v._snap.getData()[0]) == 0
    v.pickCancelled.emit()
    # in pick mode a click does not select a field, a right click cancels
    win.pick_markers("local")
    v.pickCancelled.emit()
    assert not v.picking


def test_the_full_fov_and_a_reduced_imaging_scan(win):
    """Green: the full field at the stage; red: the SEM imaging scan, only when reduced or shifted."""
    win.hw.stage = win.hw.readback = np.array(ORIGIN)
    win._tick()
    x, _ = win.view._fov.getData()
    assert np.nanmax(x) - np.nanmin(x) == pytest.approx(200.0) and win.view._scan.getData()[0] is None or \
        len(win.view._scan.getData()[0]) == 0
    win.hw.scan = ((0.0, 0.0), 200.0, 200.0, 0.0)  # the whole field: nothing more
    win._tick()
    assert not win.view._scan_label.isVisible()
    win.hw.scan = ((20.0, -10.0), 40.0, 30.0, 0.0)  # zoomed and shifted
    win._tick()
    x, y = win.view._scan.getData()
    assert win.view._scan_label.isVisible()
    assert np.nanmax(x) - np.nanmin(x) == pytest.approx(40.0) and np.nanmax(y) - np.nanmin(y) == pytest.approx(30.0)
    assert np.mean(x[:4]) == pytest.approx(5000.0 - 20.0) and np.mean(y[:4]) == pytest.approx(5000.0 - 10.0)  # X mirrored
    win.hw.scan = None
    win._tick()
    assert not win.view._scan_label.isVisible()


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


def test_move_stage_dialog(win, demo, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import MoveStageDialog

    assert win.open_schedule(demo["schedule"])
    win.set_chip_height("Demo chip", ChipHeight("constant", wd_mm=11.3, z_um=ZC))
    click, markers, positions = win.stage_targets((4980.0, 5010.0))
    assert click == ("the clicked position (chip Demo chip)", (4980.0, 5010.0), ZC, 11.3)  # on the chip: WD2
    assert win.stage_targets((100.0, 100.0))[0] == ("the clicked position", (100.0, 100.0), None, None)  # off
    ms = markers["Demo chip"]
    assert len(ms) == 2 + 9 and ms[0][1] == pytest.approx((5045.0, 4955.0)) and ms[0][2:] == (ZC, 11.3)  # X mirrored
    assert [p[0] for p in positions] == [CAL_NAME, "Faraday cup"] and positions[1][2:] == (ZC + 20.0, WD1)
    d = win.move_stage_dialog(click=(4980.0, 5010.0), exec_=False)
    qtbot.addWidget(d)
    assert d.b_click.isChecked() and d.value() == (4980.0, 5010.0, ZC, 11.3)
    d.marker.setCurrentIndex(1)  # → the marker option
    assert d.b_marker.isChecked() and d.value() == (pytest.approx(ms[1][1][0]), pytest.approx(ms[1][1][1]), ZC, 11.3)
    d.position.setCurrentIndex(1)
    assert d.b_position.isChecked() and d.value() == (4700.0, 5000.0, ZC + 20.0, WD1)
    d.focus.setChecked(False)
    assert d.value() == (4700.0, 5000.0, None, None)
    # from the plot's context menu: the stage goes there
    picked = {}

    def fake_exec(dlg):
        dlg.b_position.setChecked(True)
        dlg.position.setCurrentIndex(0)
        picked["v"] = dlg.value()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(MoveStageDialog, "exec", fake_exec)
    win.hw.wd = 12.0
    win.view.contextRequested.emit(5.0, 6.0, win.mapToGlobal(win.rect().center()))
    win._last_menu.actions()[0].trigger()
    win._last_menu.close()
    assert picked["v"] == (2000.0, 2000.0, ZC, WD1)
    qtbot.waitUntil(lambda: win.hw.calls[-2:] == [("move", 2000.0, 2000.0, ZC), ("wd", WD1)])


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
    assert w.hw.scan_area()[:2] == (pytest.approx((3.0, -2.0)), pytest.approx(10.0))
    w.hw.bundle.stage.refresh(("X", "Y")).result(30)
    w._tick()
    assert w.view._scan_label.isVisible()  # the zoomed imaging scan, red, inside the green field
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


def _program_setup(w, mag=1000):
    """On the program's simulators: a field size "t" at ×*mag* selected, the calibration sample where the stage
    is now (Z_cal, WD1 = the WD now), a stored (identity) beam calibration, the column at ×*mag*."""
    mic = w.hw.bundle
    mic.stage.refresh(("X", "Y", "Z")).result(30)
    x, y, z = (float(mic.stage.positions[k]) for k in ("X", "Y", "Z"))
    wd = float(mic.sem.working_distance_mm())
    st = w.settings
    st.set_field_sizes(st.field_sizes + [FieldSize("t", mag, round(wd, 3))])
    st.field_size = "t"
    st.replace_position(CAL_NAME, x, y, z)
    st.record_wd(wd, z)
    if "tex" not in st.positions():
        st.add_position("tex", 300.0, 300.0, 0.0)
    st.set_beamcal(BeamCalibration(condition=Condition(float(mag), None, wd, "t")))
    mic.sem.set_magnification(mag).result(30)
    return x, y, z, wd


def test_litho_runs_on_the_simulated_backend(ui, qtbot, demo):
    """Beam preset, magnification, stage moves and a real (simulated-DAQ) exposure through BackendHardware."""
    from easyebl.microscope.sem import PRESET_KEYS

    mic = ui.backend.microscope
    mic.settings.beam_presets.append(dict({k: 10 for k in PRESET_KEYS}, name="test"))
    w = ui.open("litho")
    w.ask_on_close = False
    gds = Schedule.load(demo["schedule"]).exposures[0].path
    x, y, z, wd = _program_setup(w, mag=900)
    s = Schedule([ColumnSetting(preset="test", field_size="t", current_pa=200.0),
                  Exposure(chip="A", path=gds, layers=[(1, 0)], dwell_us=0.5, step_nm=500.0)],
                 chips=[Chip("A", (300.0, 400.0))])
    s[1].reuse = True
    w.set_schedule(s)
    assert w.run(confirm=False)
    qtbot.waitUntil(lambda: w.runner is None, timeout=180000)
    assert w.tab_schedule.states == ["done"] * 3, w.log_box.toPlainText()
    assert mic.sem.magnification() == 900
    assert w.hw.stage_position() == pytest.approx((x, y), abs=1.0)  # parked at the calibration sample
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
    the stage – the standalone field calibration finds the simulators' hidden map (an identity beam calibration:
    as measured)."""
    from easyebl.app.backend import SIM_STAGE_TO_BEAM

    mic = ui.backend.microscope
    w = ui.open("litho")
    w.ask_on_close = False
    _program_setup(w)
    w.settings.fieldcal_config.position = "tex"
    w.settings.fieldcal_config.settle_s = 0.1
    w.settings.fieldcal_config.pattern = "cross"
    assert w.run_field_calibration()
    cw = _calibration_waits(w, qtbot, timeout=180000)
    assert cw.banner.isHidden(), cw.banner_text.text()
    cw.b_accept.click()
    qtbot.waitUntil(lambda: w.runner is None, timeout=30000)
    assert w.calibration is not None, w.log_box.toPlainText()
    assert np.allclose(w.calibration.stage_to_beam, np.asarray(SIM_STAGE_TO_BEAM), atol=5e-3)
    assert mic.stage.positions["X"] == pytest.approx(300.0, abs=0.5)
    assert LithoSettings.load(w.settings.path).fieldcal("t") is not None  # kept for the field size


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


def test_the_alignment_prompt_in_the_program(ui, qtbot):
    """The Global Align passes disagree: Repeat / Continue anyway / Abort."""
    w = ui.open("litho")
    w.ask_on_close = False
    for k, want in enumerate(("repeat", "continue", "abort")):
        out = {}
        threading.Thread(target=lambda: out.setdefault("a", w.hw.ask_alignment("passes differ")), daemon=True).start()
        qtbot.waitUntil(lambda: getattr(w.hw, "alignment_box", None) is not None
                        and w.hw.alignment_box.isVisible(), timeout=10000)
        box = w.hw.alignment_box
        assert "passes differ" in box.text()
        box.buttons()[[b.text() for b in box.buttons()].index(("Repeat", "Continue anyway", "Abort")[k])].click()
        qtbot.waitUntil(lambda: "a" in out, timeout=10000)
        assert out["a"] == want
