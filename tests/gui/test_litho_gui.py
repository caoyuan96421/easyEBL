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
from easyebl.litho.beamcal import BeamCalConfig  # noqa: E402
from easyebl.litho.coords import S3000_AXES  # noqa: E402
from easyebl.litho.fieldcal import FieldCalConfig  # noqa: E402
from easyebl.litho.runner import MarkRequest, SimulatedHardware  # noqa: E402
from easyebl.litho.schedule import Chip, ColumnSetting, Exposure, FieldCorrection, Schedule  # noqa: E402
from easyebl.litho.settings import LithoSettings  # noqa: E402

ORIGIN = (5000.0, 5000.0)


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
                           beam_linear=[[1.002, 0.0], [0.0, 0.999]], beam_k=1e-7, grid_at=((2000.0, 2000.0), 60.0))
    settings = LithoSettings.load(str(tmp_path / "litho.json"))
    settings.fieldcal = FieldCalConfig(x=4700.0, y=5000.0, settle_s=0.0)
    settings.add_position("grid", 2000.0, 2000.0, 11.0)
    settings.beamcal = BeamCalConfig(pixels=800, position="grid")
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


def test_window_open_edit_save_run(win, tmp_path, demo, qtbot):
    tab = win.tab_schedule
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Stage Positions", "Chips",
                                                                      "Exposure Schedule"]
    assert win.axes == S3000_AXES and win.view.flipped == (True, False)
    assert isinstance(win.schedule[0], ColumnSetting) and tab.list.count() == 1 and not tab.b_del.isEnabled()

    assert win.open_schedule(demo["schedule"])
    assert tab.list.count() == 3 and "(9 fields)" in tab.list.item(2).text()
    assert win.tab_chips.table.rowCount() == 1 and win.tab_chips.table.item(0, 0).text() == "Demo chip"
    x, y = win.view._patterns.getData()
    assert np.nanmin(x) == pytest.approx(4760.0, abs=0.5) and np.nanmax(x) == pytest.approx(5040.0, abs=0.5)
    assert np.nanmax(y) == pytest.approx(5040.0, abs=0.5)  # the layout in SCS around the chip origin

    # the step editor
    tab.select(2)
    win.replan()
    ed = tab.editor
    assert isinstance(ed.op, Exposure) and "9 field(s) of 100 µm" in ed.info.text()
    assert "Global Align in FIELD_R1C1" in ed.info.text() and "8 field(s) with a Local marker" in ed.info.text()
    boxes = {b.title().split(" (")[0]: b for b in ed.findChildren(QtWidgets.QGroupBox)}
    boxes["Local markers"].setChecked(False)
    assert not win.schedule[2].local_align and win.modified
    qtbot.waitUntil(lambda: isinstance(tab.editor.op, Exposure) and not tab.editor.op.local_align)
    win.schedule[2].local_align = True
    tab.editor.set_op(win.schedule[2])
    n = tab.editor.set_local_per_field(-45.0, 45.0)
    assert n == 9 and len(win.schedule[2].local_markers) == 9
    win.view.stepClicked.emit(2)
    assert tab.selected() == 2

    # stage positions persist with Z; Go to XYZ / XY; chip origin from a position
    win.tab_positions.add("Faraday cup", 100.0, 200.0, 12.0)
    assert LithoSettings.load(win.settings.path).positions()["Faraday cup"] == (100.0, 200.0, 12.0)
    win.tab_positions.table.selectRow(1)
    win.tab_positions.b_go.click()
    qtbot.waitUntil(lambda: win.hw.calls and win.hw.calls[-1] == ("move", 100.0, 200.0, 12.0))
    win.tab_positions.b_go_xy.click()
    qtbot.waitUntil(lambda: win.hw.calls[-1] == ("move", 100.0, 200.0))

    # save / reopen
    p = str(tmp_path / "edited.ebls.json")
    assert win.save_to(p) and not win.modified
    before = win.schedule.to_dict(str(tmp_path))
    assert win.open_schedule(p) and win.schedule.to_dict(str(tmp_path)) == before

    # run the demo: Field calibration (Accept), then the Exposure with Global Align + Local markers
    assert win.open_schedule(demo["schedule"])
    win.hw.stage = win.hw.readback = np.array(ORIGIN)
    assert win.run(confirm=False)
    _calibration_waits(win, qtbot).b_accept.click()
    assert win.wait(30)
    assert tab.states == ["done"] * 3, win.log_box.toPlainText()
    assert len(win.hw.exposures) == 9 and "9 of 9 field(s) exposed" in tab.list.item(2).toolTip()
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
    ws = win.validate()
    assert any("outside the field" in w for i, w in ws if i == 1)  # 80 µm fields in a 50 µm field
    assert "⚠" in win.tab_schedule.list.item(1).text()
    tab = win.tab_schedule
    tab.select(0)
    combo = tab.editor.findChildren(QtWidgets.QComboBox)[0]
    assert combo.itemText(0) == "Keep as is" and combo.itemData(1) == "default"
    combo.setCurrentIndex(1)
    assert win.schedule[0].preset == "default"
    tab.select(1)
    tab.add_menu.actions()[1].trigger()  # Column setting
    tab.add_menu.actions()[2].trigger()  # Field calibration
    tab.add_menu.actions()[0].trigger()  # Exposure on the selected chip
    assert [o.TYPE for o in win.schedule] == ["column", "exposure", "column", "field_correction", "exposure",
                                              "field_correction"]
    assert win.schedule[4].chip == "A" and win.schedule[4].alignment_only
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
    op = win.add_exposure_from_gds(win.schedule.exposures[0].path)
    assert op is not None and op.chip == "B" and "9 field(s)" in win.log_box.toPlainText()


def test_menu_bar(win):
    menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert menus == ["Schedule", "Step", "Calibration", "View", "Stage"]
    cal = [a.text() for a in win.menuBar().actions()[2].menu().actions() if a.text()]
    assert cal == ["Beam calibration settings…", "Run beam calibration now", "Forget the beam calibration of this "
                   "condition", "Field calibration settings…", "Run field calibration now",
                   "Forget the field calibration"]
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
    assert [x.text() for x in win._last_menu.actions() if x.text()][0] == "Move the chip…"
    win._last_menu.close()


def test_beam_then_field_calibration_from_the_menu(win, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import BeamCalDialog

    assert win.l_beam.text().startswith("Beam calibration: none for ×1000, WD 15.00 mm")
    d = BeamCalDialog(win.settings.beamcal, win.settings.positions())
    qtbot.addWidget(d)
    assert d.pos.currentData() == "grid" and "Z 11.0" in d.pos.currentText() and d.value().pixels == 800

    monkeypatch.setattr(BeamCalDialog, "exec", lambda dlg: QtWidgets.QDialog.Accepted)
    win.beamcal_dialog()
    assert LithoSettings.load(win.settings.path).beamcal.position == "grid"
    win.hw.stage_noise = 0.0
    assert win.run_beam_calibration()
    bw = _waits(qtbot, lambda: win.beamcal_window)
    assert bw.isVisible() and "X scale" in bw.numbers.text() and "+2" in bw.numbers.text()
    assert win.hw.z == 11.0 and win.beamcal() is None  # not before Accept
    bw.b_accept.click()
    assert win.wait(30)
    bc = win.beamcal()
    assert bc is not None and bc.sx == pytest.approx(1.002, abs=3e-5) and bc.sy == pytest.approx(0.999, abs=3e-5)
    assert len(LithoSettings.load(win.settings.path).beamcals) == 1  # saved
    win._update_status()
    assert win.l_beam.text().startswith("Beam calibration: sx")
    # the field calibration goes through it and is valid with it
    assert win.run_field_calibration()
    cw = _calibration_waits(win, qtbot)
    assert cw.banner.isHidden()
    cw.b_accept.click()
    assert win.wait(30) and win.calibration.beamcal_id == bc.id
    win._update_status()
    assert win.l_cal.text().startswith("Field calibration: rot")
    win.hw.wd = 15.3  # another WD: neither applies
    win._update_status()
    assert "none for" in win.l_beam.text() and "not ×1000, WD 15.30 mm" in win.l_cal.text()
    win.hw.wd = 15.0
    win.forget_beamcal()
    win._update_status()
    assert win.beamcal() is None and "another beam calibration" in win.l_cal.text()
    # Discard, and Abort while it waits
    assert win.run_beam_calibration()
    _waits(qtbot, lambda: win.beamcal_window).b_discard.click()
    assert win.wait(30) and win.beamcal() is None and "discarded" in win.log_box.toPlainText()
    assert win.run_field_calibration()
    _calibration_waits(win, qtbot)
    win.abort()
    assert win.wait(30) and not win.fieldcal_window.awaiting


def test_field_calibration_mirror_mismatch_points_to_the_hardware_settings(win, qtbot):
    win.hw.B = np.eye(2)  # this instrument's stage X runs along the image's x, the settings say mirrored
    win.hw.stage_noise = 0.0
    assert win.run_field_calibration()
    cw = _calibration_waits(win, qtbot)
    assert not cw.banner.isHidden() and "Hardware → Configure" in cw.banner_text.text() and cw.b_fix.isHidden()
    cw.b_discard.click()
    assert win.wait(30) and win.calibration is None


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


def test_litho_runs_on_the_simulated_backend(ui, qtbot, demo):
    """Beam preset, magnification, stage moves and a real (simulated-DAQ) exposure through BackendHardware."""
    from easyebl.microscope.sem import PRESET_KEYS

    mic = ui.backend.microscope
    mic.settings.beam_presets.append(dict({k: 10 for k in PRESET_KEYS}, name="test"))
    w = ui.open("litho")
    w.ask_on_close = False
    gds = Schedule.load(demo["schedule"]).exposures[0].path
    w.set_schedule(Schedule([ColumnSetting(preset="test", magnification=900.0, current_pa=200.0),
                             Exposure(chip="A", path=gds, layers=[(1, 0)], dwell_us=0.5, step_nm=500.0)],
                            chips=[Chip("A", (300.0, 400.0))]))
    w.schedule[1].top = ""
    assert w.run(confirm=False)
    qtbot.waitUntil(lambda: w.runner is None, timeout=180000)
    assert w.tab_schedule.states == ["done", "done"], w.log_box.toPlainText()
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


def test_field_calibration_in_the_program(ui, qtbot):
    """BackendHardware on the simulators: real stage moves, SEM captures and the simulated image following
    the stage – the calibration finds the simulators' hidden map (no beam calibration: as measured)."""
    from easyebl.app.backend import SIM_STAGE_TO_BEAM

    mic = ui.backend.microscope
    w = ui.open("litho")
    w.ask_on_close = False
    w.settings.fieldcal = FieldCalConfig(x=300.0, y=300.0, settle_s=0.1, image_px=256, pattern="cross")
    assert w.run_field_calibration()
    cw = _calibration_waits(w, qtbot, timeout=180000)
    assert cw.banner.isHidden(), cw.banner_text.text()
    cw.b_accept.click()
    qtbot.waitUntil(lambda: w.runner is None, timeout=30000)
    assert w.calibration is not None, w.log_box.toPlainText()
    assert np.allclose(w.calibration.stage_to_beam, np.asarray(SIM_STAGE_TO_BEAM), atol=5e-3)
    assert mic.stage.positions["X"] == pytest.approx(300.0, abs=0.5)
