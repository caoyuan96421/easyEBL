"""Lithography window (offscreen Qt): stage view, tabs, step editor, schedule file, simulated run, SEM mark picking."""
import math
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("gdstk")
pytestmark = pytest.mark.gui

from PySide6 import QtWidgets  # noqa: E402

from easyebl.app.config import SystemConfig  # noqa: E402
from easyebl.litho.runner import MarkRequest, SimulatedHardware  # noqa: E402
from easyebl.litho.schedule import ColumnSetting, ExposeField, FieldCorrection, PatternSpec, Schedule  # noqa: E402
from easyebl.litho.settings import LithoSettings  # noqa: E402
from easyebl.litho.fieldcal import FieldCalConfig  # noqa: E402


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

    hw = SimulatedHardware(_rot(0.3, 1.001), stage_noise_um=0.3, pivot=(5000.0, 5000.0), start=(5000.0, 5000.0))
    settings = LithoSettings.load(str(tmp_path / "litho.json"))
    settings.fieldcal = FieldCalConfig(x=4700.0, y=5000.0, settle_s=0.0)
    w = LithoWindow(hw, settings)
    w.ask_on_close = False
    qtbot.addWidget(w)
    w.show()
    return w


def test_window_open_edit_save_run(win, tmp_path, demo):
    tab = win.tab_schedule
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Stage Positions", "Exposure Schedule"]
    assert win.l_global.text() == "Global alignment: none (identity)"
    assert isinstance(win.schedule[0], ColumnSetting) and tab.list.count() == 1
    assert not tab.b_del.isEnabled()  # the first Column setting stays

    assert win.open_schedule(demo["schedule"])
    assert tab.list.count() == 11 and not win.warnings()
    x, y = win.view._patterns.getData()
    assert np.nanmin(x) == pytest.approx(4960.0, abs=0.5) and np.nanmax(x) == pytest.approx(5240.0, abs=0.5)
    assert np.nanmax(y) == pytest.approx(5040.0, abs=0.5)  # the layout in SCS around the origin (5000, 5000)

    # the step editor
    tab.select(3)
    win.replan()  # (debounced after a selection)
    ed = tab.editor
    assert isinstance(ed.op, ExposeField) and len(win.view._wcs_grid.getData()[0])
    assert not [b for b in ed.findChildren(QtWidgets.QToolButton) if "stage" in b.text()]  # no stage button
    boxes = {b.title(): b for b in ed.findChildren(QtWidgets.QGroupBox)}
    boxes["Global Align (markers in WCS: µm from the field centre)"].setChecked(True)
    assert win.schedule[3].global_align and win.schedule[3].markers == [(-10.0, -10.0), (10.0, 10.0)]
    assert win.modified
    win.view.stepClicked.emit(5)
    assert tab.selected() == 5
    tab.select(0)
    tab.delete()
    assert isinstance(win.schedule[0], ColumnSetting) and len(win.schedule) == 11

    # stage positions persist; "New field here" adds an Expose field there
    win.tab_positions.add("Faraday cup", 100.0, 200.0)
    assert LithoSettings.load(win.settings.path).positions() == {"Faraday cup": (100.0, 200.0)}
    win.tab_positions.table.selectRow(0)
    tab.select(len(win.schedule) - 1)
    win.tab_positions.b_step.click()
    assert win.schedule[-1].centre == (100.0, 200.0)
    tab.delete()

    # save / reopen
    p = str(tmp_path / "edited.ebls.json")
    assert win.save_to(p) and not win.modified
    before = [o.to_dict() for o in win.schedule]
    assert win.open_schedule(p) and [o.to_dict() for o in win.schedule] == before

    # run the unedited demo on the simulated hardware
    assert win.open_schedule(demo["schedule"])
    assert win.run(confirm=False)
    assert win.wait(30)
    assert tab.states == ["done"] * 11, win.log_box.toPlainText()
    assert len(win.hw.exposures) == 9
    for op, e in zip(win.schedule.fields, win.hw.exposures):
        assert win.hw.placement_error(e, op.centre) < 1e-6
    assert win.glob is not None and "Global alignment: rot +5.24 mrad" in win.l_global.text()
    assert tab.b_run.isEnabled() and not tab.b_abort.isEnabled()
    assert "schedule finished" in win.log_box.toPlainText()
    win.reset_global()
    assert win.glob is None


def test_window_warnings_and_editor_types(win):
    s = Schedule([ColumnSetting(magnification=1000),
                  ExposeField(centre=(500.0, 500.0), pattern=PatternSpec("dummy", size_um=150)),
                  FieldCorrection()])
    win.set_schedule(s)
    ws = win.validate()
    assert any("outside the field" in w for i, w in ws if i == 1)
    assert "⚠" in win.tab_schedule.list.item(1).text()
    tab = win.tab_schedule
    tab.select(0)
    combo = tab.editor.findChildren(QtWidgets.QComboBox)[0]
    assert combo.itemText(0) == "Keep as is" and combo.itemData(1) == "default"
    combo.setCurrentIndex(1)
    assert win.schedule[0].preset == "default"
    from easyebl.litho.schedule import MAGNIFICATIONS

    mag = tab.editor.findChildren(QtWidgets.QComboBox)[1]
    assert [mag.itemData(i) for i in range(mag.count())] == list(MAGNIFICATIONS)  # only what the SEM accepts
    mag.setCurrentIndex(mag.findData(900))
    assert win.schedule[0].magnification == 900 and type(win.schedule[0].magnification) is int
    tab.select(1)
    tab.add_menu.actions()[1].trigger()  # Column setting
    tab.add_menu.actions()[2].trigger()  # Field correction
    tab.add_menu.actions()[0].trigger()  # Expose field at the stage
    assert [o.TYPE for o in win.schedule] == ["column", "expose", "column", "field_correction", "expose",
                                              "field_correction"]
    assert win.schedule[4].centre == win.hw.stage_position()


def test_import_dialog(qtbot, demo):
    from easyebl.litho.module.dialogs import ImportDialog
    from easyebl.litho.schedule import PatternLibrary

    d = ImportDialog(demo["gds"], PatternLibrary(), presets=["p"], stage_position=lambda: (7000.0, 8000.0))
    qtbot.addWidget(d)
    assert len(d.fields) == 9 and "9 field(s)" in d.info.text()
    d._use_stage()
    d.glob.setChecked(True)
    d.layers.setText("1/0")
    s = d.schedule()
    assert len(s) == 10 and s[1].global_align and s[1].centre[0] == pytest.approx(7000.0)


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
    assert w.hw.field_um() == pytest.approx(ui.backend.engine.calibration.field_size_um)
    assert w.hw.field_at_1x_um() == pytest.approx(ui.backend.microscope.field_scale.k_um)

    req = MarkRequest("mark 1", (1003.0, 998.0), (3.0, -2.0), (3.0, -2.0), 10.0)
    out = {}
    threading.Thread(target=lambda: out.setdefault("xy", w.hw.measure_marks([req])), daemon=True).start()
    qtbot.waitUntil(lambda: "sem" in ui.windows and ui.windows["sem"].view.pick_mode, timeout=10000)
    sem = ui.windows["sem"]
    cfg = ui.backend.engine.config
    assert cfg.width_um == pytest.approx(10.0) and cfg.center_um == pytest.approx((3.0, -2.0))  # zoomed on the mark
    assert sem.view._pick_targets == [(3.0, -2.0)]
    sem.view.pointPicked.emit(3.4, -1.7)  # the user clicks the mark
    qtbot.waitUntil(lambda: "xy" in out, timeout=10000)
    assert out["xy"] == [(3.4, -1.7)] and not sem.view.pick_mode
    qtbot.waitUntil(lambda: not ui.backend.engine.is_scanning, timeout=10000)  # no scanning over the resist

    out.clear()  # Esc / cancel aborts the measurement
    threading.Thread(target=lambda: out.setdefault("xy", w.hw.measure_marks([req])), daemon=True).start()
    qtbot.waitUntil(lambda: sem.view.pick_mode, timeout=10000)
    sem.cancel_pick()
    qtbot.waitUntil(lambda: "xy" in out, timeout=10000)
    assert out["xy"] is None


def test_litho_runs_on_the_simulated_backend(ui, qtbot):
    """Beam preset, magnification, stage move and a real (simulated-DAQ) exposure through BackendHardware."""
    from easyebl.microscope.sem import PRESET_KEYS

    mic = ui.backend.microscope
    mic.settings.beam_presets.append(dict({k: 10 for k in PRESET_KEYS}, name="test"))
    w = ui.open("litho")
    w.ask_on_close = False
    w.set_schedule(Schedule([ColumnSetting(preset="test", magnification=900.0, current_pa=200.0),
                             ExposeField(centre=(300.0, 400.0), pattern=PatternSpec("dummy", size_um=6.0),
                                         dwell_us=0.5, step_nm=200.0)]))
    assert w.run(confirm=False)
    qtbot.waitUntil(lambda: w.runner is None, timeout=60000)  # the finished event has been handled
    assert w.tab_schedule.states == ["done", "done"], w.log_box.toPlainText()
    assert mic.sem.magnification() == 900  # a float magnification reaches the SEM as the int it needs
    assert w.hw.field_um() == pytest.approx(mic.field_scale.k_um / 900)
    assert w.hw.stage_position() == pytest.approx((300.0, 400.0), abs=1.0)


def test_menu_bar(win):
    menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert menus == ["Schedule", "Field", "Import", "View", "Stage"]
    field = [a.text() for a in win.menuBar().actions()[1].menu().actions() if a.text()]
    assert "Move all fields…" in field and "Add Expose field" in field
    assert not win.a_abort.isEnabled() and win.a_run.isEnabled()
    assert not win.findChildren(QtWidgets.QToolBar)


def test_move_all_fields(win, demo, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import MoveFieldsDialog

    assert win.open_schedule(demo["schedule"])
    a = win.anchors()
    assert a["first"][1] == (5000.0, 5000.0)
    assert a["bbox"][1] == pytest.approx((5100.0, 4900.0))  # 3 × 3 fields of 100 µm, first at the top left
    assert a["marker"][1] == pytest.approx((4955.0, 4955.0))  # marker 1 = WCS (−45, −45) of field 1
    win.hw.stage = win.hw.readback = np.array([7000.0, 6000.0])

    d = MoveFieldsDialog(a, win.hw.stage_position(), click=(100.0, 200.0))
    qtbot.addWidget(d)
    assert d.from_buttons["first"].isChecked() and d.to_buttons["click"].isChecked()
    assert d.value() == (-4900.0, -4800.0)
    d.from_buttons["marker"].setChecked(True)
    d.to_buttons["stage"].setChecked(True)
    assert d.value() == pytest.approx((2045.0, 1045.0))
    d.to_buttons["abs"].setChecked(True)
    d.ax.setValue(10000.0)
    d.ay.setValue(10000.0)
    assert d.value() == pytest.approx((5045.0, 5045.0))
    assert not MoveFieldsDialog({"first": ("x", None)}).bb.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()

    # through the window: bbox centre → the stage
    def fake_exec(dlg):
        dlg.from_buttons["bbox"].setChecked(True)
        dlg.to_buttons["stage"].setChecked(True)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(MoveFieldsDialog, "exec", fake_exec)
    markers = [list(f.markers) for f in win.schedule.fields]
    assert win.move_fields_dialog(click=(1.0, 2.0))
    assert win.anchors()["bbox"][1] == pytest.approx((7000.0, 6000.0))
    assert win.schedule.fields[0].centre == pytest.approx((6900.0, 6100.0))
    assert [list(f.markers) for f in win.schedule.fields] == markers  # WCS markers travel with their field
    assert win.modified and "moved 9 field(s)" in win.log_box.toPlainText()
    win.view.contextRequested.emit(5.0, 6.0, win.mapToGlobal(win.rect().center()))
    acts = [x.text() for x in win._last_menu.actions() if x.text()]
    assert acts[0] == "Move…"
    win._last_menu.close()



def test_field_calibration_from_the_menu(win, qtbot, monkeypatch):
    from easyebl.litho.module.dialogs import FieldCalDialog

    menu = [a.text() for a in win.menuBar().actions()[1].menu().actions() if a.text()]
    assert {"Field calibration settings…", "Run field calibration now", "Forget the field calibration"} <= set(menu)
    assert win.l_cal.text() == "Field calibration: none"
    # the settings dialog round trip
    win.settings.add_position("Faraday cup", 4600.0, 5100.0)
    d = FieldCalDialog(win.settings.fieldcal, win.settings.positions(), win.hw.stage_position)
    qtbot.addWidget(d)
    d.pos.setCurrentIndex(d.pos.findData("Faraday cup"))
    d.pattern.setCurrentIndex(d.pattern.findData("cross"))
    v = d.value()
    assert v.position == "Faraday cup" and v.pattern == "cross" and not d.x.isEnabled()

    def fake_exec(dlg):
        dlg.pos.setCurrentIndex(dlg.pos.findData("Faraday cup"))
        dlg.settle.setValue(0.0)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(FieldCalDialog, "exec", fake_exec)
    win.fieldcal_dialog()
    assert LithoSettings.load(win.settings.path).fieldcal.position == "Faraday cup"
    # run it: the simulated hardware has the nominal map (B = I); its stage lands exactly here
    win.hw.stage_noise = 0.0
    assert win.run_field_calibration()
    assert win.wait(30)
    assert win.calibration is not None and np.allclose(win.calibration.stage_to_beam, np.eye(2), atol=3e-3)
    assert win.calibration.position == (4600.0, 5100.0)
    win._update_status()
    assert win.l_cal.text().startswith("Field calibration: rot") and "×1000" in win.l_cal.text()
    win.hw.mag = 2000.0  # another magnification: no longer valid
    win._update_status()
    assert "not ×2000" in win.l_cal.text()
    win.forget_calibration()
    assert win.calibration is None


def test_stage_settings_dialog(win, monkeypatch):
    from easyebl.litho.module.dialogs import StageAxesDialog

    def fake_exec(dlg):
        dlg.my.setChecked(True)
        dlg.tol.setValue(0.5)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(StageAxesDialog, "exec", fake_exec)
    win.edit_axes()
    s = LithoSettings.load(win.settings.path)
    assert s.axes.mirror_y and s.stage_tolerance_um == 0.5


def test_field_calibration_in_the_program(ui, qtbot):
    """BackendHardware on the simulators: real stage moves (one-way approach), SEM captures and the
    simulated image following the stage – the calibration finds the simulators' hidden map."""
    from easyebl.app.backend import SIM_STAGE_TO_BEAM

    mic = ui.backend.microscope
    w = ui.open("litho")
    w.ask_on_close = False
    w.settings.fieldcal = FieldCalConfig(x=300.0, y=300.0, settle_s=0.1, image_px=256, pattern="cross")
    assert w.run_field_calibration()
    qtbot.waitUntil(lambda: w.runner is None, timeout=180000)
    assert w.calibration is not None, w.log_box.toPlainText()
    assert np.allclose(w.calibration.stage_to_beam, np.asarray(SIM_STAGE_TO_BEAM), atol=5e-3), \
        (w.calibration.stage_to_beam, w.log_box.toPlainText())
    assert mic.stage.positions["X"] == pytest.approx(300.0, abs=0.1)
