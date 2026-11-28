"""Microscope window and its wiring into SEM imaging and EDS (simulator, offscreen Qt)."""
import json
import os
import time

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend  # noqa: E402
from easyebl.imaging.engine import EngineOptions, ImagingEngine  # noqa: E402
from easyebl.microscope import MicroscopeLink, SEMControl, StageControl  # noqa: E402
from easyebl.microscope.bundle import MicroscopeBundle  # noqa: E402
from easyebl.microscope.settings import MicroscopeSettings  # noqa: E402
from PySide6 import QtCore  # noqa: E402
from easyebl.microscope.gui import MoveDialog  # noqa: E402


@pytest.fixture
def bundle(tmp_path):
    link = MicroscopeLink.simulated(command_delay_s=0.01)
    sem = SEMControl(link, polling=False)
    stage = StageControl(link, polling=False)
    sem.refresh().result(10)
    stage.refresh().result(10)
    b = MicroscopeBundle(link, sem, stage, MicroscopeSettings(path=str(tmp_path / "mic.json")))
    yield b
    b.close()


def test_microscope_window(qtbot, bundle):
    from easyebl.microscope.gui import MicroscopeWindow

    w = MicroscopeWindow(bundle.link, bundle.sem, bundle.stage, bundle.settings, owns_link=False)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitUntil(lambda: w.l_mag.text() == "×1k" and "20 kV" in w.l_hv.text(), timeout=3000)
    assert "CONNECTED" in w.l_link.text() and w.dro["X"].text() not in ("", "—")
    # SEM settings from the controls
    w.mag_combo.setCurrentIndex(w.mag_combo.findData(5000))
    w.mag_combo.activated.emit(w.mag_combo.currentIndex())
    w.cont.setValue(1234)
    qtbot.waitUntil(lambda: bundle.sem.magnification() == 5000 and bundle.sem.state["contrast"] == 1234, timeout=5000)
    qtbot.waitUntil(lambda: w.l_mag.text() == "×5k", timeout=2000)
    w.hv_set.setValue(12.5)
    w.hv_set.editingFinished.emit()  # Enter: HV is on, sent
    qtbot.waitUntil(lambda: bundle.sem.state.get("hv_setpoint_kv") == 12.5, timeout=3000)
    # readbacks changed elsewhere (the SEM panel) appear without sending anything back
    n_sent = len(bundle.link.session.simulator.received)
    bundle.link.session.simulator.sem["FOCUS C"] = "9262"  # focused at WD 22 on the SEM panel
    bundle.sem.refresh(("FOCUS C",)).result(5)
    qtbot.waitUntil(lambda: w.focus_c.value() == 9262 and abs(w.wd.value() - 22.0) < 0.01, timeout=2000)
    time.sleep(0.3)
    assert len(bundle.link.session.simulator.received) == n_sent + 1  # only the R FOCUS C read
    # stage: the SEM PC has it (#STAGE OFF) between moves; no enable button any more
    sim = bundle.link.session.simulator
    assert sim.sem["STAGE"] == "OFF" and not bundle.stage.busy and w.g_jog.isEnabled()
    assert not hasattr(w, "b_enable")
    x0 = bundle.stage.positions["X"]
    assert w.dro["X"].text() == f"{x0:,.1f}"
    w.jog("X", +1)  # default jog step 100 µm
    qtbot.waitUntil(lambda: abs(bundle.stage.positions["X"] - (x0 + w.jog_um.value())) <= 0.5, timeout=10000)
    qtbot.waitUntil(lambda: not bundle.stage.busy and w.motion_enabled, timeout=5000)  # S2, #STAGE OFF
    # double-click on the Y readout: move to a position
    qtbot.mouseDClick(w.dro["Y"], QtCore.Qt.LeftButton)
    d = w.findChild(MoveDialog)
    assert d is not None and not d.relative() and d.value() == pytest.approx(bundle.stage.positions["Y"], abs=0.1)
    d.val.setValue(30000.0)
    d.accept()
    qtbot.waitUntil(lambda: abs(bundle.stage.positions["Y"] - 30000.0) <= 0.5, timeout=15000)
    qtbot.waitUntil(lambda: w.dro["Y"].text() == f"{bundle.stage.positions['Y']:,.1f}", timeout=2000)
    # STOP ALL during a long (relative) move
    d = w.open_move_dialog("X")
    d.rb_rel.setChecked(True)
    assert d.value() == 0.0
    d.val.setValue(20000.0)
    d.accept()
    qtbot.waitUntil(lambda: bundle.stage.moving, timeout=5000)
    w.b_stop.click()
    qtbot.waitUntil(lambda: not bundle.stage.moving, timeout=5000)
    qtbot.waitUntil(lambda: "halted" in w.log.toPlainText() or "Halted" in w.log.toPlainText(), timeout=5000)
    qtbot.waitUntil(lambda: not bundle.stage.busy and sim.sem["STAGE"] == "OFF", timeout=5000)
    # during a move: controls greyed out, a second move refused; afterwards the SEM PC has the stage
    d = w.open_move_dialog("X")
    d.rb_rel.setChecked(True)
    d.val.setValue(3000.0)
    d.accept()
    qtbot.waitUntil(lambda: bundle.stage.moving, timeout=5000)
    qtbot.waitUntil(lambda: not w.g_jog.isEnabled(), timeout=2000)
    assert w.move({"X": 10}, relative=True) is None and w.open_move_dialog("X") is None
    qtbot.waitUntil(lambda: not bundle.stage.busy and sim.sem["STAGE"] == "OFF", timeout=20000)
    qtbot.waitUntil(lambda: w.g_jog.isEnabled(), timeout=2000)
    # stage auto-read on / off (stored)
    w.cb_poll.setChecked(True)
    assert bundle.stage.polling and bundle.settings.stage_polling is True
    w.cb_poll.setChecked(False)
    assert not bundle.stage.polling and bundle.settings.stage_polling is False
    assert "off" in w._poll_text()
    # the continuous SEM read can be switched off and on
    assert not w.sem_poll.isChecked()  # the fixture's SEM module does not poll
    w.sem_poll.setChecked(True)
    assert bundle.sem.polling and bundle.settings.sem_polling is True
    w.sem_poll.setChecked(False)
    assert not bundle.sem.polling and bundle.settings.sem_polling is False
    # raw port log
    w._raw_tabs.setCurrentIndex(1)
    assert "TX" in w.raw.toPlainText() and "MAG" in w.raw.toPlainText()
    w.close()


def test_sem_window_with_microscope(qtbot, bundle, tmp_path):
    from easyebl.imaging.sem.window import SEMWindow

    be = SimulatedBackend(DAQConfig(max_ao_rate=2e6, max_ai_aggregate_rate=2e6),
                          SimParams(realtime=True, field_size_um=100.0))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, FieldCalibration(100, 10), options=EngineOptions(rate=500e3))
    eng.update(nx=128, ny=96, dwell_s=2e-6)
    bundle.tie_field([eng], sim_backend=be)
    w = SEMWindow(eng, simulated=True, microscope_bundle=bundle)
    assert w.field_scale is bundle.field_scale
    w.save_settings.folder = str(tmp_path)
    qtbot.addWidget(w)
    w.show()
    try:
        # data bar and column panel show the microscope readbacks
        qtbot.waitUntil(lambda: w.view.info.get("kv") == "20 kV" and w.view.info.get("wd") == "WD 15.00 mm",
                        timeout=3000)
        assert "×1k" in w.column.l_mag.text() or "×1000" in w.column.l_mag.text()
        # the mouse wheel zooms the swept range only: the column magnification stays
        from easyebl.imaging.sem.display import image_magnification
        w._update_status()
        m0 = image_magnification(w.view.info["column_mag"], w.view.info["field_um"], eng.config.width_um)
        assert m0 == pytest.approx(1000 * 100.0 / eng.config.width_um)  # ×1k at the full field
        w0 = eng.config.width_um
        w.view.zoomSteps.emit(3)
        assert eng.config.width_um == pytest.approx(w0 / 2) and bundle.sem.magnification() == 1000
        w._update_status()
        m1 = image_magnification(w.view.info["column_mag"], w.view.info["field_um"], eng.config.width_um)
        assert m1 == pytest.approx(2 * m0)
        eng.full_field()
        # the column's own magnification (the column panel); the field follows
        w.column._call(lambda: bundle.sem.step_magnification(1))
        qtbot.waitUntil(lambda: bundle.sem.magnification() == 1200, timeout=5000)
        qtbot.waitUntil(lambda: abs(eng.calibration.field_size_um - 100.0 / 1.2) < 1e-6, timeout=3000)
        assert be.params.field_size_um == pytest.approx(100.0 / 1.2)  # the simulated SEM zooms too
        w.column._call(lambda: bundle.sem.focus(+1))
        qtbot.waitUntil(lambda: bundle.sem.state["focus_fine"] == 2048 + bundle.sem.focus_step, timeout=3000)
        # a capture saves the microscope state with the image
        w.scan.b_run.click()
        qtbot.waitUntil(lambda: w.view.frame() is not None, timeout=15000)
        w.save_frame(w.view.frame(), os.path.join(str(tmp_path), "cap"))
        meta = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))
        assert meta["microscope"]["mag"] == 1200 and meta["microscope"]["hv_kv"] == 20.0
        assert "stage_x" in meta["microscope"]
        # no separate microscope window from the SEM (the main window has the column in its side bar)
        assert not [a for a in w.findChildren(type(w.menuBar().addAction("x"))) if a.text() == "Microscope…"]
    finally:
        w.close()
        mgr.close()


def test_beam_preset_dialog(qtbot, bundle, monkeypatch):
    from PySide6 import QtWidgets

    from easyebl.microscope.gui import MicroscopeWindow
    from easyebl.microscope.settings import MicroscopeSettings

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    sim = bundle.link.session.simulator
    w = MicroscopeWindow(bundle.link, bundle.sem, bundle.stage, bundle.settings, owns_link=False)
    qtbot.addWidget(w)
    d = w.open_presets()
    assert w.open_presets() is d  # one dialog
    # save two presets from the SEM's settings
    sim.sem.update({"BEAM C": "20", "BEAM F": "30", "STG X": "2000", "STG Y": "2100", "CONT": "1000", "BRT": "1100"})
    d.save_current("imaging")
    qtbot.waitUntil(lambda: d.table.rowCount() == 1, timeout=5000)
    sim.sem.update({"BEAM C": "70", "BEAM F": "5", "STG X": "2200", "STG Y": "1900", "CONT": "3000", "BRT": "900"})
    d.save_current("EDS high current")
    qtbot.waitUntil(lambda: d.table.rowCount() == 2, timeout=5000)
    assert [p["name"] for p in bundle.settings.beam_presets] == ["imaging", "EDS high current"]
    # saved to the settings file at once
    on_disk = MicroscopeSettings.load(bundle.settings.path).beam_presets
    assert on_disk[1] == {"name": "EDS high current", "beam_coarse": 70, "beam_fine": 5, "stig_x": 2200,
                          "stig_y": 1900, "contrast": 3000, "brightness": 900}
    # rename and edit in the table; out-of-range values are refused
    d.table.item(0, 0).setText("SE imaging")
    d.table.item(0, 5).setText("1234")
    d.table.item(0, 1).setText("150")
    qtbot.wait(50)
    p0 = bundle.settings.beam_presets[0]
    assert p0["name"] == "SE imaging" and p0["contrast"] == 1234 and p0["beam_coarse"] == 20
    assert d.table.item(0, 1).text() == "20" and "0…99" in d.l_msg.text()
    d.table.item(1, 0).setText("SE imaging")  # duplicate name refused
    qtbot.wait(50)
    assert bundle.settings.beam_presets[1]["name"] == "EDS high current"
    # recall the first preset: all six settings reach the SEM
    d.table.selectRow(0)
    d.recall().result(10)
    qtbot.waitUntil(lambda: "recalled" in d.l_msg.text(), timeout=3000)
    assert (sim.sem["BEAM C"], sim.sem["BEAM F"], sim.sem["STG X"], sim.sem["STG Y"], sim.sem["CONT"],
            sim.sem["BRT"]) == ("20", "30", "2000", "2100", "1234", "1100")
    qtbot.waitUntil(lambda: w.beam_c.value() == 20 and w.cont.value() == 1234, timeout=2000)
    # reorder, overwrite from the SEM, delete
    d.table.selectRow(1)
    d.move_selected(-1)
    assert [p["name"] for p in bundle.settings.beam_presets] == ["EDS high current", "SE imaging"]
    sim.sem["BEAM C"] = "75"
    d.table.selectRow(0)
    d.update_selected()
    qtbot.waitUntil(lambda: bundle.settings.beam_presets[0]["beam_coarse"] == 75, timeout=5000)
    d.table.selectRow(1)
    d.delete_selected()
    assert [p["name"] for p in bundle.settings.beam_presets] == ["EDS high current"]
    d.close()
    w.close()


def test_stage_move_wait_dialog(qtbot, bundle):
    """Programmed moves: a modal dialog until the move ends; STOP from the dialog; a second move refused."""
    from easyebl.microscope import StageBusy
    from easyebl.microscope.movewait import StageMoveDialog, run_blocking_move

    stage = bundle.stage
    x0 = stage.positions["X"]
    pos = run_blocking_move(stage, {"X": 200.0}, relative=True)
    assert pos["X"] == pytest.approx(x0 + 200.0, abs=0.5) and not stage.busy
    assert stage.settling is None  # the move ended after the simulated X correction (settle_tol 0.5 µm)
    # STOP from the dialog of a move started elsewhere (e.g. a worker thread)
    fut = stage.move({"X": 20000.0}, relative=True)
    d = StageMoveDialog(stage, fut, "X by +20000 µm")
    qtbot.addWidget(d)
    d.open()
    with pytest.raises(StageBusy):
        stage.move({"Y": 10.0}, relative=True)
    qtbot.waitUntil(lambda: stage.moving, timeout=5000)
    d.b_stop.click()
    qtbot.waitUntil(lambda: fut.done(), timeout=15000)
    qtbot.waitUntil(lambda: not d.isVisible(), timeout=2000)
    assert d.stopped and d.result() == 0 and not stage.busy



def test_sem_scan_mode_is_read_back(qtbot, bundle):
    """R SCAN (answered by the S-3000 though not in its manual): the window shows the SEM's own scan mode;
    nothing is set on connect."""
    from easyebl.microscope.gui import MicroscopeWindow

    sim = bundle.link.session.simulator
    assert sim.sem["SCAN"] == "F" and bundle.sem.state.get("scan") == "F"  # read on start, not changed
    assert not any(ln.startswith("SCAN") for ln in sim.received)
    sim.sem["SCAN"] = "1"  # set on the SEM's own panel
    bundle.sem.refresh(["SCAN"]).result(10)
    w = MicroscopeWindow(bundle.link, bundle.sem, bundle.stage, bundle.settings, owns_link=False)
    qtbot.addWidget(w)
    qtbot.waitUntil(lambda: w.scan_combo.currentText() == "Slow scan 1", timeout=5000)
    w.scan_combo.setCurrentIndex(w.scan_combo.findData("2"))
    w.scan_combo.activated.emit(w.scan_combo.currentIndex())
    qtbot.waitUntil(lambda: sim.sem["SCAN"] == "2", timeout=5000)
    w.close()


def test_auto_gun_alignment_button(qtbot, bundle, monkeypatch):
    from PySide6 import QtWidgets

    from easyebl.microscope.gui import MicroscopeWindow

    sim = bundle.link.session.simulator
    w = MicroscopeWindow(bundle.link, bundle.sem, bundle.stage, bundle.settings, owns_link=False)
    qtbot.addWidget(w)
    w.confirm = True
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.No)
    w.b_aga.click()  # declined: nothing sent
    assert "AGA ON" not in sim.received
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    w.b_aga.click()
    qtbot.waitUntil(lambda: "AGA ON" in sim.received, timeout=5000)
    w.close()


def test_stage_positions_in_the_microscope_window(qtbot, bundle):
    """The Lithography window's Stage Positions, in the Microscope window: the same list (shared settings); ΔZ
    against the calibration sample, Record WD (WD1 of the field size, Z_cal), Go to XYZ: Z_cal + ΔZ and WD1."""
    from easyebl.litho.module.tabs import StagePositionsTab
    from easyebl.litho.settings import LithoSettings
    from easyebl.microscope.gui import MicroscopeWindow

    w = MicroscopeWindow(bundle.link, bundle.sem, bundle.stage, bundle.settings, owns_link=False)
    qtbot.addWidget(w)
    shared = LithoSettings.shared()
    pt = w.positions_tab
    assert pt.settings is shared
    litho = StagePositionsTab(shared)  # as in the Lithography window
    qtbot.addWidget(litho)
    shared.field_size = "250 µm"
    wd = bundle.sem.working_distance_mm()
    x, y, z = (bundle.stage.positions[k] for k in ("X", "Y", "Z"))
    assert pt.record_wd(confirm=False)  # focused on the calibration sample
    cal = shared.calibration
    assert cal.wd_mm == pytest.approx(wd, abs=1e-4) and cal.z_um == pytest.approx(z) and cal.wd1 == {"250 µm": cal.wd_mm}
    assert "WD1" in litho.l_cal.text() and "not recorded" not in litho.l_cal.text()
    pt.add_current("Cup")
    p = shared.positions()["Cup"]
    assert (p.x, p.y) == pytest.approx((x, y)) and p.dz == pytest.approx(0.0)
    assert litho.table.rowCount() == 3 and litho.table.item(2, 0).text() == "Cup"
    assert LithoSettings.load(shared.path).positions()["Cup"] == p  # saved
    moves, wds = [], []
    w.move = lambda targets, relative=False: moves.append((targets, relative)) or object()
    w._call = lambda fn, *a, what="": wds.append((fn.__name__, a))
    pt.table.selectRow(2)
    pt.b_go.click()
    assert moves == [({"X": p.x, "Y": p.y, "Z": pytest.approx(z)}, False)]
    assert wds == [("set_working_distance", (cal.wd_mm,))]
    pt.b_go_xy.click()
    assert moves[-1] == ({"X": p.x, "Y": p.y}, False) and len(wds) == 1
    litho.table.selectRow(2)
    litho.delete_selected()
    assert pt.table.rowCount() == 2
    # the field size selector: the magnification, Z to 1000 × the nominal WD, the focus to its WD1 (else nominal)
    assert [w.field_size.itemData(i) for i in range(w.field_size.count())] == ["", "100 µm", "250 µm", "500 µm"]
    assert w.field_size.currentData() == "250 µm"
    moves.clear()
    wds.clear()
    assert w.select_field_size("250 µm")
    assert ("set_magnification", (500,)) in wds and ("set_working_distance", (cal.wd_mm,)) in wds
    assert moves == [({"Z": 11000.0}, False)]
    moves.clear()
    wds.clear()
    assert w.select_field_size("500 µm")  # no WD1 recorded for it: the nominal WD
    assert ("set_working_distance", (24.0,)) in wds and moves == [({"Z": 24000.0}, False)]
    assert shared.field_size == "500 µm" and cal.wd_mm is None and LithoSettings.load(shared.path).field_size == "500 µm"
    assert not w.select_field_size("nope")
