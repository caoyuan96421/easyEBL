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
    sem = SEMControl(link, status_poll_s=0)
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
    w.apply_hv()
    qtbot.waitUntil(lambda: bundle.sem.state.get("hv_setpoint_kv") == 12.5, timeout=3000)
    # readbacks changed elsewhere (the SEM panel) appear without sending anything back
    n_sent = len(bundle.link.session.simulator.received)
    bundle.link.session.simulator.sem["WD"] = "22"
    bundle.sem.refresh(("WD",)).result(5)
    qtbot.waitUntil(lambda: w.wd.value() == 22, timeout=2000)
    time.sleep(0.3)
    assert len(bundle.link.session.simulator.received) == n_sent + 1  # only the R WD read
    # stage: the SEM PC has it (#STAGE OFF) until enabled
    sim = bundle.link.session.simulator
    assert sim.sem["STAGE"] == "OFF" and bundle.stage.hand_control and not w.g_jog.isEnabled()
    x0 = bundle.stage.positions["X"]
    assert w.dro["X"].text() == f"{x0:,.1f}"
    assert w.move({"X": 100}, relative=True) is None and w.open_move_dialog("X") is None
    w.b_enable.click()
    qtbot.waitUntil(lambda: sim.sem["STAGE"] == "ON" and not bundle.stage.hand_control, timeout=5000)
    w.jog("X", +1)  # default jog step 100 µm
    qtbot.waitUntil(lambda: abs(bundle.stage.positions["X"] - (x0 + w.jog_um.value())) < 0.05, timeout=10000)
    # double-click on the Y readout: move to a position
    qtbot.mouseDClick(w.dro["Y"], QtCore.Qt.LeftButton)
    d = w.findChild(MoveDialog)
    assert d is not None and not d.relative() and d.value() == pytest.approx(bundle.stage.positions["Y"], abs=0.1)
    d.val.setValue(30000.0)
    d.accept()
    qtbot.waitUntil(lambda: abs(bundle.stage.positions["Y"] - 30000.0) < 0.05, timeout=15000)
    qtbot.waitUntil(lambda: w.dro["Y"].text() == "30,000.0", timeout=2000)
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
    # disable: back to the SEM PC (#STAGE OFF, the trackball works); a running move is stopped
    d = w.open_move_dialog("X")
    d.rb_rel.setChecked(True)
    d.val.setValue(20000.0)
    d.accept()
    qtbot.waitUntil(lambda: bundle.stage.moving, timeout=5000)
    w.b_enable.click()
    qtbot.waitUntil(lambda: sim.sem["STAGE"] == "OFF" and not bundle.stage.moving, timeout=8000)
    assert bundle.stage.hand_control and not w.g_jog.isEnabled() and "SEM PC" in w.b_enable.text()
    # the poll interval cannot go below 20 s
    w.poll_s.setValue(5)
    assert w.poll_s.value() == 20 and bundle.stage.poll_interval_s == 20
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
    w = SEMWindow(eng, simulated=True, microscope_bundle=bundle)
    bundle.follow_magnification([eng], sim_backend=be)
    w.save_settings.folder = str(tmp_path)
    qtbot.addWidget(w)
    w.show()
    try:
        # data bar and column panel show the microscope readbacks
        qtbot.waitUntil(lambda: w.view.info.get("kv") == "20 kV" and w.view.info.get("wd") == "WD 15 mm",
                        timeout=3000)
        assert "×1k" in w.column.l_mag.text() or "×1000" in w.column.l_mag.text()
        # the mouse wheel on the image changes the SEM magnification; the field follows
        w.view.wheelSteps.emit(1)
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
        # the microscope window opens from the toolbar
        act = [a for a in w.findChildren(type(w.menuBar().addAction("x"))) if a.text() == "Microscope…"]
        assert act
        act[0].trigger()
        assert bundle.window().isVisible()
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
