"""Auto focus / stigmation in the SEM window, and the compact microscope window (simulators)."""
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6 import QtWidgets  # noqa: E402

from easyebl.app.backend import SystemBackend  # noqa: E402
from easyebl.app.config import SystemConfig  # noqa: E402


@pytest.fixture
def system(tmp_path):
    b = SystemBackend(SystemConfig(), eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    b.microscope.sem.set_magnification(2000).result(10)
    b.microscope.sem.refresh().result(30)
    yield b
    b.close()


def test_sem_window_auto_focus_button_shows_progress_and_result(qtbot, system):
    from easyebl.imaging.sem.window import SEMWindow

    w = SEMWindow(system.engine, owns_engine=False, simulated=True, microscope_bundle=system.microscope)
    qtbot.addWidget(w)
    w.show()
    col = w.column
    assert w.autotune is system.microscope.autotune  # the backend's, shared with lithography
    assert col.b_af.isEnabled() and col.b_as.isEnabled() and not col.b_tune_cancel.isVisible()
    # the searches themselves are verified on the instrument; here: the wiring of the progress events
    col.tuneEvent.emit({"type": "start", "text": "focus (auto): started"})
    assert col.b_tune_cancel.isVisible() and not col.b_af.isEnabled()
    col.tuneEvent.emit({"type": "progress", "text": "focus (auto): focus fine 2560 → 0.3"})
    assert "2560" in col.l_tune.text()
    col.tuneEvent.emit({"type": "done", "text": "focus (auto): done – sharper than before"})
    assert col.b_af.isEnabled() and not col.b_tune_cancel.isVisible()
    assert "focus (auto): done" in w.log.toPlainText()
    w.close()


def test_sem_window_without_remote_control_disables_auto_buttons(qtbot):
    from easyebl.core import FieldCalibration
    from easyebl.hal import DAQConfig, DAQManager, SimulatedBackend
    from easyebl.imaging.engine import ImagingEngine
    from easyebl.imaging.sem.window import SEMWindow

    mgr = DAQManager(SimulatedBackend(DAQConfig())).open()
    eng = ImagingEngine(mgr, FieldCalibration(100, 10))
    w = SEMWindow(eng, simulated=True)  # a stub microscope: no remote control
    qtbot.addWidget(w)
    assert w.autotune is None and not w.column.b_af.isEnabled() and not w.column.b_fs.isEnabled()
    assert w.auto_focus() is None
    w.close()


def test_microscope_window_has_no_sem_auto_focus_or_stigma_buttons(qtbot, system):
    from easyebl.microscope.gui import MicroscopeWindow

    b = system.microscope
    w = MicroscopeWindow(b.link, b.sem, b.stage, b.settings, owns_link=False)
    qtbot.addWidget(w)
    texts = [x.text() for x in w.findChildren(QtWidgets.QAbstractButton)]
    assert not any("Auto focus" in t or "Auto stigma" in t for t in texts)
    assert "Auto gun alignment" in texts
    assert w.b_stop.minimumHeight() <= 32 and w.dro["X"].minimumWidth() <= 160  # compact stage panel
    w.close()


def test_advanced_settings_set_the_auto_focus_and_stigma_test_images(qtbot, system):
    from easyebl.imaging.sem.window import SEMWindow

    w = SEMWindow(system.engine, owns_engine=False, simulated=True, microscope_bundle=system.microscope)
    qtbot.addWidget(w)
    w.open_advanced()
    adv = w._adv
    sp_f, px_f, info_f = adv.tune["focus"]
    sp_s, px_s, _ = adv.tune["stig"]
    assert sp_f.currentData() == "Slow 1" and px_f.value() == 256 and sp_s.currentData() == "Slow 1"
    assert "s per image" in info_f.text()
    sp_f.setCurrentIndex(sp_f.findData("Fast 2"))
    px_s.setValue(384)
    adv.apply()
    o = w.autotune.options
    assert (o.focus_speed, o.focus_px, o.stig_speed, o.stig_px) == ("Fast 2", 256, "Slow 1", 384)
    adv.close()
    w.close()
