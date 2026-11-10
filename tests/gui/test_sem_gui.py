"""SEM window smoke test on the simulator (offscreen Qt)."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend  # noqa: E402
from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine  # noqa: E402


@pytest.fixture
def sem(qtbot, tmp_path):
    from easyebl.imaging.sem.window import SEMWindow

    be = SimulatedBackend(DAQConfig(max_ao_rate=2e6, max_ai_aggregate_rate=2e6), SimParams(realtime=True))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, FieldCalibration(100, 10), options=EngineOptions(rate=500e3))
    eng.update(nx=128, ny=96, dwell_s=2e-6)
    w = SEMWindow(eng, simulated=True)
    w.save_settings.folder = str(tmp_path)
    qtbot.addWidget(w)
    w.show()
    yield w
    w.close()
    assert not eng.thread_alive
    mgr.close()


def test_sem_window_live_controls_capture(qtbot, sem, tmp_path):
    w, eng = sem, sem.engine
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: w.view.frame() is not None and w.view.frame().complete, timeout=15000)
    assert eng.state == EngineState.LIVE and w.scan.b_run.text() == "Freeze"
    # speed / resolution / beam shift / contrast
    w.scan.step_speed(+1)
    assert eng.config.dwell_s == pytest.approx(5e-6)
    w.scan.res.setCurrentIndex(w.scan.res.findData("256x256"))
    w.scan._res_chosen(w.scan.res.currentIndex())
    assert (eng.config.nx, eng.config.ny) == (256, 256)
    w.shift.step(1, 0)
    assert eng.config.center_um[0] > 0
    w.view.centerRequested.emit(0.0, 0.0)
    assert eng.config.center_um == (0.0, 0.0)
    w.auto_contrast()
    lo, hi = w.levels()
    assert hi > lo
    w.image.s_con.setValue(800)
    assert w.levels()[1] - w.levels()[0] < hi - lo
    # console-style change from another thread is reflected in the panels
    import threading

    threading.Thread(target=lambda: eng.set_resolution(320, 240)).start()
    qtbot.waitUntil(lambda: w.scan.res.currentData() == "320x240", timeout=5000)
    # advanced dialog applies options
    w.open_advanced()
    w._adv.latency.setValue(150)
    w._adv.apply()
    assert eng.options.latency_s == pytest.approx(0.15) and "Applied" in w._adv.msg.text()
    w._adv.close()
    # capture → auto-saved files
    w.capture.res.setCurrentIndex(w.capture.res.findData("256x256"))
    w.capture.speed.setCurrentIndex(1)
    w.capture.request()
    qtbot.waitUntil(lambda: w._capture_future is None, timeout=30000)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert any(f.endswith(".tif") for f in files) and any(f.endswith("_databar.png") for f in files)
    assert any(f.endswith(".json") for f in files)
    qtbot.waitUntil(lambda: eng.state == EngineState.FROZEN, timeout=5000)
    assert w.scan.b_run.text() == "Run"
    # stubs answer politely
    w.column._call(lambda: w.mic.step_magnification(1))
    assert "not available" in w.log.toPlainText()


def test_sem_window_run_freeze_cycle(qtbot, sem):
    w, eng = sem, sem.engine
    for _ in range(3):
        w.scan.b_run.click()
        qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
        w.scan.b_run.click()  # freeze at end of frame
        w.scan.b_run.click()  # … and immediately
        qtbot.waitUntil(lambda: eng.state in (EngineState.FROZEN, EngineState.IDLE), timeout=10000)
    assert eng.handle.manager.beam_state.blank


def test_external_scan_toggle(qtbot, sem):
    w, eng = sem, sem.engine
    mgr = eng.handle.manager
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
    w.cb_ext.setChecked(True)
    assert mgr.external_control is True and mgr.backend.external_control is True
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)  # resumed
    w.cb_ext.setChecked(False)
    assert mgr.external_control is False and not w.cb_ext.isChecked()
    w.cb_ext.setChecked(True)
    w.close()  # closing the SEM window hands the scan back to the SEM
    assert mgr.backend.external_control is False


def test_sem_view_annotated_render():
    from easyebl.imaging.engine import Frame
    from easyebl.imaging.raster import RasterConfig
    from easyebl.imaging.sem.view import render_annotated

    cfg = RasterConfig(nx=200, ny=150, width_um=50)
    f = Frame({"video": np.random.rand(150, 200).astype(np.float32)}, 1, 0, True, 150, None, cfg,
              FieldCalibration(100, 10), 0.0)
    img = render_annotated(f, (0.0, 1.0))
    assert img.width() == 200 and img.height() > 150
    _ = time.time()


def test_sem_window_shows_disconnect_and_resumes(qtbot, sem):
    w, eng = sem, sem.engine
    be = eng.handle.manager.backend
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
    be.unplug()
    qtbot.waitUntil(lambda: eng.state == EngineState.DISCONNECTED, timeout=5000)
    qtbot.waitUntil(lambda: "DISCONNECTED" in w.l_buf.text() and w.scan.b_run.text() == "Stop", timeout=3000)
    assert "lost" in w.log.toPlainText()
    be.replug()
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=5000)
    qtbot.waitUntil(lambda: "DISCONNECTED" not in w.l_buf.text() and w.scan.b_run.text() == "Freeze", timeout=3000)
    assert "restored" in w.log.toPlainText()


def test_advanced_dialog_switches_phantom(qtbot, tmp_path):
    from easyebl.imaging.sem.window import SEMWindow

    be = SimulatedBackend(DAQConfig.usb6363(), SimParams(realtime=True, v_full_scale=5.0))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, FieldCalibration(100, 5.0), options=EngineOptions(rate=500e3))
    eng.update(nx=128, ny=96, dwell_s=4e-6)
    w = SEMWindow(eng, simulated=True)
    qtbot.addWidget(w)
    try:
        w.scan.b_run.click()
        qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
        w.open_advanced()
        d = w._adv
        assert not d.ph_on.isChecked() and d.ph_out.count() == 2
        d.ph_on.setChecked(True)
        d.ph_noise.setValue(0.05)
        d.apply()
        assert "Applied" in d.msg.text(), d.msg.text()
        assert eng.handle.phantom == {"output": "aux0", "physical": "ao2", "noise_v": 0.05}
        assert w.view.info.get("detector", "").startswith("PHANTOM")
        qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)  # resumed
        d.ph_on.setChecked(False)
        d.apply()
        assert eng.handle.phantom["output"] is None and "detector" not in w.view.info
        qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
        d.close()
    finally:
        w.close()
        mgr.close()
