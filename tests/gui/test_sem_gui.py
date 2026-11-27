"""SEM window smoke test on the simulator (offscreen Qt)."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6 import QtCore  # noqa: E402
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
    # field of view: the scan starts at the full field; the wheel zooms the swept range
    # (the widest scan whose pre-line overshoot still fits the AO range: at most the full field)
    full = eng.max_width_um()
    w_full = eng.config.width_um
    assert w_full <= full + 1e-9 and eng.zoom_factor >= 1.0
    w.view.zoomSteps.emit(3)  # three notches: ×2
    assert eng.config.width_um == pytest.approx(w_full / 2)
    w.fov.step(1, 0)
    assert eng.config.center_um[0] > 0
    w.view.centerRequested.emit(0.0, 0.0)
    assert eng.config.center_um == (0.0, 0.0)
    w.fov.py.setValue(10.0)
    assert eng.config.center_um[1] == pytest.approx(10.0)
    w.fov.rot.setValue(30.0)  # the rotated scan must stay inside the AO range: it may shrink
    assert eng.config.rotation_rad == pytest.approx(np.radians(30.0))
    w.view.zoomSteps.emit(-20)  # zooming out stops at the full field; the position recentres
    assert eng.config.width_um <= full + 1e-9
    w.fov.rot.setValue(0.0)
    w.fov._do(eng.full_field)
    assert w_full - 1e-9 <= eng.config.width_um <= full + 1e-9 and eng.config.center_um == (0.0, 0.0)
    w.fov.size.setValue(full / 4)
    assert eng.zoom_factor == pytest.approx(4.0)
    # the data bar: column magnification at the full field × zoom (no column: unknown); the inset
    w._update_status()
    assert w.view.info["field_um"] == eng.calibration.field_size_um and w.view.info["scan"][1] == eng.config.width_um
    from easyebl.imaging.sem.display import display_magnification, image_magnification
    assert display_magnification(image_magnification(1000, 100.0, 25.0)) == "×4k"
    assert display_magnification(image_magnification(None, 100.0, 25.0)) == "×—"
    # speed presets set the line timing (shown in the status bar, not adjustable)
    assert "pre-line 20 µs" in w.l_timing.text() and "flyback 40 µs" in w.l_timing.text()
    w.auto_contrast()
    lo, hi = w.levels()
    assert hi > lo
    w.image.s_con.setValue(800)
    assert w.levels()[1] - w.levels()[0] < hi - lo
    assert w.image.b_acb.text() == "Auto levels" and w.image.avg.value() == eng.options.averaging
    # the tools at the top: icon + text
    texts = [a.text() for a in w.tools.actions() if a.text()]
    assert texts[0] in ("Run", "Stop")  # Stop while scanning
    for t in ("Single", "Capture", "Full FoV", "Autofocus", "Auto stigmation", "Auto gun align", "Auto C/B",
              "Crosshair", "Scan line", "Field inset", "Advanced…", "Full screen"):
        assert t in texts, t
    assert all(not a.icon().isNull() for a in w.tools.actions() if a.text())
    assert w.tools.toolButtonStyle() == QtCore.Qt.ToolButtonTextUnderIcon
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


def test_external_scan_follows_scanning(qtbot, sem):
    """No manual switch: scanning claims the SEM's external scan, stopping gives it back."""
    w, eng = sem, sem.engine
    mgr = eng.handle.manager
    assert mgr.external_control is not True
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
    assert mgr.external_control is True and mgr.backend.external_control is True
    w._update_status()
    assert "EXT SCAN" in w.l_ext.text()
    eng.update(nx=160, ny=120)  # a restart of the stream keeps it
    qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
    assert mgr.external_control is True
    eng.stop()
    qtbot.waitUntil(lambda: mgr.external_control is False, timeout=5000)
    assert mgr.backend.external_control is False
    eng.single()
    qtbot.waitUntil(lambda: mgr.external_control is True, timeout=5000)
    qtbot.waitUntil(lambda: eng.state == EngineState.FROZEN and mgr.external_control is False, timeout=15000)
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: mgr.external_control is True, timeout=5000)
    w.close()  # closing the SEM window hands the scan back to the SEM
    qtbot.waitUntil(lambda: mgr.backend.external_control is False, timeout=5000)


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


def test_saturation_shown_and_input_mode_fix(qtbot):
    from easyebl.imaging.sem.window import SEMWindow

    cfg = DAQConfig.usb6363()
    cfg.ai_terminal = "DIFF"  # the faulty setup: DIFF with AI0− unconnected
    be = SimulatedBackend(cfg, SimParams(realtime=True, v_full_scale=5.0, ai_minus_floating=True))
    mgr = DAQManager(be).open()
    eng = ImagingEngine(mgr, FieldCalibration(100, 5.0), options=EngineOptions(rate=500e3))
    eng.update(nx=128, ny=96, width_um=60.0, dwell_s=4e-6)
    w = SEMWindow(eng, simulated=True)
    qtbot.addWidget(w)
    try:
        w.scan.b_run.click()
        qtbot.waitUntil(lambda: "VIDEO SATURATED" in w.l_buf.text(), timeout=10000)
        assert "saturated" in w.log.toPlainText() and "AI−" in w.log.toPlainText()
        assert "Hardware → Configure… → Wiring" in w.log.toPlainText()  # the input mode is hardware wiring
        w.open_advanced()
        d = w._adv
        assert not hasattr(d, "ai_mode") and not hasattr(d, "ai_range")  # no SEM-side copy of the setting
        d.close()
        eng.set_ai_terminal("RSE")  # what Apply & reconnect does with the Wiring tab's input mode
        qtbot.waitUntil(lambda: eng.state == EngineState.LIVE, timeout=10000)
        qtbot.waitUntil(lambda: "VIDEO SATURATED" not in w.l_buf.text(), timeout=10000)
    finally:
        w.close()
        mgr.close()


def test_measure_line_and_global_calibration(qtbot, sem):
    w, eng = sem, sem.engine
    fs = w.field_scale
    assert not fs.from_microscope and fs.magnification == 1000  # no column: the manual magnification
    assert fs.k_um == pytest.approx(100.0 * 1000)
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: w.view.frame() is not None and w.view.frame().complete, timeout=15000)
    eng.freeze()
    # left drag: a line across the image, its length in field µm
    v = w.view
    r = v.image_rect()
    a, b = QtCore.QPoint(int(r.left() + 0.2 * r.width()), int(r.center().y())), \
        QtCore.QPoint(int(r.left() + 0.7 * r.width()), int(r.center().y()))
    c0 = eng.config.center_um
    qtbot.mousePress(v, QtCore.Qt.LeftButton, pos=a)
    qtbot.mouseMove(v, b)
    qtbot.mouseRelease(v, QtCore.Qt.LeftButton, pos=b)
    length = v.measure_length_um()
    assert length == pytest.approx(0.5 * v.frame().config.width_um, rel=0.03)
    assert eng.config.center_um == c0  # measuring does not drag the beam
    assert w.l_meas.text().startswith("line")
    # the calibration tab: global value, magnification, field; calibrate with the line
    w.open_advanced()
    d = w._adv
    d._refresh_cal()
    assert d.mag.isEnabled() and "enter" in d.l_mag_src.text()
    assert d.b_cal.isEnabled() and d.l_field.text().startswith("100 µm")
    d.true_len.setValue(round(length * 1.25, 4))
    d.calibrate_from_line()
    assert fs.k_um == pytest.approx(125000.0, rel=1e-4)
    assert eng.calibration.field_size_um == pytest.approx(125.0, rel=1e-4)
    assert v.measure_length_um() == pytest.approx(length * 1.25, rel=1e-6)  # the line keeps its feature
    # the frozen image was taken at the old field: no second calibration from it
    d._refresh_cal()
    assert not d.b_cal.isEnabled() and "new image" in d.l_line.text()
    # manual magnification and the global value, through Apply
    d.mag.setValue(500)
    d.k_field.setValue(100000.0)
    d.apply()
    assert eng.calibration.field_size_um == pytest.approx(200.0)
    d.close()
    v.clear_measure()
    assert v.measure_line() is None and w.l_meas.text() == ""


def test_sem_mouse_shift_pan_zoom_digital_zoom_and_menu(qtbot, sem):
    from PySide6 import QtGui

    w, eng = sem, sem.engine
    w.scan.b_run.click()
    qtbot.waitUntil(lambda: w.view.frame() is not None and w.view.frame().complete, timeout=15000)
    eng.freeze()
    v = w.view
    r = v.image_rect()
    c = r.center().toPoint()

    def wheel(steps, mods=QtCore.Qt.NoModifier, pos=c):
        ev = QtGui.QWheelEvent(QtCore.QPointF(pos), QtCore.QPointF(v.mapToGlobal(pos)), QtCore.QPoint(0, 0),
                               QtCore.QPoint(0, 120 * steps), QtCore.Qt.NoButton, mods, QtCore.Qt.NoScrollPhase,
                               False)
        v.wheelEvent(ev)

    # plain wheel: digital zoom only (the scan is untouched); about the cursor
    w0, c0 = eng.config.width_um, eng.config.center_um
    corner = QtCore.QPoint(int(r.left() + 0.25 * r.width()), int(r.top() + 0.25 * r.height()))
    before = v._to_pixel(QtCore.QPointF(corner))
    wheel(3, pos=corner)
    assert v.digital_zoom == pytest.approx(2.0) and eng.config.width_um == w0 and eng.config.center_um == c0
    after = v._to_pixel(QtCore.QPointF(corner))
    assert after == pytest.approx(before, abs=0.6)  # the point under the cursor stays put
    assert v.image_rect().width() == pytest.approx(2 * r.width())
    for _ in range(40):
        wheel(3)
    assert v.digital_zoom == pytest.approx(v._max_zoom())
    v.grab()  # paints only the visible part
    v.reset_digital_zoom()
    # Shift + wheel: the scanned region zooms
    wheel(3, QtCore.Qt.ShiftModifier)
    assert eng.config.width_um == pytest.approx(w0 / 2) and v.digital_zoom == 1.0
    # Shift + left drag: beam shift; a plain drag measures instead
    r = v.image_rect()
    a, b = QtCore.QPoint(int(r.center().x()), int(r.center().y())), QtCore.QPoint(int(r.center().x() + 60),
                                                                                 int(r.center().y()))
    c1 = eng.config.center_um
    qtbot.mousePress(v, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, a)
    qtbot.mouseMove(v, b)
    qtbot.mouseRelease(v, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, b)
    assert eng.config.center_um[0] < c1[0] and v.measure_line() is None  # the image followed the mouse
    # right click: the menu
    got = []
    v.contextRequested.connect(lambda xy, pos: got.append(xy))
    qtbot.mousePress(v, QtCore.Qt.RightButton, pos=a)
    assert got and got[0] is not None
    acts = {x.text(): x for x in w._last_menu.actions() if x.text()}
    assert list(acts) == ["Center with beam", "Center with stage", "Clear measurement", "Reset to Full FoV"]
    assert not acts["Center with stage"].isEnabled()  # no microscope here
    w._last_menu.close()
    w._center_at(*got[0])
    assert eng.config.center_um == pytest.approx(got[0])
    wheel(3)
    w.reset_full_fov()
    assert v.digital_zoom == 1.0 and eng.config.center_um == (0.0, 0.0)
    # Center with stage: the stage moves the clicked point to the image centre (S-3000 mirror: X)
    assert w.stage_move_for(eng.config.center_um[0] + 5.0, 3.0) == pytest.approx((-5.0, 3.0))
    w.stage_mirror = (False, False)
    assert w.stage_move_for(5.0, 3.0) == pytest.approx((5.0, 3.0))


def test_custom_ultrahigh_resolution_scan_and_capture(qtbot, sem, monkeypatch):
    from PySide6 import QtWidgets

    from easyebl.imaging.sem.panels import CustomSizeDialog

    w, eng = sem, sem.engine

    def fake_exec(dlg):
        dlg.set_size(8000, 8000)
        assert "64 Mpixel" in dlg.info.text() and "256 MB" in dlg.info.text() and "frame" in dlg.info.text()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(CustomSizeDialog, "exec", fake_exec)
    res = w.scan.res
    i = res.findData(None)
    assert res.itemText(i) == "Custom…"
    res.setCurrentIndex(i)
    w.scan._res_chosen(i)
    assert (eng.config.nx, eng.config.ny) == (8000, 8000)
    assert res.currentText() == "8000 × 8000 (custom)"  # kept in the list
    w.scan._res_chosen(res.findData("640x480"))
    res.setCurrentIndex(res.findData("640x480"))
    w.scan._res_chosen(res.currentIndex())
    assert (eng.config.nx, eng.config.ny) == (640, 480)
    # the capture size too
    cres = w.capture.res
    j = cres.findData(None)
    cres.setCurrentIndex(j)
    w.capture._res_chosen(j)
    assert cres.currentData() == "8000x8000"
    got = []
    w.capture.captureRequested.connect(got.append)
    w.capture.request()
    assert got[0]["nx"] == 8000 and got[0]["ny"] == 8000
    monkeypatch.setattr(CustomSizeDialog, "exec", lambda dlg: QtWidgets.QDialog.Rejected)
    cres.setCurrentIndex(cres.findData(None))
    w.capture._res_chosen(cres.currentIndex())
    assert cres.currentData() == "8000x8000"  # cancelled: back to the previous size
