"""The easyEBL main window: tabs, side bars, bottom bar, menus / shortcuts per tab, Esc, fonts (offscreen Qt)."""
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from easyebl.app.config import SystemConfig  # noqa: E402


@pytest.fixture
def ui(qtbot, tmp_path, monkeypatch):
    from easyebl.app.gui import EasyEBLApp

    qapp = QtWidgets.QApplication.instance()
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda *a: None)
    cfg = SystemConfig()
    cfg.path = str(tmp_path / "system.json")
    app = EasyEBLApp(cfg, eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01, use_tray=False,
                     confirm=lambda title, text: True, main_window=True, layout_path=str(tmp_path / "main.ini"))
    app.start()
    yield app
    if not app.closing:
        lw = app.main.litho if app.main is not None else None
        if lw is not None:
            lw.modified = False  # no "discard changes?" question at the end
        app.shutdown(ask=False)
    qapp.setQuitOnLastWindowClosed(True)


def _keys(action):
    return [k.toString() for k in action.shortcuts()]


def test_the_main_window_holds_every_subsystem(ui):
    mw = ui.main
    assert mw.isVisible() and not ui.menu.isVisible()  # the main window replaces the menu window
    assert [mw.tabs.tabText(i) for i in range(mw.tabs.count())] == ["SEM", "EDS", "Lithography"]
    assert mw.sem is not None and mw.eds is not None and mw.litho is not None and mw.mic is not None
    assert [mw.left_tabs.tabText(i) for i in range(mw.left_tabs.count())] == ["Column", "Stage"]
    assert mw.monitor is not None
    # bottom: messages | calibration, alignment, small SEM view | EDS live spectrum
    assert [mw.bottom_split.widget(i) for i in range(3)] == [mw.messages_box, mw.bottom, mw.spectrum_box]
    assert [mw.bottom.tabText(i) for i in range(mw.bottom.count())] == ["Calibration", "Alignment", "SEM view"]
    assert mw.live_spectrum is not None and mw.live_spectrum.parent() is mw.spectrum_box
    # left: Column / Stage, the stage positions, the beam monitor; the Stage tab's own copy hidden
    assert [mw.left_split.widget(i) for i in range(3)] == [mw.left_tabs, mw.positions_box, mw.monitor_box]
    assert mw.litho.tab_positions.parent() is mw.positions_box and mw.litho.tab_positions.isVisibleTo(mw)
    stage_tab = mw.left_tabs.widget(1)
    assert not any(g.isVisibleTo(stage_tab) for g in stage_tab.findChildren(QtWidgets.QGroupBox)
                   if g.title().startswith("Stage positions"))
    # the EDS Data Manager under the EDS settings (right)
    assert mw.eds.data_tabs.parent() is mw.eds_right
    # no toolbar of the main window: each tab has its own tools at its top
    assert not [tb for tb in mw.findChildren(QtWidgets.QToolBar) if tb.parent() is mw]
    assert not hasattr(mw, "b_stop")
    assert not mw.mic.isVisible()  # its panels are in the left side bar, the window itself is not shown
    # the subsystems share the backend
    assert mw.sem.engine is ui.backend.engine and mw.eds.ctrl is ui.backend.eds


def test_the_right_bar_menus_and_shortcuts_follow_the_tab(ui):
    mw = ui.main
    lw, ew = mw.litho, mw.eds
    visible = lambda: [a.text().replace("&", "") for a in mw.menuBar().actions() if a.isVisible()]  # noqa: E731
    mw.show_tab("litho")
    assert mw.right_stack.currentWidget() is mw.hosts["litho"].right
    assert "Schedule" in visible() and "File" not in visible()
    assert _keys(lw.a_save) == ["Ctrl+S"] and _keys(lw.a_run) == ["F5"]
    eds_save = next(a for m in mw.hosts["eds"].menus for a in m.actions() if a.text().startswith("Save selected"))
    assert eds_save.shortcuts() == []  # Ctrl+S is the lithography's while its tab is shown
    sem_f5 = [s for s in mw.hosts["sem"].shortcuts if s.key().toString() == "F5"]
    assert sem_f5 and not sem_f5[0].isEnabled()
    mw.show_tab("eds")
    assert "File" in visible() and "Schedule" not in visible()
    assert eds_save.shortcuts() and lw.a_save.shortcuts() == [] and lw.a_run.shortcuts() == []
    mw.show_tab("sem")
    assert sem_f5[0].isEnabled() and lw.a_run.shortcuts() == []
    assert mw.right_stack.currentWidget() is mw.hosts["sem"].right
    assert all(tb.isVisible() for tb in mw.hosts["sem"].toolbars)
    # the embedded windows' own Close is not offered
    texts = [a.text() for m in mw.hosts["eds"].menus for a in m.actions() if a.isVisible()]
    assert "Close" not in texts


def test_esc_stops_everything(ui, qtbot):
    mw = ui.main
    stops = []
    mw.mic.stage.stop = lambda *a, **k: stops.append("stage")
    sem = mw.sem
    sem.engine.run()
    qtbot.waitUntil(lambda: sem.engine.status().state.value != "idle", timeout=5000)
    picked = []
    sem.request_pick("a mark", picked.append, targets=[(1.0, 2.0)])
    # every Esc of the subsystems is off: the main window's is the only one
    for h in mw.hosts.values():
        assert all(s.key() != QtGui.QKeySequence("Esc") for s in h.shortcuts)
    assert all(not s.isEnabled() for s in mw.mic.findChildren(QtGui.QShortcut)
               if s.key() == QtGui.QKeySequence("Esc"))
    mw.esc.activated.emit()
    assert stops == ["stage"] and picked == [None]
    qtbot.waitUntil(lambda: sem.engine.status().state.value == "idle", timeout=5000)
    assert "STOP (Esc): stage, lithography, EDS, SEM scan" in mw.log.toPlainText()


def test_open_shows_tabs_and_separate_windows_stay_available(ui):
    mw = ui.main
    assert ui.open("eds") is mw.eds and mw.tabs.currentIndex() == 1
    assert ui.open("litho") is mw.litho and mw.tabs.currentIndex() == 2
    assert ui.open("microscope") is mw.mic
    sem2 = ui.open_window("sem")  # the SEM can also have a window of its own
    assert sem2 is not None and sem2 is not mw.sem and ui.windows["sem"] is sem2
    titles = [a.text() for a in mw.m_windows.actions()]
    assert "SEM imaging (separate window)" in titles and not any(t.startswith("EDS") for t in titles)
    sem2.close()
    ui.show_classic_menu()
    assert ui.menu.isVisible()


def test_marker_picks_in_the_small_sem_view(ui):
    """The lithography's SEM window is the SEM tab's; with the Lithography tab in front the pick
    shows in the small SEM view of the bottom bar too (brought forward, highlighted), and a click
    there answers it."""
    mw = ui.main
    mw.show_tab("litho")
    mw.bottom.setCurrentIndex(0)
    sem = ui._sem_for_litho()
    mini = mw.sem_mini
    assert sem is mw.sem and mw.bottom.currentWidget() is mini
    mw.bottom.setCurrentIndex(0)
    got = []
    sem.request_pick("Global marker 1", got.append, targets=[(3.0, -4.0)], centre=False)
    mini.refresh()
    assert mw.bottom.currentWidget() is mini and mini.highlighted  # forward when a pick starts
    assert mini.view.pick_mode and mini.b_cancel.isVisibleTo(mini)
    mini.view.pointPicked.emit(3.5, -4.25)
    assert got == [(3.5, -4.25)]
    mini.refresh()
    assert not mini.view.pick_mode and not mini.highlighted


def test_the_small_sem_view_tools_and_mouse(ui, qtbot):
    mw = ui.main
    mini, sem = mw.sem_mini, mw.sem
    assert [a.text() for a in mini.toolbar.actions()] == ["Zoom in", "Zoom out", "Run", "Single", "Auto C/B"]
    w0 = sem.engine.config.width_um
    mini.a_zoom_in.trigger()
    assert sem.engine.config.width_um < w0
    mini.a_zoom_out.trigger()
    assert sem.engine.config.width_um == pytest.approx(w0)
    # simple view: no measuring / digital zoom, the data bar with the magnification and scale bar only
    v = mini.view
    assert v.simple and not v.show_fov_inset
    mini.refresh()
    assert v.info.get("simple") and v.info.get("column_mag") == sem.view.info.get("column_mag")
    # the wheel zooms the scan without Shift; a drag shifts the beam; a click without a drag picks
    zooms, shifts, picks = [], [], []
    v.zoomSteps.connect(zooms.append)
    v.shiftDragged.connect(lambda dx, dy: shifts.append((dx, dy)))
    v.pointPicked.connect(lambda x, y: picks.append((x, y)))
    sem.engine.single()
    qtbot.waitUntil(lambda: sem.engine.latest() is not None, timeout=10000)
    v.resize(400, 320)
    v.set_frame(sem.engine.latest(), sem.levels())
    c = v.image_rect().center()
    ev = QtGui.QWheelEvent(c, v.mapToGlobal(c.toPoint()), QtCore.QPoint(), QtCore.QPoint(0, 120),
                           QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
    v.wheelEvent(ev)
    assert zooms == [1] and v.digital_zoom == 1.0

    def mouse(kind, pos, button=QtCore.Qt.LeftButton):
        typ = {"press": QtCore.QEvent.MouseButtonPress, "move": QtCore.QEvent.MouseMove,
               "release": QtCore.QEvent.MouseButtonRelease}[kind]
        held = QtCore.Qt.NoButton if kind == "release" else button
        e = QtGui.QMouseEvent(typ, pos, v.mapToGlobal(pos.toPoint()), button if kind != "move" else QtCore.Qt.NoButton,
                              held, QtCore.Qt.NoModifier)
        {"press": v.mousePressEvent, "move": v.mouseMoveEvent, "release": v.mouseReleaseEvent}[kind](e)

    mouse("press", c)
    mouse("move", c + QtCore.QPointF(30, 10))
    mouse("release", c + QtCore.QPointF(30, 10))
    assert shifts and v.measure_line() is None and picks == []
    v.start_pick("pick", [])
    mouse("press", c)
    mouse("release", c)
    assert len(picks) == 1 and v.measure_line() is None
    v.stop_pick()


def test_messages_of_every_subsystem_in_one_list(ui):
    mw = ui.main
    mw.sem.notify("hello from the SEM")
    mw.litho.log("error: something failed")
    mw.eds.notify("EDS says hi")
    mw.notify("the program")
    text = mw.log.toPlainText()
    for part in ("[SEM]", "hello from the SEM", "[Lithography]", "[EDS] ", "[easyEBL]"):
        assert part in text, part
    assert mw.log.kind("error: something failed") == "error" and mw.log.kind("WARNING: x") == "warning"
    assert "#ff6b6b" in mw.log.document().toHtml().lower()  # the error in red
    mw.mic.notify("column says hi")  # the microscope's messages – not its port log
    assert "[Microscope]" in mw.log.toPlainText()


def test_status_bar_lights(ui):
    mw = ui.main
    ui.refresh_status()
    assert set(mw.status_lights) == set(mw.status_labels)
    for light in mw.status_lights.values():
        assert light.property("state") in ("ok", "sim") and "border-radius" in light.styleSheet()


def test_one_hv_button(ui):
    mic = ui.main.mic
    assert not hasattr(mic, "b_hv_on") and not hasattr(mic, "b_hv_off")
    mic._hv_button(True)
    assert mic.b_hv.text() == "HV OFF"
    mic._hv_button(False)
    assert mic.b_hv.text() == "HV ON"
    asked = []
    mic._confirmed = lambda text, fn, what: asked.append(what)
    mic.sem.state["hv_on"] = True
    mic.toggle_hv()
    mic.sem.state["hv_on"] = False
    mic.toggle_hv()
    assert asked == ["HV OFF", "HV ON"]


def test_one_font_size_everywhere(ui):
    mw = ui.main
    pt = QtWidgets.QApplication.instance().font().pointSizeF()
    for w in mw.findChildren(QtWidgets.QWidget):
        assert "font-size" not in w.styleSheet(), w
        if w.testAttribute(QtCore.Qt.WA_SetFont):
            assert w.font().pointSizeF() == pt, (w, w.font().pointSizeF())


def test_calibration_and_alignment_pages(ui):
    mw = ui.main
    rows = mw.calibration_view.rows()
    assert [r[0][0].split()[0] for r in rows] == ["100", "250", "500"]
    assert rows[0][3][0] == "not recorded" and rows[0][5][0] == "none"
    lw = mw.litho
    assert mw.alignment_view.rows() == [] or all(len(r) == 4 for r in mw.alignment_view.rows())
    assert lw.tab_positions.parent() is mw.positions_box


def test_layout_is_saved_and_restored(ui, tmp_path):
    mw = ui.main
    mw.tabs.setCurrentIndex(2)
    mw.save_layout()
    assert (tmp_path / "main.ini").exists()
    mw.tabs.setCurrentIndex(0)
    assert mw.restore_layout() and mw.tabs.currentIndex() == 2


def test_reconnect_rebuilds_the_subsystems(ui):
    mw = ui.main
    old = mw.sem
    assert ui.restart_backend(ask=False)
    assert mw.sem is not None and mw.sem is not old and mw.sem.engine is ui.backend.engine
    assert mw.tabs.count() == 3 and mw.left_tabs.count() == 2 and mw.bottom.count() == 3
    assert mw.litho.tab_positions.parent() is mw.positions_box and mw.live_spectrum.parent() is mw.spectrum_box
    assert sum(1 for a in mw.menuBar().actions() if a.text() == "&Schedule") == 1


def test_closing_shuts_down_unless_in_the_background(ui, qtbot):
    """By default closing the main window shuts easyEBL down (it asks first: declined, the window
    stays); with background=True (--background) it hides and easyEBL keeps running in the tray."""
    mw = ui.main
    ui.background = True
    mw.close()
    assert (not mw.isVisible() or mw.isMinimized()) and not ui.closing
    ui.show_menu()
    assert mw.isVisible()
    mw.hide_to_tray()  # the easyEBL menu's Hide to tray: always just hides
    assert not ui.closing
    ui.show_menu()
    ui.background = False
    ui.confirm = lambda title, text: False  # "Shut down easyEBL?" – no
    mw.close()
    qtbot.wait(100)
    assert mw.isVisible() and not ui.closing
    ui.confirm = lambda title, text: True
    mw.close()
    qtbot.waitUntil(lambda: ui.closing, timeout=5000)


def test_auto_cb_is_the_sems_abc_with_the_scan_handed_back(ui, qtbot):
    """Auto C/B (SEM toolbar, small SEM view): a modal window; our scan stops – the SEM gets its scan
    back – the SEM's ABC runs (its reply ends it), the live scan resumes and Auto levels runs."""
    sem = ui.main.sem
    eng, mic = sem.engine, sem.mic
    calls = []
    real = mic.auto_brightness_contrast

    def abc():
        calls.append(eng.state.value)  # the scan when ABC is sent
        return real()

    mic.auto_brightness_contrast = abc
    eng.run()
    qtbot.waitUntil(lambda: eng.state.value == "live", timeout=10000)
    assert sem.a_acb.isEnabled() and ui.main.sem_mini.a_acb.text() == "Auto C/B"
    t = sem.auto_contrast_brightness()
    assert t is not None and sem.abc_running and not sem.abc_available
    dlg = sem.busy
    assert dlg is not None and dlg.isModal() and dlg.b_cancel is None and dlg.isVisible()
    assert sem.auto_contrast_brightness() is None  # one at a time
    qtbot.waitUntil(lambda: not sem.abc_running, timeout=30000)
    assert dlg.is_done and not dlg.isVisible() and sem.busy is None
    assert calls and calls[0] in ("idle", "frozen")  # not scanning while the SEM does ABC
    qtbot.waitUntil(lambda: eng.state.value == "live", timeout=10000)  # resumed
    qtbot.waitUntil(lambda: "auto contrast/brightness" in sem.log.toPlainText(), timeout=10000)  # Auto levels
    assert "Auto C/B: done" in ui.main.log.toPlainText()
    # Esc in its window: ABC ends, the scan stays stopped, no Auto levels
    sem.log.clear()
    sem.auto_contrast_brightness()
    qtbot.keyClick(sem.busy, QtCore.Qt.Key_Escape)
    assert sem.busy is not None and sem.busy.isVisible()  # the SEM's ABC cannot be interrupted
    qtbot.waitUntil(lambda: not sem.abc_running, timeout=30000)
    qtbot.wait(500)
    assert eng.state.value != "live" and "auto contrast/brightness" not in sem.log.toPlainText()


def test_auto_gun_alignment_and_auto_focus_wait_in_a_modal_window(ui, qtbot):
    sem = ui.main.sem
    eng, mic = sem.engine, sem.mic
    calls = []
    real = mic.alignment
    mic.alignment = lambda kind: (calls.append((kind, eng.state.value)), real(kind))[1]
    eng.run()
    qtbot.waitUntil(lambda: eng.state.value == "live", timeout=10000)
    t = ui.main.mic.b_aga  # the column's button: the same routine (no question)
    t.click()
    dlg = sem.busy
    assert dlg is not None and dlg.isModal() and dlg.b_cancel is None and sem.abc_running
    assert not sem.abc_available  # one SEM routine at a time (ABC waits)
    qtbot.waitUntil(lambda: dlg.is_done, timeout=30000)
    assert calls and calls[0][0] == "gun" and calls[0][1] in ("idle", "frozen")  # the SEM had its scan
    assert sem.busy is None and "Auto gun alignment: done" in sem.log.toPlainText()
    qtbot.waitUntil(lambda: eng.state.value == "live", timeout=10000)  # our scan again
    eng.stop()
    # our auto focus: progress, Cancel (the start values again)
    qtbot.waitUntil(lambda: sem.autotune is not None and sem.autotune.available, timeout=10000)
    fut = sem.auto_focus("fine")
    dlg = sem.busy
    assert fut is not None and dlg.b_cancel is not None and dlg.isModal()
    qtbot.keyClick(dlg, QtCore.Qt.Key_Escape)  # Esc = Cancel
    qtbot.waitUntil(lambda: fut.done(), timeout=60000)
    qtbot.waitUntil(lambda: dlg.is_done, timeout=5000)
    assert not fut.result().ok and sem.busy is None
