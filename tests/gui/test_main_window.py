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
    pages = [mw.bottom.tabText(i) for i in range(mw.bottom.count())]
    assert pages == ["Messages", "Stage positions", "Calibration", "Alignment", "SEM view", "EDS data",
                     "EDS live spectrum", "Microscope log"]
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
    shows in the small SEM view of the bottom bar too, and a click there answers it."""
    mw = ui.main
    mw.show_tab("litho")
    sem = ui._sem_for_litho()
    assert sem is mw.sem and mw.bottom.currentWidget() is mw.sem_mini
    got = []
    sem.request_pick("Global marker 1", got.append, targets=[(3.0, -4.0)], centre=False)
    mw.sem_mini.refresh()
    assert mw.sem_mini.view.pick_mode and mw.sem_mini.b_cancel.isVisible()
    mw.sem_mini.view.pointPicked.emit(3.5, -4.25)
    assert got == [(3.5, -4.25)]
    mw.sem_mini.refresh()
    assert not mw.sem_mini.view.pick_mode


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
    assert lw.tab_positions.parent() is not None and mw.bottom.indexOf(lw.tab_positions) >= 0


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
    assert mw.tabs.count() == 3 and mw.left_tabs.count() == 2 and mw.bottom.count() == 8
    assert sum(1 for a in mw.menuBar().actions() if a.text() == "&Schedule") == 1


def test_closing_hides_and_shutdown_ends(ui):
    mw = ui.main
    mw.close()
    assert not mw.isVisible() or mw.isMinimized()
    ui.show_menu()
    assert mw.isVisible()
    assert ui.shutdown(ask=False) and ui.closing
