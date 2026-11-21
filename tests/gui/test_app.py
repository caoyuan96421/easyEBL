"""The easyEBL program: menu window, subsystem windows, hardware dialog, single instance (offscreen Qt)."""
import uuid

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from easyebl.app.config import SystemConfig  # noqa: E402


def _flush(app):
    for _ in range(3):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.fixture
def ui(qtbot, tmp_path, monkeypatch):
    from easyebl.app.gui import EasyEBLApp

    qapp = QtWidgets.QApplication.instance()
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda *a: None)
    cfg = SystemConfig()
    cfg.path = str(tmp_path / "system.json")
    asked = []
    app = EasyEBLApp(cfg, eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01, use_tray=False,
                     confirm=lambda title, text: asked.append(text) or True)
    app.asked = asked
    app.start()
    yield app
    if not app.closing:
        app.shutdown(ask=False)
    qapp.setQuitOnLastWindowClosed(True)


def test_menu_shows_the_hardware(ui):
    assert ui.menu.isVisible()
    texts = {k: t.text() for k, (_dot, t) in ui.menu.rows.items()}
    assert texts == {"daq": "simulator", "eds": "DP5 mock device", "microscope": "S-3000 simulator"}
    assert not QtWidgets.QApplication.instance().quitOnLastWindowClosed()  # the backend outlives windows
    titles = [a.text() for a in ui.menu.m_open.actions()]
    assert titles[:3] == ["SEM imaging", "EDS", "Microscope (column && stage)"]
    assert all(a.isEnabled() for a in ui._open_actions)


def test_subsystem_windows_share_the_backend(ui, qtbot):
    app = QtWidgets.QApplication.instance()
    sem = ui.open("sem")
    eds = ui.open("eds")
    mic = ui.open("microscope")
    assert type(sem).__name__ == "SEMWindow" and sem.engine is ui.backend.engine
    assert eds.ctrl is ui.backend.eds and mic.link is ui.backend.microscope.link
    assert ui.open("sem") is sem  # one window per subsystem
    # SEM window's "Microscope…" goes to the app's microscope window
    act = [a for a in sem.findChildren(QtGui.QAction) if a.text() == "Microscope…"]
    act[0].trigger()
    assert ui.windows["microscope"] is mic
    # closing windows leaves the hardware connected; opening again makes a new window
    for w in (sem, eds, mic):
        w.close()
    _flush(app)
    assert ui.windows == {}
    assert ui.backend.manager.is_open and ui.backend.microscope.link.state == "connected"
    sem2 = ui.open("sem")
    assert sem2 is not None and sem2 is not sem
    sem2.scan.b_run.click()
    qtbot.waitUntil(lambda: sem2.view.frame() is not None, timeout=15000)
    sem2.close()
    _flush(app)


def test_tools_open_and_release_their_handles(ui):
    app = QtWidgets.QApplication.instance()
    mgr = ui.backend.manager
    for key in ("vector", "dose", "alignment", "raster", "hal"):
        w = ui.open(key)
        assert type(w).__name__ == "ToolWindow", key
    owners = {h.owner for h in mgr.handles() if not h.closed}
    assert {"hal-debug", "imaging", "alignment", "litho-vector"} <= owners
    ui.close_windows()
    _flush(app)
    left = {h.owner for h in mgr.handles() if not h.closed}
    assert not left & {"hal-debug", "imaging", "alignment", "litho-vector"}


def test_closing_the_menu_keeps_the_system_running(ui):
    ui.menu.close()
    assert not ui.closing and ui.backend.started and ui.backend.manager.is_open
    ui.show_menu()
    assert ui.menu.isVisible()


def test_tray_menu(ui):
    m = QtWidgets.QMenu()
    ui._fill_tray_menu(m)
    texts = [a.text() for a in m.actions() if a.text()]
    assert texts[0] == "Show easyEBL menu" and "SEM imaging" in texts and "EDS" in texts
    assert texts[-1] == "Shut down easyEBL…" and "Hardware…" in texts
    tools = [a.menu() for a in m.actions() if a.menu() is not None]
    assert tools and "GDS lithography" in [a.text() for a in tools[0].actions()]


def test_reconfigure_reconnects(ui, tmp_path):
    app = QtWidgets.QApplication.instance()
    ui.open("sem")
    new = SystemConfig()
    new.path = ui.config.path
    new.microscope.port = "none"
    new.eds.mca = "none"
    assert ui.restart_backend(new)
    _flush(app)
    assert ui.asked and ui.windows == {}  # asked first, windows closed
    st = ui.backend.status()
    assert st["microscope"].state == "off" and st["eds"].state == "off" and st["daq"].state == "sim"
    assert SystemConfig.load(new.path).microscope.port == "none"  # saved
    ok, why = ui.available("eds")
    assert not ok and "not used" in why
    eds_action = [a for a in ui._open_actions if a.data() == "eds"][0]
    ui.refresh_status()
    assert not eds_action.isEnabled()


def test_shutdown_disconnects_everything(ui):
    ui.open("microscope")
    link = ui.backend.microscope.link
    assert ui.shutdown()
    assert ui.closing and not ui.backend.started and link.state == "closed" and ui.windows == {}


def test_hardware_dialog_roundtrip(qtbot):
    from easyebl.app.gui import HardwareDialog

    c = SystemConfig.hardware(device="Dev3", port="COM9")
    c.daq.rate, c.daq.ai_terminal, c.microscope.baudrate = 2e6, "RSE", 19200
    d = HardwareDialog(c)
    qtbot.addWidget(d)
    out = d.config()
    assert out == c
    d.set_config(c.as_simulated())
    sim = d.config()
    assert (sim.daq.backend, sim.eds.mca, sim.microscope.port) == ("sim", "mock", "sim")
    d.mic_port.setEditText("COM12")
    assert d.port() == "COM12"


def test_single_instance(qtbot):
    """A second start (another process) hands its request to the running program."""
    import subprocess
    import sys

    from easyebl.app import _listen, _send_to_running

    name = "easyebl-test-" + uuid.uuid4().hex[:8]
    assert not _send_to_running(name, "show")  # nothing running yet
    got = []

    class Target(QtCore.QObject):
        def handle_command(self, cmd):
            got.append(cmd)

    t = Target()
    server = _listen(name, t)
    code = (f"from easyebl.app import _send_to_running; "
            f"raise SystemExit(0 if _send_to_running({name!r}, 'open sem,eds') else 3)")
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        qtbot.waitUntil(lambda: got == ["open sem,eds"], timeout=15000)
        qtbot.waitUntil(lambda: p.poll() is not None, timeout=10000)
        assert p.returncode == 0
    finally:
        if p.poll() is None:
            p.kill()
        server.close()
