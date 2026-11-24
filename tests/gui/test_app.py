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


def _pick(combo, data):
    i = combo.findData(data)
    assert i >= 0, data
    combo.setCurrentIndex(i)


def test_hardware_dialog_wiring_tab(ui, qtbot):
    """Wiring tab: USB-6363 dropdowns, conflicts block Apply, the wiring is saved and used by the (simulated) DAQ."""
    from easyebl.app.gui import HardwareDialog

    d = HardwareDialog(ui.config)
    qtbot.addWidget(d)
    w = d.wiring
    assert [d.tabs.tabText(i) for i in range(d.tabs.count())] == ["Devices", "Wiring"]
    assert d.config().wiring is None  # untouched: still the preset's wiring (no section written)
    assert [w.ao_x.itemData(i) for i in range(w.ao_x.count())] == ["ao0", "ao1", "ao2", "ao3"]
    assert w.blank.count() == 32 and w.blank.currentData() == "port0/line23"
    assert [n for n, ch, _ in w.ext.items()] == [None, None] and [ch for _, ch, _ in w.ext.items()] == ["PFI6", "PFI7"]
    ai = w.ai.rows[0]["combo"]
    assert ai.count() == 32 and ai.currentData() == "ai0"
    # DIFF: only the AI+ channels of the pairs are offered
    _pick(w.ai_terminal, "DIFF")
    assert ai.count() == 16 and ai.findData("ai8") < 0 and ai.findData("ai16") >= 0
    _pick(w.ai_terminal, "RSE")
    # the video input range lives here too (not in the SEM window)
    assert w.ai_range.currentData() is None and "±10 V" in w.ai_range.itemText(0)
    _pick(w.ai_range, 2.0)

    # a conflict: two signals on one output
    _pick(w.ao_y, "ao0")
    assert not d._ok.isEnabled() and d.tabs.tabText(1).startswith("Wiring ⚠")
    assert "share the analog output ao0" in w.issues_label.text() and "#f55" in w.ao_y.styleSheet()
    _pick(w.ao_y, "ao1")
    assert d._ok.isEnabled() and d.tabs.tabText(1) == "Wiring" and w.ao_y.styleSheet() == ""
    # PFI6 is port1/line6: a static line there clashes with the relay line
    r = w.static.add_row("relay", "PFI6")
    assert not d._ok.isEnabled() and "#f55" in r["combo"].styleSheet()
    _pick(r["combo"], "PFI9")
    assert d._ok.isEnabled()

    # a second SE input, used by EDS; clock export; X/Y on other outputs
    _pick(w.ao_x, "ao2")
    _pick(w.aux0, "ao0")
    r = w.ai.add_row("bse", "ai4")
    _pick(w.eds_video, "bse")
    _pick(w.clock_out, "PFI12")
    out = d.config()
    assert out.wiring.ai_channels == {"video": "ai0", "bse": "ai4"} and out.wiring.eds_video == "bse"
    assert (out.wiring.ao_x, out.wiring.aux0, out.wiring.sample_clock_out) == ("ao2", "ao0", "PFI12")
    assert out.wiring.static_lines == {"relay": "PFI9"} and out.daq.ai_terminal == "RSE" and out.daq.ai_range_v == 2.0

    # Apply & reconnect: saved with the system config and used by the HAL and its consumers
    assert ui.restart_backend(out, ask=False)
    back = SystemConfig.load(ui.config.path)
    assert back.wiring == out.wiring
    cfg = ui.backend.manager.config
    assert (cfg.ao_x, cfg.aux_physical("aux0"), cfg.sample_clock_out) == ("ao2", "ao0", "PFI12")
    assert cfg.ai_channels == {"video": "ai0", "bse": "ai4"} and cfg.static_lines == {"relay": "PFI9"}
    assert ui.backend.eds.options.video_channel == "bse" and cfg.ai_range_v == 2.0
    assert ui.backend.engine.options.ai_range_v is None  # the SEM uses the hardware range

    # reopened, the dialog shows the saved wiring; Reset to preset brings back today's defaults
    d2 = HardwareDialog(ui.config)
    qtbot.addWidget(d2)
    assert d2.wiring.wiring() == out.wiring and d2.config().wiring == out.wiring
    d2.reset_wiring()
    assert d2.wiring.wiring() == d2.preset_wiring() and d2.wiring.ao_x.currentData() == "ao0"
    _pick(d2.daq_preset, "default")  # another preset loads its wiring
    assert d2.wiring.blank.currentData() == "port0/line0"


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


def test_scan_monitor_in_the_menu(ui, qtbot):
    import numpy as np

    from easyebl.core import OutputChunk

    mon = ui.menu.monitor
    assert ui.menu.monitor_box.isVisible() and mon.worker.running and ui.scan_tap.enabled
    assert ui.backend.manager.backend.output_tap is ui.scan_tap
    mon.retention.setValue(0)  # keep everything: the test must not race the persistence
    # unblanked at (+5 V, +5 V), blanked at (−5 V, −5 V)
    n = 20000
    c = OutputChunk(np.r_[np.full(n, 5.0), np.full(n, -5.0)], np.r_[np.full(n, 5.0), np.full(n, -5.0)],
                    np.r_[np.zeros(n, bool), np.ones(n, bool)])
    h = ui.backend.manager.get_handle("monitor-test")
    h.run_stream(500e3, [c])
    h.close()
    qtbot.waitUntil(lambda: mon.stats.total >= n, timeout=5000)
    img = mon.screen.image
    res = mon.settings()[0]
    assert img.width() == img.height() == res == 256

    def lit(xv, yv):
        col = int((xv + 10) / 20 * res)
        row = res - 1 - int((yv + 10) / 20 * res)
        return img.pixelIndex(col, row)

    assert lit(5, 5) == 255 and lit(-5, -5) == 0 and lit(0, 0) == 0
    # the beam spot is always shown: parked and blanked after the stream (hollow) …
    cfg = ui.backend.manager.config
    qtbot.waitUntil(lambda: mon.screen.spot == (cfg.park_x_v, cfg.park_y_v, True), timeout=5000)
    # … and solid where the beam is switched on
    h = ui.backend.manager.get_handle("monitor-test-2")
    h.park(-3.0, 7.0, blank=False)
    qtbot.waitUntil(lambda: mon.screen.spot == (-3.0, 7.0, False), timeout=5000)
    h.park()
    h.close()
    pt = mon.screen.spot_point()
    r = mon.screen.screen_rect()
    assert r.contains(pt.toPoint())
    mon.grab()  # paints offscreen
    # settings are saved with the configuration
    mon.res.setCurrentIndex(mon.res.findData(128))
    mon.retention.setValue(3.5)
    saved = SystemConfig.load(ui.config.path).monitor
    assert (saved.resolution, saved.retention_s) == (128, 3.5)
    qtbot.waitUntil(lambda: mon.screen.image.width() == 128, timeout=5000)
    # View → Scan monitor hides it, stops the worker and remembers that
    ui.menu.a_monitor.trigger()
    assert not ui.menu.monitor_box.isVisible() and not mon.worker.running and not ui.scan_tap.enabled
    assert SystemConfig.load(ui.config.path).monitor.visible is False
    ui.menu.a_monitor.trigger()
    assert mon.worker.running


def test_scan_monitor_follows_the_scan_orientation(ui):
    new = SystemConfig()
    new.path = ui.config.path
    new.daq.invert_x, new.daq.swap_xy = True, True
    assert ui.restart_backend(new, ask=False)
    mon = ui.menu.monitor
    mon.worker.step()  # the worker applies the setting
    assert mon.worker.acc.orientation == (True, False, True)
    assert ui.backend.engine.calibration.swap_xy and ui.backend.engine.calibration.invert_x
