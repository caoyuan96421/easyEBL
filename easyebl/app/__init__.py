"""The easyEBL program: one entry point for every subsystem.

    easyebl                  # the instrument configuration saved in Hardware → Configure…
    easyebl --sim            # this session on simulators only (nothing saved)
    easyebl --open sem,eds   # also open these windows (sem, eds, microscope, litho, …)
    python -m easyebl        # the same

Starting it opens a small menu window showing whether the scan DAQ, the EDS
pulse processor and the microscope are connected, and puts an icon in the
system tray.  The hardware stays connected in the background when windows
(or the menu itself) are closed; *Shut down easyEBL* in the tray menu ends it.
A second start only brings the running program's menu forward.

* :mod:`.config`  – :class:`SystemConfig` (``%APPDATA%\\easyEBL\\system.json``)
* :mod:`.backend` – :class:`SystemBackend`: DAQ + imaging engine, DP5 + EDS controller, microscope port
* :mod:`.gui`     – :class:`EasyEBLApp`: menu window, tray icon, subsystem windows
"""
from __future__ import annotations

import argparse
import getpass
import sys

from .backend import ComponentStatus, SystemBackend
from .config import SystemConfig

__all__ = ["SystemConfig", "SystemBackend", "ComponentStatus", "main", "instance_name"]

SUBSYSTEM_KEYS = ("sem", "eds", "microscope", "litho", "vector", "gds", "dose", "alignment", "raster", "hal")


def instance_name() -> str:
    """Local-socket name of the running easyEBL of this user."""
    user = "".join(c for c in getpass.getuser() if c.isalnum()) or "user"
    return f"easyebl-{user}"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="easyebl", description="easyEBL – SEM imaging, EDS, microscope control "
                                                            "and e-beam lithography")
    p.add_argument("--sim", action="store_true",
                   help="use simulators for every device in this session (the saved configuration is kept)")
    p.add_argument("--config", default=None, help="hardware configuration JSON (default: per-user file)")
    p.add_argument("--open", default="", metavar="LIST",
                   help="open these subsystems at start, comma separated: " + ", ".join(SUBSYSTEM_KEYS))
    p.add_argument("--tray", action="store_true", help="start in the tray (no menu window)")
    return p


def _send_to_running(name: str, message: str) -> bool:
    from PySide6 import QtNetwork

    s = QtNetwork.QLocalSocket()
    s.connectToServer(name)
    if not s.waitForConnected(500):
        return False
    s.write((message + "\n").encode("utf-8"))
    s.flush()
    s.waitForBytesWritten(1000)
    s.waitForReadyRead(3000)  # the running program answers "ok" once it has the message
    s.disconnectFromServer()
    return True


def _listen(name: str, app_obj):
    from PySide6 import QtCore, QtNetwork

    QtNetwork.QLocalServer.removeServer(name)  # a stale socket left by a crash
    server = QtNetwork.QLocalServer(app_obj)
    server.listen(name)

    def on_conn():
        while server.hasPendingConnections():
            c = server.nextPendingConnection()
            buf = bytearray()

            def read(c=c, buf=buf):
                buf.extend(bytes(c.readAll()))
                while b"\n" in buf:
                    line, _, rest = bytes(buf).partition(b"\n")
                    buf[:] = rest
                    c.write(b"ok\n")
                    c.flush()
                    # handled after this slot: a dialog opened by the command must not block the reply
                    QtCore.QTimer.singleShot(0, lambda t=line.decode("utf-8", "replace"): app_obj.handle_command(t))

            def done(c=c, buf=buf):
                read()
                if buf:
                    app_obj.handle_command(bytes(buf).decode("utf-8", "replace"))
                    buf.clear()
                c.deleteLater()

            c.readyRead.connect(read)
            c.disconnected.connect(done)
            read()  # what arrived before the signals were connected

    server.newConnection.connect(on_conn)
    return server


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = _parser().parse_args(argv)
    opens = [k.strip() for k in args.open.split(",") if k.strip()]
    bad = [k for k in opens if k not in SUBSYSTEM_KEYS]
    if bad:
        _parser().error(f"unknown subsystem(s): {', '.join(bad)}")

    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("easyEBL")
    name = instance_name()
    # one easyEBL per user: the DAQ, the DP5 and the serial port can be opened by one process only
    if _send_to_running(name, "open " + ",".join(opens) if opens else "show"):
        print("easyEBL is already running – its menu was brought to the front.")
        return 0

    from ..imaging.sem.window import apply_dark_theme
    from .gui import EasyEBLApp

    apply_dark_theme(app)
    config = SystemConfig.load(args.config)
    if args.sim:
        config = config.as_simulated()
    ui = EasyEBLApp(config, persistent=not args.sim)
    server = _listen(name, ui)
    ui.start(show_menu=not args.tray)
    for k in opens:
        ui.open(k)
    try:
        return app.exec()
    finally:
        server.close()
        if not ui.closing:
            ui.shutdown(ask=False)
