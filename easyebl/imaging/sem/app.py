"""Entry point of the SEM GUI (``easyebl-sem`` / ``python -m easyebl.imaging.sem``)."""
from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    from PySide6 import QtWidgets

    from ...microscope.launch import add_microscope_args, open_from_args
    from ..cli import start_console
    from ..launch import add_hardware_args, open_engine
    from .window import SEMWindow, apply_dark_theme

    p = add_microscope_args(add_hardware_args(argparse.ArgumentParser(description="easyEBL SEM imaging")))
    p.add_argument("--console", action="store_true",
                   help="also accept console commands on stdin (same engine, thread-safe)")
    p.add_argument("--run", action="store_true", help="start live scanning immediately")
    args = p.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    mgr, engine = open_engine(args)
    mic = open_from_args(args)
    w = SEMWindow(engine, simulated=args.backend == "sim", microscope_bundle=mic)
    mic.follow_magnification([engine], sim_backend=mgr.backend if args.backend == "sim" else None)
    w.resize(1500, 950)
    w.show()
    if args.console:
        start_console(engine, on_quit=lambda: QtWidgets.QApplication.instance().quit())
    if args.run:
        engine.run()
    try:
        return app.exec()
    finally:
        mic.close()
        engine.close()
        mgr.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
