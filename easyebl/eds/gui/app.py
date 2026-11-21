"""Entry points: the EDS window alone (``easyebl-eds``) and SEM + EDS in one process (``easyebl-suite``).

Run SEM and EDS together through the suite: both share one DAQ manager, so an
EDS job pauses the live SEM image (HAL lease) and the SEM resumes by itself
afterwards.  Two separate processes cannot share the NI device (the second one
gets a clear "in use" error) – nor the microscope's serial port, which the
suite opens once for the SEM, EDS and microscope windows.
"""
from __future__ import annotations

import argparse
import sys

__all__ = ["main", "suite_main"]


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _parser(desc: str) -> argparse.ArgumentParser:
    from ...imaging.launch import add_hardware_args
    from ...microscope.launch import add_microscope_args
    from ..launch import add_eds_args

    return add_microscope_args(add_eds_args(add_hardware_args(argparse.ArgumentParser(description=desc))))


def main(argv=None) -> int:
    """EDS window only."""
    _utf8()
    from PySide6 import QtWidgets

    from ...imaging.launch import build_manager
    from ...imaging.sem.window import apply_dark_theme
    from ...microscope.launch import open_from_args
    from ..launch import build_eds, prepare_config_args
    from .window import EDSWindow

    args = _parser("easyEBL EDS").parse_args(argv)
    prepare_config_args(args)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    mgr, cal = build_manager(args)
    ctrl, mca, settings = build_eds(args, mgr, cal)
    mic = open_from_args(args)
    w = EDSWindow(ctrl, settings, simulated=args.backend == "sim", microscope_bundle=mic)
    mic.follow_magnification([ctrl], sim_backend=mgr.backend if args.backend == "sim" else None)
    w.resize(1600, 980)
    w.show()
    try:
        return app.exec()
    finally:
        mic.close()
        ctrl.close()
        mca.close()
        mgr.close()


def suite_main(argv=None) -> int:
    """SEM imaging + EDS (+ the microscope window) sharing one DAQ and one microscope port."""
    _utf8()
    from PySide6 import QtWidgets

    from ...imaging.launch import open_engine
    from ...imaging.sem.window import SEMWindow, apply_dark_theme
    from ...microscope.launch import open_from_args
    from ..launch import build_eds, prepare_config_args
    from .window import EDSWindow

    p = _parser("easyEBL SEM + EDS")
    p.add_argument("--run", action="store_true", help="start live SEM scanning immediately")
    args = p.parse_args(argv)
    prepare_config_args(args)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    mgr, engine = open_engine(args)
    ctrl, mca, settings = build_eds(args, mgr, engine.calibration)
    mic = open_from_args(args)
    sem = SEMWindow(engine, simulated=args.backend == "sim", microscope_bundle=mic)
    sem.resize(1300, 900)
    eds = EDSWindow(ctrl, settings, sem_engine=engine, simulated=args.backend == "sim", microscope_bundle=mic)
    eds.resize(1600, 980)
    mic.follow_magnification([engine, ctrl], sim_backend=mgr.backend if args.backend == "sim" else None)
    sem.show()
    eds.show()
    if mic.connected:
        mic.show_window()
    if args.run:
        engine.run()
    try:
        return app.exec()
    finally:
        mic.close()
        ctrl.close()
        engine.close()
        mca.close()
        mgr.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
