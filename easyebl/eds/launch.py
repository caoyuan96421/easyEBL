"""Building the EDS backend from command-line arguments (no Qt).

    p = add_eds_args(add_hardware_args(argparse.ArgumentParser()))
    args = p.parse_args()
    prepare_config_args(args)                 # default pulse processor for the backend
    mgr, cal = build_manager(args)            # easyebl.imaging.launch
    ctrl, mca, settings = build_eds(args, mgr, cal)
"""
from __future__ import annotations

import argparse

from ..core.calibration import FieldCalibration
from ..hal.mca import create_pulse_processor
from .controller import EDSController, EDSOptions
from .settings import EDSSettings

__all__ = ["add_eds_args", "prepare_config_args", "build_eds"]


def add_eds_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--mca", default=None, choices=["dp5", "mock"],
                   help="pulse processor: dp5 (USB) or mock (the DP5 driver's mock device); "
                        "default: mock with --backend sim, else dp5")
    p.add_argument("--kv", type=float, default=None, help="beam voltage (kV) for the analysis (default: saved)")
    p.add_argument("--settings", default=None, help="EDS settings JSON (default: per-user file)")
    p.add_argument("--no-external", dest="no_external", action="store_true",
                   help="do not switch the SEM to external scan for EDS jobs")
    return p


def prepare_config_args(args) -> None:
    """Default pulse processor: the DP5 mock device with the simulator backend, else the USB DP5."""
    if getattr(args, "mca", None) is None:
        args.mca = "mock" if getattr(args, "backend", "sim") == "sim" else "dp5"


def build_eds(args, manager, calibration: FieldCalibration | None = None, owner: str = "eds"):
    """(EDSController, PulseProcessor, EDSSettings) for *manager*."""
    settings = EDSSettings.load(getattr(args, "settings", None))
    if getattr(args, "kv", None):
        settings.beam_kv = float(args.kv)
    mca = create_pulse_processor(getattr(args, "mca", None) or "dp5").open()
    cal = calibration or FieldCalibration(settings.field_um, manager.config.ao_limit_v)
    opts = EDSOptions(external_control=not getattr(args, "no_external", False))
    ctrl = EDSController(manager, mca, cal, settings.energy_calibration, settings.beam_kv, settings.takeoff_deg,
                         options=opts, owner=owner)
    return ctrl, mca, settings
