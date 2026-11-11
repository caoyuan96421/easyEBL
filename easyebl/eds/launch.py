"""Building the EDS backend from command-line arguments (no Qt).

    p = add_eds_args(add_hardware_args(argparse.ArgumentParser()))
    args = p.parse_args()
    prepare_config_args(args)                 # sim / --phantom eds → EDS phantom on AO2 + AO3
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
    p.add_argument("--mca", default=None, choices=["dp5", "mock", "sim"],
                   help="pulse processor: dp5 (USB), mock (DP5 driver simulator, no beam coupling) or sim "
                        "(fed by the EDS phantom); default: sim with --backend sim, else dp5")
    p.add_argument("--kv", type=float, default=None, help="beam voltage (kV) for the analysis (default: saved)")
    p.add_argument("--settings", default=None, help="EDS settings JSON (default: per-user file)")
    p.add_argument("--no-external", dest="no_external", action="store_true",
                   help="do not switch the SEM to external scan for EDS jobs")
    return p


def prepare_config_args(args) -> None:
    """With the simulator, default to the EDS phantom (SE image + SDD) so spectra mean something."""
    if getattr(args, "backend", "sim") == "sim" and not getattr(args, "phantom", None):
        args.phantom = "eds"
    if getattr(args, "mca", None) is None:
        args.mca = "sim" if getattr(args, "backend", "sim") == "sim" else "dp5"


def build_eds(args, manager, calibration: FieldCalibration | None = None, owner: str = "eds"):
    """(EDSController, PulseProcessor, EDSSettings) for *manager*."""
    settings = EDSSettings.load(getattr(args, "settings", None))
    if getattr(args, "kv", None):
        settings.beam_kv = float(args.kv)
    kind = getattr(args, "mca", None) or "dp5"
    if kind == "sim":
        mca = create_pulse_processor("sim", manager=manager).open()
        if not settings.calibration_source.startswith("Simulated"):
            # the simulated DPP's true calibration as a starting point (calibrate to check it)
            p = mca.params
            from .spectrum import EnergyCalibration

            settings.energy_calibration = EnergyCalibration(p.offset_kev + p.kev_per_channel / 2, p.kev_per_channel)
            settings.calibration_source = "Simulated DPP (true calibration)"
    else:
        mca = create_pulse_processor(kind).open()
    cal = calibration or FieldCalibration(settings.field_um, manager.config.ao_limit_v)
    opts = EDSOptions(external_control=not getattr(args, "no_external", False))
    ctrl = EDSController(manager, mca, cal, settings.energy_calibration, settings.beam_kv, settings.takeoff_deg,
                         options=opts, owner=owner)
    return ctrl, mca, settings
