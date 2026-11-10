"""Building an :class:`ImagingEngine` from command-line arguments (no Qt needed)."""
from __future__ import annotations

import argparse

from ..core.calibration import FieldCalibration
from ..hal import DAQConfig, DAQManager, SimParams, SimulatedBackend, create_backend
from .engine import EngineOptions, ImagingEngine

__all__ = ["add_hardware_args", "build_manager", "open_engine"]


def add_hardware_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--backend", default="sim", choices=["sim", "nidaqmx"], help="hardware backend")
    p.add_argument("--device", default=None, help="NI device name (e.g. Dev1)")
    p.add_argument("--config", default=None, help="DAQConfig JSON file")
    p.add_argument("--preset", default="usb6363", choices=["usb6363", "default"], help="wiring preset")
    p.add_argument("--field-um", dest="field_um", type=float, default=100.0,
                   help="field (µm) covered by the full AO range at the current SEM magnification")
    p.add_argument("--rate", type=float, default=None, help="sample clock (S/s); default: device maximum")
    p.add_argument("--latency-ms", dest="latency_ms", type=float, default=100.0,
                   help="output buffer depth in ms (setting-change latency)")
    p.add_argument("--ai-terminal", dest="ai_terminal", default=None,
                   choices=["DIFF", "RSE", "NRSE", "PSEUDO_DIFF"],
                   help="video input mode (default: the preset's; DIFF needs AI+ and AI- wired)")
    p.add_argument("--phantom", action="store_true",
                   help="dummy-image test: drive a test chart on an aux output (default aux0 = AO2); "
                        "wire it to the video input (AO2 -> AI0)")
    p.add_argument("--phantom-output", dest="phantom_output", default="aux0", choices=["aux0", "aux1"],
                   help="aux output for the phantom (default aux0)")
    p.add_argument("--phantom-noise", dest="phantom_noise", type=float, default=0.0, metavar="V",
                   help="rms white noise added to the phantom signal (V), e.g. 0.2")
    return p


def build_manager(args) -> tuple[DAQManager, FieldCalibration]:
    if getattr(args, "config", None):
        with open(args.config, encoding="utf-8") as fh:
            cfg = DAQConfig.from_json(fh.read())
    elif getattr(args, "preset", "usb6363") == "usb6363":
        cfg = DAQConfig.usb6363()
    else:
        cfg = DAQConfig()
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "ai_terminal", None):
        cfg.ai_terminal = args.ai_terminal
    if getattr(args, "phantom", False):
        cfg.phantom_output = args.phantom_output
        cfg.phantom_noise_v = float(args.phantom_noise)
        cfg.validate()
        if cfg.phantom_physical is None:
            print(f"note: {args.phantom_output} is a dummy output on this configuration – no phantom signal")
    cal = FieldCalibration(float(args.field_um), cfg.ao_limit_v)
    if args.backend == "sim":
        be = SimulatedBackend(cfg, SimParams(field_size_um=float(args.field_um), v_full_scale=cfg.ao_limit_v,
                                             realtime=True, noise_sigma=0.05))
    else:
        be = create_backend(args.backend, cfg)
    # a missing/unplugged device is not fatal: the HAL connects as soon as it appears
    return DAQManager(be).open(allow_missing=True), cal


def open_engine(args, manager: DAQManager | None = None, calibration: FieldCalibration | None = None):
    """(manager, engine) from parsed :func:`add_hardware_args` arguments."""
    if manager is None:
        manager, cal = build_manager(args)
        calibration = calibration or cal
    opts = EngineOptions(rate=args.rate, latency_s=max(0.005, args.latency_ms / 1e3))
    if args.backend == "sim" and args.rate is None:
        # the simulator cannot synthesise 2 MS/s in real time: use a lighter clock
        opts = opts.with_(rate=500e3)
    return manager, ImagingEngine(manager, calibration, options=opts)
