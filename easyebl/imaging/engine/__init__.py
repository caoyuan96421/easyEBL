"""Threaded, low-latency SEM imaging engine.

Modules:

* :mod:`.settings`  – scan speed / resolution presets, :class:`EngineOptions`
* :mod:`.source`    – fast line-by-line waveform generation (:class:`FrameSource`)
* :mod:`.program`   – the stream program that follows commands at line boundaries
* :mod:`.assembler` – AI stream → image rows, display composite, capture integration
* :mod:`.frames`    – published :class:`Frame` objects, :class:`FrameSlot`, events
* :mod:`.core`      – :class:`ImagingEngine` (scan thread + thread-safe commands)
"""
from .core import EngineStatus, ImagingEngine, default_scan_config
from .frames import EngineEvent, EngineState, Frame, FrameSlot
from .settings import AI_RANGES_V, RESOLUTIONS, SCAN_SPEEDS, EngineOptions, ScanSpeed, speed_by_name, valid_rates
from .source import FrameSource

__all__ = [
    "ImagingEngine",
    "EngineStatus",
    "EngineOptions",
    "EngineState",
    "EngineEvent",
    "Frame",
    "FrameSlot",
    "FrameSource",
    "ScanSpeed",
    "SCAN_SPEEDS",
    "RESOLUTIONS",
    "AI_RANGES_V",
    "speed_by_name",
    "valid_rates",
    "default_scan_config",
]
