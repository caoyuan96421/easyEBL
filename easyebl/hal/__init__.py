"""Hardware abstraction layer over the scan DAQ.

Typical use::

    from easyebl.hal import DAQConfig, DAQManager, create_backend

    mgr = DAQManager(create_backend("sim", DAQConfig())).open()
    h = mgr.get_handle("imaging")
    result = h.run_stream(rate, chunks, ai_channels=["video"])
"""
from .backend import (
    BackendCapabilities,
    DAQBackend,
    DeviceBusyError,
    HALError,
    ResourceConflictError,
    StreamResult,
    VoltageRangeError,
)
from .config import DAQConfig
from .manager import BeamState, DAQHandle, DAQManager
from .simulated import (
    DoseMap,
    FunctionSample,
    ImageSample,
    SampleModel,
    SimParams,
    SimulatedBackend,
    make_demo_sample,
)

__all__ = [
    "BackendCapabilities",
    "DAQBackend",
    "DeviceBusyError",
    "HALError",
    "ResourceConflictError",
    "StreamResult",
    "VoltageRangeError",
    "DAQConfig",
    "BeamState",
    "DAQHandle",
    "DAQManager",
    "DoseMap",
    "FunctionSample",
    "ImageSample",
    "SampleModel",
    "SimParams",
    "SimulatedBackend",
    "make_demo_sample",
    "create_backend",
]


def create_backend(kind: str = "sim", config: DAQConfig | None = None, **kwargs) -> DAQBackend:
    """Factory: ``"sim"`` → :class:`SimulatedBackend`, ``"nidaqmx"`` → :class:`NIDAQBackend`."""
    config = config or DAQConfig()
    if kind in ("sim", "simulated", "simulator"):
        return SimulatedBackend(config, **kwargs)
    if kind in ("ni", "nidaq", "nidaqmx"):
        from .nidaq import NIDAQBackend

        return NIDAQBackend(config)
    raise ValueError(f"unknown backend kind '{kind}'")
