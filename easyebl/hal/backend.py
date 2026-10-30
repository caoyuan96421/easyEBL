"""Abstract DAQ backend interface.

A backend knows how to talk to one piece of hardware (or a simulator).  It is
*not* thread-safe and is never used directly by application modules; they go
through :class:`easyebl.hal.manager.DAQManager` / :class:`DAQHandle`, which
serialise access and enforce the safe state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core.waveform import OutputChunk
from .config import DAQConfig

__all__ = [
    "DAQBackend",
    "BackendCapabilities",
    "StreamResult",
    "HALError",
    "DeviceBusyError",
    "ResourceConflictError",
    "VoltageRangeError",
    "StreamAborted",
    "AIConsumer",
]

#: Called with each block of acquired AI samples, shape (n_channels, n_samples),
#: in acquisition order.  Sample ``k`` of the stream was taken on the same
#: clock edge as output sample ``k``.
AIConsumer = Callable[[np.ndarray], None]


class HALError(RuntimeError):
    """Base class for HAL errors."""


class DeviceBusyError(HALError):
    """The device is executing another hardware-timed operation."""


class ResourceConflictError(HALError):
    """A resource (line/channel) is claimed by another handle."""


class VoltageRangeError(HALError):
    """An output sample exceeds the configured AO limit."""


class StreamAborted(HALError):
    """Raised (internally) when a stream is aborted by request."""


@dataclass
class BackendCapabilities:
    name: str
    max_ao_rate: float
    max_ai_aggregate_rate: float
    ai_channels: tuple
    timed_lines: tuple
    static_lines: tuple
    simulated: bool = False


@dataclass
class StreamResult:
    """Outcome of :meth:`DAQBackend.run_stream`."""

    samples_written: int
    rate: float
    aborted: bool = False
    ai: np.ndarray | None = None  # (n_ch, n) when no consumer was given
    ai_channels: tuple = ()
    elapsed_s: float = 0.0
    info: dict = field(default_factory=dict)


class DAQBackend(ABC):
    """Hardware (or simulator) implementation."""

    def __init__(self, config: DAQConfig):
        config.validate()
        self.config = config
        self._open = False

    # ---------------------------------------------------------------- lifecycle
    @property
    def is_open(self) -> bool:
        return self._open

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    # ------------------------------------------------------------------- static
    @abstractmethod
    def write_static(self, x_v: float, y_v: float, blank: bool) -> None:
        """Immediately set deflection voltages and the blanker (software timed)."""

    @abstractmethod
    def set_static_line(self, name: str, value: bool) -> None:
        """Set a software-timed auxiliary line (logical value, polarity applied)."""

    # ------------------------------------------------------------------ streams
    @abstractmethod
    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        abort: threading.Event | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> StreamResult:
        """Play a sample-clocked output stream, acquiring AI on the same clock.

        Implementations must:
        * generate X, Y, blank and marker lines from one sample clock;
        * acquire the requested AI channels on that same clock, one AI sample
          per output sample;
        * check *abort* between chunks and stop promptly when set;
        * leave the hardware in the safe state (blanked, parked) on exit, even
          on error.
        """

    # ------------------------------------------------------------------ helpers
    def blank_level(self, blank: bool) -> bool:
        """Electrical level for a logical blank state."""
        return bool(blank) if self.config.blank_active_high else not bool(blank)

    def static_level(self, name: str, value: bool) -> bool:
        active_high = self.config.static_lines_active_high.get(name, True)
        return bool(value) if active_high else not bool(value)

    def safe_state(self) -> None:
        self.write_static(self.config.park_x_v, self.config.park_y_v, True)
