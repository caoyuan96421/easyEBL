"""Convenience wrapper that opens the port(s) and builds both clients."""

from __future__ import annotations

from dataclasses import dataclass, field

from .channel import LineChannel
from .sem import SemClient, Timeouts
from .simulator import SimOptions, SimulatedInstrument
from .stage import CoordinateCodec, StageClient
from .transport import SerialTransport, Transport


@dataclass
class Session:
    """A SEM client and a stage client over one shared port or two separate ports.

    The manual does not establish how the stage protocol is routed relative to
    the SEM text interface. ``Session.open(port)`` assumes one shared port;
    pass ``stage_port=`` if the stage controllers are on their own line.
    """

    sem: SemClient
    stage: StageClient
    channels: list[LineChannel] = field(default_factory=list)
    simulator: SimulatedInstrument | None = None

    @classmethod
    def from_transports(
        cls,
        sem_transport: Transport,
        stage_transport: Transport | None = None,
        *,
        low_vacuum_capable: bool = False,
        timeouts: Timeouts | None = None,
        codec: CoordinateCodec | None = None,
        **stage_kw,
    ) -> "Session":
        sem_ch = LineChannel(sem_transport)
        channels = [sem_ch]
        if stage_transport is None:
            stage_ch = sem_ch
        else:
            stage_ch = LineChannel(stage_transport)
            channels.append(stage_ch)
        sem = SemClient(sem_ch, timeouts=timeouts, low_vacuum_capable=low_vacuum_capable)
        stage = StageClient(stage_ch, codec=codec, **stage_kw)
        return cls(sem=sem, stage=stage, channels=channels)

    @classmethod
    def open(
        cls,
        port: str,
        *,
        stage_port: str | None = None,
        rtscts: bool = False,
        dsrdtr: bool = False,
        xonxoff: bool = False,
        **kw,
    ) -> "Session":
        """Open real serial port(s) at 9600 8N1."""
        flow = dict(rtscts=rtscts, dsrdtr=dsrdtr, xonxoff=xonxoff)
        sem_t = SerialTransport(port, **flow)
        stage_t = SerialTransport(stage_port, **flow) if stage_port and stage_port != port else None
        return cls.from_transports(sem_t, stage_t, **kw)

    @classmethod
    def simulated(cls, options: SimOptions | None = None, **kw) -> "Session":
        """A session backed by the in-process simulator (shared port)."""
        sim = SimulatedInstrument(options or SimOptions())
        kw.setdefault("low_vacuum_capable", sim.options.low_vacuum_capable)
        s = cls.from_transports(sim, **kw)
        s.simulator = sim
        return s

    def close(self) -> None:
        for ch in self.channels:
            ch.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
