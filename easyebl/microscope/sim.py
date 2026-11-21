"""The S-3000 simulator, tuned to behave like the instrument at the link level.

The vendored :class:`~easyebl.microscope.s3000.SimulatedInstrument` answers
both protocols exactly as the S-3000 did on the bench.  Here it also gets the
instrument's *response time* (a delay before every reply, SEM and stage
alike – the real SEM needs 0.25–1 s), and a plausible starting state: HV on at
20 kV, ×1k, WD 15 mm, the stage near the positions read on the instrument,
``#STAGE`` off (the HAL switches it on).
"""
from __future__ import annotations

import time

from .s3000 import Session, SimOptions, SimulatedInstrument

__all__ = ["SlowSimulatedInstrument", "simulated_session", "SIM_START_PULSES"]

#: stage start positions (pulses) – the ones read on the S-3000 (X 17077D, Y 0E8EB7, R 0A2819, Z 051BEB, T 000062)
SIM_START_PULSES = {(0, 0): 0x17077D, (0, 1): 0x0E8EB7, (1, 0): 0x0A2819, (2, 0): 0x051BEB, (2, 1): 0x000062}


class SlowSimulatedInstrument(SimulatedInstrument):
    """Simulator whose stage lines are delayed like the SEM lines (``options.command_delay``)."""

    def _handle(self, line: str) -> None:
        if line.startswith("#") and len(line) >= 5 and line[1:3].isdigit() and self.options.command_delay:
            time.sleep(self.options.command_delay)
        super()._handle(line)


def simulated_session(command_delay_s: float = 0.25, stage_speed_pulses_s: float = 25.6 * 2000,
                      hv_on: bool = True, hv_kv: str = "20.0", mag: str = "1k") -> Session:
    """A :class:`Session` on the simulator (shared port), with realistic timing and state."""
    sim = SlowSimulatedInstrument(SimOptions(command_delay=float(command_delay_s), stage_speed=stage_speed_pulses_s,
                                             auto_delay=max(0.3, 4 * command_delay_s)))
    sim.sem["HV_SET"] = hv_kv
    sim.sem["HV_ON"] = bool(hv_on)
    sim.sem["HV"] = hv_kv if hv_on else "OFF"
    sim.sem["MAG"] = mag
    sim.sem["EMI"] = "12" if hv_on else "0"
    for key, p in SIM_START_PULSES.items():
        sim.axes[key].position = p
    s = Session.from_transports(sim)
    s.simulator = sim
    return s
