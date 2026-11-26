"""Simulated column optics: the simulated SEM image blurs with the simulated focus and stigmator.

For the simulators only (auto focus / stigmation tests and demos): the S-3000 simulator's
``FOCUS C`` / ``FOCUS F`` / ``STG X`` / ``STG Y`` give a defocus and an astigmatism, and
:meth:`SimOptics.blur` the elliptical Gaussian beam blur on the sample that the simulated DAQ
backend applies (``SimParams.beam_blur``).

* defocus Δz = WD(focus) − WD(best) (µm, through the focus table);
* astigmatism: the stigmator error (ex, ey) from the best values gives two line foci at
  Δz ± A with ``A = |e| · astig_um_per_count``, along ½·atan2(ey, ex) (X acts at 0°, Y at 45°);
* blur σ = √(σ₀² + (α · (Δz ∓ A))²) along the two axes (α: convergence semi-angle).
"""
from __future__ import annotations

import math

from .s3000.focus_wd import wd_from_focus

__all__ = ["SimOptics", "couple_sim_optics"]


class SimOptics:
    def __init__(self, instrument, best_focus=(10670, 2300), best_stig=(2048 + 352, 2048 - 240),
                 convergence_rad: float = 5e-3, sigma0_um: float = 0.08, astig_um_per_count: float = 0.25):
        self.instrument = instrument  # the S-3000 simulator (its ``sem`` state dict)
        self.best_focus = tuple(int(v) for v in best_focus)
        self.best_stig = tuple(int(v) for v in best_stig)
        self.alpha = float(convergence_rad)
        self.sigma0 = float(sigma0_um)
        self.k_astig = float(astig_um_per_count)

    def _get(self, item, default):
        try:
            return int(self.instrument.sem.get(item, default))
        except (TypeError, ValueError):
            return default

    def defocus_um(self) -> float:
        c, f = self._get("FOCUS C", 10670), self._get("FOCUS F", 2048)
        return (wd_from_focus(c, f) - wd_from_focus(*self.best_focus)) * 1000.0

    def blur(self) -> tuple:
        dz = self.defocus_um()
        ex = self._get("STG X", 2048) - self.best_stig[0]
        ey = self._get("STG Y", 2048) - self.best_stig[1]
        a = math.hypot(ex, ey) * self.k_astig
        ang = 0.5 * math.atan2(ey, ex)
        sa = math.hypot(self.sigma0, self.alpha * (dz - a))
        sb = math.hypot(self.sigma0, self.alpha * (dz + a))
        return sa, sb, ang

    __call__ = blur


def couple_sim_optics(sim_backend, link, **kw) -> SimOptics | None:
    """Make the simulated DAQ image follow the simulated SEM's focus and stigmator."""
    sim = getattr(getattr(link, "session", None), "simulator", None)
    if sim is None or sim_backend is None or not hasattr(sim_backend, "params"):
        return None
    optics = SimOptics(sim, **kw)
    sim_backend.params.beam_blur = optics
    return optics
