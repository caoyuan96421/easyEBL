"""Synthetic EDS spectra for tests: the EDS phantom's photons through the simulated DPP (no DAQ)."""
import functools

import numpy as np

from easyebl.eds import EnergyCalibration, Spectrum, SpectrumMap
from easyebl.hal.eds_phantom import EDSPhantomSpecimen, Phase
from easyebl.hal.mca import SimMCAParams, SimulatedPulseProcessor

FS = 1.25e6


class _Src:
    def __init__(self):
        self.listeners = []

    def add_photon_listener(self, fn):
        self.listeners.append(fn)

    def remove_photon_listener(self, fn):
        self.listeners.remove(fn)


def true_calibration(p: SimMCAParams | None = None) -> EnergyCalibration:
    p = p or SimMCAParams()
    return EnergyCalibration(p.offset_kev + p.kev_per_channel / 2, p.kev_per_channel)


def spectrum_of(comp: dict, seconds: float = 10.0, cps: float = 20e3, beam_kv: float = 20.0, seed: int = 3,
                params: SimMCAParams | None = None) -> Spectrum:
    """Spectrum of a homogeneous material *comp* (mass fractions) at *cps* input rate."""
    ph = EDSPhantomSpecimen(n=8, seed=seed, cps=cps, beam_kv=beam_kv,
                            phases=(Phase("x", tuple(comp.items()), 0.5),), labels=np.zeros((8, 8), np.uint8))
    src = _Src()
    p = params or SimMCAParams(seed=seed)
    d = SimulatedPulseProcessor(src, p).open()
    ph._listeners = src.listeners
    d.set_preset_real_time(seconds)
    d.clear()
    d.enable()
    n = 262144
    for _ in range(int(seconds * FS / n) + 2):
        ph.photons(np.zeros(n), np.zeros(n), np.zeros(n, bool), FS)
    r = d.read()
    return Spectrum(r.counts, true_calibration(p), r.real_time, r.live_time, r.fast_count, r.slow_count, beam_kv)


@functools.lru_cache(maxsize=8)
def _expected(comp_items: tuple, beam_kv: float) -> np.ndarray:
    s = spectrum_of(dict(comp_items), seconds=20.0, cps=20e3, beam_kv=beam_kv, seed=11)
    return s.counts / s.real_time  # counts/s at 20 kcps input


def synthetic_map(labels: np.ndarray, comps: list, dwell_s: float = 0.05, beam_kv: float = 20.0, seed: int = 0,
                  channels: int = 2048) -> SpectrumMap:
    """Spectrum image of a phase label map (Poisson counts from each phase's mean spectrum)."""
    rng = np.random.default_rng(seed)
    ny, nx = labels.shape
    cal = true_calibration()
    rates = [_expected(tuple(sorted(c.items())), beam_kv)[:channels] for c in comps]
    sm = SpectrumMap.empty(nx, ny, channels, cal, (0.0, 0.0, 10.0, 10.0), dwell_s, beam_kv=beam_kv)
    for j in range(ny):
        for i in range(nx):
            lam = rates[labels[j, i]] * dwell_s
            sm.cube[j, i] = rng.poisson(lam)
    sm.real_time[:] = dwell_s
    sm.live_time[:] = dwell_s * 0.8
    sm.fast_count[:] = sm.cube.sum(axis=2) * 1.25
    sm.done[:] = True
    sm.se = labels.astype(float)
    return sm
