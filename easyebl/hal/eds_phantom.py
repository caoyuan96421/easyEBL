"""EDS phantom: a synthetic multi-phase specimen for the SEM *and* the X-ray detector.

Like the imaging phantom (:mod:`easyebl.hal.phantom`) it lives in
deflection-voltage space and is evaluated sample by sample at the commanded
beam position, on the stream's sample clock.  It drives two aux outputs:

* **image** (``DAQConfig.phantom_output``, AO2 on the USB-6363): an SE-like image
  of the specimen (phase contrast, bright edges, a little texture) – wire it to
  the video input (AO2 → AI0);
* **SDD** (``DAQConfig.sdd_output``, AO3): the output of a *reset-type* SDD
  charge-sensitive preamplifier – a staircase of photon steps (height ∝ energy)
  on a leakage ramp, with fast resets – wire it to the pulse processor input
  (AO3 → DP5).  Photon arrival rates and energies follow the composition
  under the beam: each phase emits the characteristic lines and bremsstrahlung
  given by :mod:`easyebl.xray.generation` at the phantom's beam voltage.

With the simulator backend there is no analogue SDD signal: photons go straight
to listeners (``add_photon_listener``), which is how
:class:`~easyebl.hal.mca.SimulatedPulseProcessor` receives them.

Ground truth for tests: :meth:`phase_at`, :meth:`phase_image`,
:attr:`phases`, :meth:`phase_spectrum`.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import threading

import numpy as np

from .phantom import PhantomSpecimen

__all__ = ["Phase", "ResetPreamp", "EDSPhantomSpecimen", "default_phases", "make_eds_specimen"]


@dataclass(frozen=True)
class Phase:
    name: str
    composition: tuple  # ((element, mass fraction), …)
    se_yield: float  # 0…1 image brightness

    @property
    def comp(self) -> dict:
        return dict(self.composition)


def default_phases() -> tuple:
    """The phases of the default specimen (index = label in :meth:`EDSPhantomSpecimen.phase_image`)."""
    return (
        Phase("Si substrate", (("Si", 1.0),), 0.22),
        Phase("Al pad", (("Al", 1.0),), 0.32),
        Phase("Cu grid", (("Cu", 1.0),), 0.55),
        Phase("stainless", (("Fe", 0.70), ("Cr", 0.19), ("Ni", 0.09), ("Mn", 0.02)), 0.62),
        Phase("TiO2", (("Ti", 0.60), ("O", 0.40)), 0.45),
        Phase("NaCl", (("Na", 0.39), ("Cl", 0.61)), 0.38),
        Phase("Au", (("Au", 1.0),), 0.92),
    )


def _box(u, v, x0, x1, y0, y1):
    return (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)


def make_eds_specimen(n: int = 1024, seed: int = 7) -> np.ndarray:
    """(n, n) uint8 phase label map, row 0 = top (+y), coordinates u, v ∈ [-1, 1].

    Layout (asymmetric, so mirrored/rotated axes are obvious):
    Si background; a Cu grid (bars every 0.5); four Al pads along the top;
    stainless-steel particles (left half); TiO2 disks (right half); NaCl cubes
    (bottom); the letters "EDS" in Au (lower left).
    """
    rng = np.random.default_rng(seed)
    c = (np.arange(n) + 0.5) * (2.0 / n) - 1.0
    u, v = np.meshgrid(c, c[::-1])
    lab = np.zeros((n, n), np.uint8)
    # Cu grid bars
    for g in (-0.5, 0.0, 0.5):
        lab[np.abs(u - g) < 0.025] = 2
        lab[np.abs(v - g) < 0.025] = 2
    # Al pads along the top
    for k in range(4):
        x0 = -0.9 + k * 0.45
        lab[_box(u, v, x0, x0 + 0.3, 0.62, 0.9)] = 1
    # stainless particles (left), TiO2 disks (right)
    for _ in range(14):
        cx, cy, r = rng.uniform(-0.95, -0.08), rng.uniform(-0.45, 0.55), rng.uniform(0.03, 0.08)
        lab[np.hypot(u - cx, v - cy) < r] = 3
    for _ in range(10):
        cx, cy, r = rng.uniform(0.08, 0.95), rng.uniform(-0.45, 0.55), rng.uniform(0.04, 0.09)
        lab[np.hypot(u - cx, v - cy) < r] = 4
    # NaCl cubes along the bottom right
    for _ in range(8):
        cx, cy, s = rng.uniform(0.05, 0.9), rng.uniform(-0.92, -0.6), rng.uniform(0.04, 0.08)
        lab[_box(u, v, cx - s, cx + s, cy - s, cy + s)] = 5
    # "EDS" in Au, lower left
    w = 0.03
    x, y0, y1 = -0.9, -0.92, -0.6
    E = (_box(u, v, x, x + w, y0, y1) | _box(u, v, x, x + 0.16, y1 - w, y1)
         | _box(u, v, x, x + 0.13, (y0 + y1) / 2 - w / 2, (y0 + y1) / 2 + w / 2) | _box(u, v, x, x + 0.16, y0, y0 + w))
    x = -0.68
    D = (_box(u, v, x, x + w, y0, y1) | (((np.hypot((u - x) / 0.16, (v - (y0 + y1) / 2) / 0.16) < 1.0)
                                         & (np.hypot((u - x) / 0.13, (v - (y0 + y1) / 2) / 0.13) > 1.0)) & (u > x)))
    x = -0.45
    S = (_box(u, v, x, x + 0.16, y1 - w, y1) | _box(u, v, x, x + w, (y0 + y1) / 2, y1)
         | _box(u, v, x, x + 0.16, (y0 + y1) / 2 - w / 2, (y0 + y1) / 2 + w / 2)
         | _box(u, v, x + 0.16 - w, x + 0.16, y0, (y0 + y1) / 2) | _box(u, v, x, x + 0.16, y0, y0 + w))
    lab[E | D | S] = 6
    return lab


@functools.lru_cache(maxsize=2)
def _default_labels(n: int) -> np.ndarray:
    a = make_eds_specimen(n)
    a.setflags(write=False)
    return a


class ResetPreamp:
    """Reset-type SDD charge-sensitive preamplifier output, fed with photon events.

    Each photon is a step of ``mv_per_kev × E`` (linear rise over ``rise_s``) on a
    leakage ramp; when the output reaches ``v_high`` a fast reset pulls it to
    ``v_low`` and holds it for ``reset_hold_s`` (photons arriving then are lost,
    as in a real inhibit window).  State carries across chunks.
    """

    def __init__(self, mv_per_kev: float = 6.0, leak_v_per_s: float = 100.0, v_low: float = -4.5,
                 v_high: float = 4.5, rise_s: float = 0.5e-6, reset_s: float = 1e-6, reset_hold_s: float = 5e-6,
                 noise_mv: float = 0.2, polarity: int = 1, seed: int | None = None):
        self.gain = mv_per_kev * 1e-3
        self.leak = float(leak_v_per_s)
        self.v_low, self.v_high = float(v_low), float(v_high)
        self.rise_s, self.reset_s, self.hold_s = float(rise_s), float(reset_s), float(reset_hold_s)
        self.noise = noise_mv * 1e-3
        self.polarity = 1.0 if polarity >= 0 else -1.0
        self.rng = np.random.default_rng(seed)
        self.level = self.v_low
        self.carry = np.zeros(0)
        self.pending = np.zeros(0)
        self.n_resets = 0
        self.n_photons = 0
        self.n_lost = 0

    def waveform(self, n: int, fs: float, idx: np.ndarray, energies: np.ndarray) -> np.ndarray:
        """*n* output samples (V) at sample rate *fs* with photons at sample *idx* of energy *energies* keV."""
        rise_n = max(1, int(round(self.rise_s * fs)))
        reset_n = max(1, int(round(self.reset_s * fs)))
        hold_n = max(0, int(round(self.hold_s * fs)))
        leak = self.leak / fs
        incr = np.full(n + rise_n, leak)
        k = min(self.carry.size, n + rise_n)
        incr[:k] += self.carry[:k]
        if idx.size:
            amp = energies * self.gain / rise_n
            for j in range(rise_n):
                np.add.at(incr, idx + j, amp)
            self.n_photons += idx.size
        self.carry = incr[n:] - leak
        self.carry[self.carry < 1e-15] = 0.0
        incr = incr[:n]
        out = np.empty(n)
        pos = 0
        level = self.level
        if self.pending.size:
            m = min(self.pending.size, n)
            out[:m] = self.pending[:m]
            self.pending = self.pending[m:]
            self.n_lost += int(np.count_nonzero((idx >= 0) & (idx < m))) if idx.size else 0
            pos, level = m, self.v_low
        while pos < n:
            seg = level + np.cumsum(incr[pos:])
            over = np.nonzero(seg >= self.v_high)[0]
            if over.size == 0:
                out[pos:] = seg
                level = seg[-1]
                break
            i = over[0]
            out[pos:pos + i] = seg[:i]
            start = seg[i - 1] if i > 0 else level
            rst = np.concatenate([np.linspace(start, self.v_low, reset_n + 1)[1:], np.full(hold_n, self.v_low)])
            self.n_resets += 1
            p = pos + i
            m = min(rst.size, n - p)
            out[p:p + m] = rst[:m]
            self.pending = rst[m:]
            if idx.size:
                self.n_lost += int(np.count_nonzero((idx >= p) & (idx < p + m)))
            pos, level = p + m, self.v_low
        self.level = level
        if self.polarity < 0:
            out = -out
        if self.noise > 0:
            out += self.rng.normal(0.0, self.noise, n)
        return out


class EDSPhantomSpecimen(PhantomSpecimen):
    """Image + SDD phantom (see module docstring).

    Parameters (all optional, also accepted from ``DAQConfig.eds_phantom``):

    ``cps``          photon rate on the brightest phase (counts/s at the detector)
    ``beam_kv``      beam voltage of the X-ray model
    ``mv_per_kev``   SDD step height (6 mV/keV suits the DP5 on the NI simulator)
    ``leak_v_per_s``, ``noise_mv``, ``rise_s``  preamp details
    ``fano``         Fano factor of the step heights on the SDD output (0 = ideal steps)
    ``n``            label map size
    """

    def __init__(self, extent_v: float = 5.0, noise_v: float = 0.0, cps: float = 20e3, beam_kv: float = 20.0,
                 mv_per_kev: float = 6.0, leak_v_per_s: float = 100.0, noise_mv: float = 0.2,
                 rise_s: float = 0.5e-6, n: int = 1024, phases=None, labels: np.ndarray | None = None,
                 seed: int | None = 0, takeoff_deg: float = 35.0, fano: float = 0.115, **_ignored):
        self.labels = np.ascontiguousarray(_default_labels(n) if labels is None else labels, dtype=np.uint8)
        self.phases = tuple(phases) if phases is not None else default_phases()
        if int(self.labels.max()) >= len(self.phases):
            raise ValueError("label map refers to undefined phases")
        img = self._se_image()
        super().__init__(img, extent_v=extent_v, low_v=0.0, high_v=4.0, blank_v=0.0, noise_v=noise_v, seed=seed)
        self.cps = float(cps)
        self.beam_kv = float(beam_kv)
        self.takeoff_deg = float(takeoff_deg)
        self.preamp = ResetPreamp(mv_per_kev=mv_per_kev, leak_v_per_s=leak_v_per_s, noise_mv=noise_mv,
                                  rise_s=rise_s, seed=None if seed is None else seed + 1)
        self._rng_x = np.random.default_rng(None if seed is None else seed + 2)
        self.fano = float(fano)
        self._build_emission()
        self._listeners: list = []
        self._llock = threading.Lock()
        self.clock_s = 0.0  # stream time seen by the detector (s), across streams
        self.photons_emitted = 0

    # ---------------------------------------------------------------- image
    def _se_image(self) -> np.ndarray:
        lab = self.labels
        se = np.array([p.se_yield for p in self.phases], np.float32)[lab]
        # bright edges (SE edge effect) and a little texture
        edge = np.zeros(lab.shape, bool)
        edge[1:, :] |= lab[1:, :] != lab[:-1, :]
        edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        e = edge.astype(np.float32)
        for _ in range(2):  # widen to ~3 px
            e = np.maximum.reduce([e, np.roll(e, 1, 0), np.roll(e, -1, 0), np.roll(e, 1, 1), np.roll(e, -1, 1)])
        tex = np.random.default_rng(11).normal(0.0, 0.025, lab.shape).astype(np.float32)
        return np.clip(se + 0.18 * e + tex, 0.0, 1.0)

    # ---------------------------------------------------------------- X-rays
    def _build_emission(self) -> None:
        from ..xray import generation as gen

        cond = gen.Conditions(beam_kv=self.beam_kv, takeoff_deg=self.takeoff_deg)
        de = 0.01
        grid = np.arange(0.1 + de / 2, self.beam_kv, de)
        self._emit_e, self._emit_cdf, yields = [], [], []
        self.phase_lines = []
        for ph in self.phases:
            lines = gen.characteristic(ph.comp, cond)
            le = np.array([ln.energy for ln, _ in lines])
            li = np.array([i for _, i in lines])
            ci = gen.continuum(grid, ph.comp, cond) * de
            e = np.concatenate([le, grid])
            w = np.concatenate([li, ci])
            total = float(w.sum())
            yields.append(total)
            self._emit_e.append(e)
            self._emit_cdf.append(np.cumsum(w) / total)
            self.phase_lines.append(lines)
        y = np.array(yields)
        self._yield_rel = y / y.max()
        self.phase_rates = self.cps * self._yield_rel  # counts/s on each phase
        self._is_line = [np.arange(len(e)) < len(ls) for e, ls in zip(self._emit_e, self.phase_lines)]

    def set_cps(self, cps: float) -> None:
        """Photon rate on the brightest phase (≈ beam current); takes effect at the next chunk."""
        if not (np.isfinite(cps) and cps >= 0):
            raise ValueError("cps must be ≥ 0")
        self.cps = float(cps)
        self.phase_rates = self.cps * self._yield_rel

    def phase_spectrum(self, phase: int, energies: np.ndarray) -> np.ndarray:
        """Expected photon density (counts/s/keV) of *phase* before detector resolution (lines as spikes)."""
        h = np.zeros_like(energies, dtype=float)
        e, cdf = self._emit_e[phase], self._emit_cdf[phase]
        p = np.diff(np.concatenate([[0.0], cdf])) * self.phase_rates[phase]
        de = np.gradient(energies)
        idx = np.clip(np.searchsorted(energies, e) - 1, 0, len(energies) - 1)
        np.add.at(h, idx, p)
        return h / de

    def phase_at(self, x_v, y_v) -> np.ndarray:
        """Phase label at commanded positions (nearest pixel; outside → background 0)."""
        x_v = np.asarray(x_v, dtype=np.float64)
        y_v = np.asarray(y_v, dtype=np.float64)
        L = self.extent_v
        ny, nx = self.labels.shape
        col = np.floor((x_v + L) * (nx / (2 * L))).astype(np.int64)
        row = np.floor((L - y_v) * (ny / (2 * L))).astype(np.int64)
        inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        flat = self.labels.ravel()
        return np.where(inside, flat[np.where(inside, row * nx + col, 0)], 0)

    def phase_image(self, config, calibration) -> np.ndarray:
        """Phase labels a raster scan with *config* should see (pixel centres)."""
        nx, ny = config.nx, config.ny
        pw, ph = config.pixel_size_um
        u = -config.width_um / 2 + (np.arange(nx) + 0.5) * pw
        v = config.height / 2 - (np.arange(ny) + 0.5) * ph
        uu, vv = np.meshgrid(u, v)
        c, s = np.cos(config.rotation_rad), np.sin(config.rotation_rad)
        x = config.center_um[0] + c * uu - s * vv
        y = config.center_um[1] + s * uu + c * vv
        volts = calibration.um_to_volts(np.column_stack([x.ravel(), y.ravel()]), check=False)
        return self.phase_at(volts[:, 0], volts[:, 1]).reshape(ny, nx)

    # ---------------------------------------------------------------- photons
    def add_photon_listener(self, fn) -> None:
        with self._llock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def remove_photon_listener(self, fn) -> None:
        with self._llock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def photons(self, x_v: np.ndarray, y_v: np.ndarray, blank: np.ndarray, fs: float):
        """Photon events of one chunk: (sample index, energy keV).  Advances the clock, notifies listeners."""
        n = len(x_v)
        dt = 1.0 / fs
        lab = self.phase_at(x_v, y_v)
        rate = self.phase_rates[lab]
        rate = np.where(blank, 0.0, rate)
        k = self._rng_x.poisson(rate * dt)
        tot = int(k.sum())
        if tot:
            idx = np.repeat(np.arange(n), k)
            ph = lab[idx]
            e = np.empty(tot)
            for p in np.unique(ph):
                m = ph == p
                cdf = self._emit_cdf[p]
                j = np.minimum(np.searchsorted(cdf, self._rng_x.random(int(m.sum()))), cdf.size - 1)
                ee = self._emit_e[p][j]
                # continuum bins: spread uniformly over the 10 eV bin
                cont = ~self._is_line[p][j]
                ee = ee + np.where(cont, self._rng_x.uniform(-0.005, 0.005, ee.size), 0.0)
                e[m] = ee
        else:
            idx = np.zeros(0, np.int64)
            e = np.zeros(0)
        t0 = self.clock_s
        self.clock_s = t0 + n * dt
        self.photons_emitted += tot
        with self._llock:
            ls = list(self._listeners)
        for fn in ls:
            try:
                fn(t0, dt, idx, e, n)
            except Exception:  # noqa: BLE001 - a listener must never break the stream
                pass
        return idx, e

    def sdd_output(self, x_v: np.ndarray, y_v: np.ndarray, blank: np.ndarray, fs: float) -> np.ndarray:
        """Per-sample SDD preamp output (V) for one chunk (what goes to the SDD aux output)."""
        idx, e = self.photons(x_v, y_v, blank, fs)
        if self.fano > 0 and e.size:
            # charge statistics in the silicon: the step height (collected charge) fluctuates
            # with the Fano factor, as on a real SDD.  (Photon listeners get the true energy –
            # the simulated pulse processor applies the Fano + noise resolution itself.)
            e = e + self._rng_x.normal(0.0, 1.0, e.size) * np.sqrt(self.fano * 3.85e-3 * e)
        return self.preamp.waveform(len(x_v), fs, idx, e)
