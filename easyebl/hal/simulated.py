"""A physics-flavoured simulator standing in for DAQ + SEM.

The simulator closes the loop so every subsystem can be exercised without
hardware:

* deflection voltages → true beam position (µm) through a configurable *true*
  field calibration (scale/rotation/offset errors);
* first-order **deflection lag** (coil + driver), carried across chunks;
* **blanker delay** and leakage;
* **video signal** sampled from a :class:`SampleModel` at the *actual*
  (lagged) beam position, with detector delay, bandwidth and noise;
* **dose map**: every unblanked sample deposits ``I·dt`` of charge at the
  actual beam position, so lithography results can be checked numerically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import threading
import time
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage, signal

from ..core.transforms import AffineTransform
from ..core.waveform import OutputChunk
from .backend import AIConsumer, BackendCapabilities, DAQBackend, StreamResult
from .config import DAQConfig

__all__ = [
    "SampleModel",
    "ImageSample",
    "FunctionSample",
    "DoseMap",
    "SimParams",
    "SimulatedBackend",
    "make_demo_sample",
    "default_demo_sample",
    "render_cross",
]


# --------------------------------------------------------------------- samples
class SampleModel:
    """Secondary-electron yield as a function of true field position (µm)."""

    def intensity(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class FunctionSample(SampleModel):
    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.fn = fn

    def intensity(self, x_um, y_um):
        return np.asarray(self.fn(x_um, y_um), dtype=float)


class ImageSample(SampleModel):
    """Bilinear interpolation of an image covering ``extent`` = (x0, x1, y0, y1) µm.

    ``image[0, 0]`` is the top-left corner (x0, y1).
    """

    def __init__(self, image: np.ndarray, extent: tuple, background: float = 0.0):
        self.image = np.asarray(image, dtype=float)
        self.extent = tuple(float(v) for v in extent)
        self.background = background

    def intensity(self, x_um, y_um):
        x0, x1, y0, y1 = self.extent
        ny, nx = self.image.shape
        col = (np.asarray(x_um) - x0) / (x1 - x0) * nx - 0.5
        row = (y1 - np.asarray(y_um)) / (y1 - y0) * ny - 0.5
        return ndimage.map_coordinates(
            self.image, [row.ravel(), col.ravel()], order=1, mode="constant", cval=self.background
        ).reshape(np.shape(x_um))


def _coverage(c: np.ndarray, lo: float, hi: float, pix: float) -> np.ndarray:
    """Fraction of pixel [c-pix/2, c+pix/2] covered by the interval [lo, hi]."""
    return np.clip(np.minimum(c + pix / 2, hi) - np.maximum(c - pix / 2, lo), 0.0, pix) / pix


def render_cross(
    xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, size: float, width: float, pixel: float | None = None
) -> np.ndarray:
    """Area-coverage (anti-aliased) cross on a regular coordinate grid.

    Sub-pixel mark positions are therefore represented faithfully, which the
    alignment tests rely on.
    """
    if pixel is None:
        pixel = float(abs(xx[0, 1] - xx[0, 0])) if xx.shape[1] > 1 else 1.0
    x, y = xx - cx, yy - cy
    h = _coverage(x, -size / 2, size / 2, pixel) * _coverage(y, -width / 2, width / 2, pixel)
    v = _coverage(x, -width / 2, width / 2, pixel) * _coverage(y, -size / 2, size / 2, pixel)
    return np.clip(h + v - _coverage(x, -width / 2, width / 2, pixel) * _coverage(y, -width / 2, width / 2, pixel), 0, 1)


def make_demo_sample(
    extent_um: float = 120.0,
    pixels: int = 1200,
    marks: Sequence[tuple] = ((-35.0, -35.0), (35.0, -35.0), (35.0, 35.0), (-35.0, 35.0)),
    mark_size: float = 8.0,
    mark_width: float = 1.0,
    seed: int = 1,
    particles: int = 60,
    grating: bool = True,
) -> ImageSample:
    """A synthetic specimen: Au-like alignment crosses, particles, a grating."""
    rng = np.random.default_rng(seed)
    half = extent_um / 2
    ax = (np.arange(pixels) + 0.5) / pixels * extent_um - half
    xx, yy = np.meshgrid(ax, ax[::-1])
    img = 0.15 + 0.05 * ndimage.gaussian_filter(rng.standard_normal((pixels, pixels)), 6) * 6
    for (mx, my) in marks:
        img += 0.7 * render_cross(xx, yy, mx, my, mark_size, mark_width, extent_um / pixels)
    for _ in range(particles):
        px, py = rng.uniform(-half * 0.9, half * 0.9, 2)
        r = rng.uniform(0.2, 1.2)
        img += 0.4 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * r * r))
    if grating:
        g = (np.abs(xx) < 15) & (np.abs(yy - 5) < 6)
        img += 0.35 * g * (np.sin(2 * np.pi * xx / 2.0) > 0)
    img = ndimage.gaussian_filter(img, 0.8)
    return ImageSample(img, (-half, half, -half, half), background=0.15)


@functools.lru_cache(maxsize=1)
def default_demo_sample() -> ImageSample:
    """Shared (read-only) default specimen – built once per process."""
    return make_demo_sample()


# ------------------------------------------------------------------- dose map
class DoseMap:
    """Accumulated charge on a regular grid (row 0 = top)."""

    def __init__(self, extent_um: float = 110.0, pixel_um: float = 0.1):
        self.extent_um = float(extent_um)
        self.pixel_um = float(pixel_um)
        self.n = int(round(self.extent_um / self.pixel_um))
        self.charge_c = np.zeros((self.n, self.n))

    @property
    def extent(self) -> tuple:
        h = self.n * self.pixel_um / 2
        return (-h, h, -h, h)

    def reset(self) -> None:
        self.charge_c[:] = 0.0

    def deposit(self, x_um: np.ndarray, y_um: np.ndarray, charge_c) -> None:
        h = self.n * self.pixel_um / 2
        col = np.floor((np.asarray(x_um) + h) / self.pixel_um).astype(np.int64)
        row = np.floor((h - np.asarray(y_um)) / self.pixel_um).astype(np.int64)
        ok = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        if not np.any(ok):
            return
        q = np.broadcast_to(np.asarray(charge_c, dtype=float), np.shape(x_um))[ok]
        idx = row[ok] * self.n + col[ok]
        self.charge_c.ravel()[:] += np.bincount(idx, weights=q, minlength=self.n * self.n)

    def dose_uC_cm2(self) -> np.ndarray:
        area_cm2 = (self.pixel_um * 1e-4) ** 2
        return self.charge_c / area_cm2 * 1e6

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates (x along columns, y along rows)."""
        h = self.n * self.pixel_um / 2
        c = -h + (np.arange(self.n) + 0.5) * self.pixel_um
        return c, c[::-1]

    def total_charge(self) -> float:
        return float(self.charge_c.sum())


# ------------------------------------------------------------------ parameters
@dataclass
class SimParams:
    """Physical parameters of the simulated SEM."""

    field_size_um: float = 100.0  # true field at ±v_full_scale
    v_full_scale: float = 10.0
    distortion: AffineTransform = field(default_factory=AffineTransform.identity)
    lag_tau_s: float = 4e-6  # deflection first-order lag
    blank_delay_s: float = 0.0  # blanker responds this much after the DO edge
    blank_leakage: float = 0.0  # fraction of current reaching sample when blanked
    video_delay_s: float = 1e-6  # detector/amplifier delay
    video_tau_s: float = 0.0  # detector bandwidth (first order)
    beam_current_a: float = 100e-12
    noise_sigma: float = 0.02  # volts rms on the video signal
    video_gain_v: float = 4.0  # volts per unit SE yield
    video_offset_v: float = 0.0
    dose_extent_um: float = 110.0
    dose_pixel_um: float = 0.1
    realtime: bool = False  # sleep to emulate the real duration
    seed: int | None = 0


# --------------------------------------------------------------------- backend
class SimulatedBackend(DAQBackend):
    """Drop-in replacement for :class:`NIDAQBackend`."""

    def __init__(
        self,
        config: DAQConfig | None = None,
        params: SimParams | None = None,
        sample: SampleModel | None = None,
        record: bool = False,
    ):
        super().__init__(config or DAQConfig())
        self.params = params or SimParams()
        self.sample = sample if sample is not None else default_demo_sample()
        self.dose = DoseMap(self.params.dose_extent_um, self.params.dose_pixel_um)
        self.record = record
        self.recorded: list[OutputChunk] = []
        self.static_lines: dict[str, bool] = {}
        self.static_xyb = (0.0, 0.0, True)
        self._static_t = time.perf_counter()
        self._rng = np.random.default_rng(self.params.seed)
        self._lock = threading.Lock()
        self.reset_state()

    # ------------------------------------------------------------- lifecycle
    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._deposit_static()
        self._open = False

    def capabilities(self) -> BackendCapabilities:
        c = self.config
        return BackendCapabilities(
            name="simulator",
            max_ao_rate=c.max_ao_rate,
            max_ai_aggregate_rate=c.max_ai_aggregate_rate,
            ai_channels=tuple(c.ai_channels),
            timed_lines=tuple(c.timed_lines),
            static_lines=tuple(c.static_lines),
            simulated=True,
        )

    def reset_state(self) -> None:
        """Reset beam filter states (beam at park) – call between unrelated tests."""
        p0 = self.volts_to_true_um(np.array([[self.config.park_x_v, self.config.park_y_v]]))[0]
        self._beam = p0.copy()
        self._video_state = 0.0
        self._blank_hist = np.ones(0, dtype=bool)
        self._video_hist = np.zeros((0,))

    # --------------------------------------------------------------- physics
    def volts_to_true_um(self, v: np.ndarray) -> np.ndarray:
        p = self.params
        nominal = np.asarray(v, dtype=float) * (p.field_size_um / (2 * p.v_full_scale))
        return p.distortion.apply(nominal)

    def true_um_to_volts(self, xy: np.ndarray) -> np.ndarray:
        p = self.params
        nominal = p.distortion.inverse().apply(xy)
        return nominal / (p.field_size_um / (2 * p.v_full_scale))

    def _deposit_static(self) -> None:
        now = time.perf_counter()
        x, y, b = self.static_xyb
        frac = self.params.blank_leakage if b else 1.0
        if frac > 0 and self.is_open:
            q = self.params.beam_current_a * (now - self._static_t) * frac
            pos = self.volts_to_true_um(np.array([[x, y]]))[0]
            self.dose.deposit(np.array([pos[0]]), np.array([pos[1]]), q)
        self._static_t = now

    # ----------------------------------------------------------------- static
    def write_static(self, x_v: float, y_v: float, blank: bool) -> None:
        with self._lock:
            self._deposit_static()
            self.static_xyb = (float(x_v), float(y_v), bool(blank))
            self._beam = self.volts_to_true_um(np.array([[x_v, y_v]]))[0]

    def set_static_line(self, name: str, value: bool) -> None:
        if name not in self.config.static_lines:
            raise KeyError(name)
        self.static_lines[name] = bool(value)

    # ----------------------------------------------------------------- stream
    def _process(self, c: OutputChunk, rate: float, n_ai: int) -> np.ndarray:
        p = self.params
        dt = 1.0 / rate
        n = len(c)
        cmd = self.volts_to_true_um(np.column_stack([c.x, c.y]))
        # deflection lag (first-order IIR with carried state)
        if p.lag_tau_s > 0:
            a = 1.0 - np.exp(-dt / p.lag_tau_s)
            b_, a_ = [a], [1.0, -(1.0 - a)]
            pos = np.empty_like(cmd)
            for k in range(2):
                pos[:, k], _ = signal.lfilter(b_, a_, cmd[:, k], zi=[(1.0 - a) * self._beam[k]])
        else:
            pos = cmd
        self._beam = pos[-1].copy()
        # blanker delay
        d_b = int(round(p.blank_delay_s * rate))
        hist = np.concatenate([self._blank_hist, c.blank])
        if d_b > 0:
            if len(hist) < n + d_b:
                hist = np.concatenate([np.ones(n + d_b - len(hist), bool), hist])
            blank_eff = hist[len(hist) - n - d_b : len(hist) - d_b]
            self._blank_hist = hist[-d_b:]
        else:
            blank_eff = c.blank
        # dose
        frac = np.where(blank_eff, p.blank_leakage, 1.0)
        if np.any(frac > 0):
            m = frac > 0
            self.dose.deposit(pos[m, 0], pos[m, 1], p.beam_current_a * dt * frac[m])
        # video
        if n_ai == 0:
            return np.zeros((0, n))
        sig = self.sample.intensity(pos[:, 0], pos[:, 1]) * p.video_gain_v
        sig = np.where(blank_eff, 0.0, sig)
        if p.video_tau_s > 0:
            a = 1.0 - np.exp(-dt / p.video_tau_s)
            sig, zf = signal.lfilter([a], [1.0, -(1.0 - a)], sig, zi=[(1.0 - a) * self._video_state])
            self._video_state = sig[-1]
        d_v = int(round(p.video_delay_s * rate))
        if d_v > 0:
            vh = np.concatenate([self._video_hist, sig])
            if len(vh) < n + d_v:
                vh = np.concatenate([np.zeros(n + d_v - len(vh)), vh])
            sig = vh[len(vh) - n - d_v : len(vh) - d_v]
            self._video_hist = vh[-d_v:]
        out = np.empty((n_ai, n))
        for i in range(n_ai):
            s = sig if i == 0 else (p.video_gain_v * 0.5 - 0.5 * sig)
            out[i] = s + p.video_offset_v + self._rng.normal(0.0, p.noise_sigma, n)
        lim = self.config.ai_range_v
        return np.clip(out, -lim, lim)

    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        abort: threading.Event | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> StreamResult:
        n_ai = len(ai_channels)
        written = 0
        collected = []
        aborted = False
        t0 = time.perf_counter()
        with self._lock:
            self._deposit_static()
            try:
                for c in chunks:
                    if abort is not None and abort.is_set():
                        aborted = True
                        break
                    if self.record:
                        self.recorded.append(c)
                    ai = self._process(c, rate, n_ai)
                    written += len(c)
                    if n_ai:
                        if on_ai is not None:
                            on_ai(ai)
                        else:
                            collected.append(ai)
                    if progress is not None:
                        progress(written)
                    if self.params.realtime:
                        lag = written / rate - (time.perf_counter() - t0)
                        if lag > 0:
                            time.sleep(lag)
            finally:
                self._static_t = time.perf_counter()
        ai_all = None
        if n_ai and on_ai is None:
            ai_all = np.concatenate(collected, axis=1) if collected else np.zeros((n_ai, 0))
        return StreamResult(
            samples_written=written,
            rate=rate,
            aborted=aborted,
            ai=ai_all,
            ai_channels=tuple(ai_channels),
            elapsed_s=time.perf_counter() - t0,
        )

    def write_safe(self) -> None:  # pragma: no cover - alias
        self.safe_state()
