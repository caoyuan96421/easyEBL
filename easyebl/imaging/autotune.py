"""Auto focus and auto stigmation from our own images (S-3000 over its remote control).

The SEM's own AFC / ASF routines are unreliable; these search the objective focus and the
stigmator with small, fast test images from the imaging engine and a noise-robust sharpness
measure, and fall back to the values they started from when they do not find a clear optimum.

Cost model (S-3000, measured 2026-11-25): every RS-232 transaction takes ≈ 0.25 s, and the SEM
reads back a new focus / stigmator value at once.  One step = one setting with its readback
(:class:`~easyebl.microscope.sem.SEMControl` confirms it: ≈ 0.5 s) + one test image (by default
Slow 1 at 256²: ≈ 0.35 s; :class:`AutoTuneOptions`, the SEM window's Advanced settings).  The background reads are held for the duration, so no step waits behind one.

* **Near the present value first** (:meth:`_Ctx.local`) – focus and stigmator: the start value and
  two steps either side; a peak there is refined, a clear trend (best / worst ≥ ``local_contrast``)
  is walked with doubling steps until the sharpness falls (at most ``local_range`` steps), then
  refined – halving down to the depth of focus (:meth:`AutoTuner.focus_tolerance_counts`; the
  stigmator: 16 counts) and a parabola.  No peak or trend nearby: the whole range, as below.
* **Focus** – one knob at a time.  *Fine* moves FOCUS F only, over its whole range: a grid of
  512 counts (9 images), then the step halved around the sharpest value – two images per halving
  – down to the depth of focus; the sharpest value measured is kept.  At high magnification the grid is finer, at most 4 focus steps (one step blurs the test
  image by ≈ ``blur_px`` pixels, ``Δz = blur_px · pixel / convergence`` µm of WD, turned into focus
  counts with the focus table), so it cannot step over the focus curve: 128 counts at ×10k.
  ≈ 27 images (≈ 23 s) up to ≈ ×5k, more above.
  *Coarse* moves FOCUS C only, and carefully (the image is very sensitive to it): never more than
  ``coarse_step_mm`` (0.1 mm of WD) per setting – on the way back and when the start value is set
  again too – and never beyond ``coarse_range_mm`` (±1 mm of WD) from where it started.  It sweeps
  in 0.1 mm steps one way until the sharpness has clearly fallen past its peak, the other way if
  the peak is not bracketed, then fits the peak.  No time limit: it runs when the fine focus cannot
  find the focus.  *Auto*: fine; if the whole FOCUS F range has no clear optimum (or its peak is
  at an end), the fine focus is set back, the coarse sweep runs, then fine again.
* **Stigmation** – when nothing is found nearby: full-range golden-section search on X (Y fixed), then on Y: each round keeps
  0.62 of the bracket and needs one new image; after 6 rounds a parabola through the best three
  gives the value.  ≈ 9–10 steps per axis (≈ 16 s for both).

A result is kept only when it is sharper than the start; otherwise – and on any error, a
readback that does not match, an image without structure or a peak at the end of the range, and
on cancel – the starting values are set again.

Hooks: :class:`AutoTuner` is shared by the backend (``bundle.autotune``), the SEM window and the
lithography hardware layer; :meth:`AutoTuner.auto_focus` / :meth:`auto_stigmate` /
:meth:`focus_and_stigmate` return futures, ``run_*`` block.
"""
from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

__all__ = ["AutoTuneOptions", "AutoTuneResult", "AutoTuneCancelled", "AutoTuneFailed", "AutoTuner", "sharpness",
           "noise_sigma",
           "fit_log_parabola"]

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618


@dataclass(frozen=True)
class AutoTuneOptions:
    # test images (square, over the present scan field): a scan speed preset and a size, one for the
    # focus and one for the stigmator searches (Slow 1 at 256²: ≈ 0.35 s per image)
    focus_speed: str = "Slow 1"
    focus_px: int = 256
    stig_speed: str = "Slow 1"
    stig_px: int = 256
    settle_s: float = 0.0  # extra wait after a confirmed setting (if the lens lags the readback)
    convergence_rad: float = 5e-3  # beam convergence semi-angle (blur radius = α · defocus)
    blur_px: float = 2.0  # blur per focus step (test-image pixels)
    coarse_step_mm: float = 0.1  # FOCUS C: at most this change of WD per setting
    coarse_range_mm: float = 1.0  # FOCUS C: never beyond ± this WD from the start
    coarse_drop: float = 0.7  # sweeps: past the peak after 2 steps below this × the best
    fine_initial_step: int = 512  # FOCUS F: the first grid over the whole range (then halved); at
    # high magnification smaller: at most 4 focus steps, so the grid cannot step over the peak
    stig_rounds: int = 6  # golden-section rounds per axis (then a parabola)
    stig_min_step: int = 32  # stigmator values at least this far apart (a single write; see SEMControl)
    min_contrast: float = 1.30  # best / worst sharpness of a search below this: no structure
    min_sharpness: float = 0.0  # best sharpness at or below this: no visible detail (all blurred / noise)
    keep_margin: float = 0.0  # a result is kept when sharpness ≥ start · (1 + margin)
    # local search first (focus and stigmator): the start value and two steps each side; a peak or a
    # clear trend there (best / worst ≥ local_contrast) is followed – walking with doubling steps at
    # most local_range steps from the start – else the whole range is searched
    local_contrast: float = 1.10
    local_range: int = 16
    stig_local_step: int = 64  # stigmator counts per probe step (the result to 16 counts, its step)
    timeout_s: float = 10.0  # per setting / per image


@dataclass
class AutoTuneResult:
    kind: str  # "focus (fine)", "focus (coarse)", "stigmation", ...
    ok: bool
    message: str
    start: dict  # values before
    final: dict  # values after (the start values again when not ok)
    score_start: float | None = None
    score_final: float | None = None
    samples: list = field(default_factory=list)  # (knob, value, score) in the order measured
    steps: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        gain = (f", sharpness ×{self.score_final / self.score_start:.2f}"
                if self.ok and self.score_start and self.score_final else "")
        vals = ", ".join(f"{k} {self.start.get(k)}→{v}" for k, v in self.final.items() if self.start.get(k) != v)
        return (f"{self.kind}: {'done' if self.ok else 'not changed'} – {self.message}"
                + (f" ({vals})" if vals and self.ok else "") + gain
                + f" [{self.steps} steps, {self.elapsed_s:.1f} s]")


class AutoTuneCancelled(RuntimeError):
    pass


class AutoTuneFailed(RuntimeError):
    pass


# ======================================================================= sharpness
_LAPLACE = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def noise_sigma(image: np.ndarray) -> float:
    """White-noise rms of an image, from the median absolute second difference (the kernel
    removes smooth structure and the median ignores edges; white noise of rms σ gives 6σ)."""
    from scipy import ndimage

    lap = ndimage.convolve(np.asarray(image, dtype=np.float64), _LAPLACE, mode="reflect")[1:-1, 1:-1]
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))) / 6.0)


_DIRECTIONS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
_NOISE_GAIN: dict = {}


def _noise_gain(smooth_px: float) -> list:
    """Mean squared response of each directional gradient (Gaussian *smooth_px*, then Sobel) to
    white noise of unit rms: the sum of the squared impulse response."""
    g = _NOISE_GAIN.get(smooth_px)
    if g is None:
        from scipy import ndimage

        n = 2 * int(4 * smooth_px + 4) + 1
        d = np.zeros((n, n))
        d[n // 2, n // 2] = 1.0
        sm = ndimage.gaussian_filter(d, smooth_px, mode="constant")
        gx = ndimage.sobel(sm, axis=1, mode="constant")
        gy = ndimage.sobel(sm, axis=0, mode="constant")
        g = [float(np.sum((gx * math.cos(t) + gy * math.sin(t)) ** 2)) for t in _DIRECTIONS]
        _NOISE_GAIN[smooth_px] = g
    return g


def sharpness(image: np.ndarray, smooth_px: float = 1.0, detail_snr: float = 0.5) -> float:
    """Gradient-based sharpness of an image (larger = sharper; 0: no detail above the noise).

    Tenengrad on a lightly de-noised image (Gaussian *smooth_px*, then Sobel), per direction (0°,
    45°, 90°, 135°): the mean squared directional gradient, less what the detector's white noise
    contributes (its rms measured in the image itself, :func:`noise_sigma`).  The four are averaged
    harmonically (dominated by the most blurred direction; ≈ 1 / the mean squared blur over the
    directions), so a line focus – sharp one way, blurred the other – scores below the round spot
    between the two line foci (the focus finds the circle of least confusion, and the stigmator
    search sees the astigmatism), and divided by the variance of the heavily smoothed image (the
    gross contrast, which defocus hardly changes), so brightness and contrast cancel.  0 when the
    gradients stand less than *detail_snr* × the noise's above it."""
    from scipy import ndimage

    img = np.asarray(image, dtype=np.float64)
    sig = noise_sigma(img)
    sm = ndimage.gaussian_filter(img, smooth_px, mode="reflect")
    gx = ndimage.sobel(sm, axis=1, mode="reflect")
    gy = ndimage.sobel(sm, axis=0, mode="reflect")
    b = int(2 * smooth_px + 2)  # leave out the border (the filters' reflection)
    if min(img.shape) <= 4 * b:
        return 0.0
    gx, gy = gx[b:-b, b:-b], gy[b:-b, b:-b]
    noise = [sig * sig * k for k in _noise_gain(float(smooth_px))]
    energy = [float(np.mean((gx * math.cos(t) + gy * math.sin(t)) ** 2)) for t in _DIRECTIONS]
    signal = [e - n for e, n in zip(energy, noise)]
    if sum(signal) <= detail_snr * sum(noise) or sum(signal) <= 0:
        return 0.0
    contrast = float(np.var(ndimage.gaussian_filter(img, 8.0, mode="reflect")[b:-b, b:-b]))
    if contrast <= 0:
        return 0.0
    floor = [max(1e-3 * n, 1e-30) for n in noise]
    h = len(signal) / sum(1.0 / max(v, f) for v, f in zip(signal, floor))  # harmonic mean
    return h / contrast


def fit_log_parabola(xs, ys) -> float | None:
    """Vertex of the parabola through log(y) (least squares, ≥ 3 points); None unless it opens
    downwards."""
    xs = np.asarray(xs, float)
    ys = np.log(np.maximum(np.asarray(ys, float), 1e-12))
    if len(xs) < 3 or np.ptp(xs) == 0:
        return None
    x0, sx = xs.mean(), max(np.ptp(xs), 1e-9)
    a, b, _c = np.polyfit((xs - x0) / sx, ys, 2)
    if a >= 0:
        return None
    return float(x0 + sx * (-b / (2 * a)))


def image_time_s(speed: str, px: int, rate: float) -> float:
    """Time (s) of one *px* × *px* image at the scan speed preset *speed* (sample clock *rate*)."""
    from .engine import speed_by_name

    f = speed_by_name(speed).fields()
    dwell = max(f["dwell_s"], 1.0 / rate)
    return px * (px * dwell + f.get("preline_s", 0.0) + f.get("flyback_s", 0.0)) + f.get("frame_settle_s", 0.0)


# ======================================================================= tuner
class AutoTuner:
    """Auto focus / stigmation on *engine* (test images) and *sem* (a SEMControl: ``set_focus``,
    ``set_stigmator``, ``state``, ``hold_polling``).  One procedure at a time, on its own thread."""

    def __init__(self, engine, sem, options: AutoTuneOptions | None = None):
        self.engine = engine
        self.sem = sem
        self.options = options or AutoTuneOptions()
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="easyebl-autotune")
        self._lock = threading.Lock()
        self._busy = False
        self._cancel = threading.Event()
        self._listeners: list[Callable[[dict], None]] = []
        self.last_result: AutoTuneResult | None = None

    # ------------------------------------------------------------------ public
    @property
    def available(self) -> bool:
        return all(hasattr(self.sem, a) for a in ("set_focus", "set_stigmator", "state")) and \
            bool(getattr(self.sem, "connected", True))

    @property
    def busy(self) -> bool:
        return self._busy

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        """``fn(event)`` on the tuner's thread: ``{"type": "progress"|"done", "text", ...}``."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def cancel(self) -> None:
        self._cancel.set()

    def auto_focus(self, mode: str = "auto") -> concurrent.futures.Future:
        """*mode*: ``"auto"`` (coarse first only when needed), ``"fine"`` or ``"coarse"``."""
        return self._submit(lambda: self.run_focus(mode))

    def auto_stigmate(self) -> concurrent.futures.Future:
        return self._submit(self.run_stigmate)

    def focus_and_stigmate(self) -> concurrent.futures.Future:
        return self._submit(self.run_focus_and_stigmate)

    def close(self) -> None:
        self.cancel()
        self._pool.shutdown(wait=False)

    def set_options(self, **changes) -> AutoTuneOptions:
        """Change :attr:`options` (e.g. ``focus_speed="Slow 2", stig_px=384``); the next run uses them."""
        from .engine import speed_by_name

        for k in ("focus_speed", "stig_speed"):
            if k in changes:
                try:
                    speed_by_name(changes[k])
                except KeyError:
                    raise ValueError(f"{k}: no scan speed {changes[k]!r}") from None
        for k in ("focus_px", "stig_px"):
            if k in changes and not 32 <= int(changes[k]) <= 4096:
                raise ValueError(f"{k} must be 32…4096")
        self.options = replace(self.options, **changes)
        return self.options

    def image_time_s(self, which: str = "focus") -> float:
        """Time of one test image (s) with the present options."""
        o = self.options
        return image_time_s(o.focus_speed if which == "focus" else o.stig_speed,
                                 o.focus_px if which == "focus" else o.stig_px, float(self.engine.rate))

    # blocking versions (any thread but the GUI's) ---------------------------------
    def run_focus(self, mode: str = "auto") -> AutoTuneResult:
        if mode not in ("auto", "fine", "coarse"):
            raise ValueError("mode must be auto, fine or coarse")
        return self._procedure(f"focus ({mode})", ("focus_coarse", "focus_fine"), lambda ctx: self._focus(ctx, mode))

    def run_stigmate(self) -> AutoTuneResult:
        return self._procedure("stigmation", ("stig_x", "stig_y"), self._stigmate)

    def run_focus_and_stigmate(self) -> AutoTuneResult:
        def body(ctx):
            self._focus(ctx, "auto")
            self._stigmate(ctx)
            self._focus(ctx, "fine")
        return self._procedure("focus + stigmation", ("focus_coarse", "focus_fine", "stig_x", "stig_y"), body)

    # ------------------------------------------------------------------ plumbing
    def _submit(self, fn) -> concurrent.futures.Future:
        if self._busy:
            raise RuntimeError("an auto focus / stigmation is already running")
        return self._pool.submit(fn)

    def _emit(self, **ev) -> None:
        for fn in list(self._listeners):
            try:
                fn(dict(ev))
            except Exception:  # noqa: BLE001 - a listener must not break the search
                pass

    def _procedure(self, kind: str, keys, body) -> AutoTuneResult:
        with self._lock:
            if self._busy:
                raise RuntimeError("an auto focus / stigmation is already running")
            self._busy = True
        self._cancel.clear()
        try:
            return self._procedure_body(kind, keys, body)
        finally:
            self._busy = False

    def _procedure_body(self, kind: str, keys, body) -> AutoTuneResult:
        t0 = time.perf_counter()
        start = {k: self.sem.state.get(k) for k in keys} if hasattr(self.sem, "state") else {}
        why = ("the microscope is not connected (no focus / stigmator control)" if not self.available else
               "focus / stigmator values not read yet" if any(not isinstance(v, int) for v in start.values())
               else None)
        if why is not None:  # nothing was changed
            res = AutoTuneResult(kind, False, why, start, dict(start))
            self.last_result = res
            self._emit(type="done", text=res.summary(), result=res)
            return res
        was_live = self._engine_live()
        hold = getattr(self.sem, "hold_polling", None)
        ext = getattr(self.engine, "hold_external_scan", None)  # claim the SEM once, not per image
        ctx = _Ctx(self, kind, start)
        ctx.image = "stig" if kind == "stigmation" else "focus"  # start and final scores: the same images
        first_image = ctx.image
        self._emit(type="start", text=f"{kind}: started", kind=kind)
        try:
            with (hold() if hold is not None else _nullcontext()), (ext() if ext is not None else _nullcontext()):
                try:
                    ctx.score_start = ctx.measure()
                    body(ctx)
                    ctx.use(first_image)
                    final_score = ctx.score_now if ctx.score_now is not None else ctx.measure()
                    keep = final_score >= ctx.score_start * (1.0 + self.options.keep_margin)
                    if not keep:
                        raise AutoTuneFailed(f"no improvement (sharpness {final_score:.3g} ≤ start {ctx.score_start:.3g})")
                    res = ctx.result(True, "sharper than before", final_score, t0)
                except Exception as e:  # noqa: BLE001 - restore the start values, then report
                    restore_err = ctx.restore()
                    if isinstance(e, AutoTuneCancelled):
                        msg = "cancelled"
                    elif isinstance(e, AutoTuneFailed):
                        msg = str(e)
                    else:
                        msg = f"{type(e).__name__}: {e}"
                    if restore_err:
                        msg += f"; restoring the start values failed: {restore_err}"
                    res = ctx.result(False, msg, None, t0)
        finally:
            if was_live:
                try:
                    self.engine.run()
                except Exception:  # noqa: BLE001
                    pass
        self.last_result = res
        self._emit(type="done", text=res.summary(), result=res)
        return res

    def _engine_live(self) -> bool:
        st = getattr(self.engine, "state", None)
        return getattr(st, "value", str(st)).lower() in ("live", "starting")

    # ------------------------------------------------------------------ step size
    def counts_per_mm(self) -> float:
        """Coarse focus counts per mm of WD at the present focus (the focus table's slope)."""
        from ..microscope.s3000.focus_wd import effective_focus, focus_for_wd, wd_from_focus

        c, f = self.sem.state.get("focus_coarse"), self.sem.state.get("focus_fine")
        wd = wd_from_focus(c, f) if isinstance(c, int) and isinstance(f, int) else 15.0
        wd = min(max(wd, 4.05), 59.95)
        e1 = effective_focus(*focus_for_wd(wd - 0.05))
        e2 = effective_focus(*focus_for_wd(wd + 0.05))
        return abs(e1 - e2) / 0.1

    def focus_step_counts(self) -> float:
        """Focus step (coarse counts) that blurs a test image by ≈ blur_px pixels at this field."""
        o = self.options
        pixel_um = float(self.engine.config.width_um) / o.focus_px
        dz_mm = o.blur_px * pixel_um / o.convergence_rad / 1000.0
        return dz_mm * self.counts_per_mm()

    def focus_tolerance_counts(self) -> float:
        """FOCUS F counts of the largest acceptable defocus: the defocus blur 2·α·Δz stays below the
        larger of the live image's pixel and the diffraction-limited probe (0.61 λ / α, λ of the
        accelerating voltage); Δz turned into counts with the focus table (WD calibration).  ≥ 1."""
        import math as _m

        from ..microscope.s3000.focus_wd import FINE_WEIGHT

        a = float(self.options.convergence_rad)
        try:
            px = float(self.engine.config.pixel_size_um[0])
        except Exception:  # noqa: BLE001
            px = 0.0
        kv = None
        try:
            kv = self.sem.accelerating_voltage_kv() if hasattr(self.sem, "accelerating_voltage_kv") else None
        except Exception:  # noqa: BLE001
            kv = None
        v = 1e3 * float(kv or getattr(self.sem, "state", {}).get("hv_kv") or 20.0)
        lam_um = 1.226 / _m.sqrt(v * (1.0 + 0.9785e-6 * v)) * 1e-3  # electron wavelength
        blur_um = max(px, 0.61 * lam_um / a)
        dz_mm = blur_um / (2.0 * a) / 1000.0
        return max(1.0, dz_mm * self.counts_per_mm() / FINE_WEIGHT)

    def coarse_step_counts(self) -> int:
        """The largest FOCUS C change per setting (``coarse_step_mm`` of WD), ≥ 1."""
        return max(1, int(round(self.options.coarse_step_mm * self.counts_per_mm())))

    # ------------------------------------------------------------------ searches
    def _focus(self, ctx: "_Ctx", mode: str) -> None:
        from ..microscope.s3000.focus_wd import FINE_WEIGHT

        ctx.use("focus")
        hf = min(600.0, max(4.0, self.focus_step_counts() / FINE_WEIGHT))
        if mode == "coarse":
            ctx.coarse_sweep()
            return
        tol = self.focus_tolerance_counts()
        ctx.focus_tol = tol
        x = ctx.local("focus_fine", max(4, int(round(hf))), max(1, int(tol)), 1)  # near the present value
        if x is not None:
            return
        self._emit(type="progress", text=f"{ctx.kind}: no peak or trend near FOCUS F {ctx.start['focus_fine']} – "
                                         "the whole range")
        try:
            ctx.fine_focus(hf)  # the whole FOCUS F range: 512-count grid, then halving
            return
        except AutoTuneFailed as e:
            if mode != "auto":
                raise
            self._emit(type="progress", text=f"{ctx.kind}: no focus within FOCUS F ({e}) – coarse focus sweep")
        ctx.set("focus_fine", ctx.start["focus_fine"])  # back to where the fine search started
        ctx.coarse_sweep()
        ctx.fine_focus(hf)

    def _stigmate(self, ctx: "_Ctx") -> None:
        ctx.use("stig")
        for key in ("stig_x", "stig_y"):
            q = 16  # the stigmator's step
            if ctx.local(key, max(q, int(self.options.stig_local_step) // q * q), q, q) is None:
                self._emit(type="progress", text=f"{ctx.kind}: no peak or trend near {key.replace('_', ' ')} "
                                                 f"{ctx.cur[key]} – the whole range")
                ctx.golden("stig_" + key[-1], key)


class _OutOfRange(AutoTuneFailed):
    pass


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_LIMITS = {"focus_coarse": (5174, 16383), "focus_fine": (0, 4095), "stig_x": (0, 4095), "stig_y": (0, 4095)}


class _Ctx:
    """One procedure: settings, measurements, the start values to fall back to."""

    def __init__(self, tuner: AutoTuner, kind: str, start: dict):
        self.t = tuner
        self.kind = kind
        self.start = dict(start)
        self.cur = dict(start)
        self.samples: list = []
        self.steps = 0
        self.score_start: float | None = None
        self.image = "focus"  # which test images: "focus" or "stig" (AutoTuneOptions)
        self.coarse_step = tuner.coarse_step_counts() if "focus_coarse" in start else 1
        self._score: tuple | None = None  # (image, sharpness) at the present settings
        self.focus_tol = 1.0  # FOCUS F counts: the depth of focus (AutoTuner.focus_tolerance_counts)

    @property
    def score_now(self) -> float | None:
        """Sharpness at the present settings with the present test images (None: not measured)."""
        return self._score[1] if self._score is not None and self._score[0] == self.image else None

    @score_now.setter
    def score_now(self, v) -> None:
        self._score = None if v is None else (self.image, v)

    def use(self, image: str) -> None:
        self.image = image

    # -------------------------------------------------------------- hardware
    def check(self):
        if self.t._cancel.is_set():
            raise AutoTuneCancelled("cancelled")

    def set(self, key: str, value: int, check: bool = True) -> int:
        """Set one knob (SEMControl confirms it by readback); returns the value read back.  FOCUS C
        moves at most one coarse step (0.1 mm of WD) per setting: a larger change is walked."""
        if check:
            self.check()
        lo, hi = _LIMITS[key]
        v = int(min(hi, max(lo, round(value))))
        if key == "focus_coarse":
            step = self.coarse_step
            while abs(v - self.cur[key]) > step:
                self._write(key, self.cur[key] + (step if v > self.cur[key] else -step))
                if check:
                    self.check()
        if v == self.cur.get(key):
            return v
        return self._write(key, v)

    def _write(self, key: str, v: int) -> int:
        sem, to = self.t.sem, self.t.options.timeout_s
        if key == "focus_coarse":
            got = sem.set_focus("C", v).result(to)
        elif key == "focus_fine":
            got = sem.set_focus("F", v).result(to)
        else:
            got = sem.set_stigmator(key[-1].upper(), v).result(to)
        got = int(got) if got is not None else v
        self.cur[key] = got
        self.score_now = None
        if self.t.options.settle_s > 0:
            time.sleep(self.t.options.settle_s)
        return got

    def measure(self) -> float:
        """Sharpness of a test image at the present settings."""
        self.check()
        o = self.t.options
        speed, n = (o.focus_speed, o.focus_px) if self.image == "focus" else (o.stig_speed, o.stig_px)
        fut = self.t.engine.capture(frames=1, nx=n, ny=n, speed=speed)
        frame = fut.result(o.timeout_s + 5.0)
        self.check()
        img = np.asarray(frame.image, dtype=float)
        if not np.isfinite(img).all() or float(np.std(img)) <= 0:
            raise AutoTuneFailed("blank test image")
        s = sharpness(img)
        self.score_now = s
        self.steps += 1
        return s

    def evaluate(self, key: str, value: int) -> tuple:
        got = self.set(key, value)
        s = self.measure() if self.score_now is None else self.score_now
        self.samples.append((key, got, s))
        self.t._emit(type="progress", text=f"{self.kind}: {key.replace('_', ' ')} {got} → {s:.4g}",
                     key=key, value=got, score=s, step=self.steps)
        return got, s

    def restore(self) -> str | None:
        """Set the start values again (FOCUS C walked back in coarse steps; not cancellable)."""
        err = None
        for k, v in self.start.items():
            if self.cur.get(k) != v:
                try:
                    self.set(k, v, check=False)
                    self.cur[k] = v
                except Exception as e:  # noqa: BLE001
                    err = f"{k}: {e}"
        return err

    def result(self, ok: bool, msg: str, final_score, t0) -> AutoTuneResult:
        return AutoTuneResult(self.kind, ok, msg, dict(self.start), dict(self.cur), self.score_start, final_score,
                              list(self.samples), self.steps, time.perf_counter() - t0)

    # -------------------------------------------------------------- searches
    def _check_structure(self, pts: dict, what: str) -> None:
        vals = list(pts.values())
        if vals and max(vals) <= self.t.options.min_sharpness:
            raise AutoTuneFailed(f"{what}: no visible detail (sharpness ≤ {max(vals):.2g}; far out of focus, "
                                 "or nothing to focus on)")
        if len(vals) >= 3 and max(vals) < self.t.options.min_contrast * max(min(vals), 1e-12):
            raise AutoTuneFailed(f"{what}: no clear optimum (sharpness changes < "
                                 f"{(self.t.options.min_contrast - 1) * 100:.0f} %; not enough structure?)")

    def local(self, key: str, step: int, min_step: int, quantum: int):
        """Near the present value first: it and *step* / 2·*step* either side.  A peak (the best not
        at an end of the five) is refined; a clear trend (the best at an end, best / worst ≥
        ``local_contrast``) is walked that way with doubling steps until the sharpness falls (the
        peak bracketed), at most ``local_range`` steps from the start, then refined.  Refining halves
        the step around the best value down to *min_step* (the depth of focus; the stigmator's 16),
        then a parabola through the best three.  Values are multiples of *quantum*.  Returns the
        value set, or None (no peak or trend: the caller searches the whole range; the start value
        is set again)."""
        o = self.t.options
        lo, hi = _LIMITS[key]
        x0 = self.cur[key]
        pts: dict[int, float] = {}

        def snap(x):
            return int(min(hi, max(lo, round(x / quantum) * quantum)))

        def ev(x):
            x = snap(x)
            if x not in pts:
                got, sc = self.evaluate(key, x)
                pts[got] = sc
                return got
            return x

        if self.score_now is None:
            self.measure()
        pts[x0] = self.score_now
        self.samples.append((key, x0, self.score_now))
        for k in (-1, 1, -2, 2):
            ev(x0 + k * step)
        probe = sorted(pts)
        vals = [pts[x] for x in probe]
        best = max(pts, key=pts.get)
        if max(vals) <= o.min_sharpness or max(vals) < o.local_contrast * max(min(vals), 1e-12):
            self.set(key, x0)
            return None
        if best in (probe[0], probe[-1]) and best not in (lo, hi):  # a trend: walk that way
            d = 1 if best == probe[-1] else -1
            gap = step
            while True:
                if abs(best - x0) >= o.local_range * step:
                    self.set(key, x0)
                    return None
                gap *= 2
                nxt = ev(best + d * gap)
                if pts[nxt] < pts[best] or nxt in (lo, hi):
                    break
                best = nxt
            if pts[nxt] >= pts[best]:  # the end of the range: no peak bracketed
                self.set(key, x0)
                return None
            step = gap // 2
        elif best in (lo, hi):
            self.set(key, x0)
            return None
        # refine: halve the step around the best value down to min_step
        while step > min_step:
            step = max(min_step, step // 2)
            b = max(pts, key=pts.get)
            ev(b - step)
            ev(b + step)
        xs = sorted(pts)
        b = max(pts, key=pts.get)
        i = xs.index(b)
        if 0 < i < len(xs) - 1:
            trio = xs[i - 1:i + 2]
            v = fit_log_parabola(trio, [pts[x] for x in trio])
            if v is not None and trio[0] <= v <= trio[-1]:
                ev(v)
        self.t._emit(type="progress", text=f"{self.kind}: {key.replace('_', ' ')} – peak near the start value")
        return self._end_at_best(key, pts)

    def _end_at_best(self, key: str, pts: dict) -> int:
        best = max(pts, key=pts.get)
        if self.cur[key] != best:
            got, s = self.evaluate(key, best)
            pts[got] = s
        return self.cur[key]

    def sweep(self, key: str, step: int, lo: int, hi: int, what: str) -> dict:
        """Step *key* from its present value by *step* towards *hi* until the sharpness has clearly
        fallen past its peak (or *hi*), then towards *lo* unless the peak is bracketed above.
        Returns the measurements; raises when there is no clear optimum or the sharpest image is
        at an end of [lo, hi]."""
        o = self.t.options
        x0 = self.cur[key]
        if self.score_now is None:
            self.measure()
        pts: dict[int, float] = {x0: self.score_now}
        self.samples.append((key, x0, self.score_now))

        def go(direction: int) -> bool:
            """True when the sweep passed the peak (it is bracketed on this side)."""
            below = 0
            i = 0
            while True:
                i += 1
                x = x0 + direction * i * step
                if x > hi or x < lo:
                    end = hi if direction > 0 else lo
                    if (x - direction * step) == end:
                        return False
                    x = end  # the last point: the end of the range itself
                got, sc = self.evaluate(key, x)
                pts[got] = sc
                best = max(pts, key=pts.get)
                if (pts[best] > o.min_sharpness and (got - best) * direction > 0
                        and sc < o.coarse_drop * pts[best]):
                    below += 1
                    if below >= 2:
                        return True
                else:
                    below = 0
                if got in (lo, hi):
                    return False

        passed_up = go(+1)
        best = max(pts, key=pts.get)
        if not (passed_up and best > x0):  # the peak is not bracketed above the start
            go(-1)
        self._check_structure(pts, key)
        best = max(pts, key=pts.get)
        if best in (lo, hi):
            raise _OutOfRange(f"{key}: the sharpest image is at the end of {what}")
        return pts

    def coarse_sweep(self) -> int:
        """FOCUS C in coarse steps (0.1 mm of WD) within ±coarse_range_mm of its start (see
        :meth:`sweep`), then the peak of a parabola through the best point and its neighbours."""
        key = "focus_coarse"
        o = self.t.options
        step = self.coarse_step
        n = max(1, int(round(o.coarse_range_mm / o.coarse_step_mm)))
        x0 = self.start[key]
        self.set(key, x0)
        lo, hi = _LIMITS[key]
        pts = self.sweep(key, step, max(lo, x0 - n * step), min(hi, x0 + n * step),
                         f"the ±{o.coarse_range_mm:g} mm range")
        best = max(pts, key=pts.get)
        nb = [x for x in sorted(pts) if abs(x - best) <= step]
        v = fit_log_parabola(nb, [pts[x] for x in nb]) if len(nb) >= 3 else None
        if v is not None and min(nb) <= v <= max(nb):
            ev_x = int(round(v))
            if ev_x not in pts:
                got, sc = self.evaluate(key, ev_x)
                pts[got] = sc
        return self._end_at_best(key, pts)

    def fine_focus(self, h: float) -> int:
        """FOCUS F over its whole range: a grid of ``fine_initial_step`` (512) counts (finer at high
        magnification: at most 4 steps *h*), then halving the step around the sharpest value down to
        1 count (two new values per halving); the sharpest value measured is kept."""
        key = "focus_fine"
        lo, hi = _LIMITS[key]
        o = self.t.options
        pts: dict[int, float] = {}

        def ev(x):
            x = int(min(hi, max(lo, round(x))))
            if x not in pts:
                got, sc = self.evaluate(key, x)
                pts[got] = sc

        x0 = self.cur[key]
        if self.score_now is None:
            self.measure()
        pts[x0] = self.score_now
        self.samples.append((key, x0, self.score_now))
        # the grid must not step over the focus curve: at most 4 magnification steps apart (a power
        # of two, so the halving lands on whole counts)
        s0 = int(o.fine_initial_step)
        while s0 > 8 and s0 > 4.0 * max(4.0, float(h)):
            s0 //= 2
        grid = sorted(set(range(lo, hi + 1, s0)) | {hi})
        # from the end nearer to the present value (smaller first move)
        for x in (grid if x0 - lo <= hi - x0 else grid[::-1]):
            ev(x)
        self._check_structure(pts, key)
        # halving down to the depth of focus (focus_tolerance_counts)
        stop = max(1, int(getattr(self, "focus_tol", 1)))
        step = s0 // 2
        while step >= stop:
            best = max(pts, key=pts.get)
            ev(best - step)
            ev(best + step)
            step //= 2
        best = max(pts, key=pts.get)
        if best in (lo, hi):
            raise _OutOfRange(f"{key}: the sharpest image is at the end of the fine focus range")
        return self._end_at_best(key, pts)

    def golden(self, label: str, key: str) -> int:
        """Full-range golden-section search of one stigmator axis, then a parabola."""
        lo, hi = _LIMITS[key]
        q = 16  # the stigmator's step
        minstep = self.t.options.stig_min_step
        pts: dict[int, float] = {}

        def snap(x):
            return int(min(hi, max(lo, round(x / q) * q)))

        def ev(x):
            x = snap(x)
            if x in pts:
                return x, pts[x]
            got, s = self.evaluate(key, x)
            pts[got] = s
            return got, s

        a, b = float(lo), float(hi)
        c = b - GOLDEN * (b - a)
        d = a + GOLDEN * (b - a)
        c, fc = ev(c)
        d, fd = ev(d)
        for _ in range(self.t.options.stig_rounds):
            if d - c < minstep:
                break
            if fc >= fd:
                b, d, fd = d, c, fc
                c, fc = ev(b - GOLDEN * (b - a))
            else:
                a, c, fc = c, d, fd
                d, fd = ev(a + GOLDEN * (b - a))
        self._check_structure(pts, key)
        xs = sorted(pts)
        best = max(pts, key=pts.get)
        i = xs.index(best)
        if 0 < i < len(xs) - 1:
            trio = xs[i - 1:i + 2]
            v = fit_log_parabola(trio, [pts[x] for x in trio])
            target = snap(v) if v is not None and trio[0] <= v <= trio[-1] else best
        else:
            target = best
        ev(target)
        return self._end_at_best(key, pts)
