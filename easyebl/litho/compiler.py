"""Vector-scan compiler: exposure runs → sample-clocked X/Y/blank streams.

Each :class:`ExposureRun` is a sequence of points exposed back-to-back with
the beam on (one scan-line span of a fill, one polyline, one dot).  Between
runs the compiler inserts::

    blank-lead (hold, blanked) → jump (blanked) → settle (blanked, at target)

with the settle time growing with jump distance.  Within a run every point
is held for ``round(dwell · rate)`` samples; optionally each point can be
preceded by a short blanked settle (``blank_between_points``, "step-and-
blank" mode, useful for dot arrays).

``blanker_delay_s`` compensates blanker latency relative to deflection by
delaying X/Y (equivalently: advancing the blank edges).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterator, Sequence

import numpy as np

from ..core.calibration import FieldCalibration
from ..core.waveform import OutputChunk

__all__ = ["ExposureRun", "VectorScanSettings", "CompiledJob", "VectorScanCompiler", "CompileStats"]


@dataclass
class ExposureRun:
    points_um: np.ndarray  # (M, 2) field coordinates (after alignment)
    dwell_s: np.ndarray  # (M,) requested dwell per point
    element: int = -1  # index of the source element (bookkeeping)
    kind: str = "area"

    def __post_init__(self):
        self.points_um = np.atleast_2d(np.asarray(self.points_um, float))
        d = np.asarray(self.dwell_s, float)
        if d.ndim == 0:
            d = np.full(len(self.points_um), float(d))
        if d.shape != (len(self.points_um),):
            raise ValueError("dwell_s must be scalar or one value per point")
        if np.any(d < 0):
            raise ValueError("dwell must be ≥ 0")
        self.dwell_s = d

    def __len__(self) -> int:
        return len(self.points_um)


@dataclass
class VectorScanSettings:
    rate: float = 500e3
    settle_base_s: float = 10e-6  # after every blanked jump
    settle_per_um_s: float = 0.5e-6  # additional settle per µm of jump
    settle_max_s: float = 300e-6
    blank_lead_s: float = 2.5e-6  # blanked hold before leaving a run
    blanker_delay_s: float = 0.0  # blanker latency to compensate
    max_unblanked_step_um: float | None = None  # larger in-run steps are split into jumps
    merge_runs_within_um: float | None = None  # runs starting this close continue without blanking
    blank_between_points: bool = False
    point_settle_s: float = 0.0  # blanked settle before each point (if blank_between_points)
    min_samples_per_point: int = 1
    chunk_samples: int = 65536
    park_um: tuple | None = (0.0, 0.0)  # final blanked position (None = stay)
    park_samples: int = 8
    quantization: str = "diffuse"  # "round": each point independently; "diffuse": error diffusion along a run

    def __post_init__(self):
        if self.quantization not in ("round", "diffuse"):
            raise ValueError("quantization must be 'round' or 'diffuse'")

    def n(self, t: float) -> int:
        return max(0, int(round(t * self.rate)))

    def settle_samples(self, dist_um: float) -> int:
        t = min(self.settle_max_s, self.settle_base_s + self.settle_per_um_s * dist_um)
        return max(1, self.n(t))


@dataclass
class CompileStats:
    n_runs: int = 0
    n_points: int = 0
    n_jumps: int = 0
    exposure_samples: int = 0
    overhead_samples: int = 0
    rate: float = 1.0
    requested_exposure_s: float = 0.0
    dose_error_rms: float = 0.0  # relative, per point (dwell-weighted)
    dose_error_max: float = 0.0  # relative, worst single point
    dose_error_total: float = 0.0  # relative error of the total exposure time
    clamped_points: int = 0  # points whose request was below min_samples_per_point (overdosed)
    zero_dose_points: int = 0

    @property
    def total_samples(self) -> int:
        return self.exposure_samples + self.overhead_samples

    @property
    def exposure_time_s(self) -> float:
        return self.exposure_samples / self.rate

    @property
    def total_time_s(self) -> float:
        return self.total_samples / self.rate

    @property
    def duty_cycle(self) -> float:
        return self.exposure_samples / self.total_samples if self.total_samples else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.update(
            total_samples=self.total_samples,
            exposure_time_s=self.exposure_time_s,
            total_time_s=self.total_time_s,
            duty_cycle=self.duty_cycle,
        )
        return d


class CompiledJob:
    """Validated, ready-to-stream vector-scan job."""

    def __init__(self, runs: list, settings: VectorScanSettings, calibration: FieldCalibration, start_um):
        self.runs = runs
        self.settings = settings
        self.calibration = calibration
        self.start_um = np.asarray(start_um, float)
        self.samples = [self._point_samples(r) for r in runs]
        self.stats = self._compute_stats()

    # ---------------------------------------------------------------- analysis
    def _point_samples(self, run: ExposureRun) -> np.ndarray:
        """Whole samples per point.

        ``diffuse`` carries the rounding remainder from point to point along
        the run (1-D error diffusion), so the *integrated* dose of an area is
        exact even when a single dwell is only a few clock ticks; each point is
        still within ±1 sample of its request.
        """
        s = self.settings
        pos = run.dwell_s > 0
        # clamp the *request* first so diffusion never hands a point 0 samples
        req = np.where(pos, np.maximum(run.dwell_s * s.rate, s.min_samples_per_point), 0.0)
        if s.quantization == "diffuse" and len(req) > 1:
            c = np.rint(np.cumsum(req))
            n = np.diff(np.concatenate([[0.0], c])).astype(np.int64)
            # rounding can still yield min-1 for a point next to a large one; enforce the floor
            n = np.where(pos, np.maximum(n, s.min_samples_per_point), 0)
        else:
            n = np.where(pos, np.maximum(np.rint(req).astype(np.int64), s.min_samples_per_point), 0)
        return n

    def _segments(self):
        """Yield (kind, payload) tuples describing the stream (shared by stats and chunks)."""
        s = self.settings
        cur = self.start_um.copy()
        exposing = False
        for run, ns in zip(self.runs, self.samples):
            if len(run) == 0:
                continue
            p0 = run.points_um[0]
            dist = float(np.hypot(*(p0 - cur)))
            merge = exposing and s.merge_runs_within_um is not None and dist <= s.merge_runs_within_um
            if not merge:
                yield ("jump", (cur.copy(), p0.copy(), dist))
            yield ("run", (run, ns))
            cur = run.points_um[-1].copy()
            exposing = True
        yield ("end", (cur.copy(),))

    def _compute_stats(self) -> CompileStats:
        s = self.settings
        st = CompileStats(rate=s.rate)
        num = den = 0.0
        for kind, payload in self._segments():
            if kind == "jump":
                _, _, dist = payload
                st.n_jumps += 1
                st.overhead_samples += s.n(s.blank_lead_s) + s.settle_samples(dist)
            elif kind == "run":
                run, ns = payload
                st.n_runs += 1
                st.n_points += len(run)
                st.exposure_samples += int(ns.sum())
                if s.blank_between_points:
                    st.overhead_samples += s.n(s.point_settle_s) * len(run)
                req = run.dwell_s
                st.requested_exposure_s += float(req.sum())
                pos = req > 0
                st.zero_dose_points += int(np.count_nonzero(~pos))
                if np.any(pos):
                    err = ns[pos] / s.rate / req[pos] - 1.0
                    w = req[pos]
                    num += float((w * err * err).sum())
                    den += float(w.sum())
                    st.dose_error_max = max(st.dose_error_max, float(np.max(np.abs(err))))
                    st.clamped_points += int(np.count_nonzero(req[pos] * s.rate < s.min_samples_per_point))
            else:
                st.overhead_samples += s.n(s.blank_lead_s)
                if s.park_um is not None:
                    st.overhead_samples += s.park_samples
        st.overhead_samples += max(0, int(math.ceil(s.blanker_delay_s * s.rate)))
        st.dose_error_rms = math.sqrt(num / den) if den else 0.0
        if st.requested_exposure_s > 0:
            st.dose_error_total = st.exposure_samples / s.rate / st.requested_exposure_s - 1.0
        return st

    def check_field(self) -> None:
        """Raise :class:`OutOfFieldError` if any point maps outside the deflection range."""
        for run in self.runs:
            if len(run):
                self.calibration.um_to_volts(run.points_um, check=True)

    # ----------------------------------------------------------------- stream
    def _raw_pieces(self) -> Iterator[tuple]:
        """(points_um (n,2), blank (n,)) pieces before delay compensation/chunking."""
        s = self.settings
        n_lead = s.n(s.blank_lead_s)
        for kind, payload in self._segments():
            if kind == "jump":
                a, b, dist = payload
                if n_lead:
                    yield np.repeat(a[None], n_lead, 0), np.ones(n_lead, bool)
                n_set = s.settle_samples(dist)
                yield np.repeat(b[None], n_set, 0), np.ones(n_set, bool)
            elif kind == "run":
                run, ns = payload
                pts = run.points_um
                if s.blank_between_points and s.n(s.point_settle_s) > 0:
                    k = s.n(s.point_settle_s)
                    reps = np.empty(2 * len(ns), np.int64)
                    reps[0::2], reps[1::2] = k, ns
                    p2 = np.repeat(pts, 2, axis=0)
                    bl = np.tile([True, False], len(ns))
                    yield np.repeat(p2, reps, axis=0), np.repeat(bl, reps)
                else:
                    yield np.repeat(pts, ns, axis=0), np.zeros(int(ns.sum()), bool)
            else:
                (cur,) = payload
                if n_lead:
                    yield np.repeat(cur[None], n_lead, 0), np.ones(n_lead, bool)
                if s.park_um is not None:
                    yield np.repeat(np.asarray(s.park_um, float)[None], s.park_samples, 0), np.ones(
                        s.park_samples, bool
                    )

    def chunks(self) -> Iterator[OutputChunk]:
        s = self.settings
        cal = self.calibration
        d = max(0, int(math.ceil(s.blanker_delay_s * s.rate)))
        size = s.chunk_samples
        buf_p: list = []
        buf_b: list = []
        n_buf = 0
        carry_p = None  # last d positions (XY delayed by d samples)

        def emit(pts: np.ndarray, bl: np.ndarray) -> OutputChunk:
            nonlocal carry_p
            if d:
                if carry_p is None:
                    carry_p = np.repeat(pts[:1], d, 0)
                allp = np.vstack([carry_p, pts])
                carry_p = allp[-d:]
                pts = allp[:-d]
            v = cal.um_to_volts(pts, check=False)
            return OutputChunk(v[:, 0], v[:, 1], bl)

        for pts, bl in self._raw_pieces():
            if len(pts) == 0:
                continue
            buf_p.append(pts)
            buf_b.append(bl)
            n_buf += len(pts)
            while n_buf >= size:
                P, B = np.vstack(buf_p), np.concatenate(buf_b)
                yield emit(P[:size], B[:size])
                buf_p, buf_b = [P[size:]], [B[size:]]
                n_buf -= size
        if n_buf:
            yield emit(np.vstack(buf_p), np.concatenate(buf_b))
        if d and carry_p is not None:  # flush delayed XY, blanked
            v = cal.um_to_volts(carry_p, check=False)
            yield OutputChunk(v[:, 0], v[:, 1], np.ones(d, bool))

    def preview_arrays(self, max_samples: int = 2_000_000) -> OutputChunk:
        """Whole stream as one chunk (refuses very long jobs)."""
        if self.stats.total_samples > max_samples:
            raise ValueError(f"job has {self.stats.total_samples} samples (> {max_samples})")
        from ..core.waveform import concat_chunks

        return concat_chunks(self.chunks())


class VectorScanCompiler:
    def __init__(self, calibration: FieldCalibration, settings: VectorScanSettings | None = None):
        self.calibration = calibration
        self.settings = settings or VectorScanSettings()

    def _split(self, run: ExposureRun) -> list:
        lim = self.settings.max_unblanked_step_um
        if lim is None or len(run) < 2:
            return [run]
        steps = np.hypot(*np.diff(run.points_um, axis=0).T)
        cuts = np.nonzero(steps > lim)[0] + 1
        if len(cuts) == 0:
            return [run]
        out = []
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(run)]):
            out.append(ExposureRun(run.points_um[a:b], run.dwell_s[a:b], run.element, run.kind))
        return out

    def compile(self, runs: Sequence[ExposureRun], start_um=None, check: bool = True) -> CompiledJob:
        if start_um is None:
            start_um = self.settings.park_um if self.settings.park_um is not None else (0.0, 0.0)
        split = []
        for r in runs:
            split.extend(self._split(r))
        job = CompiledJob(split, self.settings, self.calibration, start_um)
        if check:
            job.check_field()
        return job
