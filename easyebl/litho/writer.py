"""High-level lithography: pattern + doses → compiled vector-scan job → DAQ."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Sequence

import numpy as np

from ..core.calibration import FieldCalibration
from ..core.transforms import AffineTransform
from . import dose as dosemod
from .compiler import CompiledJob, CompileStats, ExposureRun, VectorScanCompiler, VectorScanSettings
from .fill import line_points, raster_fill
from .geometry import Dot, Pattern, Polygon, Polyline

__all__ = ["ExposureParameters", "LithoWriter", "ExposureReport", "build_runs", "order_elements"]


@dataclass
class ExposureParameters:
    beam_current_a: float = 100e-12
    area_dose_uC_cm2: float = 300.0
    line_dose_pC_cm: float = 1000.0
    dot_dose_fC: float = 10.0
    area_step_um: float = 0.02
    line_step_um: float = 0.01
    fill_direction: str = "x"
    serpentine: bool = False
    grid_origin_um: tuple = (0.0, 0.0)
    order: str = "as_is"  # as_is | nearest | kind (dots, lines, areas)

    def area_dwell(self, factor: float = 1.0) -> float:
        return dosemod.area_dwell_s(self.area_dose_uC_cm2, self.area_step_um, self.beam_current_a) * factor

    def line_dwell_per_um(self, factor: float = 1.0) -> float:
        """Dwell per µm of line length (multiply by point spacing)."""
        return dosemod.line_dwell_s(self.line_dose_pC_cm, 1.0, self.beam_current_a) * factor

    def dot_dwell(self, factor: float = 1.0) -> float:
        return dosemod.dot_dwell_s(self.dot_dose_fC, self.beam_current_a) * factor


@dataclass
class ExposureReport:
    stats: CompileStats
    aborted: bool
    samples_written: int
    elapsed_s: float
    deposited_charge_c: float = float("nan")  # filled by callers that can measure it
    info: dict = field(default_factory=dict)


def _start_point(e) -> np.ndarray:
    if isinstance(e, Dot):
        return np.array([e.x, e.y])
    return np.asarray(e.points[0], float)


def order_elements(elements: list, mode: str = "as_is", start=(0.0, 0.0)) -> list:
    if mode == "as_is":
        return list(elements)
    if mode == "kind":
        rank = {"dot": 0, "line": 1, "area": 2}
        return sorted(elements, key=lambda e: rank[e.kind])
    if mode == "nearest":
        if not elements:
            return []
        pts = np.array([np.mean(np.atleast_2d(e.points), axis=0) for e in elements])
        left = np.ones(len(elements), bool)
        cur = np.asarray(start, float)
        out = []
        for _ in range(len(elements)):
            d = np.hypot(*(pts - cur).T)
            d[~left] = np.inf
            i = int(np.argmin(d))
            left[i] = False
            out.append(elements[i])
            cur = pts[i]
        return out
    raise ValueError("order must be 'as_is', 'nearest' or 'kind'")


def build_runs(pattern: Pattern, params: ExposureParameters, transform: AffineTransform | None = None) -> list:
    """Convert a pattern into exposure runs (fill/discretise + dwell per point).

    *transform* (design → field, e.g. from marker alignment) is applied to the
    exposure **points** after filling, so the fill grid stays aligned with the
    design and only the placement changes.
    """
    runs: list = []
    els = order_elements(list(pattern), params.order)
    t = transform if transform is not None and not transform.is_identity() else None
    for idx, e in enumerate(els):
        if e.dose_factor == 0:
            continue
        if isinstance(e, Polygon):
            fill = raster_fill(
                e,
                params.area_step_um,
                direction=params.fill_direction,
                serpentine=params.serpentine,
                origin_um=params.grid_origin_um,
            )
            dw = params.area_dwell(e.dose_factor)
            for r in fill.runs:
                pts = t.apply(r) if t else r
                runs.append(ExposureRun(pts, np.full(len(r), dw), idx, "area"))
        elif isinstance(e, Polyline):
            lp = line_points(e, params.line_step_um)
            pts = t.apply(lp.points) if t else lp.points
            runs.append(ExposureRun(pts, lp.spacing_um * params.line_dwell_per_um(e.dose_factor), idx, "line"))
        elif isinstance(e, Dot):
            p = np.array([[e.x, e.y]])
            runs.append(ExposureRun(t.apply(p) if t else p, np.array([params.dot_dwell(e.dose_factor)]), idx, "dot"))
        else:  # pragma: no cover
            raise TypeError(f"unsupported element {type(e).__name__}")
    return runs


class LithoWriter:
    """Expose patterns through a DAQ handle."""

    def __init__(self, handle, calibration: FieldCalibration, settings: VectorScanSettings | None = None):
        self.handle = handle
        self.calibration = calibration
        self.settings = settings or VectorScanSettings()

    def compile(
        self,
        pattern: Pattern,
        params: ExposureParameters,
        transform: AffineTransform | None = None,
        check: bool = True,
    ) -> CompiledJob:
        runs = build_runs(pattern, params, transform)
        return VectorScanCompiler(self.calibration, self.settings).compile(runs, check=check)

    def expose_job(
        self,
        job: CompiledJob,
        progress: Callable[[float], None] | None = None,
        timeout: float = 0.0,
    ) -> ExposureReport:
        total = max(1, job.stats.total_samples)
        cb = (lambda n: progress(min(1.0, n / total))) if progress else None
        t0 = time.perf_counter()
        res = self.handle.run_stream(job.settings.rate, job.chunks(), progress=cb, timeout=timeout)
        return ExposureReport(job.stats, res.aborted, res.samples_written, time.perf_counter() - t0)

    def expose(
        self,
        pattern: Pattern,
        params: ExposureParameters,
        transform: AffineTransform | None = None,
        progress: Callable[[float], None] | None = None,
        timeout: float = 0.0,
    ) -> ExposureReport:
        return self.expose_job(self.compile(pattern, params, transform), progress, timeout)

    # -------------------------------------------------------------- shortcuts
    def write_polygon(self, polygon: Polygon, params: ExposureParameters, **kw) -> ExposureReport:
        return self.expose(Pattern([polygon]), params, **kw)

    def write_lines(self, lines: Sequence[Polyline], params: ExposureParameters, **kw) -> ExposureReport:
        return self.expose(Pattern(list(lines)), params, **kw)

    def write_dots(self, dots: Sequence[Dot], params: ExposureParameters, **kw) -> ExposureReport:
        return self.expose(Pattern(list(dots)), params, **kw)
