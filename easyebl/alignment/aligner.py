"""Automatic marker alignment on top of raster imaging + template matching.

Lag-free mark positions
-----------------------
Imaging lag (deflection + video delay) shifts a mark by ``v·τ`` along the
fast-scan direction, whereas lithography exposes quasi-statically.  With
``bidirectional_average=True`` every mark is imaged twice, once scanning
+x and once −x; the lag shifts the two images in opposite directions, so the
average of the two matched positions is free of it (to first order).  The
half-difference is reported as ``lag_shift_um`` (a free health check of τ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.transforms import AffineTransform
from ..imaging.raster import RasterConfig, RasterImage, RasterImager
from .fitting import FitResult, fit_transform
from .matching import MatchResult, cross_template, match_template, preprocess

__all__ = ["MarkSpec", "MarkMeasurement", "AlignmentResult", "MarkerAligner", "AlignmentError"]

TemplateFactory = Callable[[float], np.ndarray]  # pixel size (µm) -> template image


class AlignmentError(RuntimeError):
    pass


@dataclass
class MarkSpec:
    nominal_um: tuple
    name: str = ""
    window_um: float | None = None


@dataclass
class MarkMeasurement:
    name: str
    nominal_um: np.ndarray
    measured_um: np.ndarray
    score: float
    distinctness: float
    lag_shift_um: np.ndarray | None = None
    matches: list = field(default_factory=list)  # MatchResult per scan
    images: list = field(default_factory=list)  # RasterImage per scan
    ok: bool = True
    message: str = ""


@dataclass
class AlignmentResult:
    """``transform`` maps *design* (nominal) coordinates to actual field coordinates."""

    transform: AffineTransform
    model: str
    measurements: list
    fit: FitResult | None = None

    @property
    def rms_um(self) -> float:
        return self.fit.rms if self.fit else 0.0

    @property
    def residuals_um(self) -> np.ndarray:
        return self.fit.residuals if self.fit else np.zeros((0, 2))

    def summary(self) -> str:
        d = self.transform.decompose()
        return (
            f"{self.model}: shift=({d['tx']:.4f}, {d['ty']:.4f}) µm, rot={d['rotation_rad'] * 1e3:.3f} mrad, "
            f"scale=({d['sx']:.6f}, {d['sy']:.6f}), rms={self.rms_um * 1e3:.1f} nm, "
            f"marks={sum(m.ok for m in self.measurements)}/{len(self.measurements)}"
        )


def default_cross_factory(size_um: float = 8.0, width_um: float = 1.0) -> TemplateFactory:
    return lambda px: cross_template(size_um, width_um, px)


class MarkerAligner:
    """Measure marker positions and fit a design→field transform.

    Parameters
    ----------
    imager:
        :class:`RasterImager` bound to a DAQ handle.
    scan:
        Base raster settings (pixels, dwell, rate, pre-line…).  Centre and
        width are overridden per mark.
    template:
        Either a fixed template array (must match the scan pixel size) or a
        factory ``f(pixel_um) -> array``.
    window_um:
        Default scan window around each nominal mark position.
    min_score:
        NCC threshold below which a mark is rejected.
    bidirectional_average:
        Image each mark scanning +x and −x and average (cancels lag).
    """

    def __init__(
        self,
        imager: RasterImager,
        scan: RasterConfig | None = None,
        template: np.ndarray | TemplateFactory | None = None,
        window_um: float = 20.0,
        min_score: float = 0.5,
        bidirectional_average: bool = True,
        search_radius_um: float | None = None,
        background_sigma_px: float | None = None,
    ):
        self.imager = imager
        self.scan = scan or RasterConfig(nx=128, ny=128, dwell_s=4e-6, preline_s=20e-6, blank_preline=True)
        self.template = template if template is not None else default_cross_factory()
        self.window_um = window_um
        self.min_score = min_score
        self.bidirectional_average = bidirectional_average
        self.search_radius_um = search_radius_um
        self.background_sigma_px = background_sigma_px

    # ------------------------------------------------------------------ helpers
    def _template_for(self, pixel_um: float) -> np.ndarray:
        t = self.template(pixel_um) if callable(self.template) else np.asarray(self.template, float)
        return t

    def _scan_config(self, center, window, x_direction: int) -> RasterConfig:
        return self.scan.with_(
            center_um=(float(center[0]), float(center[1])),
            width_um=float(window),
            height_um=None,
            x_direction=x_direction,
            bidirectional=False,
        )

    def locate_in_image(self, img: RasterImage, channel: str | None = None) -> tuple[np.ndarray, MatchResult]:
        """Match the template in an existing image; returns (field µm, MatchResult)."""
        px = img.pixel_size_um[0]
        t = self._template_for(px)
        data = preprocess(img.image(channel), self.background_sigma_px)
        kw = {}
        if self.search_radius_um is not None:
            kw = dict(
                search_center=((img.config.nx - 1) / 2, (img.config.ny - 1) / 2),
                search_radius_px=self.search_radius_um / px,
            )
        m = match_template(data, t, **kw)
        return img.pixel_to_field(m.col, m.row), m

    # -------------------------------------------------------------- measuring
    def measure(self, mark: MarkSpec | Sequence[float], channel: str | None = None) -> MarkMeasurement:
        spec = mark if isinstance(mark, MarkSpec) else MarkSpec(tuple(mark))
        window = spec.window_um or self.window_um
        nominal = np.asarray(spec.nominal_um, float)
        dirs = (1, -1) if self.bidirectional_average else (self.scan.x_direction,)
        positions, matches, images = [], [], []
        for d in dirs:
            img = self.imager.acquire(self._scan_config(nominal, window, d))
            if img.aborted:
                raise AlignmentError("scan aborted")
            pos, m = self.locate_in_image(img, channel)
            positions.append(pos)
            matches.append(m)
            images.append(img)
        positions = np.array(positions)
        measured = positions.mean(axis=0)
        score = float(min(m.score for m in matches))
        dist = float(min(m.distinctness for m in matches))
        lag = (positions[0] - positions[1]) / 2 if len(positions) == 2 else None
        ok = score >= self.min_score
        msg = "" if ok else f"score {score:.2f} < {self.min_score:.2f}"
        if ok and len(positions) == 2:
            spread = float(np.hypot(*(positions[0] - positions[1])))
            if spread > 0.25 * window:
                ok, msg = False, f"forward/backward disagree by {spread:.2f} µm (> 25% of window)"
        return MarkMeasurement(
            name=spec.name,
            nominal_um=nominal,
            measured_um=measured,
            score=score,
            distinctness=dist,
            lag_shift_um=lag,
            matches=matches,
            images=images,
            ok=ok,
            message=msg,
        )

    def align(
        self,
        marks: Sequence[MarkSpec | Sequence[float]],
        model: str = "similarity",
        channel: str | None = None,
        on_mark: Callable[[MarkMeasurement], None] | None = None,
    ) -> AlignmentResult:
        """Measure all marks and fit ``model`` (translation/rigid/similarity/affine)."""
        meas = []
        for mk in marks:
            m = self.measure(mk, channel)
            meas.append(m)
            if on_mark:
                on_mark(m)
        good = [m for m in meas if m.ok]
        if not good:
            raise AlignmentError("no mark was found with sufficient score")
        src = np.array([m.nominal_um for m in good])
        dst = np.array([m.measured_um for m in good])
        try:
            fit = fit_transform(src, dst, model)
        except ValueError as e:
            raise AlignmentError(f"cannot fit '{model}' with {len(good)} good mark(s): {e}") from e
        return AlignmentResult(fit.transform, model, meas, fit)
