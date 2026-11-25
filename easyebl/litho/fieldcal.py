"""Field calibration: the stage → beam map ``B`` from images at known stage shifts (no Qt).

At a fixed stage position with a recognisable feature (a Faraday cup, a calibration
sample …) the procedure

1. moves the stage there and takes a reference image (scan rotation 0, centred);
2. moves the stage by known shifts ``Δs`` (a 3 × 3 grid or a cross of ±d, d a fraction of
   the field) and takes an image at each;
3. measures each image's shift against the reference: both images are de-noised (Gaussian
   low-pass) and flattened (a heavily blurred copy subtracted: gradients, charging), then
   matched by partially whitened phase correlation (sub-pixel) – a feature moves by ``−B·Δs``
   in BCS; the match quality is the peak-to-sidelobe ratio of the correlation;
4. fits ``B`` (and a constant drift) by least squares, dropping an image whose residual is far
   beyond the others; the residuals tell how well the images matched.

The image shifts are converted to BCS (true µm) through the beam calibration of the condition
(its linear part), so ``B`` holds the rotation between stage and beam axes and the mirror of the
stage axes; its scale (stage encoder against the calibration grid) is a cross-check only – the
exposure uses the rotation (:func:`~easyebl.litho.coords.stage_rotation`).  The result is valid
for the column condition it was measured at (magnification, beam preset, WD) and the beam
calibration it was measured through.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import time
from typing import Callable

import numpy as np

from .coords import StageAxes, linear_params

__all__ = ["FieldCalConfig", "FieldCalResult", "Condition", "phase_correlation", "prefilter", "fit_robust",
           "WEAK_MATCH_PSR", "shift_pattern", "fit_stage_to_beam",
           "measure_field_calibration", "FieldCalError"]


class FieldCalError(RuntimeError):
    """A calibration that failed; *result* holds what was measured when the fit was refused."""

    def __init__(self, message: str, result: "FieldCalResult | None" = None):
        super().__init__(message)
        self.result = result


#: two working distances closer than this are the same condition (mm)
WD_TOLERANCE_MM = 0.1


@dataclass(frozen=True)
class Condition:
    """The column condition a calibration belongs to (preset None: not known / kept as it was;
    wd_mm: the working distance computed from the focus, None: not known)."""

    magnification: float | None
    preset: str | None = None
    wd_mm: float | None = None

    def describe(self) -> str:
        m = f"×{self.magnification:g}" if self.magnification else "×?"
        if self.wd_mm is not None:
            m += f", WD {self.wd_mm:.2f} mm"
        return m + (f", beam {self.preset!r}" if self.preset else "")

    def same_wd(self, other: "Condition", tol_mm: float = WD_TOLERANCE_MM) -> bool:
        if self.wd_mm is None or other.wd_mm is None:
            return self.wd_mm is None and other.wd_mm is None
        return abs(float(self.wd_mm) - float(other.wd_mm)) <= tol_mm + 1e-9

    def matches(self, other: "Condition", preset: bool = True) -> bool:
        """Same magnification and WD (±:data:`WD_TOLERANCE_MM`), and preset unless *preset* is False."""
        if other is None or self.magnification != other.magnification or not self.same_wd(other):
            return False
        return not preset or self.preset == other.preset

    def to_dict(self) -> dict:
        return {"magnification": self.magnification, "preset": self.preset, "wd_mm": self.wd_mm}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Condition":
        d = d or {}
        return cls(d.get("magnification"), d.get("preset"), d.get("wd_mm"))


@dataclass
class FieldCalConfig:
    """How to calibrate (stored in the litho settings)."""

    position: str = ""  # a named stage position ("" : the x / y below)
    x: float = 0.0
    y: float = 0.0
    pattern: str = "grid"  # grid (3 × 3) | cross (5)
    step_fraction: float = 0.3  # stage shift d / field
    image_fraction: float = 0.8  # scan width / field
    image_px: int = 256
    frames: int = 1
    settle_s: float = 0.5  # after each stage move, before the image
    max_residual_px: float = 1.5  # a fit worse than this (rms, pixels) is refused
    denoise_px: float = 2.5  # Gaussian low-pass before matching (σ, pixels; 0: off)
    background_fraction: float = 0.125  # subtracted blur (σ as a fraction of the image; 0: off)
    whitening: float = 0.5  # cross-power spectrum ÷ |·|^w: 1 pure phase correlation, 0 plain correlation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FieldCalConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class FieldCalResult:
    stage_to_beam: np.ndarray  # B (2 × 2)
    condition: Condition
    field_um: float
    position: tuple  # SCS of the reference image
    residual_um: float  # rms of the fit
    n_images: int
    timestamp: float = field(default_factory=time.time)
    shifts: list = field(default_factory=list)  # [(Δs, measured image shift)] for display
    predicted: list = field(default_factory=list)  # nominal image shift of each image (stage axes)
    quality: list = field(default_factory=list)  # peak-to-sidelobe ratio of each image's match
    rejected: list = field(default_factory=list)  # indices of images left out of the fit (outliers)
    drift: tuple = (0.0, 0.0)  # constant image shift of the fit (µm)
    pixel_um: float = 0.0
    problem: str | None = None  # why the fit was refused (None: usable)
    beamcal_id: str | None = None  # the beam calibration the images went through (None: none)

    def __post_init__(self):
        self.stage_to_beam = np.asarray(self.stage_to_beam, float).reshape(2, 2)

    def residuals(self) -> np.ndarray:
        """Measured − fitted image shift of each image (N, 2) µm."""
        if not self.shifts:
            return np.zeros((0, 2))
        s = np.array([a for a, _ in self.shifts], float)
        m = np.array([b for _, b in self.shifts], float)
        return m - (-(s @ self.stage_to_beam.T) + np.asarray(self.drift, float))

    def mirror_mismatch(self, axes: StageAxes) -> bool:
        """The images move against what the stage axes say (``B·M`` mirrored)."""
        return bool(np.linalg.det(self.correction(axes)) < 0)

    def suggested_axes(self) -> StageAxes:
        """The stage axes the measured map matches (the sign of each axis)."""
        b = self.stage_to_beam
        return StageAxes(mirror_x=bool(b[0, 0] < 0), mirror_y=bool(b[1, 1] < 0))

    def diagnostics(self, axes: StageAxes | None = None) -> dict:
        """Numbers for the result window: of ``B·M`` (the correction against the stage axes).  When
        the images move against the stage axes, the numbers are against the axes the measurement
        says (``axes_used``) – against the wrong ones they would be a 180° rotation and a mirror."""
        axes = axes or StageAxes()
        mismatch = self.mirror_mismatch(axes)
        used = self.suggested_axes() if mismatch else axes
        c = self.correction(used)
        p = linear_params(c)
        res = self.residuals()
        px = self.pixel_um or 1.0
        return {
            "rotation_mrad": p["rotation_mrad"], "rotation_deg": math.degrees(p["rotation_mrad"] * 1e-3),
            "scale": p["scale"], "scale_x": float(np.hypot(*c[:, 0])), "scale_y": float(np.hypot(*c[:, 1])),
            "stretch": p["stretch"], "shear_mrad": p["shear"] * 1e3,
            "mirror_mismatch": mismatch, "suggested_axes": self.suggested_axes(), "axes_used": used,
            "drift_um": tuple(map(float, self.drift)),
            "rms_um": self.residual_um, "rms_px": self.residual_um / px,
            "max_residual_um": float(np.max(np.hypot(res[:, 0], res[:, 1]))) if len(res) else 0.0,
            "weakest_match": float(min(q for i, q in enumerate(self.quality) if i not in self.rejected))
            if self.quality else float("nan"),
            "rejected": list(self.rejected),
            "matrix": c,
        }

    def valid_for(self, condition: Condition, beamcal=None) -> bool:
        """Measured at *condition* (magnification, preset, WD) through the beam calibration
        *beamcal* that applies now (None: none)."""
        bid = getattr(beamcal, "id", None) if beamcal is not None else None
        return self.condition.matches(condition) and self.beamcal_id == bid

    def correction(self, axes: StageAxes) -> np.ndarray:
        """``B·M``: how true WCS µm along the stage axes appear in BCS (identity when nominal)."""
        return self.stage_to_beam @ axes.matrix

    def describe(self, axes: StageAxes | None = None) -> str:
        p = linear_params(self.correction(axes or StageAxes()))
        mirror = self.mirror_mismatch(axes or StageAxes())
        s = (f"rot {p['rotation_mrad']:+.2f} mrad, scale {p['scale']:.4f}"
             + (f", stretch {p['stretch'] * 1e3:+.1f}‰" if abs(p["stretch"]) > 5e-4 else "")
             + (" (MIRRORED: check the stage axes)" if mirror else ""))
        s += f" @ {self.condition.describe()}, rms {self.residual_um * 1e3:.0f} nm"
        return s + ("" if self.beamcal_id else " (no beam calibration)")


def shift_pattern(pattern: str, d: float) -> np.ndarray:
    """Stage shifts (N, 2) of the calibration, the centre (0, 0) first."""
    if pattern == "grid":
        pts = [(i * d, j * d) for j in (0, -1, 1) for i in (0, -1, 1)]
    elif pattern == "cross":
        pts = [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]
    else:
        raise ValueError("pattern must be 'grid' or 'cross'")
    out = np.array(pts, float)
    return out[np.argsort(np.hypot(out[:, 0], out[:, 1]), kind="stable")]


# ====================================================================== image shift
def _window(shape) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _upsampled_dft(data, region, factor, offsets):
    """Upsampled inverse DFT of *data* (a cross-power spectrum) on a *region* × *region* grid at
    1/*factor* pixel around *offsets* (row, col) – Guizar-Sicairos et al., Opt. Lett. 33, 156 (2008)."""
    ny, nx = data.shape
    kc = np.fft.ifftshift(np.arange(nx)) - nx // 2
    kr = np.fft.ifftshift(np.arange(ny)) - ny // 2
    c = np.arange(region) - offsets[1]
    r = np.arange(region) - offsets[0]
    col_kernel = np.exp((-2j * np.pi / (nx * factor)) * np.outer(kc, c))
    row_kernel = np.exp((-2j * np.pi / (ny * factor)) * np.outer(r, kr))
    return row_kernel @ data @ col_kernel


def _crops(a, b, gx: int, gy: int):
    """The overlapping parts of *a* and *b* when *b*'s content is *a*'s moved by (gx, gy) pixels."""
    ny, nx = a.shape
    x0, x1 = max(0, gx), nx + min(0, gx)
    y0, y1 = max(0, gy), ny + min(0, gy)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise FieldCalError("the images hardly overlap (stage shift too large for the image)")
    return a[y0 - gy:y1 - gy, x0 - gx:x1 - gx], b[y0:y1, x0:x1]


#: a match with a peak-to-sidelobe ratio below this is weak (a featureless or very noisy image)
WEAK_MATCH_PSR = 6.0


def prefilter(img: np.ndarray, denoise_px: float = 1.5, background_px: float = 0.0) -> np.ndarray:
    """De-noise (Gaussian low-pass, σ *denoise_px*) and flatten (minus a blur of σ *background_px*:
    brightness gradients, charging) an image for matching.  Both images of a pair must get the same."""
    from scipy import ndimage

    a = np.asarray(img, float)
    out = ndimage.gaussian_filter(a, denoise_px, mode="reflect") if denoise_px > 0 else a.copy()
    if background_px > 0:
        out -= ndimage.gaussian_filter(a, background_px, mode="reflect")
    return out


def _psr(c: np.ndarray, iy: int, ix: int, exclude: int = 5) -> float:
    """Peak-to-sidelobe ratio of a (circular) correlation surface around its peak at (iy, ix)."""
    ny, nx = c.shape
    mask = np.ones_like(c, bool)
    ry = (np.arange(-exclude, exclude + 1) + iy) % ny
    rx = (np.arange(-exclude, exclude + 1) + ix) % nx
    mask[np.ix_(ry, rx)] = False
    side = c[mask]
    sd = float(side.std())
    return float((c[iy, ix] - side.mean()) / sd) if sd > 0 else float("inf")


def phase_correlation(ref: np.ndarray, img: np.ndarray, guess=(0.0, 0.0), upsample: int = 20,
                      whitening: float = 1.0) -> tuple[float, float, float]:
    """Shift (dcol, drow) in pixels that moves *ref*'s content onto *img*'s, and the match quality
    (the correlation's peak-to-sidelobe ratio; > :data:`WEAK_MATCH_PSR` is a clear match).
    *guess*: the expected shift – only the overlapping parts are compared, so shifts of a large
    part of the image work.  *whitening*: the cross-power spectrum is divided by its magnitude to
    this power (1: pure phase correlation – sharp, but noise counts as much as the feature;
    ≈ 0.6: robust on noisy, low-contrast images); sub-pixel by an upsampled DFT."""
    a = np.asarray(ref, float)
    b = np.asarray(img, float)
    if a.shape != b.shape:
        raise ValueError("images must have the same shape")
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    a, b = _crops(a, b, gx, gy)
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * w)
    fb = np.fft.fft2((b - b.mean()) * w)
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    r = cross / (mag ** float(whitening) + 1e-12 * (mag.max() ** float(whitening) + 1e-300))
    c = np.fft.ifft2(r).real
    ny, nx = c.shape
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    quality = _psr(c, iy, ix)
    sy = iy - ny if iy > ny // 2 else iy
    sx = ix - nx if ix > nx // 2 else ix
    if upsample > 1:
        region = int(np.ceil(upsample * 3))  # ±1.5 px: the integer peak may be one pixel off in noise
        centre = region // 2
        # sub-pixel on the plain cross-power (a whitened peak is biased towards whole pixels)
        up = _upsampled_dft(np.conj(cross), region, upsample, (centre - sy * upsample, centre - sx * upsample))
        uy, ux = np.unravel_index(int(np.argmax(np.abs(up))), up.shape)
        sy = sy + (uy - centre) / upsample
        sx = sx + (ux - centre) / upsample
    return float(sx + gx), float(sy + gy), quality


def fit_robust(stage_shifts, image_shifts, floor: float = 0.0, min_images: int = 5,
               k: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """:func:`fit_stage_to_beam`, dropping the worst image while its residual is beyond *k* × the
    robust spread of the others (and above *floor*), keeping at least *min_images*.  Returns
    (B, drift, residuals of every image (N, 2), indices dropped)."""
    s = np.atleast_2d(np.asarray(stage_shifts, float))
    d = np.atleast_2d(np.asarray(image_shifts, float))
    keep = list(range(len(s)))
    dropped: list = []
    while True:
        b, drift, _ = fit_stage_to_beam(s[keep], d[keep])
        res = d - (-(s @ b.T) + drift)
        if len(keep) <= min_images:
            break
        n = np.hypot(res[keep, 0], res[keep, 1])
        worst = int(np.argmax(n))
        others = np.delete(n, worst)
        spread = 1.4826 * float(np.median(others)) if len(others) else 0.0
        if n[worst] <= max(k * spread, floor):
            break
        dropped.append(keep.pop(worst))
    return b, drift, res, sorted(dropped)


def fit_stage_to_beam(stage_shifts, image_shifts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares ``image_shift = −B·Δs + drift``: returns (B, drift, residuals (N, 2))."""
    s = np.atleast_2d(np.asarray(stage_shifts, float))
    d = np.atleast_2d(np.asarray(image_shifts, float))
    if len(s) < 3 or np.linalg.matrix_rank(s - s.mean(0)) < 2:
        raise FieldCalError("need stage shifts along two directions")
    x = np.column_stack([-s, np.ones(len(s))])
    sol, *_ = np.linalg.lstsq(x, d, rcond=None)
    b = sol[:2].T
    drift = sol[2]
    res = d - x @ sol
    return b, drift, res


# ====================================================================== procedure
def measure_field_calibration(hw, config: FieldCalConfig, position, condition: Condition,
                              log: Callable[[str], None] | None = None, check: Callable[[], None] | None = None,
                              axes: StageAxes | None = None, on_image: Callable | None = None,
                              beamcal=None) -> FieldCalResult:
    """Run the calibration on *hw* (a :class:`~easyebl.litho.runner.LithoHardware` with
    ``capture_image``) at stage *position* (SCS).  *check()* raises to abort between images;
    ``on_image(k, n, shift, image, reference)`` after each image.  The stage is back at *position*
    afterwards.  A refused fit raises :class:`FieldCalError` with the result attached.

    Each image is matched for the stage axes as set and for the other mirrorings too (the best
    match wins), so a wrong mirror setting shows in the result instead of spoiling it.  The image
    shifts (XCS) are converted to BCS through *beamcal* (its linear part; None: XCS = BCS)."""
    log = log or (lambda text: None)
    check = check or (lambda: None)
    field_um = float(hw.field_um())
    d = config.step_fraction * field_um
    shifts = shift_pattern(config.pattern, d)
    pos = np.asarray(position, float)
    if np.any(pos + shifts.min(0) < 0):
        raise FieldCalError("the calibration shifts would take the stage below 0")
    width = config.image_fraction * field_um
    if d > 0.45 * width:
        raise FieldCalError("the stage shifts are too large for the image (lower the step or widen the image)")
    log(f"field calibration at ({pos[0]:.1f}, {pos[1]:.1f}) µm: {len(shifts)} images, shifts ±{d:.2f} µm, "
        f"field {field_um:.4g} µm, {condition.describe()}")
    images, pixel = [], None
    try:
        for k, ds in enumerate(shifts):
            check()
            hw.move_stage(float(pos[0] + ds[0]), float(pos[1] + ds[1]))
            if config.settle_s > 0:
                time.sleep(config.settle_s)
            check()
            img, px = hw.capture_image(width, config.image_px, config.frames)
            pixel = px
            images.append(np.asarray(img, float))
            if on_image is not None:
                on_image(k, len(shifts), (float(ds[0]), float(ds[1])), images[-1], images[0])
            log(f"field calibration image {k + 1}/{len(shifts)}: stage shift ({ds[0]:+.2f}, {ds[1]:+.2f}) µm")
    finally:
        try:
            hw.move_stage(float(pos[0]), float(pos[1]))
        except Exception:  # noqa: BLE001 - the measurement error (if any) is the one to report
            pass
    bg = config.background_fraction * float(config.image_px)
    filtered = [prefilter(im, config.denoise_px, bg) for im in images]
    ref = filtered[0]
    meas, quality, predicted = [], [], []
    nominal = axes.matrix if axes is not None else np.eye(2)
    mirrors = [nominal] + [np.diag(d) for d in ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))
                           if not np.allclose(np.diag(d), nominal)]
    for img, ds in zip(filtered, shifts):
        predicted.append(tuple(map(float, -nominal @ ds)))  # the nominal image shift (µm)
        best = None
        for m in mirrors:
            e = -m @ ds
            try:
                dc, dr, q = phase_correlation(ref, img, guess=(e[0] / pixel[0], -e[1] / pixel[1]),
                                              whitening=config.whitening)
            except FieldCalError:
                continue
            if best is None or q > best[2] + 1e-9:
                best = (dc, dr, q)
            if not ds.any():
                break  # the reference itself
        if best is None:
            raise FieldCalError("the images hardly overlap (stage shift too large for the image)")
        dc, dr, q = best
        meas.append((dc * pixel[0], -dr * pixel[1]))  # rows run down, y up
        quality.append(q)
    meas = np.array(meas)
    if beamcal is not None:
        meas = np.atleast_2d(beamcal.linear_to_bcs(meas))
    b, drift, res, dropped = fit_robust(shifts, meas, floor=0.5 * max(pixel))
    used = [i for i in range(len(shifts)) if i not in dropped]
    rms = float(np.sqrt(np.mean(np.sum(res[used] ** 2, axis=1))))
    out = FieldCalResult(b, condition, field_um, (float(pos[0]), float(pos[1])), rms, len(shifts),
                         shifts=[(tuple(map(float, s)), tuple(map(float, m))) for s, m in zip(shifts, meas)],
                         predicted=predicted, quality=quality, drift=(float(drift[0]), float(drift[1])),
                         pixel_um=float(max(pixel)), beamcal_id=getattr(beamcal, "id", None), rejected=dropped)
    if dropped:
        log("field calibration: image(s) " + ", ".join(str(i + 1) for i in dropped)
            + " left out (far off the others: a bad match)")
    if rms > config.max_residual_px * max(pixel):
        out.problem = (f"the images do not match well (fit rms {rms * 1e3:.0f} nm = {rms / max(pixel):.1f} px; "
                       "is there a feature with structure in the field?)")
        raise FieldCalError(out.problem, out)
    log(f"field calibration: {out.describe(axes)} (drift {drift[0] * 1e3:+.0f}, {drift[1] * 1e3:+.0f} nm, "
        f"weakest match {min(quality[i] for i in used):.1f})")
    if axes is not None and out.mirror_mismatch(axes):
        sug = out.suggested_axes()
        log("field calibration: the images move against the stage axes setting; it should be mirror X "
            f"{'on' if sug.mirror_x else 'off'}, mirror Y {'on' if sug.mirror_y else 'off'} (Hardware → Configure… → Microscope)")
    return out
