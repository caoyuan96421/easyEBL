"""Field calibration: the stage → beam map ``B`` from images at known stage shifts (no Qt).

At a fixed stage position with a recognisable feature (a Faraday cup, a calibration
sample …) the procedure

1. moves the stage there and takes a reference image (scan rotation 0, centred);
2. moves the stage by known shifts ``Δs`` (a 3 × 3 grid or a cross of ±d, d a fraction of
   the field) and takes an image at each;
3. measures each image's shift against the reference by phase correlation (sub-pixel) –
   a feature moves by ``−B·Δs`` in BCS;
4. fits ``B`` (and a constant drift) by least squares; the residuals tell how well the
   images matched.

The stage is the length reference (encoders): ``B`` holds the rotation between stage and
beam axes, the mirror of the stage axes and the beam's scale error against the global
field calibration (typically ±1 %).  The result is valid for the column condition it
was measured at (magnification and beam preset).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import time
from typing import Callable

import numpy as np

from .coords import StageAxes, linear_params

__all__ = ["FieldCalConfig", "FieldCalResult", "Condition", "phase_correlation", "shift_pattern", "fit_stage_to_beam",
           "measure_field_calibration", "FieldCalError"]


class FieldCalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Condition:
    """The column condition a calibration belongs to (preset None: not known / kept as it was)."""

    magnification: float | None
    preset: str | None = None

    def describe(self) -> str:
        m = f"×{self.magnification:g}" if self.magnification else "×?"
        return m + (f", beam {self.preset!r}" if self.preset else "")


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

    def __post_init__(self):
        self.stage_to_beam = np.asarray(self.stage_to_beam, float).reshape(2, 2)

    def valid_for(self, condition: Condition) -> bool:
        return self.condition == condition

    def correction(self, axes: StageAxes) -> np.ndarray:
        """``B·M``: how true WCS µm along the stage axes appear in BCS (identity when nominal)."""
        return self.stage_to_beam @ axes.matrix

    def describe(self, axes: StageAxes | None = None) -> str:
        p = linear_params(self.correction(axes or StageAxes()))
        mirror = np.linalg.det(self.correction(axes or StageAxes())) < 0
        s = (f"rot {p['rotation_mrad']:+.2f} mrad, scale {p['scale']:.4f}"
             + (f", stretch {p['stretch'] * 1e3:+.1f}‰" if abs(p["stretch"]) > 5e-4 else "")
             + (" (MIRRORED: check the stage axes)" if mirror else ""))
        return s + f" @ {self.condition.describe()}, rms {self.residual_um * 1e3:.0f} nm"


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


def phase_correlation(ref: np.ndarray, img: np.ndarray, guess=(0.0, 0.0), upsample: int = 20
                      ) -> tuple[float, float, float]:
    """Shift (dcol, drow) in pixels that moves *ref*'s content onto *img*'s, and the peak height
    (0…1, the match quality).  *guess*: the expected shift – only the overlapping parts are
    compared, so shifts of a large part of the image work; sub-pixel by an upsampled DFT."""
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
    r = cross / (np.abs(cross) + 1e-12)  # phase correlation: a sharp peak for the integer search
    c = np.fft.ifft2(r).real
    ny, nx = c.shape
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[iy, ix])
    sy = iy - ny if iy > ny // 2 else iy
    sx = ix - nx if ix > nx // 2 else ix
    if upsample > 1:
        region = int(np.ceil(upsample * 3))  # ±1.5 px: the phase-only integer peak may be one pixel off in noise
        centre = region // 2
        # sub-pixel on the plain cross-power (the phase-only peak is biased towards whole pixels)
        up = _upsampled_dft(np.conj(cross), region, upsample, (centre - sy * upsample, centre - sx * upsample))
        uy, ux = np.unravel_index(int(np.argmax(np.abs(up))), up.shape)
        sy = sy + (uy - centre) / upsample
        sx = sx + (ux - centre) / upsample
    return float(sx + gx), float(sy + gy), peak


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
                              axes: StageAxes | None = None) -> FieldCalResult:
    """Run the calibration on *hw* (a :class:`~easyebl.litho.runner.LithoHardware` with
    ``capture_image``) at stage *position* (SCS).  *check()* raises to abort between images.
    The stage is back at *position* afterwards."""
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
            log(f"field calibration image {k + 1}/{len(shifts)}: stage shift ({ds[0]:+.2f}, {ds[1]:+.2f}) µm")
    finally:
        try:
            hw.move_stage(float(pos[0]), float(pos[1]))
        except Exception:  # noqa: BLE001 - the measurement error (if any) is the one to report
            pass
    ref = images[0]
    meas, quality = [], []
    nominal = axes.matrix if axes is not None else np.eye(2)
    for img, ds in zip(images, shifts):
        e = -nominal @ ds  # the expected image shift (BCS µm) for the nominal stage → beam map
        dc, dr, q = phase_correlation(ref, img, guess=(e[0] / pixel[0], -e[1] / pixel[1]))
        meas.append((dc * pixel[0], -dr * pixel[1]))  # rows run down, y up
        quality.append(q)
    meas = np.array(meas)
    b, drift, res = fit_stage_to_beam(shifts, meas)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    if rms > config.max_residual_px * max(pixel):
        raise FieldCalError(f"the images do not match well (fit rms {rms * 1e3:.0f} nm = {rms / max(pixel):.1f} px; "
                            "is there a feature with structure in the field?)")
    out = FieldCalResult(b, condition, field_um, (float(pos[0]), float(pos[1])), rms, len(shifts),
                         shifts=[(tuple(s), tuple(m)) for s, m in zip(shifts, meas)])
    log(f"field calibration: {out.describe()} (drift {drift[0] * 1e3:+.0f}, {drift[1] * 1e3:+.0f} nm, "
        f"weakest match {min(quality):.2f})")
    return out
