"""Beam calibration: the scanner → beam map from one image of a square calibration grid (no Qt).

Coordinates
-----------
* **XCS** – scanner coordinates: the nominal µm of the scan (deflection volts ÷ V/µm), i.e. the
  coordinates of the image pixels.  Origin at the field centre, +x along the image columns, +y
  *up*: image rows run down, so row 0 is the top.  Pixel ``(r, c)`` of an ``ny × nx`` image with
  pixel size ``(px, py)`` sits at ``x = (c − (nx−1)/2)·px``, ``y = ((ny−1)/2 − r)·py``.
* **BCS** – true µm of the beam on the sample, along the scan axes, same origin.

Model
-----
::

    bcs = D(L · xcs),   D(q) = q · (1 + k·|q|²),   L = [[sx, sh], [0, sy]]   (sx, sy > 0)

``L`` holds the scale of each scan axis and the shear (non-orthogonality) – a rotation is not part
of it (BCS is along the scan axes by definition); ``k`` (1/µm²) is the radial (pincushion for
k > 0) distortion.  The writing direction uses the lowest-order, single-step inverse
(``k·r² ≪ 1``)::

    xcs ≈ L⁻¹ · b · (1 − k·|b|²)

whose error is ≈ ``3·k²·r⁵`` (i.e. ``3·(k r²)²·r``).

Measurement
-----------
One ultra-high-resolution image (e.g. 5000² px over the full field) of a grid with a known square
pitch (lines or squares: only a 2-D periodic square lattice is assumed).  The grid lies at an
unknown rotation θ and shift ``s``: in the grid's own frame ``u = R(−θ)·(bcs − s)`` the lattice
sites are at ``u = pitch·(n, m)`` (for the synthetic "lines": the line centres).  θ and ``s`` are
nuisance parameters, fitted and reported only as info (θ modulo 90°, in (−45°, 45°]; ``s``
modulo the lattice).

1. the two fundamental lattice peaks from a windowed FFT of the central part (g1 ≈ +x, g2 ≈ +y);
2. per fundamental, complex demodulation ``img·exp(−2πi·Φ(x))`` with a carrier phase Φ, a
   Gaussian low-pass of ``filter_periods`` periods and subsampling on a coarse grid (done as two
   small matrix products per row chunk, float32/complex64), giving the lattice phase there;
3. least-squares fit of ``Φ_i/2π = e_iᵀ·D(W·x)/pitch + c_i`` (``W = R(−θ)·L``, general 2 × 2);
4. two more passes with the fitted (non-linear) model as carrier: the remaining phase is then
   small and flat, so the low-pass does not bias it (a chirped carrier would bias the scale by
   ≈ 4·k·σ²) and no unwrapping is needed;
5. ``W = Q·U`` (QR, U with positive diagonal): ``L = U``, θ from ``Q``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
import math
from typing import Callable

import numpy as np

__all__ = ["BeamCalError", "BeamCalibration", "BeamCalConfig", "fit_grid", "synthetic_grid",
           "measure_beam_calibration"]

_FFT_CROP = 2048  # px: the central part used for the coarse lattice
_PEAK_SNR = 12.0  # fundamental / median spectrum in the search annulus
_SQUARE_TOL = 0.05  # |g2| / |g1| within ±5 %
_ORTHO_TOL = 0.08  # rad: angle between g1 and g2 within 90° ± this
_REFINE_PASSES = 2


class BeamCalError(RuntimeError):
    """A calibration that failed; *result* holds what was measured when the fit was refused."""

    def __init__(self, message: str, result: "BeamCalibration | None" = None):
        super().__init__(message)
        self.result = result


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class BeamCalibration:
    """``bcs = D(L·xcs)`` with ``L = [[sx, shear], [0, sy]]``, ``D(q) = q·(1 + k|q|²)``."""

    sx: float = 1.0
    sy: float = 1.0
    shear: float = 0.0
    k: float = 0.0  # 1/µm²
    # info
    rotation_mrad: float = 0.0  # grid rotation (nuisance, modulo 90°)
    shift_um: tuple = (0.0, 0.0)  # grid shift in BCS (nuisance, modulo the lattice)
    rms_um: float = 0.0  # fit residual
    max_um: float = 0.0
    pitch_um: float = 0.0
    field_um: float = 0.0  # image width (XCS µm)
    pixels: tuple = (0, 0)  # (nx, ny)
    used: float = 0.0  # fraction of the image area that went into the fit
    condition: object = None  # set by the caller
    created: str = field(default_factory=_now)
    position: object = None
    name: str = ""

    # ------------------------------------------------------------------ the map
    @property
    def L(self) -> np.ndarray:
        return np.array([[self.sx, self.shear], [0.0, self.sy]], float)

    @property
    def id(self) -> str:
        """Identifies this calibration (a field calibration records the one it was measured through)."""
        return f"{self.created}|{self.sx:.9g},{self.sy:.9g},{self.shear:.9g},{self.k:.6g}"

    @classmethod
    def identity(cls) -> "BeamCalibration":
        return cls()

    @property
    def is_identity(self) -> bool:
        return self.sx == 1.0 and self.sy == 1.0 and self.shear == 0.0 and self.k == 0.0

    def to_bcs(self, xcs) -> np.ndarray:
        """Exact forward map, (2,) or (N, 2) µm."""
        x = np.asarray(xcs, float)
        q = x @ self.L.T
        return q * (1.0 + self.k * np.sum(q * q, axis=-1, keepdims=True))

    def to_xcs(self, bcs) -> np.ndarray:
        """Single-step inverse ``L⁻¹·b·(1 − k|b|²)`` (error ≈ 3k²r⁵), (2,) or (N, 2) µm."""
        b = np.asarray(bcs, float)
        q = b * (1.0 - self.k * np.sum(b * b, axis=-1, keepdims=True))
        return q @ np.linalg.inv(self.L).T

    def linear_to_bcs(self, v) -> np.ndarray:
        """``L·v`` only (shift vectors)."""
        return np.asarray(v, float) @ self.L.T

    def linear_to_xcs(self, v) -> np.ndarray:
        """``L⁻¹·v`` only (shift vectors)."""
        return np.asarray(v, float) @ np.linalg.inv(self.L).T

    def distortion_at(self, r_um) -> np.ndarray | float:
        """Radial displacement ``k·r³`` (µm) at radius *r_um*."""
        r = np.asarray(r_um, float)
        out = self.k * r ** 3
        return float(out) if out.ndim == 0 else out

    def edge_distortion_um(self) -> float:
        """The radial distortion at the field corner (radius field/√2)."""
        return float(self.distortion_at(self.field_um / math.sqrt(2.0)))

    def describe(self) -> str:
        if self.is_identity:
            return "identity (no beam calibration)"
        text = (f"sx {(self.sx - 1) * 1e6:+.0f} ppm, sy {(self.sy - 1) * 1e6:+.0f} ppm, "
                f"shear {self.shear * 1e3:+.3f} mrad, k {self.k:.3g} µm⁻²")
        if self.field_um:
            text += f" ({self.edge_distortion_um() * 1e3:+.0f} nm at the corner)"
        if self.rms_um:
            text += f", rms {self.rms_um * 1e3:.1f} nm"
        return text

    # ------------------------------------------------------------------ storage
    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["shift_um"] = [float(v) for v in self.shift_um]
        d["pixels"] = [int(v) for v in self.pixels]
        for key in ("condition", "position"):
            v = d[key]
            if hasattr(v, "to_dict"):
                d[key] = v.to_dict()
            elif isinstance(v, tuple):
                d[key] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict | None, condition_from_dict: Callable | None = None) -> "BeamCalibration":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "shift_um" in d:
            d["shift_um"] = tuple(float(v) for v in d["shift_um"])
        if "pixels" in d:
            d["pixels"] = tuple(int(v) for v in d["pixels"])
        if condition_from_dict is not None and d.get("condition") is not None:
            d["condition"] = condition_from_dict(d["condition"])
        return cls(**d)


@dataclass
class BeamCalConfig:
    """How to calibrate (the Beam calibration step of the schedule)."""

    pitch_um: float = 1.0
    pixels: int = 5000  # image: pixels × pixels over the current full field
    frames: int = 1
    max_rms_um: float = 0.05  # a fit worse than this (rms) is refused
    position: str = ""  # named stage position of the calibration grid
    filter_periods: float = 3.0  # low-pass sigma, in grid periods
    #: before the image: zoom to *focus_magnification* at the grid and wait for the user to focus and stigmate
    #: by hand (the WD then read back becomes the run's WD); off: image at once
    manual_focus: bool = True
    focus_magnification: int = 10000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BeamCalConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ====================================================================== helpers
def _pixel_pair(pixel_um) -> tuple[float, float]:
    p = np.atleast_1d(np.asarray(pixel_um, float)).ravel()
    px, py = (float(p[0]), float(p[0])) if p.size == 1 else (float(p[0]), float(p[1]))
    if px <= 0 or py <= 0:
        raise BeamCalError("the pixel size must be positive")
    return px, py


def _model_cycles(params, x, y, pitch: float, r0: float):
    """Lattice phase (cycles) of both fundamentals at XCS points *x*, *y* (broadcastable)."""
    w00, w01, w10, w11, kk, c1, c2 = params
    q0 = w00 * x + w01 * y
    q1 = w10 * x + w11 * y
    f = 1.0 + (kk / (r0 * r0)) * (q0 * q0 + q1 * q1)
    return q0 * f / pitch + c1, q1 * f / pitch + c2


def _gauss_matrix(n: int, centres: np.ndarray, sigma: float) -> np.ndarray:
    """(len(centres), n) normalised Gaussian weights, truncated at 4σ."""
    idx = np.arange(n, dtype=float)
    d = idx[None, :] - centres[:, None]
    g = np.exp(-0.5 * (d / sigma) ** 2)
    g[np.abs(d) > 4.0 * sigma] = 0.0
    g /= g.sum(axis=1, keepdims=True)
    return g.astype(np.float32)


def _demodulate(img, mean, params, pitch, r0, px, py, gr, gc, rows_s, sigma_r):
    """Low-passed, subsampled ``img·exp(−2πi·Φ_i)`` for both fundamentals: (2, my, mx) complex."""
    ny, nx = img.shape
    xs = (np.arange(nx) - (nx - 1) / 2.0) * px
    my = gr.shape[0]
    acc = np.zeros((2, my, nx), np.complex64)
    chunk = max(16, (1 << 20) // nx)
    reach = 4.0 * sigma_r + 1
    for r0_ in range(0, ny, chunk):
        r1 = min(ny, r0_ + chunk)
        sel = np.nonzero((rows_s >= r0_ - reach) & (rows_s <= r1 - 1 + reach))[0]
        if sel.size == 0:
            continue
        ys = ((ny - 1) / 2.0 - np.arange(r0_, r1))[:, None] * py
        block = img[r0_:r1].astype(np.float32) - np.float32(mean)
        weights = gr[sel, r0_:r1]
        for i, m in enumerate(_model_cycles(params, xs[None, :], ys, pitch, r0)):
            frac = (m - np.floor(m)).astype(np.float32)
            frac *= np.float32(-2.0 * np.pi)
            z = np.empty(frac.shape, np.complex64)
            z.real = np.cos(frac)
            z.imag = np.sin(frac)
            z *= block
            acc[i, sel] += weights @ z
    return np.stack([a @ gc.T for a in acc])


def _mag_at(mag, r, c):
    cy = mag.shape[0]
    if c < 0:
        r, c = -r, -c
    if c >= mag.shape[1]:
        return np.nan
    return mag[r % cy, c]


def _refine_peak(mag, r, c) -> tuple[float, float]:
    """Sub-bin peak position: parabola through log magnitudes along each axis."""
    out = []
    for dr, dc in ((1, 0), (0, 1)):
        a = _mag_at(mag, r - dr, c - dc)
        b = _mag_at(mag, r, c)
        e = _mag_at(mag, r + dr, c + dc)
        if not (np.isfinite(a) and np.isfinite(e)) or min(a, b, e) <= 0:
            out.append(0.0)
            continue
        la, lb, le = np.log(a), np.log(b), np.log(e)
        den = la - 2 * lb + le
        out.append(float(np.clip(0.5 * (la - le) / den, -0.5, 0.5)) if den < 0 else 0.0)
    return r + out[0], c + out[1]


def _coarse_lattice(img, px, py, pitch) -> tuple[np.ndarray, np.ndarray]:
    """The two fundamentals (cycles/µm, XCS) from the FFT of the central part: g1 ≈ +x, g2 ≈ +y."""
    from scipy import fft as sfft

    ny, nx = img.shape
    cy, cx = min(ny, _FFT_CROP), min(nx, _FFT_CROP)
    r0, c0 = (ny - cy) // 2, (nx - cx) // 2
    a = img[r0:r0 + cy, c0:c0 + cx].astype(np.float32)
    a -= a.mean()
    a *= np.outer(np.hanning(cy), np.hanning(cx)).astype(np.float32)
    mag = np.abs(sfft.rfft2(a, workers=-1))
    fr = sfft.fftfreq(cy)  # cycles / row
    fc = sfft.rfftfreq(cx)  # cycles / column
    fx = (fc / px)[None, :]
    fy = (-fr / py)[:, None]  # rows run down, y up
    fmag = np.hypot(fx, fy)
    ann = (fmag > 0.75 / pitch) & (fmag < 1.33 / pitch)
    if ann.sum() < 16:
        raise BeamCalError("the grid pitch is not resolved by the image (too few pixels per period)")
    ang = np.arctan2(np.broadcast_to(fy, mag.shape), np.broadcast_to(fx, mag.shape))
    peaks = []
    mask = ann.copy()
    for _ in range(2):
        m = np.where(mask, mag, 0.0)
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        med = float(np.median(mag[ann]))
        if not med > 0 or mag[r, c] < _PEAK_SNR * med:
            raise BeamCalError(f"no grid found in the image (lattice peak {mag[r, c] / max(med, 1e-30):.1f} × "
                               f"the background, need {_PEAK_SNR:g})")
        rr, cc = _refine_peak(mag, r, c)
        g = np.array([cc / cx / px, -(rr if rr <= cy / 2 else rr - cy) / cy / py])
        peaks.append(g)
        a0 = math.atan2(g[1], g[0])
        d = np.angle(np.exp(2j * (ang - a0))) / 2  # angle difference modulo 180°
        mask &= np.abs(d) > math.radians(30)
    ga, gb = peaks
    ratio = np.linalg.norm(gb) / np.linalg.norm(ga)
    cosang = abs(float(ga @ gb)) / (np.linalg.norm(ga) * np.linalg.norm(gb))
    if abs(ratio - 1) > _SQUARE_TOL or math.asin(min(1.0, cosang)) > _ORTHO_TOL:
        raise BeamCalError(f"the lattice is not square (period ratio {ratio:.3f}, angle "
                           f"{math.degrees(math.acos(min(1.0, cosang))):.1f}°)")
    cands = [ga, -ga, gb, -gb]
    angs = [math.atan2(g[1], g[0]) for g in cands]
    i1 = int(np.argmin([abs(a) for a in angs]))
    target = angs[i1] + math.pi / 2
    i2 = int(np.argmin([abs(math.remainder(a - target, 2 * math.pi)) for a in angs]))
    return cands[i1], cands[i2]


def _unwrap2d(phase: np.ndarray) -> np.ndarray:
    """Smooth-field unwrap: along each row, then the first column aligns the rows."""
    u = np.unwrap(phase, axis=1)
    col = np.unwrap(phase[:, 0])
    return u + (col - u[:, 0])[:, None]


def _fit(params0, xs, ys, meas, keep, pitch, r0):
    from scipy.optimize import least_squares

    def resid(p, x, y, m):
        a, b = _model_cycles(p, x, y, pitch, r0)
        return np.concatenate([a - m[0], b - m[1]])

    x, y = xs[keep], ys[keep]
    m = (meas[0][keep], meas[1][keep])
    p = np.asarray(params0, float)
    for _ in range(3):  # fit, drop outliers (unwrap slips, defects), refit
        sol = least_squares(resid, p, args=(x, y, m), x_scale="jac", method="lm")
        p = sol.x
        r = sol.fun.reshape(2, -1)
        e = np.hypot(r[0], r[1])
        mad = 1.4826 * np.median(np.abs(r))
        bad = e > max(6.0 * mad * math.sqrt(2), 0.02)
        if not bad.any():
            break
        x, y, m = x[~bad], y[~bad], (m[0][~bad], m[1][~bad])
        if x.size < 20:
            raise BeamCalError("too few usable points in the image")
    return p, r, x.size


# ====================================================================== fit
def fit_grid(img, pixel_um, pitch_um: float = 1.0, max_rms_um: float = 0.05,
             on_progress: Callable[[float, str], None] | None = None,
             filter_periods: float = 3.0) -> BeamCalibration:
    """Fit the beam calibration to an image of a square grid of pitch *pitch_um* (see the module
    docstring for the conventions).  *pixel_um*: scalar or (px, py).  ``on_progress(fraction,
    text)`` reports the stages.  Raises :class:`BeamCalError` (with the result when the fit was
    refused for its residual)."""
    progress = on_progress or (lambda f, t: None)
    img = np.asarray(img)
    if img.ndim != 2:
        raise BeamCalError("the image must be 2-D")
    ny, nx = img.shape
    px, py = _pixel_pair(pixel_um)
    pitch = float(pitch_um)
    if pitch / max(px, py) < 3.0:
        raise BeamCalError(f"{pitch / max(px, py):.1f} px per grid period: need at least 3")
    if not np.isfinite(img).all() or float(img.max()) == float(img.min()):
        raise BeamCalError("the image is blank")
    progress(0.0, "lattice")
    g1, g2 = _coarse_lattice(img, px, py, pitch)
    r0 = 0.5 * math.hypot(nx * px, ny * py)  # scale for k

    # subsample grid, away from where the filter runs off the image
    sig_r = filter_periods * pitch / py
    sig_c = filter_periods * pitch / px
    mr, mc = int(math.ceil(4 * sig_r)), int(math.ceil(4 * sig_c))
    step_r = max(1, int(round(0.75 * sig_r)))
    step_c = max(1, int(round(0.75 * sig_c)))
    rows_s = np.arange(mr, ny - mr, step_r, dtype=float)
    cols_s = np.arange(mc, nx - mc, step_c, dtype=float)
    if rows_s.size < 8 or cols_s.size < 8:
        raise BeamCalError("the image is too small for the grid pitch and filter")
    gr = _gauss_matrix(ny, rows_s, sig_r)
    gc = _gauss_matrix(nx, cols_s, sig_c)
    xs = np.broadcast_to(((cols_s - (nx - 1) / 2.0) * px)[None, :], (rows_s.size, cols_s.size))
    ys = np.broadcast_to((((ny - 1) / 2.0 - rows_s) * py)[:, None], (rows_s.size, cols_s.size))
    mean = float(np.mean(img, dtype=np.float64))

    params = np.array([pitch * g1[0], pitch * g1[1], pitch * g2[0], pitch * g2[1], 0.0, 0.0, 0.0])
    npass = 1 + _REFINE_PASSES
    resid = None
    n_used = 0
    for k in range(npass):
        progress(0.05 + 0.9 * k / npass, f"demodulation {k + 1}/{npass}")
        s = _demodulate(img, mean, params, pitch, r0, px, py, gr, gc, rows_s, sig_r)
        amp = np.abs(s).min(axis=0)
        keep = amp > 0.3 * np.median(amp)
        phase = np.angle(s)
        if k == 0:
            phase = np.stack([_unwrap2d(p) for p in phase])
        base = _model_cycles(params, xs, ys, pitch, r0)
        meas = (base[0] + phase[0] / (2 * np.pi), base[1] + phase[1] / (2 * np.pi))
        if k == 0:  # pull the integer cycles of the unwrap into the offsets
            med = [np.median(meas[i][keep] - base[i][keep]) for i in range(2)]
            params[5:] += med
        params, resid, n_used = _fit(params, xs, ys, meas, keep, pitch, r0)
    progress(0.97, "decomposition")

    w = params[:4].reshape(2, 2)
    q, u = np.linalg.qr(w)
    sgn = np.sign(np.diag(u))
    q, u = q * sgn[None, :], u * sgn[:, None]
    theta = -math.atan2(q[1, 0], q[0, 0])  # W = R(−θ)·L
    c = (params[5:] + 0.5) % 1.0 - 0.5
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    shift = -pitch * (rot @ c)
    e = np.hypot(resid[0], resid[1]) * pitch
    rms = float(np.sqrt(np.mean(e ** 2)))
    area = (rows_s.size * step_r * cols_s.size * step_c) / float(nx * ny)
    out = BeamCalibration(sx=float(u[0, 0]), sy=float(u[1, 1]), shear=float(u[0, 1]),
                          k=float(params[4] / r0 ** 2), rotation_mrad=theta * 1e3,
                          shift_um=(float(shift[0]), float(shift[1])), rms_um=rms, max_um=float(e.max()),
                          pitch_um=pitch, field_um=float(nx * px), pixels=(int(nx), int(ny)),
                          used=float(min(1.0, area) * n_used / rows_s.size / cols_s.size))
    progress(1.0, "done")
    if not np.isfinite([out.sx, out.sy, out.shear, out.k]).all() or out.sx <= 0 or out.sy <= 0:
        raise BeamCalError("the fit did not converge", out)
    if rms > max_rms_um:
        raise BeamCalError(f"the grid does not fit well (rms {rms * 1e3:.0f} nm > {max_rms_um * 1e3:.0f} nm)", out)
    return out


# ====================================================================== synthetic
def synthetic_grid(n_px: int, field_um: float, pitch_um: float = 1.0, sx: float = 1.0, sy: float = 1.0,
                   shear: float = 0.0, k: float = 0.0, rotation_rad: float = 0.0, shift_um=(0.0, 0.0),
                   line_width_um: float = 0.2, noise: float = 0.0, seed: int = 0, kind: str = "lines"
                   ) -> tuple[np.ndarray, tuple[float, float]]:
    """An ``n_px × n_px`` image of a square grid over *field_um* (XCS) as the scanner records it:
    each pixel's XCS → ``bcs = D(L·xcs)`` → grid frame ``u = R(−rotation)·(bcs − shift)``.

    ``kind="lines"``: bright lines of *line_width_um* centred at ``u_x = n·pitch`` and
    ``u_y = m·pitch``; ``"squares"``: bright squares of side ``pitch − line_width_um`` between
    those lines (centred at ``(n + ½, m + ½)·pitch``).  Edges blurred by a Gaussian of 0.7 px,
    contrast 0…1, plus Gaussian noise of sd *noise*.  Returns ``(image float32, (px, py))``."""
    from scipy.special import erf

    if kind not in ("lines", "squares"):
        raise ValueError(f"unknown grid kind {kind!r}")
    n = int(n_px)
    px = float(field_um) / n
    blur = 0.7 * px
    lmat = np.array([[sx, shear], [0.0, sy]], float)
    ct, st = math.cos(rotation_rad), math.sin(rotation_rad)
    s0, s1 = float(shift_um[0]), float(shift_um[1])
    xs = (np.arange(n) - (n - 1) / 2.0) * px
    img = np.empty((n, n), np.float32)
    hw = 0.5 * line_width_um
    sq2 = math.sqrt(2.0) * blur

    def line(u):
        d = np.abs(u - pitch_um * np.round(u / pitch_um)).astype(np.float32)
        return 0.5 * (erf((hw - d) / sq2) + erf((hw + d) / sq2))

    chunk = max(1, (1 << 19) // n)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        y = ((n - 1) / 2.0 - np.arange(r0, r1))[:, None] * px
        q0 = lmat[0, 0] * xs[None, :] + lmat[0, 1] * y
        q1 = lmat[1, 1] * y + 0.0 * xs[None, :]
        f = 1.0 + k * (q0 * q0 + q1 * q1)
        d0, d1 = q0 * f - s0, q1 * f - s1
        a, b = line(ct * d0 + st * d1), line(-st * d0 + ct * d1)
        img[r0:r1] = np.maximum(a, b) if kind == "lines" else (1 - a) * (1 - b)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img += (noise * rng.standard_normal(img.shape, dtype=np.float32)).astype(np.float32)
    return img, (px, px)


# ====================================================================== procedure
def measure_beam_calibration(hw, config: BeamCalConfig, condition, log: Callable[[str], None] = print,
                             check: Callable[[], None] = lambda: None,
                             on_image: Callable | None = None) -> BeamCalibration:
    """Image the full field (``hw.capture_image(width_um, px, frames)`` → ``(image, (px_um,
    py_um))``, centred) at ``config.pixels`` and fit the grid.  *check()* raises to abort
    between stages; ``on_image(image, pixel_um)`` after the capture.  A refused fit raises
    :class:`BeamCalError` with the result attached (its condition set)."""
    log = log or (lambda text: None)
    check = check or (lambda: None)
    field_um = float(hw.field_um())
    check()
    log(f"beam calibration: imaging {field_um:.4g} µm at {config.pixels} px "
        f"({field_um / config.pixels * 1e3:.0f} nm pixels), grid pitch {config.pitch_um:g} µm")
    img, pixel = hw.capture_image(field_um, int(config.pixels), int(config.frames))
    if on_image is not None:
        on_image(img, pixel)
    check()
    try:
        out = fit_grid(img, pixel, config.pitch_um, config.max_rms_um, filter_periods=config.filter_periods)
    except BeamCalError as e:
        if e.result is not None:
            e.result.condition = condition
            e.result.position = config.position or None
        raise
    out.condition = condition
    out.position = config.position or None
    check()
    log(f"beam calibration: {out.describe()} (grid rotation {out.rotation_mrad:+.1f} mrad, "
        f"{out.used * 100:.0f} % of the image used)")
    return out
