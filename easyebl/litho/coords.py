"""Coordinate systems of the lithography module and the maps between them.

==========  =====================================================================================
GDS         coordinates of an Exposure's GDS file (top-level cell, µm)
CCS         chip coordinates (µm): ``ccs = gds + o`` (*o*: the Exposure's offset, no rotation).
            The axes are the layout's (as seen in the SEM image).
FCS         field coordinates: ``fcs = ccs − c_f`` (*c_f*: the field centre, a cell of the GDS)
SCS         stage coordinates (µm, the encoder read-back)
BCS         true µm of the nominal beam position along the scan axes, from the field centre
XCS         scanner coordinates: deflection volts ÷ (V/µm) of the global field calibration
==========  =====================================================================================

The chain, for a point of field *f* with the stage read back at ``ŝ``::

    scs = G(ccs) = A_g·ccs + t_g                       (ChipTransform; Global Align)
    s_f = G(c_f)                                       (stage target of the field)
    bcs = B·(A_g·fcs + G(c_f) − ŝ + δ_f)               (FieldMap; δ_f: Local marker, SCS)
    xcs = L⁻¹·bcs·(1 − k·|bcs|²)                       (beam calibration, one-step inverse)

* ``G`` – nominal ``A_g = M`` (the stage axes' mirror, :class:`StageAxes`), ``t_g`` = the chip
  origin on the stage; a Global Align fits it (2 markers: rotation, scale, shift on top of
  ``M``; 3–4: full affine).  It stays for every later Exposure of the chip.
* ``B`` – SCS → BCS: the rotation of the field calibration (:func:`stage_rotation`) with the
  mirror ``M``; nominal ``B = M``.  Its scale / shear are not written with (they come from the
  beam calibration, or from the alignment).
* Marker picking runs the chain backwards with the same maps (clicked xcs → bcs → scs), so
  within a field ``L`` and ``B`` cancel between measuring and writing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..alignment.fitting import fit_transform
from ..core.transforms import AffineTransform

__all__ = ["StageAxes", "S3000_AXES", "ChipTransform", "fit_chip", "stage_rotation", "FieldMap", "FieldCheck",
           "check_field", "linear_params", "describe_linear"]


def _pt(p) -> np.ndarray:
    a = np.asarray(p, float).reshape(-1)
    if a.shape != (2,):
        raise ValueError("a point is (x, y)")
    return a


@dataclass(frozen=True)
class StageAxes:
    """Stage axes against the beam axes: ``mirror_x`` – stage +X moves the sample image along −x.
    Set in Hardware → Configure… (Microscope)."""

    mirror_x: bool = False
    mirror_y: bool = False

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([-1.0 if self.mirror_x else 1.0, -1.0 if self.mirror_y else 1.0])

    def to_dict(self) -> dict:
        return {"mirror_x": self.mirror_x, "mirror_y": self.mirror_y}

    @classmethod
    def from_dict(cls, d: dict | None) -> "StageAxes":
        if not d:
            return S3000_AXES
        return cls(bool(d.get("mirror_x", False)), bool(d.get("mirror_y", False)))

    def describe(self) -> str:
        return f"mirror X {'on' if self.mirror_x else 'off'}, mirror Y {'on' if self.mirror_y else 'off'}"


#: the Hitachi S-3000: stage X runs against the SEM image's x (X → −x), Y along y
S3000_AXES = StageAxes(mirror_x=True, mirror_y=False)


# ====================================================================== helpers
def linear_params(a) -> dict:
    """Rotation (mrad), mean scale, stretch (sy/sx − 1) and shear of a 2×2 matrix."""
    d = AffineTransform.from_linear(np.asarray(a, float)).decompose()
    return {"rotation_mrad": d["rotation_rad"] * 1e3, "scale": math.sqrt(abs(d["sx"] * d["sy"])),
            "stretch": d["sy"] / d["sx"] - 1.0 if d["sx"] else 0.0, "shear": d["shear"]}


def describe_linear(a) -> str:
    p = linear_params(a)
    s = f"rot {p['rotation_mrad']:+.2f} mrad, scale {p['scale']:.5f}"
    if abs(p["stretch"]) > 1e-6 or abs(p["shear"]) > 1e-6:
        s += f", stretch {p['stretch'] * 1e3:+.2f}‰, shear {p['shear'] * 1e3:+.2f} mrad"
    return s


def stage_rotation(stage_to_beam, axes: StageAxes) -> np.ndarray:
    """The rotation part of a measured stage → beam map *B*, with the mirror of *axes*: ``R·M``
    (``R`` from the polar decomposition of ``B·M``; scale and shear dropped)."""
    m = axes.matrix
    c = np.asarray(stage_to_beam, float).reshape(2, 2) @ m
    u, _, vt = np.linalg.svd(c)
    r = u @ vt
    if np.linalg.det(r) < 0:  # mirrored against the axes setting: keep the nearest rotation
        u[:, -1] *= -1
        r = u @ vt
    return r @ m


# ====================================================================== CCS → SCS
@dataclass
class ChipTransform:
    """``scs = A·ccs + t`` – where the chip's coordinates are on the stage."""

    linear: np.ndarray = field(default_factory=lambda: np.eye(2))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    measured: bool = False  # from a Global Align (else nominal: the chip origin and the stage axes)
    rms_um: float | None = None  # Global Align residual (3–4 markers)
    step: int | None = None  # schedule step that measured it

    def __post_init__(self):
        self.linear = np.asarray(self.linear, float).reshape(2, 2)
        self.offset = _pt(self.offset)

    @classmethod
    def nominal(cls, origin, axes: StageAxes) -> "ChipTransform":
        return cls(axes.matrix.copy(), _pt(origin))

    def copy(self) -> "ChipTransform":
        return ChipTransform(self.linear.copy(), self.offset.copy(), self.measured, self.rms_um, self.step)

    def to_scs(self, ccs) -> np.ndarray:
        p = np.asarray(ccs, float)
        out = np.atleast_2d(p) @ self.linear.T + self.offset
        return out[0] if p.ndim == 1 else out

    def to_ccs(self, scs) -> np.ndarray:
        p = np.asarray(scs, float)
        out = (np.atleast_2d(p) - self.offset) @ np.linalg.inv(self.linear).T
        return out[0] if p.ndim == 1 else out

    def correction(self, axes: StageAxes) -> np.ndarray:
        """``A·M``: rotation / scale / shear of the chip on the stage beyond the stage axes."""
        return self.linear @ axes.matrix

    def describe(self, axes: StageAxes) -> str:
        o = self.offset
        s = describe_linear(self.correction(axes)) + f", origin ({o[0]:.2f}, {o[1]:.2f}) µm"
        if self.rms_um is not None:
            s += f", rms {self.rms_um * 1e3:.0f} nm"
        return s if self.measured else "nominal: " + s

    def to_dict(self) -> dict:
        return {"linear": self.linear.tolist(), "offset": self.offset.tolist(), "measured": self.measured,
                "rms_um": self.rms_um, "step": self.step}

    @classmethod
    def from_dict(cls, d: dict) -> "ChipTransform":
        return cls(d["linear"], d["offset"], bool(d.get("measured", True)), d.get("rms_um"), d.get("step"))


def fit_chip(ccs_points, scs_points, axes: StageAxes) -> tuple[ChipTransform, np.ndarray]:
    """Global Align: 2 markers – rotation, uniform scale and shift on top of the stage axes;
    3–4 – full affine (scale and shear too).  Returns (transform, residuals µm (N, 2))."""
    c = np.atleast_2d(np.asarray(ccs_points, float))
    s = np.atleast_2d(np.asarray(scs_points, float))
    if not 2 <= len(c) <= 4 or len(c) != len(s):
        raise ValueError("Global Align needs 2 to 4 markers")
    d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
    if np.any(d[np.triu_indices(len(c), 1)] < 1e-9):
        raise ValueError("the Global Align markers must be distinct")
    m = axes.matrix
    if len(c) == 2:
        fit = fit_transform(c @ m.T, s, "similarity")  # scs = S·(M·ccs) + t
        lin = fit.transform.linear @ m
    else:
        fit = fit_transform(c, s, "affine")
        lin = fit.transform.linear
    t = ChipTransform(lin, fit.transform.offset, measured=True)
    res = s - t.to_scs(c)
    if len(c) > 2:
        t.rms_um = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return t, res


# ====================================================================== FCS → XCS
@dataclass
class FieldMap:
    """One field written with the stage at its read-back: ``bcs = linear·fcs + shift``, then the
    beam calibration (BCS → XCS).  :meth:`apply` is FCS → XCS (what the writer and the field check
    use); ``linear = B·A_g``, ``shift = B·(G(c_f) − ŝ + δ_f)``."""

    linear: np.ndarray = field(default_factory=lambda: np.eye(2))
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))
    beamcal: object = None  # BeamCalibration (None: XCS = BCS)

    def __post_init__(self):
        self.linear = np.asarray(self.linear, float).reshape(2, 2)
        self.shift = _pt(self.shift)

    @classmethod
    def make(cls, chip: ChipTransform, centre_ccs, stage, b: np.ndarray, delta=(0.0, 0.0),
             beamcal=None) -> "FieldMap":
        b = np.asarray(b, float)
        c = chip.to_scs(_pt(centre_ccs))
        return cls(b @ chip.linear, b @ (c - _pt(stage) + _pt(delta)), beamcal)

    def copy(self) -> "FieldMap":
        return FieldMap(self.linear.copy(), self.shift.copy(), self.beamcal)

    def to_bcs(self, fcs) -> np.ndarray:
        p = np.asarray(fcs, float)
        out = np.atleast_2d(p) @ self.linear.T + self.shift
        return out[0] if p.ndim == 1 else out

    def bcs_to_fcs(self, bcs) -> np.ndarray:
        p = np.asarray(bcs, float)
        out = (np.atleast_2d(p) - self.shift) @ np.linalg.inv(self.linear).T
        return out[0] if p.ndim == 1 else out

    def bcs_to_xcs(self, bcs) -> np.ndarray:
        return np.asarray(bcs, float) if self.beamcal is None else self.beamcal.to_xcs(bcs)

    def xcs_to_bcs(self, xcs) -> np.ndarray:
        return np.asarray(xcs, float) if self.beamcal is None else self.beamcal.to_bcs(xcs)

    def apply(self, fcs) -> np.ndarray:
        """FCS → XCS."""
        return self.bcs_to_xcs(self.to_bcs(fcs))

    def is_identity(self, tol: float = 1e-12) -> bool:
        return (np.allclose(self.linear, np.eye(2), atol=tol) and np.allclose(self.shift, 0.0, atol=tol)
                and (self.beamcal is None or self.beamcal.is_identity))

    def describe(self) -> str:
        s = describe_linear(self.linear) + f", shift ({self.shift[0]:+.3f}, {self.shift[1]:+.3f}) µm"
        return s + (", beam calibrated" if self.beamcal is not None and not self.beamcal.is_identity else "")


# ====================================================================== field check
@dataclass
class FieldCheck:
    """How much of a pattern lands outside the deflection field (XCS)."""

    n_points: int
    n_outside: int
    max_abs_um: float
    half_field_um: float
    bbox_xcs: tuple | None

    @property
    def ok(self) -> bool:
        return self.n_outside == 0

    def message(self) -> str:
        if self.ok:
            return ""
        return (f"{self.n_outside} of {self.n_points} vertices outside the field: extends to ±{self.max_abs_um:.2f} µm, "
                f"the field is ±{self.half_field_um:.2f} µm")


def check_field(pattern, transform, field_um: float) -> FieldCheck:
    """Map every vertex (the field is a convex square, so vertices suffice) through *transform*
    (anything with ``apply``: FCS → XCS) and count those outside ``±field_um / 2``."""
    pts = [np.atleast_2d(e.points) for e in pattern]
    half = float(field_um) / 2.0
    if not pts:
        return FieldCheck(0, 0, 0.0, half, None)
    b = np.atleast_2d(transform.apply(np.vstack(pts)))
    a = np.abs(b)
    out = np.any(a > half * (1 + 1e-9), axis=1)
    mn, mx = b.min(0), b.max(0)
    return FieldCheck(len(b), int(np.count_nonzero(out)), float(a.max()), half,
                      (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])))
