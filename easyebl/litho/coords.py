"""Coordinate systems of the lithography module: SCS, WCS (per field) and BCS.

* **SCS** – stage coordinates (µm): what the litho window plots.  Field centres and
  alignment markers are entered in SCS (typically: centre a feature, take the stage position).
* **WCS** – working coordinates of one field (µm): (0, 0) is the field centre, the
  pattern's GDS cell coordinates are WCS.  A sample point at stage position ``p`` has
  ``wcs = M · (p − centre)``: WCS has true (stage) µm and the stage axes, ``M``
  (:class:`StageAxes`) mirroring stage axes that run against the beam axes.
* **BCS** – beam coordinates (µm, the deflection field, origin at its centre); internal.

The stage → beam map ``B`` (:class:`StageFrame`): a stage displacement ``d`` moves a
feature of the sample by ``−B·d`` in BCS.  Nominally ``B = M`` (beam and stage parallel,
equal scale); the **field calibration** measures it (rotation between stage and beam,
the beam's scale error).  A field without Global Align is written through
``A = B·M`` (true µm along the stage axes).

With the stage at the field's target, a field is written through :class:`FieldTransform`::

    bcs = A · wcs + t

``A`` comes from the last **Global Align** (rotation, scale, stretch; kept until the next
one), ``t`` is measured by the Global Align / **Local XY** of this field (0 otherwise).

:class:`GlobalAlignment` keeps where the last Global Align was measured, so a later
field's centre can be corrected: the stage goes to

    origin_ref + M · A · M · (centre − centre_ref)

where ``origin_ref = target_ref + M · t_ref`` is the measured WCS origin of the reference
field in SCS – the nominal spacing between the field centres, transformed by the measured
global alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..alignment.fitting import fit_transform
from ..core.transforms import AffineTransform

__all__ = ["StageAxes", "StageFrame", "FieldTransform", "GlobalAlignment", "fit_global", "fit_local", "FieldCheck", "check_field"]


def _pt(p) -> np.ndarray:
    a = np.asarray(p, float).reshape(-1)
    if a.shape != (2,):
        raise ValueError("a point is (x, y)")
    return a


@dataclass(frozen=True)
class StageAxes:
    """Stage axes against the beam axes: ``mirror_x`` – stage +X moves the sample image along −x."""

    mirror_x: bool = False
    mirror_y: bool = False

    @property
    def matrix(self) -> np.ndarray:
        return np.diag([-1.0 if self.mirror_x else 1.0, -1.0 if self.mirror_y else 1.0])

    def scs_to_wcs(self, scs, centre) -> np.ndarray:
        """Nominal WCS of stage points (a field centred at *centre*)."""
        return (np.atleast_2d(np.asarray(scs, float)) - _pt(centre)) @ self.matrix.T

    def wcs_to_scs(self, wcs, centre) -> np.ndarray:
        return np.atleast_2d(np.asarray(wcs, float)) @ self.matrix.T + _pt(centre)

    def to_dict(self) -> dict:
        return {"mirror_x": self.mirror_x, "mirror_y": self.mirror_y}

    @classmethod
    def from_dict(cls, d: dict | None) -> "StageAxes":
        d = d or {}
        return cls(bool(d.get("mirror_x", False)), bool(d.get("mirror_y", False)))


@dataclass(frozen=True)
class StageFrame:
    """Stage axes plus the stage → beam map ``B`` (None: nominal ``B = M``)."""

    axes: StageAxes = StageAxes()
    stage_to_beam: tuple | None = None  # 2×2 as nested tuples (hashable), from the field calibration

    @classmethod
    def make(cls, axes: StageAxes | None = None, stage_to_beam=None) -> "StageFrame":
        b = None if stage_to_beam is None else tuple(map(tuple, np.asarray(stage_to_beam, float).reshape(2, 2)))
        return cls(axes or StageAxes(), b)

    @property
    def calibrated(self) -> bool:
        return self.stage_to_beam is not None

    @property
    def B(self) -> np.ndarray:
        return self.axes.matrix if self.stage_to_beam is None else np.array(self.stage_to_beam, float)

    @property
    def B_inv(self) -> np.ndarray:
        return np.linalg.inv(self.B)

    def nominal_transform(self) -> "FieldTransform":
        """WCS → BCS of a field without Global Align: ``A = B·M``."""
        return FieldTransform(self.B @ self.axes.matrix, (0.0, 0.0))

    def beam_to_stage(self, v) -> np.ndarray:
        """Stage displacement of BCS vectors (rows)."""
        return np.atleast_2d(np.asarray(v, float)) @ self.B_inv.T


@dataclass
class FieldTransform:
    """``bcs = A · wcs + t`` for one field (see module docstring)."""

    linear: np.ndarray = field(default_factory=lambda: np.eye(2))
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.linear = np.asarray(self.linear, float).reshape(2, 2)
        self.shift = _pt(self.shift)

    def copy(self) -> "FieldTransform":
        return FieldTransform(self.linear.copy(), self.shift.copy())

    @property
    def wcs_to_bcs(self) -> AffineTransform:
        return AffineTransform.from_linear(self.linear, self.shift)

    def to_bcs(self, wcs) -> np.ndarray:
        return self.wcs_to_bcs.apply(wcs)

    def to_wcs(self, bcs) -> np.ndarray:
        return self.wcs_to_bcs.inverse().apply(bcs)

    @property
    def is_identity(self) -> bool:
        return self.wcs_to_bcs.is_identity(1e-12)

    def params(self) -> dict:
        return linear_params(self.linear) | {"shift_um": (float(self.shift[0]), float(self.shift[1]))}

    def describe(self) -> str:
        p = self.params()
        return describe_linear(self.linear) + f", shift ({p['shift_um'][0]:+.3f}, {p['shift_um'][1]:+.3f}) µm"


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


@dataclass
class GlobalAlignment:
    """The last Global Align: its ``A`` and where it was measured (see module docstring)."""

    linear: np.ndarray
    ref_centre: np.ndarray  # nominal centre of the reference field (SCS)
    ref_target: np.ndarray  # where the stage was for it (SCS)
    ref_shift: np.ndarray  # t measured there (BCS)
    magnification: float | None = None
    step: int | None = None  # schedule step that measured it

    def __post_init__(self):
        self.linear = np.asarray(self.linear, float).reshape(2, 2)
        self.ref_centre, self.ref_target, self.ref_shift = _pt(self.ref_centre), _pt(self.ref_target), \
            _pt(self.ref_shift)

    def copy(self) -> "GlobalAlignment":
        return GlobalAlignment(self.linear.copy(), self.ref_centre.copy(), self.ref_target.copy(),
                               self.ref_shift.copy(), self.magnification, self.step)

    def ref_origin(self, frame) -> np.ndarray:
        """Measured SCS position of the reference field's WCS origin (*frame*: StageFrame or StageAxes)."""
        frame = _frame(frame)
        return self.ref_target + frame.B_inv @ self.ref_shift

    def corrected_centre(self, centre, frame) -> np.ndarray:
        """Stage target for a field nominally centred at *centre* (SCS): the reference origin + the
        nominal spacing (WCS) through the measured A, back to stage µm through ``B⁻¹``."""
        frame = _frame(frame)
        return self.ref_origin(frame) + frame.B_inv @ self.linear @ frame.axes.matrix @ (_pt(centre) - self.ref_centre)

    def describe(self) -> str:
        where = f" (step {self.step + 1})" if self.step is not None else ""
        return describe_linear(self.linear) + where

    def to_dict(self) -> dict:
        return {"linear": self.linear.tolist(), "ref_centre": self.ref_centre.tolist(),
                "ref_target": self.ref_target.tolist(), "ref_shift": self.ref_shift.tolist(),
                "magnification": self.magnification, "step": self.step}

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalAlignment":
        return cls(d["linear"], d["ref_centre"], d["ref_target"], d["ref_shift"], d.get("magnification"),
                   d.get("step"))


def _frame(f) -> StageFrame:
    return f if isinstance(f, StageFrame) else StageFrame(f if f is not None else StageAxes())


def fit_global(wcs_points, bcs_points) -> tuple[FieldTransform, np.ndarray]:
    """Global Align: 2 markers – rotation, uniform scale, shift; 3–4 – full affine (stretch, shear).
    Returns (transform, residuals µm)."""
    w = np.atleast_2d(np.asarray(wcs_points, float))
    b = np.atleast_2d(np.asarray(bcs_points, float))
    if not 2 <= len(w) <= 4 or len(w) != len(b):
        raise ValueError("Global Align needs 2 to 4 markers")
    d = np.hypot(*(w[:, None, :] - w[None, :, :]).transpose(2, 0, 1))
    if np.any(d[np.triu_indices(len(w), 1)] < 1e-9):
        raise ValueError("the Global Align markers must be distinct")
    fit = fit_transform(w, b, "similarity" if len(w) == 2 else "affine")
    return FieldTransform(fit.transform.linear, fit.transform.offset), fit.residuals


def fit_local(wcs_point, bcs_point, previous: FieldTransform) -> FieldTransform:
    """Local XY: the shift only; A carries over."""
    return FieldTransform(previous.linear.copy(), _pt(bcs_point) - previous.linear @ _pt(wcs_point))


# ====================================================================== field check
@dataclass
class FieldCheck:
    """How much of a pattern lands outside the deflection field (BCS)."""

    n_points: int
    n_outside: int
    max_abs_um: float
    half_field_um: float
    bbox_bcs: tuple | None

    @property
    def ok(self) -> bool:
        return self.n_outside == 0

    def message(self) -> str:
        if self.ok:
            return ""
        return (f"{self.n_outside} of {self.n_points} vertices outside the field: extends to ±{self.max_abs_um:.2f} µm, "
                f"the field is ±{self.half_field_um:.2f} µm")


def check_field(pattern, transform: AffineTransform, field_um: float) -> FieldCheck:
    """Map every vertex (the field is a convex square, so vertices suffice) through
    *transform* (WCS → BCS) and count those outside ``±field_um / 2``."""
    pts = [np.atleast_2d(e.points) for e in pattern]
    half = float(field_um) / 2.0
    if not pts:
        return FieldCheck(0, 0, 0.0, half, None)
    b = transform.apply(np.vstack(pts))
    a = np.abs(b)
    out = np.any(a > half * (1 + 1e-9), axis=1)
    mn, mx = b.min(0), b.max(0)
    return FieldCheck(len(b), int(np.count_nonzero(out)), float(a.max()), half,
                      (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])))
