"""Mapping between field coordinates (µm) and deflection voltages (V).

The SEM's external scan input is scaled by the microscope magnification, so a
given full-scale voltage corresponds to the field of view at the current
magnification.  :class:`FieldCalibration` captures that relation for one
magnification setting plus a *correction* transform (measured field
distortion / alignment) applied in the µm domain before the nominal scaling.

    volts = (nominal µm→V scaling) ∘ correction ∘ (µm field coordinates)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json

import numpy as np

from .transforms import AffineTransform

__all__ = ["FieldCalibration", "OutOfFieldError"]


class OutOfFieldError(ValueError):
    """Raised when a coordinate maps outside the usable deflection range."""


@dataclass(frozen=True)
class FieldCalibration:
    """Field-of-view ↔ voltage calibration for one magnification.

    Parameters
    ----------
    field_size_um:
        Width (= height) of the field covered by ``-v_full_scale .. +v_full_scale``.
    v_full_scale:
        Voltage at the field edge (positive number, e.g. 10 V).
    correction:
        Affine transform applied to µm coordinates before scaling.  This is
        where measured scale/rotation/offset errors (from the calibration grid
        or from marker alignment) live.
    invert_x / invert_y:
        Flip an axis if the SEM's input polarity is reversed.
    name:
        Free-form label (e.g. ``"x1000 @ 15 kV, WD 10"``).
    """

    field_size_um: float = 100.0
    v_full_scale: float = 10.0
    correction: AffineTransform = field(default_factory=AffineTransform.identity)
    invert_x: bool = False
    invert_y: bool = False
    name: str = ""

    def __post_init__(self):
        if self.field_size_um <= 0:
            raise ValueError("field_size_um must be positive")
        if self.v_full_scale <= 0:
            raise ValueError("v_full_scale must be positive")

    # ----------------------------------------------------------------- scaling
    @property
    def volts_per_um(self) -> float:
        return 2.0 * self.v_full_scale / self.field_size_um

    @property
    def half_field_um(self) -> float:
        return self.field_size_um / 2.0

    def dac_step_um(self, bits: int = 16, ao_range_v: float = 10.0) -> float:
        """Beam step corresponding to one DAC LSB."""
        return (2.0 * ao_range_v / 2**bits) / self.volts_per_um

    def _nominal(self) -> AffineTransform:
        k = self.volts_per_um
        return AffineTransform.scaling(-k if self.invert_x else k, -k if self.invert_y else k)

    @property
    def um_to_volts_transform(self) -> AffineTransform:
        return self._nominal() @ self.correction

    def um_to_volts(self, points, check: bool = True, limit_v: float | None = None) -> np.ndarray:
        """Convert (N, 2) µm field coordinates to (N, 2) volts."""
        v = self.um_to_volts_transform.apply(points)
        if check:
            lim = self.v_full_scale if limit_v is None else limit_v
            bad = np.abs(v) > lim * (1 + 1e-9)
            if np.any(bad):
                worst = float(np.max(np.abs(v)))
                raise OutOfFieldError(
                    f"{int(np.count_nonzero(np.any(np.atleast_2d(bad), axis=-1)))} point(s) outside "
                    f"±{lim:g} V (max |V| = {worst:.4g} V); pattern does not fit the field"
                )
        return v

    def volts_to_um(self, volts) -> np.ndarray:
        return self.um_to_volts_transform.inverse().apply(volts)

    def contains(self, points, margin_um: float = 0.0) -> np.ndarray:
        """Boolean mask of points that map inside the full-scale voltage range."""
        v = self.um_to_volts_transform.apply(np.atleast_2d(points))
        lim = self.v_full_scale - margin_um * self.volts_per_um
        return np.all(np.abs(v) <= lim + 1e-12, axis=1)

    # ------------------------------------------------------------- composition
    def with_correction(self, correction: AffineTransform) -> "FieldCalibration":
        return replace(self, correction=correction)

    def with_additional_transform(self, t: AffineTransform) -> "FieldCalibration":
        """Prepend a design→field transform (e.g. from marker alignment)."""
        return replace(self, correction=self.correction @ t)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "field_size_um": self.field_size_um,
            "v_full_scale": self.v_full_scale,
            "correction": self.correction.to_list(),
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldCalibration":
        d = dict(d)
        d["correction"] = AffineTransform.from_list(d.get("correction", np.eye(3).tolist()))
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FieldCalibration":
        return cls.from_dict(json.loads(s))
