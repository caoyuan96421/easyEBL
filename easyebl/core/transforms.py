"""2-D affine transforms used throughout easyEBL.

All geometry in easyEBL is expressed in micrometres in *field coordinates*:
origin at the centre of the deflection field, +x to the right, +y up.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = ["AffineTransform"]


@dataclass(frozen=True)
class AffineTransform:
    """Immutable 2-D affine transform ``p' = A @ p + t``.

    Stored as a 3x3 homogeneous matrix.  Use the factory methods instead of
    building the matrix by hand.
    """

    matrix: np.ndarray

    def __post_init__(self):
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError("AffineTransform matrix must be 3x3")
        if not np.allclose(m[2], [0.0, 0.0, 1.0]):
            raise ValueError("last row of an affine matrix must be [0, 0, 1]")
        m = m.copy()
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)

    # ------------------------------------------------------------------ factories
    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls(np.eye(3))

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineTransform":
        m = np.eye(3)
        m[0, 2], m[1, 2] = tx, ty
        return cls(m)

    @classmethod
    def scaling(cls, sx: float, sy: float | None = None) -> "AffineTransform":
        sy = sx if sy is None else sy
        return cls(np.diag([sx, sy, 1.0]))

    @classmethod
    def rotation(cls, angle_rad: float) -> "AffineTransform":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))

    @classmethod
    def from_params(
        cls,
        tx: float = 0.0,
        ty: float = 0.0,
        rotation_rad: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        shear: float = 0.0,
    ) -> "AffineTransform":
        """Build ``T @ R @ Shear @ S`` (scale first, translation last)."""
        sh = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        m = (
            cls.translation(tx, ty).matrix
            @ cls.rotation(rotation_rad).matrix
            @ sh
            @ cls.scaling(sx, sy).matrix
        )
        return cls(m)

    @classmethod
    def from_linear(cls, a: np.ndarray, t=(0.0, 0.0)) -> "AffineTransform":
        m = np.eye(3)
        m[:2, :2] = np.asarray(a, dtype=float)
        m[:2, 2] = np.asarray(t, dtype=float)
        return cls(m)

    # ---------------------------------------------------------------- operations
    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:2, :2]

    @property
    def offset(self) -> np.ndarray:
        return self.matrix[:2, 2]

    def apply(self, points) -> np.ndarray:
        """Transform an (N, 2) array (or a single (2,) point)."""
        p = np.asarray(points, dtype=float)
        single = p.ndim == 1
        p2 = np.atleast_2d(p)
        out = p2 @ self.linear.T + self.offset
        return out[0] if single else out

    __call__ = apply

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return ``self ∘ other`` (apply *other* first, then *self*)."""
        return AffineTransform(self.matrix @ other.matrix)

    def __matmul__(self, other: "AffineTransform") -> "AffineTransform":
        return self.compose(other)

    def inverse(self) -> "AffineTransform":
        return AffineTransform(np.linalg.inv(self.matrix))

    def decompose(self) -> dict:
        """Approximate decomposition into translation/rotation/scale/shear."""
        a = self.linear
        sx = math.hypot(a[0, 0], a[1, 0])
        rot = math.atan2(a[1, 0], a[0, 0])
        shear_term = (a[0, 0] * a[0, 1] + a[1, 0] * a[1, 1]) / sx
        sy = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) / sx
        return {
            "tx": float(self.offset[0]),
            "ty": float(self.offset[1]),
            "rotation_rad": rot,
            "sx": sx,
            "sy": sy,
            "shear": shear_term / sy if sy else 0.0,
        }

    def is_identity(self, tol: float = 1e-12) -> bool:
        return bool(np.allclose(self.matrix, np.eye(3), atol=tol))

    def to_list(self) -> list:
        return self.matrix.tolist()

    @classmethod
    def from_list(cls, values) -> "AffineTransform":
        return cls(np.asarray(values, dtype=float))

    def __eq__(self, other) -> bool:  # pragma: no cover - trivial
        return isinstance(other, AffineTransform) and np.allclose(self.matrix, other.matrix)

    def __hash__(self):  # pragma: no cover - trivial
        return hash(self.matrix.tobytes())

    def __repr__(self) -> str:
        d = self.decompose()
        return (
            "AffineTransform(t=({tx:.4g}, {ty:.4g}), rot={r:.4g} mrad, "
            "s=({sx:.6g}, {sy:.6g}), shear={shear:.3g})".format(r=d["rotation_rad"] * 1e3, **d)
        )
