"""Working distance from the objective focus values (Hitachi S-3000) – one lens setting, two views.

``WD n`` and ``FOCUS C`` / ``FOCUS F`` drive the same objective lens: writing ``WD n``
sets the coarse focus from a fixed table and centres the fine focus (≈ 2048), and
the ``WD`` readback is that table read backwards from the focus, rounded to whole
millimetres.  The fine focus weighs :data:`FINE_WEIGHT` of a coarse count, the same
at every WD, so the lens is set by one number

    effective focus = FOCUS C + FINE_WEIGHT × (FOCUS F − 2048)

and :func:`wd_from_focus` gives the working distance to a few µm (1 coarse count is
≈1 µm at 5 mm, ≈4 µm at 15 mm, ≈27 µm at 60 mm; 1 fine count ≈ 1/12 of that).

Measured on the S-3000 with HV off (2026-11-23):

* the table: ``WD n`` for n = 4…60, the coarse / fine value the SEM set (``WD 3``
  saturates at C 16383 / F 4095 and is left out);
* the readback steps n → n+1 fall at the midpoint between two table values: the
  interpolated table reproduces 11 of them to 0.002 mm rms (0.007 mm max);
* FINE_WEIGHT from the shift of those steps with the fine value at 5, 15 and 30 mm
  (0.0817, 0.0816, 0.0814);
* the coarse value is clamped at 5174 (below WD 60) by the SEM.

Not measured: whether the table changes with the accelerating voltage (it was taken at
one HV setpoint; :class:`~easyebl.microscope.sem.SEMControl` compares with the SEM's
own ``WD`` readback and reports a disagreement).
"""
from __future__ import annotations

import numpy as np

__all__ = ["FINE_WEIGHT", "FINE_CENTRE", "WD_TABLE", "effective_focus", "wd_from_focus", "focus_for_wd",
           "WD_RANGE_MM", "COARSE_MIN"]

#: one fine count in coarse counts
FINE_WEIGHT = 0.08155
#: the fine value ``WD n`` centres on
FINE_CENTRE = 2048
#: the SEM's range for ``WD n``
WD_RANGE_MM = (3, 60)
#: the SEM clamps the coarse focus here (≈ WD 60 + a few mm)
COARSE_MIN = 5174

#: WD (mm) -> (FOCUS C, FOCUS F) the SEM sets for ``WD n`` (S-3000, measured 2026-11-23)
WD_TABLE: dict[int, tuple[int, int]] = {
    4: (16129, 2051), 5: (15221, 2044), 6: (14458, 2043), 7: (13805, 2050), 8: (13239, 2052),
    9: (12743, 2045), 10: (12302, 2052), 11: (11909, 2043), 12: (11554, 2045), 13: (11232, 2047),
    14: (10939, 2042), 15: (10670, 2042), 16: (10422, 2046), 17: (10193, 2047), 18: (9980, 2052),
    19: (9783, 2043), 20: (9598, 2046), 21: (9425, 2046), 22: (9262, 2052), 23: (9110, 2042),
    24: (8965, 2051), 25: (8829, 2048), 26: (8700, 2048), 27: (8578, 2044), 28: (8461, 2052),
    29: (8351, 2045), 30: (8245, 2051), 31: (8145, 2045), 32: (8049, 2043), 33: (7957, 2044),
    34: (7869, 2043), 35: (7784, 2049), 36: (7703, 2048), 37: (7625, 2050), 38: (7550, 2051),
    39: (7478, 2050), 40: (7409, 2046), 41: (7342, 2047), 42: (7277, 2051), 43: (7215, 2047),
    44: (7155, 2045), 45: (7097, 2042), 46: (7040, 2049), 47: (6986, 2043), 48: (6933, 2045),
    49: (6882, 2042), 50: (6832, 2046), 51: (6784, 2044), 52: (6737, 2046), 53: (6691, 2052),
    54: (6647, 2050), 55: (6604, 2050), 56: (6562, 2053), 57: (6522, 2046), 58: (6482, 2050),
    59: (6444, 2044), 60: (6406, 2049),
}


def effective_focus(coarse: float, fine: float) -> float:
    """The one number the lens is set by, in coarse counts."""
    return float(coarse) + FINE_WEIGHT * (float(fine) - FINE_CENTRE)


_WD = np.array(sorted(WD_TABLE), float)
_EFF = np.array([effective_focus(*WD_TABLE[int(w)]) for w in _WD])  # falls as WD rises
_interp = None


def _curve():
    global _interp
    if _interp is None:
        from scipy.interpolate import PchipInterpolator

        _interp = (PchipInterpolator(_EFF[::-1], _WD[::-1]), PchipInterpolator(_WD, _EFF))
    return _interp


def wd_from_focus(coarse: float, fine: float) -> float:
    """Working distance (mm) for the focus values, within :data:`WD_RANGE_MM`.

    Monotone cubic interpolation of the table; beyond it (WD < 4 or > 60) the end slope
    is extended and the result held to the SEM's 3–60 mm."""
    e = effective_focus(coarse, fine)
    if _EFF[-1] <= e <= _EFF[0]:
        wd = float(_curve()[0](e))
    elif e > _EFF[0]:  # nearer than WD 4
        wd = _WD[0] + (e - _EFF[0]) * (_WD[1] - _WD[0]) / (_EFF[1] - _EFF[0])
    else:  # farther than WD 60
        wd = _WD[-1] + (e - _EFF[-1]) * (_WD[-1] - _WD[-2]) / (_EFF[-1] - _EFF[-2])
    return float(min(max(wd, WD_RANGE_MM[0]), WD_RANGE_MM[1]))


def focus_for_wd(wd_mm: float) -> tuple[int, int]:
    """(FOCUS C, FOCUS F) for a working distance within the table (4–60 mm): the coarse value
    the SEM would use, the remainder in fine counts."""
    wd = float(min(max(wd_mm, _WD[0]), _WD[-1]))
    e = float(_curve()[1](wd))
    c = int(round(e))
    f = int(round(FINE_CENTRE + (e - c) / FINE_WEIGHT))
    return c, f
