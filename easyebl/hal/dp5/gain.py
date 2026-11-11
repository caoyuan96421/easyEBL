"""
dp5.gain - DP5 analog (coarse) gain tables and helpers.

Total gain = coarse_table[GAIA - 1] * GAIF, with GAIF in 0.75 .. 1.25.
Tables from the SDK (``vcDP5/DP5Cfg/DppGain.h``). Which DP5 table applies is decided from the
status block: Rev Dx boards (FW+build >= 6.08.06 and DPP_ECO < 0xFF) use the 24-step table.
"""

from __future__ import annotations

DP5_GAINS = (1.122, 2.491, 3.778, 5.278, 6.556, 8.387, 10.091, 11.322,
             14.554, 17.772, 22.402, 30.839, 38.124, 47.468, 66.157, 101.828)
DP5_DX_GAINS = (1.000, 1.250, 1.563, 1.952, 2.433, 3.040, 3.801, 4.749,
                5.975, 7.466, 9.337, 11.666, 14.605, 18.250, 22.822, 28.515,
                35.530, 44.395, 55.518, 69.368, 87.269, 109.043, 136.364, 170.383)
PX5_GAINS = (1.000, 1.250, 1.560, 1.950, 2.433, 3.041, 3.797, 4.746, 5.975, 7.472, 9.340, 11.656,
             14.570, 18.179, 22.724, 28.371, 35.464, 44.645, 55.831, 69.788, 87.096, 108.622,
             135.836, 169.795, 211.989, 264.987, 330.704, 413.379)
DP5X_GAINS = (4.008, 8.016, 13.987, 27.974, 55.948, 111.896)

FINE_MIN, FINE_MAX = 0.75, 1.25
TOTAL_LIMITS = {"DP5": (0.75, 150.0), "DP5Dx": (0.75, 250.0), "PX5": (0.75, 500.0), "DP5X": (3.0, 150.0)}


def table_for(status=None, rev_dx: bool | None = None) -> tuple:
    """Coarse gain table for a DP5 given its decoded Status (or an explicit rev_dx flag)."""
    if rev_dx is None:
        rev_dx = bool(status is not None and status.is_rev_dx)
    return DP5_DX_GAINS if rev_dx else DP5_GAINS


def total_limits(rev_dx: bool) -> tuple[float, float]:
    return TOTAL_LIMITS["DP5Dx" if rev_dx else "DP5"]


def total_gain(gaia: int, gaif: float, rev_dx: bool) -> float:
    """Total gain from a 1-based coarse index and fine gain."""
    t = DP5_DX_GAINS if rev_dx else DP5_GAINS
    return t[gaia - 1] * gaif


def split_total_gain(total: float, rev_dx: bool) -> tuple[int, float]:
    """Choose (GAIA 1-based, GAIF) for a requested total gain, preferring GAIF close to 1.

    The DP5 does this itself when it receives GAIN; this helper is for display / planning.
    """
    t = DP5_DX_GAINS if rev_dx else DP5_GAINS
    lo, hi = total_limits(rev_dx)
    total = min(max(total, lo), hi)
    best = None
    for i, g in enumerate(t, start=1):
        f = total / g
        if FINE_MIN <= f <= FINE_MAX:
            score = abs(f - 1.0)
            if best is None or score < best[0]:
                best = (score, i, f)
    if best is None:                       # outside reachable range: clamp
        i = 1 if total < t[0] else len(t)
        return i, min(max(total / t[i - 1], FINE_MIN), FINE_MAX)
    return best[1], round(best[2], 4)
