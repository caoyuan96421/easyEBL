"""Contrast / brightness ↔ display levels, scale-bar maths (no Qt)."""
from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["ContrastBrightness", "nice_scale_bar", "format_length", "display_magnification"]

#: Polaroid / 4×5" convention: magnification refers to a 127 mm wide image.
REFERENCE_WIDTH_UM = 127_000.0


@dataclass
class ContrastBrightness:
    """SEM-style controls (both 0…1) mapped onto display levels in volts.

    * contrast → span of the displayed voltage window, logarithmic over
      ``decades`` (0 = the whole AI range, 1 = range / 10^decades);
    * brightness → centre of the window (0 = −range, 1 = +range).
    """

    ai_range_v: float = 10.0
    contrast: float = 0.35
    brightness: float = 0.55
    decades: float = 3.0

    @property
    def span(self) -> float:
        return 2 * self.ai_range_v * 10 ** (-self.decades * self.contrast)

    @property
    def center(self) -> float:
        return -self.ai_range_v + 2 * self.ai_range_v * self.brightness

    def levels(self) -> tuple[float, float]:
        return (self.center - self.span / 2, self.center + self.span / 2)

    def set_levels(self, lo: float, hi: float) -> None:
        span = max(hi - lo, 2 * self.ai_range_v * 10 ** (-self.decades))
        c = -math.log10(span / (2 * self.ai_range_v)) / self.decades
        self.contrast = min(1.0, max(0.0, c))
        b = ((lo + hi) / 2 + self.ai_range_v) / (2 * self.ai_range_v)
        self.brightness = min(1.0, max(0.0, b))


def nice_scale_bar(width_um: float, fraction: float = 0.2) -> float:
    """1-2-5 length (µm) close to *fraction* of the image width."""
    if width_um <= 0:
        return 1.0
    target = width_um * fraction
    exp = math.floor(math.log10(target))
    best = None
    for m in (1, 2, 5, 10):
        v = m * 10**exp
        if v <= target * 1.5 and (best is None or abs(v - target) < abs(best - target)):
            best = v
    return float(best or 10**exp)


def format_length(um: float) -> str:
    if um >= 1000:
        return f"{um / 1000:g} mm"
    if um >= 1:
        return f"{um:g} µm"
    return f"{um * 1000:g} nm"


def display_magnification(width_um: float) -> str:
    """Magnification of the scan field on the 127 mm reference width ("×1.5k")."""
    if width_um <= 0:
        return "×—"
    m = REFERENCE_WIDTH_UM / width_um
    if m >= 1e5:
        return f"×{m / 1e3:.0f}k"
    if m >= 1e3:
        return f"×{m / 1e3:.1f}k"
    return f"×{m:.0f}"
