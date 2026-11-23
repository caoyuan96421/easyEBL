"""Contrast / brightness ↔ display levels, scale-bar maths (no Qt)."""
from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["ContrastBrightness", "nice_scale_bar", "format_length", "display_magnification", "image_magnification"]

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


def format_length(um: float, digits: int = 6) -> str:
    """A length with its unit (nm / µm / mm), at most *digits* significant figures."""
    if um >= 1000:
        return f"{um / 1000:.{digits}g} mm"
    if um >= 1:
        return f"{um:.{digits}g} µm"
    return f"{um * 1000:.{digits}g} nm"


def image_magnification(column_mag: float | None, field_um: float | None, width_um: float) -> float | None:
    """Magnification of the image: the column's magnification is that of the full field of view;
    a narrower sweep zooms in from there (column × field / scan width)."""
    if not column_mag or not field_um or width_um <= 0:
        return None
    return float(column_mag) * float(field_um) / float(width_um)


def display_magnification(mag: float | None) -> str:
    """"×1.5k"-style text of a magnification (None: unknown)."""
    if not mag or mag <= 0:
        return "×—"
    if mag >= 1e5:
        return f"×{mag / 1e3:.0f}k"
    if mag >= 1e3:
        return f"×{mag / 1e3:.3g}k"
    return f"×{mag:.3g}"
