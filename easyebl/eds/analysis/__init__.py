"""EDS spectrum analysis (pure numpy): background, peak search, AutoID, fitting, quantification, maps."""
from .background import smooth, snip, snip_background
from .beam import DuaneHunt, duane_hunt, estimate_beam_kv
from .peaksearch import Peak, find_peaks, refine_peak
from .fitting import FamilyRef, FitResult, escape_fraction, family_reference, fit_spectrum
from .maps import ElementMap, PhaseResult, bin_map, fitted_maps, phase_map, quant_maps, window_maps
from .peaks import Identification, identify_elements
from .quant import QuantResult, QuantRow, choose_family, quantify

__all__ = [
    "smooth", "snip", "snip_background", "DuaneHunt", "duane_hunt", "estimate_beam_kv", "Peak", "find_peaks", "refine_peak", "FamilyRef", "FitResult",
    "escape_fraction", "family_reference", "fit_spectrum", "ElementMap", "PhaseResult", "bin_map", "fitted_maps",
    "phase_map", "quant_maps", "window_maps", "Identification", "identify_elements", "QuantResult", "QuantRow",
    "choose_family", "quantify",
]
