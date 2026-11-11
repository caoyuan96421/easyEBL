"""EDS (energy-dispersive X-ray spectroscopy) for easyEBL.

Backend (no Qt): :mod:`.controller` (acquisitions over the HAL and a pulse
processor), :mod:`.spectrum` (data + files), :mod:`.analysis` (calibration,
background, peak identification, fitting, quantification, maps).
Front-end: :mod:`easyebl.eds.gui`.
"""
from .controller import EDSController, EDSJob, EDSOptions, JobCancelled
from .spectrum import ElectronImage, EnergyCalibration, Spectrum, SpectrumMap, load_map, load_spectrum

__all__ = ["EDSController", "EDSJob", "EDSOptions", "JobCancelled", "ElectronImage", "EnergyCalibration",
           "Spectrum", "SpectrumMap", "load_map", "load_spectrum"]
