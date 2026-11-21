"""easyEBL – DIY SEM imaging / e-beam lithography control on top of NI-DAQmx.

Subpackages
-----------
core       transforms, field calibration, waveform containers
hal        hardware abstraction layer (shared DAQ handles, NI + simulator)
imaging    raster scanning with synchronised X/Y/blanker and video acquisition
alignment  template matching and marker-based alignment
litho      dose utilities, fills, vector-scan compiler, writer, GDS import
gui        shared GUI helpers (lithography and debug tools)
microscope SEM column and stage control over RS-232 (Hitachi S-3000)
eds        EDS acquisition and analysis
app        the easyEBL program: menu window, tray icon, backend (run: easyebl)
"""
__version__ = "0.1.0"
