"""The Lithography module window (Qt): stage view, stage positions, chips, exposure schedule.

* :mod:`.window`          – :class:`LithoWindow`
* :mod:`.view`            – :class:`StageView` (SCS: chips, fields, patterns, markers, stage positions)
* :mod:`.tabs`            – the Stage Positions, Chips and Exposure Schedule tabs, the step editor
* :mod:`.dialogs`         – stage settings, moving a chip, field / beam calibration settings
* :mod:`.fieldcal_window` / :mod:`.beamcal_window` – calibration progress and results
* :mod:`.hardware`        – :class:`BackendHardware` (the easyEBL backend), :class:`GuiBridge`

The schedule model, coordinate maps, calibrations and runner have no Qt: :mod:`easyebl.litho.schedule`,
:mod:`easyebl.litho.coords`, :mod:`easyebl.litho.beamcal`, :mod:`easyebl.litho.fieldcal`,
:mod:`easyebl.litho.runner`.
"""
from .window import LithoWindow

__all__ = ["LithoWindow"]
