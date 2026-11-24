"""The Lithography module window (Qt): stage view, stage positions, exposure schedule.

* :mod:`.window`   – :class:`LithoWindow`
* :mod:`.view`     – :class:`StageView` (SCS: fields, patterns, each field's WCS, markers, stage positions)
* :mod:`.tabs`     – the Stage Positions and Exposure Schedule tabs, the step editor
* :mod:`.dialogs`  – stage axes, fractured-GDS import
* :mod:`.hardware` – :class:`BackendHardware` (the easyEBL backend), :class:`GuiBridge`

The schedule model, transforms, importer and runner have no Qt: :mod:`easyebl.litho.schedule`,
:mod:`easyebl.litho.coords`, :mod:`easyebl.litho.fracture`, :mod:`easyebl.litho.runner`.
"""
from .window import LithoWindow

__all__ = ["LithoWindow"]
