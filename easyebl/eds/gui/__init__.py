"""EDS user interface (PySide6 + pyqtgraph) – front end only; all work happens in
:class:`easyebl.eds.EDSController` and :mod:`easyebl.eds.analysis`.

* :mod:`.window`      – :class:`EDSWindow` (electron image, sites, spectrum, quant, maps)
* :mod:`.calibration` – the energy calibration dialog
* :mod:`.processor`   – pulse-processor (DP5) settings
* :mod:`.views`       – spectrum / image / map widgets, periodic table

The window is opened from the easyEBL program (``easyebl``, :mod:`easyebl.app`).
"""
