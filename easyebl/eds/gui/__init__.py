"""EDS user interface (PySide6 + pyqtgraph) – front end only; all work happens in
:class:`easyebl.eds.EDSController` and :mod:`easyebl.eds.analysis`.

* :mod:`.window`      – :class:`EDSWindow` (electron image, sites, spectrum, quant, maps)
* :mod:`.calibration` – the energy calibration dialog
* :mod:`.processor`   – pulse-processor (DP5) settings
* :mod:`.views`       – spectrum / image / map widgets, periodic table
* :mod:`.app`         – ``easyebl-eds`` and ``easyebl-suite`` (SEM + EDS in one process)
"""


def main(argv=None) -> int:
    from .app import main as _main

    return _main(argv)


def suite_main(argv=None) -> int:
    from .app import suite_main as _main

    return _main(argv)
