"""SEM user interface (PySide6): commercial-SEM-style imaging front end.

Modules:

* :mod:`.window`   – :class:`SEMWindow`, the main window
* :mod:`.view`     – image display with scan line, data bar and scale bar
* :mod:`.panels`   – scan / capture / image / beam-shift / column panels
* :mod:`.advanced` – the "Advanced…" hardware & waveform dialog
* :mod:`.display`  – contrast/brightness model, scale-bar maths (no Qt)
* :mod:`.optics`   – microscope (column) control interface – stubs for now

Run:  ``python -m easyebl.imaging.sem [--backend nidaqmx --device Dev1] [--console]``
"""


def main(argv=None) -> int:
    from .app import main as _main

    return _main(argv)
