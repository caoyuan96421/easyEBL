"""SEM user interface (PySide6): commercial-SEM-style imaging front end.

Modules:

* :mod:`.window`   – :class:`SEMWindow`, the main window
* :mod:`.view`     – image display with scan line, data bar and scale bar
* :mod:`.panels`   – scan / capture / image / beam-shift / column panels
* :mod:`.advanced` – the "Advanced…" hardware & waveform dialog
* :mod:`.display`  – contrast/brightness model, scale-bar maths (no Qt)

The column (magnification, focus …) is controlled through :mod:`easyebl.microscope`.

The window is opened from the easyEBL program (``easyebl``, :mod:`easyebl.app`).
"""
