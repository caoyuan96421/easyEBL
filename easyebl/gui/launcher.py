"""All debug/demo GUIs as tabs sharing one DAQ manager.

Run:  ``easyebl-demo`` or ``python -m easyebl.gui.launcher [--backend nidaqmx --device Dev1]``

Each tab obtains its own :class:`~easyebl.hal.DAQHandle` from the shared
manager; the HAL tab shows the handles and which one is streaming.
"""
from __future__ import annotations

from .common import QtWidgets, Session, ensure_app, standalone_parser

__all__ = ["MainWindow", "main"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session: Session):
        super().__init__()
        from ..alignment.gui import AlignmentWidget
        from ..hal.gui import HALDebugWidget
        from ..imaging.gui import ImagingWidget
        from ..litho.gui_dose import DoseCalculatorWidget
        from ..litho.gui_gds import GDSWidget
        from ..litho.gui_vector import VectorLithoWidget

        self.session = session
        tabs = QtWidgets.QTabWidget()
        self.hal = HALDebugWidget(session)
        self.imaging = ImagingWidget(session)
        self.alignment = AlignmentWidget(session)
        self.vector = VectorLithoWidget(session)
        self.gds = GDSWidget(session)
        self.dose = DoseCalculatorWidget(session)
        tabs.addTab(self.hal, "HAL")
        tabs.addTab(self.imaging, "Imaging")
        tabs.addTab(self.alignment, "Alignment")
        tabs.addTab(self.vector, "Vector litho")
        tabs.addTab(self.gds, "GDS litho")
        tabs.addTab(self.dose, "Dose calculator")
        self.tabs = tabs
        self.setCentralWidget(tabs)
        cap = session.manager.capabilities()
        self.setWindowTitle(f"easyEBL demo – {cap.name}")
        if session.simulator is not None:
            self.alignment._apply_sim()
        tabs.currentChanged.connect(lambda _: self.hal.refresh())

    def closeEvent(self, ev):  # pragma: no cover - interactive
        self.session.manager.abort_all()
        super().closeEvent(ev)


def main(argv=None) -> int:
    args = standalone_parser("easyEBL demo").parse_args(argv)
    app = ensure_app()
    session = Session.from_args(args)
    w = MainWindow(session)
    w.resize(1400, 900)
    w.show()
    try:
        return app.exec()
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
