"""A modal window while an automatic function runs (auto focus, stigmation, the SEM's ABC and gun
alignment): the program waits for it, the window shows what is happening and closes by itself.

``BusyDialog(title, text, parent, cancel=fn)``: *cancel* (optional) is called by the Cancel button or
Esc; without it the function cannot be interrupted (the SEM's own routines) and Esc does nothing –
or *esc* (optional) is called instead (e.g. "do not restart the scan afterwards").
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["BusyDialog"]


class BusyDialog(QtWidgets.QDialog):
    def __init__(self, title: str, text: str, parent=None, cancel=None, cancel_text: str = "Cancel", esc=None,
                 esc_text: str = ""):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlags((self.windowFlags() | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
                            & ~QtCore.Qt.WindowCloseButtonHint & ~QtCore.Qt.WindowContextHelpButtonHint)
        self._cancel = cancel
        self._esc = esc
        self._esc_text = esc_text
        self.finished_ok = False
        self._done = False
        lay = QtWidgets.QVBoxLayout(self)
        self.l_text = QtWidgets.QLabel(text)
        self.l_text.setWordWrap(True)
        f = self.l_text.font()
        f.setBold(True)
        self.l_text.setFont(f)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)  # busy
        self.bar.setTextVisible(False)
        self.l_progress = QtWidgets.QLabel("")
        self.l_progress.setWordWrap(True)
        self.l_progress.setStyleSheet("color:#9aa")
        lay.addWidget(self.l_text)
        lay.addWidget(self.bar)
        lay.addWidget(self.l_progress)
        self.b_cancel = None
        if cancel is not None:
            self.b_cancel = QtWidgets.QPushButton(cancel_text)
            self.b_cancel.clicked.connect(self.cancel)
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(self.b_cancel)
            lay.addLayout(row)
        self.setMinimumWidth(420)

    def show_busy(self) -> "BusyDialog":
        """Show it (modal: the rest of the program waits) without blocking the caller."""
        self.open()
        return self

    def set_progress(self, text: str) -> None:
        self.l_progress.setText(text)

    def cancel(self) -> None:
        if self._cancel is not None and self.b_cancel is not None and self.b_cancel.isEnabled():
            self.b_cancel.setEnabled(False)
            self.b_cancel.setText("Cancelling…")
            self._cancel()

    def finish(self, ok: bool = True) -> None:
        self.finished_ok = ok
        self._done = True
        self.done(0)

    @property
    def is_done(self) -> bool:
        return self._done

    def keyPressEvent(self, ev):  # noqa: N802
        if ev.key() == QtCore.Qt.Key_Escape:
            if self._cancel is not None:
                self.cancel()
            elif self._esc is not None:
                self._esc()
                if self._esc_text:
                    self.set_progress(self._esc_text)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def reject(self):  # Esc / the window's close: only through finish()
        if self._done:
            super().reject()

    def closeEvent(self, ev):  # noqa: N802
        if not self._done:
            ev.ignore()
            return
        super().closeEvent(ev)
