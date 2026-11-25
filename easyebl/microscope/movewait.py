"""A modal "stage moving" dialog for programmed moves (lithography, scripts with a GUI).

While the stage moves the microscope port carries nothing else, so a subsystem
that moves the stage on its own shows :class:`StageMoveDialog`: it blocks the
user's interaction with the application, shows the target and the live
positions / axis states (and how far each axis is still off while it settles),
offers STOP, and closes by itself when the move ends.

* :func:`run_blocking_move` – on the GUI thread: start a move, show the dialog
  until it ends, return the final positions (raises the move's error).
* :class:`StageMoveDialog` ``(stage, future)`` – for a move started elsewhere (e.g.
  on a worker thread): open it on the GUI thread with :meth:`QDialog.exec` or
  :meth:`QDialog.open`; it closes when *future* is done.
"""
from __future__ import annotations

import concurrent.futures

from PySide6 import QtCore, QtWidgets

from .stage import AXES, StageControl

__all__ = ["StageMoveDialog", "run_blocking_move"]


def _describe(targets: dict, relative: bool, stage: StageControl) -> str:
    return ", ".join(f"{a} {'by ' if relative else ''}{v:{'+' if relative else ''}g} {stage.unit(a)}"
                     for a, v in targets.items())


class StageMoveDialog(QtWidgets.QDialog):
    """Modal wait dialog for one stage move (see the module docstring)."""

    def __init__(self, stage: StageControl, future: concurrent.futures.Future, description: str = "",
                 parent=None):
        super().__init__(parent)
        self.stage, self.future = stage, future
        self.setWindowTitle("Stage moving")
        self.setModal(True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel("Stage moving" + (f": {description}" if description else ""))
        head.setStyleSheet("font-size:15px;font-weight:bold")
        head.setWordWrap(True)
        lay.addWidget(head)
        note = QtWidgets.QLabel("The microscope port is reserved for the move: SEM settings wait until it ends.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#999")
        lay.addWidget(note)
        self.l_pos = QtWidgets.QLabel()
        self.l_pos.setStyleSheet("font-family:Consolas;font-size:13px")
        lay.addWidget(self.l_pos)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)  # busy indicator
        lay.addWidget(bar)
        self.b_stop = QtWidgets.QPushButton("STOP")
        self.b_stop.setMinimumHeight(40)
        self.b_stop.setStyleSheet("QPushButton{background:#c62828;color:white;font-size:16px;font-weight:bold}")
        self.b_stop.clicked.connect(self.stop)
        lay.addWidget(self.b_stop)
        self.stopped = False
        self._timer = QtCore.QTimer(self, interval=200, timeout=self._tick)
        self._timer.start()
        self._tick()

    def stop(self):
        self.stopped = True
        self.stage.stop()
        self.b_stop.setEnabled(False)
        self.b_stop.setText("STOP sent – waiting for the axes")

    def _tick(self):
        st = self.stage
        rows = []
        for a in AXES:
            p = st.positions.get(a)
            v = "—" if p is None else (f"{p:12,.1f}" if st.unit(a) == "µm" else f"{p:12.3f}")
            rows.append(f"{a}  {v} {st.unit(a):2s}  {st.motion_state(a)}")
        off = st.settling
        if off:
            rows.append("settling: " + ", ".join(
                f"{a} {d:+.2f} {st.unit(a)}" if st.unit(a) == "µm" else f"{a} {d:+.4f} {st.unit(a)}"
                for a, d in off.items() if abs(d) > st.settle_tol.get(a, 0.0)) + " off")
        self.l_pos.setText("\n".join(rows))
        if self.future.done():
            self._timer.stop()
            self.done(QtWidgets.QDialog.Accepted if self._ok() else QtWidgets.QDialog.Rejected)

    def _ok(self) -> bool:
        f = self.future
        return not f.cancelled() and f.exception() is None

    def reject(self):  # Esc: STOP, the dialog stays until the move has ended
        if not self.future.done():
            self.stop()
            return
        super().reject()


def run_blocking_move(stage: StageControl, targets: dict, relative: bool = False, approach: bool = True,
                      parent=None) -> dict:
    """Move and wait with :class:`StageMoveDialog` (GUI thread); the final positions, or the move's
    exception (:class:`~easyebl.microscope.stage.StageBusy`, ``MotionError``, ``CancelledError``)."""
    fut = stage.move(targets, relative=relative, approach=approach)
    d = StageMoveDialog(stage, fut, _describe(targets, relative, stage), parent)
    if not fut.done():
        d.exec()
    d.deleteLater()
    return fut.result(0)
