"""HAL debug GUI: device state, handles/claims, static control, test waveforms.

Run:  ``python -m easyebl.hal.gui [--backend nidaqmx --device Dev1]``
"""
from __future__ import annotations

import numpy as np

from ..core.waveform import OutputChunk, chunk_arrays
from ..gui.common import (
    LogBox,
    QtCore,
    QtWidgets,
    Session,
    WaveformView,
    Worker,
    run_standalone,
)

__all__ = ["HALDebugWidget", "test_waveform", "main"]


def test_waveform(kind: str, amp_v: float, freq_hz: float, duration_s: float, rate: float, blank_square: bool):
    """Simple diagnostic stream: X follows *kind*, Y is its quadrature, blank toggles."""
    n = max(2, int(duration_s * rate))
    t = np.arange(n) / rate
    ph = 2 * np.pi * freq_hz * t
    if kind == "sine":
        x, y = np.sin(ph), np.cos(ph)
    elif kind == "square":
        x = np.sign(np.sin(ph))
        y = np.sign(np.cos(ph))
    elif kind == "triangle":
        x = 2 * np.abs(2 * ((freq_hz * t) % 1) - 1) - 1
        y = 2 * np.abs(2 * ((freq_hz * t + 0.25) % 1) - 1) - 1
    elif kind == "step":
        x = np.where(t < duration_s / 2, -1.0, 1.0)
        y = np.zeros(n)
    else:
        raise ValueError(kind)
    blank = (np.sin(ph) < 0) if blank_square else np.zeros(n, bool)
    return OutputChunk(amp_v * x, amp_v * y, blank)


class HALDebugWidget(QtWidgets.QWidget):
    _event = QtCore.Signal(str, dict)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.mgr = session.manager
        self.handle = self.mgr.get_handle("hal-debug")
        self._worker = None

        # --- info + handles
        self.info = QtWidgets.QLabel()
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.handles = QtWidgets.QTreeWidget()
        self.handles.setHeaderLabels(["handle", "claims", "active"])
        self.handles.setMaximumHeight(150)

        # --- static control
        self.x_v = QtWidgets.QDoubleSpinBox(decimals=4, minimum=-10, maximum=10, singleStep=0.1, suffix=" V")
        self.y_v = QtWidgets.QDoubleSpinBox(decimals=4, minimum=-10, maximum=10, singleStep=0.1, suffix=" V")
        lim = self.mgr.config.ao_limit_v
        self.x_v.setRange(-lim, lim)
        self.y_v.setRange(-lim, lim)
        self.blank = QtWidgets.QCheckBox("blanked")
        self.blank.setChecked(True)
        b_move = QtWidgets.QPushButton("Set beam")
        b_move.clicked.connect(self._set_static)
        b_park = QtWidgets.QPushButton("Park + blank")
        b_park.clicked.connect(self._park)
        static = QtWidgets.QGroupBox("Static beam / blanker")
        g = QtWidgets.QGridLayout(static)
        g.addWidget(QtWidgets.QLabel("X"), 0, 0)
        g.addWidget(self.x_v, 0, 1)
        g.addWidget(QtWidgets.QLabel("Y"), 1, 0)
        g.addWidget(self.y_v, 1, 1)
        g.addWidget(self.blank, 2, 0, 1, 2)
        g.addWidget(b_move, 3, 0)
        g.addWidget(b_park, 3, 1)

        lines = QtWidgets.QGroupBox("Static lines (claim → switch)")
        lg = QtWidgets.QGridLayout(lines)
        self._line_boxes = {}
        for i, name in enumerate(self.mgr.config.static_lines):
            claim = QtWidgets.QCheckBox(f"claim {name}")
            val = QtWidgets.QCheckBox("on")
            claim.toggled.connect(lambda on, n=name: self._claim(n, on))
            val.toggled.connect(lambda on, n=name: self._line(n, on))
            lg.addWidget(claim, i, 0)
            lg.addWidget(val, i, 1)
            self._line_boxes[name] = (claim, val)

        # --- test stream
        tg = QtWidgets.QGroupBox("Test stream")
        f = QtWidgets.QFormLayout(tg)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["sine", "square", "triangle", "step"])
        self.amp = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0, maximum=lim, value=1.0, suffix=" V")
        self.freq = QtWidgets.QDoubleSpinBox(decimals=1, minimum=0.1, maximum=1e5, value=1000, suffix=" Hz")
        self.dur = QtWidgets.QDoubleSpinBox(decimals=4, minimum=1e-4, maximum=60, value=0.01, suffix=" s")
        self.rate = QtWidgets.QDoubleSpinBox(decimals=0, minimum=1e3, maximum=self.mgr.config.max_ao_rate, value=400e3, suffix=" S/s")
        self.blank_sq = QtWidgets.QCheckBox("toggle blanker")
        self.ai = QtWidgets.QComboBox()
        self.ai.addItem("(none)")
        self.ai.addItems(list(self.mgr.config.ai_channels))
        if self.ai.count() > 1:
            self.ai.setCurrentIndex(1)
        for lab, w in [("waveform", self.kind), ("amplitude", self.amp), ("frequency", self.freq),
                       ("duration", self.dur), ("rate", self.rate), ("AI channel", self.ai), ("", self.blank_sq)]:
            f.addRow(lab, w)
        self.b_run = QtWidgets.QPushButton("Run test stream")
        self.b_run.clicked.connect(self._run_test)
        b_abort = QtWidgets.QPushButton("ABORT ALL")
        b_abort.setStyleSheet("background:#a33;color:white;font-weight:bold")
        b_abort.clicked.connect(self.mgr.abort_all)
        f.addRow(self.b_run, b_abort)

        self.wave = WaveformView()
        self.log = LogBox()

        left = QtWidgets.QVBoxLayout()
        for w in (self.info, self.handles, static, lines, tg, self.log):
            left.addWidget(w)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(430)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.wave, 1)

        self._event.connect(self._on_event)
        self._listener = lambda ev, d: self._event.emit(ev, d)
        self.mgr.add_listener(self._listener)
        self.refresh()

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        cap = self.mgr.capabilities()
        st = self.mgr.beam_state
        self.info.setText(
            f"<b>{cap.name}</b>  {'(simulated)' if cap.simulated else ''}<br>"
            f"AO ≤ {cap.max_ao_rate / 1e3:.0f} kS/s, AI ≤ {cap.max_ai_aggregate_rate / 1e3:.0f} kS/s aggregate<br>"
            f"AI: {', '.join(cap.ai_channels) or '-'} &nbsp; timed DO: {', '.join(cap.timed_lines) or '-'}<br>"
            f"beam: X={st.x_v:+.4f} V Y={st.y_v:+.4f} V {'BLANKED' if st.blank else '<span style=color:red>ON</span>'}"
            f" &nbsp; busy: {self.mgr.active_owner or 'no'}"
        )
        self.handles.clear()
        claims = self.mgr.claims()
        for h in self.mgr.handles():
            it = QtWidgets.QTreeWidgetItem(
                [h.owner, ", ".join(r for r, o in claims.items() if o == h.owner), "●" if h.is_active else ""]
            )
            self.handles.addTopLevelItem(it)

    def _on_event(self, ev: str, d: dict) -> None:
        if ev in ("stream_started", "stream_finished", "claim", "release", "line"):
            self.log.log(f"[{ev}] {d}")
        self.refresh()

    # ---------------------------------------------------------------- actions
    def _guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.log.log(f"ERROR {type(e).__name__}: {e}")
        self.refresh()

    def _set_static(self):
        self._guard(self.handle.park, self.x_v.value(), self.y_v.value(), self.blank.isChecked())

    def _park(self):
        self._guard(self.handle.park)

    def _claim(self, name, on):
        self._guard(self.handle.claim if on else self.handle.release, name)

    def _line(self, name, on):
        self._guard(self.handle.set_line, name, on)

    def _run_test(self):
        rate = self.rate.value()
        chunk = test_waveform(
            self.kind.currentText(), self.amp.value(), self.freq.value(), self.dur.value(), rate, self.blank_sq.isChecked()
        )
        ai = [] if self.ai.currentIndex() == 0 else [self.ai.currentText()]
        self.b_run.setEnabled(False)

        def job():
            res = self.handle.run_stream(rate, chunk_arrays(chunk.x, chunk.y, chunk.blank, 50_000), ai_channels=ai)
            return chunk, res

        self._worker = Worker(job)
        self._worker.result.connect(lambda r: self.wave.show_chunk(r[0], rate, r[1].ai))
        self._worker.error.connect(lambda e: self.log.log("ERROR " + e))
        self._worker.finished.connect(lambda: self.b_run.setEnabled(True))
        self._worker.start()

    def closeEvent(self, ev):  # pragma: no cover - interactive
        self.mgr.remove_listener(self._listener)
        self.handle.close()
        super().closeEvent(ev)


def main(argv=None) -> int:
    return run_standalone(HALDebugWidget, "HAL debug", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
