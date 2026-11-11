"""Pulse-processor dialog: device info, shaping preset, channel count and raw DP5 parameters."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...hal.mca import MCAError, PulseProcessor

__all__ = ["ProcessorDialog"]

PARAMS = [
    ("MCAC", "MCA channels"),
    ("GAIN", "Total gain"),
    ("TPEA", "Peaking time (µs)"),
    ("TFLA", "Flat top (µs)"),
    ("THSL", "Slow threshold (%)"),
    ("THFA", "Fast threshold"),
    ("PURE", "Pile-up rejection"),
    ("RTDE", "Risetime discrimination"),
    ("AINP", "Input polarity"),
    ("INOF", "Input offset"),
    ("RESL", "Reset lockout (µs)"),
    ("PRER", "Preset real time"),
]


class ProcessorDialog(QtWidgets.QDialog):
    """Settings of the pulse processor.  Changes are written on Apply (MCAC changes the calibration!)."""

    channelsChanged = QtCore.Signal(int, int)  # old, new

    def __init__(self, mca: PulseProcessor, parent=None):
        super().__init__(parent)
        self.mca = mca
        self.setWindowTitle(f"Pulse processor – {mca.name}")
        self.resize(520, 560)
        info = mca.info()
        self.l_info = QtWidgets.QLabel("<br>".join(f"<b>{k}</b>: {v}" for k, v in info.items()))
        self.l_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.cb_speed = QtWidgets.QComboBox()
        self.speeds = mca.speeds()
        for k, d in self.speeds.items():
            self.cb_speed.addItem(d, k)
        cur = mca.current_speed()
        if cur in self.speeds:
            self.cb_speed.setCurrentIndex(list(self.speeds).index(cur))
        self.b_speed = QtWidgets.QPushButton("Apply preset")
        self.b_speed.clicked.connect(self._apply_speed)
        self.table = QtWidgets.QTableWidget(len(PARAMS), 3)
        self.table.setHorizontalHeaderLabels(["Parameter", "Meaning", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.b_read = QtWidgets.QPushButton("Read")
        self.b_read.clicked.connect(self.read)
        self.b_write = QtWidgets.QPushButton("Write changed")
        self.b_write.clicked.connect(self.write)
        self.b_thr = QtWidgets.QPushButton("Auto thresholds (beam off!)")
        self.b_thr.setVisible(hasattr(mca, "auto_threshold"))
        self.b_thr.clicked.connect(self._auto_thr)
        self.msg = QtWidgets.QLabel()
        self.msg.setWordWrap(True)
        top = QtWidgets.QGroupBox("Device")
        QtWidgets.QVBoxLayout(top).addWidget(self.l_info)
        sp = QtWidgets.QGroupBox("Shaping (speed vs resolution)")
        sl = QtWidgets.QHBoxLayout(sp)
        sl.addWidget(self.cb_speed, 1)
        sl.addWidget(self.b_speed)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(top)
        lay.addWidget(sp)
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        for b in (self.b_read, self.b_write, self.b_thr):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(self.msg)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._values: dict = {}
        self.read()

    def _busy(self) -> bool:
        if self.mca.owner is not None:
            self.msg.setText(f"busy: in use by '{self.mca.owner}' – stop the acquisition first")
            return True
        return False

    def read(self):
        try:
            vals = self.mca.get_settings([p for p, _ in PARAMS])
        except MCAError as e:
            self.msg.setText(str(e))
            return
        self._values = dict(vals)
        for r, (p, d) in enumerate(PARAMS):
            for c, v in enumerate((p, d, vals.get(p, ""))):
                it = QtWidgets.QTableWidgetItem(str(v))
                if c < 2:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                if p == "PRER" and c == 2:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)  # managed by the acquisitions
                self.table.setItem(r, c, it)
        self.msg.setText("read")

    def write(self):
        if self._busy():
            return
        changes = {}
        for r, (p, _d) in enumerate(PARAMS):
            v = self.table.item(r, 2).text().strip()
            if v and v != str(self._values.get(p, "")) and p != "PRER":
                changes[p] = v
        if not changes:
            self.msg.setText("nothing changed")
            return
        old = self.mca.channels
        try:
            self.mca.configure(changes)
        except MCAError as e:
            self.msg.setText(str(e))
            return
        if self.mca.channels != old:
            self.channelsChanged.emit(old, self.mca.channels)
        self.read()
        self.msg.setText("written: " + ", ".join(f"{k}={v}" for k, v in changes.items()))

    def _apply_speed(self):
        if self._busy():
            return
        k = self.cb_speed.currentData()
        try:
            self.mca.apply_speed(int(k))
        except MCAError as e:
            self.msg.setText(str(e))
            return
        self.read()
        self.msg.setText(f"speed preset {k} applied (peak positions are kept; re-check the calibration)")

    def _auto_thr(self):
        if self._busy():
            return
        if QtWidgets.QMessageBox.question(self, "Auto thresholds",
                                          "Blank the beam / no X-rays on the detector.  Continue?") != \
                QtWidgets.QMessageBox.Yes:
            return
        try:
            res = self.mca.auto_threshold()
        except Exception as e:  # noqa: BLE001
            self.msg.setText(str(e))
            return
        self.read()
        self.msg.setText(f"thresholds: {res}")
