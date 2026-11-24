"""Scan monitor widget: the X/Y deflection and blanker as an oscilloscope in XY mode with persistence.

The image is built by a :class:`~easyebl.hal.scanmonitor.ScanMonitorWorker`
in its own thread (draining the DAQ's :class:`~easyebl.hal.scanmonitor.ScanTap`,
binning, rendering to a :class:`QImage`); this widget only paints the finished
frame and its graticule.  The worker runs while the widget is shown.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..hal.scanmonitor import FULL_SCALE_V, RESOLUTIONS, MonitorStats, ScanMonitorWorker, ScanTap, phosphor_lut

__all__ = ["ScanMonitorWidget", "ScopeScreen"]

_LUT = phosphor_lut()


def frame_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """uint8 intensity → indexed QImage with the phosphor colour table (owns its data)."""
    h, w = frame.shape
    buf = np.ascontiguousarray(frame)
    img = QtGui.QImage(buf.data, w, h, w, QtGui.QImage.Format_Indexed8)
    img.setColorTable(_LUT)
    return img.copy()


class ScopeScreen(QtWidgets.QWidget):
    """Square screen: the latest frame (nearest-neighbour, pixelated) under a 10×10 graticule."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: QtGui.QImage | None = None
        #: the beam now: (x, y field volts, blanked); drawn as a solid (beam on) or hollow (blanked) dot
        self.spot: tuple[float, float, bool] | None = None
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(320, 320)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, w):  # noqa: N802
        return w

    def screen_rect(self) -> QtCore.QRect:
        s = min(self.width(), self.height()) - 2
        return QtCore.QRect((self.width() - s) // 2, (self.height() - s) // 2, s, s)

    SPOT_RADIUS = 6.0
    SPOT_COLOR = QtGui.QColor(255, 255, 160)

    def spot_point(self) -> QtCore.QPointF | None:
        """Screen position of :attr:`spot` (clipped to the field edge)."""
        if self.spot is None:
            return None
        r = self.screen_rect()
        fs = FULL_SCALE_V
        x = min(max(self.spot[0], -fs), fs)
        y = min(max(self.spot[1], -fs), fs)
        return QtCore.QPointF(r.left() + (x + fs) / (2 * fs) * r.width(),
                              r.top() + (fs - y) / (2 * fs) * r.height())

    def _paint_spot(self, p: QtGui.QPainter) -> None:
        c = self.spot_point()
        if c is None:
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rad = self.SPOT_RADIUS
        halo = QtGui.QRadialGradient(c, 2.5 * rad)
        halo.setColorAt(0.0, QtGui.QColor(255, 255, 160, 110))
        halo.setColorAt(1.0, QtGui.QColor(255, 255, 160, 0))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(c, 2.5 * rad, 2.5 * rad)
        if self.spot[2]:  # blanked: an empty ring
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(self.SPOT_COLOR, 2.0))
        else:  # beam on: a solid dot
            p.setBrush(self.SPOT_COLOR)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.0))
        p.drawEllipse(c, rad, rad)

    def paintEvent(self, ev):  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        r = self.screen_rect()
        p.fillRect(r, QtGui.QColor("#000"))
        if self.image is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
            p.drawImage(r, self.image)
        grid = QtGui.QPen(QtGui.QColor(90, 110, 90, 110), 1, QtCore.Qt.DotLine)
        p.setPen(grid)
        for k in range(1, 10):
            x = r.left() + r.width() * k / 10
            y = r.top() + r.height() * k / 10
            p.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom()))
            p.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
        p.setPen(QtGui.QPen(QtGui.QColor(110, 140, 110, 160), 1))
        p.drawLine(QtCore.QPointF(r.center().x(), r.top()), QtCore.QPointF(r.center().x(), r.bottom()))
        p.drawLine(QtCore.QPointF(r.left(), r.center().y()), QtCore.QPointF(r.right(), r.center().y()))
        p.drawRect(r)
        p.setPen(QtGui.QColor(140, 170, 140))
        f = p.font()
        f.setPointSizeF(max(6.0, f.pointSizeF() * 0.8))
        p.setFont(f)
        fs = f"{FULL_SCALE_V:g} V"
        p.drawText(r.adjusted(4, 2, -4, -2), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter, f"Y +{fs}")
        p.drawText(r.adjusted(4, 2, -4, -2), QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight, f"X +{fs}")
        p.drawText(r.adjusted(4, 2, -4, -2), QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft, f"−{fs}")
        self._paint_spot(p)  # always on top, blanked or not
        p.end()


class ScanMonitorWidget(QtWidgets.QWidget):
    """Screen + resolution / retention controls.  ``on_settings(resolution, retention_s, fade)``
    is called when the user changes a setting (to save it)."""

    _frame = QtCore.Signal(QtGui.QImage, object)

    def __init__(self, tap: ScanTap, resolution: int = 256, retention_s: float = 2.0, fade: bool = True,
                 on_settings: Callable[[int, float, bool], None] | None = None, interval_s: float = 0.1,
                 orientation: tuple[bool, bool, bool] = (False, False, False), parent=None):
        super().__init__(parent)
        self.on_settings = on_settings
        self._busy = False  # a frame is queued for painting: the worker skips the next ones
        self.stats = MonitorStats()
        self.frames = 0
        self.worker = ScanMonitorWorker(tap, self._emit_frame, resolution, retention_s, fade, interval_s,
                                        orientation)
        self.worker.skip_frame = lambda: self._busy
        self._frame.connect(self._show_frame, QtCore.Qt.QueuedConnection)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.screen = ScopeScreen()
        lay.addWidget(self.screen, 1)
        row = QtWidgets.QHBoxLayout()
        self.res = QtWidgets.QComboBox()
        for r in RESOLUTIONS:
            self.res.addItem(f"{r}²", r)
        if self.res.findData(int(resolution)) < 0:
            self.res.addItem(f"{int(resolution)}²", int(resolution))
        self.res.setCurrentIndex(self.res.findData(int(resolution)))
        self.res.setToolTip("display resolution (pixels across the ±10 V field)")
        self.retention = QtWidgets.QDoubleSpinBox()
        self.retention.setRange(0.0, 3600.0)
        self.retention.setDecimals(1)
        self.retention.setSingleStep(0.5)
        self.retention.setSuffix(" s")
        self.retention.setSpecialValueText("∞")
        self.retention.setValue(float(retention_s))
        self.retention.setToolTip("persistence: how long a sample stays on the screen (∞: until cleared)")
        self.fade = QtWidgets.QCheckBox("Fade")
        self.fade.setChecked(bool(fade))
        self.fade.setToolTip("older samples fade out over the persistence time")
        b = QtWidgets.QToolButton()
        b.setText("Clear")
        b.clicked.connect(self.clear)
        row.addWidget(QtWidgets.QLabel("Resolution"))
        row.addWidget(self.res)
        row.addWidget(QtWidgets.QLabel("Persistence"))
        row.addWidget(self.retention)
        row.addWidget(self.fade)
        row.addStretch(1)
        row.addWidget(b)
        lay.addLayout(row)
        self.l_stats = QtWidgets.QLabel("idle")
        self.l_stats.setStyleSheet("color:#9aa")
        lay.addWidget(self.l_stats)
        self.res.currentIndexChanged.connect(self._changed)
        self.retention.valueChanged.connect(self._changed)
        self.fade.toggled.connect(self._changed)

    # ---------------------------------------------------------------- settings
    def settings(self) -> tuple[int, float, bool]:
        return int(self.res.currentData()), float(self.retention.value()), self.fade.isChecked()

    def _changed(self, *_):
        res, ret, fade = self.settings()
        self.worker.configure(res, ret, fade)
        if self.on_settings is not None:
            self.on_settings(res, ret, fade)

    def clear(self):
        self.worker.clear()

    def set_orientation(self, invert_x: bool, invert_y: bool, swap_xy: bool):
        """The scan outputs' invert / swap (as in the hardware configuration): shown in image coordinates."""
        self.worker.configure(orientation=(invert_x, invert_y, swap_xy))

    # ---------------------------------------------------------------- worker
    def start(self):
        self.worker.start()

    def stop(self):
        self.worker.stop()

    def _emit_frame(self, frame: np.ndarray, stats: MonitorStats):  # worker thread
        img = frame_to_qimage(frame)
        self._busy = True
        try:
            self._frame.emit(img, stats)
        except RuntimeError:  # the widget is gone
            pass

    def _show_frame(self, img: QtGui.QImage, stats: MonitorStats):
        self._busy = False
        self.frames += 1
        self.stats = stats
        self.screen.image = img
        self.screen.spot = stats.spot
        self.screen.update()
        if stats.rate > 0:
            txt = f"{stats.rate / 1e3:.0f} kS/s · {stats.unblanked * 100:.0f}% unblanked"
        else:
            txt = "no stream"
        if stats.spot is not None:
            x, y, b = stats.spot
            txt += f" · beam ({x:+.2f}, {y:+.2f}) V {'blanked' if b else 'ON'}"
        if stats.dropped:
            txt += f" · {stats.dropped:,} samples skipped"
        self.l_stats.setText(txt)

    def showEvent(self, ev):  # noqa: N802
        super().showEvent(ev)
        self.start()

    def hideEvent(self, ev):  # noqa: N802
        self.stop()
        super().hideEvent(ev)

    def closeEvent(self, ev):  # noqa: N802
        self.stop()
        super().closeEvent(ev)
