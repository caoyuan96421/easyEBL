"""SEM image display: fitted image, scan-line marker, data bar with scale bar.

Mouse:
* double-click – move the beam so that the clicked point becomes the centre
* drag (left button) – beam shift, the image follows the mouse
* wheel – magnification (emitted; the SEM control is a stub for now)
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..export import to_uint8
from .display import display_magnification, format_length, nice_scale_bar

__all__ = ["SEMImageView", "render_annotated"]

BAR_H = 30


def _pixel_to_field(cfg, col: float, row: float) -> tuple[float, float]:
    pw, ph = cfg.pixel_size_um
    u = -cfg.width_um / 2 + (col + 0.5) * pw
    v = cfg.height / 2 - (row + 0.5) * ph
    c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
    return (cfg.center_um[0] + c * u - s * v, cfg.center_um[1] + s * u + c * v)


def _draw_data_bar(p: QtGui.QPainter, rect: QtCore.QRectF, frame, info: dict) -> None:
    p.fillRect(rect, QtGui.QColor(0, 0, 0))
    cfg = frame.config if frame is not None else None
    font = QtGui.QFont("Segoe UI", 9)
    font.setStyleHint(QtGui.QFont.SansSerif)
    p.setFont(font)
    p.setPen(QtGui.QColor(235, 235, 235))
    fm = QtGui.QFontMetrics(font)
    y = rect.center().y() + fm.ascent() / 2 - 2
    items = [
        info.get("kv", "-- kV"),
        info.get("wd", "WD --"),
        display_magnification(cfg.width_um) if cfg else "×—",
        info.get("detector", "SE"),
    ]
    x = rect.left() + 8
    for t in items:
        p.drawText(QtCore.QPointF(x, y), t)
        x += fm.horizontalAdvance(t) + 16
    # scale bar on the right
    if cfg is not None and rect.width() > 200:
        img_w_px = info.get("image_width_px", rect.width())
        um_per_px = cfg.width_um / max(img_w_px, 1)
        L = nice_scale_bar(cfg.width_um, 0.18)
        bar_px = L / um_per_px
        label = format_length(L)
        lw = fm.horizontalAdvance(label)
        right = rect.right() - 10 - fm.horizontalAdvance("0000-00-00 00:00") - 16
        x1 = right
        x0 = x1 - bar_px
        if x0 > x + lw + 10:
            yb = rect.center().y() + 1
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 3)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x0, yb), QtCore.QPointF(x1, yb))
            pen.setWidth(1)
            p.setPen(pen)
            for k in range(11):  # Hitachi-style tick dots
                xt = x0 + bar_px * k / 10
                p.drawLine(QtCore.QPointF(xt, yb - (5 if k in (0, 10) else 3)), QtCore.QPointF(xt, yb - 1))
            p.drawText(QtCore.QPointF(x0 - lw - 8, y), label)
        p.setPen(QtGui.QColor(200, 200, 200))
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(frame.timestamp))
        p.drawText(QtCore.QPointF(rect.right() - 8 - fm.horizontalAdvance(ts), y), ts)


def render_annotated(frame, levels, info: dict | None = None) -> QtGui.QImage:
    """Full-resolution image with the data bar burned in (for 'Save with data bar')."""
    u8 = to_uint8(frame.image, levels)
    h, w = u8.shape
    img = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    bar = max(BAR_H, h // 24)
    out = QtGui.QImage(w, h + bar, QtGui.QImage.Format_RGB32)
    out.fill(QtGui.QColor(0, 0, 0))
    p = QtGui.QPainter(out)
    p.drawImage(0, 0, img)
    inf = dict(info or {})
    inf["image_width_px"] = w
    _draw_data_bar(p, QtCore.QRectF(0, h, w, bar), frame, inf)
    p.end()
    return out


class SEMImageView(QtWidgets.QWidget):
    centerRequested = QtCore.Signal(float, float)  # absolute beam shift target (µm)
    shiftDragged = QtCore.Signal(float, float)  # incremental beam shift (µm)
    wheelSteps = QtCore.Signal(int)  # magnification steps (stub)
    pointerMoved = QtCore.Signal(object)  # (x_um, y_um) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 380)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self._qimg: QtGui.QImage | None = None
        self._u8 = None
        self._frame = None
        self.info: dict = {}
        self.show_scan_line = True
        self.show_crosshair = False
        self._drag = None
        self._paint_ms = 0.0

    # ---------------------------------------------------------------- data
    def set_frame(self, frame, levels) -> None:
        self._frame = frame
        self._u8 = np.ascontiguousarray(to_uint8(frame.image, levels))
        h, w = self._u8.shape
        self._qimg = QtGui.QImage(self._u8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        self.update()

    def frame(self):
        return self._frame

    def clear(self) -> None:
        self._frame = self._qimg = self._u8 = None
        self.update()

    # ------------------------------------------------------------ geometry
    def image_rect(self) -> QtCore.QRectF:
        area = QtCore.QRectF(0, 0, self.width(), self.height() - BAR_H)
        if self._qimg is None:
            return area
        iw, ih = self._qimg.width(), self._qimg.height()
        s = min(area.width() / iw, area.height() / ih)
        w, h = iw * s, ih * s
        return QtCore.QRectF((area.width() - w) / 2, (area.height() - h) / 2, w, h)

    def _to_pixel(self, pos: QtCore.QPointF):
        if self._qimg is None:
            return None
        r = self.image_rect()
        if not r.contains(pos):
            return None
        col = (pos.x() - r.left()) / r.width() * self._qimg.width() - 0.5
        row = (pos.y() - r.top()) / r.height() * self._qimg.height() - 0.5
        return col, row

    # --------------------------------------------------------------- paint
    def paintEvent(self, ev):  # noqa: N802
        t0 = time.perf_counter()
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(16, 16, 18))
        r = self.image_rect()
        if self._qimg is not None:
            down = r.width() < self._qimg.width()
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, down)
            p.drawImage(r, self._qimg)
            f = self._frame
            if self.show_scan_line and f is not None and not f.complete and f.scan_row is not None:
                y = r.top() + (f.scan_row + 1) / f.config.ny * r.height()
                p.setPen(QtGui.QPen(QtGui.QColor(255, 210, 0, 200), 1))
                p.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
            if self.show_crosshair:
                p.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255, 160), 1, QtCore.Qt.DashLine))
                c = r.center()
                p.drawLine(QtCore.QPointF(c.x() - 20, c.y()), QtCore.QPointF(c.x() + 20, c.y()))
                p.drawLine(QtCore.QPointF(c.x(), c.y() - 20), QtCore.QPointF(c.x(), c.y() + 20))
        else:
            p.setPen(QtGui.QColor(120, 120, 120))
            p.drawText(self.rect().adjusted(0, 0, 0, -BAR_H), QtCore.Qt.AlignCenter, "No image – press Run")
        info = dict(self.info)
        info["image_width_px"] = r.width()
        bar = QtCore.QRectF(0, self.height() - BAR_H, self.width(), BAR_H)
        if self._frame is not None:
            # the data bar spans the image width, like on the SEM screen
            bar = QtCore.QRectF(r.left(), self.height() - BAR_H, r.width(), BAR_H)
        _draw_data_bar(p, bar, self._frame, info)
        p.end()
        self._paint_ms = (time.perf_counter() - t0) * 1e3

    # --------------------------------------------------------------- mouse
    def mouseDoubleClickEvent(self, ev):  # noqa: N802
        px = self._to_pixel(ev.position())
        if px is not None and self._frame is not None:
            self.centerRequested.emit(*_pixel_to_field(self._frame.config, *px))

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == QtCore.Qt.LeftButton and self._frame is not None:
            self._drag = ev.position()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):  # noqa: N802
        f = self._frame
        if self._drag is not None and f is not None:
            d = ev.position() - self._drag
            r = self.image_rect()
            if r.width() > 0 and (abs(d.x()) + abs(d.y())) >= 3:
                cfg = f.config
                um_x = -d.x() / r.width() * cfg.width_um
                um_y = d.y() / r.height() * cfg.height
                c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
                self.shiftDragged.emit(c * um_x - s * um_y, s * um_x + c * um_y)
                self._drag = ev.position()
        px = self._to_pixel(ev.position())
        self.pointerMoved.emit(None if px is None or f is None else _pixel_to_field(f.config, *px))

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._drag = None
        self.unsetCursor()

    def wheelEvent(self, ev):  # noqa: N802
        steps = int(round(ev.angleDelta().y() / 120))
        if steps:
            self.wheelSteps.emit(steps)

    def leaveEvent(self, ev):  # noqa: N802
        self.pointerMoved.emit(None)
