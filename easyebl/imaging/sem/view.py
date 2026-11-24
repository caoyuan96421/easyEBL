"""SEM image display: fitted image, scan-line marker, data bar with scale bar.

Mouse:
* double-click – move the beam so that the clicked point becomes the centre
* drag (left button) – beam shift, the image follows the mouse
* wheel – zoom: the swept range narrows / widens (the column magnification is not touched)
* Measure mode (:attr:`SEMImageView.measure_mode`): drag draws a line instead; its length is shown on
  the image.  The line is kept in field µm, so it stays on the feature while the scan pans or zooms.
* Pick mode (:meth:`SEMImageView.start_pick`): a click emits :attr:`SEMImageView.pointPicked` with the
  field position (µm); the prompt is shown on top of the image and the targets (e.g. a predicted
  mark position) as crosses.  Used by the lithography alignment.

The data bar shows the image magnification (column magnification × full field / scan
width); the inset in the lower-left corner shows the scanned region within the full field.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..export import to_uint8
from .display import display_magnification, format_length, image_magnification, nice_scale_bar

__all__ = ["SEMImageView", "render_annotated"]

BAR_H = 30


def _pixel_to_field(cfg, col: float, row: float) -> tuple[float, float]:
    pw, ph = cfg.pixel_size_um
    u = -cfg.width_um / 2 + (col + 0.5) * pw
    v = cfg.height / 2 - (row + 0.5) * ph
    c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
    return (cfg.center_um[0] + c * u - s * v, cfg.center_um[1] + s * u + c * v)


def _field_to_pixel(cfg, x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`_pixel_to_field`: (col, row) of a field point (may be outside the image)."""
    pw, ph = cfg.pixel_size_um
    dx, dy = x - cfg.center_um[0], y - cfg.center_um[1]
    c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (u + cfg.width_um / 2) / pw - 0.5, (cfg.height / 2 - v) / ph - 0.5


def _draw_measure(p: QtGui.QPainter, a: QtCore.QPointF, b: QtCore.QPointF, label: str) -> None:
    """The measuring line: yellow on a dark outline, end ticks, the length in a box at the middle."""
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    d = b - a
    n = math.hypot(d.x(), d.y())
    tick = QtCore.QPointF(-d.y() / n * 7, d.x() / n * 7) if n > 0 else QtCore.QPointF(0, 7)
    lines = [QtCore.QLineF(a, b), QtCore.QLineF(a - tick, a + tick), QtCore.QLineF(b - tick, b + tick)]
    for pen in (QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 4), QtGui.QPen(QtGui.QColor(255, 220, 0), 2)):
        pen.setCapStyle(QtCore.Qt.FlatCap)
        p.setPen(pen)
        p.drawLines(lines)
    font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold)
    p.setFont(font)
    fm = QtGui.QFontMetrics(font)
    m = (a + b) / 2
    box = QtCore.QRectF(0, 0, fm.horizontalAdvance(label) + 10, fm.height() + 4)
    box.moveCenter(m + QtCore.QPointF(0, -(box.height() / 2 + 6)))
    p.fillRect(box, QtGui.QColor(0, 0, 0, 190))
    p.setPen(QtGui.QColor(255, 220, 0))
    p.drawText(box, QtCore.Qt.AlignCenter, label)
    p.setRenderHint(QtGui.QPainter.Antialiasing, False)


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
        display_magnification(image_magnification(info.get("column_mag"), info.get("field_um"), cfg.width_um)
                              if cfg else None),
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


def scan_corners(center_um, width_um: float, height_um: float, rotation_rad: float) -> np.ndarray:
    """Corners (4, 2) of a scan rectangle in field µm."""
    u = np.array([-1, 1, 1, -1]) * width_um / 2
    v = np.array([1, 1, -1, -1]) * height_um / 2
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    return np.column_stack([center_um[0] + c * u - s * v, center_um[1] + s * u + c * v])


def _draw_fov_inset(p: QtGui.QPainter, image_rect: QtCore.QRectF, field_um: float, scan) -> None:
    """Lower-left corner of the image: the full field of view and the scanned region in it."""
    size = max(48.0, min(110.0, 0.22 * min(image_rect.width(), image_rect.height())))
    box = QtCore.QRectF(image_rect.left() + 8, image_rect.bottom() - 8 - size, size, size)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    p.fillRect(box, QtGui.QColor(0, 0, 0, 150))
    p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200, 220), 1))
    p.drawRect(box)
    center, w, h, rot = scan
    pts = scan_corners(center, w, h, rot)
    poly = QtGui.QPolygonF([QtCore.QPointF(box.left() + (x / field_um + 0.5) * size,
                                           box.top() + (0.5 - y / field_um) * size) for x, y in pts])
    p.setBrush(QtGui.QColor(255, 210, 0, 70))
    p.setPen(QtGui.QPen(QtGui.QColor(255, 210, 0), 1.5))
    p.drawPolygon(poly)
    p.setBrush(QtCore.Qt.NoBrush)
    p.setRenderHint(QtGui.QPainter.Antialiasing, False)


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
    zoomSteps = QtCore.Signal(int)  # wheel notches: + zoom in, − zoom out (the scan range, not the column)
    pointerMoved = QtCore.Signal(object)  # (x_um, y_um) or None
    measureChanged = QtCore.Signal(object)  # the measuring line ((x0, y0), (x1, y1)) in field µm, or None
    pointPicked = QtCore.Signal(float, float)  # pick mode: the clicked field position (µm)

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
        self.show_fov_inset = True
        self._measure_mode = False
        self._line = None  # measuring line: [(x0, y0), (x1, y1)] in field µm
        self._drawing = False
        self._drag = None
        self._paint_ms = 0.0
        self._pick_prompt: str | None = None  # pick mode on when not None
        self._pick_targets: list = []  # crosses (field µm)

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

    # ------------------------------------------------------------- measure
    @property
    def measure_mode(self) -> bool:
        return self._measure_mode

    @measure_mode.setter
    def measure_mode(self, on: bool) -> None:
        """On: left drag draws the measuring line (no beam shift).  Off: the line is removed."""
        self._measure_mode = bool(on)
        if on:
            self.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.unsetCursor()
            self._drawing = False
            self.clear_measure()

    def measure_line(self):
        """((x0, y0), (x1, y1)) in field µm, or None."""
        return None if self._line is None else (tuple(self._line[0]), tuple(self._line[1]))

    def measure_length_um(self) -> float | None:
        if self._line is None:
            return None
        (x0, y0), (x1, y1) = self._line
        return math.hypot(x1 - x0, y1 - y0)

    def set_measure_line(self, a, b) -> None:
        self._line = [tuple(map(float, a)), tuple(map(float, b))]
        self.update()
        self.measureChanged.emit(self.measure_line())

    def clear_measure(self) -> None:
        if self._line is not None:
            self._line = None
            self.update()
            self.measureChanged.emit(None)

    def scale_measure(self, r: float) -> None:
        """All field µm changed by *r* (a new field calibration): keep the line on the same feature."""
        if self._line is not None:
            self.set_measure_line(*[(x * r, y * r) for x, y in self._line])

    # ---------------------------------------------------------------- pick
    @property
    def pick_mode(self) -> bool:
        return self._pick_prompt is not None

    def start_pick(self, prompt: str, targets=()) -> None:
        """Pick mode: the next left click emits :attr:`pointPicked` (beam shift by drag is off)."""
        self._pick_prompt = str(prompt)
        self._pick_targets = [tuple(map(float, t)) for t in targets]
        self.setCursor(QtCore.Qt.CrossCursor)
        self.update()

    def stop_pick(self) -> None:
        self._pick_prompt = None
        self._pick_targets = []
        if not self._measure_mode:
            self.unsetCursor()
        self.update()

    def _widget_point(self, x: float, y: float) -> QtCore.QPointF:
        cfg, r = self._frame.config, self.image_rect()
        col, row = _field_to_pixel(cfg, x, y)
        return QtCore.QPointF(r.left() + (col + 0.5) / cfg.nx * r.width(),
                              r.top() + (row + 0.5) / cfg.ny * r.height())

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
                # solid, with a dark outline so it reads on bright and dark specimens alike
                c = r.center()
                arm = max(40.0, 0.08 * min(r.width(), r.height()))
                lines = (QtCore.QLineF(c.x() - arm, c.y(), c.x() + arm, c.y()),
                         QtCore.QLineF(c.x(), c.y() - arm, c.x(), c.y() + arm))
                for pen in (QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 4),
                            QtGui.QPen(QtGui.QColor(0, 230, 255, 235), 2)):
                    pen.setCapStyle(QtCore.Qt.FlatCap)
                    p.setPen(pen)
                    p.drawLines(lines)
            if self._line is not None and f is not None:
                p.save()
                p.setClipRect(r)
                _draw_measure(p, self._widget_point(*self._line[0]), self._widget_point(*self._line[1]),
                              format_length(self.measure_length_um(), 4))
                p.restore()
            field, scan = self.info.get("field_um"), self.info.get("scan")
            if self.show_fov_inset and field and scan is not None:
                _draw_fov_inset(p, r, float(field), scan)
            if self._pick_targets and f is not None:
                p.save()
                p.setClipRect(r)
                for t in self._pick_targets:
                    c = self._widget_point(*t)
                    for pen in (QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 4), QtGui.QPen(QtGui.QColor(255, 80, 200), 2)):
                        p.setPen(pen)
                        p.drawLine(QtCore.QPointF(c.x() - 14, c.y()), QtCore.QPointF(c.x() - 4, c.y()))
                        p.drawLine(QtCore.QPointF(c.x() + 4, c.y()), QtCore.QPointF(c.x() + 14, c.y()))
                        p.drawLine(QtCore.QPointF(c.x(), c.y() - 14), QtCore.QPointF(c.x(), c.y() - 4))
                        p.drawLine(QtCore.QPointF(c.x(), c.y() + 4), QtCore.QPointF(c.x(), c.y() + 14))
                p.restore()
        else:
            p.setPen(QtGui.QColor(120, 120, 120))
            p.drawText(self.rect().adjusted(0, 0, 0, -BAR_H), QtCore.Qt.AlignCenter, "No image – press Run")
        if self._pick_prompt is not None:
            font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold)
            p.setFont(font)
            box = QtCore.QRectF(8, 8, self.width() - 16, QtGui.QFontMetrics(font).height() * 2 + 12)
            p.fillRect(box, QtGui.QColor(120, 20, 90, 215))
            p.setPen(QtGui.QColor(255, 255, 255))
            p.drawText(box.adjusted(8, 4, -8, -4), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap,
                       self._pick_prompt)
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
        if px is not None and self._frame is not None and not self._measure_mode and not self.pick_mode:
            self.centerRequested.emit(*_pixel_to_field(self._frame.config, *px))

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() != QtCore.Qt.LeftButton or self._frame is None:
            return
        if self._pick_prompt is not None:
            px = self._to_pixel(ev.position())
            if px is not None:
                self.pointPicked.emit(*_pixel_to_field(self._frame.config, *px))
            return
        if self._measure_mode:
            px = self._to_pixel(ev.position())
            if px is not None:
                xy = _pixel_to_field(self._frame.config, *px)
                self._drawing = True
                self.set_measure_line(xy, xy)
            return
        self._drag = ev.position()
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):  # noqa: N802
        f = self._frame
        if self._drawing and f is not None:
            r = self.image_rect()
            pos = QtCore.QPointF(min(max(ev.position().x(), r.left()), r.right() - 1e-6),
                                 min(max(ev.position().y(), r.top()), r.bottom() - 1e-6))
            px = self._to_pixel(pos)
            if px is not None:
                self.set_measure_line(self._line[0], _pixel_to_field(f.config, *px))
        elif self._drag is not None and f is not None:
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
        if self._drawing:
            self._drawing = False
            if (self.measure_length_um() or 0.0) <= 0.0:
                self.clear_measure()  # a click without a drag
            return
        self._drag = None
        if self._measure_mode:
            self.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.unsetCursor()

    def wheelEvent(self, ev):  # noqa: N802
        steps = int(round(ev.angleDelta().y() / 120))
        if steps:
            self.zoomSteps.emit(steps)

    def leaveEvent(self, ev):  # noqa: N802
        self.pointerMoved.emit(None)
