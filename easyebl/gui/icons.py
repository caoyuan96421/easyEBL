"""Toolbar icons, drawn in code (no image files): ``icon(name)`` → QIcon.

Simple line drawings on a 32 × 32 grid in a light colour that reads on the dark theme; every
toolbar of the main window (SEM, Lithography, the small SEM view, the EDS image tools) uses them,
with the text under the icon (:func:`tool_bar`).
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = ["icon", "NAMES", "tool_bar"]

FG = QtGui.QColor(225, 230, 238)
ACCENT = QtGui.QColor(90, 190, 255)
GREEN = QtGui.QColor(90, 210, 110)
RED = QtGui.QColor(235, 85, 85)
YELLOW = QtGui.QColor(255, 210, 70)

_cache: dict = {}


def _pen(p: QtGui.QPainter, color=FG, w: float = 2.2) -> None:
    pen = QtGui.QPen(color, w)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)


def _poly(pts) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts])


def _arrow(p, x0, y0, x1, y1, head=5.0):
    p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
    a = math.atan2(y1 - y0, x1 - x0)
    for s in (+1, -1):
        p.drawLine(QtCore.QPointF(x1, y1),
                   QtCore.QPointF(x1 - head * math.cos(a + s * 0.5), y1 - head * math.sin(a + s * 0.5)))


def _lens(p, x=16, y=14, r=8.0):
    p.drawEllipse(QtCore.QPointF(x, y), r, r)
    p.drawLine(QtCore.QPointF(x + r * 0.7, y + r * 0.7), QtCore.QPointF(x + r * 1.5, y + r * 1.5))


def _frame(p, x0=5, y0=6, x1=27, y1=26):
    p.drawRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))


# ----------------------------------------------------------------------- drawings
def _run(p):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(GREEN)
    p.drawPolygon(_poly([(9, 6), (26, 16), (9, 26)]))


def _stop(p):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(RED)
    p.drawRoundedRect(QtCore.QRectF(8, 8, 16, 16), 2, 2)


def _single(p):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(GREEN)
    p.drawPolygon(_poly([(6, 7), (19, 16), (6, 25)]))
    p.setBrush(FG)
    p.drawRect(QtCore.QRectF(21, 7, 4, 18))


def _capture(p):
    _pen(p)
    p.drawRoundedRect(QtCore.QRectF(4, 10, 24, 16), 3, 3)
    p.drawRect(QtCore.QRectF(11, 6, 10, 4))
    p.drawEllipse(QtCore.QPointF(16, 18), 5, 5)


def _record(p):
    """A red dot in a ring (start recording)."""
    _pen(p, FG, 2.0)
    p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(RED)
    p.drawEllipse(QtCore.QPointF(16, 16), 7, 7)


def _record_stop(p):
    """A red square in a red ring (recording: stop it)."""
    _pen(p, RED, 2.0)
    p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(RED)
    p.drawRect(QtCore.QRectF(11, 11, 10, 10))


def _full_fov(p):
    _pen(p)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        cx, cy = 16 + sx * 11, 16 + sy * 11
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx - sx * 6, cy))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx, cy - sy * 6))
    _pen(p, ACCENT)
    p.drawRect(QtCore.QRectF(11, 11, 10, 10))


def _focus(p):
    _pen(p)
    p.drawPolygon(_poly([(4, 6), (28, 6), (17, 16), (28, 26), (4, 26), (15, 16)]))
    p.setBrush(YELLOW)
    p.setPen(QtCore.Qt.NoPen)
    p.drawEllipse(QtCore.QPointF(16, 16), 2.5, 2.5)


def _stig(p):
    _pen(p)
    p.drawEllipse(QtCore.QRectF(4, 10, 24, 12))
    _pen(p, ACCENT)
    p.drawEllipse(QtCore.QRectF(10, 4, 12, 24))
    p.setBrush(YELLOW)
    p.setPen(QtCore.Qt.NoPen)
    p.drawEllipse(QtCore.QPointF(16, 16), 2.5, 2.5)


def _gun(p):
    _pen(p)
    p.drawLine(QtCore.QPointF(8, 4), QtCore.QPointF(24, 4))
    p.drawPolygon(_poly([(10, 5), (22, 5), (16, 13)]))
    _pen(p, ACCENT, 2.6)
    p.drawLine(QtCore.QPointF(16, 13), QtCore.QPointF(16, 27))
    _pen(p, FG, 1.6)
    p.drawEllipse(QtCore.QPointF(16, 21), 6, 2.5)


def _acb(p):
    _pen(p)
    p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(FG)
    path = QtGui.QPainterPath()
    path.moveTo(16, 5)
    path.arcTo(QtCore.QRectF(5, 5, 22, 22), 90, -180)
    path.closeSubpath()
    p.drawPath(path)
    f = p.font()
    f.setPixelSize(9)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QtGui.QColor(20, 20, 24))
    p.drawText(QtCore.QRectF(16, 5, 11, 22), QtCore.Qt.AlignCenter, "A")


def _crosshair(p):
    _pen(p)
    p.drawLine(QtCore.QPointF(16, 3), QtCore.QPointF(16, 29))
    p.drawLine(QtCore.QPointF(3, 16), QtCore.QPointF(29, 16))
    p.drawEllipse(QtCore.QPointF(16, 16), 6, 6)


def _scan_line(p):
    _pen(p, FG, 1.4)
    _frame(p)
    _pen(p, GREEN, 2.4)
    p.drawLine(QtCore.QPointF(5, 18), QtCore.QPointF(27, 18))


def _inset(p):
    _pen(p, FG, 1.6)
    _frame(p)
    _pen(p, ACCENT, 1.6)
    p.drawRect(QtCore.QRectF(7, 17, 9, 7))
    _pen(p, RED, 1.6)
    p.drawRect(QtCore.QRectF(10, 19, 3, 3))


def _advanced(p):
    _pen(p)
    p.drawEllipse(QtCore.QPointF(16, 16), 5, 5)
    for k in range(8):
        a = k * math.pi / 4
        p.drawLine(QtCore.QPointF(16 + 8 * math.cos(a), 16 + 8 * math.sin(a)),
                   QtCore.QPointF(16 + 12 * math.cos(a), 16 + 12 * math.sin(a)))


def _full_screen(p):
    _pen(p)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        _arrow(p, 16 + sx * 3, 16 + sy * 3, 16 + sx * 11, 16 + sy * 11, 5)


def _column(p):
    _pen(p)
    p.drawRect(QtCore.QRectF(11, 3, 10, 6))
    p.drawPolygon(_poly([(9, 9), (23, 9), (20, 20), (12, 20)]))
    _pen(p, ACCENT, 2.4)
    p.drawLine(QtCore.QPointF(16, 20), QtCore.QPointF(16, 26))
    _pen(p, FG, 2.0)
    p.drawLine(QtCore.QPointF(6, 28), QtCore.QPointF(26, 28))


def _pan(p):
    _pen(p)
    _arrow(p, 16, 16, 16, 4)
    _arrow(p, 16, 16, 16, 28)
    _arrow(p, 16, 16, 4, 16)
    _arrow(p, 16, 16, 28, 16)


def _point(p):
    _pen(p)
    p.drawLine(QtCore.QPointF(16, 4), QtCore.QPointF(16, 11))
    p.drawLine(QtCore.QPointF(16, 21), QtCore.QPointF(16, 28))
    p.drawLine(QtCore.QPointF(4, 16), QtCore.QPointF(11, 16))
    p.drawLine(QtCore.QPointF(21, 16), QtCore.QPointF(28, 16))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(YELLOW)
    p.drawEllipse(QtCore.QPointF(16, 16), 3, 3)


def _rect(p):
    _pen(p, YELLOW)
    p.drawRect(QtCore.QRectF(5, 8, 22, 16))


def _ellipse(p):
    _pen(p, YELLOW)
    p.drawEllipse(QtCore.QRectF(4, 8, 24, 16))


def _line(p):
    _pen(p, YELLOW, 2.4)
    p.drawLine(QtCore.QPointF(5, 26), QtCore.QPointF(27, 6))
    p.setBrush(FG)
    p.setPen(QtCore.Qt.NoPen)
    p.drawEllipse(QtCore.QPointF(5, 26), 2.5, 2.5)
    p.drawEllipse(QtCore.QPointF(27, 6), 2.5, 2.5)


def _map(p):
    p.setPen(QtCore.Qt.NoPen)
    cols = (QtGui.QColor(40, 60, 140), QtGui.QColor(60, 160, 200), QtGui.QColor(240, 200, 60), QtGui.QColor(220, 80, 60))
    for i in range(4):
        for j in range(4):
            p.setBrush(cols[(i * 3 + j * 2) % 4])
            p.drawRect(QtCore.QRectF(5 + 5.5 * i, 5 + 5.5 * j, 5.5, 5.5))
    _pen(p, FG, 1.4)
    p.drawRect(QtCore.QRectF(5, 5, 22, 22))


def _zoom_in(p):
    _pen(p)
    _lens(p, 13, 13, 8)
    p.drawLine(QtCore.QPointF(9, 13), QtCore.QPointF(17, 13))
    p.drawLine(QtCore.QPointF(13, 9), QtCore.QPointF(13, 17))


def _zoom_out(p):
    _pen(p)
    _lens(p, 13, 13, 8)
    p.drawLine(QtCore.QPointF(9, 13), QtCore.QPointF(17, 13))


def _zoom_pattern(p):
    _pen(p, ACCENT, 1.8)
    p.drawRect(QtCore.QRectF(6, 6, 7, 7))
    p.drawRect(QtCore.QRectF(15, 9, 5, 9))
    _pen(p)
    _lens(p, 13, 13, 10)


def _move_pattern(p):
    _pen(p, ACCENT, 1.8)
    p.drawRect(QtCore.QRectF(4, 4, 11, 11))
    p.drawRect(QtCore.QRectF(7, 7, 5, 5))
    _pen(p)
    _arrow(p, 13, 13, 27, 27, 6)


def _move_stage(p):
    _pen(p)
    p.drawEllipse(QtCore.QPointF(16, 16), 4, 4)
    _arrow(p, 16, 11, 16, 3)
    _arrow(p, 16, 21, 16, 29)
    _arrow(p, 11, 16, 3, 16)
    _arrow(p, 21, 16, 29, 16)


def _color_field(p):
    p.setPen(QtCore.Qt.NoPen)
    for (x, y), c in zip(((4, 4), (17, 4), (4, 17), (17, 17)),
                         (QtGui.QColor(90, 190, 255), QtGui.QColor(255, 150, 60), QtGui.QColor(120, 220, 110),
                          QtGui.QColor(230, 110, 220))):
        p.setBrush(c)
        p.drawRect(QtCore.QRectF(x, y, 11, 11))


def _color_dose(p):
    g = QtGui.QLinearGradient(4, 0, 28, 0)
    g.setColorAt(0.0, QtGui.QColor(50, 40, 140))
    g.setColorAt(0.5, QtGui.QColor(40, 180, 160))
    g.setColorAt(1.0, QtGui.QColor(250, 220, 50))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QBrush(g))
    p.drawRect(QtCore.QRectF(4, 10, 24, 12))
    _pen(p, FG, 1.2)
    p.drawRect(QtCore.QRectF(4, 10, 24, 12))


def _order(p):
    _pen(p, FG, 1.6)
    pts = ((6, 7), (26, 7), (6, 25), (26, 25))
    for a, b in zip(pts, pts[1:]):
        _arrow(p, *a, *b, 4)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(YELLOW)
    p.drawEllipse(QtCore.QPointF(6, 7), 2.5, 2.5)


def _dots(p):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(ACCENT)
    for i in range(5):
        for j in range(5):
            p.drawEllipse(QtCore.QPointF(6 + 5 * i, 6 + 5 * j), 1.4, 1.4)


_DRAW = {
    "run": _run, "stop": _stop, "single": _single, "capture": _capture, "record": _record,
    "record_stop": _record_stop, "full_fov": _full_fov,
    "focus": _focus, "stig": _stig, "gun": _gun, "acb": _acb, "crosshair": _crosshair,
    "scan_line": _scan_line, "inset": _inset, "advanced": _advanced, "full_screen": _full_screen, "column": _column,
    "pan": _pan, "point": _point, "rect": _rect, "ellipse": _ellipse, "line": _line, "map": _map,
    "zoom_in": _zoom_in, "zoom_out": _zoom_out, "zoom_pattern": _zoom_pattern,
    "move_pattern": _move_pattern, "move_stage": _move_stage, "color_field": _color_field,
    "color_dose": _color_dose, "order": _order, "dots": _dots,
}
NAMES = tuple(_DRAW)


def icon(name: str) -> QtGui.QIcon:
    """The icon *name* (one of :data:`NAMES`)."""
    ic = _cache.get(name)
    if ic is not None:
        return ic
    draw = _DRAW[name]
    ic = QtGui.QIcon()
    for size in (16, 24, 32, 48):
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(size / 32.0, size / 32.0)
        draw(p)
        p.end()
        ic.addPixmap(pm)
    _cache[name] = ic
    return ic


def tool_bar(name: str, parent=None, size: int = 22) -> QtWidgets.QToolBar:
    """A toolbar with the text under each icon, not movable."""
    tb = QtWidgets.QToolBar(name, parent)
    tb.setObjectName(name)
    tb.setMovable(False)
    tb.setFloatable(False)
    tb.setIconSize(QtCore.QSize(size, size))
    tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
    return tb
