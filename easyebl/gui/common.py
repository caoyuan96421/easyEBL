"""Shared helpers for the debug/demo GUIs (PySide6 + pyqtgraph).

Every subsystem GUI is a ``QWidget`` that takes a :class:`Session`, so the
same widgets can run standalone (``python -m easyebl.imaging.gui``) or as tabs
of the launcher, where they share one DAQ manager – demonstrating the HAL's
handle sharing.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
import typing
from typing import Any, Callable

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as e:  # pragma: no cover
    raise ImportError("GUI requires PySide6 and pyqtgraph: pip install easyebl[gui]") from e

from ..core.calibration import FieldCalibration
from ..hal import DAQConfig, DAQManager, SimParams, SimulatedBackend, create_backend

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "pg",
    "Session",
    "ensure_app",
    "DataclassForm",
    "Worker",
    "ImageView",
    "WaveformView",
    "LogBox",
    "parse_points",
    "format_points",
    "run_standalone",
    "standalone_parser",
]

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


def ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    return app


# --------------------------------------------------------------------- session
class Session:
    """What a GUI needs: a shared DAQ manager and the current field calibration."""

    def __init__(self, manager: DAQManager, calibration: FieldCalibration | None = None):
        self.manager = manager
        self.calibration = calibration or FieldCalibration(100.0, 10.0)

    @classmethod
    def simulated(cls, sample=None, params: SimParams | None = None, config: DAQConfig | None = None):
        be = SimulatedBackend(config or DAQConfig(), params or SimParams(), sample)
        return cls(DAQManager(be).open())

    @classmethod
    def from_args(cls, args) -> "Session":
        preset = getattr(args, "preset", None)
        if args.config:
            cfg = DAQConfig.from_json(open(args.config).read())
        elif preset == "usb6363":
            cfg = DAQConfig.usb6363()
        else:
            cfg = DAQConfig()
        if args.device:
            cfg.device = args.device
        field = getattr(args, "field_um", None) or 100.0
        cal = FieldCalibration(field, cfg.ao_limit_v)
        if args.backend == "sim":
            s = cls.simulated(config=cfg, params=SimParams(field_size_um=field, v_full_scale=cfg.ao_limit_v))
            s.calibration = cal
            return s
        return cls(DAQManager(create_backend(args.backend, cfg)).open(), cal)

    @property
    def simulator(self) -> SimulatedBackend | None:
        be = self.manager.backend
        return be if isinstance(be, SimulatedBackend) else None

    def close(self) -> None:
        self.manager.close()


# ------------------------------------------------------------------ dataclass form
def _unwrap_optional(tp):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


class DataclassForm(QtWidgets.QWidget):
    """Auto-generated editor for a dataclass instance.

    ``overrides[name]`` may contain ``min``, ``max``, ``decimals``, ``step``,
    ``scale`` (display = value·scale), ``suffix``, ``choices`` and ``label``.
    Fields listed in ``hidden`` are not shown (their values are preserved).
    """

    changed = QtCore.Signal()

    def __init__(self, instance, overrides: dict | None = None, hidden=(), parent=None):
        super().__init__(parent)
        self._cls = type(instance)
        self._base = instance
        self._ov = overrides or {}
        self._widgets: dict[str, tuple] = {}
        hints = typing.get_type_hints(self._cls)
        lay = QtWidgets.QFormLayout(self)
        lay.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        for f in dataclasses.fields(self._cls):
            if f.name in hidden or not f.init:
                continue
            w = self._make(f.name, hints.get(f.name, type(getattr(instance, f.name))), getattr(instance, f.name))
            if w is None:
                continue
            lay.addRow(self._ov.get(f.name, {}).get("label", f.name), w[0])
            self._widgets[f.name] = w

    def _make(self, name, tp, value):
        ov = self._ov.get(name, {})
        tp, optional = _unwrap_optional(tp)
        if "choices" in ov:
            w = QtWidgets.QComboBox()
            for c in ov["choices"]:
                w.addItem(str(c), c)
            idx = w.findData(value)
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(self.changed)
            return (w, "choice", optional)
        if tp is bool:
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(self.changed)
            return (w, "bool", optional)
        if tp is int and not optional:
            w = QtWidgets.QSpinBox()
            w.setRange(int(ov.get("min", -(2**31) + 1)), int(ov.get("max", 2**31 - 1)))
            w.setValue(int(value))
            if "suffix" in ov:
                w.setSuffix(ov["suffix"])
            w.valueChanged.connect(self.changed)
            return (w, "int", optional)
        if tp is float and not optional:
            scale = ov.get("scale", 1.0)
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(int(ov.get("decimals", 4)))
            w.setRange(float(ov.get("min", -1e12)), float(ov.get("max", 1e12)))
            w.setSingleStep(float(ov.get("step", 1.0)))
            if "suffix" in ov:
                w.setSuffix(ov["suffix"])
            w.setValue(float(value) * scale)
            w.valueChanged.connect(self.changed)
            return (w, ("float", scale), optional)
        # generic: text (tuples, optional numbers, strings)
        w = QtWidgets.QLineEdit("" if value is None else self._fmt(value))
        w.editingFinished.connect(self.changed)
        return (w, ("text", tp), optional)

    @staticmethod
    def _fmt(v) -> str:
        if isinstance(v, (tuple, list)):
            return ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v)
        return f"{v:g}" if isinstance(v, float) else str(v)

    def value(self):
        kw = {}
        for name, (w, kind, optional) in self._widgets.items():
            if kind == "choice":
                v = w.currentData()
            elif kind == "bool":
                v = w.isChecked()
            elif kind == "int":
                v = w.value()
            elif isinstance(kind, tuple) and kind[0] == "float":
                v = w.value() / kind[1]
            else:
                tp = kind[1]
                txt = w.text().strip()
                if txt == "" or txt.lower() == "none":
                    v = None if optional else getattr(self._base, name)
                elif tp in (tuple, list) or typing.get_origin(tp) in (tuple, list) or isinstance(
                    getattr(self._base, name), (tuple, list)
                ):
                    parts = [p.strip() for p in txt.split(",") if p.strip()]
                    vals = []
                    for p in parts:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            vals.append(p)
                    v = tuple(vals)
                elif tp is float:
                    v = float(txt)
                elif tp is int:
                    v = int(txt)
                else:
                    v = txt
            kw[name] = v
        return dataclasses.replace(self._base, **kw)

    def set_value(self, instance) -> None:
        self._base = instance
        for name, (w, kind, optional) in self._widgets.items():
            v = getattr(instance, name)
            w.blockSignals(True)
            if kind == "choice":
                w.setCurrentIndex(max(0, w.findData(v)))
            elif kind == "bool":
                w.setChecked(bool(v))
            elif kind == "int":
                w.setValue(int(v))
            elif isinstance(kind, tuple) and kind[0] == "float":
                w.setValue(float(v) * kind[1])
            else:
                w.setText("" if v is None else self._fmt(v))
            w.blockSignals(False)
        self.changed.emit()


# --------------------------------------------------------------------- worker
class Worker(QtCore.QThread):
    """Run ``fn(*args, **kw)`` in a thread; results delivered via signals."""

    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(object)

    def __init__(self, fn: Callable, *args, **kw):
        super().__init__()
        self._fn, self._args, self._kw = fn, args, kw

    def run(self):
        try:
            self.result.emit(self._fn(*self._args, **self._kw))
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


# ----------------------------------------------------------------- image view
class ImageView(pg.PlotWidget):
    """Image in field coordinates (µm) with overlay helpers."""

    def __init__(self, parent=None, cmap: str = "gray"):
        super().__init__(parent)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel("bottom", "x", "µm")
        self.setLabel("left", "y", "µm")
        self.img = pg.ImageItem()
        self.addItem(self.img)
        try:
            self.img.setColorMap(pg.colormap.get(cmap, source="matplotlib"))
        except Exception:  # pragma: no cover - colormap availability
            pass
        self._overlays: list = []
        self._extent = None

    def set_image(self, data: np.ndarray, extent: tuple, levels=None) -> None:
        """``extent`` = (x0, x1, y0, y1); ``data[0]`` is the top row."""
        d = np.nan_to_num(np.asarray(data, float), nan=float(np.nanmin(data)) if np.any(np.isfinite(data)) else 0.0)
        x0, x1, y0, y1 = extent
        self.img.setImage(d[::-1], autoLevels=levels is None, levels=levels)
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        if self._extent != tuple(extent):
            self._extent = tuple(extent)
            self.getPlotItem().vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.02)

    def clear_overlays(self) -> None:
        for it in self._overlays:
            self.removeItem(it)
        self._overlays = []

    def add_points(self, xy, color="r", size=8, symbol="+") -> None:
        xy = np.atleast_2d(xy)
        it = pg.ScatterPlotItem(xy[:, 0], xy[:, 1], pen=pg.mkPen(color), brush=pg.mkBrush(color), size=size, symbol=symbol)
        self.addItem(it)
        self._overlays.append(it)

    def add_path(self, xy, color="y", width=1, style=None, closed=False) -> None:
        xy = np.atleast_2d(xy)
        if closed and len(xy):
            xy = np.vstack([xy, xy[:1]])
        pen = pg.mkPen(color, width=width)
        if style == "dash":
            pen.setStyle(QtCore.Qt.DashLine)
        it = pg.PlotDataItem(xy[:, 0], xy[:, 1], pen=pen)
        self.addItem(it)
        self._overlays.append(it)


class WaveformView(pg.GraphicsLayoutWidget):
    """X/Y/blank (and optional AI) traces versus time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p_xy = self.addPlot(row=0, col=0)
        self.p_xy.setLabel("left", "deflection", "V")
        self.p_xy.addLegend()
        self.p_bl = self.addPlot(row=1, col=0)
        self.p_bl.setLabel("left", "blank")
        self.p_bl.setXLink(self.p_xy)
        self.p_ai = self.addPlot(row=2, col=0)
        self.p_ai.setLabel("left", "AI", "V")
        self.p_ai.setLabel("bottom", "time", "s")
        self.p_ai.setXLink(self.p_xy)
        self.p_bl.setMaximumHeight(120)

    def show_chunk(self, chunk, rate: float, ai: np.ndarray | None = None, max_points: int = 200_000):
        n = len(chunk)
        step = max(1, n // max_points)
        t = np.arange(0, n, step) / rate
        self.p_xy.clear()
        self.p_xy.addLegend()
        self.p_xy.plot(t, chunk.x[::step], pen="c", name="X")
        self.p_xy.plot(t, chunk.y[::step], pen="m", name="Y")
        self.p_bl.clear()
        self.p_bl.plot(t, chunk.blank[::step].astype(float), pen="y", stepMode=None)
        for i, (k, v) in enumerate(chunk.markers.items()):
            self.p_bl.plot(t, v[::step].astype(float) * 0.8 + 1.2 * (i + 1), pen=pg.intColor(i))
        self.p_ai.clear()
        if ai is not None and ai.size:
            for i, row in enumerate(np.atleast_2d(ai)):
                m = min(len(row), n)
                self.p_ai.plot(t[: (m + step - 1) // step], row[:m:step], pen=pg.intColor(i + 3))


class LogBox(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))

    def log(self, msg: str) -> None:
        self.appendPlainText(msg)


# ---------------------------------------------------------------- points text
def parse_points(text: str) -> np.ndarray:
    """Parse ``"x, y"`` pairs separated by newlines or ';'."""
    pts = []
    for part in text.replace(";", "\n").splitlines():
        part = part.strip()
        if not part or part.startswith("#"):
            continue
        vals = [float(v) for v in part.replace(",", " ").split()]
        if len(vals) != 2:
            raise ValueError(f"cannot parse point '{part}'")
        pts.append(vals)
    return np.array(pts, float).reshape(-1, 2)


def format_points(pts) -> str:
    return "\n".join(f"{x:g}, {y:g}" for x, y in np.atleast_2d(pts))


# ----------------------------------------------------------------- standalone
def standalone_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--backend", default="sim", choices=["sim", "nidaqmx"], help="hardware backend")
    p.add_argument("--device", default=None, help="NI device name (e.g. Dev1)")
    p.add_argument("--config", default=None, help="DAQConfig JSON file")
    p.add_argument("--preset", default=None, choices=["usb6363"], help="built-in wiring preset")
    p.add_argument("--field-um", dest="field_um", type=float, default=None,
                   help="field size (µm) that the full AO range maps to (default 100)")
    return p


def run_standalone(widget_factory: Callable[[Session], QtWidgets.QWidget], title: str, argv=None) -> int:
    args = standalone_parser(title).parse_args(argv)
    app = ensure_app()
    session = Session.from_args(args)
    w = widget_factory(session)
    w.setWindowTitle(f"easyEBL – {title} [{session.manager.capabilities().name}]")
    w.resize(1300, 850)
    w.show()
    try:
        return app.exec()
    finally:
        session.close()
