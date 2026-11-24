"""The *Wiring* tab of the hardware dialog: which DAQ signal is on which USB-6363 channel.

Every dropdown lists only channels the NI USB-6363 has (AI in the selected input
mode).  Conflicts – two signals on one output, an AI pin used twice, a static
line on the hardware-timed port – are listed under the form and the widgets
concerned turn red; the dialog cannot be applied while there are any.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..hal.wiring import USB6363, Wiring, WiringIssue, canonical_line, pin_key
from .config import AI_RANGES_V, AI_TERMINALS

__all__ = ["WiringEditor"]

_BAD = "QComboBox, QLineEdit { border: 1px solid #f55; }"


def _ao_choices():
    return [(c, c) for c in USB6363.ao]


def _ai_choices(terminal: str):
    diff = str(terminal).upper() in ("DIFF", "PSEUDO_DIFF")
    out = []
    for c in USB6363.ai_for(terminal):
        n = int(c[2:])
        out.append((f"{c}  (AI{n}+ / AI{n + 8}−)" if diff else c, c))
    return out


def _port0_choices():
    return [(f"{c}  (P0.{c.split('line')[1]})", c) for c in USB6363.timed_lines]


def _pfi_choices():
    out = []
    for c in USB6363.pfi:
        n = int(c[3:])
        out.append((f"{c}  (P{1 + n // 8}.{n % 8})", c))
    return out


def _fill(combo: QtWidgets.QComboBox, choices, value, none_label: str | None = None):
    """Replace the items; keep *value* selected (an unknown value is shown, marked, so it is flagged)."""
    combo.blockSignals(True)
    combo.clear()
    if none_label is not None:
        combo.addItem(none_label, None)
    for label, data in choices:
        combo.addItem(label, data)
    i = combo.findData(value) if value is not None else (0 if none_label is not None else -1)
    if i < 0 and value is not None:
        combo.addItem(f"{value}  (not available)", value)
        i = combo.count() - 1
    combo.setCurrentIndex(max(0, i))
    combo.blockSignals(False)


def _combo(choices, value, none_label=None, on_change=None) -> QtWidgets.QComboBox:
    c = QtWidgets.QComboBox()
    _fill(c, choices, value, none_label)
    if on_change is not None:
        c.currentIndexChanged.connect(on_change)
    return c


class _Rows(QtWidgets.QWidget):
    """An editable list of signals: [name] channel [polarity] ✕, plus an Add button."""

    changed = QtCore.Signal()

    def __init__(self, choices, named: bool, polarity: str | None, add_text: str, default_name: str = "line"):
        super().__init__()
        self._choices, self._named, self._polarity, self._default = choices, named, polarity, default_name
        self.rows: list[dict] = []
        self.used_pins = None  # callable: pins taken by every other signal (a new row avoids them)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self._box = QtWidgets.QVBoxLayout()
        v.addLayout(self._box)
        self.add_button = QtWidgets.QPushButton(add_text)
        self.add_button.clicked.connect(lambda: self.add_row())
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.add_button)
        h.addStretch(1)
        v.addLayout(h)

    def set_choices(self, choices):
        self._choices = choices
        for r in self.rows:
            _fill(r["combo"], choices, r["combo"].currentData())

    def _free_channel(self):
        used = {pin_key(r["combo"].currentData() or "") for r in self.rows}
        if self.used_pins is not None:
            used |= set(self.used_pins())
        return next((d for _, d in self._choices if pin_key(d) not in used), self._choices[0][1])

    def add_row(self, name: str | None = None, channel: str | None = None, active: bool = True, emit: bool = True):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        r = {"widget": w, "name": None, "pol": None}
        if self._named:
            if name is None:
                taken = {x["name"].text() for x in self.rows}
                name = next(f"{self._default}{i}" for i in range(1, 1000) if f"{self._default}{i}" not in taken)
            r["name"] = QtWidgets.QLineEdit(str(name))
            r["name"].setPlaceholderText("name")
            r["name"].setMaximumWidth(140)
            r["name"].textChanged.connect(self.changed)
            h.addWidget(r["name"])
        r["combo"] = _combo(self._choices, channel if channel is not None else self._free_channel(),
                            on_change=self.changed)
        h.addWidget(r["combo"], 1)
        if self._polarity:
            r["pol"] = QtWidgets.QCheckBox(self._polarity)
            r["pol"].setChecked(bool(active))
            r["pol"].toggled.connect(self.changed)
            h.addWidget(r["pol"])
        rm = QtWidgets.QToolButton()
        rm.setText("✕")
        rm.setToolTip("remove")
        rm.clicked.connect(lambda: self.remove_row(r))
        h.addWidget(rm)
        self.rows.append(r)
        self._box.addWidget(w)
        if emit:
            self.changed.emit()
        return r

    def remove_row(self, r: dict):
        self.rows.remove(r)
        r["widget"].setParent(None)
        r["widget"].deleteLater()
        self.changed.emit()

    def set_items(self, items):
        for r in list(self.rows):
            r["widget"].setParent(None)
            r["widget"].deleteLater()
        self.rows.clear()
        for name, ch, active in items:
            self.add_row(name, ch, active, emit=False)
        self.changed.emit()

    def items(self):
        return [((r["name"].text().strip() if r["name"] else None), r["combo"].currentData(),
                 (r["pol"].isChecked() if r["pol"] else True)) for r in self.rows]


class WiringEditor(QtWidgets.QWidget):
    """Edit a :class:`Wiring` (and the AI input mode) with USB-6363 channel dropdowns."""

    changed = QtCore.Signal()

    def __init__(self, wiring: Wiring, ai_terminal: str | None = None, preset_terminal: str = "RSE", parent=None,
                 ai_range_v: float | None = None, preset_range_v: float = 10.0):
        super().__init__(parent)
        self._preset_terminal = preset_terminal
        self._preset_range = float(preset_range_v)
        self._loading = False
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        ch = self._emit

        # --- AO
        g = QtWidgets.QGroupBox("Scan outputs (AO, ±10 V)")
        f = QtWidgets.QFormLayout(g)
        self.ao_x = _combo(_ao_choices(), None, on_change=ch)
        self.ao_y = _combo(_ao_choices(), None, on_change=ch)
        self.aux0 = _combo(_ao_choices(), None, "not connected (dummy)", ch)
        self.aux1 = _combo(_ao_choices(), None, "not connected (dummy)", ch)
        f.addRow("Scan X (DBC X-IN)", self.ao_x)
        f.addRow("Scan Y (DBC Y-IN)", self.ao_y)
        f.addRow("Aux output 0", self.aux0)
        f.addRow("Aux output 1", self.aux1)
        lay.addWidget(g)

        # --- port0
        g = QtWidgets.QGroupBox("Blanker and hardware-timed lines (port0, clocked with the scan)")
        f = QtWidgets.QFormLayout(g)
        self.blank = _combo(_port0_choices(), None, on_change=ch)
        self.blank_high = QtWidgets.QCheckBox("HIGH blanks the beam")
        self.blank_high.setToolTip("the S-3000 blanker input is active LOW (unchecked)")
        self.blank_high.toggled.connect(ch)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.blank, 1)
        row.addWidget(self.blank_high)
        f.addRow("Blanker", row)
        self.timed = _Rows(_port0_choices(), named=True, polarity=None, add_text="Add timed line",
                           default_name="marker")
        self.timed.changed.connect(ch)
        f.addRow("Marker lines", self.timed)
        lay.addWidget(g)

        # --- AI
        g = QtWidgets.QGroupBox("Video / SE inputs (AI)")
        f = QtWidgets.QFormLayout(g)
        self.ai_terminal = QtWidgets.QComboBox()
        self.ai_terminal.addItem("", None)
        for t in AI_TERMINALS:
            self.ai_terminal.addItem(t, t)
        self.ai_terminal.setToolTip("RSE: AI+ against AI GND.  DIFF: AI n+ / AI n+8− (both wired).")
        self.ai_terminal.currentIndexChanged.connect(self._terminal_changed)
        self.ai_range = QtWidgets.QComboBox()
        self.ai_range.addItem("", None)
        for v in AI_RANGES_V:
            self.ai_range.addItem(f"±{v:g} V", v)
        self.ai_range.setToolTip("Video input range: the smallest range that holds the signal uses the full "
                                 "16-bit resolution (used by SEM imaging and the EDS SE image).")
        self._set_range(ai_range_v)
        self.ai = _Rows(_ai_choices("RSE"), named=True, polarity=None, add_text="Add video input",
                        default_name="video")
        self.ai.changed.connect(self._ai_changed)
        self.eds_video = QtWidgets.QComboBox()
        self.eds_video.currentIndexChanged.connect(ch)
        f.addRow("Input mode", self.ai_terminal)
        f.addRow("Input range", self.ai_range)
        f.addRow("Inputs (first = SEM image)", self.ai)
        f.addRow("EDS SE input", self.eds_video)
        lay.addWidget(g)

        # --- static DIO / PFI
        g = QtWidgets.QGroupBox("Static lines (PFI0–15 = port1/port2)")
        f = QtWidgets.QFormLayout(g)
        self.ext = _Rows(_pfi_choices(), named=False, polarity="active HIGH", add_text="Add relay line")
        self.ext.changed.connect(ch)
        self.ext.setToolTip("the level 'active' switches the SEM to external scan (all lines together)")
        self.static = _Rows(_pfi_choices(), named=True, polarity="active HIGH", add_text="Add static line",
                            default_name="aux")
        self.static.changed.connect(ch)
        f.addRow("External scan control", self.ext)
        f.addRow("Other static lines", self.static)
        lay.addWidget(g)

        # --- clock / trigger exports
        g = QtWidgets.QGroupBox("Clock and trigger outputs (PFI)")
        f = QtWidgets.QFormLayout(g)
        self.clock_out = _combo(_pfi_choices(), None, "not exported", ch)
        self.clock_out.setToolTip("every scan stream puts its sample clock on this terminal")
        self.trigger_out = _combo(_pfi_choices(), None, "not exported", ch)
        self.trigger_out.setToolTip("a pulse on this terminal when a scan stream starts")
        f.addRow("Sample clock out", self.clock_out)
        f.addRow("Start trigger out", self.trigger_out)
        lay.addWidget(g)
        lay.addStretch(1)

        for rows in (self.timed, self.ai, self.ext, self.static):
            rows.used_pins = self._used_pins

        self.issues_label = QtWidgets.QLabel()
        self.issues_label.setWordWrap(True)
        self.issues_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        outer.addWidget(self.issues_label)
        self.set_wiring(wiring, ai_terminal)

    # ---------------------------------------------------------------- state
    def _set_range(self, v):
        v = float(v) if v else None
        self.ai_range.setItemText(0, f"preset default (±{self._preset_range:g} V)")
        i = self.ai_range.findData(v) if v is not None else 0
        if i < 0:  # a range set by hand in system.json
            self.ai_range.addItem(f"±{v:g} V", v)
            i = self.ai_range.count() - 1
        self.ai_range.setCurrentIndex(i)

    def selected_range(self) -> float | None:
        """The stored input range (None: the preset's)."""
        return self.ai_range.currentData()

    def set_preset_range(self, v: float):
        self._preset_range = float(v)
        self.ai_range.setItemText(0, f"preset default (±{self._preset_range:g} V)")

    def set_preset_terminal(self, terminal: str):
        self._preset_terminal = terminal or "RSE"
        self.ai_terminal.setItemText(0, f"preset default ({self._preset_terminal})")
        self._terminal_changed()

    def terminal(self) -> str:
        """The input mode in effect (the preset's when 'preset default' is selected)."""
        return self.ai_terminal.currentData() or self._preset_terminal

    def set_wiring(self, w: Wiring, ai_terminal: str | None = None, keep_terminal: bool = False):
        self._loading = True
        try:
            if not keep_terminal:
                self.ai_terminal.blockSignals(True)
                self.ai_terminal.setCurrentIndex(max(0, self.ai_terminal.findData(ai_terminal)))
                self.ai_terminal.blockSignals(False)
            self.ai_terminal.setItemText(0, f"preset default ({self._preset_terminal})")
            _fill(self.ao_x, _ao_choices(), w.ao_x)
            _fill(self.ao_y, _ao_choices(), w.ao_y)
            _fill(self.aux0, _ao_choices(), w.aux0, "not connected (dummy)")
            _fill(self.aux1, _ao_choices(), w.aux1, "not connected (dummy)")
            _fill(self.blank, _port0_choices(), w.blank_line)
            self.blank_high.setChecked(bool(w.blank_active_high))
            self.timed.set_items([(n, line, True) for n, line in w.timed_lines.items()])
            self.ai.set_choices(_ai_choices(self.terminal()))
            self.ai.set_items([(n, c, True) for n, c in w.ai_channels.items()])
            self.ext.set_items([(canonical_line(line), canonical_line(line), hi)
                                for line, hi in w.external_control_lines.items()])
            self.static.set_items([(n, canonical_line(line), w.static_lines_active_high.get(n, True))
                                   for n, line in w.static_lines.items()])
            _fill(self.clock_out, _pfi_choices(), canonical_line(w.sample_clock_out) if w.sample_clock_out else None,
                  "not exported")
            _fill(self.trigger_out, _pfi_choices(),
                  canonical_line(w.start_trigger_out) if w.start_trigger_out else None, "not exported")
            self._refresh_eds(w.eds_video)
        finally:
            self._loading = False
        self._emit()

    def _refresh_eds(self, keep=...):
        cur = self.eds_video.currentData() if keep is ... else keep
        names = [n for n, _, _ in self.ai.items() if n]
        self.eds_video.blockSignals(True)
        self.eds_video.clear()
        self.eds_video.addItem("none (no SE image with EDS)", None)
        for n in names:
            self.eds_video.addItem(n, n)
        i = self.eds_video.findData(cur) if cur is not None else 0
        if i < 0:  # renamed / removed: follow the first input
            i = 1 if names else 0
        self.eds_video.setCurrentIndex(i)
        self.eds_video.blockSignals(False)

    def _terminal_changed(self):
        self.ai.set_choices(_ai_choices(self.terminal()))
        self._emit()

    def _ai_changed(self):
        if not self._loading:
            self._refresh_eds()
        self._emit()

    def _used_pins(self):
        from ..hal.wiring import ai_pins

        w = self.wiring()
        pins = {pin_key(v) for k, v in w.signals().items() if not k.startswith("ai:")}
        for ch in w.ai_channels.values():
            pins.update(ai_pins(ch, self.terminal()))
        return pins

    def wiring(self) -> Wiring:
        return self._collect()[0]

    def selected_terminal(self) -> str | None:
        """The stored input mode (None: the preset's)."""
        return self.ai_terminal.currentData()

    def _collect(self):
        """(wiring, {signal key: widgets}, editor-level issues)."""
        widgets: dict[str, list] = {}
        extra: list[WiringIssue] = []

        def reg(key, *ws):
            widgets.setdefault(key, []).extend(w for w in ws if w is not None)

        def named(rows: _Rows, prefix: str, what: str):
            out, pol = {}, {}
            for r, (n, line, hi) in zip(rows.rows, rows.items()):
                if n in out:
                    extra.append(WiringIssue(f"Two {what}s are named '{n}'", (f"{prefix}{n}",)))
                out[n], pol[n] = line, hi
                reg(f"{prefix}{n}", r["name"], r["combo"])
            return out, pol

        timed, _ = named(self.timed, "timed:", "timed line")
        ai, _ = named(self.ai, "ai:", "video input")
        static, static_hi = named(self.static, "static:", "static line")
        ext = {}
        for r, (_, line, hi) in zip(self.ext.rows, self.ext.items()):
            if line in ext:
                extra.append(WiringIssue(f"External control {line} is listed twice", (f"ext:{line}",)))
            ext[line] = hi
            reg(f"ext:{line}", r["combo"])
        w = Wiring(
            ao_x=self.ao_x.currentData(), ao_y=self.ao_y.currentData(),
            aux0=self.aux0.currentData(), aux1=self.aux1.currentData(),
            blank_line=self.blank.currentData(), blank_active_high=self.blank_high.isChecked(),
            timed_lines=timed, external_control_lines=ext, static_lines=static,
            static_lines_active_high={n: hi for n, hi in static_hi.items() if not hi},
            ai_channels=ai, eds_video=self.eds_video.currentData(),
            sample_clock_out=self.clock_out.currentData(), start_trigger_out=self.trigger_out.currentData(),
        )
        for key, wd in (("ao_x", self.ao_x), ("ao_y", self.ao_y), ("aux0", self.aux0), ("aux1", self.aux1),
                        ("blank", self.blank), ("clock_out", self.clock_out), ("trigger_out", self.trigger_out),
                        ("eds_video", self.eds_video)):
            reg(key, wd)
        return w, widgets, extra

    def issues(self) -> list:
        w, _, extra = self._collect()
        return extra + w.check(self.terminal())

    def _emit(self, *_):
        if self._loading:
            return
        w, widgets, extra = self._collect()
        issues = extra + w.check(self.terminal())
        bad = {k for i in issues for k in i.signals}
        for key, ws in widgets.items():
            for wd in ws:
                wd.setStyleSheet(_BAD if key in bad else "")
        if issues:
            self.issues_label.setStyleSheet("color:#f55")
            self.issues_label.setText("Conflicts – fix before applying:\n" + "\n".join(f"• {i.message}" for i in issues))
        else:
            self.issues_label.setStyleSheet("color:#4c4")
            self.issues_label.setText(f"Wiring OK ({USB6363.name}, AI {self.terminal()}).")
        self.changed.emit()
