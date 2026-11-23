"""SEM (column) HAL module: magnification, beam, focus, stigmation, detector settings.

:class:`SEMControl` implements :class:`~easyebl.microscope.control.MicroscopeControl`
on top of a :class:`~easyebl.microscope.link.MicroscopeLink`:

* **readbacks** are cached (``state``) and returned at once; they are refreshed
  when the module starts, after every setting, and by a slow status poll
  (``MAG``/``HV``/``WD``, default every 10 s) so changes made on the SEM's own
  panel are picked up;
* **settings** are queued at the highest priority (``SEM_SET``) and return a
  Future.  Repeated settings of the same quantity *coalesce*: ten mouse-wheel
  steps while one ``MAG`` transaction is on the wire become one more
  transaction with the final value, not ten.

Raw numbers (``BEAM C 0–99``, ``FOCUS``, ``STG``, ``CONT``/``BRT``, ``EMI``)
are control settings, not physical units – the S-3000 protocol reports no
calibrated probe current.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Callable

from .control import MicroscopeControl, NotAvailable
from .link import POLL, READ, SEM_SET, MicroscopeLink
from .s3000 import commands as C

__all__ = ["SEMControl", "STATUS_ITEMS", "ALL_ITEMS", "MAGNIFICATIONS", "SCAN_MODES", "PRESET_KEYS"]

#: read on start and by "Refresh"
ALL_ITEMS = ("HV", "MAG", "WD", "EMI", "BEAM C", "BEAM F", "FOCUS C", "FOCUS F", "STG X", "STG Y",
             "CONT", "BRT", "RUN", "INPUTS AF")
#: polled in the background (changes on the SEM console that matter to imaging / EDS)
STATUS_ITEMS = ("MAG", "HV", "WD")
MAGNIFICATIONS: tuple[int, ...] = C.ALLOWED_MAGNIFICATIONS
SCAN_MODES = C.SCAN_MODES  # {"F": "TV scan 1", …}

# readback item -> state key
_KEYS = {"HV": "hv_kv", "MAG": "mag", "WD": "wd_mm", "EMI": "emission", "BEAM C": "beam_coarse",
         "BEAM F": "beam_fine", "FOCUS C": "focus_coarse", "FOCUS F": "focus_fine", "STG X": "stig_x",
         "STG Y": "stig_y", "CONT": "contrast", "BRT": "brightness", "RUN": "running", "INPUTS AF": "detector"}
_LIMITS = {"focus_coarse": (0, 16383), "focus_fine": (0, 4095), "stig_x": (0, 4095), "stig_y": (0, 4095),
           "beam_coarse": (0, 99), "beam_fine": (0, 99), "contrast": (0, 4095), "brightness": (0, 4095)}
#: what a beam preset holds (state key -> readback item), in the order a recall sends them
PRESET_KEYS = {"beam_coarse": "BEAM C", "beam_fine": "BEAM F", "stig_x": "STG X", "stig_y": "STG Y",
               "contrast": "CONT", "brightness": "BRT"}


def _parse(item: str, data: str) -> dict:
    """Readback data string -> {state key: value} (raw strings stay where the format is unknown)."""
    d = data.strip()
    if item == "HV":
        if d.upper() == "OFF":
            return {"hv_on": False, "hv_kv": None}
        try:
            return {"hv_on": True, "hv_kv": float(d), "hv_setpoint_kv": float(d)}
        except ValueError:
            return {"hv_on": None, "hv_kv": None}
    if item == "MAG":
        return {"mag": C.MAG_VALUES.get(d.lower())}
    if item == "RUN":
        return {"running": d.upper() == "ON"}
    if item == "INPUTS AF":
        inv = {v: k for k, v in C.SIGNAL_TOKENS.items()}
        return {"detector": inv.get(d.upper(), d)}
    try:
        return {_KEYS[item]: int(d)}
    except ValueError:
        return {_KEYS[item]: d}


class SEMControl(MicroscopeControl):
    """The S-3000 column through the shared link (see module docstring)."""

    def __init__(self, link: MicroscopeLink, status_poll_s: float = 10.0, focus_step: int = 8,
                 coarse_focus_step: int = 64, stig_step: int = 16, refresh_on_start: bool = True):
        self.link = link
        link.sem = self
        self.name = f"Hitachi S-3000 ({link.port})"
        self.focus_step, self.coarse_focus_step, self.stig_step = int(focus_step), int(coarse_focus_step), int(stig_step)
        self.state: dict = {"hv_on": None, "hv_kv": None, "hv_setpoint_kv": None, "mag": None, "wd_mm": None}
        self.t_read: dict = {}  # state key -> time of the last readback
        self._lock = threading.RLock()
        self._targets: dict = {}
        self._jobs: dict = {}
        self._listeners: list[Callable[[dict], None]] = []
        self._poll_s = float(status_poll_s)
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, name="easyebl-microscope-sem-poll", daemon=True)
        self._poller.start()
        if refresh_on_start:
            self.refresh()

    # ------------------------------------------------------------------ MicroscopeControl
    @property
    def connected(self) -> bool:
        return self.link.connected

    def add_listener(self, fn) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self.state.items() if v is not None}

    def magnification(self) -> float | None:
        return self.state.get("mag")

    def working_distance_mm(self) -> float | None:
        return self.state.get("wd_mm")

    def accelerating_voltage_kv(self) -> float | None:
        return self.state.get("hv_kv") if self.state.get("hv_on") else None

    @property
    def hv_setpoint_kv(self) -> float | None:
        return self.state.get("hv_setpoint_kv")

    # ------------------------------------------------------------------ cache
    def _update(self, changes: dict) -> None:
        now = time.time()
        with self._lock:
            diff = {k: v for k, v in changes.items() if self.state.get(k, object()) != v}
            self.state.update(changes)
            for k in changes:
                self.t_read[k] = now
        if diff:
            for fn in list(self._listeners):
                try:
                    fn(dict(diff))
                except Exception:  # noqa: BLE001 - a listener must not break the link
                    pass
            self.link.events.emit("sem", "", changes=diff)

    def _read_items(self, session, items) -> dict:
        out = {}
        for item in items:
            out.update(_parse(item, session.sem.read(item)))
        self._update(out)
        return out

    def refresh(self, items=ALL_ITEMS, priority: int = READ) -> concurrent.futures.Future:
        """Read *items* (default: everything shown in the GUI)."""
        items = tuple(items)
        return self.link.submit(lambda s: self._read_items(s, items), priority, "SEM read " + ", ".join(items),
                                key=("sem-read", items))

    # ------------------------------------------------------------------ status poll
    @property
    def status_poll_s(self) -> float:
        return self._poll_s

    def set_status_poll(self, seconds: float) -> None:
        """Background read of MAG / HV / WD every *seconds* (0: off)."""
        self._poll_s = max(0.0, float(seconds))

    def _poll_loop(self) -> None:
        t_next = time.monotonic() + max(self._poll_s, 1.0)
        while not self._stop.wait(0.25):
            if self._poll_s <= 0:
                t_next = time.monotonic() + 1.0
                continue
            if time.monotonic() >= t_next:
                t_next = time.monotonic() + self._poll_s
                if self.link.connected:
                    try:
                        self.refresh(STATUS_ITEMS, priority=POLL)
                    except RuntimeError:  # link closed
                        return

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ coalesced settings
    def _set_latest(self, name: str, value, send: Callable, what: str) -> concurrent.futures.Future:
        """Queue *send(session, value)*; a newer value for *name* replaces one not yet sent."""
        with self._lock:
            self._targets[name] = value
            fut = self._jobs.get(name)
            if fut is not None and not fut.done():
                return fut

            def job(session):
                result = None
                while True:
                    with self._lock:
                        if name not in self._targets:
                            self._jobs.pop(name, None)
                            return result
                        v = self._targets.pop(name)
                    try:
                        result = send(session, v)
                    except BaseException:
                        with self._lock:
                            self._jobs.pop(name, None)
                            self._targets.pop(name, None)
                        raise

            fut = self.link.submit(job, SEM_SET, what)
            self._jobs[name] = fut
            return fut

    def _target(self, name: str, state_key: str):
        with self._lock:
            return self._targets.get(name, self.state.get(state_key))

    def _command(self, text_fn, what: str) -> concurrent.futures.Future:
        return self.link.submit(lambda s: s.sem.command(text_fn()), SEM_SET, what)

    # magnification ------------------------------------------------------
    def set_magnification(self, mag) -> concurrent.futures.Future:
        m = C.parse_magnification(mag)  # refused before sending if not in the list

        def send(session, v):
            rb = session.sem.set_magnification(v)  # returns the R MAG readback (the SEM may limit it)
            self._update(_parse("MAG", rb))
            return self.state.get("mag")

        return self._set_latest("mag", m, send, "MAG (latest)")

    def step_magnification(self, steps: int) -> concurrent.futures.Future:
        cur = self._target("mag", "mag")
        if cur is None:
            raise NotAvailable("magnification (not read yet)")
        lst = MAGNIFICATIONS
        i = min(range(len(lst)), key=lambda k: abs(lst[k] - cur))
        return self.set_magnification(lst[max(0, min(len(lst) - 1, i + int(steps)))])

    # HV --------------------------------------------------------------------
    def set_hv(self, kv) -> concurrent.futures.Future:
        """Accelerating-voltage setpoint (0.30–9.99 kV in 0.01 kV, 10.0–30.0 kV in 0.1 kV steps).
        Does not switch HV on."""
        text = C.hv_setting(kv)

        def send(session, v):
            session.sem.command(v)
            kv_ = float(v.split()[1])
            ch = {"hv_setpoint_kv": kv_}
            if self.state.get("hv_on"):
                ch["hv_kv"] = kv_
            self._update(ch)

        return self._set_latest("hv", text, send, text)

    def hv_on(self) -> concurrent.futures.Future:
        def job(session):
            session.sem.hv_on()
            return self._read_items(session, ("HV", "EMI"))

        return self.link.submit(job, SEM_SET, "HV ON")

    def hv_off(self) -> concurrent.futures.Future:
        def job(session):
            session.sem.hv_off()
            return self._read_items(session, ("HV", "EMI"))

        return self.link.submit(job, SEM_SET, "HV OFF")

    # beam current, focus, stigmator, contrast ------------------------------------
    def _set_int(self, state_key: str, value: int, text_fn, what: str) -> concurrent.futures.Future:
        lo, hi = _LIMITS[state_key]
        v = int(round(value))
        if not lo <= v <= hi:
            raise ValueError(f"{what} must be {lo}…{hi}")
        text = text_fn(v)  # validated by the command builder too

        def send(session, t):
            session.sem.command(t)
            self._update({state_key: int(t.rsplit(" ", 1)[1])})

        return self._set_latest(state_key, text, send, text)

    def set_beam(self, stage: str, n: int) -> concurrent.futures.Future:
        """Beam current setting 1, coarse ("C") or fine ("F"), 0–99 (raw)."""
        key = "beam_coarse" if stage.upper() == "C" else "beam_fine"
        return self._set_int(key, n, lambda v: C.beam(1, stage, v), f"BEAM {stage}")

    def set_focus(self, stage: str, n: int) -> concurrent.futures.Future:
        key = "focus_coarse" if stage.upper() == "C" else "focus_fine"
        return self._set_int(key, n, lambda v: C.focus(stage, v), f"FOCUS {stage}")

    def focus(self, delta: float, coarse: bool = False) -> concurrent.futures.Future:
        stage, key = ("C", "focus_coarse") if coarse else ("F", "focus_fine")
        cur = self._target_int(key)
        lo, hi = _LIMITS[key]
        step = self.coarse_focus_step if coarse else self.focus_step
        return self.set_focus(stage, max(lo, min(hi, cur + int(round(delta * step)))))

    def _target_int(self, key: str) -> int:
        with self._lock:
            t = self._targets.get(key)
            cur = int(t.rsplit(" ", 1)[1]) if isinstance(t, str) else self.state.get(key)
        if cur is None or not isinstance(cur, int):
            raise NotAvailable(f"{key.replace('_', ' ')} (not read yet)")
        return cur

    #: writes of the same stigmator value after a wrong readback (see :meth:`set_stigmator`)
    STIG_RETRIES = 2

    def set_stigmator(self, axis: str, n: int) -> concurrent.futures.Future:
        """Stigmator X / Y, 0–4095 (raw; the SEM moves in steps of 16).

        The S-3000 misapplies a single write that changes the value by one step (±16): within
        ~0.3 s it moves to ±240 instead – the step lands on the coarse part (×256) and the fine
        part moves the other way – and stays there.  The same value written twice holds (seen
        stable for 3 s), so every value is written twice, read back, and written again if the SEM
        reports something else (values off the 16-step grid are rounded by the SEM: a readback
        within one step is accepted)."""
        axis = axis.upper()
        key = "stig_x" if axis == "X" else "stig_y"
        item = f"STG {axis}"
        lo, hi = _LIMITS[key]
        v = int(round(n))
        if not lo <= v <= hi:
            raise ValueError(f"{item} must be {lo}…{hi}")
        text = C.stigmator(axis, v)

        def send(session, t):
            want = int(t.rsplit(" ", 1)[1])
            for _ in range(1 + self.STIG_RETRIES):
                session.sem.command(t)
                session.sem.command(t)  # a single one-step write drifts by ±240 (see above)
                got = _parse(item, session.sem.read(item)).get(key)
                if got is None or abs(got - want) < 16:  # the SEM's stigmator step
                    break
            self._update({key: got if got is not None else want})

        return self._set_latest(key, text, send, text)

    def stigmate(self, dx: float, dy: float) -> concurrent.futures.Future:
        fut = None
        for axis, d in (("X", dx), ("Y", dy)):
            if d:
                key = "stig_x" if axis == "X" else "stig_y"
                lo, hi = _LIMITS[key]
                fut = self.set_stigmator(axis, max(lo, min(hi, self._target_int(key) + int(round(d * self.stig_step)))))
        if fut is None:
            raise ValueError("no stigmator change")
        return fut

    def set_contrast(self, n: int) -> concurrent.futures.Future:
        return self._set_int("contrast", n, C.contrast, "CONT")

    def set_brightness(self, n: int) -> concurrent.futures.Future:
        return self._set_int("brightness", n, C.brightness, "BRT")

    # beam presets --------------------------------------------------------------
    @staticmethod
    def preset_limits(key: str) -> tuple[int, int]:
        return _LIMITS[key]

    def read_preset(self) -> concurrent.futures.Future:
        """Read the preset settings (beam 1, stigmator, contrast / brightness) fresh from the SEM.
        Resolves to ``{state key: value}`` (keys of :data:`PRESET_KEYS`)."""
        items = tuple(PRESET_KEYS.values())

        def job(session):
            got = self._read_items(session, items)
            missing = [k for k in PRESET_KEYS if not isinstance(got.get(k), int)]
            if missing:
                raise ValueError("unreadable: " + ", ".join(missing))
            return {k: got[k] for k in PRESET_KEYS}

        return self.link.submit(job, READ, "SEM read preset")

    def apply_preset(self, values: dict) -> concurrent.futures.Future:
        """Send a beam preset: BEAM C/F, STG X/Y, CONT, BRT.  Every value is checked before
        anything is sent; the future resolves once all six settings have gone out."""
        vals = {}
        for k in PRESET_KEYS:
            lo, hi = _LIMITS[k]
            v = values.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= round(v) <= hi:
                raise ValueError(f"preset {k.replace('_', ' ')} must be {lo}…{hi} (got {v!r})")
            vals[k] = int(round(v))
        futs = [self.set_beam("C", vals["beam_coarse"]), self.set_beam("F", vals["beam_fine"]),
                self.set_stigmator("X", vals["stig_x"]), self.set_stigmator("Y", vals["stig_y"]),
                self.set_contrast(vals["contrast"]), self.set_brightness(vals["brightness"])]

        def barrier(_session):  # queued after the six settings (same priority, FIFO): they are done
            for f in futs:
                if not f.cancelled() and f.exception(timeout=0) is not None:
                    raise f.exception()
            return vals

        return self.link.submit(barrier, SEM_SET, "beam preset")

    def set_working_distance(self, mm: int) -> concurrent.futures.Future:
        """``WD n`` (3–60 mm): the working distance the column computes for (focus tracking)."""
        text = C.working_distance(int(mm))

        def send(session, t):
            session.sem.command(t)
            self._update({"wd_mm": int(t.split()[1])})

        return self._set_latest("wd", text, send, text)

    # scan, detector -----------------------------------------------------------------
    def set_scan(self, mode: str) -> concurrent.futures.Future:
        text = C.scan(mode)

        def job(session):
            session.sem.command(text)
            self._update({"scan": mode})

        return self.link.submit(job, SEM_SET, text)

    def set_running(self, on: bool) -> concurrent.futures.Future:
        """``RUN ON`` / ``RUN OFF`` (freeze the SEM's own display; does not blank the beam)."""
        def job(session):
            session.sem.run() if on else session.sem.freeze()
            self._update({"running": bool(on)})

        return self.link.submit(job, SEM_SET, "RUN ON" if on else "RUN OFF")

    def set_detector(self, signal: str) -> concurrent.futures.Future:
        text = C.inputs("AF", signal)

        def job(session):
            session.sem.command(text)
            self._update(_parse("INPUTS AF", text.split()[-1]))

        return self.link.submit(job, SEM_SET, text)

    # automatic routines (long: they hold the port until the SEM answers) ---------
    def _auto(self, what: str, fn, reread=()) -> concurrent.futures.Future:
        def job(session):
            fn(session.sem)
            return self._read_items(session, reread) if reread else None

        return self.link.submit(job, SEM_SET, what)

    def auto_focus(self, coarse: bool = False) -> concurrent.futures.Future:
        return self._auto("AFC " + ("C" if coarse else "F"), lambda c: c.afc("C" if coarse else "F"),
                          ("FOCUS C", "FOCUS F"))

    def auto_stigma(self) -> concurrent.futures.Future:
        return self._auto("ASF ON", lambda c: c.asf(), ("STG X", "STG Y"))

    def auto_brightness_contrast(self) -> concurrent.futures.Future:
        return self._auto("ABC ON", lambda c: c.abc(), ("CONT", "BRT"))

    def alignment(self, kind: str) -> concurrent.futures.Future:
        if kind.lower() in ("gun", "beam"):
            return self._auto("AGA ON", lambda c: c.aga())  # automatic gun alignment
        raise NotAvailable(f"{kind} alignment")
