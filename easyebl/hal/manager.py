"""Shared access to one DAQ device.

There is exactly one :class:`DAQManager` per physical device.  Application
modules (imaging, alignment, lithography, GUIs, …) each obtain their own
:class:`DAQHandle`::

    mgr = DAQManager(SimulatedBackend(cfg)).open()
    imaging = mgr.get_handle("imaging")
    litho = mgr.get_handle("litho")

Rules enforced here (not in the backends):

* **One hardware-timed operation at a time.**  ``run_stream`` takes the device
  operation lock.  A second caller either waits (``timeout``) or gets
  :class:`DeviceBusyError`.  The manager records which handle is active.
* **External scan control** is a HAL function of its own: ``set_external_control``
  switches the SEM between its internal scan and the DAQ, whatever wires
  (relay enables) the configuration maps it to.  It is claimed exclusively
  (resource ``"external_control"``), refused while a stream runs, blanks the
  beam before switching, and is handed back to the SEM (internal scan) when
  the holding handle is closed or the manager shuts down.
* **Exclusive claims for auxiliary resources.**  Software-timed lines can only
  be driven by the handle that claimed them.  Arbitrary named resources
  (``"stage"``, ``"sem_serial"`` …) can be claimed too, so the same mechanism
  coordinates future subsystems.
* **Safe state.**  After every stream – normal end, abort or exception – the
  beam is blanked and parked.  Static writes are rejected while a stream is
  running.
* **Output validation.**  Rate limits, AO voltage limits (and NaN/inf), AI
  channel names and marker line names are checked before/while streaming.
* **Lost connection.**  If the device stops answering (USB unplugged, power
  cycled), the failing call raises :class:`DeviceLostError`, the manager
  marks the device disconnected (beam state unknown), every further operation
  fails immediately instead of hanging, and a :class:`ReconnectWatchdog`
  probes the device.  When it is back the manager reopens it, writes the safe
  state, selects the SEM's internal scan and then re-applies the external-scan
  request and static lines of handles that still hold them.  Listeners get
  ``device_lost`` / ``device_restored`` events; :meth:`wait_connected` blocks
  until the device is back.

Callbacks (``on_ai``, ``progress``, listeners) run in the streaming thread
while the device lock is held: keep them short and never block on another
thread that may itself be waiting for the device (e.g. a GUI thread calling
``close()``).  GUIs should forward data through queued signals.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import itertools
import threading
import time
from typing import Callable, Iterable, Iterator, Sequence
import weakref

import numpy as np

from ..core.waveform import OutputChunk
from .backend import (
    AIConsumer,
    DAQBackend,
    DeviceBusyError,
    DeviceLostError,
    HALError,
    ResourceConflictError,
    StreamResult,
    VoltageRangeError,
)
from .config import AI_TERMINALS, AUX_OUTPUTS, PHANTOM_KINDS
from .connection import ReconnectWatchdog

_KEEP = object()  # "leave unchanged" marker for optional arguments

__all__ = ["DAQManager", "DAQHandle", "BeamState", "EXTERNAL_CONTROL"]

#: Resource name of the SEM internal/external scan switch.
EXTERNAL_CONTROL = "external_control"

Listener = Callable[[str, dict], None]

#: Errors that describe a *request* problem, never a lost device (no probing needed).
_LOGICAL_ERRORS = (ValueError, TypeError, KeyError, DeviceBusyError, ResourceConflictError, VoltageRangeError)


@dataclass
class BeamState:
    x_v: float
    y_v: float
    blank: bool


def _unknown_beam() -> BeamState:
    # never claim the beam is blanked if we could not blank it
    return BeamState(float("nan"), float("nan"), False)


class DAQManager:
    """Owner of a backend; hands out :class:`DAQHandle` objects."""

    def __init__(self, backend: DAQBackend, auto_reconnect: bool = True, reconnect_interval_s: float = 0.5):
        self.backend = backend
        self.auto_reconnect = auto_reconnect
        self._op_lock = threading.Lock()
        self._state = threading.RLock()
        self._claims: dict[str, "DAQHandle"] = {}
        self._handles: "weakref.WeakSet[DAQHandle]" = weakref.WeakSet()
        self._active: DAQHandle | None = None
        self._active_abort: threading.Event | None = None
        self._listeners: list[Listener] = []
        self._ids = itertools.count(1)
        self._beam = BeamState(backend.config.park_x_v, backend.config.park_y_v, True)
        self._line_state: dict[str, bool] = {}
        self._external: bool | None = None  # actual state (None = unknown / not configured)
        self._external_wanted = False  # what the holder of "external_control" asked for
        self._aux_state: dict[str, float] = {k: 0.0 for k in AUX_OUTPUTS}
        self._opened = False
        self._connected = threading.Event()
        self._lost_reason: str | None = None
        self.disconnects = 0
        self.reconnects = 0
        self._watchdog = ReconnectWatchdog(self, reconnect_interval_s)
        self._atexit = None

    # ------------------------------------------------------------------ lifecycle
    def open(self, allow_missing: bool = False) -> "DAQManager":
        """Open the device (safe state, SEM internal scan).

        With *allow_missing* a device that is not there (yet) is not an error:
        the manager starts disconnected and connects as soon as it appears.
        """
        if self._opened:
            return self
        try:
            with self._op_lock:
                self.backend.open()
                self._bring_up()
        except Exception as e:  # noqa: BLE001
            lost = self._as_lost(e)
            if lost is None or not allow_missing:
                try:
                    self.backend.close()
                except Exception:  # noqa: BLE001
                    pass
                raise lost if lost is not None and lost is not e else e
            self._opened = True
            self._register_atexit()
            self._emit("opened", {"connected": False})
            self._mark_lost(lost)
            return self
        self._opened = True
        self._connected.set()
        self._register_atexit()
        # build the phantom now (≈1–2 s once), not inside the first stream start
        self.backend.prepare_phantom()
        self._emit("opened", {"connected": True})
        return self

    def _bring_up(self) -> None:
        """Safe state, internal scan, then re-apply what handles still hold (op lock held)."""
        self.backend.safe_state()
        with self._state:
            self._beam = BeamState(self.config.park_x_v, self.config.park_y_v, True)
        if self.has_external_control:
            # a defined start: the SEM drives its own scan
            self.backend.set_external_control(False)
            with self._state:
                self._external = False
                want = self._external_wanted and EXTERNAL_CONTROL in self._claims
            if want:
                self.backend.set_external_control(True)
                with self._state:
                    self._external = True
        with self._state:
            lines = {n: v for n, v in self._line_state.items() if n in self._claims}
            aux = {n: (v if n in self._claims else 0.0) for n, v in self._aux_state.items()}
        for name, value in lines.items():
            self.backend.set_static_line(name, value)
        for name, volts in aux.items():  # aux outputs: 0 V unless a handle holds a value
            if name not in self.config.phantom_outputs:
                self.backend.set_aux_output(name, volts)

    def _register_atexit(self) -> None:
        # if the program ends without close() (e.g. an unhandled exception), still blank the
        # beam and give the scan back to the SEM; a weak reference keeps the manager collectable
        ref = weakref.ref(self)

        def _at_exit():
            m = ref()
            if m is not None:
                try:
                    m.close()
                except Exception:  # noqa: BLE001
                    pass

        self._atexit = _at_exit
        atexit.register(_at_exit)

    def close(self) -> None:
        self._watchdog.stop()
        with self._state:
            if self._active_abort is not None:
                self._active_abort.set()
        # wait for a running stream to finish its cleanup
        with self._op_lock:
            if self._opened:
                try:
                    if self._connected.is_set():
                        self.backend.safe_state()
                        if self.has_external_control and self._external:
                            self.backend.set_external_control(False)  # give the scan back to the SEM
                            self._external = False
                        for name, volts in list(self._aux_state.items()):
                            if volts and name not in self.config.phantom_outputs:
                                self.backend.set_aux_output(name, 0.0)
                                self._aux_state[name] = 0.0
                except Exception as e:  # noqa: BLE001 - closing must always complete
                    if self._as_lost(e) is None:
                        self._emit("error", {"message": f"close: {e}"})
                finally:
                    try:
                        self.backend.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._opened = False
                    self._connected.clear()
                    if self._atexit is not None:
                        atexit.unregister(self._atexit)
                        self._atexit = None
                    self._emit("closed", {})

    def __enter__(self) -> "DAQManager":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- connection
    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def lost_reason(self) -> str | None:
        with self._state:
            return self._lost_reason

    def wait_connected(self, timeout: float | None = None) -> bool:
        """Block until the device is connected (True) or *timeout* expires (False)."""
        return self._connected.wait(timeout)

    @property
    def reconnecting(self) -> bool:
        return self._watchdog.running

    def _as_lost(self, e: BaseException) -> DeviceLostError | None:
        """Classify a backend failure: a :class:`DeviceLostError` if the device is gone."""
        if isinstance(e, DeviceLostError):
            return e
        if isinstance(e, _LOGICAL_ERRORS):
            return None
        try:
            lost = self.backend.is_device_lost(e)
        except Exception:  # noqa: BLE001
            lost = False
        if not lost:
            # unknown failure: ask the device whether it is still there (harmless probe)
            try:
                lost = not self.backend.probe()
            except Exception:  # noqa: BLE001
                lost = True
        if lost:
            err = DeviceLostError(f"device {self.config.device} not responding: {e}")
            err.__cause__ = e
            return err
        return None

    def _mark_lost(self, err: BaseException) -> None:
        with self._state:
            was = self._connected.is_set()
            self._connected.clear()
            self._beam = _unknown_beam()
            if self.has_external_control:
                self._external = None
            self._lost_reason = str(err)
            if self._active_abort is not None:
                self._active_abort.set()
        if was:
            self.disconnects += 1
        self._emit("device_lost", {"reason": str(err)})
        if self.auto_reconnect and self._opened:
            self._watchdog.start()

    def _try_reconnect(self) -> bool:
        """One reconnect attempt (watchdog thread).  True when the device is back."""
        if not self._opened or self._connected.is_set():
            return True
        if not self.backend.probe():
            return False
        if not self._op_lock.acquire(timeout=1.0):
            return False
        try:
            self.backend.recover()
            self._bring_up()
        except Exception as e:  # noqa: BLE001
            with self._state:
                self._beam = _unknown_beam()
                self._lost_reason = f"reconnect failed: {e}"
            return False
        finally:
            self._op_lock.release()
        with self._state:
            self._lost_reason = None
        self.reconnects += 1
        self._connected.set()
        self._emit("device_restored", {"external_control": self._external})
        return True

    def reconnect(self, timeout: float = 5.0) -> bool:
        """Try to reconnect now (also starts the watchdog); True if connected within *timeout*."""
        if self._connected.is_set():
            return True
        if not self._opened:
            raise HALError("manager is not open")
        self._watchdog.start()
        return self._connected.wait(timeout)

    def _call(self, fn, *args):
        """Run a backend call; a lost device is reported and turned into DeviceLostError."""
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            lost = self._as_lost(e)
            if lost is None:
                raise
            self._mark_lost(lost)
            raise lost from e

    def _require_connected(self) -> None:
        if not self._connected.is_set():
            raise DeviceLostError(
                f"device {self.config.device} is not connected"
                + (f" ({self._lost_reason})" if self._lost_reason else "")
                + (" – reconnecting" if self._watchdog.running else "")
            )

    # -------------------------------------------------------------------- queries
    @property
    def config(self):
        return self.backend.config

    def capabilities(self):
        return self.backend.capabilities()

    @property
    def is_busy(self) -> bool:
        return self._op_lock.locked()

    @property
    def active_owner(self) -> str | None:
        with self._state:
            return self._active.owner if self._active else None

    @property
    def beam_state(self) -> BeamState:
        with self._state:
            return BeamState(self._beam.x_v, self._beam.y_v, self._beam.blank)

    @property
    def has_external_control(self) -> bool:
        return bool(self.config.external_control_lines)

    @property
    def external_control(self) -> bool | None:
        """True = the SEM scans from the DAQ, False = internal scan, None = not configured/unknown."""
        with self._state:
            return self._external

    @property
    def aux_state(self) -> dict:
        """{aux output: (physical channel or None for a dummy, last value in V)}."""
        with self._state:
            vals = dict(self._aux_state)
        return {n: (self.config.aux_physical(n), v) for n, v in vals.items()}

    @property
    def line_state(self) -> dict:
        with self._state:
            return dict(self._line_state)

    def handles(self) -> list["DAQHandle"]:
        return [h for h in list(self._handles) if not h.closed]

    def claims(self) -> dict[str, str]:
        with self._state:
            return {r: h.owner for r, h in self._claims.items()}

    # ------------------------------------------------------------------ handles
    def get_handle(self, owner: str) -> "DAQHandle":
        h = DAQHandle(self, owner, next(self._ids))
        self._handles.add(h)
        self._emit("handle_created", {"owner": owner})
        return h

    def abort_all(self) -> None:
        """Abort whatever stream is running (emergency stop)."""
        with self._state:
            if self._active_abort is not None:
                self._active_abort.set()

    # ---------------------------------------------------------------- listeners
    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, event: str, data: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(event, data)
            except Exception:  # listeners must never break hardware control
                pass

    # -------------------------------------------------------------- internals
    def _claim(self, handle: "DAQHandle", resources: Sequence[str]) -> None:
        with self._state:
            for r in resources:
                holder = self._claims.get(r)
                if holder is not None and holder is not handle:
                    raise ResourceConflictError(f"'{r}' is claimed by '{holder.owner}'")
            for r in resources:
                self._claims[r] = handle
        self._emit("claim", {"owner": handle.owner, "resources": list(resources)})

    def _release(self, handle: "DAQHandle", resources: Sequence[str] | None) -> None:
        with self._state:
            if resources is None:
                resources = [r for r, h in self._claims.items() if h is handle]
            resources = list(resources)
            give_back = EXTERNAL_CONTROL in resources and self._claims.get(EXTERNAL_CONTROL) is handle
        if give_back:
            with self._state:
                self._external_wanted = False
            if self._external is not False:
                try:
                    self._switch_external(handle, False, wait=5.0)
                except DeviceLostError:
                    pass  # the device comes back in internal scan (reconnect selects it)
                except Exception as e:  # noqa: BLE001 - releasing must not fail; report instead
                    self._emit("error", {"owner": handle.owner, "message": f"could not return scan to the SEM: {e}"})
        for r in resources:
            if r in AUX_OUTPUTS and self._claims.get(r) is handle and self._aux_state.get(r):
                try:  # a released aux output goes back to 0 V
                    self._write_aux(handle, r, 0.0, wait=5.0)
                except DeviceLostError:
                    with self._state:
                        self._aux_state[r] = 0.0
                except Exception as e:  # noqa: BLE001
                    self._emit("error", {"owner": handle.owner, "message": f"could not reset {r}: {e}"})
        with self._state:
            for r in resources:
                if self._claims.get(r) is handle:
                    del self._claims[r]
        self._emit("release", {"owner": handle.owner, "resources": list(resources)})

    def _holds(self, handle: "DAQHandle", resource: str) -> bool:
        with self._state:
            return self._claims.get(resource) is handle

    def _static(self, handle: "DAQHandle", x_v: float, y_v: float, blank: bool) -> None:
        lim = self.config.ao_limit_v
        if not (np.isfinite(x_v) and np.isfinite(y_v)):
            raise VoltageRangeError("static position must be finite")
        if abs(x_v) > lim or abs(y_v) > lim:
            raise VoltageRangeError(f"static position ({x_v:.4g}, {y_v:.4g}) V exceeds ±{lim} V")
        self._require_connected()
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError(f"device busy (stream owned by '{self.active_owner}')")
        try:
            self._call(self.backend.write_static, float(x_v), float(y_v), bool(blank))
            with self._state:
                self._beam = BeamState(float(x_v), float(y_v), bool(blank))
        finally:
            self._op_lock.release()
        self._emit("static", {"owner": handle.owner, "x_v": x_v, "y_v": y_v, "blank": blank})

    def _blank_here(self) -> None:
        """Blank at the current position (park if the position is unknown).  Op lock held."""
        b = self._beam
        x, y = (b.x_v, b.y_v) if np.isfinite(b.x_v) and np.isfinite(b.y_v) else (
            self.config.park_x_v, self.config.park_y_v)
        self._call(self.backend.write_static, x, y, True)
        with self._state:
            self._beam = BeamState(x, y, True)

    def _set_line(self, handle: "DAQHandle", name: str, value: bool) -> None:
        if name not in self.config.static_lines:
            raise HALError(f"unknown static line '{name}'")
        if not self._holds(handle, name):
            holder = self.claims().get(name)
            raise ResourceConflictError(
                f"line '{name}' must be claimed first" + (f" (held by '{holder}')" if holder else "")
            )
        self._require_connected()
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError("cannot switch static lines while a stream is running")
        try:
            # the line may switch deflection sources: always blank first
            self._blank_here()
            self._call(self.backend.set_static_line, name, bool(value))
            with self._state:
                self._line_state[name] = bool(value)
        finally:
            self._op_lock.release()
        self._emit("line", {"owner": handle.owner, "line": name, "value": bool(value)})

    def _switch_external(self, handle: "DAQHandle", on: bool, wait: float = 0.0) -> None:
        if not self.has_external_control:
            raise HALError("this DAQ configuration has no external-control lines")
        self._require_connected()
        ok = self._op_lock.acquire(timeout=wait) if wait > 0 else self._op_lock.acquire(blocking=False)
        if not ok:
            raise DeviceBusyError(
                f"cannot switch external scan control while a stream runs (owner '{self.active_owner}')"
            )
        try:
            # the switch changes the deflection source: blank first, at the current position
            self._blank_here()
            with self._state:
                self._external = None  # unknown while switching
            self._call(self.backend.set_external_control, bool(on))
            with self._state:
                self._external = bool(on)
        finally:
            self._op_lock.release()
        self._emit("external_control", {"owner": handle.owner, "on": bool(on)})

    def _set_external(self, handle: "DAQHandle", on: bool) -> None:
        if not self.has_external_control:
            raise HALError("this DAQ configuration has no external-control lines")
        if on and not self._holds(handle, EXTERNAL_CONTROL):
            self._claim(handle, [EXTERNAL_CONTROL])  # raises if another handle holds it
        elif not on and not self._holds(handle, EXTERNAL_CONTROL):
            holder = self.claims().get(EXTERNAL_CONTROL)
            if holder is not None:
                raise ResourceConflictError(f"external control is held by '{holder}'")
        with self._state:
            self._external_wanted = bool(on)
        self._switch_external(handle, on)

    def _write_aux(self, handle: "DAQHandle", name: str, volts: float, wait: float = 0.0) -> None:
        self._require_connected()
        ok = self._op_lock.acquire(timeout=wait) if wait > 0 else self._op_lock.acquire(blocking=False)
        if not ok:
            raise DeviceBusyError(f"cannot set {name} while a stream runs (owner '{self.active_owner}')")
        try:
            self._call(self.backend.set_aux_output, name, float(volts))
            with self._state:
                self._aux_state[name] = float(volts)
        finally:
            self._op_lock.release()
        self._emit("aux", {"owner": handle.owner, "output": name, "volts": float(volts),
                           "physical": self.config.aux_physical(name)})

    def _set_aux(self, handle: "DAQHandle", name: str, volts: float) -> None:
        if name not in AUX_OUTPUTS:
            raise HALError(f"unknown aux output '{name}'; available: {list(AUX_OUTPUTS)}")
        if not np.isfinite(volts) or abs(volts) > self.config.ao_limit_v:
            raise VoltageRangeError(f"{name}: {volts:g} V outside ±{self.config.ao_limit_v} V")
        if name in self.config.phantom_outputs:
            raise ResourceConflictError(f"{name} is driven by the phantom specimen (DAQConfig.phantom_output/sdd_output)")
        if not self._holds(handle, name):
            self._claim(handle, [name])  # exclusive, like the other resources
        self._write_aux(handle, name, volts)

    @property
    def phantom(self) -> dict:
        """{"output", "physical", "noise_v", "kind", "sdd_output", "sdd_physical", "eds"}.

        ``output``/``physical``: image output (aux name / AO channel, None = off or dummy);
        ``sdd_*``: the EDS phantom's SDD signal; ``eds``: EDS phantom parameters.
        """
        cfg = self.config
        return {"output": cfg.phantom_output, "physical": cfg.phantom_physical, "noise_v": cfg.phantom_noise_v,
                "kind": cfg.phantom_kind, "sdd_output": cfg.sdd_output, "sdd_physical": cfg.sdd_physical,
                "eds": dict(cfg.eds_phantom)}

    def _set_ai_terminal(self, handle: "DAQHandle", mode: str) -> None:
        mode = str(mode).upper()
        if mode not in AI_TERMINALS:
            raise HALError(f"unknown AI input mode '{mode}'; choose from {list(AI_TERMINALS)}")
        if mode == self.config.ai_terminal:
            return
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError(f"the AI input mode cannot change while a stream runs (owner '{self.active_owner}')")
        try:
            self.config.ai_terminal = mode  # used by every following stream (and the probe)
        finally:
            self._op_lock.release()
        self._emit("ai_terminal", {"owner": handle.owner, "mode": mode})

    def _set_phantom(self, handle: "DAQHandle", output, noise_v: float | None, kind: str | None = None,
                     sdd_output=_KEEP, eds: dict | None = None) -> None:
        cfg = self.config
        if output is not _KEEP and output is not None and output not in AUX_OUTPUTS:
            raise HALError(f"unknown aux output '{output}'; available: {list(AUX_OUTPUTS)} or None")
        if sdd_output is not _KEEP and sdd_output is not None and sdd_output not in AUX_OUTPUTS:
            raise HALError(f"unknown aux output '{sdd_output}'; available: {list(AUX_OUTPUTS)} or None")
        if kind is not None and kind not in PHANTOM_KINDS:
            raise HALError(f"unknown phantom kind '{kind}'; choose from {list(PHANTOM_KINDS)}")
        if noise_v is not None and not (np.isfinite(noise_v) and 0 <= noise_v <= cfg.ao_limit_v):
            raise VoltageRangeError(f"phantom noise must be 0…{cfg.ao_limit_v} V rms")
        noise = cfg.phantom_noise_v if noise_v is None else float(noise_v)
        new_out = cfg.phantom_output if output is _KEEP else output
        new_kind = cfg.phantom_kind if kind is None else kind
        new_sdd = cfg.sdd_output if sdd_output is _KEEP else sdd_output
        new_eds = dict(cfg.eds_phantom)
        if eds:
            new_eds.update(eds)
        live = {"cps"}  # parameters that may change while a stream runs
        if (new_out, new_kind, new_sdd) == (cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output) and all(
                k in live or cfg.eds_phantom.get(k) == v for k, v in new_eds.items()):
            # same routing and model: noise / photon rate change immediately, even mid-stream
            cfg.phantom_noise_v = noise
            cfg.eds_phantom = new_eds
            if self.backend.phantom is not None:
                self.backend.prepare_phantom()
            self._emit("phantom", {"owner": handle.owner, **self.phantom})
            return
        claims = self.claims()
        for name in {new_out, new_sdd if new_kind == "eds" else None} - {None}:
            holder = claims.get(name)
            if holder is not None:
                raise ResourceConflictError(f"{name} is in use by '{holder}' (aux output)")
        self._require_connected()
        if not self._op_lock.acquire(blocking=False):
            raise DeviceBusyError(
                f"the phantom cannot change while a stream runs (owner '{self.active_owner}')"
            )
        try:
            saved = (cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output, cfg.eds_phantom, cfg.phantom_noise_v)
            old_outputs = cfg.phantom_outputs
            cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output = new_out, new_kind, new_sdd
            cfg.eds_phantom, cfg.phantom_noise_v = new_eds, noise
            try:
                cfg.validate()
                self.backend.prepare_phantom()  # build it now, not at the next stream start
            except Exception:
                (cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output, cfg.eds_phantom,
                 cfg.phantom_noise_v) = saved
                raise
            for name in old_outputs - cfg.phantom_outputs:
                # outputs the phantom no longer drives go back to their static value (0 V unless held)
                self._call(self.backend.set_aux_output, name, self._aux_state.get(name, 0.0))
        finally:
            self._op_lock.release()
        self._emit("phantom", {"owner": handle.owner, **self.phantom})

    def _validate_stream(self, rate: float, ai_channels: Sequence[str]) -> None:
        cfg = self.config
        if rate <= 0:
            raise ValueError("rate must be positive")
        ao_lim = cfg.max_ao_rate_for(cfg.stream_ao_channels)
        if rate > ao_lim * (1 + 1e-9):
            raise HALError(
                f"rate {rate:g} S/s exceeds the AO limit {ao_lim:g} S/s ({cfg.stream_ao_channels} AO channels)"
            )
        unknown = [c for c in ai_channels if c not in cfg.ai_channels]
        if unknown:
            raise HALError(f"unknown AI channel(s) {unknown}; configured: {list(cfg.ai_channels)}")
        if ai_channels and rate > cfg.max_ai_rate(len(ai_channels)) * (1 + 1e-9):
            raise HALError(
                f"AI rate {rate:g} S/s × {len(ai_channels)} channel(s) exceeds the device limit "
                f"({cfg.max_ai_rate(len(ai_channels)):g} S/s per channel)"
            )

    def _checked(self, chunks: Iterable[OutputChunk]) -> Iterator[OutputChunk]:
        lim = self.config.ao_limit_v * (1 + 1e-9)
        allowed = set(self.config.timed_lines)
        for c in chunks:
            if len(c) and not (np.all(np.isfinite(c.x)) and np.all(np.isfinite(c.y))):
                raise VoltageRangeError("output stream contains non-finite (NaN/inf) samples")
            if c.max_abs_voltage() > lim:
                raise VoltageRangeError(
                    f"output sample {c.max_abs_voltage():.4g} V exceeds ±{self.config.ao_limit_v} V"
                )
            extra = set(c.markers) - allowed
            if extra:
                raise HALError(f"marker line(s) {sorted(extra)} not configured as timed_lines")
            yield c

    def _run(
        self,
        handle: "DAQHandle",
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str],
        on_ai: AIConsumer | None,
        progress: Callable[[int], None] | None,
        timeout: float,
        latency_s: float | None = None,
        monitor=None,
        ai_range_v: float | None = None,
    ) -> StreamResult:
        self._validate_stream(rate, ai_channels)
        if latency_s is not None and not (np.isfinite(latency_s) and latency_s > 0):
            raise ValueError("latency_s must be a positive number")
        if ai_range_v is not None and not (np.isfinite(ai_range_v) and ai_range_v > 0):
            raise ValueError("ai_range_v must be a positive number")
        if self._active is handle:
            raise DeviceBusyError("handle already running a stream (re-entrant call)")
        self._require_connected()
        if timeout is None or timeout < 0:
            ok = self._op_lock.acquire()
        else:
            ok = self._op_lock.acquire(timeout=timeout) if timeout > 0 else self._op_lock.acquire(False)
        if not ok:
            raise DeviceBusyError(f"device busy (stream owned by '{self.active_owner}')")
        if not self._connected.is_set():  # lost while we waited for the lock
            self._op_lock.release()
            self._require_connected()
        abort = threading.Event()
        with self._state:
            self._active = handle
            self._active_abort = abort
        self._emit("stream_started", {"owner": handle.owner, "rate": rate})
        t0 = time.perf_counter()
        result: StreamResult | None = None
        lost: DeviceLostError | None = None
        try:
            try:
                result = self.backend.run_stream(
                    rate,
                    self._checked(chunks),
                    tuple(ai_channels),
                    on_ai,
                    abort,
                    progress,
                    latency_s=latency_s,
                    monitor=monitor,
                    ai_range_v=ai_range_v,
                )
            except Exception as e:  # noqa: BLE001
                lost = self._as_lost(e)
                if lost is None:
                    raise
                raise lost from e
            return result
        finally:
            safe = False
            if lost is None:
                try:
                    self.backend.safe_state()
                    safe = True
                except Exception as e:  # noqa: BLE001
                    lost = self._as_lost(e)
                    if lost is None:
                        raise
            with self._state:
                self._beam = (BeamState(self.config.park_x_v, self.config.park_y_v, True) if safe
                              else _unknown_beam())
                self._active = None
                self._active_abort = None
            self._op_lock.release()
            if lost is not None:
                self._mark_lost(lost)
            self._emit(
                "stream_finished",
                {
                    "owner": handle.owner,
                    "aborted": bool(result.aborted) if result else True,
                    "elapsed_s": time.perf_counter() - t0,
                    "samples": result.samples_written if result else 0,
                    "device_lost": lost is not None,
                },
            )
            if lost is not None and result is not None:
                raise lost  # the stream itself ended, but the device vanished during cleanup

    def _abort(self, handle: "DAQHandle") -> bool:
        with self._state:
            if self._active is handle and self._active_abort is not None:
                self._active_abort.set()
                return True
        return False


class DAQHandle:
    """Per-module access token for a shared :class:`DAQManager`."""

    def __init__(self, manager: DAQManager, owner: str, uid: int):
        self._mgr = manager
        self.owner = owner
        self.uid = uid
        self.closed = False

    def __repr__(self) -> str:
        return f"<DAQHandle {self.owner!r} #{self.uid}{' closed' if self.closed else ''}>"

    def _check(self) -> None:
        if self.closed:
            raise HALError(f"handle '{self.owner}' is closed")
        if not self._mgr.is_open:
            raise HALError("device is not open")

    # ----------------------------------------------------------------- queries
    @property
    def manager(self) -> DAQManager:
        return self._mgr

    @property
    def config(self):
        return self._mgr.config

    def capabilities(self):
        return self._mgr.capabilities()

    @property
    def is_active(self) -> bool:
        return self._mgr._active is self

    @property
    def connected(self) -> bool:
        return self._mgr.connected

    def wait_connected(self, timeout: float | None = None) -> bool:
        return self._mgr.wait_connected(timeout)

    @property
    def claimed(self) -> set[str]:
        with self._mgr._state:
            return {r for r, o in self._mgr._claims.items() if o is self}

    # ----------------------------------------------------------------- claims
    def claim(self, *resources: str) -> "DAQHandle":
        self._check()
        self._mgr._claim(self, resources)
        return self

    def release(self, *resources: str) -> None:
        self._mgr._release(self, resources if resources else None)

    # -------------------------------------------------------------- static I/O
    def park(self, x_v: float | None = None, y_v: float | None = None, blank: bool = True) -> None:
        """Move the beam statically (default: configured park position, blanked)."""
        self._check()
        cfg = self._mgr.config
        self._mgr._static(
            self, cfg.park_x_v if x_v is None else x_v, cfg.park_y_v if y_v is None else y_v, blank
        )

    def set_blank(self, blank: bool) -> None:
        self._check()
        st = self._mgr.beam_state
        if not (np.isfinite(st.x_v) and np.isfinite(st.y_v)):
            cfg = self._mgr.config
            st = BeamState(cfg.park_x_v, cfg.park_y_v, st.blank)
        self._mgr._static(self, st.x_v, st.y_v, blank)

    def set_line(self, name: str, value: bool) -> None:
        self._check()
        self._mgr._set_line(self, name, value)

    # -------------------------------------------------------- aux analog outputs
    def set_aux_output(self, name: str, volts: float) -> None:
        """Set auxiliary analog output ``"aux0"`` / ``"aux1"`` (static, claimed by this handle).

        On boards without the matching AO channel the output is a dummy: the
        value is accepted and recorded but has no hardware effect.  Not allowed
        while a stream runs; released outputs return to 0 V.
        """
        self._check()
        self._mgr._set_aux(self, name, float(volts))

    @property
    def aux_outputs(self) -> dict:
        """{name: (physical AO channel or None if dummy, value in V)}."""
        return self._mgr.aux_state

    # ------------------------------------------------------------ AI input mode
    @property
    def ai_terminal(self) -> str:
        return self._mgr.config.ai_terminal

    def set_ai_terminal(self, mode: str) -> None:
        """Video input mode: ``"DIFF"`` (AI+ and AI− wired), ``"RSE"`` (AI+ vs AI GND), …

        A property of the wiring, so it applies to every module's following
        streams; refused while a stream runs.
        """
        self._check()
        self._mgr._set_ai_terminal(self, mode)

    # ------------------------------------------------------------ phantom specimen
    @property
    def phantom(self) -> dict:
        """Current phantom routing: {"output", "physical", "noise_v"}."""
        return self._mgr.phantom

    def set_phantom(self, output=_KEEP, noise_v: float | None = None, *, kind: str | None = None,
                    sdd_output=_KEEP, eds: dict | None = None) -> None:
        """Route the phantom specimen's image to aux output *output* (``None`` = off).

        *kind*: ``"chart"`` (test chart) or ``"eds"`` (multi-phase EDS specimen,
        whose reset-type SDD preamp signal goes to *sdd_output*, e.g. ``"aux1"``);
        *eds*: EDS phantom parameters to change (``cps``, ``beam_kv``,
        ``mv_per_kev``, ``noise_mv``, ``leak_v_per_s``).  Omitted arguments keep
        their current value.

        Changing the routing or the model is refused while a stream runs (the AO
        task layout changes, and with it the maximum sample clock); *noise_v*
        and ``eds["cps"]`` (≈ beam current) change at any time.  An output
        claimed by another handle cannot be used.  On a dummy output the
        phantom has no effect.
        """
        self._check()
        self._mgr._set_phantom(self, output, noise_v, kind=kind, sdd_output=sdd_output, eds=eds)

    # --------------------------------------------------------- external scan
    @property
    def has_external_control(self) -> bool:
        return self._mgr.has_external_control

    @property
    def external_control(self) -> bool | None:
        """True = SEM scans from the DAQ, False = SEM internal scan (None = unknown)."""
        return self._mgr.external_control

    def set_external_control(self, on: bool) -> None:
        """Switch the SEM to external (DAQ) scan or back to its internal scan.

        Claims the ``"external_control"`` resource for this handle (exclusive).
        Not allowed while a stream runs.  The beam is blanked before switching.
        Closing this handle (or releasing the resource) returns the SEM to its
        internal scan.  After a lost connection the HAL re-applies the request
        when the device is back.
        """
        self._check()
        self._mgr._set_external(self, bool(on))

    # ----------------------------------------------------------------- streams
    def run_stream(
        self,
        rate: float,
        chunks: Iterable[OutputChunk],
        ai_channels: Sequence[str] = (),
        on_ai: AIConsumer | None = None,
        progress: Callable[[int], None] | None = None,
        timeout: float = 0.0,
        latency_s: float | None = None,
        monitor=None,
        ai_range_v: float | None = None,
    ) -> StreamResult:
        """Run a hardware-timed stream.

        ``timeout``: seconds to wait for the device if another handle is
        streaming (0 = fail immediately, negative/None = wait forever).
        ``latency_s``: output buffer depth (how far the host runs ahead of
        the beam); None = the configured, conservative default.  ``monitor``:
        a :class:`~easyebl.hal.backend.StreamMonitor` updated while streaming.
        ``ai_range_v``: AI input range for this stream (None = configured).
        Raises :class:`DeviceLostError` if the device is (or goes) away.
        """
        self._check()
        return self._mgr._run(
            self, rate, chunks, ai_channels, on_ai, progress, timeout, latency_s, monitor, ai_range_v
        )

    def abort(self) -> bool:
        """Abort this handle's running stream.  Returns True if one was running."""
        return self._mgr._abort(self)

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if not self.closed:
            self.abort()
            self.release()
            self.closed = True

    def __enter__(self) -> "DAQHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
