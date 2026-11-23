"""Stage HAL module: five-axis position readback, moves and stop, over the shared link.

:class:`StageControl` works in physical units – µm for X, Y, Z and degrees
for R (rotation) and T (tilt) – and converts to motor pulses with the
calibrated scales of the driver (X/Y 25.6 /µm, Z 22.32 /µm, R 1849 /°,
T 3528 /°).

* **Polling** – positions are read in the background at most every
  :attr:`MIN_POLL_S` (20 s; stage queries take ~1 s each and share the port with
  the SEM).  One query per controller (``R3`` reads both motors of X/Y and of
  Z/T), at the lowest priority, so SEM settings always go first.  No polling
  while the stage moves; the moved axes are read once when the motion ends.
* **Moves** – absolute or relative targets for any axes.  ``C0`` starts both
  motors of a controller, so the partner axis is re-targeted at its own
  position (``hold_partners``).  A move returns a Future that resolves when
  every axis has reported completion (or raises :class:`MotionError`).
* **Stop** never waits in the queue: S0 goes out at once (also while a long
  SEM transaction holds the port) and queued moves are cancelled.
* ``#STAGE ON`` (the SEM relays stage lines only then) is sent automatically
  before the first stage request.
* **Hand control** (the default) – the SEM PC drives the stage (its trackball works
  only with ``#STAGE OFF``; it decides about motor power).  Positions can still be
  read: each read switches ``#STAGE ON`` for that request only and ``#STAGE OFF``
  again.  Moves are refused until :meth:`StageControl.engage` (``#STAGE ON``);
  :meth:`StageControl.release` stops the stage and hands it back.

Coordinates are unsigned until the instrument's encoding of negative values is
confirmed (driver ``CoordinateCodec``) – a target below 0 is refused.
"""
from __future__ import annotations

import concurrent.futures
import math
import threading
import time

from .link import POLL, READ, STAGE_CMD, MicroscopeLink
from .s3000 import Axis, MotionError, MotionState, StageRejectedBySem, axis_unit, from_pulses, to_pulses
from .s3000.stage import AxisEvent

__all__ = ["StageControl", "AXES", "MIN_POLL_S"]

AXES = ("X", "Y", "Z", "R", "T")
MIN_POLL_S = 20.0
_CONTROLLER = {"X": 0, "Y": 0, "R": 1, "Z": 2, "T": 2}


def _gather(futures) -> concurrent.futures.Future:
    """One future for several (result: list of results; first exception wins)."""
    out: concurrent.futures.Future = concurrent.futures.Future()
    futures = list(futures)
    left = [len(futures)]
    lock = threading.Lock()

    def done(_):
        with lock:
            left[0] -= 1
            if left[0] or out.done():
                return
        for f in futures:
            if f.cancelled():
                out.cancel()
                return
            if f.exception() is not None:
                out.set_exception(f.exception())
                return
        out.set_result([f.result() for f in futures])

    if not futures:
        out.set_result([])
    for f in futures:
        f.add_done_callback(done)
    return out


class StageControl:
    """The five-axis stage through the shared link (see module docstring)."""

    MIN_POLL_S = MIN_POLL_S

    def __init__(self, link: MicroscopeLink, poll_s: float = 30.0, polling: bool = True, move_timeout_s: float = 600.0,
                 hand_control: bool = True):
        self.link = link
        link.stage = self
        self.move_timeout_s = float(move_timeout_s)
        self.positions: dict = {a: None for a in AXES}  # physical (µm / deg)
        self.pulses: dict = {a: None for a in AXES}
        self.t_read: dict = {a: None for a in AXES}
        self._lock = threading.RLock()
        self._stage_on = False
        self._hand = bool(hand_control)  # the SEM PC has the stage (see release())
        self._moves: list = []  # futures of queued / running moves (cancelled by stop)
        self._stop_gen = 0  # incremented by every stop(): a move being sent at that moment is halted too
        # one halt sequence at a time: S0 lines interleaved on the port (a second STOP, or the re-halt
        # of a move whose start was on the wire) make the SEM drop an acknowledgement
        self._halt_lock = threading.Lock()
        self._poll_s = max(self.MIN_POLL_S, float(poll_s))
        self._polling = bool(polling)
        self._t_next_poll = time.monotonic()  # first read right away
        self.t_last_poll: float | None = None
        self._stop = threading.Event()
        link.session.stage.add_listener(self._on_stage_message)
        self._poller = threading.Thread(target=self._poll_loop, name="easyebl-microscope-stage-poll", daemon=True)
        self._poller.start()

    # ------------------------------------------------------------------ state
    @staticmethod
    def unit(axis: str) -> str:
        return axis_unit(axis)

    def motion_state(self, axis: str) -> str:
        return self.link.session.stage.motion_state(axis).value

    @property
    def moving(self) -> bool:
        return any(s == MotionState.MOVING for s in self.link.session.stage.motion_states().values())

    def snapshot(self) -> dict:
        with self._lock:
            return {f"stage_{a.lower()}": v for a, v in self.positions.items() if v is not None}

    # ------------------------------------------------------------------ polling
    @property
    def poll_interval_s(self) -> float:
        return self._poll_s

    def set_poll_interval(self, seconds: float) -> float:
        """Background read interval; never shorter than :attr:`MIN_POLL_S`.  Returns the value used."""
        self._poll_s = max(self.MIN_POLL_S, float(seconds))
        self._t_next_poll = min(self._t_next_poll, time.monotonic() + self._poll_s)
        return self._poll_s

    @property
    def polling(self) -> bool:
        return self._polling

    def set_polling(self, on: bool) -> None:
        self._polling = bool(on)

    @property
    def next_poll_in_s(self) -> float | None:
        return max(0.0, self._t_next_poll - time.monotonic()) if self._polling else None

    def _poll_loop(self) -> None:
        while not self._stop.wait(0.2):
            if not self._polling or time.monotonic() < self._t_next_poll:
                continue
            if self.moving or not self.link.connected:
                self._t_next_poll = time.monotonic() + 1.0
                continue
            self._t_next_poll = time.monotonic() + self._poll_s
            try:
                self.refresh(priority=POLL)
                self.t_last_poll = time.time()
            except RuntimeError:  # link closed
                return

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ reads
    def _ensure_on(self, session) -> None:
        if not self._stage_on:
            session.sem.stage_control(True)
            self._stage_on = True

    def _stage_call(self, session, fn):
        """Run a stage request; if the SEM is not relaying (``#STAGE`` off), switch it on once.
        Under hand control ``#STAGE OFF`` follows the request (the trackball works again)."""
        hand = self._hand  # as the request started (release() switches off itself)
        self._ensure_on(session)
        try:
            try:
                return fn()
            except StageRejectedBySem:
                self._stage_on = False
                self._ensure_on(session)
                return fn()
        finally:
            if hand:
                self._stage_off(session)

    def _stage_off(self, session) -> None:
        session.sem.stage_control(False)
        self._stage_on = False

    # ------------------------------------------------------------------ hand control
    @property
    def hand_control(self) -> bool:
        """True while the SEM PC has the stage (``#STAGE OFF``): moves are refused."""
        return self._hand

    def release(self) -> concurrent.futures.Future:
        """Hand the stage back to the SEM PC: stop, then ``#STAGE OFF`` (the trackball works)."""
        moving = self.moving
        halt = self.stop() if moving or self._moves else None  # goes out at once
        self._hand = True

        def run(session):
            if halt is not None:
                try:
                    halt.result(15)  # every S0 sent before #STAGE OFF
                except Exception:  # noqa: BLE001 - reported by stop()
                    pass
                session.stage.wait_for_motion(list(AXES), timeout=10)
            self._stage_off(session)
            self.link.events.emit("stage", "stage handed to the SEM PC (#STAGE OFF)", hand=True)
            return True

        return self.link.submit(run, STAGE_CMD, "stage release (hand control)")

    def engage(self) -> concurrent.futures.Future:
        """Back to remote control: ``#STAGE ON``; the next move drives the motors again."""

        def run(session):
            self._hand = False
            self._stage_on = False
            self._ensure_on(session)
            self.link.events.emit("stage", "stage under easyEBL control (#STAGE ON)", hand=False)
            return True

        return self.link.submit(run, STAGE_CMD, "stage engage (remote control)")

    def _read_controller(self, session, u: int) -> dict:
        st = session.stage
        if u == 1:
            r = self._stage_call(session, lambda: st.read_position("R"))
            vals = {"R": r.value}
        else:
            r = self._stage_call(session, lambda: st.read_both(u))
            a0, a1 = (("X", "Y") if u == 0 else ("Z", "T"))
            vals = {a0: st.codec.decode(r.axis0), a1: st.codec.decode(r.axis1)}
        now = time.time()
        with self._lock:
            for a, p in vals.items():
                self.pulses[a] = p
                self.positions[a] = from_pulses(a, p)
                self.t_read[a] = now
            pos = {a: self.positions[a] for a in vals}
        self.link.events.emit("stage", "", positions=pos)
        return pos

    def refresh(self, axes=AXES, priority: int = READ) -> concurrent.futures.Future:
        """Read the positions of *axes* (one query per controller involved)."""
        ctrls = sorted({_CONTROLLER[a.upper()] for a in axes})
        return _gather(self.link.submit(lambda s, u=u: self._read_controller(s, u), priority,
                                        f"stage read (controller {u})", key=("stage-read", u))
                       for u in ctrls)

    # ------------------------------------------------------------------ motion
    def move(self, targets: dict, relative: bool = False) -> concurrent.futures.Future:
        """Move to *targets* {axis: value} in µm (X/Y/Z) or degrees (R/T); ``relative`` adds
        them to the current positions.  The Future resolves with the final positions when
        every axis has stopped normally, else raises :class:`MotionError`."""
        tg = {}
        for a, v in targets.items():
            a = a.upper()
            if a not in AXES:
                raise ValueError(f"unknown axis {a!r} (X, Y, Z, R, T)")
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"{a}: not a number")
            tg[a] = v
        if not tg:
            raise ValueError("no targets")
        if relative and all(v == 0 for v in tg.values()):
            raise ValueError("relative move of zero")
        if self._hand:
            raise RuntimeError("the stage is under hand control (SEM PC): engage() it first")
        outer: concurrent.futures.Future = concurrent.futures.Future()
        desc = ", ".join(f"{a} {'+' if relative and v >= 0 else ''}{v:g} {axis_unit(a)}" for a, v in tg.items())

        def start(session):
            if self._hand:
                raise RuntimeError("the stage was released for hand control")
            gen = self._stop_gen
            st = session.stage
            pulses = {}
            for a, v in tg.items():
                if relative:
                    cur = self._stage_call(session, lambda a=a: st.read_position(a)).value
                    pulses[a] = cur + to_pulses(a, v)
                else:
                    pulses[a] = to_pulses(a, v)
                if pulses[a] < 0:
                    raise ValueError(f"{a}: target {from_pulses(a, pulses[a]):.1f} {axis_unit(a)} is below the "
                                     "stage zero (negative coordinates are not enabled)")
            if self._stop_gen != gen:
                raise concurrent.futures.CancelledError("stopped before the move started")
            self._stage_call(session, lambda: st.move(pulses, hold_partners=True, wait=False))
            moved = sorted({ax.name for ax in Axis if ax.controller in {_CONTROLLER[a] for a in pulses}})
            if self._stop_gen != gen:  # STOP arrived while M0 / C0 were on the wire: halt what C0 started
                with self._halt_lock:  # after STOP's own halts, never interleaved with them
                    for a in moved:
                        st.halt(a)
            self.link.events.emit("stage", f"stage move: {desc}", moving=moved, targets=dict(tg), relative=relative)
            return moved

        fut = self.link.submit(start, STAGE_CMD, f"stage move {desc}")
        # C0 starts both motors of a controller: every axis on an involved controller may move
        outer._axes = tuple(a for a in AXES if _CONTROLLER[a] in {_CONTROLLER[t] for t in tg})
        with self._lock:
            self._moves.append(outer)

        def started(f):
            if f.cancelled():
                outer.cancel()
                return
            if f.exception() is not None:
                outer.set_exception(f.exception())
                self._forget(outer)
                return
            threading.Thread(target=self._wait_motion, args=(f.result(), outer), daemon=True,
                             name="easyebl-stage-wait").start()

        fut.add_done_callback(started)
        outer._start = fut  # the queued request (cancelled by stop())
        return outer

    def _wait_motion(self, axes, outer) -> None:
        st = self.link.session.stage
        states = st.wait_for_motion(axes, timeout=self.move_timeout_s)  # does not hold the port
        try:
            pos = self.refresh(axes, priority=STAGE_CMD).result(60)
        except Exception:  # noqa: BLE001
            pos = None
        bad = {a.name: s.value for a, s in states.items() if s != MotionState.COMPLETED}
        self.link.events.emit("stage", "stage move done" if not bad else
                              "stage move ended: " + ", ".join(f"{a} {s}" for a, s in bad.items()),
                              done=[a.name for a in states], states={a.name: s.value for a, s in states.items()})
        if not outer.done():
            if bad:
                outer.set_exception(MotionError("move did not complete: " +
                                                ", ".join(f"{a}={s}" for a, s in bad.items()), states))
            else:
                outer.set_result({a: self.positions[a] for a in AXES if pos is not None})
        self._forget(outer)

    def _forget(self, fut) -> None:
        with self._lock:
            if fut in self._moves:
                self._moves.remove(fut)

    def _halt_axes(self) -> list[str]:
        """Axes STOP halts: those of the moves in progress and any the controllers report moving;
        every axis when none is known to move (motion started elsewhere, state unknown)."""
        st = self.link.session.stage
        with self._lock:
            busy = {a for m in self._moves for a in getattr(m, "_axes", ())}
        busy |= {ax.name for ax, s in st.motion_states().items()
                 if s in (MotionState.MOVING, MotionState.UNCERTAIN)}
        return [a for a in AXES if a in busy] or list(AXES)

    def stop(self, axis: str | None = None) -> concurrent.futures.Future:
        """Halt one axis or the moving ones *now* (bypasses the queue); queued moves are cancelled,
        and a move being sent right now is halted as soon as it has started."""
        with self._lock:
            self._stop_gen += 1
            moves = list(self._moves)
        for m in moves:
            start = getattr(m, "_start", None)
            if start is not None and start.cancel():
                m.cancel()
                self._forget(m)
        out: concurrent.futures.Future = concurrent.futures.Future()

        def run():
            try:
                st = self.link.session.stage
                if self._hand and not self._stage_on:  # the SEM PC has the stage: S0 would be refused
                    out.set_result(False)
                    return
                problems = []
                with self._halt_lock:
                    for a in ([axis.upper()] if axis is not None else self._halt_axes()):
                        try:
                            st.halt(a)
                        except Exception as exc:  # noqa: BLE001 - keep going: every axis gets its S0
                            problems.append(f"{a}: {exc}")
                if problems:
                    raise MotionError("halt: " + "; ".join(problems))
                out.set_result(True)
            except BaseException as e:  # noqa: BLE001
                out.set_exception(e)
                self.link.events.emit("error", f"stage stop: {e}")

        threading.Thread(target=run, name="easyebl-stage-stop", daemon=True).start()
        self.link.events.emit("stage", "STOP " + (axis or "all axes"), stop=axis or "all")
        return out

    def acknowledge(self) -> None:
        """Return finished / halted / uncertain axes to idle (after checking the stage)."""
        st = self.link.session.stage
        for a in AXES:
            if st.motion_state(a) != MotionState.MOVING:
                st.acknowledge_motion(a)

    # ------------------------------------------------------------------ async events
    def _on_stage_message(self, msg) -> None:
        if isinstance(msg, AxisEvent):
            try:
                ax = Axis.from_address(msg.controller, msg.motor).name
            except Exception:  # noqa: BLE001
                ax = f"{msg.controller}/{msg.motor}"
            self.link.events.emit("stage_event", f"stage {ax}: {msg.meaning}", axis=ax, code=msg.code)
