"""Stage HAL module: five-axis position readback, moves and stop, over the shared link.

:class:`StageControl` works in physical units – µm for X, Y, Z and degrees
for R (rotation) and T (tilt) – and converts to motor pulses with the
calibrated scales of the driver (X/Y 25.6 /µm, Z 22.32 /µm, R 1849 /°,
T 3528 /°).

* **The SEM PC has the stage** between requests (``#STAGE OFF``: its trackball works and
  it decides about motor power).  Every stage request – a position read or a move –
  switches ``#STAGE ON`` immediately before and ``#STAGE OFF`` right after (also when
  it fails or is stopped).
* **A move is one exclusive job** on the link: ``#STAGE ON``, the start positions,
  every leg (the one-way approach included), the wait until all axes have stopped,
  the final positions, ``S2`` (motors free – the worm gears hold the stage) for the
  moved controllers, ``#STAGE OFF``.  The port carries nothing else meanwhile:
  SEM settings sent during a move wait in the queue and go out after it, the
  background reads pause (:attr:`MicroscopeLink.exclusive`), and a second move is
  refused (:class:`StageBusy`).  Only STOP gets through.
* **Polling** – each controller (``R3`` reads both motors of X/Y and of Z/T; R
  alone) is a task of the link's :class:`~easyebl.microscope.poll.Poller`, next to
  the SEM readbacks: X/Y and Z/T at level 3, R at level 4 (:data:`POLL_LEVELS`;
  at least every 15 s).  Lowest priority, so SEM settings always go first.
* **Moves** – absolute or relative targets for any axes.  ``C0`` starts both
  motors of a controller, so the partner axis is re-targeted at its own
  position (``hold_partners``).  A move returns a Future that resolves with the
  final positions when every axis has reported completion (or raises
  :class:`MotionError`).  ``move_started`` / ``move_done`` events go out on
  ``link.events`` (kind ``"stage"``); :attr:`StageControl.busy` is True in between.
* **One-way approach (X / Y)** – against backlash every X / Y move ends travelling
  in + for at least :attr:`StageControl.backlash_um` (default 50 µm): a move that
  would end going − (or + by less) first goes to *target − backlash* (not below 0),
  then to the target.  Z, R and T move directly; ``approach=False`` skips it.
* **Settling** – the controllers report STOPPED before the stage is really there: X / Y
  keep correcting on their encoders for several seconds.  So after the last leg the move
  keeps reading the commanded axes until each reads within :attr:`StageControl.settle_tol`
  of its setpoint (X / Y / Z 0.5 µm, R / T 0.01°) and only then resolves;
  :attr:`StageControl.settling` holds the offsets meanwhile.  X / Y get up to
  :attr:`StageControl.settle_timeout_s`; Z, R and T do not correct, so one that is off and
  no longer changes fails at once.  Either way the Future raises :class:`MotionError`.
* **Stop** never waits in the queue: S0 goes out at once (also during the move job)
  and a queued move is cancelled.

Coordinates are unsigned until the instrument's encoding of negative values is
confirmed (driver ``CoordinateCodec``) – a target below 0 is refused.
"""
from __future__ import annotations

import concurrent.futures
import math
import threading
import time

from .link import READ, STAGE_CMD, MicroscopeLink
from .s3000 import Axis, MotionError, MotionState, StageRejectedBySem, axis_unit, from_pulses, to_pulses
from .s3000.errors import CommandFailed
from .s3000.stage import AxisEvent

__all__ = ["StageControl", "StageBusy", "AXES", "POLL_LEVELS"]

#: background-read priority levels per controller (easyebl.microscope.poll); T comes with Z
POLL_LEVELS = {"stage X/Y": 3, "stage Z/T": 3, "stage R": 4}
#: X / Y always end a move travelling in + by at least this much (backlash clearing)
DEFAULT_BACKLASH_UM = 50.0
APPROACH_AXES = ("X", "Y")
#: a move ends when each commanded axis reads within this of its setpoint (µm / deg)
DEFAULT_SETTLE_TOL = {"X": 0.5, "Y": 0.5, "Z": 0.5, "R": 0.01, "T": 0.01}
#: axes that keep correcting on their encoders after reporting STOPPED
CORRECTING_AXES = ("X", "Y")

AXES = ("X", "Y", "Z", "R", "T")
_CONTROLLER = {"X": 0, "Y": 0, "R": 1, "Z": 2, "T": 2}


class StageBusy(RuntimeError):
    """A stage move was requested while another one is in progress."""


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


    def __init__(self, link: MicroscopeLink, polling: bool = True, move_timeout_s: float = 600.0,
                 backlash_um: float = DEFAULT_BACKLASH_UM, settle_timeout_s: float = 30.0,
                 settle_poll_s: float = 0.2):
        self.link = link
        link.stage = self
        #: one-way approach distance for X and Y (µm; 0: off), see the module docstring
        self.backlash_um = float(backlash_um)
        self.move_timeout_s = float(move_timeout_s)
        #: settling (module docstring): tolerance per axis, the X / Y wait limit, the pause between reads
        self.settle_tol: dict = dict(DEFAULT_SETTLE_TOL)
        self.settle_timeout_s = float(settle_timeout_s)
        self.settle_poll_s = float(settle_poll_s)
        #: {axis: read − setpoint} while a move waits for the axes to settle, else None
        self.settling: dict | None = None
        self.positions: dict = {a: None for a in AXES}  # physical (µm / deg)
        self.pulses: dict = {a: None for a in AXES}
        self.t_read: dict = {a: None for a in AXES}
        self._lock = threading.RLock()
        self._stage_on = False
        self._in_move = False  # the move job holds #STAGE ON (reads inside it leave it on)
        self._busy = False  # a move was accepted and has not ended
        self._moves: list = []  # futures of queued / running moves (cancelled by stop)
        self._stop_gen = 0  # incremented by every stop(): a move being sent at that moment is halted too
        # one halt sequence at a time: S0 lines interleaved on the port (a second STOP, or the re-halt
        # of a move whose start was on the wire) make the SEM drop an acknowledgement
        self._halt_lock = threading.Lock()
        self._polling = bool(polling)
        link.session.stage.add_listener(self._on_stage_message)
        for u, name in ((0, "stage X/Y"), (2, "stage Z/T"), (1, "stage R")):
            link.poller.add(name, lambda s, u=u: self._read_controller(s, u), level=POLL_LEVELS[name], owner=self,
                            key=("stage-read", u), what=f"stage read (controller {u})",
                            enabled=lambda: self._polling and not self._busy, cost_s=1.1)

    # ------------------------------------------------------------------ state
    @staticmethod
    def unit(axis: str) -> str:
        return axis_unit(axis)

    def motion_state(self, axis: str) -> str:
        return self.link.session.stage.motion_state(axis).value

    @property
    def moving(self) -> bool:
        return any(s == MotionState.MOVING for s in self.link.session.stage.motion_states().values())

    @property
    def busy(self) -> bool:
        """True from an accepted :meth:`move` until it has ended (the port is reserved for it)."""
        return self._busy

    def snapshot(self) -> dict:
        with self._lock:
            return {f"stage_{a.lower()}": v for a, v in self.positions.items() if v is not None}

    # ------------------------------------------------------------------ polling
    @property
    def polling(self) -> bool:
        return self._polling

    def set_polling(self, on: bool) -> None:
        self._polling = bool(on)

    def poll_period_s(self, name: str = "stage X/Y") -> float | None:
        """Measured time between background reads of one controller (None: not known yet)."""
        task = self.link.poller.task(name)
        return task.period if task is not None else None

    def close(self) -> None:
        self.link.poller.remove(self)

    # ------------------------------------------------------------------ reads
    def _ensure_on(self, session) -> None:
        if not self._stage_on:
            session.sem.stage_control(True)
            self._stage_on = True

    def _stage_call(self, session, fn):
        """Run a stage request between ``#STAGE ON`` and ``#STAGE OFF`` (inside the move job the
        relay stays on until the job ends).  If the SEM refused it as not relaying, switch on again once."""
        self._ensure_on(session)
        try:
            try:
                return fn()
            except StageRejectedBySem:
                self._stage_on = False
                self._ensure_on(session)
                return fn()
        finally:
            if not self._in_move:
                self._stage_off(session)

    def _stage_off(self, session) -> None:
        was_on = self._stage_on
        try:
            session.sem.stage_control(False)
        except CommandFailed as e:
            # the S-3000 answers ``#STAGE OFF`` with E6 when the relay is already off: nothing to do
            if was_on or getattr(e.reply, "result", None) != "E6":
                raise
        self._stage_on = False

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
    def approach_plan(self, targets: dict, current: dict) -> list[dict]:
        """The legs of a move to absolute *targets* from *current* positions: ``[targets]`` or
        ``[pre, targets]`` where *pre* puts X / Y at ``target − backlash`` (clamped at 0) for those
        whose final travel would not be at least ``+backlash``."""
        b = self.backlash_um
        pre = {}
        for a in APPROACH_AXES:
            if a not in targets or b <= 0:
                continue
            cur = current.get(a)
            t = float(targets[a])
            if cur is None or t - float(cur) < b - 1e-6:
                p = max(0.0, t - b)
                if cur is None or abs(p - float(cur)) > 1e-6:
                    pre[a] = p
        return [pre, dict(targets)] if pre else [dict(targets)]

    def move(self, targets: dict, relative: bool = False, approach: bool = True) -> concurrent.futures.Future:
        """Move to *targets* {axis: value} in µm (X/Y/Z) or degrees (R/T); ``relative`` adds
        them to the current positions.  X / Y end with the one-way approach (module docstring)
        unless *approach* is False.  One exclusive link job from ``#STAGE ON`` to ``#STAGE OFF``;
        raises :class:`StageBusy` while another move is in progress.  The Future resolves with
        the final positions when every axis has stopped normally, else raises :class:`MotionError`."""
        tg = self._check_targets(targets, relative)
        with self._lock:
            if self._busy:
                raise StageBusy("a stage move is in progress: wait for it to end (or STOP)")
            self._busy = True
            gen = self._stop_gen
        desc = ", ".join(f"{a} {'+' if relative and v >= 0 else ''}{v:g} {axis_unit(a)}" for a, v in tg.items())
        # C0 starts both motors of a controller: every axis on an involved controller may move
        axes = tuple(a for a in AXES if _CONTROLLER[a] in {_CONTROLLER[t] for t in tg})
        self.link.events.emit("stage", f"stage move: {desc}", move_started=True, targets=dict(tg),
                              relative=relative)

        def job(session):
            self.link.exclusive = f"stage move {desc}"
            self._in_move = True
            try:
                return self._run_move(session, tg, relative, approach, gen)
            finally:
                self._in_move = False
                if self._stage_on:  # S2: motors of the moved controllers free (the worm gears hold them)
                    for u in sorted({_CONTROLLER[a] for a in tg}):
                        try:
                            session.stage.motor_free(u, confirm=True)
                        except Exception as e:  # noqa: BLE001 - reported; #STAGE OFF still follows
                            self.link.events.emit("error", f"stage: S2 (motor free) controller {u}: {e}")
                try:
                    self._stage_off(session)  # the SEM PC has the stage again
                except Exception as e:  # noqa: BLE001 - reported; the move result stands
                    self.link.events.emit("error", f"stage: #STAGE OFF after the move: {e}")
                self.link.exclusive = None

        try:
            fut = self.link.submit(job, STAGE_CMD, f"stage move {desc}")
        except BaseException:
            self._busy = False
            raise
        fut._axes = axes
        with self._lock:
            self._moves.append(fut)

        def ended(f):
            self._forget(f)
            self._busy = False
            if f.cancelled():
                msg, ok = "stage move cancelled", False
            elif f.exception() is not None:
                msg, ok = f"stage move ended: {f.exception()}", False
            else:
                msg, ok = "stage move done", True
            self.link.events.emit("stage", msg, move_done=True, ok=ok, done=list(axes))

        fut.add_done_callback(ended)
        return fut

    @staticmethod
    def _check_targets(targets: dict, relative: bool) -> dict:
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
        return tg

    def _run_move(self, session, tg: dict, relative: bool, approach: bool, gen: int) -> dict:
        """The move job (on the link worker): start positions, legs, final positions."""
        st = session.stage
        self._ensure_on(session)
        need = [a for a in tg if relative or (approach and a in APPROACH_AXES and self.backlash_um > 0)]
        for u in sorted({_CONTROLLER[a] for a in need}):
            self._read_controller(session, u)
        cur = {a: self.positions.get(a) for a in need}
        if relative and any(cur.get(a) is None for a in tg):
            raise RuntimeError("relative move: the start position is not known")
        absolute = {a: (cur[a] + v if relative else v) for a, v in tg.items()}
        legs = self.approach_plan(absolute, cur) if approach else [absolute]
        for a, v in absolute.items():  # before anything moves
            if to_pulses(a, v) < 0:
                raise ValueError(f"{a}: target {v:.1f} {axis_unit(a)} is below the "
                                 "stage zero (negative coordinates are not enabled)")
        moved_all: set = set()
        for leg in legs:
            if self._stop_gen != gen:
                raise concurrent.futures.CancelledError("stopped")
            pulses = {a: to_pulses(a, v) for a, v in leg.items()}
            self._stage_call(session, lambda: st.move(pulses, hold_partners=True, wait=False))
            moved = sorted({ax.name for ax in Axis if ax.controller in {_CONTROLLER[a] for a in pulses}})
            moved_all |= set(moved)
            if self._stop_gen != gen:  # STOP arrived while M0 / C0 were on the wire: halt what C0 started
                with self._halt_lock:  # after STOP's own halts, never interleaved with them
                    for a in moved:
                        st.halt(a)
            self.link.events.emit("stage", "", moving=moved, targets=dict(leg))
            states = st.wait_for_motion(moved, timeout=self.move_timeout_s)
            bad = {a.name: s.value for a, s in states.items() if s != MotionState.COMPLETED}
            if bad:
                for u in sorted({_CONTROLLER[a] for a in moved}):
                    try:
                        self._read_controller(session, u)
                    except Exception:  # noqa: BLE001
                        pass
                raise MotionError("move did not complete: " + ", ".join(f"{a}={s}" for a, s in bad.items()),
                                  states)
        for u in sorted({_CONTROLLER[a] for a in moved_all}):
            self._read_controller(session, u)
        try:
            self._settle(session, absolute, gen)
        finally:
            self.settling = None
        return {a: self.positions[a] for a in AXES}

    def _settle(self, session, setpoints: dict, gen: int) -> None:
        """Read the commanded axes until each is within :attr:`settle_tol` of its setpoint (the
        positions were just read).  X / Y may take up to :attr:`settle_timeout_s`; Z / R / T do not
        correct, so one that is off and unchanged since the previous read fails at once."""
        t0 = time.monotonic()
        ctrls = sorted({_CONTROLLER[a] for a in setpoints})
        prev: dict = {}
        while True:
            with self._lock:
                pos = {a: self.positions[a] for a in setpoints}
            if any(v is None for v in pos.values()):
                raise MotionError("settling: the position could not be read")
            off = {a: pos[a] - v for a, v in setpoints.items()}
            bad = {a: d for a, d in off.items() if abs(d) > self.settle_tol.get(a, 0.0)}
            if not bad:
                return
            self.settling = off
            text = ", ".join(f"{a} {d:+.3g} {axis_unit(a)}" for a, d in bad.items())
            self.link.events.emit("stage", f"stage settling: {text} off", settling=dict(off))
            # one pulse of change counts as "still moving"
            stuck = [a for a in bad if a not in CORRECTING_AXES and a in prev
                     and abs(pos[a] - prev[a]) < 1.5 / abs(to_pulses(a, 1.0) or 1)]
            if stuck:
                raise MotionError("stage did not reach the setpoint: " + ", ".join(
                    f"{a} {bad[a]:+.3g} {axis_unit(a)} off" for a in stuck))
            if time.monotonic() - t0 > self.settle_timeout_s:
                raise MotionError(f"stage did not settle within {self.settle_timeout_s:g} s: {text} off")
            if self._stop_gen != gen:
                raise concurrent.futures.CancelledError("stopped")
            prev = pos
            time.sleep(self.settle_poll_s)
            if self._stop_gen != gen:
                raise concurrent.futures.CancelledError("stopped")
            for u in ctrls:
                self._read_controller(session, u)

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
            if m.cancel():  # still queued: it never starts
                self._forget(m)
        out: concurrent.futures.Future = concurrent.futures.Future()

        def run():
            try:
                st = self.link.session.stage
                if not self._stage_on:  # #STAGE OFF: no easyEBL motion, and S0 would be refused
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
