"""Running an exposure schedule (no Qt): :class:`ScheduleRunner` over a :class:`LithoHardware`.

Per step (on the calling thread; the litho window runs it on a worker thread):

* **Column setting** – recall the beam preset (unless kept as is), set the magnification
  (the field follows the global field calibration), take the nominal current.  The global
  alignment is kept; an Expose field after a magnification change without a Global Align
  is logged as a warning.
* **Field correction** – not implemented: skipped.
* **Expose field**
  1. stage → the field centre, corrected by the last Global Align if the step asks for it;
     the stage does not move again until this field is exposed;
  2. transform ``bcs = A·wcs + t`` with ``A`` of the last Global Align (identity before
     any) and ``t = 0``;
  3. Global Align: every marker's BCS position is predicted, measured by the hardware (the
     litho window: zoomed SEM scan, the user clicks), and A, t are fitted (2 markers:
     similarity, 3–4: affine); this becomes the global alignment for later fields;
  4. Local XY: the same with one marker, ``t`` only;
  5. the pattern (unless none) is checked against the field and exposed with
     ``dwell × dose factor`` per point.

:class:`SimulatedHardware` has a hidden truth (sample rotation / stretch, stage errors) and
tells where each exposure really landed.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from ..core.transforms import AffineTransform
from . import dose as dosemod
from .compiler import VectorScanSettings
from .coords import FieldTransform, GlobalAlignment, StageAxes, check_field, fit_global, fit_local
from .schedule import ColumnSetting, ExposeField, FieldCorrection, PatternLibrary, Schedule, field_target
from .writer import ExposureParameters

__all__ = ["LithoHardware", "SimulatedHardware", "ScheduleRunner", "MarkRequest", "RunAborted", "STEP_STATES",
           "params_for_dwell"]

STEP_STATES = ("pending", "running", "done", "skipped", "failed", "aborted")
_REF_CURRENT_A = 1e-12  # dwell → dose bookkeeping for the writer (any value: the dwell is what counts)


class RunAborted(RuntimeError):
    pass


@dataclass
class MarkRequest:
    """One marker to measure: nominal stage position, WCS, predicted BCS, zoomed view width."""

    label: str
    nominal_scs: tuple
    wcs: tuple
    predicted_bcs: tuple
    view_um: float


def params_for_dwell(dwell_s: float, step_um: float, settings=None) -> ExposureParameters:
    """Exposure parameters giving *dwell_s* per point (areas and lines at *step_um*, dots) × dose factor."""
    i = _REF_CURRENT_A
    p = ExposureParameters(beam_current_a=i, area_dose_uC_cm2=dosemod.area_dose_uC_cm2(dwell_s, step_um, i),
                           line_dose_pC_cm=dosemod.line_dose_pC_cm(dwell_s, step_um, i),
                           dot_dose_fC=dosemod.dot_dose_fC(dwell_s, i), area_step_um=step_um, line_step_um=step_um)
    if settings is not None:
        p.fill_direction, p.serpentine, p.order = settings.fill_direction, settings.serpentine, settings.order
    return p


class LithoHardware:
    """What the runner needs from the instrument.  Blocking calls (the runner has its own thread)."""

    def stage_position(self) -> tuple | None:
        return None

    def move_stage(self, x: float, y: float) -> None:
        raise NotImplementedError

    def set_magnification(self, mag: float) -> float:
        """Set the column magnification; return the new field (µm)."""
        raise NotImplementedError

    def field_um(self) -> float:
        raise NotImplementedError

    def field_at_1x_um(self) -> float:
        raise NotImplementedError

    def preset_names(self) -> list | None:
        return None

    def apply_preset(self, name: str) -> None:
        raise NotImplementedError

    def measure_marks(self, requests: list) -> list | None:
        """Measured BCS position of each marker (µm), or None to abort."""
        raise NotImplementedError

    def ask(self, text: str) -> bool:
        return False

    def expose(self, pattern, params: ExposureParameters, transform: AffineTransform,
               settings: VectorScanSettings) -> object:
        raise NotImplementedError

    def abort(self) -> None:
        pass


# ====================================================================== simulated
class SimulatedHardware(LithoHardware):
    """Hardware stand-in with a hidden truth.

    A feature entered at nominal stage position ``p`` really sits at ``c + M·S·M·(p − c)``: the
    sample rotated / stretched by *sample_linear* ``S`` (beam frame; ``M``: the stage axes) about
    the *pivot* ``c`` – where the positions were taken from the stage, so they are right there and
    off by more the farther away.  Each stage move lands with a random error of *stage_noise_um*.
    With the stage at ``s`` a sample point at true stage position ``q`` is seen at
    ``bcs = M·(q − s)``.
    """

    def __init__(self, sample_linear=None, axes: StageAxes | None = None, field_at_1x_um: float = 100_000.0,
                 magnification: float = 1000.0, stage_noise_um: float = 0.0, mark_noise_um: float = 0.0,
                 presets=("default",), seed: int = 0, delay_s: float = 0.0, start=(0.0, 0.0), pivot=(0.0, 0.0)):
        self.sample = np.eye(2) if sample_linear is None else np.asarray(sample_linear, float)
        self.pivot = np.asarray(pivot, float)
        self.axes = axes or StageAxes()
        self.k = float(field_at_1x_um)
        self.mag = float(magnification)
        self.stage_noise = float(stage_noise_um)
        self.mark_noise = float(mark_noise_um)
        self.presets = list(presets) if presets is not None else None
        self.rng = np.random.default_rng(seed)
        self.delay_s = float(delay_s)
        self.stage = np.asarray(start, float)
        self.preset = None
        self.exposures: list = []
        self.measurements: list = []
        self.answer = False  # what ask() answers
        self.calls: list = []
        self._abort = threading.Event()

    def _noise(self, s: float) -> np.ndarray:
        return self.rng.normal(0.0, s, 2) if s > 0 else np.zeros(2)

    def true_position(self, nominal_scs) -> np.ndarray:
        m = self.axes.matrix
        return (np.atleast_2d(np.asarray(nominal_scs, float)) - self.pivot) @ (m @ self.sample @ m).T + self.pivot

    def bcs_of(self, true_scs, stage=None) -> np.ndarray:
        s = self.stage if stage is None else np.asarray(stage, float)
        return (np.atleast_2d(true_scs) - s) @ self.axes.matrix.T

    def stage_position(self):
        return (float(self.stage[0]), float(self.stage[1]))

    def move_stage(self, x, y):
        self.calls.append(("move", x, y))
        if self.delay_s:
            time.sleep(self.delay_s)
        self.stage = np.array([x, y], float) + self._noise(self.stage_noise)

    def set_magnification(self, mag):
        from .schedule import MAGNIFICATIONS

        if type(mag) is not int or mag not in MAGNIFICATIONS:  # as strict as the SEM driver
            raise ValueError(f"Magnification {mag!r} is not in the permitted list")
        self.calls.append(("mag", mag))
        self.mag = float(mag)
        return self.field_um()

    def field_um(self):
        return self.k / self.mag

    def field_at_1x_um(self):
        return self.k

    def preset_names(self):
        return self.presets

    def apply_preset(self, name):
        if self.presets is not None and name not in self.presets:
            raise KeyError(f"no beam preset named {name!r}")
        self.calls.append(("preset", name))
        self.preset = name

    def measure_marks(self, requests):
        out = []
        for r in requests:
            m = self.bcs_of(self.true_position(r.nominal_scs))[0] + self._noise(self.mark_noise)
            self.measurements.append((r, m))
            out.append((float(m[0]), float(m[1])))
        return out

    def ask(self, text):
        self.calls.append(("ask", text))
        return self.answer

    def expose(self, pattern, params, transform, settings):
        if self._abort.is_set():
            return {"aborted": True}
        if self.delay_s:
            time.sleep(self.delay_s)
        rec = {"pattern": pattern, "transform": transform, "params": params, "field_um": self.field_um(),
               "stage": self.stage_position(), "aborted": False}
        self.exposures.append(rec)
        return rec

    def placement_error(self, rec, centre) -> float:
        """Max distance (µm) between where an exposure was written and where its WCS really is
        (*centre*: the field's nominal centre)."""
        pts = np.vstack([np.atleast_2d(e.points) for e in rec["pattern"]])
        written = rec["transform"].apply(pts)
        true = self.bcs_of(self.true_position(self.axes.wcs_to_scs(pts, centre)), rec["stage"])
        return float(np.max(np.hypot(*(written - true).T)))

    def abort(self):
        self._abort.set()


# ====================================================================== runner
class ScheduleRunner:
    """Runs a :class:`Schedule` (see module docstring).

    ``on_event(kind, data)`` is called on the runner thread with kinds ``"step"``
    ({index, state, message}), ``"log"`` ({text}), ``"alignment"`` ({glob, transform, target,
    index}), ``"exposed"`` ({index, report}) and ``"finished"`` ({state, message})."""

    def __init__(self, schedule: Schedule, hardware: LithoHardware, axes: StageAxes | None = None,
                 library: PatternLibrary | None = None, glob: GlobalAlignment | None = None,
                 on_event: Callable | None = None):
        self.schedule = schedule
        self.hw = hardware
        self.axes = axes or StageAxes()
        self.library = library or PatternLibrary()
        self.glob = glob.copy() if glob is not None else None
        self.transform = FieldTransform()
        self.on_event = on_event or (lambda kind, data: None)
        self.current_pa = None
        self.field = None
        self.magnification = None
        self.mag_changed = False
        self.states = ["pending"] * len(schedule)
        self.state = "idle"
        self._abort = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    # ---------------------------------------------------------------- control
    def abort(self) -> None:
        self._abort.set()
        self._resume.set()
        self.hw.abort()

    def pause(self, on: bool = True) -> None:
        """Pause before the next step (the running step finishes)."""
        if on:
            self._resume.clear()
        else:
            self._resume.set()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def _check(self) -> None:
        self._resume.wait()
        if self._abort.is_set():
            raise RunAborted("aborted")

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001 - a GUI callback must not stop the run
            pass

    def log(self, text: str) -> None:
        self._emit("log", text=text)

    def _step(self, i: int, state: str, message: str = "") -> None:
        self.states[i] = state
        self._emit("step", index=i, state=state, message=message)

    # ---------------------------------------------------------------- run
    def run(self, start: int = 0) -> str:
        """Run from step *start* (the first Column setting always runs first); returns the final
        state (done / aborted / failed)."""
        self.state = "running"
        msg = ""
        steps = [0] + list(range(max(1, start), len(self.schedule)))
        i = 0
        try:
            for i in steps:
                self._check()
                op = self.schedule[i]
                if not op.enabled and i > 0:
                    self._step(i, "skipped", "disabled")
                    continue
                self._step(i, "running")
                result = self._execute(i, op)
                self._step(i, "skipped" if result == "skipped" else "done", "" if result == "skipped" else result)
            self.state = "done"
            msg = f"schedule finished ({len(steps)} steps)"
        except RunAborted as e:
            self.state = "aborted"
            msg = str(e) or "aborted"
            if self.states[i] == "running":
                self._step(i, "aborted", msg)
        except Exception as e:  # noqa: BLE001 - reported
            self.state = "failed"
            msg = f"step {i + 1}: {type(e).__name__}: {e}"
            self._step(i, "failed", f"{type(e).__name__}: {e}")
        self.log(msg)
        self._emit("finished", state=self.state, message=msg)
        return self.state

    def _execute(self, i: int, op) -> str:
        if isinstance(op, ColumnSetting):
            if op.preset:
                self.hw.apply_preset(op.preset)
            if self.magnification is not None and op.magnification != self.magnification:
                self.mag_changed = True
            if not op.permitted:
                raise ValueError(f"magnification ×{op.magnification:g} is not one the SEM accepts")
            self.magnification = op.magnification
            self.field = float(self.hw.set_magnification(op.magnification))
            if op.current_pa:
                self.current_pa = float(op.current_pa)
            return f"×{op.magnification:g}, field {self.field:.4g} µm, beam {op.preset or 'kept as is'}"
        if isinstance(op, FieldCorrection):
            self.log(f"step {i + 1}: field correction is not implemented yet – skipped")
            return "skipped"
        if isinstance(op, ExposeField):
            return self._expose_field(i, op)
        raise TypeError(f"unknown operation {type(op).__name__}")

    def _measure(self, i: int, op: ExposeField, marks: list, kind: str) -> tuple:
        """Predict and measure markers [(name, wcs)] → (wcs (N, 2), measured bcs (N, 2))."""
        wcs = np.array([m for _, m in marks], float).reshape(-1, 2)
        nominal = op.nominal_scs(wcs, self.axes)
        pred = self.transform.to_bcs(wcs)
        reqs = [MarkRequest(f"step {i + 1} {name}", tuple(map(float, s)), tuple(map(float, w)),
                            (float(p[0]), float(p[1])), op.view_um) for (name, _), s, w, p in zip(marks, nominal, wcs,
                                                                                                    pred)]
        got = self.hw.measure_marks(reqs)
        if got is None:
            raise RunAborted(f"{kind} cancelled")
        self._check()
        got = np.asarray(got, float).reshape(len(marks), 2)
        dev = ", ".join(f"({d[0]:+.2f}, {d[1]:+.2f})" for d in got - pred)
        self.log(f"step {i + 1}: {kind} – markers off the prediction by {dev} µm")
        return wcs, got

    def _expose_field(self, i: int, op: ExposeField) -> str:
        target = field_target(op, self.glob, self.axes)
        corrected = op.correct_centre and self.glob is not None
        self.log(f"step {i + 1}: stage → ({target[0]:.2f}, {target[1]:.2f}) µm"
                 + (f" (centre ({op.centre[0]:.2f}, {op.centre[1]:.2f}) corrected by the global alignment)"
                    if corrected else ""))
        self.hw.move_stage(float(target[0]), float(target[1]))
        self.transform = FieldTransform(self.glob.linear if self.glob is not None else np.eye(2), (0.0, 0.0))
        if self.mag_changed and not op.global_align:
            self.log(f"WARNING step {i + 1}: the magnification changed since the last Global Align and this field "
                     "has none")
        if op.global_align:
            if not 2 <= len(op.markers) <= 4:
                raise ValueError(f"Global Align needs 2–4 markers (has {len(op.markers)})")
            marks = [(f"global marker {k + 1}", m) for k, m in enumerate(op.markers)]
            wcs, got = self._measure(i, op, marks, "Global Align")
            self.transform, res = fit_global(wcs, got)
            self.glob = GlobalAlignment(self.transform.linear, op.centre, target, self.transform.shift,
                                        self.magnification, i)
            self.mag_changed = False
            if len(marks) > 2:
                self.log(f"step {i + 1}: Global Align residuals rms "
                         f"{float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))):.3f} µm")
            self.log(f"step {i + 1}: global alignment {self.glob.describe()}")
        if op.local_xy:
            wcs, got = self._measure(i, op, [("local XY marker", op.local_marker)], "Local XY")
            self.transform = fit_local(wcs[0], got[0], self.transform)
        self._emit("alignment", glob=self.glob.copy() if self.glob is not None else None,
                   transform=self.transform.copy(), target=(float(target[0]), float(target[1])), index=i)
        if op.pattern.kind == "none":
            return "aligned (no pattern)"
        return self._write(i, op)

    def _write(self, i: int, op: ExposeField) -> str:
        pat = self.library.get(op.pattern)
        t = self.transform.wcs_to_bcs
        fc = check_field(pat, t, self.field)
        if not fc.ok:
            msg = f"step {i + 1} ({op.summary()}): {fc.message()}"
            self.log("WARNING " + msg)
            policy = self.schedule.settings.on_out_of_field
            if policy == "abort":
                raise RunAborted("pattern outside the field of view")
            if policy == "ask" and not self.hw.ask(msg + "\n\nSkip this field and continue? (No aborts the schedule.)"):
                raise RunAborted("pattern outside the field of view")
            return "skipped"
        params = params_for_dwell(op.dwell_us * 1e-6, op.step_nm * 1e-3, self.schedule.settings)
        settings = VectorScanSettings(**self.schedule.vector_settings)
        self._check()
        rep = self.hw.expose(pat, params, t, settings)
        self._emit("exposed", index=i, report=rep)
        if (isinstance(rep, dict) and rep.get("aborted")) or getattr(rep, "aborted", False):
            raise RunAborted("exposure aborted")
        return f"{len(pat)} element(s) exposed, {self.transform.describe()}"
