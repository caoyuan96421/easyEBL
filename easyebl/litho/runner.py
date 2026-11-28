"""Running an exposure schedule (no Qt): :class:`ScheduleRunner` over a :class:`LithoHardware`.

Per step (on the calling thread; the litho window runs it on a worker thread):

* **Column setting** – recall the beam preset (unless kept as is), take the nominal current; the
  first one also sets the schedule's field size (its magnification; the field follows the global
  field calibration) and the focus to WD1 (the calibration sample's).
* **Beam calibration** – the stage to the calibration sample (X, Y, Z_cal), the focus at WD1,
  zoomed in for the user to confirm the focus by hand (the WD then is WD1 from there on), the
  grid image fitted (:mod:`easyebl.litho.beamcal`), shown for Accept / Discard
  (``review_beam``); or a stored one of the field size (``reuse``).
* **Exposure** – every field of the GDS (the Global Align field first), each:

  1. stage → ``G(c_f)`` (the chip's CCS → SCS map, nominal from the chip origin until a Global
     Align) in X, Y only – Z stays at Z_cal – then the focus to the chip's WD at the field
     centre (WD2, or the plane through WD2x); the read-back ``ŝ`` is what the field is written
     against;
  2. Global Align (its field only): the markers' XCS positions are predicted through the chain
     (:mod:`easyebl.litho.coords`), measured by the hardware (the litho window: zoomed SEM scan,
     the user clicks) twice (1 → 2 → 1 → 2), the passes compared against the alignment
     tolerance (repeat / continue / abort), taken back to SCS and fitted: the chip's new ``G``
     (kept for all later Exposures of the chip);
  3. Local marker (if the field has one): a shift ``δ_f`` (SCS) for this field;
  4. the pattern (FCS) is checked against the field and written through
     ``xcs = beamcal⁻¹(B·(A_g·fcs + G(c_f) − ŝ + δ_f))``, ``B`` the rotation of the field size's
     field calibration (kept per field size, measured outside a schedule:
     :meth:`ScheduleRunner.field_calibrate`).

A run must be at the field size selected in the Microscope window.  A run from a step after the
two fixed ones (Column setting, Beam calibration) skips them: the column must be at the field
size's magnification, the stage at Z_cal, and a beam calibration of the field size in force.
When a run finishes the stage goes back to the calibration sample (Z_cal, WD1).

:class:`SimulatedHardware` has a hidden truth (sample rotation / stretch, the stage → beam
rotation, the beam's scale / shear / pincushion, stage errors) and tells where each exposure
really landed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Callable

import numpy as np

from . import dose as dosemod
from .compiler import VectorScanSettings
from .coords import ChipTransform, FieldMap, StageAxes, check_field, fit_chip, stage_rotation
from .fieldcal import (Condition, FieldCalConfig, FieldCalError, FieldCalResult, measure_field_calibration,
                       shift_pattern)
from .height import Z_MATCH_UM
from .schedule import (FIXED_STEPS, BeamCalibrationStep, ColumnSetting, Exposure, PatternLibrary, Schedule,
                       ScheduleError, assign_markers, calibration_problem, exposure_fields, field_size_of, find_beamcal,
                       position_problem, wd_far_problem)
from .writer import ExposureParameters

__all__ = ["LithoHardware", "SimulatedHardware", "ScheduleRunner", "MarkRequest", "RunAborted", "STEP_STATES",
           "params_for_dwell", "StageError"]

STEP_STATES = ("pending", "running", "done", "skipped", "failed", "aborted")
_REF_CURRENT_A = 1e-12  # dwell → dose bookkeeping for the writer (any value: the dwell is what counts)


class RunAborted(RuntimeError):
    pass


@dataclass
class MarkRequest:
    """One marker to measure: its CCS position, the nominal stage position (from the chip origin),
    where it is predicted in the scan (XCS) and the zoomed view width."""

    label: str
    ccs: tuple
    nominal_scs: tuple
    predicted_xcs: tuple
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

    def stage_axes(self) -> StageAxes:
        """The stage axes against the beam (Hardware → Configure… → Microscope)."""
        from .coords import S3000_AXES

        return S3000_AXES

    def stage_position(self) -> tuple | None:
        return None

    def stage_z(self) -> float | None:
        return None

    def move_stage(self, x: float, y: float, z: float | None = None) -> None:
        raise NotImplementedError

    def read_stage(self) -> tuple | None:
        """The stage position (X, Y) read fresh from the instrument (after a move)."""
        return self.stage_position()

    def magnification(self) -> float | None:
        return None

    def scan_area(self) -> tuple | None:
        """The SEM imaging scan now: (centre (x, y) XCS µm, width µm, height µm, rotation rad) – a
        reduced or shifted scan is a part of the field (None: not known)."""
        return None

    def working_distance_mm(self) -> float | None:
        """The working distance computed from the focus (None: not known)."""
        return None

    def set_working_distance(self, wd_mm: float) -> None:
        """Set the focus to the working distance *wd_mm* (and wait until it is read back)."""
        raise NotImplementedError

    #: the beam preset this program recalled last (None: not known)
    last_preset = None

    def capture_image(self, width_um: float, px: int, frames: int = 1):
        """An image (px × px, rows run down) of a centred scan *width_um* wide, and its pixel size (µm)."""
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
        """Measured XCS position of each marker (µm, scan coordinates), or None to abort."""
        raise NotImplementedError

    def ask(self, text: str) -> bool:
        return False

    def prepare_run(self) -> None:
        """At the start of a run: the SEM in RUN mode (not frozen) and on Slow scan 1 (not set back)."""

    def ask_focus(self, text: str) -> bool:
        """Wait while the user focuses and stigmates by hand (*text* says where); True: continue,
        False: abort.  The column (magnification) is set by the runner before and after."""
        return self.ask(text)

    def ask_alignment(self, text: str) -> str:
        """The Global Align passes disagree (*text*): ``"repeat"``, ``"continue"`` (use them anyway)
        or ``"abort"``."""
        return "continue" if self.ask(text + "\n\nContinue anyway? (No aborts.)") else "abort"

    def auto_focus(self, mode: str = "auto"):
        """Auto focus at the present scan field (:mod:`easyebl.imaging.autotune`; ``mode`` auto /
        fine / coarse).  Returns its :class:`~easyebl.imaging.autotune.AutoTuneResult` (``ok`` False:
        the start values were set again)."""
        raise NotImplementedError

    def auto_stigmate(self):
        """Auto stigmation at the present scan field (see :meth:`auto_focus`)."""
        raise NotImplementedError

    def expose(self, pattern, params: ExposureParameters, transform, settings: VectorScanSettings) -> object:
        """Write *pattern* (FCS) through *transform* (``apply``: FCS → XCS)."""
        raise NotImplementedError

    def abort(self) -> None:
        pass


# ====================================================================== simulated
class _TrueBeam:
    """The simulator's hidden beam: ``bcs = D(L·xcs)``; the exact inverse by iteration."""

    def __init__(self, linear=None, k: float = 0.0):
        self.L = np.eye(2) if linear is None else np.asarray(linear, float)
        self.k = float(k)

    def to_bcs(self, xcs):
        q = np.atleast_2d(np.asarray(xcs, float)) @ self.L.T
        return q * (1.0 + self.k * np.sum(q * q, axis=1))[:, None]

    def to_xcs(self, bcs):
        b = np.atleast_2d(np.asarray(bcs, float))
        q = b.copy()
        for _ in range(30):
            q = b / (1.0 + self.k * np.sum(q * q, axis=1))[:, None]
        return q @ np.linalg.inv(self.L).T


class SimulatedHardware(LithoHardware):
    """Hardware stand-in with a hidden truth.

    * A feature at nominal stage position ``p`` (the chip origin through the stage axes) really sits
      at ``c + M·S·M·(p − c)``: the sample rotated / stretched by *sample_linear* ``S`` (beam frame;
      ``M``: the stage axes) about the *pivot* ``c``.
    * The stage reads back the commanded position (+ *stall_um*); it really stands at the command +
      a random *stage_noise_um* error.
    * With the stage really at ``s`` a sample point at true stage position ``q`` is at
      ``bcs = B·(q − s)`` (*stage_to_beam* ``B``, default ``M``), and the beam reaches ``bcs`` from
      ``xcs`` with ``bcs = D(L·xcs)`` (*beam_linear* ``L``, *beam_k* – what the beam calibration
      measures).
    * :meth:`capture_image` renders a random texture, or a 1 µm grid (*grid_at*: (centre, radius)
      on the stage) – both seen through the true maps.
    """

    def __init__(self, sample_linear=None, axes: StageAxes | None = None, field_at_1x_um: float = 100_000.0,
                 magnification: float = 1000.0, stage_noise_um: float = 0.0, mark_noise_um: float = 0.0,
                 presets=("default",), seed: int = 0, delay_s: float = 0.0, start=(0.0, 0.0), pivot=(0.0, 0.0),
                 stage_to_beam=None, stall_um=(0.0, 0.0), beam_linear=None, beam_k: float = 0.0,
                 grid_at=None, grid_pitch_um: float = 1.0, grid_rotation_rad: float = 0.01,
                 wd_mm: float | None = 15.0, image_noise: float = 0.05, image_contrast: float = 1.0,
                 start_z: float = 15000.0):
        from .coords import S3000_AXES

        self.sample = np.eye(2) if sample_linear is None else np.asarray(sample_linear, float)
        self.pivot = np.asarray(pivot, float)
        self.axes = axes or S3000_AXES
        self.B = self.axes.matrix if stage_to_beam is None else np.asarray(stage_to_beam, float)
        self.beam = _TrueBeam(beam_linear, beam_k)
        self.k = float(field_at_1x_um)
        self.mag = float(magnification)
        self.stage_noise = float(stage_noise_um)
        self.mark_noise = float(mark_noise_um)
        self.stall = np.asarray(stall_um, float)
        self.presets = list(presets) if presets is not None else None
        self.rng = np.random.default_rng(seed)
        self.delay_s = float(delay_s)
        self.stage = np.asarray(start, float)  # where it really is
        self.readback = np.asarray(start, float)  # what it reports
        self.z = float(start_z)
        self.wd = wd_mm
        self.grid_at = grid_at
        self.grid_pitch = float(grid_pitch_um)
        self.grid_rotation = float(grid_rotation_rad)
        self.image_noise = float(image_noise)  # rms noise of the images (texture rms × contrast)
        self.image_contrast = float(image_contrast)
        self.preset = None
        self.last_preset = None
        self.exposures: list = []
        self.measurements: list = []
        self.images = 0
        self.answer = False  # what ask() answers
        self.focus_answer = True  # what ask_focus() answers
        self.refocus_wd = None  # the WD "the user" focuses to in ask_focus (None: unchanged)
        self.alignment_answers: list = []  # what ask_alignment() answers, in turn (then "continue")
        self.scan = None  # what scan_area() reports (None: not known)
        self.calls: list = []
        self._abort = threading.Event()
        self._texture = {}  # field µm -> (tile µm, periodic texture image)
        self._seed = seed

    def stage_axes(self) -> StageAxes:
        return self.axes

    def _noise(self, s: float) -> np.ndarray:
        return self.rng.normal(0.0, s, 2) if s > 0 else np.zeros(2)

    def true_position(self, nominal_scs) -> np.ndarray:
        m = self.axes.matrix
        return (np.atleast_2d(np.asarray(nominal_scs, float)) - self.pivot) @ (m @ self.sample @ m).T + self.pivot

    def bcs_of(self, true_scs, stage=None) -> np.ndarray:
        s = self.stage if stage is None else np.asarray(stage, float)
        return (np.atleast_2d(true_scs) - s) @ self.B.T

    def stage_position(self):
        return (float(self.readback[0]), float(self.readback[1]))

    def stage_z(self):
        return float(self.z)

    def move_stage(self, x, y, z=None):
        self.calls.append(("move", x, y) if z is None else ("move", x, y, z))
        if self.delay_s:
            time.sleep(self.delay_s)
        self.stage = np.array([x, y], float) + self._noise(self.stage_noise)
        self.readback = np.array([x, y], float) + self.stall
        if z is not None:
            self.z = float(z)

    def working_distance_mm(self):
        return self.wd

    def set_working_distance(self, wd_mm):
        self.calls.append(("wd", float(wd_mm)))
        self.wd = float(wd_mm)

    def set_magnification(self, mag):
        from .schedule import MAGNIFICATIONS

        if type(mag) is not int or mag not in MAGNIFICATIONS:  # as strict as the SEM driver
            raise ValueError(f"Magnification {mag!r} is not in the permitted list")
        self.calls.append(("mag", mag))
        self.mag = float(mag)
        return self.field_um()

    def magnification(self):
        return self.mag

    def scan_area(self):
        return self.scan

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
        self.preset = self.last_preset = name

    def measure_marks(self, requests):
        out = []
        for r in requests:
            b = self.bcs_of(self.true_position(r.nominal_scs))
            m = self.beam.to_xcs(b)[0] + self._noise(self.mark_noise)
            self.measurements.append((r, m))
            out.append((float(m[0]), float(m[1])))
        return out

    def _grid_here(self) -> bool:
        if self.grid_at is None:
            return False
        (cx, cy), r = self.grid_at
        return bool(np.hypot(self.stage[0] - cx, self.stage[1] - cy) <= r)

    def capture_image(self, width_um: float, px: int, frames: int = 1):
        """The sample seen through the true maps with the stage really where it is (rows run down): a
        random texture fixed to the sample (periodic, 3 fields wide), or the grid (:attr:`grid_at`)."""
        from scipy import ndimage

        self.images += 1
        n = int(px)
        pix = width_um / n
        u = (np.arange(n) + 0.5) * pix - width_um / 2
        bx, by = np.meshgrid(u, -u)
        xcs = np.column_stack([bx.ravel(), by.ravel()])
        q = self.beam.to_bcs(xcs) @ np.linalg.inv(self.B).T + self.stage
        if self._grid_here():
            c, s = math.cos(self.grid_rotation), math.sin(self.grid_rotation)
            g = (q - np.asarray(self.grid_at[0], float)) @ np.array([[c, s], [-s, c]]).T / self.grid_pitch
            d = np.abs(g - np.round(g)) * self.grid_pitch  # µm to the nearest line, per axis
            w = 0.1  # half line width (µm)
            prof = np.clip(0.5 - (d - w) / pix, 0.0, 1.0)  # anti-aliased line edges
            img = np.maximum(prof[:, 0], prof[:, 1]).reshape(n, n) + self.rng.normal(0.0, 0.05, (n, n))
            return img.astype(np.float32), (pix, pix)
        f = self.field_um()
        if f not in self._texture:
            n_t = 1024
            t = ndimage.gaussian_filter(np.random.default_rng(self._seed + 1).standard_normal((n_t, n_t)), 4.0,
                                        mode="wrap")
            self._texture[f] = (3.0 * f, (t - t.mean()) / t.std())
        tile, tex = self._texture[f]
        k = tex.shape[0] / tile
        img = ndimage.map_coordinates(tex, [-q[:, 1] * k, q[:, 0] * k], order=1, mode="grid-wrap")
        img = self.image_contrast * img.reshape(n, n) + self.rng.normal(0.0, self.image_noise, (n, n))
        return img, (pix, pix)

    def ask(self, text):
        self.calls.append(("ask", text))
        return self.answer

    def prepare_run(self):
        self.calls.append(("prepare",))

    def ask_focus(self, text):
        self.calls.append(("focus", self.mag))
        if self.refocus_wd is not None:
            self.wd = float(self.refocus_wd)
        return self.focus_answer

    def ask_alignment(self, text):
        self.calls.append(("align", text))
        return self.alignment_answers.pop(0) if self.alignment_answers else "continue"

    def expose(self, pattern, params, transform, settings):
        if self._abort.is_set():
            return {"aborted": True}
        if self.delay_s:
            time.sleep(self.delay_s)
        rec = {"pattern": pattern, "transform": transform, "params": params, "field_um": self.field_um(),
               "stage": (float(self.stage[0]), float(self.stage[1])), "aborted": False}
        self.exposures.append(rec)
        return rec

    def placement_error(self, rec, centre_ccs, chip_nominal: ChipTransform) -> float:
        """Max distance (true stage µm) between where an exposure was written and where its FCS really
        is (*centre_ccs*: the field centre, *chip_nominal*: the chip's nominal CCS → SCS map)."""
        pts = np.vstack([np.atleast_2d(e.points) for e in rec["pattern"]])
        xcs = rec["transform"].apply(pts)
        written = self.beam.to_bcs(xcs) @ np.linalg.inv(self.B).T + np.asarray(rec["stage"], float)
        true = self.true_position(chip_nominal.to_scs(pts + np.asarray(centre_ccs, float)))
        return float(np.max(np.hypot(*(written - true).T)))

    def abort(self):
        self._abort.set()


# ====================================================================== runner
class StageError(RuntimeError):
    pass


class ScheduleRunner:
    """Runs a :class:`Schedule` (see module docstring).

    ``on_event(kind, data)`` is called on the runner thread with kinds ``"step"``
    ({index, state, message}), ``"log"`` ({text}), ``"field"`` ({index, k, n, name, state, map}),
    ``"alignment"`` ({chip, transform, index}), ``"calibration_start"`` ({index, n, position,
    condition, field_um}), ``"calibration_image"`` ({k, n, shift, image, reference}),
    ``"calibration"`` ({result}), ``"beamcal_start"`` ({index, position, condition, field_um, pixels}),
    ``"beamcal_image"`` ({image, pixel_um}), ``"beamcal"`` ({result}), ``"focus_request"`` ({index,
    position, magnification}), ``"wd1"`` ({wd_mm, old, z_um}: the WD on the calibration sample,
    focused again), ``"exposed"`` ({index, report}) and ``"finished"`` ({state, message}).

    ``review(result) -> bool`` / ``review_beam(result) -> bool`` (optional, blocking) decide whether
    a field / beam calibration is used: False discards it (a schedule run stops).  They are also
    called with a refused result (``result.problem`` set): True uses it anyway (an override), False
    fails the step.

    *positions*: {name: :class:`~easyebl.litho.settings.StagePosition`} with the calibration sample
    (copied: WD1 focused again reaches the caller through the ``"wd1"`` event);
    *calibration*: the field calibration of the schedule's field size; *field_sizes*: [FieldSize]
    (None: the defaults); *selected*: the field size selected in the Microscope window (a run at
    another one is refused; None: not checked)."""

    def __init__(self, schedule: Schedule, hardware: LithoHardware, axes: StageAxes | None = None,
                 library: PatternLibrary | None = None, chips: dict | None = None,
                 on_event: Callable | None = None, calibration: FieldCalResult | None = None,
                 positions: dict | None = None, stage_tolerance_um: float = 0.5, review: Callable | None = None,
                 beamcals=None, review_beam: Callable | None = None, field_sizes=None, selected: str | None = None):
        self.schedule = schedule
        self.hw = hardware
        self.axes = axes or hardware.stage_axes()
        self.library = library or PatternLibrary()
        self.chips = {k: v.copy() for k, v in (chips or {}).items()}
        self.calibration = calibration
        self.positions = {k: replace(v) for k, v in (positions or {}).items()}
        self.tolerance = float(stage_tolerance_um)
        self.review = review
        self.review_beam = review_beam
        self.beamcals = list(beamcals or [])
        self.field_sizes = field_sizes
        self.selected = selected
        self.on_event = on_event or (lambda kind, data: None)
        self.current_pa = None
        self.field = None
        self.magnification = None
        self.preset = getattr(hardware, "last_preset", None)
        self.states = ["pending"] * len(schedule)
        self.state = "idle"
        self.last_map: FieldMap | None = None
        self._abort = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    # ---------------------------------------------------------------- state
    def _wd(self) -> float | None:
        try:
            v = self.hw.working_distance_mm()
        except Exception:  # noqa: BLE001
            return None
        return None if v is None else float(v)

    @property
    def field_size(self):
        """The schedule's :class:`~easyebl.litho.settings.FieldSize` (None: not chosen / not known)."""
        return field_size_of(self.schedule.field_size, self.field_sizes)

    def _need_field_size(self):
        fs = self.field_size
        if fs is None:
            raise ScheduleError(f"no field size named {self.schedule.field_size!r}" if self.schedule.field_size
                                else "the schedule has no field size (its first Column setting)")
        return fs

    @property
    def condition(self) -> Condition:
        fs = self.field_size
        return Condition(self.magnification, self.preset, self._wd(), fs.name if fs is not None else None)

    @property
    def cal_position(self):
        """The calibration sample (a copy: WD1 focused again is kept here for the run)."""
        return next((p for p in self.positions.values() if p.role == "calibration"), None)

    @property
    def z_cal(self) -> float | None:
        p = self.cal_position
        return None if p is None else p.z_um

    @property
    def wd1(self) -> float | None:
        p = self.cal_position
        return None if p is None else p.wd_mm

    def _need_calibration_sample(self):
        p = self.cal_position
        w = calibration_problem(p)
        if w:
            raise ScheduleError(w)
        return p

    def z_of(self, p) -> float | None:
        """The stage Z of stage position *p*: Z_cal + ΔZ (None: not known)."""
        zc = self.z_cal
        if p.role == "calibration":
            return zc
        return None if zc is None or p.dz is None else zc + p.dz

    def _position(self, name: str, what: str):
        w = position_problem(name, self.positions, what)
        if w:
            raise ScheduleError(w)
        return self.positions[name]

    def beamcal(self, condition: Condition | None = None):
        return find_beamcal(self.beamcals, condition or self.condition)

    def stage_to_beam(self, condition: Condition | None = None) -> tuple[np.ndarray, bool]:
        """B in use (the rotation of the field size's field calibration with the stage mirror;
        nominal ``M``) and whether a field calibration applies."""
        cond = condition or self.condition
        cal = self.calibration
        if cal is not None and cal.valid_for(cond):
            return stage_rotation(cal.stage_to_beam, self.axes), True
        return self.axes.matrix, False

    def chip_map(self, name: str) -> ChipTransform:
        chip = self.schedule.chip(name)
        if chip is None:
            raise ScheduleError(f"no chip named {name!r}")
        return self.chips.get(name) or chip.nominal(self.axes)

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
        if 0 <= i < len(self.states):
            self.states[i] = state
        self._emit("step", index=i, state=state, message=message)

    # ---------------------------------------------------------------- run
    def run(self, start: int = 0, stop: int | None = None) -> str:
        """Run the steps ``start … stop − 1`` (*stop* None: to the end); returns the final state
        (done / aborted / failed).  A *start* within the fixed steps runs from the first step; one
        after them skips them – the column must then be at the field size's magnification, the
        stage at Z_cal, and a beam calibration of the field size in force."""
        self.state = "running"
        msg = ""
        n_fixed = len(FIXED_STEPS)
        start = 0 if start < n_fixed else start
        end = len(self.schedule) if stop is None else min(int(stop), len(self.schedule))
        steps = list(range(start, end))
        i = start
        try:
            fs = self._need_field_size()
            if self.selected is not None and self.selected != fs.name:
                raise ScheduleError(f"the Microscope window is at field size {self.selected or 'none'}, the schedule "
                                    f"at {fs.name}: select {fs.name} there first")
            self._need_calibration_sample()
            self.hw.prepare_run()
            self.log("SEM: RUN mode, Slow scan 1")
            if start:
                self._resume_column()
            for i in steps:
                self._check()
                op = self.schedule[i]
                if not op.enabled and not self.schedule.is_fixed(i):
                    self._step(i, "skipped", "disabled")
                    continue
                self._step(i, "running")
                result = self._execute(i, op)
                self._step(i, "skipped" if result == "skipped" else "done", "" if result == "skipped" else result)
            self.state = "done"
            msg = f"schedule finished ({len(steps)} steps)"
            self._park()
        except RunAborted as e:
            self.state = "aborted"
            msg = str(e) or "aborted"
            if 0 <= i < len(self.states) and self.states[i] == "running":
                self._step(i, "aborted", msg)
        except Exception as e:  # noqa: BLE001 - reported
            self.state = "failed"
            msg = f"step {i + 1}: {type(e).__name__}: {e}"
            if 0 <= i < len(self.states):
                self._step(i, "failed", f"{type(e).__name__}: {e}")
        self.log(msg)
        self._emit("finished", state=self.state, message=msg)
        return self.state

    def _park(self) -> None:
        """After a finished run: the stage to the calibration sample (Z_cal), the focus at WD1.
        A failure is only logged."""
        p = self.cal_position
        if calibration_problem(p):
            self.log("the stage stays: the calibration sample is not recorded")
            return
        try:
            self.move_stage(p.x, p.y, p.z_um)
            self.hw.set_working_distance(p.wd_mm)
            self.log(f"stage parked at {p.name!r}, WD1 {p.wd_mm:g} mm")
        except Exception as e:  # noqa: BLE001 - the run is done
            self.log(f"WARNING the stage could not go to {p.name!r}: {type(e).__name__}: {e}")

    def _check_z_cal(self) -> None:
        """The stage must be at Z_cal (chip heights are WDs at that Z)."""
        z = self.hw.stage_z()
        zc = self.z_cal
        if z is None or zc is None:
            raise ScheduleError("the stage Z or the calibration sample's Z is not known")
        if abs(z - zc) > max(self.tolerance, Z_MATCH_UM):
            raise ScheduleError(f"the stage is at Z {z:.1f} µm, not at the calibration sample's Z {zc:.1f} µm "
                                "(the chip heights are WDs at that Z)")

    def _resume_column(self) -> None:
        """A run after the fixed steps: take the column as it is; it must be at the field size, the
        stage at Z_cal and a beam calibration in force."""
        fs = self._need_field_size()
        mag = self.hw.magnification()
        if mag is None or int(round(mag)) != fs.magnification:
            raise ScheduleError(f"the column is at ×{mag:g}, not the field size's ×{fs.magnification:g}: run the "
                                "schedule from the start" if mag is not None else "the magnification is not known")
        self.magnification = fs.magnification
        self.field = float(self.hw.field_um())
        first = self.schedule[0]
        if first.current_pa:
            self.current_pa = float(first.current_pa)
        if first.preset:
            self.preset = first.preset
        self._check_z_cal()
        cond = self.condition
        bc = self.beamcal(cond)
        if bc is None:
            raise ScheduleError(f"no beam calibration for {cond.describe()}: run the schedule from the start")
        self.log(f"the fixed steps are skipped: the column as it is ({cond.describe()}), calibrated")

    def _execute(self, i: int, op) -> str:
        if isinstance(op, ColumnSetting):
            return self._column(i, op)
        if isinstance(op, BeamCalibrationStep):
            return self._beam_calibrate(i, op)
        if isinstance(op, Exposure):
            return self._exposure(i, op)
        raise TypeError(f"unknown operation {type(op).__name__}")

    def _column(self, i: int, op: ColumnSetting) -> str:
        if op.preset:
            self.hw.apply_preset(op.preset)
            self.preset = op.preset
        if op.current_pa:
            self.current_pa = float(op.current_pa)
        if i == 0:
            fs = self._need_field_size()
            p = self._need_calibration_sample()
            self.magnification = fs.magnification
            self.field = float(self.hw.set_magnification(fs.magnification))
            self.hw.set_working_distance(p.wd_mm)
            self.log(f"step 1: field size {fs.name} (×{fs.magnification:g}), the focus at WD1 {p.wd_mm:g} mm")
            return (f"field {fs.name}, ×{fs.magnification:g} ({self.field:.4g} µm), beam {op.preset or 'kept as is'}, "
                    f"WD1 {p.wd_mm:g} mm")
        cond = self.condition
        if self.beamcal(cond) is None:
            self.log(f"step {i + 1}: no beam calibration for {cond.describe()}: nominal beam")
        return f"beam {op.preset or 'kept as is'}" + (f", {op.current_pa:g} pA" if op.current_pa else "")

    def _beam_calibrate(self, i: int, op: BeamCalibrationStep) -> str:
        from .beamcal import BeamCalError, measure_beam_calibration

        cond = self.condition
        if op.reuse:
            bc = self.beamcal(cond)
            if bc is not None:
                self.log(f"step {i + 1}: the stored beam calibration for {cond.describe()}: {bc.describe()}")
                return f"stored: {bc.describe()}"
        cfg = op.config
        p = self._need_calibration_sample()
        x, y, z = p.x, p.y, p.z_um
        self.move_stage(x, y, z)
        if self._wd() is None or abs(self._wd() - p.wd_mm) >= 5e-4:
            self.hw.set_working_distance(p.wd_mm)
        if cfg.manual_focus:
            self._manual_focus(i, cfg)
        cond = self.condition
        field = float(self.hw.field_um())
        self._emit("beamcal_start", index=i, position=(x, y, z), condition=cond, field_um=field,
                   pixels=int(cfg.pixels))

        def on_image(image, pixel_um):
            self._emit("beamcal_image", image=image, pixel_um=pixel_um)

        try:
            res = measure_beam_calibration(self.hw, cfg, cond, self.log, self._check, on_image=on_image)
        except BeamCalError as e:
            if e.result is None or self.review_beam is None:
                raise
            e.result.problem = str(e)
            e.result.position = (x, y, z)
            if not self.review_beam(e.result):
                raise
            res = e.result
            self.log(f"step {i + 1}: WARNING the refused beam calibration is used (override): {e}")
        else:
            res.position = (x, y, z)
            if self.review_beam is not None and not self.review_beam(res):
                raise RunAborted("beam calibration discarded")
        self.beamcals = [b for b in self.beamcals if not b.condition.matches(cond, preset=False)] + [res]
        self._emit("beamcal", result=res)
        return res.describe()

    def _manual_focus(self, i: int, cfg) -> None:
        """Zoom in on the calibration sample and wait while the user focuses and stigmates by hand;
        then back to the step's magnification.  The WD read back is WD1 from then on (``"wd1"``);
        the stage Z must not change."""
        p = self.cal_position
        mag = int(cfg.focus_magnification)
        back = self.magnification
        self.log(f"step {i + 1}: ×{mag:g} on {p.name!r}: focus and stigmate by hand")
        self._emit("focus_request", index=i, position=p.name, magnification=mag)
        self.hw.set_magnification(mag)
        try:
            ok = self.hw.ask_focus(f"Step {i + 1}: focus and stigmate on the calibration grid ({p.name}, ×{mag:,}) – "
                                   f"it should be close at WD1 {p.wd_mm:g} mm – then Continue.\n\nDo not move the "
                                   "stage Z: the chip heights are WDs at this Z.")
        finally:
            if back is not None:
                self.field = float(self.hw.set_magnification(back))
        if not ok:
            raise RunAborted("focus cancelled")
        self._check()
        self._check_z_cal()
        wd = self._wd()
        if wd is None:
            raise ScheduleError("the working distance cannot be read after the focus")
        wd = round(wd, 4)
        if not 4.0 <= wd <= 60.0:
            raise ScheduleError(f"the WD after the focus ({wd:g} mm) is outside 4–60 mm")
        old = p.wd_mm
        if abs(wd - old) >= 5e-4:
            p.wd_mm = wd
            self.log(f"step {i + 1}: WD1 {old:g} → {wd:g} mm (focused by hand)")
            self._emit("wd1", wd_mm=wd, old=old, z_um=p.z_um)

    # ---------------------------------------------------------------- field calibration (standalone)
    def field_calibrate(self, config: FieldCalConfig) -> FieldCalResult:
        """The field calibration of the field size (outside a schedule): at the stage position of
        *config* (Z_cal + ΔZ, the focus at WD1), the column as it is – at the field size's
        magnification.  Returns the result (also :attr:`calibration`, ``"calibration"`` event)."""
        fs = self._need_field_size()
        mag = self.hw.magnification()
        if mag is None or int(round(mag)) != fs.magnification:
            raise ScheduleError(f"the column is at ×{mag:g}, not field size {fs.name}'s ×{fs.magnification:g} "
                                "(select the field size in the Microscope window)" if mag is not None
                                else "the magnification is not known")
        self.magnification = fs.magnification
        self.field = float(self.hw.field_um())
        cal = self._need_calibration_sample()
        p = self._position(config.position, "field calibration")
        self.move_stage(p.x, p.y, self.z_of(p))
        if self._wd() is None or abs(self._wd() - cal.wd_mm) >= 5e-4:
            self.hw.set_working_distance(cal.wd_mm)
        cond = self.condition
        bc = self.beamcal(cond)
        if bc is None:
            self.log(f"field calibration: no beam calibration for {cond.describe()} – measured with the nominal beam")
        n = len(shift_pattern(config.pattern, 1.0))
        self._emit("calibration_start", index=-1, n=n, position=(p.x, p.y), condition=cond, field_um=self.field)

        def on_image(k, n, shift, image, reference):
            self._emit("calibration_image", k=k, n=n, shift=shift, image=image, reference=reference)

        try:
            res = measure_field_calibration(self.hw, config, (p.x, p.y), cond, self.log, self._check,
                                            self.axes, on_image=on_image, beamcal=bc)
        except FieldCalError as e:
            if e.result is None or self.review is None or not self.review(e.result):
                raise
            res = e.result
            self.log(f"field calibration: WARNING the refused field calibration is used (override): {e}")
        else:
            if self.review is not None and not self.review(res):
                raise RunAborted("field calibration discarded")
        self.calibration = res
        self._emit("calibration", result=res)
        return res

    def run_field_calibration(self, config: FieldCalConfig) -> str:
        """:meth:`field_calibrate` with the run bookkeeping (``"finished"``); returns done / aborted / failed."""
        self.state = "running"
        try:
            res = self.field_calibrate(config)
            self.state, msg = "done", f"field calibration: {res.describe(self.axes)}"
        except RunAborted as e:
            self.state, msg = "aborted", str(e) or "aborted"
        except Exception as e:  # noqa: BLE001 - reported
            self.state, msg = "failed", f"field calibration: {type(e).__name__}: {e}"
        self.log(msg)
        self._emit("finished", state=self.state, message=msg)
        return self.state

    # ---------------------------------------------------------------- exposure
    def _measure(self, i: int, op: Exposure, marks: list, g: ChipTransform, nominal: ChipTransform,
                 stage: np.ndarray, b: np.ndarray, bc, kind: str) -> np.ndarray:
        """Predict and measure markers [(name, ccs)] → their measured SCS positions (N, 2)."""
        ccs = np.array([m for _, m in marks], float).reshape(-1, 2)
        fm = FieldMap(b, b @ (-stage), bc)  # SCS → XCS with the stage at its read-back: bcs = B·(p − ŝ)
        pred_scs = g.to_scs(ccs)
        pred = np.atleast_2d(fm.apply(pred_scs))
        nom = np.atleast_2d(nominal.to_scs(ccs))
        reqs = [MarkRequest(f"step {i + 1} {name}", tuple(map(float, c)), tuple(map(float, s)),
                            (float(p[0]), float(p[1])), op.view_um)
                for (name, _), c, s, p in zip(marks, ccs, nom, pred)]
        got = self.hw.measure_marks(reqs)
        if got is None:
            raise RunAborted(f"{kind} cancelled")
        self._check()
        got = np.asarray(got, float).reshape(len(marks), 2)
        bcs = np.atleast_2d(fm.xcs_to_bcs(got))
        scs = stage + bcs @ np.linalg.inv(b).T
        dev = ", ".join(f"({d[0]:+.2f}, {d[1]:+.2f})" for d in scs - pred_scs)
        self.log(f"step {i + 1}: {kind} – markers off the prediction by {dev} µm (stage)")
        return scs

    def _global_align(self, i: int, op: Exposure, g: ChipTransform, nominal: ChipTransform, stage, b, bc):
        """Global Align, confirmed: every marker measured, then all again (1 → 2 → 1 → 2), the
        second pass predicted from the first.  The passes must agree within the Exposure's
        alignment tolerance, else ``ask_alignment``: repeat / continue (an override) / abort.
        Returns the chip map fitted to the mean of the two passes."""
        ccs = np.asarray(op.global_markers, float)
        marks = [(f"global marker {j + 1}", m) for j, m in enumerate(op.global_markers)]
        tol = float(op.align_tolerance_um)
        attempt = 0
        while True:
            attempt += 1
            first = self._measure(i, op, marks, g, nominal, stage, b, bc, "Global Align")
            g1, _ = fit_chip(ccs, first, self.axes)
            second = self._measure(i, op, [(f"{n} (again)", m) for n, m in marks], g1, nominal, stage, b, bc,
                                   "Global Align, confirmation")
            dev = np.hypot(*(second - first).T)
            worst = float(np.max(dev))
            mean = 0.5 * (first + second)
            if worst <= tol:
                self.log(f"step {i + 1}: Global Align confirmed: the two passes agree within {worst:.3f} µm "
                         f"(tolerance {tol:g} µm)")
                break
            text = (f"Step {i + 1}: the two Global Align passes differ by up to {worst:.3f} µm (marker "
                    + ", ".join(f"{j + 1}: {d:.3f}" for j, d in enumerate(dev))
                    + f" µm), more than the tolerance {tol:g} µm.")
            self.log(f"step {i + 1}: WARNING {text.split(': ', 1)[1]}")
            ans = self.hw.ask_alignment(text)
            if ans == "repeat":
                self.log(f"step {i + 1}: Global Align repeated (attempt {attempt + 1})")
                continue
            if ans == "continue":
                self.log(f"step {i + 1}: WARNING Global Align not confirmed, continued anyway (override)")
                break
            raise RunAborted("Global Align not confirmed")
        g, _ = fit_chip(ccs, mean, self.axes)
        return g

    def _exposure(self, i: int, op: Exposure) -> str:
        chip = self.schedule.chip(op.chip)
        if chip is None:
            raise ScheduleError(f"no chip named {op.chip!r}")
        nominal = chip.nominal(self.axes)
        g = self.chip_map(chip.name)
        flds = exposure_fields(op, self.library)
        if not flds:
            raise ScheduleError("the Exposure has no fields")
        field_um = self.field if self.field is not None else float(self.hw.field_um())
        asg = assign_markers(flds, op, field_um)
        flds = asg.fields
        for w in asg.warnings:
            self.log(f"step {i + 1}: WARNING {w}")
        if asg.errors:
            raise ScheduleError("; ".join(asg.errors))
        if op.global_align and asg.global_field is None:
            raise ScheduleError("the Global Align markers are not inside one field")
        if asg.field0:
            self.log(f"step {i + 1}: the Global Align markers are in no field: {flds[asg.global_field].name} at CCS "
                     f"({flds[asg.global_field].centre_ccs[0]:.2f}, {flds[asg.global_field].centre_ccs[1]:.2f}) µm")
        w = chip.height.problem(self.z_cal, check_z=True)
        if w:
            raise ScheduleError(f"chip {chip.name}: {w}")
        if chip.height.mode == "none":
            self.log(f"step {i + 1}: chip {chip.name} has no height: the focus is kept")
        w = wd_far_problem(chip.height, self.wd1)
        if w:
            self.log(f"step {i + 1}: WARNING chip {chip.name}: {w}")
        self._check_z_cal()
        cond = self.condition
        b, calibrated = self.stage_to_beam(cond)
        bc = self.beamcal(cond)
        if not calibrated:
            self.log(f"step {i + 1}: no field calibration for {cond.describe()}: nominal stage ↔ beam rotation")
        params = params_for_dwell(op.dwell_us * 1e-6, op.step_nm * 1e-3, self.schedule.settings)
        settings = VectorScanSettings(**self.schedule.vector_settings)
        n, written, skipped = len(asg.order), 0, 0
        for k_i, k in enumerate(asg.order):
            self._check()
            f = flds[k]
            self._emit("field", index=i, k=k_i, n=n, name=f.name, state="running", map=None)
            target = g.to_scs(np.asarray(f.centre_ccs, float))
            wd = chip.wd_at(f.centre_ccs)
            self.log(f"step {i + 1}: field {f.name} ({k_i + 1}/{n}): stage → ({target[0]:.2f}, {target[1]:.2f}) µm"
                     + (f", WD {wd:.4f} mm" if wd is not None else ""))
            stage = np.asarray(self.move_stage(float(target[0]), float(target[1])), float)
            if wd is not None:
                if not 4.0 <= wd <= 60.0:
                    raise ScheduleError(f"field {f.name}: WD {wd:.3f} mm is outside 4–60 mm (the height map?)")
                self.hw.set_working_distance(round(wd, 4))
            if op.global_align and k == asg.global_field:
                g = self._global_align(i, op, g, nominal, stage, b, bc)
                g.step = i
                self.chips[chip.name] = g
                if g.rms_um is not None:
                    self.log(f"step {i + 1}: Global Align residuals rms {g.rms_um:.3f} µm")
                self.log(f"step {i + 1}: chip {chip.name}: {g.describe(self.axes)}")
                self._emit("alignment", chip=chip.name, transform=g.copy(), index=i)
            delta = np.zeros(2)
            if k in asg.local:
                m = asg.local[k]
                scs = self._measure(i, op, [("local marker", m)], g, nominal, stage, b, bc, "Local marker")
                delta = scs[0] - g.to_scs(np.asarray(m, float))
            fm = FieldMap.make(g, f.centre_ccs, stage, b, delta, bc)
            self.last_map = fm
            if op.alignment_only or not f.path:  # nothing to write (alignment only, Field 0)
                self._emit("field", index=i, k=k_i, n=n, name=f.name, state="done", map=fm.copy())
                continue
            r = self._write(i, op, f, fm, field_um, params, settings)
            written += r == "done"
            skipped += r == "skipped"
            self._emit("field", index=i, k=k_i, n=n, name=f.name, state=r, map=fm.copy())
        if op.alignment_only:
            return f"aligned (chip {chip.name})"
        return f"{written} of {n} field(s) exposed" + (f", {skipped} skipped" if skipped else "")

    def _write(self, i: int, op: Exposure, f, fm: FieldMap, field_um: float, params, settings) -> str:
        pat = self.library.get(f.spec(op))
        fc = check_field(pat, fm, field_um)
        if not fc.ok:
            msg = f"step {i + 1}, field {f.name}: {fc.message()}"
            self.log("WARNING " + msg)
            policy = self.schedule.settings.on_out_of_field
            if policy == "abort":
                raise RunAborted("pattern outside the field of view")
            if policy == "ask" and not self.hw.ask(msg + "\n\nSkip this field and continue? (No aborts the schedule.)"):
                raise RunAborted("pattern outside the field of view")
            return "skipped"
        self._check()
        rep = self.hw.expose(pat, params, fm, settings)
        self._emit("exposed", index=i, report=rep)
        if (isinstance(rep, dict) and rep.get("aborted")) or getattr(rep, "aborted", False):
            raise RunAborted("exposure aborted")
        return "done"

    def move_stage(self, x: float, y: float, z: float | None = None) -> tuple:
        """Move and check the read-back position against the target (``StageError`` when off by
        more than the tolerance); returns the read-back (x, y).  *z* None: Z stays.  A Z move
        larger than the schedule's ``max_z_change_um`` is refused (``StageError``) before anything
        moves."""
        if z is None:
            self.hw.move_stage(float(x), float(y))
        else:
            z = float(z)
            if z < 0:
                raise StageError(f"stage Z {z:.1f} µm is below 0 (ΔZ or Z_cal wrong?)")
            z0 = self.hw.stage_z()
            if z0 is None:
                raise StageError("the stage Z is not known")
            limit = float(self.schedule.settings.max_z_change_um)
            if abs(z - z0) > limit:
                raise StageError(f"Z would move by {z - z0:+.1f} µm (from {z0:.1f} to {z:.1f}), more than the "
                                 f"schedule allows ({limit:g} µm): check ΔZ and Z_cal")
            self.hw.move_stage(float(x), float(y), z)
            zr = self.hw.stage_z()
            if zr is None or abs(zr - z) > max(self.tolerance, 1.0):
                raise StageError(f"the stage Z reads {zr} µm after the move, not {z:.1f}")
        pos = self.hw.read_stage()
        if pos is None:
            raise StageError("the stage position could not be read back after the move")
        err = math.hypot(pos[0] - x, pos[1] - y)
        if err > self.tolerance:
            raise StageError(f"the stage is at ({pos[0]:.2f}, {pos[1]:.2f}) µm, {err:.2f} µm from the target "
                             f"({x:.2f}, {y:.2f}) (tolerance {self.tolerance:g} µm)")
        return pos
