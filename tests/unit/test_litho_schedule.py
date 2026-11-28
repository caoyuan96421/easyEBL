"""Lithography module: coordinate maps, chips / exposures, marker assignment, plan, runner, field
sizes, the calibration sample (Z_cal, WD1), chip heights as WD2, field and beam calibration,
settings, demo files."""
import json
import math

import numpy as np
import pytest

from easyebl.litho.beamcal import BeamCalConfig, BeamCalibration
from easyebl.litho.coords import (S3000_AXES, ChipTransform, FieldMap, StageAxes, check_field, fit_chip,
                                  linear_params, stage_rotation)
from easyebl.litho.fieldcal import (Condition, FieldCalConfig, FieldCalError, FieldCalResult, fit_stage_to_beam,
                                    measure_field_calibration, phase_correlation, shift_pattern)
from easyebl.litho.geometry import Pattern, rectangle
from easyebl.litho.height import ChipHeight, fit_plane, z_problem
from easyebl.litho.runner import ScheduleRunner, SimulatedHardware, StageError
from easyebl.litho.schedule import (FIXED_STEPS, BeamCalibrationStep, Chip, ColumnSetting, Exposure, PatternLibrary,
                                    Schedule, ScheduleError, assign_markers, dose_readout, exposure_fields,
                                    FIELD0_NAME, find_beamcal, plan)
from easyebl.litho.settings import FieldSize, LithoSettings, StagePosition

CAL = FieldCalConfig(settle_s=0.0)
ORIGIN = (5000.0, 5000.0)
#: the tests' field sizes: ×1000 → 100 µm on the simulator (100 mm at ×1), ×2000 → 50 µm
SIZES = [FieldSize("100 µm", 1000, 15.0), FieldSize("50 µm", 2000, 15.0)]
#: the calibration sample: Z_cal 15000 µm (the simulator's Z), WD1 15 mm (its WD)
CALS = StagePosition("Calibration sample", 1000.0, 1000.0, role="calibration", z_um=15000.0, wd_mm=15.0)
POS = {CALS.name: CALS}
N_FIXED = len(FIXED_STEPS)  # Column setting, Beam calibration: the first steps of every schedule


def _schedule(*ops, fs="100 µm", **kw) -> Schedule:
    """A schedule at field size *fs* (its first Column setting)."""
    ops = list(ops)
    if not ops or not isinstance(ops[0], ColumnSetting):
        ops.insert(0, ColumnSetting())
    ops[0].field_size = fs
    return Schedule(ops, **kw)


def _calibrated(bc=None, mag=1000, b=None, fs="100 µm", **extra) -> dict:
    """Runner keywords for a run after the fixed steps (``run(N_FIXED)``): the field sizes, the calibration
    sample, a beam calibration (default the identity: the nominal beam) and the field size's field
    calibration (default the nominal rotation)."""
    bc = bc or BeamCalibration(condition=Condition(float(mag), None, 15.0, fs))
    cal = FieldCalResult(S3000_AXES.matrix if b is None else b, Condition(float(mag), None, 15.0, fs), 100_000.0 / mag,
                         (0.0, 0.0), 0.0, 9, beamcal_id=bc.id)
    out = dict(beamcals=[bc], calibration=cal, positions=dict(POS), field_sizes=SIZES)
    out.update(extra)
    return out


def _runnable(s: Schedule, bc=None) -> dict:
    """Make the fixed steps of *s* runnable on the simulator (a stored beam calibration *bc*, no focus
    prompt); returns the runner keywords."""
    s[1].reuse = True
    fs = next(f for f in SIZES if f.name == s.field_size)
    bc = bc or BeamCalibration(condition=Condition(float(fs.magnification), None, 15.0, fs.name))
    return dict(positions=POS, beamcals=[bc], field_sizes=SIZES)


def _rot(deg, s=1.0):
    t = math.radians(deg)
    return s * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


@pytest.fixture
def frac_gds(tmp_path):
    """TOP places F1 at (0, 0) and F2 (cell origin at its lower left) twice, at x = 100 and 200."""
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    f1 = lib.new_cell("F1")
    f1.add(gdstk.rectangle((-40, -40), (40, 40), layer=1))
    f1.add(gdstk.rectangle((10, 10), (20, 20), layer=2))
    f2 = lib.new_cell("F2")
    f2.add(gdstk.rectangle((0, 0), (40, 80), layer=1))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(f1, (0, 0)))
    top.add(gdstk.Reference(f2, (100, -40), columns=2, rows=1, spacing=(100, 0)))
    path = tmp_path / "frac.gds"
    lib.write_gds(str(path))
    return str(path)


@pytest.fixture
def demo(tmp_path):
    from easyebl.litho.demo import make_demo_files

    return make_demo_files(str(tmp_path / "demo"))


# ====================================================================== coordinate maps
def test_chip_transform_and_global_fit():
    g = ChipTransform.nominal(ORIGIN, S3000_AXES)
    assert np.allclose(g.to_scs([10.0, 20.0]), [4990.0, 5020.0])  # X mirrored
    assert np.allclose(g.to_ccs(g.to_scs([[3.0, 4.0]])), [[3.0, 4.0]]) and not g.measured
    # 2 markers: rotation + scale + shift on top of the stage mirror
    true = ChipTransform(_rot(0.3, 1.002) @ S3000_AXES.matrix, (5003.0, 4998.0))
    ccs = np.array([[-45.0, -45.0], [45.0, 45.0]])
    t, res = fit_chip(ccs, true.to_scs(ccs), S3000_AXES)
    assert np.allclose(t.linear, true.linear) and np.allclose(t.offset, true.offset) and t.measured
    assert t.rms_um is None and np.allclose(res, 0, atol=1e-9)
    p = linear_params(t.correction(S3000_AXES))
    assert p["rotation_mrad"] == pytest.approx(math.radians(0.3) * 1e3) and p["scale"] == pytest.approx(1.002)
    # 4 markers: affine (stretch, shear) and a residual
    true = ChipTransform(np.array([[-1.001, 0.002], [0.0005, 0.998]]), (10.0, 20.0))
    ccs = np.array([[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]])
    t, _ = fit_chip(ccs, true.to_scs(ccs), S3000_AXES)
    assert np.allclose(t.linear, true.linear) and t.rms_um == pytest.approx(0.0, abs=1e-9)
    assert ChipTransform.from_dict(t.to_dict()).to_dict() == t.to_dict()
    with pytest.raises(ValueError):
        fit_chip([[0, 0]], [[0, 0]], S3000_AXES)
    with pytest.raises(ValueError):
        fit_chip([[0, 0], [0, 0]], [[0, 0], [1, 1]], S3000_AXES)


def test_stage_rotation_drops_scale_and_shear():
    b = _rot(0.4) @ np.diag([1.006, 0.994]) @ S3000_AXES.matrix  # scale and stretch go, the rotation stays
    r = stage_rotation(b, S3000_AXES)
    assert np.allclose(r @ S3000_AXES.matrix @ (r @ S3000_AXES.matrix).T, np.eye(2))  # a pure rotation
    assert linear_params(r @ S3000_AXES.matrix)["rotation_mrad"] == pytest.approx(math.radians(0.4) * 1e3, abs=2e-3)
    assert np.allclose(stage_rotation(S3000_AXES.matrix, S3000_AXES), S3000_AXES.matrix)


def test_field_map_chain_and_beam_calibration():
    g = ChipTransform.nominal(ORIGIN, S3000_AXES)
    b = S3000_AXES.matrix
    fm = FieldMap.make(g, (100.0, 0.0), stage=(4900.0, 5000.0), b=b)
    assert fm.is_identity()  # nominal everything, stage at the field centre: FCS = BCS = XCS
    fm = FieldMap.make(g, (100.0, 0.0), stage=(4900.5, 5000.0), b=b, delta=(0.0, 0.25))
    assert np.allclose(fm.to_bcs([0.0, 0.0]), b @ [-0.5, 0.25])  # the stage read back 0.5 µm off, δ in SCS
    bc = BeamCalibration(sx=1.01, sy=0.99, shear=0.002, k=1e-6)
    fm = FieldMap(np.eye(2), (0.0, 0.0), bc)
    x = np.array([[30.0, -20.0], [0.0, 0.0]])
    assert np.allclose(bc.to_bcs(fm.apply(x)), x, atol=1e-3)
    pat = Pattern([rectangle(-45, -45, 45, 45)])
    assert check_field(pat, FieldMap(), 100.0).ok
    short = FieldMap(np.eye(2), (0.0, 0.0), BeamCalibration(sy=0.88))  # the beam reaches less far in y
    assert not check_field(pat, short, 100.0).ok and check_field(pat, fm, 100.0).ok


# ====================================================================== schedule model
def test_exposure_fields_centres_and_offset(frac_gds):
    lib = PatternLibrary()
    e = Exposure(chip="c", path=frac_gds, offset=(1000.0, 0.0))
    f = exposure_fields(e, lib)
    assert [x.cell for x in f] == ["F1", "F2", "F2"]
    assert [x.centre_gds for x in f] == [(0.0, 0.0), (120.0, 0.0), (220.0, 0.0)]  # bounding-box centres
    assert f[1].centre_ccs == (1120.0, 0.0)
    pat = lib.get(f[1].spec(e))  # FCS: centred on the field centre
    assert pat.bbox == pytest.approx((-20.0, -40.0, 20.0, 40.0))
    e.cell_origin = True
    f = exposure_fields(e, lib)
    assert [x.centre_gds for x in f] == [(0.0, 0.0), (100.0, -40.0), (200.0, -40.0)]  # the cells' origins
    assert lib.get(f[1].spec(e)).bbox == pytest.approx((0.0, 0.0, 40.0, 80.0))
    e.layers = [(2, 0)]
    assert [x.cell for x in exposure_fields(e, lib)] == ["F1"]  # only F1 has layer 2
    a = Exposure(chip="c", global_align=True, global_markers=[(10.0, 10.0), (30.0, 30.0)])
    (only,) = exposure_fields(a, lib)
    assert a.alignment_only and only.centre_ccs == (20.0, 20.0)


def test_marker_assignment(frac_gds):
    lib = PatternLibrary()
    e = Exposure(chip="c", path=frac_gds, global_align=True, global_markers=[(110.0, -30.0), (130.0, 30.0)],
                 local_align=True, local_markers=[(-40.0, 40.0), (115.0, 0.0), (240.0, 0.0), (500.0, 0.0),
                                                  (210.0, 5.0)])
    f = exposure_fields(e, lib)
    a = assign_markers(f, e, 100.0)
    assert a.global_field == 1 and a.order == [1, 0, 2]  # the Global Align field first
    assert a.local == {0: (-40.0, 40.0), 2: (210.0, 5.0)}  # nearest wins, the global field takes none
    assert a.marker_field == [0, None, None, None, 2]
    w = "\n".join(a.warnings)
    assert "only in the Global Align field" in w and "(500, 0) is in no field" in w
    assert "(240, 0) is not used: its field(s) F2 #3 have another" in w
    # in no field's field of view, but within one field size: a dummy Field 0 (last), aligned first,
    # then every field in cell-name order
    e.global_markers = [(55.0, -10.0), (65.0, 10.0)]
    a = assign_markers(f, e, 100.0)
    assert a.field0 and a.global_field == 3 and len(a.fields) == 4 and not a.errors
    f0 = a.fields[3]
    assert f0.name == FIELD0_NAME and f0.centre_ccs == (60.0, 0.0) and not f0.path
    assert a.order == [3, 0, 1, 2] and 3 not in a.local
    # wider than a field: an error (the step cannot run)
    e.global_markers = [(0.0, 0.0), (120.0, 0.0)]
    a = assign_markers(f, e, 100.0)
    assert a.global_field is None and a.errors and "do not fit in one field" in a.errors[0]
    e.global_markers = [(0.0, 0.0)]
    assert any("2–4 markers" in x for x in assign_markers(f, e, 100.0).warnings)
    e2 = Exposure(chip="c", path=frac_gds, local_align=True, local_markers=[(0.0, 0.0)])
    a = assign_markers(f, e2, 100.0)
    assert a.global_field is None and any("2 field(s) without a Local marker" in x for x in a.warnings)


# ====================================================================== stage, field calibration
def test_one_way_approach_plan():
    from types import SimpleNamespace

    from easyebl.microscope.stage import StageControl

    st = SimpleNamespace(backlash_um=50.0)
    plan_ = lambda t, c: StageControl.approach_plan(st, t, c)  # noqa: E731
    assert plan_({"X": 300.0}, {"X": 100.0}) == [{"X": 300.0}]
    assert plan_({"X": 300.0}, {"X": 280.0}) == [{"X": 250.0}, {"X": 300.0}]
    assert plan_({"X": 20.0}, {"X": 300.0}) == [{"X": 0.0}, {"X": 20.0}]
    assert plan_({"Z": 10.0}, {"Z": 300.0}) == [{"Z": 10.0}]


def test_phase_correlation_and_fit():
    from scipy import ndimage

    big = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((600, 600)), 3)

    def crop(dx, dy):
        return ndimage.shift(big, (dy, dx), order=3, mode="wrap")[172:428, 172:428]

    a = crop(0, 0)
    for dx, dy in ((3.3, -2.6), (-80.4, 30.7), (0.25, 0.0)):
        sx, sy, q = phase_correlation(a, crop(dx, dy), guess=(dx, dy))
        assert (sx, sy) == pytest.approx((dx, dy), abs=0.06) and q > 0.1
    with pytest.raises(FieldCalError):
        phase_correlation(a, a, guess=(250, 0))
    b = _rot(0.5, 1.01)
    ds = shift_pattern("grid", 30.0)
    bb, drift, res = fit_stage_to_beam(ds, -ds @ b.T + (0.2, -0.1))
    assert np.allclose(bb, b) and np.allclose(drift, (0.2, -0.1)) and np.allclose(res, 0, atol=1e-12)


def test_simulated_sem_follows_the_stage():
    from types import SimpleNamespace

    from easyebl.app.backend import couple_sim_stage
    from easyebl.hal import SimulatedBackend

    be = SimulatedBackend()
    stage = SimpleNamespace(positions={"X": 1000.0, "Y": 2000.0})
    couple_sim_stage(be, stage, stage_to_beam=np.eye(2))
    x = np.array([5.0, -3.0])
    y = np.array([2.0, 7.0])
    before = be._sample_at(x, y)
    stage.positions = {"X": 1010.0, "Y": 2000.0}
    assert np.allclose(be._sample_at(x - 10.0, y), before)


def test_field_calibration_on_noisy_low_contrast_images():
    """De-noise + flatten + partial whitening: a map is found where pure phase correlation fails."""
    from easyebl.litho.fieldcal import WEAK_MATCH_PSR

    b = _rot(0.5) @ S3000_AXES.matrix
    old = FieldCalConfig(settle_s=0.0, denoise_px=0.0, background_fraction=0.0, whitening=1.0,
                         max_residual_px=1e9)
    for contrast, noise in ((0.1, 0.3), (0.05, 0.5)):  # signal-to-noise 0.33 and 0.1
        for seed in range(3):
            hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0), image_contrast=contrast, image_noise=noise,
                                   seed=seed)
            cfg = FieldCalConfig(settle_s=0.0, max_residual_px=3.0)  # noisy: more scatter
            res = measure_field_calibration(hw, cfg, (1000.0, 1000.0), Condition(1000, None), axes=S3000_AXES)
            assert np.abs(res.stage_to_beam - b).max() < 0.012, (contrast, seed, res.stage_to_beam)
            assert res.problem is None and min(res.quality) > WEAK_MATCH_PSR
    hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0), image_contrast=0.1, image_noise=0.3)
    res = measure_field_calibration(hw, old, (1000.0, 1000.0), Condition(1000, None), axes=S3000_AXES)
    assert np.abs(res.stage_to_beam - b).max() > 0.1  # the old matcher on the same images
    # clean images stay as precise as before
    hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0))
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), Condition(1000, None), axes=S3000_AXES)
    assert np.abs(res.stage_to_beam - b).max() < 3e-4 and res.rejected == []


def test_robust_fit_drops_a_bad_match():
    from easyebl.litho.fieldcal import fit_robust, prefilter

    b = _rot(0.3, 1.002)
    ds = shift_pattern("grid", 30.0)
    meas = -ds @ b.T + np.random.default_rng(0).normal(0, 0.005, ds.shape)
    meas[4] += (3.0, -2.0)  # one image matched the wrong feature
    bb, drift, res, dropped = fit_robust(ds, meas)
    assert dropped == [4] and np.allclose(bb, b, atol=2e-3) and np.hypot(*res[4]) > 3.0
    bb, _, _, dropped = fit_robust(ds, -ds @ b.T)
    assert dropped == [] and np.allclose(bb, b)
    img = np.random.default_rng(1).normal(size=(64, 64)) + np.linspace(0, 50, 64)[None, :]  # a strong gradient
    flat = prefilter(img, 1.0, 8.0)
    slope = lambda a: abs(np.polyfit(np.arange(64), a.mean(0), 1)[0])  # noqa: E731
    assert slope(flat) < 0.1 * slope(img)  # the gradient is (nearly) gone


def test_schedule_chips_roundtrip_and_errors(tmp_path, frac_gds):
    s = _schedule(name="t")
    c = s.add_chip(origin=ORIGIN)
    assert c.name == "Chip 1" and s.unique_chip_name() == "Chip 2"
    with pytest.raises(ScheduleError):
        s.add_chip("Chip 1")
    s.add(Exposure(chip="Chip 1", path=frac_gds, layers=[(1, 0)], offset=(5.0, -5.0), global_align=True,
                   global_markers=[(-10.0, -10.0), (10.0, 10.0)], local_align=True, local_markers=[(120.0, 0.0)],
                   align_tolerance_um=0.2))
    s.rename_chip("Chip 1", "A")
    assert s.exposures[0].chip == "A" and s.chip("A").origin == ORIGIN
    s.shift_chip("A", 10.0, -5.0)
    assert s.chip("A").origin == (5010.0, 4995.0)
    p = s.save(str(tmp_path / "sub" / "x.ebls.json"))
    t = Schedule.load(p)
    assert t.to_dict() == s.to_dict() and t.exposures[0].path == s.exposures[0].path
    assert t.exposures[0].layers == [(1, 0)] and t.chips == s.chips and t.exposures[0].align_tolerance_um == 0.2
    assert t.field_size == "100 µm" and t.to_dict()["version"] == 5
    d = s.to_dict()
    d["version"] = 2
    with pytest.raises(ScheduleError, match="version 2"):
        Schedule.from_dict(d)
    with pytest.raises(ScheduleError):
        Schedule.from_dict({"format": "other"})
    with pytest.raises(ScheduleError, match="tolerance"):
        Exposure(align_tolerance_um=0.0)
    assert "µC/cm²" in dose_readout(1.0, 20.0, 100.0) and "unknown" in dose_readout(1.0, 20.0, None)


def test_field_sizes_and_the_magnification():
    from easyebl.litho.schedule import MAGNIFICATIONS, nearest_magnification
    from easyebl.litho.settings import DEFAULT_FIELD_SIZES

    assert [f.describe() for f in DEFAULT_FIELD_SIZES()] == ["100 µm: ×1200, WD 7 mm", "250 µm: ×500, WD 11 mm",
                                                             "500 µm: ×250, WD 24 mm"]
    assert all(f.magnification in MAGNIFICATIONS for f in DEFAULT_FIELD_SIZES())
    for bad in (dict(name="", magnification=100, wd_mm=10), dict(name="x", magnification=0, wd_mm=10),
                dict(name="x", magnification=100, wd_mm=3.0)):
        with pytest.raises(ValueError):
            FieldSize(**bad)
    assert 900 in MAGNIFICATIONS and nearest_magnification(920) == 900 and nearest_magnification(1050) == 1000
    bad = _schedule(fs="odd")
    sizes = SIZES + [FieldSize("odd", 920, 15.0)]
    assert any("not a magnification the SEM accepts (nearest: ×900)" in w
               for w in plan(bad, 100_000.0, field_sizes=sizes)[0].warnings)
    assert any("no field size named 'nope'" in w for w in plan(_schedule(fs="nope"), 1e5, field_sizes=SIZES)[0].warnings)
    assert any("no field size chosen" in w for w in plan(Schedule(), 1e5, field_sizes=SIZES)[0].warnings)
    r = ScheduleRunner(bad, SimulatedHardware(), positions=POS, field_sizes=sizes)
    assert r.run() == "failed" and r.states[0] == "failed"
    # the first Column setting: the field size's magnification (a whole number), the focus to WD1
    hw = SimulatedHardware(wd_mm=14.0)
    assert ScheduleRunner(_schedule(fs="50 µm"), hw, positions=POS, field_sizes=SIZES).run(stop=1) == "done"
    assert hw.calls[:3] == [("prepare",), ("mag", 2000), ("wd", 15.0)] and type(hw.calls[1][1]) is int
    # a version 4 file: the magnification becomes the default field size with it, the WD and the Field
    # calibration steps go
    v4 = {"format": "easyebl-schedule", "version": 4, "name": "old", "wd_mm": 12.0,
          "operations": [{"type": "column", "magnification": 500, "preset": "p"},
                         {"type": "beam_calibration", "config": {"position": "grid", "pixels": 3000}},
                         {"type": "field_correction", "config": {"position": "cup"}},
                         {"type": "exposure", "chip": "A"}],
          "chips": [{"name": "A", "origin": [1, 2], "height": {"mode": "constant", "value_um": 3000.0}}]}
    t = Schedule.from_dict(v4)
    assert [o.TYPE for o in t] == ["column", "beam_calibration", "exposure"]
    assert t.field_size == "250 µm" and t[0].preset == "p" and t[1].config.pixels == 3000
    assert t.chips[0].height.mode == "none"  # H = WD − Z is not convertible
    v4["operations"][0]["magnification"] = 777
    assert Schedule.from_dict(v4).field_size == ""


def test_plan_fields_targets_and_warnings(frac_gds):
    s = _schedule(chips=[Chip("A", ORIGIN)])
    s.add(Exposure(chip="A", path=frac_gds), Exposure(chip="nope", path=frac_gds),
          Exposure(chip="A", path=frac_gds, offset=(0.0, 0.0), global_align=True,
                   global_markers=[(-30.0, -30.0), (30.0, 30.0)]))
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), field_sizes=SIZES)
    p = ps[2]
    assert [f.target for f in p.fields] == [(5000.0, 5000.0), (4880.0, 5000.0), (4780.0, 5000.0)]  # X mirrored
    assert all(f.fov_check.ok for f in p.fields) and not p.calibrated
    assert any(w.startswith("no field calibration for field size 100 µm") for w in p.warnings)
    assert all(f.wd is None for f in p.fields) and any("has no height: the focus is kept" in w for w in p.warnings)
    assert any("no calibration sample" in w for w in ps[0].warnings)
    assert any("no chip named 'nope'" in w for w in ps[3].warnings) and not ps[3].fields
    assert ps[4].fields[0].is_global and ps[4].fields[0].name.startswith("F1")
    # the 50 µm field (×2000): F2 (40 × 80) does not fit
    s.field_size = "50 µm"
    p = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), field_sizes=SIZES)[2]
    assert any("pattern outside the field of view" in w for w in p.warnings)
    # a measured chip map moves the targets
    g = ChipTransform(_rot(0.5) @ S3000_AXES.matrix, (5010.0, 4990.0), measured=True)
    s.field_size = "100 µm"
    p = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), chips={"A": g}, field_sizes=SIZES)[2]
    assert np.allclose(p.fields[1].target, g.to_scs([120.0, 0.0]))
    # the field size's field calibration applies (at any WD), with its beam calibration
    kw = _calibrated()
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), start=N_FIXED, calibration=kw["calibration"],
              beamcals=kw["beamcals"], positions=POS, field_sizes=SIZES)
    assert ps[2].calibrated and ps[2].beamcal is kw["beamcals"][0] and not ps[0].warnings
    # the Microscope window at another field size: warned (the run refuses)
    ps = plan(s, 100_000.0, positions=POS, field_sizes=SIZES, selected="50 µm")
    assert any("the Microscope window is at field size 50 µm" in w for w in ps[0].warnings)


def test_plan_calibrations_by_field_size(frac_gds):
    s = _schedule(Exposure(chip="A", path=frac_gds), ColumnSetting(preset="other"), Exposure(chip="A", path=frac_gds),
                  chips=[Chip("A", ORIGIN)])
    bc = BeamCalibration(sx=1.001, condition=Condition(1000, None, 15.0, "100 µm"))
    cal = _calibrated(bc)["calibration"]
    ps = plan(s, 1e5, S3000_AXES, library=PatternLibrary(), beamcals=[bc], positions=POS, calibration=cal,
              field_sizes=SIZES)
    assert ps[2].calibrated and ps[2].beamcal is bc
    assert ps[4].calibrated and ps[4].beamcal is bc  # another beam preset: the same calibrations (per field size)
    assert find_beamcal([bc], Condition(1000, "anything", 12.0, "100 µm")) is bc  # any preset, any WD
    assert find_beamcal([bc], Condition(1000, None, 15.0, "other size")) is None
    old = BeamCalibration(condition=Condition(1000, None, 15.0))  # stored without a field size (an older file)
    assert find_beamcal([old], Condition(1000, None, 15.0, "100 µm")) is None
    assert find_beamcal([old, bc], Condition(1000, None, 15.0)) is bc
    assert find_beamcal([bc], Condition(2000, None, 15.0)) is None
    assert cal.valid_for(Condition(1000, "x", 9.0, "100 µm")) and not cal.valid_for(Condition(1000, None, 15.0, "x"))
    # WD1 far from the nominal WD: no warning; a WD2 more than 1 mm from WD1: a warning (logged in the run)
    far = dict(POS)
    far[CALS.name] = StagePosition(CALS.name, 1.0, 1.0, role="calibration", z_um=15000.0, wd_mm=16.5)
    assert not any("WD1" in w for w in plan(s, 1e5, positions=far, field_sizes=SIZES)[0].warnings)
    s.chips[0].height = ChipHeight("map", points=[(0, 0, 15.2), (100, 0, 16.1), (0, 100, 15.0)], z_um=15000.0)
    ws = plan(s, 1e5, positions=POS, field_sizes=SIZES)[2].warnings
    assert any("WD2 16.100 mm more than 1 mm from WD1 15 mm" in w for w in ws)
    far[CALS.name].wd_mm = 15.6  # every point within 1 mm of WD1
    assert not any("WD2" in w for w in plan(s, 1e5, positions=far, field_sizes=SIZES)[2].warnings)


def _demo_run(demo, hw, bc=None, local=True, **kw):
    """The demo Exposure run after the fixed steps, with the beam calibration *bc* (None: nominal beam)."""
    s = Schedule.load(demo["schedule"])
    e = s[N_FIXED]
    e.local_align = local
    lib = PatternLibrary()
    cal = _calibrated(bc, mag=500, fs="250 µm")
    cal["field_sizes"] = None  # the defaults: "250 µm" is ×500
    r = ScheduleRunner(s, hw, library=lib, **cal, **kw)
    state = r.run(N_FIXED)
    flds = exposure_fields(e, lib)
    order = assign_markers(flds, e, hw.field_um()).order
    nom = s.chips[0].nominal(hw.axes)
    errs = [hw.placement_error(e, flds[k].centre_ccs, nom) for e, k in zip(hw.exposures, order)]
    return state, r, errs


TRUE_BEAM = dict(beam_linear=[[1.003, 0.001], [0.0, 0.998]], beam_k=2e-7)


TRUE_BC = BeamCalibration(sx=1.003, sy=0.998, shear=0.001, k=2e-7, condition=Condition(500.0, None, 15.0, "250 µm"))


DEMO_HW = dict(pivot=ORIGIN, start=ORIGIN, magnification=500)


def test_runner_global_and_local_align_land_exactly_with_the_beam_calibration(demo):
    hw = SimulatedHardware(_rot(0.25, 1.0005), **DEMO_HW, **TRUE_BEAM)
    state, r, errs = _demo_run(demo, hw, TRUE_BC, local=False)
    assert state == "done" and len(errs) == 9 and max(errs) < 2e-3  # the one-step inverse: ≈ nm
    g = r.chips["Demo chip"]
    assert g.measured and linear_params(g.correction(S3000_AXES))["rotation_mrad"] == pytest.approx(
        -math.radians(0.25) * 1e3, abs=1e-3)  # the sample turned in the beam frame: −θ in the mirrored stage frame
    assert len(hw.measurements) == 4  # the two Global Align markers, twice (confirmed)
    # without the beam calibration the beam's scale / shear / pincushion is written as it is
    hw = SimulatedHardware(_rot(0.25, 1.0005), **DEMO_HW, **TRUE_BEAM)
    _, _, errs = _demo_run(demo, hw, None, local=False)
    assert max(errs) > 0.3
    # Local markers absorb stage errors (the stage goes 0.3 µm off, reading back right)
    hw = SimulatedHardware(_rot(0.25), stage_noise_um=0.3, seed=3, **DEMO_HW, **TRUE_BEAM)
    state, _, errs = _demo_run(demo, hw, TRUE_BC, local=True)
    assert state == "done" and max(errs[1:]) < 2e-3 and len(hw.measurements) == 4 + 8
    hw = SimulatedHardware(_rot(0.25), stage_noise_um=0.3, seed=3, **DEMO_HW, **TRUE_BEAM)
    _, _, errs = _demo_run(demo, hw, TRUE_BC, local=False)
    assert max(errs) > 0.1


def test_global_align_holds_for_later_exposures_of_the_chip(demo):
    from easyebl.litho.demo import MARK

    gds = Schedule.load(demo["schedule"]).exposures[0].path
    chip = Chip("C", ORIGIN)
    s = _schedule(Exposure(chip="C", global_align=True, global_markers=[(-MARK, -MARK), (MARK, MARK)]),
                  Exposure(chip="C", path=gds, layers=[1]), fs="250 µm", chips=[chip])
    hw = SimulatedHardware(_rot(0.3), **DEMO_HW)
    events = []
    kw = _calibrated(mag=500, fs="250 µm")
    kw["field_sizes"] = None
    r = ScheduleRunner(s, hw, on_event=lambda k, d: events.append((k, d)), **kw)
    assert r.run(N_FIXED) == "done" and r.states == ["pending"] * 2 + ["done"] * 2 and len(hw.exposures) == 9
    lib = PatternLibrary()
    flds = exposure_fields(s[3], lib)
    errs = [hw.placement_error(e, f.centre_ccs, chip.nominal(hw.axes)) for e, f in zip(hw.exposures, flds)]
    assert max(errs) < 1e-6
    assert [d["chip"] for k, d in events if k == "alignment"] == ["C"]
    fields = [d for k, d in events if k == "field"]
    assert fields[0]["state"] == "running" and fields[-1]["state"] == "done" and fields[-1]["n"] == 9


def test_global_align_is_confirmed_by_a_second_pass(demo):
    """1 → 2 → 1 → 2: the passes must agree within the tolerance, else repeat / continue / abort."""
    from easyebl.litho.demo import MARK

    s = _schedule(Exposure(chip="C", global_align=True, global_markers=[(-MARK, -MARK), (MARK, MARK)],
                           align_tolerance_um=0.5), fs="250 µm", chips=[Chip("C", ORIGIN)])

    def run(noise, answers=(), seed=0):
        hw = SimulatedHardware(_rot(0.3), mark_noise_um=noise, seed=seed, **DEMO_HW)
        hw.alignment_answers = list(answers)
        kw = _calibrated(mag=500, fs="250 µm")
        kw["field_sizes"] = None
        log = []
        r = ScheduleRunner(s, hw, on_event=lambda k, d: log.append(d.get("text", "")), **kw)
        return r.run(N_FIXED), hw, r, log

    state, hw, r, log = run(0.0)
    assert state == "done" and len(hw.measurements) == 4 and not any(c[0] == "align" for c in hw.calls)
    assert [m[0].label.endswith("(again)") for m in hw.measurements] == [False, False, True, True]
    assert any("Global Align confirmed" in t for t in log)
    # the passes disagree (2 µm noise): asked; Abort
    state, hw, r, log = run(2.0, ["abort"])
    assert state == "aborted" and sum(c[0] == "align" for c in hw.calls) == 1 and "differ by up to" in [
        c for c in hw.calls if c[0] == "align"][0][1]
    # Repeat, then Continue anyway (an override): measured 4 times, the mean of the last two passes used
    state, hw, r, log = run(2.0, ["repeat", "continue"])
    assert state == "done" and len(hw.measurements) == 8 and sum(c[0] == "align" for c in hw.calls) == 2
    assert any("continued anyway (override)" in t for t in log) and r.chips["C"].measured
    # the default hardware answer (ask: No → abort)
    from easyebl.litho.runner import LithoHardware

    assert LithoHardware().ask_alignment("x") == "abort"


def test_runner_out_of_field_abort_errors_and_stage_check(frac_gds):
    s = _schedule(Exposure(chip="A", path=frac_gds), fs="50 µm", chips=[Chip("A", ORIGIN)])
    e = s[N_FIXED]

    def runner(hw, **kw):  # after the fixed steps, calibrated (nominal) at the field size
        return ScheduleRunner(s, hw, **_calibrated(mag=2000, fs="50 µm"), **kw)

    hw = SimulatedHardware(start=ORIGIN, magnification=2000)
    hw.answer = True  # skip
    assert runner(hw).run(N_FIXED) == "done" and not hw.exposures and sum(c[0] == "ask" for c in hw.calls) == 3
    s.settings.on_out_of_field = "abort"  # (50 µm field)
    assert runner(SimulatedHardware(start=ORIGIN, magnification=2000)).run(N_FIXED) == "aborted"
    s.field_size = "100 µm"
    s.settings.on_out_of_field = "ask"

    def runner(hw, **kw):  # noqa: F811
        return ScheduleRunner(s, hw, **_calibrated(), **kw)

    r = runner(SimulatedHardware(start=ORIGIN, stall_um=(2.0, 0.5)))
    assert r.run(N_FIXED) == "failed" and r.states[N_FIXED] == "failed"
    assert runner(SimulatedHardware(start=ORIGIN, stall_um=(2.0, 0.5)), stage_tolerance_um=3.0).run(N_FIXED) == "done"
    with pytest.raises(StageError):
        ScheduleRunner(s, SimulatedHardware(stall_um=(1.0, 0.0))).move_stage(0.0, 0.0)
    e.chip = "missing"
    r = runner(SimulatedHardware(start=ORIGIN))
    assert r.run(N_FIXED) == "failed" and r.states[N_FIXED] == "failed"
    e.chip = "A"
    e.global_align, e.global_markers = True, [(0.0, 0.0), (120.0, 0.0)]  # not in one field
    assert runner(SimulatedHardware(start=ORIGIN)).run(N_FIXED) == "failed"
    e.global_align = False
    r = runner(SimulatedHardware(start=ORIGIN))
    r.abort()
    assert r.run(N_FIXED) == "aborted"


def test_field_calibration_and_its_condition():
    b = _rot(0.5) @ S3000_AXES.matrix
    hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0))
    cond = Condition(1000, None, 15.0, "100 µm")
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), cond, axes=S3000_AXES)
    assert np.allclose(res.stage_to_beam, b, atol=1e-3) and res.beamcal_id is None
    assert hw.stage_position() == (1000.0, 1000.0)
    # used at any WD, beam preset and beam calibration of the field size
    assert res.valid_for(Condition(1000, None, 12.0, "100 µm")) and res.valid_for(Condition(1000, "x", 15.0))
    assert res.valid_for(cond, TRUE_BC)
    assert not res.valid_for(Condition(2000, None, 15.0)) and not res.valid_for(Condition(1000, None, 15, "50 µm"))
    t = FieldCalResult.from_dict(json.loads(json.dumps(res.to_dict())))  # kept in the settings
    assert np.allclose(t.stage_to_beam, res.stage_to_beam) and t.condition == cond and t.shifts == res.shifts
    assert t.describe(S3000_AXES) == res.describe(S3000_AXES)
    # a wrong mirror setting is still measured, and reported
    logs = []
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), cond, axes=StageAxes(), log=logs.append)
    assert res.mirror_mismatch(StageAxes()) and res.suggested_axes() == S3000_AXES
    assert any("Hardware → Configure" in t for t in logs)


def test_field_calibration_through_the_beam_calibration():
    """With the beam's scale error in the images, the field calibration measures ≈ 1 through the beam
    calibration (the stage is the length reference) and the rotation either way."""
    b = _rot(0.35) @ S3000_AXES.matrix
    beam = dict(beam_linear=[[1.006, 0.0], [0.0, 1.006]])
    hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0), **beam)
    bc = BeamCalibration(sx=1.006, sy=1.006, condition=Condition(1000.0, None, 15.0))
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), bc.condition, axes=S3000_AXES, beamcal=bc)
    p = linear_params(res.correction(S3000_AXES))
    assert p["scale"] == pytest.approx(1.0, abs=5e-4) and res.valid_for(bc.condition, bc)
    res2 = measure_field_calibration(hw, CAL, (1000.0, 1000.0), bc.condition, axes=S3000_AXES)
    assert linear_params(res2.correction(S3000_AXES))["scale"] == pytest.approx(1 / 1.006, abs=1e-3)
    assert np.allclose(stage_rotation(res.stage_to_beam, S3000_AXES), b, atol=3e-4)


def test_the_field_calibration_outside_a_schedule():
    """Measured at a stage position (Z_cal + ΔZ, the focus at WD1) at the field size, reviewed: Discard, a refused
    one not used / used anyway."""
    tex = StagePosition("tex", 3000.0, 3000.0, dz=-50.0)
    pos = dict(POS, tex=tex)
    cfg = FieldCalConfig(position="tex", settle_s=0.0)
    s = Schedule([ColumnSetting(field_size="100 µm")])
    b = _rot(0.5) @ S3000_AXES.matrix
    seen, events = [], []
    hw = SimulatedHardware(stage_to_beam=b, wd_mm=14.0)
    r = ScheduleRunner(s, hw, positions=pos, field_sizes=SIZES, on_event=lambda k, d: events.append((k, d)),
                       review=lambda res: seen.append(res) or True)
    assert r.run_field_calibration(cfg) == "done" and r.calibration is seen[0]
    assert hw.calls[:2] == [("move", 3000.0, 3000.0, 14950.0), ("wd", 15.0)]  # Z_cal + ΔZ, WD1
    assert np.allclose(stage_rotation(r.calibration.stage_to_beam, S3000_AXES), b, atol=1e-3)
    assert r.calibration.condition.field_size == "100 µm"
    assert [k for k, _ in events].count("calibration_image") == 9 and events[-1][0] == "finished"
    # not at the field size's magnification: refused before anything moves
    hw = SimulatedHardware(magnification=2000)
    r = ScheduleRunner(s, hw, positions=pos, field_sizes=SIZES)
    assert r.run_field_calibration(cfg) == "failed" and not hw.calls
    # Discard
    r = ScheduleRunner(s, SimulatedHardware(stage_to_beam=b), positions=pos, field_sizes=SIZES,
                       review=lambda res: False)
    assert r.run_field_calibration(cfg) == "aborted" and r.calibration is None
    # refused (noise): not overridden / Use anyway
    hw = SimulatedHardware()
    hw.capture_image = lambda w, px, frames=1: (np.random.default_rng(len(hw.calls)).normal(size=(px, px)),
                                                (w / px, w / px))
    for answer, state in ((False, "failed"), (True, "done")):
        seen = []
        r = ScheduleRunner(s, hw, positions=pos, field_sizes=SIZES, review=lambda res: seen.append(res) or answer)
        assert r.run_field_calibration(cfg) == state and len(seen) == 1 and seen[0].problem
        assert (r.calibration is seen[0]) is answer


# ====================================================================== beam calibration
def test_beam_calibration_on_the_calibration_sample():
    """The Beam calibration step: on the calibration sample at Z_cal, the focus at WD1, confirmed by hand."""
    hw = SimulatedHardware(grid_at=((2000.0, 2000.0), 50.0), start_z=8100.0, wd_mm=14.0, **TRUE_BEAM)
    s = _schedule()
    s[1].config = BeamCalConfig(pixels=1000)
    cal = {"Calibration sample": StagePosition("Calibration sample", 2000.0, 2000.0, role="calibration",
                                               z_um=8000.0, wd_mm=12.5)}
    events, seen = [], []
    r = ScheduleRunner(s, hw, positions=cal, field_sizes=SIZES, on_event=lambda k, d: events.append((k, d)),
                       review_beam=lambda res: seen.append(res) or True)
    assert r.run(stop=2) == "done"
    bc = [d["result"] for k, d in events if k == "beamcal"][0]
    assert bc is seen[0] and bc.condition == Condition(1000.0, None, 12.5, "100 µm")
    assert bc.position == (2000.0, 2000.0, 8000.0)
    assert hw.calls[:7] == [("prepare",), ("mag", 1000), ("wd", 12.5), ("move", 2000.0, 2000.0, 8000.0),
                            ("mag", 10000), ("focus", 10000), ("mag", 1000)]  # confirmed by hand at ×10k first
    assert bc.sx == pytest.approx(1.003, abs=2e-5) and bc.sy == pytest.approx(0.998, abs=2e-5)
    assert bc.shear == pytest.approx(0.001, abs=5e-5) and bc.k == pytest.approx(2e-7, rel=0.05)
    assert [k for k, _ in events if k.startswith("beamcal")] == ["beamcal_start", "beamcal_image", "beamcal"]
    assert r.beamcals == [bc] and r.beamcal(Condition(1000.0, "p", 9.0, "100 µm")) is bc
    # "use a stored one": no image, no prompt
    s[1].reuse = True
    hw.calls.clear()
    images = hw.images
    r = ScheduleRunner(s, hw, positions=cal, field_sizes=SIZES, beamcals=[bc])
    assert r.run(stop=2) == "done" and not any(c[0] == "focus" for c in hw.calls) and r.beamcals == [bc]
    assert hw.images == images
    s[1].reuse = False
    # Discard; and a refused fit (no grid there) is shown, then fails
    r = ScheduleRunner(s, hw, positions=cal, field_sizes=SIZES, review_beam=lambda res: False)
    assert r.run(stop=2) == "aborted" and r.beamcals == []
    seen = []
    nogrid = {"c": StagePosition("c", 9000.0, 9000.0, role="calibration", z_um=8000.0, wd_mm=12.5)}
    s[1].config = BeamCalConfig(pixels=600)
    r = ScheduleRunner(s, hw, positions=nogrid, field_sizes=SIZES, review_beam=lambda res: seen.append(res))
    assert r.run(stop=2) == "failed" and (not seen or seen[0].problem) and r.beamcals == []  # (no fit: nothing to use)
    # a fit refused by the rms limit: Discard fails the step, Use anyway uses it
    s[1].config = BeamCalConfig(pixels=1000, max_rms_um=1e-9)
    for answer, state in ((False, "failed"), (True, "done")):
        seen = []
        r = ScheduleRunner(s, hw, positions=cal, field_sizes=SIZES, review_beam=lambda res: seen.append(res) or answer)
        assert r.run(stop=2) == state and len(seen) == 1 and seen[0].problem
        assert r.beamcals == (seen if answer else [])
    # no WD1 / no calibration sample: refused before anything happens on the column
    for pos in ({}, {"c": StagePosition("c", 1.0, 1.0, role="calibration", z_um=8000.0)}):
        hw.calls.clear()
        log = []
        r = ScheduleRunner(s, hw, positions=pos, field_sizes=SIZES, on_event=lambda k, d: log.append(str(d)))
        assert r.run(stop=2) == "failed" and not hw.calls and any("WD1" in m or "calibration sample" in m
                                                                   for m in log), log


def test_the_focus_confirmed_by_hand_is_wd1_and_the_z_must_stay():
    hw = SimulatedHardware(grid_at=((1000.0, 1000.0), 50.0), **TRUE_BEAM)
    s = _schedule()
    s[1].config = BeamCalConfig(pixels=1000)
    hw.refocus_wd = 15.21  # "the user" focuses at 15.21 mm
    events = []
    r = ScheduleRunner(s, hw, positions=POS, field_sizes=SIZES, on_event=lambda k, d: events.append((k, d)))
    assert r.run(stop=2) == "done"
    assert ("wd1", {"wd_mm": 15.21, "old": 15.0, "z_um": 15000.0}) in events
    assert r.wd1 == 15.21 and POS[CALS.name].wd_mm == 15.0  # the caller's positions are not changed
    bc = [d["result"] for k, d in events if k == "beamcal"][0]
    assert bc.condition.wd_mm == pytest.approx(15.21)
    # the user moved Z while focusing: refused
    hw = SimulatedHardware(grid_at=((1000.0, 1000.0), 50.0))
    real = hw.ask_focus

    def moved(text):
        hw.z += 30.0
        return real(text)

    hw.ask_focus = moved
    log = []
    r = ScheduleRunner(s, hw, positions=POS, field_sizes=SIZES, on_event=lambda k, d: log.append(str(d)))
    assert r.run(stop=2) == "failed" and hw.images == 0 and any("not at the calibration sample's Z" in t for t in log)
    # Abort at the prompt: back to the step's magnification
    hw = SimulatedHardware(grid_at=((1000.0, 1000.0), 50.0))
    hw.focus_answer = False
    r = ScheduleRunner(s, hw, positions=POS, field_sizes=SIZES)
    assert r.run() == "aborted" and r.states[1] == "aborted" and hw.images == 0 and hw.mag == 1000
    # off: no prompt
    s[1].config.manual_focus = False
    hw = SimulatedHardware(grid_at=((1000.0, 1000.0), 50.0))
    assert ScheduleRunner(s, hw, positions=POS, field_sizes=SIZES).run(stop=2) == "done"
    assert not any(c[0] == "focus" for c in hw.calls)
    t = Schedule.from_dict(s.to_dict())
    assert t[1].config.manual_focus is False and t[1].config.focus_magnification == 10000


# ====================================================================== settings, demo
def test_settings_positions_field_sizes_and_calibrations(tmp_path):
    s = LithoSettings(path=str(tmp_path / "l.json"))
    assert [(p.name, p.role) for p in s.stage_positions] == [("Calibration sample", "calibration"),
                                                             ("Faraday cup", "faraday")]
    with pytest.raises(ValueError, match="always kept"):
        s.delete_position("Faraday cup")
    s.replace_position("Calibration sample", 100.0, 200.0, 11000.0)
    assert s.calibration.z_um == 11000.0 and s.calibration.dz == 0.0 and s.calibration.wd_mm is None
    s.replace_position("Faraday cup", 500.0, 600.0, 11040.0)
    s.add_position("grid", 1.0, 2.0, s.dz_here(10990.0))
    assert s.faraday.dz == 40.0 and s.positions()["grid"].dz == -10.0 and s.z_of(s.faraday) == 11040.0
    with pytest.raises(ValueError):
        s.add_position("grid", 0, 0)
    # Record WD: WD1 and Z_cal; no other ΔZ changes (their Z follows Z_cal)
    s.record_wd(11.0234, 11020.0)
    assert s.calibration.wd_mm == 11.0234 and s.calibration.z_um == 11020.0
    assert s.faraday.dz == 40.0 and s.z_of(s.faraday) == 11060.0
    # Replace the calibration sample at another Z: every other ΔZ changes so their Z stays; WD1 is kept
    s.replace_position("Calibration sample", 110.0, 210.0, 11120.0)
    assert s.faraday.dz == pytest.approx(-60.0) and s.z_of(s.faraday) == pytest.approx(11060.0)
    assert s.positions()["grid"].dz == pytest.approx(-110.0) and s.calibration.dz == 0.0
    assert s.calibration.wd_mm == 11.0234
    s.delete_position("grid")
    # field sizes and the field calibration of each
    assert s.field_size == "" and s.selected_field_size is None
    s.field_size = "250 µm"
    assert s.selected_field_size.magnification == 500
    # WD1 is kept per field size: the calibration sample's WD1 is the selected one's
    assert s.calibration.wd_mm is None and s.wd1("") == 11.0234
    s.record_wd(11.1, 11130.0)  # Z_cal too is per field size: ΔZ against the selected one's
    assert s.calibration.z_um == 11130.0 and s.z_of(s.faraday) == pytest.approx(11070.0)
    assert s.dz_here(11140.0) == 10.0
    s.field_size = "100 µm"
    assert s.calibration.wd_mm is None and s.calibration.z_um is None and s.wd1("250 µm") == 11.1
    assert s.z_of(s.faraday) is None
    s.field_size = "250 µm"
    assert s.calibration.wd_mm == 11.1 and s.calibration.wd1 == {"": 11.0234, "250 µm": 11.1}
    assert s.calibration.zcal == {"": 11120.0, "250 µm": 11130.0} and s.z_cal("") == 11120.0
    with pytest.raises(ValueError, match="same name"):
        s.set_field_sizes([FieldSize("a", 100, 10), FieldSize("a", 200, 10)])
    res = FieldCalResult(S3000_AXES.matrix, Condition(500, None, 11.0, "250 µm"), 200.0, (0.0, 0.0), 0.01, 9)
    s.fieldcals["250 µm"] = res
    s.set_beamcal(BeamCalibration(sx=1.001, condition=Condition(500.0, None, 11.0, "250 µm")))
    s.set_beamcal(BeamCalibration(sx=1.002, condition=Condition(500.0, None, 11.4, "250 µm")))  # replaces
    s.set_beamcal(BeamCalibration(sx=1.003, condition=Condition(1200.0, None, 7.0, "100 µm")))
    s.save()
    t = LithoSettings.load(s.path)
    assert t.stage_positions == s.stage_positions and t.field_size == "250 µm" and t.field_sizes == s.field_sizes
    assert np.allclose(t.fieldcal().stage_to_beam, res.stage_to_beam) and t.fieldcal("100 µm") is None
    assert [b.sx for b in t.beamcals] == [1.002, 1.003]
    assert t.forget_beamcal(Condition(1200.0, None, None, "100 µm")) and len(t.beamcals) == 1
    s.set_field_sizes([FieldSize("x", 100, 10.0)])  # the selected one is gone
    assert s.field_size == ""
    # a file with one Z_cal / WD1 (before they were per field size): kept for the selected field size
    with open(s.path, "w", encoding="utf-8") as fh:
        json.dump({"field_size": "500 µm", "stage_positions": [
            {"name": "Calibration sample", "x": 1.0, "y": 2.0, "dz": 0.0, "role": "calibration", "z_um": 24500.0,
             "wd_mm": 26.1, "wd1": {"500 µm": 26.1, "100 µm": 7.7}}]}, fh)
    t = LithoSettings.load(s.path)
    assert (t.calibration.z_um, t.calibration.wd_mm) == (24500.0, 26.1) and t.calibration.zcal == {"500 µm": 24500.0}
    t.save()
    t = LithoSettings.load(s.path)  # and again after a restart
    assert (t.calibration.z_um, t.calibration.wd_mm) == (24500.0, 26.1)
    t.field_size = "100 µm"
    assert (t.calibration.z_um, t.calibration.wd_mm) == (None, 7.7)
    from easyebl.litho.schedule import calibration_problem

    assert "no Z_cal for the selected field size" in calibration_problem(t.calibration)
    # an older file: H (= WD − Z) is not ΔZ; a "Calibration sample" by name becomes the fixed one
    with open(s.path, "w", encoding="utf-8") as fh:
        json.dump({"stage_positions": [{"name": "old", "x": 1.0, "y": 2.0, "h": 8.0},
                                       {"name": "Calibration sample", "x": 5.0, "y": 6.0, "h": 3.0}]}, fh)
    t = LithoSettings.load(s.path)
    assert [p.name for p in t.stage_positions] == ["Calibration sample", "Faraday cup", "old"]
    assert t.positions()["old"].dz is None and t.calibration.x == 5.0 and t.calibration.z_um is None
    assert [f.name for f in t.field_sizes] == ["100 µm", "250 µm", "500 µm"]
    # one instance per file: the Lithography and the Microscope windows share it
    assert LithoSettings.shared(s.path) is LithoSettings.shared(str(tmp_path / "." / "l.json"))


def test_demo_files(demo):
    s = Schedule.load(demo["schedule"])
    assert [o.TYPE for o in s] == ["column", "beam_calibration", "exposure"]
    assert s.chips[0].origin == ORIGIN and s.field_size == "250 µm"
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary())
    p = ps[2]
    assert len(p.fields) == 9 and p.fields[0].is_global and sum(f.local_marker is not None for f in p.fields) == 8
    assert all(f.fov_check.ok for f in p.fields)
    d = Schedule.load(demo["dummy"])
    kw = _calibrated(mag=500, fs="250 µm")
    kw["field_sizes"] = None
    hw = SimulatedHardware(**DEMO_HW)
    assert ScheduleRunner(d, hw, **kw).run(N_FIXED) == "done" and len(hw.exposures) == 9
    # the whole schedule: the field size, WD1, a stored beam calibration; parked at the calibration sample
    hw = SimulatedHardware(**DEMO_HW)
    d[1].reuse = True
    r = ScheduleRunner(d, hw, positions=POS, beamcals=[TRUE_BC], selected="250 µm")
    assert r.run() == "done" and r.states == ["done"] * len(d) and len(hw.exposures) == 9
    assert hw.calls[:3] == [("prepare",), ("mag", 500), ("wd", 15.0)]
    assert hw.calls[-2:] == [("move", 1000.0, 1000.0, 15000.0), ("wd", 15.0)]  # parked: Z_cal, WD1


def test_heights_and_the_plane_fit():
    (a, b, c), rms = fit_plane([(0, 0, 100), (100, 0, 110), (0, 100, 90), (100, 100, 100)])
    assert (a, b, c) == pytest.approx((100.0, 0.1, -0.1)) and rms == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="at least 3"):
        fit_plane([(0, 0, 1), (1, 0, 1)])
    with pytest.raises(ValueError, match="one line"):
        fit_plane([(0, 0, 1), (1, 1, 2), (2, 2, 3)])
    none = ChipHeight()
    assert none.wd_at((0, 0)) is None and not none.ready and none.problem() is None and none.describe() == "focus kept"
    const = ChipHeight("constant")
    assert not const.ready and "not recorded" in const.problem() and const.wd_at((0, 0)) is None
    const.wd_mm, const.z_um = 11.25, 11000.0
    assert const.ready and const.wd_at((1e4, -1e4)) == 11.25 and const.tilt_mrad() is None
    assert const.problem(11000.5, check_z=True) is None  # within 1 µm
    assert "not at the calibration sample's Z" in const.problem(11010.0, check_z=True)
    assert "not recorded" in const.problem(None, check_z=True)
    assert "WD2 11.250 mm at Z 11000" in const.describe()
    m = ChipHeight("map", points=[(0, 0, 11.0), (1000, 0, 11.01), (0, 1000, 10.99)], z_um=11000.0)
    assert m.ready and m.wd_at((1200.0, 0.0)) == pytest.approx(11.012)
    assert m.tilt_mrad() == pytest.approx(math.hypot(10.0, 10.0))  # 10 µm of WD over 1 mm in x and in y
    assert ChipHeight.from_dict(m.to_dict()) == m and "3 pts" in m.describe() and m.wds() == [11.0, 11.01, 10.99]
    assert "at least 3" in ChipHeight("map", points=[(0, 0, 1)]).problem()
    with pytest.raises(ValueError):
        ChipHeight("tilted")
    assert ChipHeight.from_dict({"mode": "constant", "value_um": 3000.0}).mode == "none"  # an older file's H
    chip = Chip("A", ORIGIN, height=m)
    assert chip.wd_at((0.0, 0.0)) == pytest.approx(11.0) and Chip("B").wd_at((0, 0)) is None
    assert Chip.from_dict(chip.to_dict()) == chip
    assert z_problem(5.0, None) and z_problem(None, 5.0) and z_problem(5.0, 5.9) is None


def test_the_fixed_first_steps(tmp_path):
    s = Schedule()
    assert [type(o) for o in s] == list(FIXED_STEPS) and s.field_size == ""
    s.add(Exposure(chip="A"), Exposure(chip="B"))
    assert [s.can_delete(i) for i in range(4)] == [False, False, True, True]
    assert s.can_move(2, 3) and not s.can_move(2, 1) and not s.can_move(0, 1)
    assert Schedule.is_fixed(1) and not Schedule.is_fixed(2)
    off = Schedule([ColumnSetting(enabled=False), BeamCalibrationStep(enabled=False)])
    assert all(o.enabled for o in off)  # the fixed steps always run
    s.field_size = "250 µm"
    t = Schedule.from_dict(s.to_dict())
    assert t.field_size == "250 µm" and [o.TYPE for o in t] == [o.TYPE for o in s]
    # version 3: [Column, Field calibration, …]
    v3 = {"format": "easyebl-schedule", "version": 3, "name": "old",
          "operations": [{"type": "column", "magnification": 1200},
                         {"type": "field_correction", "position": "cup", "notes": "n"},
                         {"type": "exposure", "chip": "A"}]}
    t = Schedule.from_dict(v3)
    assert [o.TYPE for o in t] == ["column", "beam_calibration", "exposure"] and t.field_size == "100 µm"
    # the calibration settings live in the step
    s[1].config = BeamCalConfig(pixels=3000)
    t = Schedule.load(s.save(str(tmp_path / "f.ebls.json")))
    assert t[1].config == s[1].config


def test_the_focus_follows_each_chip_and_z_stays(frac_gds):
    m = ChipHeight("map", points=[(0, 0, 15.1), (100, 0, 15.11), (0, 100, 15.09)], z_um=15000.0)
    s = _schedule(Exposure(chip="A", path=frac_gds), chips=[Chip("A", ORIGIN, height=m)])
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), positions=POS, field_sizes=SIZES)
    assert [f.wd for f in ps[2].fields] == pytest.approx([15.1, 15.112, 15.122])
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(s, hw, **_calibrated()).run(N_FIXED) == "done" and len(hw.exposures) == 3
    moves = [c for c in hw.calls if c[0] == "move"]
    assert all(len(c) == 3 for c in moves[:-1])  # X, Y only: Z stays (the last one parks at the calibration sample)
    assert [c[1] for c in hw.calls if c[0] == "wd"][:3] == pytest.approx([15.1, 15.112, 15.122])
    assert hw.z == 15000.0
    # constant: one WD
    s.chips[0].height = ChipHeight("constant", wd_mm=15.3, z_um=15000.0)
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(s, hw, **_calibrated()).run(N_FIXED) == "done"
    assert {c[1] for c in hw.calls if c[0] == "wd"} == {15.3, 15.0}  # the chip, then WD1 when parked
    # refused before the stage moves: not recorded; recorded at another Z than Z_cal; the stage not at Z_cal
    for h, z, why in ((ChipHeight("constant"), 15000.0, "not recorded"),
                      (ChipHeight("constant", wd_mm=15.3, z_um=15100.0), 15000.0, "not at the calibration sample's Z"),
                      (ChipHeight("constant", wd_mm=15.3, z_um=15000.0), 15200.0, "the stage is at Z 15200.0")):
        s.chips[0].height = h
        hw = SimulatedHardware(start=ORIGIN, start_z=z)
        log = []
        r = ScheduleRunner(s, hw, **_calibrated(), on_event=lambda k, d: log.append(str(d.get("message", ""))))
        assert r.run(N_FIXED) == "failed" and not hw.exposures
        assert not any(c[0] == "move" for c in hw.calls) and any(why in t for t in log), (why, log)
    # a WD off the scale (the plane far out): refused at that field
    s.chips[0].height = ChipHeight("map", points=[(0, 0, 15.0), (1, 0, 25.0), (0, 1, 15.0)], z_um=15000.0)
    r = ScheduleRunner(s, SimulatedHardware(start=ORIGIN), **_calibrated())
    assert r.run(N_FIXED) == "failed"


def test_the_field_size_and_run_from_k(frac_gds):
    s = _schedule(Exposure(chip="A", path=frac_gds), chips=[Chip("A", ORIGIN)])
    log = []

    def on(kind, d):
        log.append(str(d.get("message", d.get("text", ""))))

    # the Microscope window at another field size: refused before anything happens
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(s, hw, **_calibrated(), selected="50 µm", on_event=on).run() == "failed" and not hw.calls
    assert any("the Microscope window is at field size 50 µm" in t for t in log)
    assert ScheduleRunner(s, SimulatedHardware(start=ORIGIN), **_calibrated(), selected="100 µm").run(N_FIXED) == "done"
    # run from k: the column at the field size's magnification, the stage at Z_cal, a beam calibration
    for hw_kw, kw, why in (({}, {"beamcals": []}, "no beam calibration"),
                           ({"magnification": 2000}, {}, "not the field size's ×1000"),
                           ({"start_z": 14000.0}, {}, "not at the calibration sample's Z")):
        log.clear()
        hw = SimulatedHardware(start=ORIGIN, **hw_kw)
        r = ScheduleRunner(s, hw, **dict(_calibrated(), **kw), on_event=on)
        assert r.run(N_FIXED) == "failed" and not hw.exposures and hw.calls == [("prepare",)], why
        assert any(why in t for t in log), (why, log)
    # no field calibration of the field size: the nominal rotation (logged)
    log.clear()
    kw = _calibrated()
    kw["calibration"] = None
    assert ScheduleRunner(s, SimulatedHardware(start=ORIGIN), **kw, on_event=on).run(N_FIXED) == "done"
    assert any("no field calibration" in t for t in log)
    # a start within the fixed steps runs from the first step
    hw = SimulatedHardware(start=ORIGIN, wd_mm=13.0)
    r = ScheduleRunner(s, hw, **_runnable(s))
    assert r.run(1) == "done" and hw.calls[1:3] == [("mag", 1000), ("wd", 15.0)] and r.states == ["done"] * 3
    # later Column settings change the beam only
    s.add(ColumnSetting(preset="default", current_pa=50.0), Exposure(chip="A", path=frac_gds))
    hw = SimulatedHardware(start=ORIGIN)
    r = ScheduleRunner(s, hw, **_calibrated())
    assert r.run(N_FIXED) == "done" and ("preset", "default") in hw.calls and r.current_pa == 50.0
    assert [c for c in hw.calls if c[0] == "mag"] == []


def test_a_run_starts_with_the_sem_running_and_parks_only_when_done(frac_gds):
    s = _schedule(Exposure(chip="A", path=frac_gds), chips=[Chip("A", ORIGIN)])
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(s, hw, **_calibrated()).run(N_FIXED) == "done"
    assert hw.calls[0] == ("prepare",) and hw.calls[-2:] == [("move", 1000.0, 1000.0, 15000.0), ("wd", 15.0)]
    s[N_FIXED].chip = "missing"  # fails: the stage stays
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(s, hw, **_calibrated()).run(N_FIXED) == "failed"
    assert hw.calls == [("prepare",)]


def test_global_align_in_a_dummy_field_0(demo):
    """The Global Align markers in no field (but within one field size): Field 0 is aligned first,
    nothing written there, then every field in cell-name order; markers wider than a field: the
    Exposure cannot run."""
    gds = Schedule.load(demo["schedule"]).exposures[0].path
    chip = Chip("C", ORIGIN)
    e = Exposure(chip="C", path=gds, layers=[1], global_align=True, global_markers=[(-200.0, -200.0), (-160.0, -160.0)])
    s = _schedule(e, fs="250 µm", chips=[chip])
    hw = SimulatedHardware(_rot(0.3), **DEMO_HW)
    events = []
    kw = _calibrated(mag=500, fs="250 µm")
    kw["field_sizes"] = None
    r = ScheduleRunner(s, hw, on_event=lambda k, d: events.append((k, d)), **kw)
    assert r.run(N_FIXED) == "done" and len(hw.exposures) == 9
    names = [d["name"] for k, d in events if k == "field" and d["state"] == "running"]
    assert names[0] == FIELD0_NAME and names[1:] == sorted(names[1:]) and len(names) == 10
    e.global_markers = [(-200.0, -200.0), (200.0, 200.0)]  # wider than a field: cannot run
    log = []
    r = ScheduleRunner(s, SimulatedHardware(_rot(0.3), **DEMO_HW), on_event=lambda k, d: log.append(d.get("text", "")),
                       **kw)
    assert r.run(N_FIXED) == "failed" and any("do not fit in one field" in t for t in log)


def test_local_markers_cover_as_many_fields_as_possible(frac_gds):
    """Nearest-centre alone would give F1 both markers; the assignment covers both fields."""
    lib = PatternLibrary()
    e = Exposure(chip="c", path=frac_gds, local_align=True, local_markers=[(50.0, 0.0), (10.0, 0.0)])
    f = exposure_fields(e, lib)
    a = assign_markers(f, e, 200.0)  # ±100 µm: (50, 0) is in F1 and F2 #2, (10, 0) only in F1
    assert a.marker_field == [1, 0] and a.local == {0: (10.0, 0.0), 1: (50.0, 0.0)}
    assert any("1 field(s) without a Local marker: F2 #3" in w for w in a.warnings)
    # several ways to cover the same fields: the nearest centres
    e.local_markers = [(20.0, 0.0), (5.0, 0.0), (130.0, 0.0)]
    a = assign_markers(f, e, 200.0)
    assert a.local[0] == (5.0, 0.0) and a.marker_field[0] in (1, 2) and a.marker_field[2] in (1, 2)
