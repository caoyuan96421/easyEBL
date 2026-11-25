"""Lithography module: coordinate maps, chips / exposures, marker assignment, plan, runner,
field and beam calibration through the runner, settings, demo files."""
import math

import numpy as np
import pytest

from easyebl.litho.beamcal import BeamCalConfig, BeamCalibration
from easyebl.litho.coords import (S3000_AXES, ChipTransform, FieldMap, StageAxes, check_field, fit_chip,
                                  linear_params, stage_rotation)
from easyebl.litho.fieldcal import (Condition, FieldCalConfig, FieldCalError, fit_stage_to_beam,
                                    measure_field_calibration, phase_correlation, shift_pattern)
from easyebl.litho.geometry import Pattern, rectangle
from easyebl.litho.runner import ScheduleRunner, SimulatedHardware, StageError
from easyebl.litho.schedule import (Chip, ColumnSetting, Exposure, FieldCorrection, PatternLibrary, Schedule,
                                    ScheduleError, assign_markers, dose_readout, exposure_fields, find_beamcal, plan)
from easyebl.litho.settings import LithoSettings

CAL = FieldCalConfig(x=1000.0, y=1000.0, settle_s=0.0)
ORIGIN = (5000.0, 5000.0)


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
    assert a.local == {0: (-40.0, 40.0), 2: (210.0, 5.0)}  # nearest wins, the global field's is dropped
    w = "\n".join(a.warnings)
    assert "is the Global Align field" in w and "(500, 0) is in no field" in w and "two Local markers" in w
    e.global_markers = [(0.0, 0.0), (120.0, 0.0)]  # in two fields
    a = assign_markers(f, e, 100.0)
    assert a.global_field is None and any("not all inside" in x for x in a.warnings)
    e.global_markers = [(0.0, 0.0)]
    assert any("2–4 markers" in x for x in assign_markers(f, e, 100.0).warnings)
    e2 = Exposure(chip="c", path=frac_gds, local_align=True, local_markers=[(0.0, 0.0)], order="raster")
    a = assign_markers(f, e2, 100.0)
    assert a.global_field is None and any("2 field(s) without a Local marker" in x for x in a.warnings)


def test_schedule_chips_roundtrip_and_errors(tmp_path, frac_gds):
    s = Schedule(name="t")
    c = s.add_chip(origin=ORIGIN)
    assert c.name == "Chip 1" and s.unique_chip_name() == "Chip 2"
    with pytest.raises(ScheduleError):
        s.add_chip("Chip 1")
    s.add(Exposure(chip="Chip 1", path=frac_gds, layers=[(1, 0)], offset=(5.0, -5.0), global_align=True,
                   global_markers=[(-10.0, -10.0), (10.0, 10.0)], local_align=True, local_markers=[(120.0, 0.0)]),
          FieldCorrection(position="cup"))
    s.rename_chip("Chip 1", "A")
    assert s.exposures[0].chip == "A" and s.chip("A").origin == ORIGIN
    s.shift_chip("A", 10.0, -5.0)
    assert s.chip("A").origin == (5010.0, 4995.0)
    p = s.save(str(tmp_path / "sub" / "x.ebls.json"))
    t = Schedule.load(p)
    assert t.to_dict() == s.to_dict() and t.exposures[0].path == s.exposures[0].path
    assert t.exposures[0].layers == [(1, 0)] and t.chips == s.chips
    d = s.to_dict()
    d["version"] = 2
    with pytest.raises(ScheduleError, match="version 2"):
        Schedule.from_dict(d)
    with pytest.raises(ScheduleError):
        Schedule.from_dict({"format": "other"})
    with pytest.raises(ScheduleError):
        Exposure(order="random")
    assert "µC/cm²" in dose_readout(1.0, 20.0, 100.0) and "unknown" in dose_readout(1.0, 20.0, None)


def test_magnification_is_a_whole_permitted_value():
    from easyebl.litho.schedule import MAGNIFICATIONS, nearest_magnification
    from easyebl.microscope.s3000.commands import parse_magnification

    c = ColumnSetting(magnification=900.0)
    assert type(c.magnification) is int and c.magnification == 900 and c.permitted
    assert parse_magnification(c.magnification) == 900
    assert type(Schedule.from_dict(Schedule([c]).to_dict())[0].magnification) is int
    assert 900 in MAGNIFICATIONS and nearest_magnification(920) == 900 and nearest_magnification(1050) == 1000
    bad = Schedule([ColumnSetting(magnification=920)])
    assert any("not one the SEM accepts (nearest: ×900)" in w for w in plan(bad, 100_000.0)[0].warnings)
    r = ScheduleRunner(bad, SimulatedHardware())
    assert r.run() == "failed" and r.states[0] == "failed"
    hw = SimulatedHardware()
    assert ScheduleRunner(Schedule([ColumnSetting(magnification=900.0)]), hw).run() == "done"
    assert hw.calls == [("mag", 900)] and type(hw.calls[0][1]) is int


# ====================================================================== plan
def test_plan_fields_targets_and_warnings(frac_gds):
    s = Schedule([ColumnSetting(magnification=1000)], chips=[Chip("A", ORIGIN)])
    s.add(Exposure(chip="A", path=frac_gds), Exposure(chip="nope", path=frac_gds),
          Exposure(chip="A", path=frac_gds, offset=(0.0, 0.0), global_align=True,
                   global_markers=[(-30.0, -30.0), (30.0, 30.0)]))
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary())
    p = ps[1]
    assert [f.target for f in p.fields] == [(5000.0, 5000.0), (4880.0, 5000.0), (4780.0, 5000.0)]  # X mirrored
    assert all(f.fov_check.ok for f in p.fields) and not p.calibrated
    assert any(w.startswith("no field calibration") for w in p.warnings)
    assert any(w.startswith("no beam calibration") for w in ps[0].warnings)
    assert any("no chip named 'nope'" in w for w in ps[2].warnings) and not ps[2].fields
    assert ps[3].fields[0].is_global and ps[3].fields[0].name.startswith("F1")
    # the same at ×2000 (50 µm field): F2 (40 × 80) does not fit
    s[0].magnification = 2000
    p = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary())[1]
    assert any("pattern outside the field of view" in w for w in p.warnings)
    # a measured chip map moves the targets
    g = ChipTransform(_rot(0.5) @ S3000_AXES.matrix, (5010.0, 4990.0), measured=True)
    s[0].magnification = 1000
    p = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), chips={"A": g})[1]
    assert np.allclose(p.fields[1].target, g.to_scs([120.0, 0.0]))


def test_plan_calibration_conditions(frac_gds):
    s = Schedule([ColumnSetting(magnification=1000), FieldCorrection(position="cup"), Exposure(chip="A", path=frac_gds),
                  ColumnSetting(magnification=1000, preset="other"), Exposure(chip="A", path=frac_gds)],
                 chips=[Chip("A", ORIGIN)])
    bc = BeamCalibration(sx=1.001, condition=Condition(1000, None, 15.0))
    ps = plan(s, 1e5, S3000_AXES, library=PatternLibrary(), beamcals=[bc], wd_mm=15.04, positions={})
    assert any("no stage position named 'cup'" in w for w in ps[1].warnings)
    assert ps[2].calibrated and ps[2].beamcal is bc  # WD within 0.1 mm
    assert not ps[4].calibrated and ps[4].beamcal is bc  # the preset changed: field calibration no longer
    ps = plan(s, 1e5, S3000_AXES, library=PatternLibrary(), beamcals=[bc], wd_mm=15.2)
    assert ps[2].beamcal is None and any(w.startswith("no beam calibration") for w in ps[0].warnings)
    assert find_beamcal([bc], Condition(1000, "anything", 14.95)) is bc  # the preset does not matter
    assert find_beamcal([bc], Condition(2000, None, 15.0)) is None


# ====================================================================== runner
def _demo_run(demo, hw, beamcals=(), local=True, **kw):
    s = Schedule.load(demo["schedule"])
    s.operations = [o for o in s.operations if not isinstance(o, FieldCorrection)]
    s[1].local_align = local
    lib = PatternLibrary()
    r = ScheduleRunner(s, hw, library=lib, beamcals=list(beamcals), **kw)
    state = r.run()
    flds = exposure_fields(s[1], lib)
    order = assign_markers(flds, s[1], hw.field_um()).order
    nom = s.chips[0].nominal(hw.axes)
    errs = [hw.placement_error(e, flds[k].centre_ccs, nom) for e, k in zip(hw.exposures, order)]
    return state, r, errs


TRUE_BEAM = dict(beam_linear=[[1.003, 0.001], [0.0, 0.998]], beam_k=2e-7)
TRUE_BC = BeamCalibration(sx=1.003, sy=0.998, shear=0.001, k=2e-7, condition=Condition(1000.0, None, 15.0))


def test_runner_global_and_local_align_land_exactly_with_the_beam_calibration(demo):
    hw = SimulatedHardware(_rot(0.25, 1.0005), pivot=ORIGIN, start=ORIGIN, **TRUE_BEAM)
    state, r, errs = _demo_run(demo, hw, [TRUE_BC], local=False)
    assert state == "done" and len(errs) == 9 and max(errs) < 2e-3  # the one-step inverse: ≈ nm
    g = r.chips["Demo chip"]
    assert g.measured and linear_params(g.correction(S3000_AXES))["rotation_mrad"] == pytest.approx(
        -math.radians(0.25) * 1e3, abs=1e-3)  # the sample turned in the beam frame: −θ in the mirrored stage frame
    assert len(hw.measurements) == 2  # the two Global Align markers
    # without the beam calibration the beam's scale / shear / pincushion is written as it is
    hw = SimulatedHardware(_rot(0.25, 1.0005), pivot=ORIGIN, start=ORIGIN, **TRUE_BEAM)
    _, _, errs = _demo_run(demo, hw, [], local=False)
    assert max(errs) > 0.3
    # Local markers absorb stage errors (the stage goes 0.3 µm off, reading back right)
    hw = SimulatedHardware(_rot(0.25), pivot=ORIGIN, start=ORIGIN, stage_noise_um=0.3, seed=3, **TRUE_BEAM)
    state, _, errs = _demo_run(demo, hw, [TRUE_BC], local=True)
    assert state == "done" and max(errs[1:]) < 2e-3 and len(hw.measurements) == 2 + 8
    hw = SimulatedHardware(_rot(0.25), pivot=ORIGIN, start=ORIGIN, stage_noise_um=0.3, seed=3, **TRUE_BEAM)
    _, _, errs = _demo_run(demo, hw, [TRUE_BC], local=False)
    assert max(errs) > 0.1


def test_global_align_holds_for_later_exposures_of_the_chip(demo):
    from easyebl.litho.demo import MARK

    gds = Schedule.load(demo["schedule"]).exposures[0].path
    chip = Chip("C", ORIGIN)
    s = Schedule([ColumnSetting(), Exposure(chip="C", global_align=True, global_markers=[(-MARK, -MARK), (MARK, MARK)]),
                  Exposure(chip="C", path=gds, layers=[(1, 0)])], chips=[chip])
    hw = SimulatedHardware(_rot(0.3), pivot=ORIGIN, start=ORIGIN)
    events = []
    r = ScheduleRunner(s, hw, on_event=lambda k, d: events.append((k, d)))
    assert r.run() == "done" and r.states == ["done"] * 3 and len(hw.exposures) == 9
    lib = PatternLibrary()
    flds = exposure_fields(s[2], lib)
    errs = [hw.placement_error(e, f.centre_ccs, chip.nominal(hw.axes)) for e, f in zip(hw.exposures, flds)]
    assert max(errs) < 1e-6
    assert [d["chip"] for k, d in events if k == "alignment"] == ["C"]
    fields = [d for k, d in events if k == "field"]
    assert fields[0]["state"] == "running" and fields[-1]["state"] == "done" and fields[-1]["n"] == 9


def test_runner_out_of_field_abort_errors_and_stage_check(frac_gds):
    s = Schedule([ColumnSetting(magnification=2000), Exposure(chip="A", path=frac_gds)], chips=[Chip("A", ORIGIN)])
    hw = SimulatedHardware(start=ORIGIN)
    hw.answer = True  # skip
    r = ScheduleRunner(s, hw)
    assert r.run() == "done" and not hw.exposures and sum(c[0] == "ask" for c in hw.calls) == 3  # 50 µm field
    s.settings.on_out_of_field = "abort"
    assert ScheduleRunner(s, SimulatedHardware(start=ORIGIN)).run() == "aborted"
    s[0].magnification = 1000
    r = ScheduleRunner(s, SimulatedHardware(start=ORIGIN, stall_um=(2.0, 0.5)))
    assert r.run() == "failed" and r.states[1] == "failed"
    assert ScheduleRunner(s, SimulatedHardware(start=ORIGIN, stall_um=(2.0, 0.5)), stage_tolerance_um=3.0).run() == "done"
    with pytest.raises(StageError):
        ScheduleRunner(s, SimulatedHardware(stall_um=(1.0, 0.0))).move_stage(0.0, 0.0)
    s[1].chip = "missing"
    r = ScheduleRunner(s, SimulatedHardware(start=ORIGIN))
    assert r.run() == "failed" and r.states[1] == "failed"
    s[1].chip = "A"
    s[1].global_align, s[1].global_markers = True, [(0.0, 0.0), (120.0, 0.0)]  # not in one field
    assert ScheduleRunner(s, SimulatedHardware(start=ORIGIN)).run() == "failed"
    hw = SimulatedHardware(start=ORIGIN)
    s[1].global_align = False
    r = ScheduleRunner(s, hw)
    r.abort()
    assert r.run() == "aborted"


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


def test_field_calibration_and_its_condition():
    b = _rot(0.5) @ S3000_AXES.matrix
    hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0))
    cond = Condition(1000, None, 15.0)
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), cond, axes=S3000_AXES)
    assert np.allclose(res.stage_to_beam, b, atol=1e-3) and res.beamcal_id is None
    assert hw.stage_position() == (1000.0, 1000.0)
    assert res.valid_for(Condition(1000, None, 15.08)) and not res.valid_for(Condition(1000, None, 15.2))
    assert not res.valid_for(Condition(1000, "x", 15.0)) and not res.valid_for(Condition(2000, None, 15.0))
    assert not res.valid_for(cond, TRUE_BC)  # measured without this beam calibration
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


def test_review_discard_and_refused():
    s = Schedule([ColumnSetting(magnification=1000), FieldCorrection()])
    seen, events = [], []
    r = ScheduleRunner(s, SimulatedHardware(stage_to_beam=_rot(0.5) @ S3000_AXES.matrix), fieldcal=CAL,
                       on_event=lambda k, d: events.append(k), review=lambda res: seen.append(res) and False)
    assert r.run() == "aborted" and r.calibration is None and len(seen) == 1
    assert events.count("calibration_image") == 9 and "calibration" not in events
    hw = SimulatedHardware()
    hw.capture_image = lambda w, px, frames=1: (np.random.default_rng(len(hw.calls)).normal(size=(px, px)),
                                                (w / px, w / px))
    seen = []
    r = ScheduleRunner(s, hw, fieldcal=CAL, review=lambda res: seen.append(res) or True)
    assert r.run() == "failed" and len(seen) == 1 and seen[0].problem and r.calibration is None


# ====================================================================== beam calibration
def test_beam_calibration_through_the_runner():
    hw = SimulatedHardware(grid_at=((2000.0, 2000.0), 50.0), wd_mm=12.5, **TRUE_BEAM)
    events, seen = [], []
    r = ScheduleRunner(Schedule(), hw, positions={"grid": (2000.0, 2000.0, 8.0)},
                       beamcal_config=BeamCalConfig(pixels=1000, position="grid"),
                       on_event=lambda k, d: events.append((k, d)), review_beam=lambda res: seen.append(res) or True)
    assert r.run_beam_calibration() == "done"
    bc = [d["result"] for k, d in events if k == "beamcal"][0]
    assert bc is seen[0] and bc.condition == Condition(1000.0, None, 12.5) and bc.position == (2000.0, 2000.0, 8.0)
    assert hw.calls[0] == ("move", 2000.0, 2000.0, 8.0) and hw.z == 8.0  # Go to XYZ
    assert bc.sx == pytest.approx(1.003, abs=2e-5) and bc.sy == pytest.approx(0.998, abs=2e-5)
    assert bc.shear == pytest.approx(0.001, abs=5e-5) and bc.k == pytest.approx(2e-7, rel=0.05)
    assert [k for k, _ in events if k.startswith("beamcal")] == ["beamcal_start", "beamcal_image", "beamcal"]
    assert r.beamcals == [bc] and r.beamcal(Condition(1000.0, "p", 12.45)) is bc
    # Discard; and a refused fit (no grid there) is shown, then fails
    r = ScheduleRunner(Schedule(), hw, positions={"grid": (2000.0, 2000.0)}, review_beam=lambda res: False,
                       beamcal_config=BeamCalConfig(pixels=1000, position="grid"))
    assert r.run_beam_calibration() == "aborted" and r.beamcals == []
    seen = []
    r = ScheduleRunner(Schedule(), hw, positions={"tex": (9000.0, 9000.0)}, review_beam=lambda res: seen.append(res),
                       beamcal_config=BeamCalConfig(pixels=600, position="tex"))
    assert r.run_beam_calibration() == "failed" and (not seen or seen[0].problem)
    assert ScheduleRunner(Schedule(), hw, beamcal_config=BeamCalConfig()).run_beam_calibration() == "failed"


# ====================================================================== settings, demo
def test_settings_persist_positions_and_beam_calibrations(tmp_path):
    s = LithoSettings(path=str(tmp_path / "l.json"))
    s.add_position("cup", 100.0, 200.0, 12.5)
    s.add_position("grid", 1.0, 2.0)
    with pytest.raises(ValueError):
        s.add_position("cup", 0, 0)
    s.beamcal = BeamCalConfig(pixels=4000, position="grid")
    s.set_beamcal(BeamCalibration(sx=1.001, condition=Condition(1000.0, None, 15.0)))
    s.set_beamcal(BeamCalibration(sx=1.002, condition=Condition(1000.0, None, 15.05)))  # replaces
    s.set_beamcal(BeamCalibration(sx=1.003, condition=Condition(2000.0, None, 15.0)))
    s.save()
    t = LithoSettings.load(s.path)
    assert t.positions() == {"cup": (100.0, 200.0, 12.5), "grid": (1.0, 2.0, None)}
    assert t.beamcal.pixels == 4000 and [b.sx for b in t.beamcals] == [1.002, 1.003]
    assert t.beamcals[0].condition == Condition(1000.0, None, 15.05)
    assert t.forget_beamcal(Condition(2000.0, None, 15.0)) and len(t.beamcals) == 1


def test_demo_files(demo):
    s = Schedule.load(demo["schedule"])
    assert [o.TYPE for o in s] == ["column", "field_correction", "exposure"] and s.chips[0].origin == ORIGIN
    ps = plan(s, 100_000.0, S3000_AXES, library=PatternLibrary(), fieldcal=CAL)
    p = ps[2]
    assert len(p.fields) == 9 and p.fields[0].is_global and sum(f.local_marker is not None for f in p.fields) == 8
    assert all(f.fov_check.ok for f in p.fields)
    d = Schedule.load(demo["dummy"])
    hw = SimulatedHardware(start=ORIGIN)
    assert ScheduleRunner(d, hw, fieldcal=CAL).run() == "done" and len(hw.exposures) == 9


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
    old = FieldCalConfig(x=1000.0, y=1000.0, settle_s=0.0, denoise_px=0.0, background_fraction=0.0, whitening=1.0,
                         max_residual_px=1e9)
    for contrast, noise in ((0.1, 0.3), (0.05, 0.5)):  # signal-to-noise 0.33 and 0.1
        for seed in range(3):
            hw = SimulatedHardware(stage_to_beam=b, start=(1000.0, 1000.0), image_contrast=contrast, image_noise=noise,
                                   seed=seed)
            cfg = FieldCalConfig(x=1000.0, y=1000.0, settle_s=0.0, max_residual_px=3.0)  # noisy: more scatter
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
