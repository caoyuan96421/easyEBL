"""Lithography module: transforms, schedule model / file, fractured-GDS importer, runner, demo files."""
import json
import math

import numpy as np
import pytest

from easyebl.core.transforms import AffineTransform
from easyebl.litho.coords import FieldTransform, GlobalAlignment, StageAxes, check_field, fit_global, fit_local
from easyebl.litho.geometry import Dot, Pattern, Polyline, rectangle
from easyebl.litho.runner import ScheduleRunner, SimulatedHardware, params_for_dwell
from easyebl.litho.schedule import (KEEP_AS_IS, ColumnSetting, ExposeField, FieldCorrection, OP_TYPES, PatternLibrary,
                                    PatternSpec, Schedule, ScheduleError, dose_readout, dummy_pattern, plan)
from easyebl.litho.settings import LithoSettings


def _nocal(warnings) -> list:
    """The warnings other than the missing field calibration."""
    return [w for w in warnings if not w.startswith("no field calibration")]


CAL = None  # set below: a FieldCalConfig at a usable position, no settling


def _rot(deg, s=1.0):
    t = math.radians(deg)
    return s * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


@pytest.fixture
def frac_gds(tmp_path):
    """Three 80 µm fields placed by a top cell (two of them through a 1×2 array)."""
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    f1 = lib.new_cell("F1")
    f1.add(gdstk.rectangle((0, 0), (80, 80), layer=1))
    f1.add(gdstk.rectangle((10, 10), (20, 20), layer=2))
    f2 = lib.new_cell("F2")
    f2.add(gdstk.rectangle((0, 0), (40, 80), layer=1))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(f1, (0, 0)))
    top.add(gdstk.Reference(f2, (100, 0), columns=2, rows=1, spacing=(100, 0)))
    path = tmp_path / "frac.gds"
    lib.write_gds(str(path))
    return path


# ====================================================================== transforms
def test_identity_and_axes():
    t = FieldTransform()
    p = np.array([[3.0, -4.0], [10.0, 20.0]])
    assert t.is_identity and np.allclose(t.to_bcs(p), p)
    ax = StageAxes(mirror_y=True)
    w = ax.scs_to_wcs([[110.0, 190.0]], (100.0, 200.0))
    assert np.allclose(w, [[10.0, 10.0]]) and np.allclose(ax.wcs_to_scs(w, (100.0, 200.0)), [[110.0, 190.0]])
    assert StageAxes.from_dict(ax.to_dict()) == ax


def test_global_from_two_markers_rotation_scale_shift():
    A, t = _rot(0.7, 1.004), np.array([0.35, -1.2])
    w = np.array([[-40.0, -40.0], [40.0, 40.0]])
    tr, res = fit_global(w, w @ A.T + t)
    assert np.allclose(tr.linear, A) and np.allclose(tr.shift, t) and np.allclose(res, 0, atol=1e-9)
    p = tr.params()
    assert p["rotation_mrad"] == pytest.approx(math.radians(0.7) * 1e3) and p["scale"] == pytest.approx(1.004)
    with pytest.raises(ValueError):
        fit_global(w[:1], w[:1])
    with pytest.raises(ValueError):
        fit_global([w[0], w[0]], [w[0], w[0]])


def test_global_from_four_markers_finds_stretch():
    A = np.array([[1.002, 0.001], [-0.0005, 0.998]])
    w = np.array([[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]])
    tr, res = fit_global(w, w @ A.T + (1.0, 2.0))
    assert np.allclose(tr.linear, A) and np.allclose(tr.shift, (1.0, 2.0))
    assert "stretch" in tr.describe()


def test_local_xy_keeps_rotation_and_scale():
    prev = FieldTransform(_rot(1.0, 0.99), (0.5, 0.5))
    tr = fit_local((20.0, 30.0), (21.0, 28.0), prev)
    assert np.allclose(tr.linear, prev.linear) and np.allclose(tr.to_bcs([20.0, 30.0]), [21.0, 28.0])


def test_corrected_centre_uses_the_spacing_transformed_by_the_global_alignment():
    A = _rot(0.5, 1.002)
    for axes in (StageAxes(), StageAxes(mirror_x=True)):
        m = axes.matrix
        g = GlobalAlignment(A, ref_centre=(1000.0, 2000.0), ref_target=(1000.5, 1999.0), ref_shift=(0.2, -0.4))
        origin = np.array([1000.5, 1999.0]) + m @ np.array([0.2, -0.4])
        assert np.allclose(g.ref_origin(axes), origin)
        assert np.allclose(g.corrected_centre((1000.0, 2000.0), axes), origin)
        c = g.corrected_centre((1300.0, 2000.0), axes)
        assert np.allclose(c, origin + m @ A @ m @ np.array([300.0, 0.0]))
        assert np.hypot(*(c - origin)) == pytest.approx(300.0 * 1.002)
    assert GlobalAlignment.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_out_of_fov_check():
    pat = Pattern([rectangle(-40, -40, 40, 40), Dot(0, 0), Polyline(np.array([[0, 0], [45, 0]]))])
    assert check_field(pat, AffineTransform.identity(), 100.0).ok
    bad = check_field(pat, AffineTransform.translation(12, 0), 100.0)
    assert not bad.ok and bad.n_outside == 3 and bad.max_abs_um == pytest.approx(57)
    assert "outside the field" in bad.message()
    assert not check_field(Pattern([rectangle(-45, -45, 45, 45)]), AffineTransform.rotation(math.radians(30)), 100).ok


# ====================================================================== schedule
def test_schedule_always_starts_with_a_column_setting():
    s = Schedule()
    assert len(s) == 1 and isinstance(s[0], ColumnSetting) and s[0].preset == KEEP_AS_IS
    s = Schedule([ExposeField()])
    assert isinstance(s[0], ColumnSetting) and isinstance(s[1], ExposeField)
    assert not s.can_delete(0) and s.can_delete(1) and not s.can_move(1, 0)
    with pytest.raises(ScheduleError):
        ColumnSetting(magnification=0)


def _every_op(gds_path):
    return Schedule([
        ColumnSetting(preset="30pA", magnification=800, current_pa=31.5),
        ExposeField(label="gds", centre=(1000.0, 2000.0), correct_centre=False,
                    pattern=PatternSpec("gds", str(gds_path), "F1", [(1, 0)],
                                        placement=AffineTransform.translation(-40, -40)),
                    dwell_us=1.5, step_nm=25, global_align=True,
                    markers=[(-40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)], view_um=8.0),
        FieldCorrection(notes="later"),
        ColumnSetting(magnification=1000),
        ExposeField(centre=(1100.0, 2000.0), pattern=PatternSpec("dummy", size_um=30.0), local_xy=True,
                    local_marker=(-20.0, -20.0)),
        ExposeField(centre=(1200.0, 2000.0), pattern=PatternSpec("none"), enabled=False),
    ], name="every op")


def test_schedule_roundtrip(tmp_path, frac_gds):
    s = _every_op(frac_gds)
    assert {type(o).TYPE for o in s} == set(OP_TYPES)
    p = s.save(tmp_path / "sub" / "run.ebls.json")
    raw = json.loads(open(p, encoding="utf-8").read())
    assert raw["format"] == "easyebl-schedule" and raw["version"] == 2
    assert raw["operations"][1]["pattern"]["path"] == "../frac.gds"  # relative to the schedule file
    t = Schedule.load(p)
    d = str(tmp_path / "sub")
    assert t.to_dict(d) == s.to_dict(d)
    assert [o.title() for o in t] == [o.title() for o in s]
    assert t[1].pattern.path == str(frac_gds) and t[1].pattern.layers == [(1, 0)]
    assert t[1].markers == [(-40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)] and not t[1].correct_centre
    assert t[4].local_marker == (-20.0, -20.0) and not t[5].enabled and t[0].current_pa == 31.5
    lib = PatternLibrary()
    assert lib.get(t[1].pattern).bbox == pytest.approx((-40.0, -40.0, 40.0, 40.0))  # placement applied, layer 1 only
    assert len(lib.get(t[4].pattern)) == len(dummy_pattern(30.0)) and len(lib.get(t[5].pattern)) == 0


def test_schedule_file_errors(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"format": "other"}')
    with pytest.raises(ScheduleError):
        Schedule.load(p)
    with pytest.raises(ScheduleError):
        Schedule.from_dict({"format": "easyebl-schedule", "version": 1, "operations": []})  # the old model
    with pytest.raises(ScheduleError):
        Schedule.from_dict({"format": "easyebl-schedule", "version": 2, "operations": [{"type": "warp"}]})
    with pytest.raises(ScheduleError):
        PatternSpec("svg")


def test_dummy_pattern_and_dose():
    pat = dummy_pattern(20.0)
    assert pat.bbox == pytest.approx((-10.0, -10.0, 10.0, 10.0))
    assert {e.kind for e in pat} == {"area", "line", "dot"}
    p = params_for_dwell(2e-6, 0.02)
    assert p.area_dwell() == pytest.approx(2e-6) and p.dot_dwell() == pytest.approx(2e-6)
    assert p.line_dwell_per_um() * 0.02 == pytest.approx(2e-6)
    assert p.area_dwell(1.5) == pytest.approx(3e-6)  # dose factor multiplies the dwell
    assert "µC/cm²" in dose_readout(1.0, 20.0, 100.0) and "unknown" in dose_readout(1.0, 20.0, None)
    # 1 µs at 100 pA on a 20 nm grid = 100e-12 * 1e-6 / (20e-7 cm)^2 = 25 µC/cm²
    assert dose_readout(1.0, 20.0, 100.0).startswith("≙ 25 µC/cm²")


def test_plan_identity_fov_and_warnings():
    s = Schedule([ColumnSetting(magnification=1000),
                  ExposeField(centre=(500.0, 500.0), pattern=PatternSpec("dummy", size_um=40)),
                  ExposeField(centre=(600.0, 500.0), pattern=PatternSpec("dummy", size_um=120)),
                  ExposeField(centre=(-1.0, 0.0), pattern=PatternSpec("none")),
                  ExposeField(centre=(700.0, 500.0), global_align=True, markers=[(0.0, 0.0)]),
                  ExposeField(centre=(800.0, 500.0), local_xy=True, local_marker=(60.0, 0.0))])
    ps = plan(s, 100_000.0)
    assert _nocal(ps[1].warnings) == [] and ps[1].target == (500.0, 500.0) and ps[1].transform.is_identity
    assert any(w.startswith("no field calibration for ×1000") for w in ps[1].warnings)
    assert any("outside the field" in w for w in ps[2].warnings)
    assert any("below 0" in w for w in ps[3].warnings) and any("only moves" in w for w in ps[3].warnings)
    assert any("2–4 markers" in w for w in ps[4].warnings)
    assert any("Local XY marker (WCS 60, 0) is outside the field" in w for w in ps[5].warnings)


def test_plan_magnification_change_needs_a_global_align():
    s = Schedule([ColumnSetting(magnification=1000, preset="nope"),
                  ExposeField(centre=(500.0, 500.0), global_align=True, markers=[(-30, -30), (30, 30)]),
                  ColumnSetting(magnification=2000),
                  ExposeField(centre=(600.0, 500.0), pattern=PatternSpec("dummy", size_um=10)),
                  ExposeField(centre=(700.0, 500.0), global_align=True, markers=[(-10, -10), (10, 10)],
                              pattern=PatternSpec("dummy", size_um=10)),
                  FieldCorrection()])
    ps = plan(s, 100_000.0, presets=["p"])
    assert any("no beam preset" in w for w in ps[0].warnings)
    assert ps[3].field_um == pytest.approx(50.0)
    assert any("magnification changed" in w for w in ps[3].warnings)  # the calibration is kept, but warned
    assert not any("magnification" in w for w in ps[4].warnings)
    assert ps[5].warnings


def test_plan_corrects_later_centres_by_a_known_global_alignment():
    g = GlobalAlignment(_rot(1.0), (500.0, 500.0), (500.0, 500.0), (0.0, 0.0))
    s = Schedule([ColumnSetting(), ExposeField(centre=(600.0, 500.0)),
                  ExposeField(centre=(600.0, 500.0), correct_centre=False)])
    ps = plan(s, 100_000.0, glob=g)
    assert np.allclose(ps[1].target, np.array([500.0, 500.0]) + _rot(1.0) @ [100.0, 0.0])
    assert ps[2].target == (600.0, 500.0)
    assert np.allclose(ps[1].transform.linear, _rot(1.0))


# ====================================================================== importer
def test_importer_references(frac_gds):
    from easyebl.litho.fracture import ImportOptions, find_fields, import_fractured_gds

    fields = find_fields(frac_gds)
    assert [f.name for f in fields] == ["F1 #1", "F2 #2", "F2 #3"]
    assert [f.center for f in fields] == [(40.0, 40.0), (120.0, 40.0), (220.0, 40.0)]
    s = import_fractured_gds(frac_gds, ImportOptions(origin_scs=(1000.0, 2000.0), magnification=1000,
                                                     beam_preset="p1", global_align=True,
                                                     global_markers=((-35, -35), (35, 35)), local_xy=True,
                                                     local_marker=(-15, -35), dwell_us=3.0))
    assert [o.TYPE for o in s] == ["column", "expose", "expose", "expose"]
    assert s[0].preset == "p1" and s[0].magnification == 1000
    assert [o.centre for o in s.operations[1:]] == [(1040.0, 2040.0), (1120.0, 2040.0), (1220.0, 2040.0)]
    assert s[1].global_align and s[1].markers == [(-35.0, -35.0), (35.0, 35.0)]  # WCS: they move with the field
    assert not s[2].global_align and s[3].local_marker == (-15.0, -35.0) and s[3].dwell_us == 3.0
    assert all(o.correct_centre for o in s.operations[1:])
    lib = PatternLibrary()
    assert lib.get(s[3].pattern).bbox == pytest.approx((-20.0, -40.0, 20.0, 40.0))  # recentred: WCS
    ps = plan(s, 100_000.0, presets=["p1"])
    assert all(_nocal(p.warnings) == [] for p in ps), [p.warnings for p in ps]


def test_importer_mirrored_axes_cells_layers_and_order(frac_gds):
    from easyebl.litho.fracture import ImportOptions, build_schedule, find_fields, order_fields

    cells = find_fields(frac_gds, ImportOptions(source="cells", cells=["F1", "F2"], layers=[1]))
    assert [f.name for f in cells] == ["F1", "F2"] and cells[0].n_elements == 1
    fields = find_fields(frac_gds)
    assert [f.name for f in order_fields(list(reversed(fields)), "raster")] == ["F1 #1", "F2 #2", "F2 #3"]
    s = build_schedule(fields, ImportOptions(origin_scs=(1000.0, 2000.0), axes=StageAxes(mirror_x=True)))
    assert s[2].centre == (880.0, 2040.0)  # layout +x runs along stage −X


def test_import_roundtrip(tmp_path, frac_gds):
    from easyebl.litho.fracture import ImportOptions, import_fractured_gds

    s = import_fractured_gds(frac_gds, ImportOptions(global_align=True))
    t = Schedule.load(s.save(tmp_path / "imported.json"))
    lib = PatternLibrary()
    for a, b in zip(s, t):
        assert a.to_dict() == b.to_dict()
        if isinstance(a, ExposeField):
            assert lib.get(a.pattern).bbox == lib.get(b.pattern).bbox


# ====================================================================== settings
def test_stage_positions_and_axes_persist(tmp_path):
    p = str(tmp_path / "litho.json")
    s = LithoSettings.load(p)
    s.add_position("Faraday cup", 12000, 3000, "corner")
    s.add_position("Cal sample", 500.5, 600.25)
    with pytest.raises(ValueError):
        s.add_position("Cal sample", 1, 1)
    s.axes = StageAxes(mirror_y=True)
    s.save()
    t = LithoSettings.load(p)
    assert t.positions() == {"Faraday cup": (12000.0, 3000.0), "Cal sample": (500.5, 600.25)}
    assert t.stage_positions[0].note == "corner" and t.axes == StageAxes(mirror_y=True)
    assert t.unique_name("Cal sample") == "Cal sample 2"


# ====================================================================== runner
def _imported(frac_gds, axes=StageAxes(), local=True):
    from easyebl.litho.fracture import ImportOptions, import_fractured_gds

    return import_fractured_gds(frac_gds, ImportOptions(origin_scs=(5000.0, 4000.0), axes=axes, global_align=True,
                                                        global_markers=((-35, -35), (35, 35)), local_xy=local,
                                                        local_marker=(-15, -35)))


@pytest.mark.parametrize("axes", [StageAxes(), StageAxes(mirror_y=True)])
def test_runner_global_align_and_corrected_centres_land_exactly(frac_gds, axes):
    """Rotated / scaled sample, exact stage: the global alignment alone (centres corrected) places every field."""
    s = _imported(frac_gds, axes, local=False)
    hw = SimulatedHardware(_rot(0.5, 1.002), axes, pivot=(5040.0, 4040.0))
    r = ScheduleRunner(s, hw, axes)
    assert r.run() == "done" and r.states == ["done"] * 4
    fields = s.operations[1:]
    assert len(hw.exposures) == 3
    for op, e in zip(fields, hw.exposures):
        assert hw.placement_error(e, op.centre) < 1e-6
    assert np.allclose(r.glob.linear, _rot(0.5, 1.002))
    assert hw.calls[0] == ("mag", 1000.0)
    # without the centre correction the far fields land off by the rotation × spacing
    for op in fields:
        op.correct_centre = False
    hw2 = SimulatedHardware(_rot(0.5, 1.002), axes, pivot=(5040.0, 4040.0))
    ScheduleRunner(s, hw2, axes).run()
    assert hw2.placement_error(hw2.exposures[2], fields[2].centre) > 1.0


def test_runner_local_xy_absorbs_stage_errors(frac_gds):
    s = _imported(frac_gds)
    hw = SimulatedHardware(_rot(0.3), stage_noise_um=1.0, pivot=(5040.0, 4040.0), seed=3)
    assert ScheduleRunner(s, hw).run() == "done"
    for op, e in zip(s.operations[1:], hw.exposures):
        assert hw.placement_error(e, op.centre) < 1e-6
    s2 = _imported(frac_gds, local=False)
    hw2 = SimulatedHardware(_rot(0.3), stage_noise_um=1.0, pivot=(5040.0, 4040.0), seed=3)
    ScheduleRunner(s2, hw2).run()
    assert max(hw2.placement_error(e, op.centre) for op, e in zip(s2.operations[1:], hw2.exposures)) > 0.1


def test_runner_dwell_preset_dummy_none_and_out_of_field():
    s = Schedule([ColumnSetting(preset="default", magnification=1000, current_pa=25.0),
                  ExposeField(centre=(100.0, 100.0), pattern=PatternSpec("dummy", size_um=20), dwell_us=2.5,
                              step_nm=10),
                  ExposeField(centre=(200.0, 100.0), pattern=PatternSpec("dummy", size_um=150)),
                  FieldCorrection(),
                  ExposeField(centre=(300.0, 100.0), pattern=PatternSpec("none"), local_xy=True,
                              local_marker=(5.0, 0.0))])
    hw = SimulatedHardware()
    hw.answer = True  # skip the field that does not fit
    r = ScheduleRunner(s, hw, fieldcal=CAL)
    assert r.run() == "done"
    assert r.states == ["done", "done", "skipped", "done", "done"]  # the Field correction calibrates
    assert len(hw.exposures) == 1 and hw.preset == "default"
    p = hw.exposures[0]["params"]
    assert p.area_dwell() == pytest.approx(2.5e-6) and p.area_step_um == pytest.approx(0.01)
    assert any(c[0] == "ask" for c in hw.calls)
    assert len(hw.measurements) == 1  # the alignment-only field measured its marker
    hw2 = SimulatedHardware()  # ask → No aborts
    r2 = ScheduleRunner(s, hw2, fieldcal=CAL)
    assert r2.run() == "aborted" and r2.states[2] == "aborted"
    s.settings.on_out_of_field = "skip"
    assert ScheduleRunner(s, SimulatedHardware(), fieldcal=CAL).run() == "done"


def test_runner_keeps_the_global_alignment_over_a_magnification_change():
    s = Schedule([ColumnSetting(magnification=1000),
                  ExposeField(centre=(500.0, 500.0), global_align=True, markers=[(-30, -30), (30, 30)],
                              pattern=PatternSpec("none")),
                  ColumnSetting(magnification=2000),
                  ExposeField(centre=(600.0, 500.0), pattern=PatternSpec("dummy", size_um=10))])
    logs = []
    hw = SimulatedHardware(_rot(0.4), pivot=(500.0, 500.0))
    r = ScheduleRunner(s, hw, on_event=lambda k, d: logs.append(d.get("text", "")))
    assert r.run() == "done"
    assert np.allclose(r.glob.linear, _rot(0.4)) and np.allclose(hw.exposures[0]["transform"].linear, _rot(0.4))
    assert hw.exposures[0]["field_um"] == pytest.approx(50.0)
    assert any("magnification changed" in t for t in logs)


def test_shift_fields_moves_centres_and_markers_with_them():
    s = Schedule([ColumnSetting(), ExposeField(centre=(100.0, 200.0), global_align=True, markers=[(-5, -5), (5, 5)]),
                  FieldCorrection(), ExposeField(centre=(300.0, 200.0), local_xy=True, local_marker=(1.0, 2.0))])
    ax = StageAxes(mirror_x=True)
    before = s[1].nominal_scs(s[1].markers, ax)
    assert s.shift_fields(1000.0, -50.0) == 2
    assert [f.centre for f in s.fields] == [(1100.0, 150.0), (1300.0, 150.0)]
    assert s[1].markers == [(-5.0, -5.0), (5.0, 5.0)] and s[3].local_marker == (1.0, 2.0)
    assert np.allclose(s[1].nominal_scs(s[1].markers, ax), before + (1000.0, -50.0))
    assert np.allclose(before, [[105.0, 195.0], [95.0, 205.0]])  # stage X mirrored


def test_runner_errors_abort_and_start():
    s = Schedule([ColumnSetting(), ExposeField(centre=(10.0, 10.0), global_align=True, markers=[(1.0, 1.0)]),
                  ExposeField(centre=(20.0, 20.0))])
    r = ScheduleRunner(s, SimulatedHardware())
    assert r.run() == "failed" and r.states[1] == "failed"
    r = ScheduleRunner(s, SimulatedHardware())
    r.abort()
    assert r.run() == "aborted"
    hw = SimulatedHardware()
    r = ScheduleRunner(s, hw)
    assert r.run(start=2) == "done" and r.states == ["done", "pending", "done"]  # the Column setting always runs

    class NoClick(SimulatedHardware):
        def measure_marks(self, requests):
            return None

    s2 = Schedule([ColumnSetting(), ExposeField(local_xy=True)])
    assert ScheduleRunner(s2, NoClick()).run() == "aborted"


def test_demo_files(tmp_path):
    pytest.importorskip("gdstk")
    from easyebl.litho.demo import make_demo_files

    out = make_demo_files(str(tmp_path / "demo"))
    s = Schedule.load(out["schedule"])
    assert len(s) == 11 and isinstance(s[1], FieldCorrection) and s[2].global_align
    assert all(o.local_xy for o in s.fields)
    assert s[2].centre == (5000.0, 5000.0) and s[3].centre == (5100.0, 5000.0)
    ps = plan(s, 100_000.0, fieldcal=CAL)
    assert all(p.warnings == [] for p in ps), [p.warnings for p in ps]  # calibrated by the Field correction
    assert any("position is not set" in w for w in plan(s, 100_000.0)[1].warnings)
    # (a stage landing 0.5 µm off at random would spoil the calibration fit: it refuses such a fit)
    hw = SimulatedHardware(_rot(0.3, 1.001), stage_noise_um=0.2, pivot=(5000.0, 5000.0))
    assert ScheduleRunner(s, hw, fieldcal=CAL).run() == "done" and len(hw.exposures) == 9
    for op, e in zip(s.fields, hw.exposures):
        assert hw.placement_error(e, op.centre) < 1e-6
    d = Schedule.load(out["dummy"])
    assert all(p.warnings == [] for p in plan(d, 100_000.0, fieldcal=CAL))
    assert ScheduleRunner(d, SimulatedHardware(), fieldcal=CAL).run() == "done"


def test_magnification_is_a_whole_permitted_value():
    from easyebl.litho.schedule import MAGNIFICATIONS, nearest_magnification
    from easyebl.microscope.s3000.commands import parse_magnification

    c = ColumnSetting(magnification=900.0)  # e.g. read back from an older file / a spin box
    assert type(c.magnification) is int and c.magnification == 900 and c.permitted
    assert parse_magnification(c.magnification) == 900  # what the SEM driver accepts
    t = Schedule.from_dict(Schedule([c]).to_dict())
    assert type(t[0].magnification) is int
    assert 900 in MAGNIFICATIONS and nearest_magnification(920) == 900 and nearest_magnification(1050) == 1000
    bad = Schedule([ColumnSetting(magnification=920)])
    assert any("not one the SEM accepts (nearest: ×900)" in w for w in plan(bad, 100_000.0)[0].warnings)
    r = ScheduleRunner(bad, SimulatedHardware())
    assert r.run() == "failed" and r.states[0] == "failed"
    hw = SimulatedHardware()
    assert ScheduleRunner(Schedule([ColumnSetting(magnification=900.0)]), hw).run() == "done"
    assert hw.calls == [("mag", 900)] and type(hw.calls[0][1]) is int



# ====================================================================== stage, field calibration
from easyebl.litho.fieldcal import (Condition, FieldCalConfig, FieldCalError, FieldCalResult,  # noqa: E402
                                    fit_stage_to_beam, measure_field_calibration, phase_correlation, shift_pattern)
from easyebl.litho.runner import StageError  # noqa: E402

CAL = FieldCalConfig(x=1000.0, y=1000.0, settle_s=0.0)


def _beam(deg=0.5, s=1.01):
    return _rot(deg, s)


def test_one_way_approach_plan():
    from types import SimpleNamespace

    from easyebl.microscope.stage import StageControl

    st = SimpleNamespace(backlash_um=50.0)
    plan_ = lambda t, c: StageControl.approach_plan(st, t, c)  # noqa: E731
    assert plan_({"X": 300.0}, {"X": 100.0}) == [{"X": 300.0}]  # ends going + by ≥ 50: direct
    assert plan_({"X": 300.0}, {"X": 280.0}) == [{"X": 250.0}, {"X": 300.0}]  # + but too short
    assert plan_({"X": 100.0, "Y": 500.0}, {"X": 300.0, "Y": 100.0}) == [{"X": 50.0}, {"X": 100.0, "Y": 500.0}]
    assert plan_({"X": 20.0}, {"X": 300.0}) == [{"X": 0.0}, {"X": 20.0}]  # never below the stage zero
    assert plan_({"Z": 10.0}, {"Z": 300.0}) == [{"Z": 10.0}]  # only X and Y
    st.backlash_um = 0.0
    assert plan_({"X": 100.0}, {"X": 300.0}) == [{"X": 100.0}]


def test_stage_position_is_verified():
    s = Schedule([ColumnSetting(), ExposeField(centre=(100.0, 100.0))])
    r = ScheduleRunner(s, SimulatedHardware(stall_um=(2.0, 0.5)))
    assert r.run() == "failed" and r.states[1] == "failed"
    assert ScheduleRunner(s, SimulatedHardware(stall_um=(2.0, 0.5)), stage_tolerance_um=3.0).run() == "done"
    assert ScheduleRunner(s, SimulatedHardware(stage_noise_um=5.0)).run() == "done"  # reads back right


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
    B = _beam()
    ds = shift_pattern("grid", 30.0)
    assert len(ds) == 9 and tuple(ds[0]) == (0.0, 0.0) and len(shift_pattern("cross", 1.0)) == 5
    b, drift, res = fit_stage_to_beam(ds, -ds @ B.T + (0.2, -0.1))
    assert np.allclose(b, B) and np.allclose(drift, (0.2, -0.1)) and np.allclose(res, 0, atol=1e-12)
    with pytest.raises(FieldCalError):
        fit_stage_to_beam([[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]])


@pytest.mark.parametrize("axes", [StageAxes(), StageAxes(mirror_y=True)])
def test_field_calibration_recovers_the_stage_to_beam_map(axes):
    B = _beam() @ axes.matrix
    hw = SimulatedHardware(axes=axes, stage_to_beam=B, start=(1000.0, 1000.0))
    res = measure_field_calibration(hw, CAL, (1000.0, 1000.0), Condition(1000, None), axes=axes)
    assert np.allclose(res.stage_to_beam, B, atol=1e-3)  # ≈ 0.1 % from noisy simulated images
    from easyebl.litho.coords import linear_params

    p = linear_params(res.correction(axes))
    assert p["rotation_mrad"] == pytest.approx(8.727, abs=0.2) and p["scale"] == pytest.approx(1.01, abs=5e-4)
    assert res.n_images == 9 and hw.images == 9 and "MIRRORED" not in res.describe(axes)
    assert hw.stage_position() == (1000.0, 1000.0)  # back at the position
    assert res.valid_for(Condition(1000, None)) and not res.valid_for(Condition(2000, None))
    assert not res.valid_for(Condition(1000, "other preset"))
    with pytest.raises(FieldCalError):  # shifts below the stage zero
        measure_field_calibration(hw, CAL, (5.0, 5.0), Condition(1000, None))


def test_calibration_corrects_exposures_until_the_column_changes():
    B = _beam()
    s = Schedule([ColumnSetting(magnification=1000), FieldCorrection(),
                  ExposeField(centre=(2000.0, 2000.0), pattern=PatternSpec("dummy", size_um=60)),
                  ColumnSetting(magnification=1000, preset="default"),
                  ExposeField(centre=(2100.0, 2000.0), pattern=PatternSpec("dummy", size_um=60))])
    hw = SimulatedHardware(stage_to_beam=B)
    events = []
    r = ScheduleRunner(s, hw, fieldcal=CAL, on_event=lambda k, d: events.append((k, d)))
    assert r.run() == "done"
    assert hw.placement_error(hw.exposures[0], s[2].centre) < 0.15  # calibrated: ≈ 0.1 µm over 60 µm
    assert np.allclose(hw.exposures[0]["transform"].linear, r.calibration.stage_to_beam)
    assert hw.exposures[1]["transform"].is_identity()  # the preset changed: the calibration no longer applies
    assert hw.placement_error(hw.exposures[1], s[4].centre) > 0.4  # 1 % of 30 µm + the rotation
    assert any(k == "calibration" for k, _ in events)
    ps = plan(s, 100_000.0, fieldcal=CAL)
    assert _nocal(ps[2].warnings) == ps[2].warnings and ps[2].calibrated  # a Field correction before it
    assert not ps[4].calibrated and any(w.startswith("no field calibration for ×1000, beam 'default'")
                                        for w in ps[4].warnings)
    assert any("no stage position named 'cup'" in w
               for w in plan(Schedule([ColumnSetting(), FieldCorrection(position="cup")]), 1e5)[1].warnings)


def test_run_calibration_alone_and_the_corrected_centre():
    B = _beam()
    hw = SimulatedHardware(stage_to_beam=B, magnification=1000)
    events = []
    r = ScheduleRunner(Schedule(), hw, fieldcal=CAL, on_event=lambda k, d: events.append((k, d)))
    assert r.run_calibration() == "done"
    res = [d["result"] for k, d in events if k == "calibration"][0]
    assert res.condition == Condition(1000.0, None)
    # with a calibration, the corrected centre goes through B⁻¹ (stage µm), not the nominal axes
    g = GlobalAlignment(_rot(0.2) @ res.stage_to_beam, (500.0, 500.0), (500.0, 500.0), (0.0, 0.0))
    from easyebl.litho.coords import StageFrame

    fr = StageFrame.make(StageAxes(), res.stage_to_beam)
    c = g.corrected_centre((600.0, 500.0), fr)
    assert np.allclose(c, [500.0, 500.0] + np.linalg.inv(fr.B) @ g.linear @ [100.0, 0.0])
    assert np.hypot(*(c - (500.0, 500.0))) == pytest.approx(100.0, abs=0.3)  # stage µm stay true µm


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
    stage.positions = {"X": 1010.0, "Y": 2000.0}  # +10 µm in X: the sample moves −10 µm in the field
    assert np.allclose(be._sample_at(x - 10.0, y), before)
    stage.positions = {"X": 1000.0 + 120.0, "Y": 2000.0}  # the demo sample repeats every 120 µm
    assert np.allclose(be._sample_at(x, y), before)
