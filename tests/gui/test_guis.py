"""Smoke tests: every debug GUI builds and runs its main action on the simulator (offscreen Qt)."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.gui

from easyebl.gui.common import DataclassForm, Session, format_points, parse_points  # noqa: E402
from easyebl.hal import SimParams  # noqa: E402
from easyebl.imaging import RasterConfig  # noqa: E402


@pytest.fixture
def session():
    s = Session.simulated(params=SimParams(beam_current_a=0.25e-9))
    yield s
    s.close()


def wait(qtbot, widget_with_worker, attr="_worker", timeout=60_000):
    w = getattr(widget_with_worker, attr)
    assert w is not None
    qtbot.waitUntil(lambda: w.isFinished(), timeout=timeout)
    qtbot.wait(20)  # deliver queued signals


def test_points_text_roundtrip():
    pts = np.array([[1.5, -2], [3, 4]])
    np.testing.assert_allclose(parse_points(format_points(pts) + "\n# comment\n5 6"), [[1.5, -2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        parse_points("1, 2, 3")


def test_dataclass_form_roundtrip(qtbot):
    from easyebl.imaging.gui import RASTER_FORM_OVERRIDES

    cfg = RasterConfig(nx=99, center_um=(1.0, -2.0), dwell_s=7e-6, preline_mode="hold", line_sync_marker=None)
    f = DataclassForm(cfg, RASTER_FORM_OVERRIDES)
    qtbot.addWidget(f)
    assert f.value() == cfg
    f._widgets["nx"][0].setValue(17)
    f._widgets["dwell_s"][0].setValue(3.0)  # displayed in µs
    f._widgets["center_um"][0].setText("5, 6")
    f._widgets["line_sync_marker"][0].setText("sync")
    v = f.value()
    assert (v.nx, v.dwell_s, v.center_um, v.line_sync_marker) == (17, pytest.approx(3e-6), (5.0, 6.0), "sync")
    f.set_value(cfg)
    assert f.value() == cfg


def test_hal_gui(qtbot, session):
    from easyebl.hal.gui import HALDebugWidget, test_waveform

    w = HALDebugWidget(session)
    qtbot.addWidget(w)
    w.x_v.setValue(1.5)
    w.blank.setChecked(False)
    w._set_static()
    assert session.manager.beam_state.x_v == 1.5 and not session.manager.beam_state.blank
    w._park()
    assert session.manager.beam_state.blank
    claim, val = w._line_boxes["scan_relay"]
    claim.setChecked(True)
    val.setChecked(True)
    assert session.manager.line_state["scan_relay"] is True
    for kind in ("sine", "square", "triangle", "step"):
        c = test_waveform(kind, 2.0, 1e3, 1e-3, 100e3, True)
        assert len(c) == 100 and c.max_abs_voltage() <= 2.0 + 1e-12
    w.dur.setValue(0.002)
    w._run_test()
    wait(qtbot, w)
    assert "hal-debug" in w.handles.findItems("hal-debug", __import__("PySide6").QtCore.Qt.MatchExactly)[0].text(0)


def test_imaging_gui(qtbot, session):
    from easyebl.imaging.gui import ImagingWidget

    w = ImagingWidget(session)
    qtbot.addWidget(w)
    w.form.set_value(w.config().with_(nx=48, ny=40, width_um=20))
    w.preview()
    w.scan()
    wait(qtbot, w)
    assert w.last is not None and w.last.image().shape == (40, 48)
    assert "frame" in w.summary.text()
    w.form._widgets["center_um"][0].setText("49, 0")
    w._update_summary()
    assert "outside" in w.summary.text()


def test_imaging_gui_live(qtbot, session):
    from easyebl.imaging.gui import ImagingWidget

    w = ImagingWidget(session)
    qtbot.addWidget(w)
    w.form.set_value(w.config().with_(nx=32, ny=24, width_um=20, rate=400e3, dwell_s=5e-6))
    # scan control claims the relay lines and drives them high
    w.cb_control.setChecked(True)
    assert all(session.manager.line_state[n] for n in w.static_names)
    w.sp_avg.setValue(4)
    frames = []
    w._frame.connect(lambda k, img, t: frames.append(k))
    w.b_live.setChecked(True)
    qtbot.waitUntil(lambda: len(frames) >= 3, timeout=10000)
    assert session.manager.is_busy  # one stream, still running
    # changing a setting restarts the stream with the new size
    w.form.set_value(w.config().with_(nx=40))
    qtbot.waitUntil(lambda: w.last is not None and w.last.image().shape == (24, 40), timeout=10000)
    w.stop()
    qtbot.waitUntil(lambda: not w._busy(), timeout=10000)
    assert not session.manager.is_busy and session.manager.beam_state.blank
    assert not w.b_live.isChecked()
    w.cb_control.setChecked(False)
    assert not any(session.manager.line_state[n] for n in w.static_names)


def test_alignment_gui(qtbot, session):
    from easyebl.alignment.gui import AlignmentWidget

    w = AlignmentWidget(session)
    qtbot.addWidget(w)
    w.scan_form.set_value(w.scan_form.value().__class__(nx=96, ny=96, dwell_s=4e-6, preline_s=30e-6, blank_preline=True))
    w._apply_sim()
    w.run_alignment()
    wait(qtbot, w)
    assert w.result is not None and w.table.rowCount() == 4
    truth = w.sim_truth()
    nominal = parse_points(w.marks.toPlainText())
    err = w.result.transform.apply(nominal) - truth.apply(nominal)
    assert np.max(np.hypot(*err.T)) < 0.08
    w.table.selectRow(2)


def test_vector_gui(qtbot, session):
    from easyebl.litho.gui_vector import PRESETS, VectorLithoWidget

    w = VectorLithoWidget(session)
    qtbot.addWidget(w)
    p = w.panel
    p.params_form.set_value(p.params().__class__(beam_current_a=0.25e-9, area_step_um=0.05))
    for name in PRESETS:
        w.preset.setCurrentText(name)
        p.compile_preview()
        assert p.job is not None and p.job.stats.n_points > 0, name
    w.preset.setCurrentText("triangle")
    p.expose()
    wait(qtbot, p)
    d = session.simulator.dose.dose_uC_cm2()
    assert np.median(d[d > 0]) == pytest.approx(300, rel=0.05)
    w.mode.setCurrentIndex(1)
    pat = w.pattern()
    assert len(pat.by_kind("line")) == 2 and len(pat.by_kind("dot")) == 15
    p.expose()
    wait(qtbot, p)
    assert "done" in p.log.toPlainText()


def test_gds_gui(qtbot, session):
    from easyebl.litho.gui_gds import GDSWidget

    w = GDSWidget(session)
    qtbot.addWidget(w)
    w._demo()
    assert w.layers.count() == 3 and "9 polygons" in w.info.text() and "fits field" in w.info.text()
    w.mode.setCurrentText("datatype")
    w.load()
    assert sorted(set(w.pattern().dose_factors())) == [1.0, 1.25, 1.6]
    w.layers.item(0).setCheckState(__import__("PySide6").QtCore.Qt.Unchecked)
    w.load()
    assert len(w.pattern()) == 5
    w.panel.params_form.set_value(w.panel.params().__class__(beam_current_a=0.25e-9, area_step_um=0.1))
    w.panel.expose()
    wait(qtbot, w.panel)
    assert session.simulator.dose.total_charge() > 0


def test_dose_gui(qtbot, session):
    from easyebl.litho.gui_dose import DoseCalculatorWidget

    w = DoseCalculatorWidget(session)
    qtbot.addWidget(w)
    assert w.out.rowCount() >= 6 and "OK" in w.warn.text()
    w.form._widgets["current_pA"][0].setValue(5000)
    w.update()
    assert "minimum" in w.warn.text()


def test_launcher(qtbot, session):
    from easyebl.gui.launcher import MainWindow

    w = MainWindow(session)
    qtbot.addWidget(w)
    assert w.tabs.count() == 6
    owners = {h.owner for h in session.manager.handles()}
    assert {"hal-debug", "imaging", "alignment", "litho-vector", "litho-gds"} <= owners
