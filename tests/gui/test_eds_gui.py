"""EDS window smoke test on the simulator + DP5 mock (offscreen Qt): image, spectra, calibration, AutoID, maps."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytestmark = pytest.mark.gui

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.eds import EDSController, EnergyCalibration  # noqa: E402
from easyebl.eds.settings import EDSSettings  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend  # noqa: E402
from easyebl.hal.mca import create_pulse_processor  # noqa: E402


@pytest.fixture
def eds(qtbot, tmp_path):
    from easyebl.eds.gui.window import EDSWindow

    cfg = DAQConfig.usb6363()
    mgr = DAQManager(SimulatedBackend(cfg, SimParams(realtime=True))).open()
    dpp = create_pulse_processor("mock").open()
    settings = EDSSettings(path=str(tmp_path / "eds.json"))
    # a deliberately wrong calibration: the dialog must fix it
    settings.energy_calibration = EnergyCalibration(0.05, 0.0099)
    ctrl = EDSController(mgr, dpp, FieldCalibration(100.0, cfg.ao_limit_v), settings.energy_calibration)
    from easyebl.microscope import MicroscopeLink, SEMControl

    settings.idle_update_s = 0.2
    link = MicroscopeLink.simulated(command_delay_s=0.0)
    link.session.simulator.sem["HV_SET"] = link.session.simulator.sem["HV"] = "15.0"
    mic = SEMControl(link)
    mic.refresh().result(10)
    w = EDSWindow(ctrl, settings, simulated=True, microscope=mic)
    qtbot.addWidget(w)
    w.show()
    yield w, mgr, dpp
    w.close()
    link.close()
    dpp.close()
    mgr.close()


def _wait_idle(qtbot, w, timeout=60000):
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None and w._analysis is None, timeout=timeout)


def test_eds_window_workflow(qtbot, eds, tmp_path):
    w, mgr, dpp = eds
    # electron image
    w.im_res.setCurrentIndex(w.im_res.findData("256x192"))
    w.im_dwell.setValue(2.0)
    w.acquire_image()
    _wait_idle(qtbot, w)
    assert w.image is not None and w.image.data.shape == (192, 256) and np.isfinite(w.image.data).all()

    # calibration from a whole-field spectrum (the DP5 mock sees a 55Fe source: Mn Kα + Kβ)
    w.sp_time.setValue(1.5)
    w.sp_autoid.setChecked(False)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()  # nothing selected: the whole image
    _wait_idle(qtbot, w)
    sp = w._current.spectrum
    assert sp is not None and sp.total > 5000
    w.open_calibration()
    d = w._calib_dialog
    d.cb_std.setCurrentText("Mn / ⁵⁵Fe source (Kα, Kβ)")
    d.auto()
    assert d.table.rowCount() >= 2 and d.b_apply.isEnabled()
    d.apply()
    cal = w.ctrl.energy_calibration
    assert abs(cal.energy(int(np.argmax(sp.counts)) + 0.5) - 5.895) < 0.05  # Mn Kα where it belongs
    assert EDSSettings.load(w.settings.path).energy_calibration == cal  # persisted
    d.close()

    # point spectrum with AutoID
    w.sp_autoid.setChecked(True)
    w.sp_time.setValue(1.0)
    w._site_requested("point", (0.0, 0.0))
    assert not w.ctrl.busy and w._current.spectrum is None  # drawing only creates the site …
    w.b_acquire.click()  # … the Acquire button measures it
    _wait_idle(qtbot, w)
    assert w._current.spectrum is not None and w._current.spectrum.total > 0

    # a small map + window maps
    w._site_requested("map", (0.0, 0.0, 50.0, 50.0))
    w.m_nx.setValue(8)
    w.m_ny.setValue(6)
    w.m_dwell.setValue(5)
    w.set_elements(["Mn", "Fe", "Cu"])
    w.start_map()
    _wait_idle(qtbot, w, 120000)
    assert w.map is not None and w.map.done.all() and w.map.shape == (6, 8)
    assert len(w._maps) == 3 and w.maps_grid.count() == 4  # SE + 3 elements
    w.m_kind.setCurrentIndex(1)
    w.compute_maps()
    _wait_idle(qtbot, w)
    w.compute_phases()
    _wait_idle(qtbot, w)
    assert any(m.label == "Phases" for m in w._maps)
    w._map_pixel(2, 3)
    assert w._current.spectrum.total > 0

    # line scan
    w._site_requested("line", (-40.0, 0.0, 40.0, 0.0))
    w.l_n.setValue(6)
    w.l_dwell.setValue(5)
    w.start_linescan()
    _wait_idle(qtbot, w)
    assert len(w.profile.getPlotItem().listDataItems()) >= 3

    # files
    w._current.spectrum.save(tmp_path / "x.msa")
    assert (tmp_path / "x.msa").exists()


def test_processor_dialog(qtbot, eds):
    from easyebl.eds.gui.processor import ProcessorDialog

    w, mgr, dpp = eds
    d = ProcessorDialog(dpp, w)
    qtbot.addWidget(d)
    assert d.table.rowCount() > 5
    d.cb_speed.setCurrentIndex(0)
    d._apply_speed()
    assert dpp.current_speed() == 1


def test_live_panel_composition_and_beam_sync(qtbot, eds):
    w, mgr, dpp = eds
    # the live MCA panel updates between acquisitions (delta mode)
    qtbot.waitUntil(lambda: w.l_icr.text() != "ICR –" and w._live_curve.yData is not None
                    and len(w._live_curve.yData) > 0, timeout=5000)
    assert w.ctrl.idle_state == "live" and "Δ" in w.l_live_t.text()
    w.cb_live.setChecked(False)
    assert w.ctrl.idle_state == "off" and not w.settings.idle_monitor
    w.cb_live.setChecked(True)
    qtbot.waitUntil(lambda: w.ctrl.idle_state == "live", timeout=3000)

    # the nominal beam energy (fallback), its SEM sync, take-off, resolution and window are advanced
    assert not w.adv.isVisibleTo(w) and not w.c_kv.isVisibleTo(w)
    w.b_adv.setChecked(True)
    assert w.adv.isVisibleTo(w)
    w.b_kv_sync.click()
    assert w.c_kv.value() == 15.0 and w.ctrl.beam_kv == 15.0
    w.c_kv.setValue(20.0)

    # AutoID only adds to the user's selection, and the selection survives new spectra
    w.set_elements(["Zr", "Mo"])
    w.sp_time.setValue(0.5)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None and w._analysis is None, timeout=30000)
    assert {"Zr", "Mo"} <= set(w.elements)
    w.analyse(autoid=True)
    qtbot.waitUntil(lambda: w._analysis is None, timeout=30000)
    assert {"Zr", "Mo"} <= set(w.elements)
    w.b_clear_el.click()
    assert w.elements == []


def test_markers_keep_their_screen_size_and_e0_shown(qtbot, eds):
    w, mgr, dpp = eds
    w.set_elements(["Fe", "Cu"])
    sv = w.specview
    (_, _), (y0, y1) = sv.viewRange()
    frac = [(m.yData[1] - y0) / (y1 - y0) for m in sv._markers]
    sv.setYRange(y0, y0 + (y1 - y0) * 0.1, padding=0)  # zoom the counts axis in 10x
    (_, _), (z0, z1) = sv.viewRange()
    frac2 = [(m.yData[1] - z0) / (z1 - z0) for m in sv._markers]
    assert frac and np.allclose(frac, frac2, atol=1e-6)  # same height on screen
    # E0 of a spectrum without a measurable continuum end falls back to the nominal value
    w.sp_time.setValue(0.5)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None, timeout=30000)
    sp = w._current.spectrum
    assert sp.metadata["beam_kv_source"] == "nominal" and sp.beam_kv == w.settings.beam_kv
    assert "nominal" in w.l_e0.text()
    # the spectrum view shows every MCA channel
    (x0, x1), _ = sv.viewRange()
    assert x1 >= float(sp.with_calibration(w.ctrl.energy_calibration).energies[-1])


def test_close_stops_a_running_acquisition(qtbot, eds):
    w, mgr, dpp = eds
    w.sp_time.setValue(60.0)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()
    qtbot.waitUntil(lambda: w.ctrl.busy and dpp.status().mca_enabled, timeout=10000)
    job = w.ctrl.job
    w.close()
    assert not dpp.status().mca_enabled  # counting stopped at once
    job.future.exception(10)  # the job thread ends promptly (cancelled)
    assert job.cancelled


def test_min_energy_control(qtbot, eds):
    w, mgr, dpp = eds
    assert w.sp_emin.value() == pytest.approx(0.22)  # default: just below C Kα
    thsl0 = float(dpp.get_settings(["THSL"])["THSL"])
    assert thsl0 == pytest.approx(w.ctrl.thsl_for(0.22), abs=0.01)  # applied when the window opened
    w.sp_emin.setValue(0.5)
    qtbot.waitUntil(lambda: abs(float(dpp.get_settings(["THSL"])["THSL"]) - w.ctrl.thsl_for(0.5)) < 0.01,
                    timeout=3000)
    assert w.settings.min_energy_kev == 0.5
