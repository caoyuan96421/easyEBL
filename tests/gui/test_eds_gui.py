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
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None and w._analysis is None and w._maps_fut is None,
                    timeout=timeout)


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
    k_ka = d.marker  # the strongest peak is marked: Mn Kα
    assert k_ka is not None and abs(k_ka - np.argmax(sp.counts)) < 3
    d.add_element("Mn")  # the calibration is far off (the mock is ≈39 eV/ch): the nearest Mn line is wrong …
    assert d.pairs[0].line.label != "Mn Kα" and d.table.rowCount() == 1
    combo = d.table.cellWidget(0, 0)  # … and is corrected in the table
    combo.setCurrentIndex([combo.itemData(i).label for i in range(combo.count())].index("Mn Kα"))
    assert d.pairs[0].line.label == "Mn Kα" and abs(d.calibration.energy(k_ka) - 5.8965) < 1e-3
    # Mn Kβ: click near it, then Mn again
    true_kb = k_ka * 6.49 / 5.895
    d.mark_peak(true_kb + 2)
    d.add_element("Mn")
    assert [p.line.label for p in d.pairs] == ["Mn Kα", "Mn Kβ"]
    # a second spectrum (another sample): the pairs so far are kept, the new one is added
    from easyebl.eds.spectrum import Spectrum as _Sp
    other = _Sp(np.roll(sp.counts, 40), sp.calibration, title="second sample", timestamp="2026-11-22T10:00:00")
    d.spectrum = other
    d._show_spectrum(mark_strongest=True)
    d.add_element("Mn")
    assert len(d.pairs) == 3 and {p.source for p in d.pairs} >= {"second sample 11-22 10:00"}
    saved = EDSSettings.load(w.settings.path).calibration_points
    assert len(saved) == 3 and {q["label"] for q in saved} == {"Mn Kα", "Mn Kβ"}
    d.remove_selected()  # nothing selected: no change
    d.table.selectRow([p.source for p in d.pairs].index("second sample 11-22 10:00"))
    d.remove_selected()
    assert len(d.pairs) == 2 and len(EDSSettings.load(w.settings.path).calibration_points) == 2
    d.spectrum = sp
    d.refit()
    d.apply()
    cal = w.ctrl.energy_calibration
    assert abs(cal.energy(k_ka) - 5.895) < 0.01  # Mn Kα where it belongs
    # typing a value and Clear
    d.sp_slope.setValue(12.5)
    assert d.calibration.gain_kev == pytest.approx(0.0125)
    d.close()
    # reopened: the pairs are back
    w.open_calibration()
    d = w._calib_dialog
    assert [p.line.label for p in d.pairs] == ["Mn Kα", "Mn Kβ"] and d.table.rowCount() == 2
    d.clear()
    assert d.calibration == EnergyCalibration() and d.pairs == [] and d.table.rowCount() == 0
    assert EDSSettings.load(w.settings.path).calibration_points == []
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
    w._site_requested("map", (0.0, 0.0, 0.5, 0.5))  # field fractions
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
    w._site_requested("line", (-0.4, 0.0, 0.4, 0.0))
    w.l_n.setValue(6)
    w.l_dwell.setValue(5)
    w.start_linescan()
    _wait_idle(qtbot, w)
    assert len(w.profile.getPlotItem().listDataItems()) >= 3

    # files
    w._current.spectrum.save(tmp_path / "x.msa")
    assert (tmp_path / "x.msa").exists()



def test_sites_follow_the_field_of_view_and_clear_all(qtbot, eds):
    """Sites are fractions of the field: a magnification change keeps them in place on the
    screen and in range; the map region cannot leave the image; Clear all resets everything."""
    from easyebl.eds.spectrum import ElectronImage

    w, mgr, dpp = eds
    calls = {}
    w.ctrl.acquire_map = lambda nx, ny, dwell, region: calls.update(map=region)
    w.ctrl.acquire_spectrum = lambda t, **kw: calls.update(spectrum=kw)
    # an SEM-like 4:3 image of the middle of a 100 µm field
    w._show_image(ElectronImage(np.zeros((30, 40)), (0.0, 0.0, 40.0, 30.0), 1e-6, metadata={"field_um": 100.0}))
    assert w._image_region() == pytest.approx((0.0, 0.0, 0.4, 0.3))
    # drawn beyond the image: limited to it
    w._site_requested("map", (0.1, 0.0, 0.4, 0.5))
    assert w.map_region == pytest.approx((0.05, 0.0, 0.3, 0.3))  # x -0.1..0.2, y -0.15..0.15
    w._site_requested("point", (0.45, 0.0))  # outside the image: ignored
    assert w._sites == []
    w._site_requested("point", (0.1, 0.1))
    assert "(10.00, 10.00) µm" in w.site_list.item(0).text()
    # ×10 magnification: same places in the field of view, 10× smaller in µm
    w.ctrl.set_calibration(FieldCalibration(10.0, w.ctrl.calibration.v_full_scale))
    w._tick()
    assert w._sites[0].params == pytest.approx((0.1, 0.1))
    assert "(1.00, 1.00) µm" in w.site_list.item(0).text()
    w.start_map()
    assert calls["map"] == pytest.approx((0.5, 0.0, 3.0, 3.0))
    w.acquire_site(w._sites[0])
    assert calls["spectrum"]["at_um"] == pytest.approx((1.0, 1.0))
    # the old image keeps its place in the field of view
    assert w._image_region() == pytest.approx((0.0, 0.0, 0.4, 0.3))
    # Clear all: no sites, no line, map region = the whole image
    w._site_requested("line", (-0.1, 0.0, 0.1, 0.0))
    assert w.b_clear.text() == "Clear all" and w.b_clear.font().bold()
    w.b_clear.click()
    assert w._sites == [] and w.site_list.count() == 0 and w.line is None and w.map_region is None
    assert w.m_region.text() == "whole image"
    calls.clear()
    w.start_map()
    assert calls["map"] == pytest.approx((0.0, 0.0, 4.0, 3.0))  # the image at ×10


def test_processor_dialog(qtbot, eds):
    from easyebl.eds.gui.processor import ProcessorDialog

    w, mgr, dpp = eds
    d = ProcessorDialog(dpp, w)
    qtbot.addWidget(d)
    assert d.table.rowCount() > 5
    assert not hasattr(d, "cb_speed") and not hasattr(d, "b_thr")  # speed: Acquire tab; no auto threshold
    d.close()

    # the speed preset is a combo box in the Acquire tab, written as soon as it is chosen
    i1 = w.sp_speed.findData(1)
    w.sp_speed.setCurrentIndex(i1)
    w.sp_speed.activated.emit(i1)
    assert dpp.current_speed() == 1 and w.sp_speed.currentData() == 1
    i5 = w.sp_speed.findData(5)
    w.sp_speed.setCurrentIndex(i5)
    w.sp_speed.activated.emit(i5)
    assert dpp.current_speed() == 5 and w.sp_speed.currentData() == 5
    # chosen during an acquisition: written when it is done
    w.sp_time.setValue(1.0)
    w.sp_autoid.setChecked(False)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()
    qtbot.waitUntil(lambda: w.ctrl.busy, timeout=5000)
    w.sp_speed.setCurrentIndex(i1)
    w.sp_speed.activated.emit(i1)
    assert w._speed_pending == 1 and dpp.current_speed() == 5
    _wait_idle(qtbot, w)
    assert dpp.current_speed() == 1 and w._speed_pending is None


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

    # during an acquisition the live panel shows the delta between read-outs, not the running total
    import re
    w.sp_autoid.setChecked(False)
    w.sp_time.setValue(2.0)
    w.site_list.setCurrentRow(-1)
    w.b_acquire.click()
    qtbot.waitUntil(lambda: w._job is not None and w._live_prev is not None
                    and w._live_prev.real_time > 1.0, timeout=10000)
    dt = float(re.search(r"Δ ([0-9.]+) s", w.l_live_t.text()).group(1))
    assert 0.05 < dt < 0.8 and w.l_icr.text() != "ICR –"  # one read-out interval (0.25 s), not > 1 s
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None and w._analysis is None, timeout=30000)
    assert w._live_prev is None
    w.sp_autoid.setChecked(True)
    # maps: the pixels finished since the last update, summed
    from easyebl.eds.spectrum import SpectrumMap
    sm = SpectrumMap.empty(4, 3, 64, w.ctrl.energy_calibration, (0, 0, 10, 10), 0.05)
    sm.cube[0, :2, 10] = 100
    sm.real_time[0, :2], sm.live_time[0, :2], sm.fast_count[0, :2] = 0.05, 0.04, 150
    sm.done[0, :2] = True
    w._live_px_t = 0.0
    w._live_from_map(sm)
    assert w.l_live_t.text().startswith("Δ 0.10 s") and "200 cts" in w.l_live_t.text()
    assert w.l_icr.text() == "ICR 3.00 kcps" and w.l_ocr.text() == "OCR 2.00 kcps"
    sm.cube[0, 2, 11] = 7
    sm.real_time[0, 2], sm.done[0, 2] = 0.05, True
    w._live_px_t = 0.0
    w._live_from_map(sm)
    assert "7 cts" in w.l_live_t.text()  # only the new pixel
    w._live_px = None

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


def test_live_maps_during_the_scan_and_dead_time_zones(qtbot, eds):
    w, mgr, dpp = eds
    # dead-time bar: grey below 5 %, green 5–50 %, red above
    for pct, col in ((2, "200,200,200"), (20, "80,200,90"), (70, "240,70,60")):
        w._style_dead(pct)
        assert f"chunk{{background:rgba({col}" in w.l_dead.styleSheet()

    # element maps appear while the map is still scanning; unscanned pixels are blank (NaN)
    w._site_requested("map", (0.0, 0.0, 0.6, 0.6))
    w.m_nx.setValue(16)
    w.m_ny.setValue(12)
    w.m_dwell.setValue(25)  # ≈ 5 s
    w.m_kind.setCurrentIndex(0)
    w.m_bin.setValue(1)
    w.set_elements(["Mn", "Fe"])
    w.start_map()
    qtbot.waitUntil(lambda: bool(w._job is not None and getattr(w, "_tiles", None)
                                 and any(t.label.startswith("Fe") for t in w._tiles)
                                 and np.isnan(w._tiles[-1].data).any()), timeout=20000)
    assert w._map_running is not None
    # changing the composition mid-scan: the live maps follow
    w.set_elements(["Mn", "Fe", "Cu"])
    qtbot.waitUntil(lambda: bool(any(t.label.startswith("Cu") for t in w._tiles)), timeout=10000)
    assert w._job is not None, "the scan should still be running"
    _wait_idle(qtbot, w, 120000)
    labels = [t.label for t in w._tiles]
    assert labels[0] == "SE" and {"Mn", "Fe", "Cu"} <= {lb.split()[0] for lb in labels[1:]}
    assert w.map is not None and w.map.done.all()
    assert all(np.isfinite(t.data).all() for t in w._tiles[1:])  # the final maps are complete
    # after the scan: changing the map kind recomputes by itself
    w.m_kind.setCurrentIndex(1)
    qtbot.waitUntil(lambda: bool(w._maps_fut is None and w._maps_done_gen == w._maps_gen), timeout=20000)
    assert any(t.unit == "counts" for t in w._tiles)
