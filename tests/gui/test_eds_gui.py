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
    # unsaved test data: no question when the window closes – pytest-qt closes it before this
    # fixture's teardown, and a modal question nobody answers would hang the test run
    w._ask_unsaved = lambda items: "discard"
    qtbot.addWidget(w)
    w.show()
    yield w, mgr, dpp
    w._ask_unsaved = lambda items: "discard"  # a test may have changed it
    w.close()
    link.close()
    dpp.close()
    mgr.close()


def _wait_idle(qtbot, w, timeout=60000):
    qtbot.waitUntil(lambda: not w.ctrl.busy and w._job is None and w._analysis is None and w._maps_fut is None,
                    timeout=timeout)


def test_sites_follow_the_field_of_view_and_clear_all(qtbot, eds):
    """Sites are fractions of the field: a magnification change keeps them in place on the
    screen and in range; the map region cannot leave the image; Clear all resets everything."""
    from easyebl.eds.spectrum import ElectronImage

    w, mgr, dpp = eds
    calls = {}
    w.ctrl.acquire_map = lambda nx, ny, dwell, region, **kw: calls.update(map=region)
    w.ctrl.acquire_spectrum = lambda t, **kw: calls.update(spectrum=kw)
    # an SEM-like 4:3 image of the middle of a 100 µm field
    w._show_image(ElectronImage(np.zeros((30, 40)), (0.0, 0.0, 40.0, 30.0), 1e-6, metadata={"field_um": 100.0}))
    assert w._image_region() == pytest.approx((0.0, 0.0, 0.4, 0.3))
    # drawn beyond the image: limited to it
    w._site_requested("map", (0.1, 0.0, 0.4, 0.5))
    assert w.map_region == pytest.approx((0.05, 0.0, 0.3, 0.3))  # x -0.1..0.2, y -0.15..0.15
    w._site_requested("point", (0.45, 0.0))  # outside the image: ignored
    assert len(w.data) == 0
    w._site_requested("point", (0.1, 0.1))
    site = w.data.items[0]
    assert "(10.00, 10.00) µm" in w._rows[site.number].text(3)
    # ×10 magnification: same places in the field of view, 10× smaller in µm
    w.ctrl.set_calibration(FieldCalibration(10.0, w.ctrl.calibration.v_full_scale))
    w._tick()
    assert site.params == pytest.approx((0.1, 0.1))
    assert "(1.00, 1.00) µm" in w._rows[site.number].text(3)
    w.start_map()
    assert calls["map"] == pytest.approx((0.5, 0.0, 3.0, 3.0))
    w.acquire_site(site)
    assert calls["spectrum"]["at_um"] == pytest.approx((1.0, 1.0))
    # the old image keeps its place in the field of view
    assert w._image_region() == pytest.approx((0.0, 0.0, 0.4, 0.3))
    # Clear all: no sites, no line, map region = the whole image
    w._site_requested("line", (-0.1, 0.0, 0.1, 0.0))
    assert w.b_clear.text() == "Clear all" and w.b_clear.font().bold()
    w.b_clear.click()
    assert len(w.data) == 0 and w.data_view.topLevelItemCount() == 0 and w.line is None and w.map_region is None
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
    w.select(None)
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
    w.select(None)
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
    w.select(None)
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
    w.select(None)
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
    w.select(None)
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


def test_data_manager(qtbot, eds, tmp_path):
    """Every acquisition is an item: maps and lines coexist, the selected item is highlighted on the
    image, items are renamed / remeasured / saved / opened / deleted, unsaved data is guarded."""
    from PySide6 import QtCore

    w, mgr, dpp = eds
    w.sp_autoid.setChecked(False)
    w.sp_time.setValue(0.5)
    # a point site, measured; Acquire again on it: a second item at the same place, the first kept
    w._site_requested("point", (0.1, -0.1))
    p1 = w._current
    assert p1.number == 1 and p1.data is None and not w.isWindowModified()
    w.b_acquire.click()
    _wait_idle(qtbot, w)
    assert p1.spectrum is not None and p1.dirty and w.isWindowModified() and w._rows[1].text(4) == "unsaved"
    first = p1.spectrum
    w.b_acquire.click()
    _wait_idle(qtbot, w)
    p2 = w._current
    assert p2.number == 2 and p2.params == p1.params and p1.spectrum is first and p2.spectrum is not first
    assert p1.conditions["request"] == {"time_s": 0.5, "live_time": False}
    assert p1.conditions["field_um"] == pytest.approx(100.0) and p1.conditions["processor"]["name"]
    # the selected item: thicker outline, number in bold; the others normal
    st = w.imgview.styles
    assert st["item2"]["bold"] and st["item2"]["width"] > st["item1"]["width"] and not st["item1"]["bold"]
    assert w.imgview._items["item2"][1].textItem.font().bold()
    w.select(p1)
    assert st["item1"]["bold"] and not st["item2"]["bold"]

    # two maps: the second does not replace the first
    w._site_requested("map", (0.0, 0.0, 0.4, 0.4))
    w.m_nx.setValue(3)
    w.m_ny.setValue(2)
    w.m_dwell.setValue(5)
    w.set_elements(["Mn"])
    m1 = w.start_map()
    _wait_idle(qtbot, w, 60000)
    w.m_nx.setValue(2)
    m2 = w.start_map()
    _wait_idle(qtbot, w, 60000)
    assert m1.kind == m2.kind == "map" and m1.data is not None and m2.data is not None
    assert m1.data.shape == (2, 3) and m2.data.shape == (2, 2) and w.map is m2.data
    assert m1.conditions["request"] == {"nx": 3, "ny": 2, "dwell_s": 0.005}
    assert st[f"item{m2.number}"]["bold"] and st[f"item{m2.number}"]["kind"] == "map"
    w.select(m1)  # selecting a map shows it
    assert w.map is m1.data and st[f"item{m1.number}"]["bold"] and not st[f"item{m2.number}"]["bold"]
    _wait_idle(qtbot, w)
    assert w._tiles and w._tiles[0].data.shape == (2, 3)
    # a left click in a map tile only shows the pixel's spectrum (no new item) …
    n = len(w.data)
    tile = w._tiles[1]
    seen = []
    tile.clicked.connect(lambda r, c: seen.append((r, c)))
    tile.menuRequested.connect(lambda r, c, pos: seen.append(("menu", r, c)))

    class _Click:  # a pyqtgraph mouse click at the centre of pixel (row 1, col 2)
        def __init__(self, button):
            self._b = button

        def button(self):
            return self._b

        def scenePos(self):
            return tile.vb.mapViewToScene(QtCore.QPointF(2.5, 0.5))  # row 1 of 2 is y 0 (flipped)

        def screenPos(self):
            return QtCore.QPointF(10, 10)

        def accept(self):
            pass

    w._map_menu = lambda r, c, pos: seen.append(("window menu", r, c))  # no modal menu in the test
    tile.menuRequested.disconnect()
    tile.menuRequested.connect(lambda r, c, pos: w._map_menu(r, c, pos))
    tile._click(_Click(QtCore.Qt.LeftButton))
    tile._click(_Click(QtCore.Qt.RightButton))
    assert seen == [(1, 2), ("window menu", 1, 2)]
    assert len(w.data) == n and w._preview is not None and w._current is m1
    assert w._spectrum_of_current() is w._preview["sp"] and "preview" in st
    assert "preview" in w.specview._curves and w._spectrum_of_current().title.startswith(f"Map {m1.number} pixel")
    # … "Add to Data Manager" (right-click menu) keeps it as an item
    px = w._map_pixel(1, 2, add=True)
    assert len(w.data) == n + 1 and w._preview is None and "preview" not in st and w._current is px
    assert px.kind == "spectrum" and px.source.startswith(f"Map {m1.number} pixel") and not px.can_remeasure
    # a line scan
    w._site_requested("line", (-0.2, 0.0, 0.2, 0.0))
    w.l_n.setValue(4)
    w.l_dwell.setValue(5)
    ln = w.start_linescan()
    _wait_idle(qtbot, w)
    assert ln.kind == "line" and ln.data is not None and ln.data.shape == (1, 4) and w.linescan is ln.data
    # the same on the line profile: a click shows the point's spectrum, "Add" keeps it
    n = len(w.data)
    sp = w._line_point(2)
    assert len(w.data) == n and w._preview["sp"] is sp and w._current is ln and w._profile_cursor.isVisible()
    assert sp.real_time > 0 and "point 3/4" in sp.title
    pt = w._line_point(2, add=True)
    assert len(w.data) == n + 1 and pt.shape == "point" and pt.source == f"Line {ln.number} point 3/4"
    assert pt.params == pytest.approx((0.0 + (2.5 / 4 - 0.5) * 0.4, 0.0), abs=1e-6)
    w.select(ln)

    # rename, properties
    w._ask_text = lambda title, label, text: "Grain A"
    w.rename_item(p1)
    assert p1.title == "Grain A" and w._rows[1].text(1) == "Grain A" and p1.spectrum.title == "Grain A"
    d = w.show_properties(m1)
    assert d.value("Conditions › request › nx") == "3" and d.value("Map › pixels") == "3 × 2"
    assert d.value("Conditions › processor › min_energy_kev") is not None and d.value("Map › timestamp")
    d.close()

    # remeasure: a new item with the same geometry and conditions (not the panel's present values)
    w.sp_time.setValue(2.0)
    r = w.remeasure(p1)
    _wait_idle(qtbot, w)
    assert r.remeasure_of == 1 and r.params == p1.params and r.spectrum is not None
    assert r.conditions["request"] == p1.conditions["request"] and r.spectrum.real_time < 1.5
    assert r.title == "Grain A (re)"
    rm = w.remeasure(m1)
    _wait_idle(qtbot, w, 60000)
    assert rm.data.shape == (2, 3) and rm.params == m1.params

    # save everything, open it again
    assert w.save_unsaved(str(tmp_path))
    assert not w.data.dirty and not w.isWindowModified() and w._rows[1].text(4) == "saved"
    n = len(w.data)
    opened = w.open_data([p1.path, m1.path, ln.path])
    assert [o.kind for o in opened] == ["spectrum", "map", "line"] and len(w.data) == n + 3
    assert opened[0].title == "Grain A" and opened[0].params == pytest.approx(p1.params)
    assert opened[1].params == pytest.approx(m1.params) and np.array_equal(opened[1].data.cube, m1.data.cube)
    assert not w.data.dirty and w._current is opened[2]

    # delete: an unsaved item asks first
    w.data.rename(m2, "to delete")
    asked = []
    w._confirm = lambda title, text: asked.append(text) or False
    w.delete_items([m2])
    assert asked and m2 in w.data
    w._confirm = lambda title, text: True
    w.delete_items([m2])
    assert m2 not in w.data and f"item{m2.number}" not in st and m2.number not in w._rows

    # closing with unsaved data: cancel keeps the window, save writes the files and closes
    from easyebl.eds.datamanager import open_file

    w.data.rename(p2, "last")  # saved before: goes back to its file
    w._site_requested("point", (0.0, 0.1))
    w.b_acquire.click()  # never saved: goes into the folder chosen at closing
    _wait_idle(qtbot, w)
    new = w._current
    assert new.dirty and new.path is None
    w._ask_unsaved = lambda items: "cancel"
    w.close()
    assert w.isVisible() and w.timer.isActive()
    w._ask_unsaved = lambda items: "save"
    w._ask_folder = lambda: None  # the folder dialog cancelled: the window stays open
    w.close()
    assert w.isVisible() and w.data.dirty
    w._ask_folder = lambda: str(tmp_path / "at_close")
    (tmp_path / "at_close").mkdir()
    w.close()
    assert not w.isVisible() and not w.data.dirty
    assert open_file(p2.path).name == "last"
    assert (tmp_path / "at_close" / f"{new.number:03d}_Spectrum {new.number}.msa").exists()
