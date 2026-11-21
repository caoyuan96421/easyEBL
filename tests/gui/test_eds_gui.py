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
    w = EDSWindow(ctrl, settings, simulated=True)
    qtbot.addWidget(w)
    w.show()
    yield w, mgr, dpp
    w.close()
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
    w._new_site("field", w._image_region())
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
