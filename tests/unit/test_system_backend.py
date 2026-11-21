"""easyEBL system configuration and backend (simulators, no Qt)."""
import json

from easyebl.app.backend import SystemBackend
from easyebl.app.config import SystemConfig


def test_config_roundtrip_and_presets(tmp_path):
    c = SystemConfig.hardware(device="Dev2", port="COM7")
    c.daq.rate = 1e6
    c.microscope.baudrate = 19200
    p = c.save(str(tmp_path / "system.json"))
    back = SystemConfig.load(p)
    assert back == c and back.daq.rate == 1e6 and back.microscope.port == "COM7"
    assert not back.simulated and back.as_simulated().simulated
    s = back.as_simulated()
    assert (s.daq.backend, s.eds.mca, s.microscope.port) == ("sim", "mock", "sim")
    assert s.daq.device == "Dev2" and s.microscope.baudrate == 19200  # addresses are kept
    # unknown keys and a broken file are tolerated
    (tmp_path / "odd.json").write_text(json.dumps({"daq": {"backend": "sim", "bogus": 1}, "x": {}}))
    assert SystemConfig.load(str(tmp_path / "odd.json")).daq.backend == "sim"
    (tmp_path / "bad.json").write_text("{not json")
    assert SystemConfig.load(str(tmp_path / "bad.json")) == SystemConfig()
    # the default is safe: everything simulated
    assert SystemConfig().simulated


def test_backend_on_simulators(tmp_path):
    b = SystemBackend(SystemConfig(), eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    try:
        st = b.status()
        assert {k: s.state for k, s in st.items()} == {"daq": "sim", "eds": "sim", "microscope": "sim"}
        assert b.engine.handle.manager is b.manager and b.eds.handle.manager is b.manager  # one DAQ for every subsystem
        assert b.microscope.follower is not None and b.busy == []
        b.microscope.sem.set_magnification(2000).result(10)  # the field follows the magnification
        assert b.engine.calibration.field_size_um == 50.0
    finally:
        b.close()
    assert b.manager is None and b.microscope.link.state == "closed"


def test_a_failing_device_does_not_stop_the_others(tmp_path):
    c = SystemConfig()
    c.microscope.port = "COM_DOES_NOT_EXIST"
    c.eds.mca = "none"
    b = SystemBackend(c, eds_settings_path=str(tmp_path / "eds.json")).start()
    try:
        st = b.status()
        assert st["daq"].state == "sim" and st["eds"].state == "off"
        assert st["microscope"].state == "error" and "COM_DOES_NOT_EXIST" in st["microscope"].text
        assert not b.microscope.connected and b.engine is not None
    finally:
        b.close()
