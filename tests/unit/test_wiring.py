"""DAQ wiring (Hardware → Configure… → Wiring): defaults, system.json round trip, conflicts, use by the HAL."""
import json

import numpy as np
import pytest

from easyebl.app.backend import SystemBackend
from easyebl.app.config import SystemConfig
from easyebl.hal import DAQConfig, DAQManager
from easyebl.hal.wiring import USB6363, Wiring, ai_pins, canonical_line, pin_key

from . import fake_nidaqmx


# ----------------------------------------------------------------- catalog / defaults
def test_usb6363_catalog_and_aliases():
    assert USB6363.ao == ("ao0", "ao1", "ao2", "ao3")
    assert len(USB6363.ai) == 32 and len(USB6363.timed_lines) == 32 and len(USB6363.pfi) == 16
    diff = USB6363.ai_for("DIFF")
    assert diff == tuple(f"ai{i}" for i in (*range(8), *range(16, 24)))
    assert USB6363.ai_for("RSE") == USB6363.ai
    assert canonical_line("port1/line6") == "PFI6" and canonical_line("port2/line0") == "PFI8"
    assert canonical_line("port0/line23") == "port0/line23" and canonical_line("pfi3") == "PFI3"
    assert pin_key("PFI7") == pin_key("port1/line7")
    assert ai_pins("ai3", "DIFF") == ("ai3", "ai11") and ai_pins("ai3", "RSE") == ("ai3",)


def test_defaults_are_todays_usb6363_wiring():
    w = Wiring()
    assert w == Wiring.from_daq_config(DAQConfig.usb6363())
    assert w.check() == []
    cfg = DAQConfig.usb6363()
    w.apply_to(cfg)
    ref = DAQConfig.usb6363()
    for k in ("ao_x", "ao_y", "aux_outputs", "blank_line", "blank_active_high", "timed_lines",
              "external_control_lines", "static_lines", "ai_channels", "sample_clock_out", "start_trigger_out"):
        assert getattr(cfg, k) == getattr(ref, k), k


def test_old_system_json_without_wiring(tmp_path):
    p = tmp_path / "system.json"
    p.write_text(json.dumps({"daq": {"backend": "sim", "preset": "usb6363"}, "eds": {"mca": "mock"}}))
    c = SystemConfig.load(str(p))
    assert c.wiring is None  # follows the preset, exactly as before the Wiring tab
    assert c.effective_wiring() == Wiring()
    d = c.daq_config()
    assert (d.ao_x, d.blank_line, d.ai_channels, d.external_control_lines) == \
        ("ao0", "port0/line23", {"video": "ai0"}, {"PFI6": True, "PFI7": True})
    assert c.eds_video_channel == "video"
    # the generic preset keeps its own wiring too (port1 lines shown as PFI, same pins)
    c.daq.preset = "default"
    g = c.effective_wiring()
    assert g.blank_line == "port0/line0" and g.external_control_lines == {"PFI0": True}
    assert c.daq_config().external_control_lines == {"port1/line0": True}  # untouched in the HAL config


def test_wiring_roundtrip(tmp_path):
    c = SystemConfig.hardware()
    c.wiring = Wiring(ao_x="ao1", ao_y="ao0", aux0=None, aux1="ao2", blank_line="port0/line5", blank_active_high=True,
                      timed_lines={"line_sync": "port0/line1"}, external_control_lines={"PFI6": True, "PFI7": False},
                      static_lines={"relay": "PFI9"}, static_lines_active_high={"relay": False},
                      ai_channels={"se": "ai2", "bse": "ai3"}, eds_video="bse",
                      sample_clock_out="PFI12", start_trigger_out="PFI13")
    p = c.save(str(tmp_path / "system.json"))
    raw = json.loads(open(p, encoding="utf-8").read())
    assert raw["wiring"]["ai_channels"] == {"se": "ai2", "bse": "ai3"}
    back = SystemConfig.load(p)
    assert back == c and list(back.wiring.ai_channels) == ["se", "bse"]  # order kept: first = SEM image
    d = back.daq_config()
    assert (d.ao_x, d.ao_y, d.aux_outputs) == ("ao1", "ao0", {"aux0": None, "aux1": "ao2"})
    assert d.blank_line == "port0/line5" and d.blank_active_high and d.timed_lines == {"line_sync": "port0/line1"}
    assert d.external_control_map() == {"PFI6": True, "PFI7": False}
    assert d.static_lines == {"relay": "PFI9"} and d.static_lines_active_high == {"relay": False}
    assert d.ai_channels == {"se": "ai2", "bse": "ai3"} and back.eds_video_channel == "bse"
    assert (d.sample_clock_out, d.start_trigger_out) == ("PFI12", "PFI13")
    assert d.ai_range_v == 10.0  # no range stored: the preset's
    back.daq.ai_range_v = 1.0
    assert SystemConfig.load(back.save(p)).daq_config().ai_range_v == 1.0
    # an explicit null is the same as no section
    raw["wiring"] = None
    (tmp_path / "null.json").write_text(json.dumps(raw))
    assert SystemConfig.load(str(tmp_path / "null.json")).wiring is None


# ----------------------------------------------------------------- conflicts
def _keys(w, term="RSE"):
    return [set(i.signals) for i in w.check(term)]


def test_two_signals_on_one_output_are_flagged():
    assert {"ao_x", "ao_y"} in _keys(Wiring(ao_y="ao0"))
    assert {"ao_x", "aux0"} in _keys(Wiring(aux0="ao0"))
    # PFI6 and port1/line6 are the same pin
    w = Wiring(static_lines={"relay": "port1/line6"})
    assert {"ext:PFI6", "static:relay"} in _keys(w)
    assert {"blank", "timed:m"} in _keys(Wiring(timed_lines={"m": "port0/line23"}))
    assert {"clock_out", "trigger_out"} in _keys(Wiring(sample_clock_out="PFI3", start_trigger_out="PFI3"))
    assert {"ext:PFI7", "clock_out"} in _keys(Wiring(sample_clock_out="PFI7"))
    assert {"ai:a", "ai:b"} in _keys(Wiring(ai_channels={"a": "ai1", "b": "ai1"}, eds_video="a"))


def test_differential_pairs_and_wrong_kinds():
    w = Wiring(ai_channels={"video": "ai0", "bse": "ai8"})
    assert w.check("RSE") == []
    keys = _keys(w, "DIFF")
    assert {"ai:bse"} in keys  # AI8 is AI0− in DIFF …
    assert {"ai:video", "ai:bse"} in keys  # … and the video input already uses it
    probs = " ".join(Wiring(ao_x="ao7", static_lines={"s": "port0/line2"}, ai_channels={"v": "ai40"},
                            sample_clock_out="port0/line1", eds_video="v").problems())
    for text in ("ao7", "ai40", "not a PFI terminal", "drives as a whole"):
        assert text in probs, text
    assert "hardware-timed" in " ".join(Wiring(blank_line="PFI3").problems())
    assert "EDS SE input" in " ".join(Wiring(eds_video="gone").problems())
    assert "At least one video input" in " ".join(Wiring(ai_channels={}, eds_video=None).problems())
    with pytest.raises(ValueError, match="share"):
        Wiring(ao_y="ao0").validate()


def test_bad_wiring_is_refused_when_opening():
    c = SystemConfig()
    c.wiring = Wiring(ao_y="ao0")
    with pytest.raises(ValueError, match="share the analog output"):
        c.daq_config()
    b = SystemBackend(c).start()
    try:
        assert b.manager is None and "share" in b.errors["daq"]
    finally:
        b.close()


# ----------------------------------------------------------------- the HAL and its consumers use it
def test_backend_uses_the_configured_channels(tmp_path):
    c = SystemConfig()
    c.wiring = Wiring(ao_x="ao2", ao_y="ao3", aux0="ao0", aux1=None, ai_channels={"se": "ai5", "bse": "ai6"},
                      eds_video="bse", external_control_lines={"PFI2": True})
    b = SystemBackend(c, eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    try:
        cfg = b.manager.config
        assert (cfg.ao_x, cfg.ao_y, cfg.aux_physical("aux0")) == ("ao2", "ao3", "ao0")
        assert cfg.ai_channels == {"se": "ai5", "bse": "ai6"} and cfg.external_control_map() == {"PFI2": True}
        assert b.engine.config.ai_channels == ("se",)  # the SEM image: the first input
        assert b.eds.options.video_channel == "bse"
    finally:
        b.close()
    c.wiring.eds_video = None
    b = SystemBackend(c, eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    try:
        assert b.eds.options.video_channel is None
    finally:
        b.close()


def test_nidaq_uses_wired_channels_and_exports(monkeypatch):
    from easyebl.core import OutputChunk
    from easyebl.hal.nidaq import NIDAQBackend

    fake = fake_nidaqmx.install(monkeypatch)
    cfg = DAQConfig(buffer_samples=5000)
    Wiring(ao_x="ao1", ao_y="ao0", aux0=None, aux1=None, blank_line="port0/line4", blank_active_high=True,
           external_control_lines={"PFI6": True}, ai_channels={"se": "ai3"}, eds_video="se",
           sample_clock_out="PFI12", start_trigger_out="PFI13").apply_to(cfg)
    mgr = DAQManager(NIDAQBackend(cfg)).open()
    try:
        n = 3000
        x = np.linspace(-1, 1, n)
        mgr.get_handle("t").run_stream(100e3, [OutputChunk(x, -x, np.zeros(n, bool), {})], ai_channels=["se"])
        t = {t.name: t for t in fake.tasks if t.name.startswith("easyebl")}
        assert [p for _, p in t["easyebl_ao"].channels] == ["Dev1/ao1", "Dev1/ao0"]
        assert [p for _, p in t["easyebl_ai"].channels] == ["Dev1/ai3"]
        assert t["easyebl_ao"].export_signals.samp_clk_output_term == "/Dev1/PFI12"
        assert t["easyebl_ao"].export_signals.start_trig_output_term == "/Dev1/PFI13"
    finally:
        mgr.close()
