"""Pulse-processor HAL: the DP5 adapter (on the driver's mock device)."""
import threading

import numpy as np
import pytest

from easyebl.hal.mca import (
    DP5PulseProcessor,
    MCABusyError,
    MCAReading,
    create_pulse_processor,
)


# ------------------------------------------------------------------ DP5 (mock transport)
@pytest.fixture
def dp5():
    p = DP5PulseProcessor(mock=True, mock_cps=20e3).open()
    yield p
    p.close()


def test_dp5_open_info_and_config(dp5):
    assert dp5.is_open and dp5.channels > 0
    info = dp5.info()
    assert info["mock"] is True and info["serial"]
    dp5.configure({"MCAC": 2048})
    assert dp5.channels == 2048
    assert dp5.get_settings(["MCAC"])["MCAC"].startswith("2048")


def test_dp5_preset_real_time_stops_the_mca(dp5):
    dp5.set_preset_real_time(0.05)
    assert dp5.preset_real_time == 0.05
    dp5.clear()
    dp5.enable()
    st = dp5.wait_done(timeout=3.0, expected_s=0.05)
    assert st.preset_done  # like the real DP5, PRER leaves "MCA enabled" set …
    assert not dp5.status().mca_enabled  # … so wait_done disabled it
    r = dp5.read(clear=True)
    assert r.real_time == pytest.approx(0.05, abs=2e-3)
    assert r.counts.size == dp5.channels and r.total > 0
    after = dp5.status()
    assert after.real_time == 0 and not after.preset_done


def test_dp5_acquire_with_updates_and_cancel(dp5):
    seen = []
    r = dp5.acquire(0.2, on_update=seen.append, update_s=0.05)
    assert r.real_time == pytest.approx(0.2, abs=5e-3) and seen and seen[-1] is r
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    r = dp5.acquire(5.0, cancel=cancel)
    assert r.real_time < 2.0


def test_session_is_exclusive(dp5):
    with dp5.session("eds"):
        with dp5.session("eds"):  # re-entrant
            assert dp5.owner == "eds"
        with pytest.raises(MCABusyError):
            with dp5.session("other"):
                pass
    assert dp5.owner is None


def test_reading_rates():
    r = MCAReading(np.zeros(4, np.int64), real_time=2.0, acc_time=1.9, fast_count=2000, slow_count=1500,
                   mca_enabled=False, preset_done=True)
    assert r.icr == 1000 and r.ocr == 750 and r.dead_fraction == pytest.approx(0.25)
    assert r.live_time == pytest.approx(1.5)


# ------------------------------------------------------------------ factory
def test_create_pulse_processor():
    p = create_pulse_processor("mock").open()
    try:
        assert isinstance(p, DP5PulseProcessor) and p.info()["mock"] is True
    finally:
        p.close()
    with pytest.raises(ValueError):
        create_pulse_processor("sim")
