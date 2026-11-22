"""Amptek DP5 over USB: communication and settings (skips when no DP5 is connected).

Nothing here depends on the spectrum: the counts may be zero (beam off).  Every
setting changed is restored by the ``dp5`` fixture.
"""
import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware


def _f(v) -> float:
    return float(str(v).split(";")[0])


def test_identity_and_configuration(dp5):
    info = dp5.info()
    assert info, "no device information"
    assert dp5.is_open and dp5.channels in (256, 512, 1024, 2048, 4096, 8192)
    s = dp5.get_settings(["MCAC", "TPEA", "THSL", "GAIN"])
    assert int(_f(s["MCAC"])) == dp5.channels
    assert _f(s["TPEA"]) > 0 and 0 <= _f(s["THSL"]) < 100 and _f(s["GAIN"]) > 0


def test_slow_threshold_write_read_back(dp5):
    thsl0 = _f(dp5.get_settings(["THSL"])["THSL"])
    new = round(thsl0 + 0.5 if thsl0 < 20 else thsl0 - 0.5, 3)
    dp5.configure({"THSL": f"{new:.3f}"})
    assert _f(dp5.get_settings(["THSL"])["THSL"]) == pytest.approx(new, abs=0.02)
    dp5.configure({"THSL": f"{thsl0:.3f}"})
    assert _f(dp5.get_settings(["THSL"])["THSL"]) == pytest.approx(thsl0, abs=0.02)


def test_shaping_preset_switch(dp5):
    before = dp5.get_settings(["TPEA", "TFLA"])
    speeds = sorted(dp5.speeds())
    cur = dp5.current_speed()
    other = next(k for k in speeds if k != cur)
    dp5.apply_speed(other)
    assert dp5.current_speed() == other
    dp5.configure(before)  # restore the exact values
    assert dp5.get_settings(["TPEA", "TFLA"]) == before


def test_counting_and_preset_real_time(dp5):
    dp5.set_preset_real_time(0.5)
    assert dp5.preset_real_time == 0.5
    assert _f(dp5.get_settings(["PRER"])["PRER"]) == pytest.approx(0.5, abs=1e-3)
    dp5.clear()
    t0 = time.monotonic()
    dp5.enable()
    st = dp5.wait_done(timeout=5.0, expected_s=0.5)
    dt = time.monotonic() - t0
    assert st.preset_done or not st.mca_enabled
    assert 0.4 < dt < 3.0
    r = dp5.read(clear=True)
    assert r.counts is not None and len(r.counts) == dp5.channels and r.counts.dtype.kind in "iu"
    assert r.real_time == pytest.approx(0.5, abs=0.05)
    assert r.fast_count >= 0 and r.slow_count >= 0 and np.all(r.counts >= 0)
    # read-and-clear: the next status starts from zero
    assert dp5.status().real_time < 0.05
    # without a preset, counting runs until disabled
    dp5.set_preset_real_time(None)
    dp5.enable()
    time.sleep(0.3)
    assert dp5.status().mca_enabled
    dp5.disable()
    st = dp5.status()
    assert not st.mca_enabled and st.real_time == pytest.approx(0.3, abs=0.2)
    dp5.clear()


def test_acquire_with_updates_and_cancel(dp5):
    import threading

    updates = []
    r = dp5.acquire(0.6, on_update=updates.append, update_s=0.15)
    assert len(updates) >= 2 and r.real_time == pytest.approx(0.6, abs=0.06)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    r = dp5.acquire(5.0, cancel=cancel)
    assert time.monotonic() - t0 < 2.0 and r.real_time < 1.5
    assert not dp5.status().mca_enabled


def test_eds_controller_live_monitor_on_the_dp5(dp5, tmp_path):
    """The EDS idle monitor reads the real DP5 in delta mode (the DAQ is simulated: no beam is moved)."""
    from easyebl.core import FieldCalibration
    from easyebl.eds.controller import EDSController, EDSOptions
    from easyebl.eds.spectrum import EnergyCalibration
    from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend

    thsl0 = _f(dp5.get_settings(["THSL"])["THSL"])
    mgr = DAQManager(SimulatedBackend(DAQConfig(), SimParams(realtime=True))).open()
    ctrl = EDSController(mgr, dp5, FieldCalibration(100.0, 10.0), EnergyCalibration(0.0, 10.0 / dp5.channels),
                         15.0, 35.0, options=EDSOptions(external_control=False))
    events = ctrl.events.subscribe()
    try:
        ctrl.start_idle_monitor(0.3)
        deadline = time.monotonic() + 10
        got = []
        while len(got) < 3 and time.monotonic() < deadline:
            got += [e for e in events.drain() if e.kind == "idle_spectrum"]
            time.sleep(0.05)
        assert len(got) >= 3, f"idle monitor state: {ctrl.idle_state}"
        assert ctrl.idle_state == "live"
        # the minimum energy becomes the DP5 slow threshold
        thsl = ctrl.set_min_energy(0.5)
        assert _f(dp5.get_settings(["THSL"])["THSL"]) == pytest.approx(thsl, abs=0.02)
        # aborting stops counting (the live read-out may finish the step it is in)
        ctrl.abort()
        t0 = time.monotonic()
        while dp5.status().mca_enabled and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        assert not dp5.status().mca_enabled
        time.sleep(0.5)
        assert not dp5.status().mca_enabled  # and stays off
    finally:
        ctrl.stop_idle_monitor()
        dp5.configure({"THSL": f"{thsl0:.3f}"})
        events.close()
        mgr.close()
