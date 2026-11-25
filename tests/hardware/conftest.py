"""Tests against the real instrument: the Amptek DP5 (USB) and the Hitachi S-3000 (RS-232).

They run with every ``pytest`` and **skip by themselves** when a device is not
connected, is switched off, or is in use (e.g. by a running easyEBL – close it
first: one program at a time can hold the DP5 and the serial port).

What they never do:

* set the accelerating voltage, switch HV on/off, or switch the beam on;
* run automatic routines (AFC, ASF, ABC, AGA – their results cannot be undone);
* touch the DP5 detector bias (HVSE) or its cooler (TECS);
* move the stage in Z, R or T (collision risk) – only small X / Y moves.

Every other setting they change is restored, also when a test fails.

Environment:

* ``EASYEBL_NO_HARDWARE_TESTS=1`` – skip them all;
* ``EASYEBL_HW_MICROSCOPE_PORT=COM4`` – the microscope port (default: the port saved
  in the easyEBL configuration, else COM4); only that port is tried;
* ``EASYEBL_HW_NO_STAGE=1`` – no stage moves (positions are still read);
* ``EASYEBL_HW_DRY_RUN=1`` – check the tests themselves without the instrument: the DP5
  driver's mock device and the S-3000 simulator stand in for the hardware.
"""
from __future__ import annotations

import json
import os
import time

import pytest

def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0")


_OFF = _flag("EASYEBL_NO_HARDWARE_TESTS")
DRY_RUN = _flag("EASYEBL_HW_DRY_RUN")
NO_STAGE = _flag("EASYEBL_HW_NO_STAGE")


def _saved_microscope_port() -> str:
    """Read before the test fixtures redirect the settings files to temporary ones."""
    env = os.environ.get("EASYEBL_HW_MICROSCOPE_PORT")
    if env:
        return env
    from easyebl.app.config import default_config_path
    from easyebl.microscope.settings import default_settings_path

    for path, get in ((default_config_path(), lambda d: (d.get("microscope") or {}).get("port")),
                      (default_settings_path(), lambda d: d.get("port"))):
        try:
            with open(path, encoding="utf-8") as fh:
                port = get(json.load(fh))
        except (OSError, ValueError, AttributeError):
            continue
        if port and str(port).lower() not in ("sim", "simulator", "none", "off"):
            return str(port)
    return "COM4"


MICROSCOPE_PORT = _saved_microscope_port()


# ====================================================================== DP5
@pytest.fixture(scope="module")
def dp5():
    if _OFF:
        pytest.skip("hardware tests switched off (EASYEBL_NO_HARDWARE_TESTS)")
    from easyebl.hal.mca import create_pulse_processor

    try:
        mca = create_pulse_processor("mock" if DRY_RUN else "dp5").open()
    except Exception as e:  # noqa: BLE001 - not connected, no driver, or in use
        pytest.skip(f"DP5 not available: {type(e).__name__}: {e}")
    from easyebl.hal.dp5.presets import SHAPING_KEYS

    keep = list(dict.fromkeys(list(SHAPING_KEYS) + ["THSL", "GAIN"]))  # all a test may change
    names = keep + ["PRER", "MCAC"]
    try:
        original = mca.get_settings(names)
    except Exception as e:  # noqa: BLE001
        mca.close()
        pytest.skip(f"DP5 does not answer: {type(e).__name__}: {e}")
    preset = mca.preset_real_time
    yield mca
    # restore what the tests may have changed
    try:
        mca.disable()
        mca.configure({k: original[k] for k in keep if k in original and k != "GAIN"})
        mca.configure({"GAIN": original["GAIN"]})  # after the shaping: the DP5 re-derives its gain on TPEA
        mca._preset = "unknown"  # force the write
        mca.set_preset_real_time(preset)
        now = mca.get_settings([k for k in keep if k in original])
        assert now == {k: original[k] for k in now}, f"DP5 not restored: {now} != {original}"
    finally:
        mca.close()


# ====================================================================== microscope
#: settings the tests may change -> how to restore them
_RESTORE = ("mag", "wd_readback_mm", "focus_coarse", "focus_fine", "stig_x", "stig_y", "beam_coarse", "beam_fine",
            "contrast", "brightness", "running", "detector")


def _restore(sem, original: dict) -> list[str]:
    sem.refresh().result(30)  # read everything again
    s = sem.state
    futs = []
    for key in _RESTORE:
        v = original.get(key)
        if v is None or s.get(key) == v:
            continue
        if key == "mag":
            futs.append(sem.set_magnification(v))
        elif key == "wd_readback_mm":  # sets the focus: the focus values are restored after it
            futs.append(sem.set_working_distance(v))
        elif key in ("focus_coarse", "focus_fine"):
            futs.append(sem.set_focus("C" if key == "focus_coarse" else "F", v))
        elif key in ("stig_x", "stig_y"):
            futs.append(sem.set_stigmator(key[-1].upper(), v))
        elif key in ("beam_coarse", "beam_fine"):
            futs.append(sem.set_beam("C" if key == "beam_coarse" else "F", v))
        elif key == "contrast":
            futs.append(sem.set_contrast(v))
        elif key == "brightness":
            futs.append(sem.set_brightness(v))
        elif key == "running":
            futs.append(sem.set_running(v))
        elif key == "detector":
            futs.append(sem.set_detector(v))
    for f in futs:
        f.result(30)
    sem.refresh().result(30)
    return [k for k in _RESTORE if original.get(k) is not None and sem.state.get(k) != original[k]]


@pytest.fixture(scope="module")
def microscope():
    """(link, sem, stage) on the real port; every setting and the X/Y position restored at the end."""
    if _OFF:
        pytest.skip("hardware tests switched off (EASYEBL_NO_HARDWARE_TESTS)")
    from easyebl.microscope import MicroscopeLink, SEMControl, StageControl

    try:
        link = MicroscopeLink.simulated(command_delay_s=0.05) if DRY_RUN else MicroscopeLink.open(MICROSCOPE_PORT)
    except Exception as e:  # noqa: BLE001 - no such port, or in use
        pytest.skip(f"microscope port {MICROSCOPE_PORT} not available: {type(e).__name__}: {e}")
    sem = SEMControl(link, polling=False, refresh_on_start=False)
    try:
        sem.refresh(("MAG",)).result(10)
    except Exception as e:  # noqa: BLE001 - port open, SEM silent
        link.close()
        pytest.skip(f"no reply from the SEM on {MICROSCOPE_PORT} (switched off or not connected): {e}")
    stage = StageControl(link, polling=False)
    sem.refresh().result(30)
    original = {k: sem.state.get(k) for k in _RESTORE}
    try:
        stage.refresh().result(30)
        xy0 = {a: stage.positions.get(a) for a in ("X", "Y")}
    except Exception:  # noqa: BLE001 - the stage tests report it
        xy0 = {}
    yield link, sem, stage
    problems = []
    try:
        if link.state == "uncertain":
            link.resync()
        if stage.moving or stage.busy:
            stage.stop().result(10)
            t_end = time.monotonic() + 30
            while stage.busy and time.monotonic() < t_end:
                time.sleep(0.1)
        if xy0 and all(v is not None for v in xy0.values()):
            stage.refresh(["X", "Y"]).result(30)
            back = {a: v for a, v in xy0.items() if abs((stage.positions.get(a) or v) - v) > 0.5}
            if back:
                stage.acknowledge()
                stage.move(back).result(120)
        problems = _restore(sem, original)  # every move ended with #STAGE OFF: the SEM PC has the stage
    finally:
        stage.close()
        sem.close()
        link.close()
    assert not problems, f"not restored: {problems} (original {original})"
