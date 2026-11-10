"""Fixtures for tests on a real NI DAQ (USB-6363 preset).

Run:   pytest tests/hardware --hw Dev1 [--hw-loopback] [--hw-soak 120] -s

The tests drive AO0/AO1 (scan), P0.23 (blanker) and read AI0, exactly like
live imaging.  They never touch the scan-relay lines (PFI6/PFI7), so the SEM
stays on its internal scan unless it was switched over by hand.  A report is
written to ``scripts/logs/hw_imaging_<timestamp>.txt``.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import gc
import os
from pathlib import Path
import sys

import pytest

from easyebl.core import FieldCalibration
from easyebl.hal import DAQConfig, DAQManager
from easyebl.imaging.engine import EngineOptions, ImagingEngine

_REPORT: list[str] = []
_LOG_DIR = Path(__file__).resolve().parents[2] / "scripts" / "logs"


def pytest_collection_modifyitems(config, items):
    dev = config.getoption("--hw")
    for it in items:
        if "tests" + os.sep + "hardware" in str(it.fspath) or "tests/hardware" in str(it.fspath):
            it.add_marker(pytest.mark.hardware)
            if not dev:
                it.add_marker(pytest.mark.skip(reason="hardware test: pass --hw <device> to run"))


def pytest_sessionfinish(session, exitstatus):
    if not _REPORT:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOG_DIR / f"hw_imaging_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(_REPORT) + f"\n\nexit status: {exitstatus}\n", encoding="utf-8")
    sys.stdout.write(f"\n\nhardware report written to {path}\n")


@pytest.fixture(scope="session")
def report():
    def say(msg: str = ""):
        _REPORT.append(msg)
        sys.stdout.write(msg + "\n")

    if not _REPORT:
        say(f"easyEBL imaging hardware tests – {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    return say


@pytest.fixture(scope="session")
def hw_device(pytestconfig):
    dev = pytestconfig.getoption("--hw")
    if not dev:
        pytest.skip("pass --hw <device>")
    nidaqmx = pytest.importorskip("nidaqmx")
    import nidaqmx.system

    names = {d.name: d.product_type for d in nidaqmx.system.System.local().devices}
    if dev not in names:
        pytest.skip(f"device {dev} not present (found {names})")
    return dev, names[dev]


@pytest.fixture(scope="session")
def hw_config(hw_device):
    cfg = DAQConfig.usb6363(hw_device[0])
    return cfg


@pytest.fixture
def hw_manager(hw_config, report):
    mgr = DAQManager(__import__("easyebl.hal", fromlist=["create_backend"]).create_backend("nidaqmx", hw_config)).open()
    yield mgr
    st = mgr.beam_state
    mgr.close()
    assert st.blank, f"beam not blanked at the end of the test: {st}"


@pytest.fixture
def hw_engine(hw_manager, hw_config):
    engines = []

    def make(**opts):
        eng = ImagingEngine(hw_manager, FieldCalibration(100.0, hw_config.ao_limit_v), options=EngineOptions(**opts))
        engines.append(eng)
        return eng

    yield make
    for e in engines:
        e.close()
        assert not e.thread_alive, "scan thread still alive after close()"


def rss_bytes() -> int:
    """Resident memory of this process (Windows API; psutil if available elsewhere)."""
    try:
        import psutil  # noqa: F401

        return psutil.Process().memory_info().rss
    except ImportError:
        pass
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return int(pmc.PagefileUsage)  # private bytes: what a leak grows
    import resource

    gc.collect()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
