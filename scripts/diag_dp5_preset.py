"""DP5 preset diagnostic: does PRER (preset real time) stop the MCA?  Also PRET for comparison.

Usage:  python scripts/diag_dp5_preset.py      (log: scripts/logs/diag_dp5_preset.txt)
The DP5 configuration is restored afterwards.
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from easyebl.hal.dp5 import DP5  # noqa: E402

out = []


def say(m):
    print(m, flush=True)
    out.append(str(m))


dp = DP5()
dp.open()
try:
    orig = dp.read_config(["PRER", "PRET", "PREC", "MCAE", "GPME"])
    say(f"before: {orig}")
    for key, val in (("PRER", "0.500"), ("PRER", "0.5"), ("PRET", "0.5"), ("PRER", "1")):
        try:
            res = dp.write_config({"PRER": "OFF", "PRET": "OFF", key: val})
            say(f"write {key}={val}: {res}")
        except Exception as e:  # noqa: BLE001
            say(f"write {key}={val} failed: {e!r}")
            continue
        say(f"  readback {dp.read_config(['PRER', 'PRET'])}")
        dp.read_spectrum(clear=True)
        dp.enable_mca()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            st = dp.status()
            say(f"  t={time.monotonic() - t0:5.2f} real={st.real_time:.3f} acc={st.accumulation_time:.3f} "
                f"mca={st.mca_enabled} prer_done={st.preset_real_time_done} flags35=0x{st.raw[35]:02x}"
                if hasattr(st, "raw") else
                f"  t={time.monotonic() - t0:5.2f} real={st.real_time:.3f} acc={st.accumulation_time:.3f} "
                f"mca={st.mca_enabled} prer_done={st.preset_real_time_done}")
            if not st.mca_enabled:
                break
            time.sleep(0.25)
        dp.disable_mca()
finally:
    try:
        dp.write_config({k: v for k, v in orig.items() if k in ("PRER", "PRET", "PREC")})
        say(f"restored: {dp.read_config(['PRER', 'PRET', 'PREC'])}")
    finally:
        dp.close()
        Path(__file__).resolve().parent.joinpath("logs").mkdir(exist_ok=True)
        Path(__file__).resolve().parent.joinpath("logs", "diag_dp5_preset.txt").write_text("\n".join(out), encoding="utf-8")
