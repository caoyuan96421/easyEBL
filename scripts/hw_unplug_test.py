"""Manual hardware test: pull the USB-6363 cable while scanning, plug it back in.

Live imaging runs on the USB-6363 while you unplug and re-plug the USB cable
(or switch the DAQ off and on) a few times.  For every cycle the script
records:

* loss detected  – last good frame → engine DISCONNECTED (how fast the loss is noticed)
* reconnected    – device answers again → HAL connected (watchdog + device reset + safe state)
* resumed        – HAL connected → first complete new frame
* the beam is blanked after the reconnect, external scan control (if used)
  is restored, the scan thread is alive and no exception escaped.

Usage:  python scripts/hw_unplug_test.py [--device Dev1] [--cycles 3] [--external] [--timeout 300]

``--external`` also switches the SEM to external scan, to check that it is
restored after each reconnect (and back to internal at the end).
The log goes to scripts/logs/hw_unplug_<timestamp>.txt.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):  # Windows consoles may use a legacy code page (e.g. GBK)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, create_backend  # noqa: E402
from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--external", action="store_true", help="also use (and check) external scan control")
    ap.add_argument("--timeout", type=float, default=300.0, help="give up after this many seconds")
    args = ap.parse_args(argv)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"hw_unplug_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt"
    lines: list[str] = []
    t_start = time.perf_counter()

    def say(msg=""):
        stamp = f"[{time.perf_counter() - t_start:8.3f}] "
        print(stamp + msg, flush=True)
        lines.append(stamp + msg)

    cfg = DAQConfig.usb6363(args.device)
    mgr = DAQManager(create_backend("nidaqmx", cfg), reconnect_interval_s=0.5).open(allow_missing=True)
    hal_events = []
    mgr.add_listener(lambda ev, d: hal_events.append((time.perf_counter(), ev, d)) if ev in (
        "device_lost", "device_restored", "external_control", "error") else None)
    eng = ImagingEngine(mgr, FieldCalibration(100.0, cfg.ao_limit_v), options=EngineOptions(latency_s=0.1))
    eng.update(nx=640, ny=480, dwell_s=0.5e-6)
    ext = None
    ok = True
    results = []
    try:
        if not mgr.wait_connected(30):
            say("device not connected – plug it in first")
            return 2
        if args.external:
            ext = mgr.get_handle("external_scan")
            ext.set_external_control(True)
            say(f"external scan ON {cfg.external_control_map()}")
        eng.run()
        f = eng.wait_frame(complete=True, timeout=10)
        say(f"live: {f.config.nx}x{f.config.ny}, {eng.status().summary()}")
        deadline = time.monotonic() + args.timeout
        for cycle in range(1, args.cycles + 1):
            say("")
            say(f"=== cycle {cycle}/{args.cycles}: UNPLUG the USB cable (or switch the DAQ off) now")
            t_last = time.perf_counter()
            seq = eng.frames.seq
            while eng.state != EngineState.DISCONNECTED:
                g = eng.wait_frame(seq, timeout=0.5, complete=True)
                if g is not None:
                    seq, t_last = g.seq, time.perf_counter()
                if time.monotonic() > deadline:
                    raise TimeoutError("no disconnect seen")
            t_lost = time.perf_counter()
            say(f"loss detected {(t_lost - t_last) * 1e3:.0f} ms after the last good frame "
                f"(state {eng.state.value}, beam {mgr.beam_state})")
            say("    now PLUG it back in (or switch the DAQ on)")
            while not mgr.connected:
                time.sleep(0.01)
                if time.monotonic() > deadline:
                    raise TimeoutError("device did not come back")
            t_conn = time.perf_counter()
            say(f"reconnected {t_conn - t_lost:.1f} s after the loss; beam {mgr.beam_state}, "
                f"external control {mgr.external_control}")
            g = eng.wait_frame(eng.frames.seq, timeout=20, complete=True)
            if g is None:
                raise TimeoutError("scanning did not resume")
            t_res = time.perf_counter()
            st = eng.status()
            say(f"resumed: first new frame {(t_res - t_conn) * 1e3:.0f} ms after reconnect; {st.summary()}")
            checks = {
                "thread alive": eng.thread_alive,
                "state live": eng.state == EngineState.LIVE,
                "external restored": (mgr.external_control is True) if args.external else True,
            }
            bad = [k for k, v in checks.items() if not v]
            if bad:
                ok = False
                say(f"  !! failed: {bad}")
            results.append((t_lost - t_last, t_conn - t_lost, t_res - t_conn))
        say("")
        say("summary (s): detect / disconnected / resume")
        for i, (a, b, c) in enumerate(results, 1):
            say(f"  cycle {i}: {a:6.3f} / {b:6.1f} / {c:6.3f}")
        say(f"HAL: disconnects {mgr.disconnects}, reconnects {mgr.reconnects}; "
            f"engine: disconnects {eng.status().disconnects}, streams {eng.status().streams}")
        for t, ev, d in hal_events:
            say(f"  hal event {t - t_start:8.3f} {ev} {d}")
    except Exception as e:  # noqa: BLE001
        ok = False
        say(f"FAILED: {type(e).__name__}: {e}")
    finally:
        eng.close()
        if ext is not None:
            ext.close()
        say(f"final: external control {mgr.external_control}, beam {mgr.beam_state}")
        mgr.close()
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"log written to {log_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
