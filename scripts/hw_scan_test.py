"""Hardware bring-up for the NI USB-6363 scan path (no SEM required).

Run on the Windows PC that has the board, from the repo folder with the venv active:

    python scripts/hw_scan_test.py --device Dev1

What it does
  1. Sets relay-enable PFI6 and PFI7 HIGH (required before taking control).
  2. Parks the beam blanked, then checks the blanker/park path with a static write.
  3. Takes dummy raster images (video input = floating / noise) at several sample
     rates, checking the actual clock rate, buffer underflow/overflow warnings,
     sample count and the noise statistics of each image.
  4. Optional loopback (--loopback): if AO0 is wired to AI0, it checks that the
     AI follows the AO ramp, and it reports the AI latency in samples.

It writes a log to scripts/hw_scan_test_<timestamp>.txt. Paste that file's contents
back if anything looks wrong.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from easyebl.core import FieldCalibration, OutputChunk  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, create_backend  # noqa: E402
from easyebl.imaging import RasterConfig, RasterImager  # noqa: E402

RATES = [100e3, 250e3, 500e3, 1e6, 2e6]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--loopback", action="store_true", help="AO0 wired to AI0 (checks latency)")
    ap.add_argument("--frames", type=int, default=2)
    args = ap.parse_args(argv)

    log_path = Path(__file__).with_name(f"hw_scan_test_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt")
    lines: list[str] = []

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    cfg = DAQConfig.usb6363(args.device)
    cfg.validate()
    say(f"config: {cfg.device} AO {cfg.ao_x},{cfg.ao_y}  blank {cfg.blank_line} "
        f"(active {'high' if cfg.blank_active_high else 'low'})  video {cfg.ai_channels}")

    mgr = DAQManager(create_backend("nidaqmx", cfg)).open()
    h = mgr.get_handle("hw_test")
    cal = FieldCalibration(field_size_um=100.0, v_full_scale=cfg.ao_limit_v)
    ok = True
    try:
        # 1. relay enables
        h.claim("relay_pfi6", "relay_pfi7")
        for name in ("relay_pfi6", "relay_pfi7"):
            h.set_line(name, True)
        say(f"relay enables set HIGH: {mgr.line_state}")

        # 2. beam parked and blanked
        h.park(0.0, 0.0, blank=True)
        say(f"beam state after park: {mgr.beam_state}")

        # 3. dummy images at several rates
        say("")
        say(f"{'rate':>10} {'requested':>12} {'samples':>10} {'frame s':>9} "
            f"{'noise std':>10} {'NaN %':>7} {'warnings'}")
        for rate in RATES:
            raster = RasterConfig(nx=256, ny=256, width_um=50, rate=rate, dwell_s=4 / rate,
                                  frames=args.frames, bidirectional=True, ai_channels=["video"])
            imager = RasterImager(h, cal)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                img = imager.acquire(raster, timeout=120)
                dt = time.perf_counter() - t0
            data = np.asarray(img.image("video"), dtype=float)
            finite = np.isfinite(data)
            noise = float(np.nanstd(data)) if finite.any() else float("nan")
            msgs = "; ".join(str(w.message) for w in caught) or "-"
            ns = raster.total_samples if hasattr(raster, "total_samples") else RasterImager(h, cal).plan(raster).total
            say(f"{rate:>10.0f} {rate:>12.0f} {ns:>10d} {dt:>9.2f} {noise:>10.4g} "
                f"{100 * (1 - finite.mean()):>7.2f} {msgs}")
            if not finite.all() and finite.mean() < 0.9:
                ok = False
                say(f"  !! more than 10% of pixels missing at {rate:g} S/s (buffer or timing problem)")

        # 4. live mode: one continuous stream, frames back-to-back
        say("")
        say(f"{'live rate':>10} {'frame':>9} {'frames':>7} {'fps':>7} {'theory':>7} {'NaN %':>6}")
        for rate, nx, ny in [(500e3, 256, 256), (1e6, 256, 256), (2e6, 128, 128), (2e6, 512, 512)]:
            raster = RasterConfig(nx=nx, ny=ny, width_um=50, rate=rate, dwell_s=4 / rate,
                                  bidirectional=True, ai_channels=["video"])
            imager = RasterImager(h, cal)
            plan = imager.plan(raster)
            n_want = max(3, int(round(3.0 / plan.frame_time_s)))  # ~3 s of streaming
            stamps, nan = [], []

            def on_frame(k, img, stamps=stamps, nan=nan):
                stamps.append(time.perf_counter())
                nan.append(float(np.isnan(img.image()).mean()))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                n = imager.live(raster, on_frame, max_frames=n_want, timeout=120)
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
            msgs = "; ".join(str(w.message) for w in caught)
            say(f"{rate:>10.0f} {nx}x{ny:<5} {n:>7d} {fps:>7.2f} {1 / plan.frame_time_s:>7.2f} "
                f"{100 * max(nan):>6.2f} {msgs}")
            if n != n_want or max(nan) > 0 or fps < 0.95 / plan.frame_time_s:
                ok = False
                say("  !! live stream lost frames or ran slow")

        # 5. optional loopback
        if args.loopback:
            n = 20000
            ramp = np.linspace(-cfg.ao_limit_v * 0.8, cfg.ao_limit_v * 0.8, n)
            r = h.run_stream(1e6, [OutputChunk(ramp, np.zeros(n), np.ones(n, bool))],
                             ai_channels=["video"], timeout=60)
            ai = r.ai[0]  # blanked: the video AI still samples the AO-driven input
            lag = _best_lag(ai, ramp)
            say(f"loopback AO0->AI0 @1 MS/s: best lag {lag} samples; "
                f"ramp corr {np.corrcoef(np.roll(ai, -lag), ramp)[0, 1]:.5f}")
    except Exception as e:  # noqa: BLE001
        ok = False
        say(f"FAILED: {type(e).__name__}: {e}")
    finally:
        try:
            h.park(0.0, 0.0, blank=True)
            h.set_line("relay_pfi6", False)
            h.set_line("relay_pfi7", False)
        except Exception as e:  # noqa: BLE001
            say(f"cleanup problem: {e}")
        mgr.close()
        say(f"final beam state: {mgr.beam_state}")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        say(f"log written to {log_path}")
    return 0 if ok else 1


def _best_lag(ai: np.ndarray, ref: np.ndarray, max_lag: int = 200) -> int:
    best, best_c = 0, -np.inf
    for lag in range(max_lag):
        a = ai[lag:]
        r = ref[: len(a)]
        if len(a) < 100 or np.std(a) == 0:
            continue
        c = np.corrcoef(a, r)[0, 1]
        if c > best_c:
            best, best_c = lag, c
    return best


if __name__ == "__main__":
    raise SystemExit(main())
