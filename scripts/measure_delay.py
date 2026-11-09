"""Measure the AO -> AI delay of the scan DAQ with a loopback cable (AO0 wired to AI0).

AO and AI run on the same sample clock, so the delay is measured in samples. That
number goes straight into ``RasterConfig.ai_latency_samples``. Two independent methods
are used at each rate:

* step: square wave, averaged over all periods; the delay is the interpolated 50 %
  crossing after the edge, and the 10-90 % rise time is reported too;
* sine: sine at rate/32, fitted for phase -> delay with sub-sample resolution.

Both ideally give ~1 sample. AI sample k converts on the clock edge that also
updates the DAC, so it sees the previous output value. AO settling adds a fraction
of a sample at high rates.

Usage:  python scripts/measure_delay.py [--device Dev1] [--terminal DIFF|RSE|NRSE] [--amp 2]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from easyebl.core import OutputChunk  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, create_backend  # noqa: E402

RATES = [100e3, 250e3, 500e3, 1e6, 2e6]


# ------------------------------------------------------------------ analysis
def step_delay(ai: np.ndarray, period: int, amp: float):
    """Delay (samples) of the rising edge at index 0 of each period, + 10-90 % rise time."""
    n = (len(ai) // period) * period
    avg = ai[:n].reshape(-1, period).mean(axis=0)  # one averaged period
    lo, hi = np.median(avg[period // 2 + period // 8 : period - 2]), np.median(avg[period // 8 : period // 2 - 2])
    pre = 4  # include a few samples before the edge
    norm = np.roll((avg - lo) / (hi - lo), pre)

    def crossing(level):
        k = int(np.argmax(norm[: period // 2] >= level))
        if k == 0:
            return float(-pre)
        y0, y1 = norm[k - 1], norm[k]
        return (k - 1) + (level - y0) / (y1 - y0) - pre

    d50 = crossing(0.5)
    rise = crossing(0.9) - crossing(0.1)
    return d50, rise, avg, (hi - lo) / (2 * amp)


def sine_delay(ao: np.ndarray, ai: np.ndarray, period: int) -> tuple[float, float]:
    """Delay (samples) of AI relative to AO from the phase at f = 1/period; also gain."""
    k = np.arange(len(ai))
    w = 2 * np.pi / period
    basis = np.column_stack([np.sin(w * k), np.cos(w * k), np.ones_like(k, float)])
    (a_i, b_i, _), *_ = np.linalg.lstsq(basis, ai, rcond=None)
    (a_o, b_o, _), *_ = np.linalg.lstsq(basis, ao, rcond=None)
    ph_i, ph_o = np.arctan2(b_i, a_i), np.arctan2(b_o, a_o)
    dphi = (ph_o - ph_i) % (2 * np.pi)  # AI lags AO
    return dphi / w, np.hypot(a_i, b_i) / np.hypot(a_o, b_o)


# ------------------------------------------------------------------ measurement
def run(h, rate: float, wave: np.ndarray, settle: int) -> np.ndarray:
    n = len(wave)
    chunk = OutputChunk(wave, np.zeros(n), np.ones(n, bool))
    r = h.run_stream(rate, [chunk], ai_channels=["video"], timeout=60)
    return r.ai[0][settle:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--terminal", default=None, help="AI terminal config (default: try DIFF and RSE)")
    ap.add_argument("--amp", type=float, default=2.0, help="test amplitude (V)")
    ap.add_argument("--backend", default="nidaqmx")
    args = ap.parse_args(argv)

    log_path = Path(__file__).with_name(f"delay_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt")
    out: list[str] = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    terminals = [args.terminal] if args.terminal else ["DIFF", "RSE"]
    for term in terminals:
        cfg = replace(DAQConfig.usb6363(args.device), ai_terminal=term)
        mgr = DAQManager(create_backend(args.backend, cfg)).open()
        h = mgr.get_handle("delay")
        say(f"== AO0 -> AI0, AI terminal {term}, amplitude ±{args.amp} V")
        say(f"{'rate':>9} | {'step d50':>9} {'rise10-90':>9} {'gain':>6} | {'sine d':>8} {'gain':>6} | "
            f"{'delay µs':>9} {'(d-1)·T µs':>10}")
        try:
            for rate in RATES:
                n = int(max(40_000, 0.2 * rate))
                settle = n // 10
                # square wave, period 64 samples, rising edge at index 0 of each period
                P = 64
                k = np.arange(n)
                sq = np.where((k % P) < P // 2, args.amp, -args.amp).astype(float)
                ai_sq = run(h, rate, sq, settle)
                # align so that a rising edge sits at index 0
                start = (P - settle % P) % P
                d50, rise, avg, g_sq = step_delay(ai_sq[start:], P, args.amp)
                d50 -= step_delay(sq[settle:][start:], P, args.amp)[0]  # relative to the AO edge
                # sine at rate/32
                Ps = 32
                sn = args.amp * np.sin(2 * np.pi * k / Ps)
                ai_sn = run(h, rate, sn, settle)
                d_sine, g_sine = sine_delay(sn[settle:], ai_sn, Ps)
                say(f"{rate:>9.0f} | {d50:>9.3f} {rise:>9.3f} {g_sq:>6.3f} | {d_sine:>8.3f} {g_sine:>6.3f} | "
                    f"{d_sine / rate * 1e6:>9.3f} {(d_sine - 1) / rate * 1e6:>10.3f}")
                if rate == RATES[-1] or rate == RATES[0]:
                    say("          step (averaged, samples -2..6 around the edge, normalised): "
                        + " ".join(f"{v:+.3f}" for v in ((np.r_[avg[-2:], avg[:7]] - avg[P // 2 + P // 4])
                                                          / (avg[P // 4] - avg[P // 2 + P // 4]))))
        except Exception as e:  # noqa: BLE001
            say(f"FAILED: {type(e).__name__}: {e}")
        finally:
            mgr.close()
        say("")
    say("Delay in samples -> RasterConfig.ai_latency_samples (default 1.0).")
    say("'(d-1)·T' is the analogue part beyond the one-sample conversion latency.")
    log_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    say(f"log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
