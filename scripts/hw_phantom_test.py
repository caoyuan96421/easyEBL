"""Dummy-image test on the USB-6363: phantom test chart on AO2, looped back to AI0.

Wiring: AO2 → AI0+.  The default RSE input mode needs nothing else; with
--ai-terminal DIFF also wire AO GND → AI0− (an open AI0− floats into the rail).
The scan outputs (AO0/AO1) and the blanker run exactly as in live imaging;
the SEM is not involved (external scan control stays OFF).

What is measured (every image is compared with the chart it should show):
  1. fidelity  – full field: gain/offset/R² of measured vs expected
  2. registration – sub-pixel shift measured vs expected, unidirectional and
     bidirectional (a residual shift means the AI latency setting is off)
  3. field of view – several scan widths (zoom)
  4. beam shift – commanded vs measured shift
  5. noise vs speed – frame-to-frame noise and residual at increasing dwell
     (with --noise V the phantom adds white noise: expect σ/√(samples per pixel))
Images (measured and expected) are saved as PNG next to the log.

Usage:  python scripts/hw_phantom_test.py [--device Dev1] [--noise 0.2] [--output aux0]
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):  # Windows consoles may use a legacy code page (e.g. GBK)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, create_backend  # noqa: E402
from easyebl.imaging.engine import EngineOptions, ImagingEngine  # noqa: E402
from easyebl.imaging.export import auto_levels, to_uint8, write_png  # noqa: E402
from easyebl.imaging.phantom_check import fit_to_expected, pair_noise, subpixel_shift  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--output", default="aux0", choices=["aux0", "aux1"], help="aux output wired to AI0")
    ap.add_argument("--noise", type=float, default=0.0, help="synthetic white noise on the phantom (V rms)")
    ap.add_argument("--backend", default="nidaqmx", choices=["nidaqmx", "sim"])
    ap.add_argument("--ai-terminal", dest="ai_terminal", default="RSE", choices=["RSE", "DIFF"],
                    help="video input mode (default RSE)")
    args = ap.parse_args(argv)

    out_dir = Path(__file__).resolve().parent / "logs" / f"phantom_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    def save(name, img, lv=None):
        write_png(out_dir / f"{name}.png", to_uint8(img, lv or auto_levels(img)))

    cfg = DAQConfig.usb6363(args.device)
    cfg.phantom_output = args.output
    cfg.phantom_noise_v = args.noise
    cfg.ai_terminal = args.ai_terminal
    cfg.validate()
    if args.backend == "sim":
        from easyebl.hal import SimParams, SimulatedBackend

        be = SimulatedBackend(cfg, SimParams(noise_sigma=0.0, v_full_scale=cfg.ao_limit_v))
    else:
        be = create_backend("nidaqmx", cfg)
    mgr = DAQManager(be).open()
    field = 100.0
    cal = FieldCalibration(field, cfg.ao_limit_v)
    eng = ImagingEngine(mgr, cal, options=EngineOptions(latency_s=0.1))
    ph = be.phantom_source()
    ok = True

    def frame(n=2):
        f = None
        for _ in range(n):
            f = eng.wait_frame(None if f is None else f.seq, complete=True, timeout=60)
            if f is None:
                raise TimeoutError("no frame")
        return f

    def two_frames():
        a = frame(2)
        b = eng.wait_frame(a.seq, complete=True, timeout=60)
        return a, b

    try:
        say(f"phantom on {args.output} = {cfg.phantom_physical} → wire to AI0 ({cfg.ai_terminal}); "
            f"clock {eng.rate / 1e6:.4f} MS/s (3 AO channels), synthetic noise {args.noise:g} V")
        T = 1 / eng.rate
        # 1. fidelity, full field
        eng.update(nx=512, ny=512, width_um=0.9 * field, dwell_s=4 * T, bidirectional=False)
        eng.run()
        f = frame(3)
        exp = ph.expected_image(f.config, cal)
        fit = fit_to_expected(f.image, exp)
        save("1_full_measured", f.image)
        save("1_full_expected", exp)
        say("")
        say(f"1. full field 512×512, 4 samples/px: {fit}")
        if fit.r2 < 0.98:
            ok = False
            say("   !! the image does not match the chart – check the AO2→AI0 wire and the AI terminal mode")
        # 2. registration
        say("")
        say("2. registration (measured vs expected, pixels; + = measured lies right/down)")
        for bidir in (False, True):
            eng.update(nx=400, ny=300, width_um=40.0, dwell_s=2 * T, bidirectional=bidir, center_um=(0.0, 0.0))
            f = frame(3)
            exp = ph.expected_image(f.config, cal)
            dx, dy = subpixel_shift(f.image, exp)
            fwd = subpixel_shift(f.image[0::2], exp[0::2])[0]
            bwd = subpixel_shift(f.image[1::2], exp[1::2])[0]
            say(f"   {'bidirectional ' if bidir else 'unidirectional'}: dx {dx:+.3f}, dy {dy:+.3f}; "
                f"even rows dx {fwd:+.3f}, odd rows dx {bwd:+.3f}")
            if abs(dx) > 0.5 or abs(fwd - bwd) > 0.5:
                say("   !! misregistered – adjust 'AI latency' (Advanced…) by the row difference / 2")
                ok = False
        # 3. field of view
        say("")
        say("3. field of view (zoom)")
        eng.update(bidirectional=False)
        for width in (0.9 * field, 45.0, 20.0, 5.0, 1.0):
            eng.update(nx=320, ny=240, width_um=width, dwell_s=4 * T)
            f = frame(3)
            exp = ph.expected_image(f.config, cal)
            fit = fit_to_expected(f.image, exp)
            save(f"3_fov_{width:g}um", f.image)
            say(f"   width {width:6.1f} µm (pixel {width / 320 * 1e3:7.1f} nm): {fit}")
            if np.var(exp) > 0 and fit.r2 < 0.95:
                ok = False
        # 4. beam shift
        say("")
        say("4. beam shift (width 20 µm, 320 px → 62.5 nm/px)")
        eng.update(nx=320, ny=320, width_um=20.0, dwell_s=4 * T, center_um=(0.0, 0.0))
        ref = frame(3)
        for sx, sy in ((2.0, 0.0), (0.0, 2.0), (-3.0, -1.5), (5.0, 5.0)):
            eng.set_beam_shift(sx, sy)
            f = frame(3)
            exp = ph.expected_image(f.config, cal)
            fit = fit_to_expected(f.image, exp)
            dx, dy = subpixel_shift(f.image, ref.image)  # how far the content moved in the image
            # the content moves opposite to the beam: +x beam → content left; +y beam → content down
            want = (-sx / f.pixel_size_um[0], sy / f.pixel_size_um[1])
            say(f"   beam shift ({sx:+.1f}, {sy:+.1f}) µm: content moved ({dx:+.2f}, {dy:+.2f}) px, "
                f"expected ({want[0]:+.2f}, {want[1]:+.2f}); {fit}")
            if max(abs(dx - want[0]), abs(dy - want[1])) > 0.5 and abs(want[0]) < 160 and abs(want[1]) < 160:
                ok = False
        eng.reset_beam_shift()
        # 5. noise vs speed
        say("")
        say("5. noise vs scan speed (256×256, width 40 µm)")
        say(f"   {'dwell µs':>9} {'spp':>4} {'frame s':>8} {'pair noise mV':>14} {'residual mV':>12}"
            + (f" {'σ/√spp mV':>10}" if args.noise else ""))
        for spp in (1, 2, 4, 8, 16, 32, 64):
            eng.update(nx=256, ny=256, width_um=40.0, dwell_s=spp * T)
            a, b = two_frames()
            exp = ph.expected_image(a.config, cal)
            fit = fit_to_expected(a.image, exp)
            pn = pair_noise(a.image, b.image)
            line = (f"   {a.config.actual_dwell_s * 1e6:9.3f} {a.config.samples_per_pixel:4d} "
                    f"{eng.frame_time_s():8.3f} {pn * 1e3:14.3f} {fit.resid_rms_v * 1e3:12.3f}")
            if args.noise:
                line += f" {args.noise / np.sqrt(spp) * 1e3:10.3f}"
                if abs(pn / (args.noise * fit.gain / np.sqrt(spp)) - 1) > 0.2:
                    ok = False
                    line += "  !!"
            say(line)
            if spp in (1, 64):
                save(f"5_noise_spp{spp}", a.image)
        st = eng.status()
        say("")
        say(f"engine: streams {st.streams}, underruns {st.underruns}, {st.summary()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        say(f"FAILED: {type(e).__name__}: {e}")
    finally:
        eng.close()
        mgr.close()
        say(f"result: {'PASS' if ok else 'CHECK THE !! LINES'}; images in {out_dir}")
        (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
