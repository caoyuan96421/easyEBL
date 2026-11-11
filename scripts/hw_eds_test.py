"""EDS hardware test: USB-6363 EDS phantom + Amptek DP5 + the easyEBL EDS backend.

Wiring:  AO2 → AI0+  (SE image of the phantom; RSE input)
         AO3 → DP5 analogue input  (reset-type SDD preamp signal of the phantom)
The SEM is not involved: external scan control stays OFF.

The DP5 configuration is read first, saved to the log folder and restored at
the end (also after an error).  During the test the DP5 runs shaping preset 3
(TPEA 9.6 µs) with its own gain, polarity and channel count.

Checks:
  1. electron image (AO2 → AI0): correlation with the phantom image
  2. pulse processor: ICR on the Cu grid vs the phantom photon rate
  3. energy calibration from Cu L + Cu Kα (auto-assignment), residuals, FWHM
  4. point spectra on every phase: AutoID and quantification vs the truth
  5. spectrum map 16×12: element windows vs the phase map, SE map, overhead / pixel
  6. sharing: a live SEM image pauses for an EDS spectrum and resumes

Usage:  python scripts/hw_eds_test.py [--device Dev1] [--cps 20000] [--quick]
Report, spectra (.msa), map (.npz) and PNGs go to scripts/logs/eds_<timestamp>/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from easyebl.core import FieldCalibration  # noqa: E402
from easyebl.eds import EDSController, EDSOptions, EnergyCalibration  # noqa: E402
from easyebl.eds import analysis as A  # noqa: E402
from easyebl.eds.beam import grid_points  # noqa: E402
from easyebl.hal import DAQConfig, DAQManager, create_backend  # noqa: E402
from easyebl.hal.mca import create_pulse_processor  # noqa: E402
from easyebl.imaging.export import auto_levels, to_uint8, write_png  # noqa: E402
from easyebl.xray.generation import Detector  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--backend", default="nidaqmx", choices=["nidaqmx", "sim"])
    ap.add_argument("--mca", default=None, choices=["dp5", "sim"], help="default: dp5 (sim with --backend sim)")
    ap.add_argument("--cps", type=float, default=20e3, help="phantom photon rate on the brightest phase")
    ap.add_argument("--mv-per-kev", dest="mv", type=float, default=6.0, help="phantom SDD step height")
    ap.add_argument("--speed", type=int, default=3, help="DP5 shaping preset during the test")
    ap.add_argument("--quick", action="store_true", help="shorter acquisitions")
    args = ap.parse_args(argv)
    mca_kind = args.mca or ("sim" if args.backend == "sim" else "dp5")

    out = Path(__file__).resolve().parent / "logs" / f"eds_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    results: dict = {"checks": {}}

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    def check(name, ok, detail=""):
        results["checks"][name] = {"ok": bool(ok), "detail": detail}
        say(f"   [{'PASS' if ok else 'FAIL'}] {name}" + (f" – {detail}" if detail else ""))
        return ok

    def png(name, img):
        write_png(out / f"{name}.png", to_uint8(np.nan_to_num(img), auto_levels(np.nan_to_num(img))))

    T = 0.5 if args.quick else 1.0
    cfg = DAQConfig.usb6363(args.device)
    cfg.ai_terminal = "RSE"
    cfg.phantom_output, cfg.phantom_kind, cfg.sdd_output = "aux0", "eds", "aux1"
    cfg.eds_phantom = {"cps": float(args.cps), "mv_per_kev": float(args.mv)}
    cfg.validate()
    if args.backend == "sim":
        from easyebl.hal import SimParams, SimulatedBackend

        be = SimulatedBackend(cfg, SimParams(realtime=True, v_full_scale=cfg.ao_limit_v))
    else:
        be = create_backend("nidaqmx", cfg)
    mgr = DAQManager(be).open()
    field = 100.0
    cal = FieldCalibration(field, cfg.ao_limit_v)
    ph = be.eds_phantom()
    saved_cfg = None
    mca = None
    ctrl = None
    ok_all = True
    try:
        say(f"easyEBL EDS hardware test  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
        say(f"DAQ: {be.capabilities().name}; phantom image {cfg.phantom_physical} → AI0, "
            f"SDD {cfg.sdd_physical} → DP5; {args.cps:g} cps, {args.mv:g} mV/keV")
        # ---------------------------------------------------------------- pulse processor
        if mca_kind == "sim":
            mca = create_pulse_processor("sim", manager=mgr).open()
        else:
            mca = create_pulse_processor("dp5").open()
            dev = mca.device
            saved_cfg = dev.read_config()
            (out / "dp5_config_before.json").write_text(json.dumps(saved_cfg, indent=1))
            say(f"DP5: {mca.info()}; saved its configuration ({len(saved_cfg)} settings)")
            mca.apply_speed(args.speed)
        st = mca.get_settings(["MCAC", "GAIN", "TPEA", "AINP", "THSL", "THFA", "RESL"])
        say(f"processor settings: {st}")
        results["processor"] = {"info": mca.info(), "settings": st}
        opts = EDSOptions(external_control=False)
        ctrl = EDSController(mgr, mca, cal, EnergyCalibration(0.0, 0.01), beam_kv=20.0, options=opts)
        say(f"EDS stream clock {ctrl.rate / 1e6:.4g} MS/s ({cfg.stream_ao_channels} AO channels)")

        # ---------------------------------------------------------------- 1. electron image
        say("")
        say("1. electron image 256×192, 4 µs/px (AO2 → AI0)")
        im = ctrl.acquire_image(256, 192, 4e-6).result(120)
        v = ctrl._to_volts(grid_points((0, 0), field, field, 256, 192))
        exp = ph.intensity_v(v[:, 0], v[:, 1]).reshape(192, 256)
        r = float(np.corrcoef(exp.ravel(), np.nan_to_num(im.data).ravel())[0, 1])
        png("1_electron_measured", im.data)
        png("1_electron_expected", exp)
        gain, off = np.polyfit(exp.ravel(), np.nan_to_num(im.data).ravel(), 1)
        ok_all &= check("electron image matches the phantom", r > 0.9,
                        f"r = {r:.3f}, gain {gain:.3f}, offset {off:+.3f} V")
        if r < 0.5:
            say("   (is AO2 wired to AI0?  AI0 may still be looped to AO0 from the delay test)")

        # ---------------------------------------------------------------- 2. rates on Cu
        say("")
        say("2. Cu grid (centre): count rate and raw spectrum")
        cu = ctrl.acquire_spectrum(10 * T, at_um=(0.0, 0.0), title="Cu grid").result(120)
        cu.save(out / "2_cu_raw.msa")
        rate_true = float(ph.phase_rates[2])
        say(f"   phantom rate {rate_true:.0f} cps; DPP ICR {cu.icr:.0f}, OCR {cu.ocr:.0f} cps, "
            f"dead {100 * cu.dead_fraction:.1f} %, {cu.total:,} counts in {cu.real_time:.2f} s")
        ok_all &= check("input count rate", 0.6 * rate_true < cu.icr < 1.3 * rate_true,
                        f"ICR/true = {cu.icr / rate_true:.2f}")

        # ---------------------------------------------------------------- 3. calibration
        say("")
        say("3. energy calibration from Cu L + Cu Kα")
        peaks = A.find_peaks(cu)
        say("   peaks (channel, counts): " + ", ".join(f"{p.channel:.1f}:{p.area:.0f}" for p in peaks[:6]))
        pairs = A.auto_assign(cu, A.standard_lines("Cu (L + Kα)"), peaks=peaks)
        if len(pairs) < 2:
            ok_all &= check("calibration peaks found", False, "fewer than two peaks matched")
            raise RuntimeError("cannot calibrate – check the DP5 gain/polarity and the AO3 wire")
        fit = A.fit_calibration([p.channel for *_, p in pairs], [e for _, e, _ in pairs],
                                labels=[lab for lab, *_ in pairs], name="Cu L + Kα (phantom)")
        ecal = fit.calibration
        ctrl.set_energy_calibration(ecal)
        say(f"   E = {ecal.offset_kev * 1e3:+.1f} eV + {ecal.gain_kev * 1e3:.4f} eV/ch; "
            f"full scale {ecal.energy(mca.channels):.1f} keV")
        ka = min(peaks, key=lambda p: abs(p.channel - pairs[-1][2].channel))
        fwhm_ka = ka.fwhm_ch * ecal.gain_kev * 1e3
        mn, fano = A.fit_resolution([(8.04, fwhm_ka / 1e3)])
        say(f"   Cu Kα FWHM {fwhm_ka:.0f} eV (→ {mn:.0f} eV at Mn Kα; the phantom's electronic noise "
            f"and the AO DAC add to the processor's own)")
        results["calibration"] = {**ecal.to_dict(), "fwhm_cu_ka_ev": fwhm_ka, "fwhm_mn_ev": mn}
        ok_all &= check("calibration spans the X-ray range", 10.0 < ecal.energy(mca.channels) < 80.0)
        det = Detector(fwhm_mn_ev=float(np.clip(mn, 110, 300)))
        cu = cu.with_calibration(ecal)
        cu.save(out / "3_cu_calibrated.msa")

        # ---------------------------------------------------------------- 4. phases
        say("")
        say("4. point spectra on every phase: AutoID + quantification")
        lab = ph.labels
        pts = grid_points((0, 0), field, field, 200, 200)
        vv = ctrl._to_volts(pts)
        labs = ph.phase_at(vv[:, 0], vv[:, 1])
        results["phases"] = {}
        for k, phase in enumerate(ph.phases):
            idx = None
            for i in np.nonzero(labs == k)[0]:
                x, y = pts[i]
                if max(abs(x), abs(y)) > 45:
                    continue
                nb = ctrl._to_volts([[x + 0.5, y], [x - 0.5, y], [x, y + 0.5], [x, y - 0.5]])
                if np.all(ph.phase_at(nb[:, 0], nb[:, 1]) == k):
                    idx = i
                    break
            if idx is None:
                say(f"   {phase.name}: no clean point found – skipped")
                continue
            at = tuple(pts[idx])
            sp = ctrl.acquire_spectrum(8 * T, at_um=at, title=phase.name).result(120)
            sp.save(out / f"4_{k}_{phase.name.replace(' ', '_')}.msa")
            ids = A.identify_elements(sp, detector=det)
            found = sorted(i.element for i in ids)
            truth = sorted(el for el, _ in phase.composition)
            major = sorted(el for el, w in phase.composition if w >= 0.05)
            q = None
            if found:
                ffit = A.fit_spectrum(sp, found, detector=det)
                q = A.quantify(ffit, 20.0, detector=det, oxygen_by_stoichiometry="O" in truth)
            wt = q.wt() if q else {}
            say(f"   {phase.name:<13} at ({at[0]:5.1f}, {at[1]:5.1f}) µm: {sp.total:7,} cts; found {found}; "
                f"truth {truth}; wt% " + ", ".join(f"{e} {w:.1f}" for e, w in wt.items()))
            results["phases"][phase.name] = {"found": found, "truth": truth, "wt": wt, "counts": sp.total}
            extra, missing = sorted(set(found) - set(truth)), sorted(set(major) - set(found))
            ok_all &= check(f"AutoID {phase.name}", not extra and not missing,
                            (f"extra {extra} " if extra else "") + (f"missing {missing}" if missing else ""))
        # ---------------------------------------------------------------- 5. map
        say("")
        say("5. spectrum map 16×12 @ 20 ms (whole field)")
        t0 = time.monotonic()
        sm = ctrl.acquire_map(16, 12, 0.02).result(600)
        el_t = time.monotonic() - t0
        sm.save(out / "5_map.npz")
        v = ctrl._to_volts(grid_points((0, 0), field, field, 16, 12))
        truth_lab = ph.phase_at(v[:, 0], v[:, 1]).reshape(12, 16)
        per = el_t / (16 * 12)
        say(f"   {el_t:.1f} s total, {per * 1e3:.0f} ms/px (dwell 20 ms → overhead {per * 1e3 - 20:.0f} ms/px)")
        maps = {m.label: m.data for m in A.window_maps(sm, ["Al", "Si", "Cu", "Fe", "Ti", "Au", "Cl"], det)}
        for m, data in maps.items():
            png(f"5_map_{m.replace(' ', '_')}", data)
        png("5_map_SE", sm.se)
        png("5_map_truth", truth_lab.astype(float))
        corr = {}
        for k, el in ((1, "Al Kα"), (0, "Si Kα"), (6, "Au Mα"), (4, "Ti Kα")):
            if el in maps and (truth_lab == k).any():
                corr[el] = float(np.corrcoef((truth_lab == k).ravel().astype(float), maps[el].ravel())[0, 1])
        say("   correlation element window ↔ phase: " + ", ".join(f"{k} {c:.2f}" for k, c in corr.items()))
        ok_all &= check("map element windows follow the phases", corr and min(corr.values()) > 0.5)
        se_exp = ph.intensity_v(v[:, 0], v[:, 1]).reshape(12, 16)
        rse = float(np.corrcoef(se_exp.ravel(), np.nan_to_num(sm.se).ravel())[0, 1])
        ok_all &= check("map SE signal matches the phantom", rse > 0.8, f"r = {rse:.2f}")
        ok_all &= check("map overhead per pixel", per - 0.02 < 0.06, f"{(per - 0.02) * 1e3:.0f} ms")
        results["map"] = {"s_per_px": per, "corr": corr, "se_r": rse}

        # ---------------------------------------------------------------- 6. sharing with the SEM
        say("")
        say("6. live SEM imaging pauses for an EDS spectrum and resumes")
        from easyebl.imaging.engine import EngineOptions, EngineState, ImagingEngine

        eng = ImagingEngine(mgr, cal, options=EngineOptions(latency_s=0.05))
        try:
            eng.update(nx=256, ny=192)
            eng.run()
            f0 = eng.wait_frame(timeout=30)
            job = ctrl.acquire_spectrum(2.0, at_um=(0.0, 0.0), title="during SEM")
            paused = False
            while not job.done():
                paused |= eng.state == EngineState.PAUSED
                time.sleep(0.01)
            job.result()
            deadline = time.monotonic() + 20
            while eng.state != EngineState.LIVE and time.monotonic() < deadline:
                time.sleep(0.05)
            f1 = eng.wait_frame(after_seq=f0.seq if f0 else None, timeout=30)
            ok_all &= check("SEM paused during the EDS job and resumed",
                            paused and eng.state == EngineState.LIVE and f1 is not None,
                            f"paused={paused}, state after={eng.state.value}")
        finally:
            eng.close()
    except Exception as e:  # noqa: BLE001
        ok_all = False
        say(f"!! ERROR: {type(e).__name__}: {e}")
        say(traceback.format_exc())
    finally:
        if ctrl is not None:
            try:
                ctrl.cancel()
                time.sleep(0.2)
            except Exception:  # noqa: BLE001
                pass
        if saved_cfg is not None and mca is not None:
            try:
                mca.device.write_config(saved_cfg)
                after = mca.device.read_config(list(saved_cfg))
                diff = {k: (saved_cfg[k], after.get(k)) for k in saved_cfg if str(after.get(k)) != str(saved_cfg[k])}
                say("")
                say("DP5 configuration restored" + (f" (differences: {diff})" if diff else ""))
            except Exception as e:  # noqa: BLE001
                say(f"!! could not restore the DP5 configuration: {e} – use run_restore_config.bat with "
                    f"{out / 'dp5_config_before.json'}")
        if mca is not None:
            try:
                mca.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            mgr.close()
        except Exception:  # noqa: BLE001
            pass
        say("")
        n_ok = sum(c["ok"] for c in results["checks"].values())
        say(f"RESULT: {'PASS' if ok_all else 'FAIL'} ({n_ok}/{len(results['checks'])} checks)  → {out}")
        (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
        (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
