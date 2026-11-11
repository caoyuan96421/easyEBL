# Acquisition and tuning guide

How the DP5 turns preamp steps into a spectrum, what each setting does, and how to use the driver's
acquisition modes, speed presets and dead-time tools.

## 1. Signal chain

```
reset preamp ──> analog front end ──> ADC ──┬──> fast channel ──> pile-up / reset logic ──┐
 (steps on a      AINP polarity      CLCK   │     TPFA, THFA        PURE, RESL             │ veto
  leakage ramp)   INOF offset        20/80  │                                              v
                  GAIN               MHz    └──> slow channel ──> peak detect ──────> MCA histogram
                                                  TPEA, TFLA        THSL, PDMD          MCAC, PRET
                                                  BLRM/BLRD/BLRU
```

* **Input:** the SDD preamp output is a staircase. Each X-ray adds a step whose height is proportional to its energy, and the
  leakage current adds a slow ramp. At the top of the range the preamp **resets** back down.
* **Front end:**
  * `AINP` must match the step polarity (`POS` for upward steps).
  * `INOF` keeps the ramp inside the ADC range.
  * `GAIN` (= coarse step `GAIA` × fine `GAIF`) sets keV per channel, i.e. the energy range of the spectrum.
* **Digital processing:** everything after the 80 MHz ADC is digital, running in the FPGA. Two filters run in parallel on the same samples:
  * **Fast channel (`TPFA`, ~50–400 ns):** noisy but narrow. It detects *when* each photon arrived and separates close
    pairs. A trigger happens when its output crosses `THFA`.
  * **Slow channel (`TPEA`, `TFLA`):** a **trapezoidal filter**. It subtracts the average of the samples before the step from
    the average after it, so the flat top of the trapezoid equals the step height, i.e. the energy. Longer averaging means less noise
    and better FWHM.

## 2. Settings that matter for an SDD

| Setting | What it does | Practical guidance |
|---|---|---|
| `TPEA` peaking time | Length of each averaging window. Sets the noise (FWHM) and each pulse's length, 2·TPEA+TFLA | 1 µs for high count rates, 4–10 µs normal, 16–25 µs best resolution. Use the speed presets. |
| `TFLA` flat top | Gap between the two windows. Must exceed the spread in step rise times | 0.2 µs for a real SDD. 0.4–0.6 µs for the NI 6363 simulator (its steps rise in 0.5 µs). |
| `TPFA` fast peaking | Fast filter length | Longer: lower fast noise, so a lower usable `THFA` and lighter elements detected, but worse pair resolution. |
| `THFA` fast threshold | Trigger level on the fast channel (0–255.93, not %) | Just above noise. Too low: noise triggers and false pile-up. Too high: low-energy photons are missed, and ICR drops below OCR. `auto_threshold()` with the **beam off**. |
| `THSL` slow threshold | Lowest measured energy stored (% of full scale) | Just above the noise, below the lowest line of interest. The presets use 2 % (speed 1), 1.25 % (speed 2) and 1 % (3–5); `auto_threshold()` sets it from the noise. |
| `PURE` pile-up rejection | Discards pulses with another trigger inside their trapezoid | Keep `ON` for EDS. It can't see pairs closer than about `TPFA`, which end up as sum peaks. |
| `RESL` reset lockout | Blanks the input for N µs after each preamp reset | Just longer than the real reset and settling time (check with a scope). Too long wastes live time at high rates. |
| `BLRM/BLRD/BLRU` | Baseline restorer: removes the offset the leakage ramp adds | `BLRM=1`. The presets set `BLRU` per speed. |
| `GAIN` | Energy scale | Calibrate on a known line (e.g. Mn Kα 5.895 keV). Speed presets never change it. |
| `MCAC` | Number of channels | 2048 is plenty for an SDD (about 25 eV/channel at GAIN ≈ 10). |
| `PRET` / `PRER` / `PREC` | Stop after an accumulation time, real time or count total | The DP5 then clears the "MCA enabled" status bit, and the driver emits `stopped` with reason `"preset"`. |
| `CUSP` | Blends a cusp shape (more weight near the step) into the trapezoid | A few % better FWHM at long peaking. Unreadable on FPGA 7.07, and not used by the presets. |
| `RTDE` | Rise-time discrimination | `OFF` for an SDD. |

Every command, with SDK limits and Amptek's defaults, is in [settings_reference.md](settings_reference.md).

## 3. MCA mode vs Delta mode

Delta is not a DP5 setting. It's the host's choice of request on each poll, exactly as in Amptek's GUIs.

| | `start(mode="mca")` | `start(mode="delta")` |
|---|---|---|
| Request per poll | spectrum + status (`02 03`) | spectrum + status, **then clear** (`02 04`) |
| `latest()` | cumulative since start/clear | counts since the previous poll only |
| `accumulated()` | same as `latest()` | host-side sum of every poll since start/clear |
| `interval_rates()` | from successive cumulative statuses | from the interval's own status |
| Typical use | normal spectra, preset-time runs | live rate display, per-pixel or per-dwell spectra for mapping, auto-tuning |

* **Polling and timing:** `poll_interval` (default 0.5 s, minimum 0.05 s) sets how often the worker reads. Each read is one USB round trip
  plus the transfer: about 3 kB at 1024 channels, about 25 kB at 8192.
* **Short dwells:** for dwells much shorter than about 50–100 ms, sum several pixels per read.
* **Switching modes:** `set_mode()` can switch while running. It resets the host-side sum.
* **Hardware MCS mode (`MCAS=MCS`, `MCST`, `MCSL`/`MCSH`):** this is different again. The DP5 itself fills one time bin per channel with counts
  in an energy window.
* **No list mode:** list mode is **not available** with FW 6.10.02 + FPGA 7.07.

## 4. Dead time

Dead time = 1 − OCR/ICR: the fraction of photons that reached the detector but not the spectrum, mostly
pile-up rejections and the reset lockout.

* **What's measured:** ICR is the fast-channel input rate and OCR the slow-channel output rate. Both come from the status block.
* **The model:** the DP5 behaves like a paralyzable system. Each counted pulse blocks τ ≈ 1.07 × (2·TPEA + TFLA); the factor was
  1.07–1.14 on this DP5 (higher with the presets' short TPFA, which rejects more pile-up). `auto_speed()` recalibrates
  it from each measurement. So OCR = ICR · exp(−ICR · τ), which peaks at ICR = 1/τ, i.e. **63 % dead time**.
  Beyond that, more beam current gives *fewer* counts.
* **Rough targets:**

| Dead time | Meaning |
|---|---|
| < 10–20 % | You could use more beam current, or a slower (better-resolution) speed. |
| **20–50 %** | **Sweet spot**: near-maximum throughput with clean spectra. |
| > 60 % | Past the throughput peak: fewer counts, more uncaught pile-up and sum peaks, less accurate live-time correction. |

For quantitative work, keep dead time similar between standards and unknowns.

## 5. Speed presets 1–5

| Speed | TPEA | TPFA | Peak output | Input rate for ~30 % dead | Use |
|---|---|---|---|---|---|
| 1 | 1.0 µs | 50 ns | ~156 kcps | ~150 kcps | fast mapping |
| 2 | 3.2 µs | 100 ns | ~52 kcps | ~50 kcps | mapping / survey |
| 3 | 9.6 µs | 100 ns | ~18 kcps | ~17 kcps | Amptek's default SDD setting |
| 4 | 16.0 µs | 200 ns | ~11 kcps | ~10 kcps | point analysis |
| 5 | 25.6 µs | 400 ns | ~7 kcps | ~6.5 kcps | best resolution, light elements |

* **Settings common to all speeds:** TFLA 0.2 µs, THFA 30, CLCK 80, PURE ON, BLRM 1, BLRD 3, with `BLRU` set per speed.
  THSL is 2 % for speed 1, 1.25 % for speed 2 and 1 % for speeds 3–5 (see section 6 for why).
* **Gain is kept:** the DP5 re-derives its gain whenever TPEA changes (measured: TPEA alone moved GAIN 8.748 to anywhere
  between 4.8 and 14.0). `apply_speed()` reads GAIN first and sends it again after the shaping, then checks the readback.
* **Where the values come from:** speeds 1, 3 and 5 use the shaping from Amptek's SDK files "SDD High Rate", "SDD Default" and "SDD Low Energy". Speeds 2 and 4 are geometric
  interpolations.
* **What doesn't change:** gain, polarity, input offset, reset lockout and channel count are never touched, so peaks don't move.

```python
dp.apply_speed(2)
dp.apply_speed(5, overrides={"THFA": 25}, autoset_fast_threshold=False)
dp.current_speed()            # 1..5 or None
```

* **Rejected settings:** if the DP5 rejects one of a preset's settings, it is skipped and listed in `dp.last_preset_skipped`.
* **Spectrum after a speed change:** when you change speed during an acquisition, the spectrum is cleared by default (`clear=True`), so you never mix two
  shapings in one spectrum.

## 6. Thresholds from the noise: `auto_threshold()`

Run it with the **beam off** (no X-rays on the detector):

```python
res = dp.auto_threshold(speeds="all")      # or "current" (default), or e.g. [1, 2]
# {1: {"THFA": 30.9, "THSL": 4.75, "noise_cps": 42}, 2: {...}, ...}
dp.auto_thresholds                         # remembered per speed; apply_speed()/auto_speed() use them
dp.clear_auto_thresholds()                 # back to the preset values
```

For each speed it:

1. runs the DP5's own fast-threshold autoset and writes the result back as `THFA` (without the write-back the fast
   channel stayed nearly blind afterwards: 1 kcps instead of 46 kcps);
2. raises `THSL` from the preset value in 0.25 % steps until noise alone triggers fewer than `max_noise_cps` (default 50)
   counts per second on the slow channel. The DP5 has no slow-threshold autoset of its own.

A running acquisition is paused and restarted. The GUI's **Auto thresholds** button does `speeds="all"`
in the background (about 20 s).

**Measured on the bench (NI simulator, 2048 channels, beam "off" = simulator at 0 cps):**

| Speed | Noise at preset THSL | `auto_threshold` result |
|---|---|---|
| 1 | ~100 kcps at 1 %, 280–400 cps at 2 % | THSL 4.25–4.75 %, < 50 cps |
| 2 | 150 cps at 1 %, 30–120 cps at 1.25 % | THSL 1.5–2.0 %, < 40 cps |
| 3–5 | 0–16 cps at 1 % | unchanged (1 %) |

(Ranges are from repeated runs.) With all five speeds the whole call took 15–17 s.

Short peaking times average less noise, so they need a higher slow threshold. On a real SDD the numbers will differ;
run `auto_threshold()` once on the microscope. With 2048 channels over ~20 keV, 4.25–4.75 % is about 0.9 keV: speed 1 then
can't see C/N/O/F lines, which is the price of 1 µs shaping.

## 7. Automatic speed selection

```python
r = dp.auto_speed(max_dead=0.40)      # measure -> choose -> apply
print(r["speed"], r["measured"], r["predicted"], r["note"])
```

1. It measures ICR, OCR and dead time. If acquiring, it uses the running polls (in either mode). Otherwise it runs a 1 s test
   acquisition, which clears the spectrum and leaves the MCA disabled.
2. It recalibrates the factor f in τ = f·(2·TPEA+TFLA) from that one measurement.
3. It predicts each speed's dead time at the measured ICR and applies the **slowest speed with predicted dead ≤ `max_dead`**.
4. If even speed 1 is over the limit, it applies speed 1 and `note` says to reduce the beam current.

`measure_dead_time(seconds)` is available on its own, and `dp5.presets.predicted_dead_fraction()` /
`choose_speed()` let you plan without a device.

**Example, simulator at 49 kcps:** auto_speed chose speed 2 (predicted 29 %, measured 30 %). Speed 3 would give 64 % and
speed 5 93 %.

## 8. Fast peaking time `TPFA`, and the NI 6363 simulator

`TPFA` only decides whether and when a pulse is *seen*; the slow filter still measures its energy, so FWHM doesn't
depend on it. Measured with the simulator (Fe-55 at 50 kcps, 6 mV/keV, 0.5 µs step rise), speed 3:

| TPFA | autoset THFA | ICR | dead (model) | FWHM |
|---|---|---|---|---|
| 50 ns | 52.7 | 1.9 kcps, **blind** | – | 4.50 ch |
| 100 ns | 49.5 | 47.4 kcps | 65.0 % (62.6) | 4.55 ch |
| 200 ns | 37.3 | 47.9 kcps | 66.2 % (63.0) | 4.54 ch |
| 400 ns | 34.3 | 46.8 kcps | 63.5 % (62.1) | 4.52 ch |
| 800 ns | 30.3 | 46.0 kcps | 61.3 % (61.5) | 4.57 ch |

* **Longer TPFA, lower noise:** the noise-derived threshold falls from 53 to 30, so smaller (lower-energy) pulses
  trigger. At 400 ns any THFA from 20 to 80 counted every pulse; 15 and below triggered on noise.
* **Shorter TPFA, better pile-up rejection:** two pulses closer than about 2·TPFA look like one. At speed 1 (TPEA 1 µs)
  TPFA 100–400 ns gave 7–12 % dead time (model 10.5 %). TPFA 800 ns gave only 2 %, because pile-up went unrejected
  and was counted. Keep TPFA well below TPEA. The DP5 rejects TPFA 1600 and 3200 ns.
* **Rise time:** the fast filter must be longer than the signal's rise. A real SDD rises in about 100 ns and works at
  the presets' 50–400 ns. The simulator's 0.5 µs edge makes 50 ns blind, so for the simulator override speed 1 only:

```python
dp.speed_overrides = {1: {"TPFA": 100}}     # integer key = one speed; string keys apply to all speeds
dp.auto_threshold(speeds="all")             # beam off
```

**Other simulator effects:** drive it at about 6 mV/keV (5.9 keV → 35 mV); at 2 mV/keV the fast pulses barely clear
the fast noise. Speed 1's 0.2 µs flat top is shorter than the 0.5 µs rise, so its peaks sit about 4.5 % low
(ballistic deficit). Speeds 2–5 agree within 1.6 %.

## 9. Recommended tuning order for a real detector

1. **`AINP`:** set it to the preamp polarity.
2. **`RESL`:** look at the reset with the DP5 scope or an oscilloscope, then set `RESL` a bit longer than the reset plus its settling.
3. **`GAIN` and `MCAC`:** pick the energy range you need (e.g. 0–20 keV across 2048 channels) and calibrate on a known line.
4. **Thresholds:** with the **beam off**, run `auto_threshold(speeds="all")`. It sets THFA and THSL for every speed from the noise.
5. **Speed:** choose a speed for the job, or use `auto_speed()` with the beam on and set the beam current for 20–50 % dead time.
6. **Resolution check:** look at the FWHM of Mn Kα (Fe-55 or a Mn standard). A good SDD gives about 125–135 eV at long peaking.
