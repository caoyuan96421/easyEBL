"""
dp5.presets - SDD speed presets 1 (fastest) .. 5 (best resolution), derived from the DP5 SDK.

=====  ======  ======  ======  ======  =====  ====  ================================
speed  TPEA    TFLA    TPFA    THSL    THFA   BLRU  origin
=====  ======  ======  ======  ======  =====  ====  ================================
1      1.0us   0.2us   50ns    2.0%    30     2     SDK "SDD High Rate Default DP5"
2      3.2us   0.2us   100ns   1.25%   30     1     geometric interpolation 1-3
3      9.6us   0.2us   100ns   1.0%    30     0     SDK "SDD Default DP5"
4      16.0us  0.2us   200ns   1.0%    30     1     geometric interpolation 3-5
5      25.6us  0.2us   400ns   1.0%    30     2     SDK "SDD Low Energy Default DP5"
=====  ======  ======  ======  ======  =====  ====  ================================

All speeds: CLCK 80 MHz, PURE ON, RTDE OFF, PDMD NORM, BLRM 1, BLRD 3. CUSP is not set.
THFA is 30 for all speeds; THSL is 1 % except for the short peaking times (speed 1: 2 %,
speed 2: 1.25 %), which count noise at 1 % (bench, beam off: ~100 kcps at speed 1, 150 cps at
speed 2). These are starting values - run ``DP5.auto_threshold()`` with the beam off to set both
thresholds from the actual noise; the result is remembered per speed.

Trade-off: a longer peaking time averages away more noise (better FWHM) but each pulse occupies
~2*TPEA+TFLA, so dead time rises with count rate. Peak output rates (at ~63 % dead):
~156 / 52 / 18 / 11 / 7 kcps for speeds 1..5 (with the measured 1.07 dead-time factor). For
30 % dead time the input rate should be about
150 / 50 / 17 / 10 / 6.5 kcps.

Only the shaping keys (SHAPING_KEYS) are written: gain, polarity, input offset, reset lockout,
channel count etc. are never changed, so switching speed does not move the peaks.

``ALL_SDK_SDD`` keeps verbatim copies of all nine SDK SDD / FastSDD / SuperSDD files for reference.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Raw SDK files (DP5 "[DP5 Configuration File]" sections). Values copied verbatim.
# --------------------------------------------------------------------------------------
_COMMON = {
    "RESC": "YES", "CLCK": "80", "PURE": "ON", "RTDE": "OFF", "MCAS": "NORM", "MCAC": "2048",
    "SOFF": "OFF", "AINP": "POS", "INOF": "DEF", "PDMD": "NORM", "TLLD": "OFF", "DACO": "SHAPED",
    "DACF": "50", "RTDS": "0", "RTDT": "0.00", "BLRM": "1", "BLRD": "3", "GATE": "OFF",
    "AUO1": "SCA8", "PRET": "OFF", "PRER": "OFF", "PREC": "OFF", "PRCL": "1", "PRCH": "8191",
    "HVSE": "-110", "TECS": "220", "PAPS": "ON", "SCOE": "FALLING", "SCOT": "12", "SCOG": "1",
    "MCSL": "1", "MCSH": "8191", "MCST": "0.00", "AUO2": "ICR", "TPMO": "OFF", "GPED": "RISING",
    "GPIN": "AUX1", "GPME": "ON", "GPGA": "ON", "GPMC": "ON", "MCAE": "ON", "VOLU": "OFF",
    "CON1": "DAC", "CON2": "AUXOUT2", "BOOT": "ON",
}


def _sdk(**diff) -> dict[str, str]:
    """SDK file = common values + the per-file values, in the SDK's command order."""
    order = ["RESC", "CLCK", "TPEA", "GAIF", "GAIN", "RESL", "TFLA", "TPFA", "PURE", "RTDE", "MCAS",
             "MCAC", "SOFF", "AINP", "INOF", "GAIA", "CUSP", "PDMD", "THSL", "TLLD", "THFA", "DACO",
             "DACF", "RTDS", "RTDT", "BLRM", "BLRD", "BLRU", "GATE", "AUO1", "PRET", "PRER", "PREC",
             "PRCL", "PRCH", "HVSE", "TECS", "PAPS", "SCOE", "SCOT", "SCOG", "MCSL", "MCSH", "MCST",
             "AUO2", "TPMO", "GPED", "GPIN", "GPME", "GPGA", "GPMC", "MCAE", "VOLU", "CON1", "CON2",
             "BOOT"]
    merged = {**_COMMON, **diff}
    return {k: merged[k] for k in order if k in merged}


ALL_SDK_SDD: dict[str, dict[str, str]] = {
    "SDD Default DP5": _sdk(TPEA="9.600", GAIF="0.9313", GAIN="64.998", RESL="204", TFLA="0.200",
                            TPFA="100", GAIA="20", CUSP="0", THSL="0.976", THFA="41.43", BLRU="0"),
    "SDD High Rate Default DP5": _sdk(TPEA="1.000", GAIF="0.9313", GAIN="65.000", RESL="204", TFLA="0.200",
                                      TPFA="50", GAIA="20", CUSP="0", THSL="1.855", THFA="41.43", BLRU="2"),
    "SDD Low Energy Default DP5": _sdk(TPEA="25.600", GAIF="1.0306", GAIN="139.997", RESL="204", TFLA="0.200",
                                       TPFA="400", GAIA="23", CUSP="50", THSL="1.464", THFA="26.75", BLRU="2"),
    "FastSDD Default DP5": _sdk(TPEA="4.000", GAIF="1.0713", GAIN="16.000", RESL="100", TFLA="0.200",
                                TPFA="100", GAIA="11", CUSP="0", THSL="0.683", THFA="12.12", BLRU="0", VOLU="ON"),
    "FastSDD High Rate Default DP5": _sdk(TPEA="1.000", GAIF="1.0713", GAIN="16.000", RESL="50", TFLA="0.300",
                                          TPFA="50", GAIA="11", CUSP="0", THSL="0.878", THFA="12.75", BLRU="0",
                                          VOLU="ON"),
    "FastSDD Low Energy Default DP5": _sdk(TPEA="8.000", GAIF="1.0533", GAIN="32.0", RESL="204", TFLA="0.200",
                                           TPFA="100", GAIA="14", CUSP="0", THSL="9.179", THFA="25.06", BLRU="0",
                                           VOLU="ON"),
}
# SuperSDD files are identical to the SDD ones in every DP5 command.
ALL_SDK_SDD["SuperSDD Default DP5"] = dict(ALL_SDK_SDD["SDD Default DP5"])
ALL_SDK_SDD["SuperSDD High Rate Default DP5"] = dict(ALL_SDK_SDD["SDD High Rate Default DP5"])
ALL_SDK_SDD["SuperSDD Low Energy Default DP5"] = dict(ALL_SDK_SDD["SDD Low Energy Default DP5"])

# --------------------------------------------------------------------------------------
# Speed presets 1 (fastest) .. 5 (best resolution)
# --------------------------------------------------------------------------------------
# Speeds 1, 3 and 5 take the shaping of Amptek's "SDD High Rate", "SDD Default" and
# "SDD Low Energy" files; 2 and 4 are geometric interpolations (peaking time is a log-scale
# parameter: sqrt(1.0*9.6) = 3.1 -> 3.2 us, sqrt(9.6*25.6) = 15.7 -> 16.0 us). TPFA only takes
# discrete values, so it steps 50/100/100/200/400 ns. BLRU (0-3) is interpolated to 1 for 2 and 4.
# Thresholds: THFA = 30 (fast-channel units, 0-255.93) for all speeds; THSL = 1.0 % of full
# scale, raised for speeds 1 and 2 whose short peaking time lets noise through at 1 % (measured,
# beam off, NI simulator: speed 1 counted ~100 kcps of noise at 1 %, ~400 cps at 2 %; speed 2
# 150 cps at 1 %). DP5.auto_threshold() refines both from the real noise.
# CUSP is not set (left at whatever the DP5 has; this FPGA cannot read it back).
SPEED_THSL = {1: "2.000", 2: "1.250", 3: "1.000", 4: "1.000", 5: "1.000"}
SPEED_THFA = "30.00"


def _speed(tpea, tpfa, blru, source, thsl):
    return {
        "source": source,
        "config": {"CLCK": "80", "TPEA": f"{tpea:.3f}", "TFLA": "0.200", "TPFA": str(tpfa), "PURE": "ON",
                   "RTDE": "OFF", "PDMD": "NORM", "THSL": thsl, "THFA": SPEED_THFA,
                   "BLRM": "1", "BLRD": "3", "BLRU": str(blru)},
    }


SPEEDS: dict[int, dict] = {
    1: _speed(1.0, 50, 2, "SDD High Rate Default DP5", SPEED_THSL[1]),
    2: _speed(3.2, 100, 1, "interpolated 1-3", SPEED_THSL[2]),
    3: _speed(9.6, 100, 0, "SDD Default DP5", SPEED_THSL[3]),
    4: _speed(16.0, 200, 1, "interpolated 3-5", SPEED_THSL[4]),
    5: _speed(25.6, 400, 2, "SDD Low Energy Default DP5", SPEED_THSL[5]),
}
SPEED_MIN, SPEED_MAX = 1, 5

# Old names kept as aliases
ALIASES = {"fast": 1, "high_rate": 1, "highrate": 1, "default": 3, "standard": 3, "normal": 3,
           "high_resolution": 5, "hires": 5, "high_res": 5, "low_energy": 5, "resolution": 5}

# Pulse-processing settings a speed preset writes. Detector-specific settings (gain, polarity,
# offset, reset lockout, channels, presets, HV/TEC, AUX/scope) are never touched.
SHAPING_KEYS = ("CLCK", "TPEA", "TFLA", "TPFA", "PURE", "RTDE", "PDMD", "THSL", "THFA",
                "BLRM", "BLRD", "BLRU")

# Effective dead time per counted pulse ~ DEADTIME_FACTOR * (2*TPEA + TFLA); 1.07 was measured on
# DP5 S/N 15791 (TPEA 4 us: ICR 49 kcps, OCR 32 kcps). auto_speed() recalibrates it from the
# current measurement, so this is only the starting guess.
DEADTIME_FACTOR = 1.07


def resolve(speed) -> int:
    """1..5, or one of the old names / aliases -> speed number."""
    if isinstance(speed, str):
        key = speed.strip().lower().replace("-", "_").replace(" ", "_")
        if key.isdigit():
            speed = int(key)
        elif key in ALIASES:
            return ALIASES[key]
        else:
            raise KeyError(f"unknown speed {speed!r}; use 1-5 (1 = fastest) or {sorted(ALIASES)}")
    speed = int(speed)
    if speed not in SPEEDS:
        raise KeyError(f"speed must be 1-5 (1 = fastest), got {speed}")
    return speed


def speed_config(speed) -> dict[str, str]:
    return dict(SPEEDS[resolve(speed)]["config"])


def pulse_width_us(speed) -> float:
    """Trapezoid length 2*TPEA + TFLA (us) for a speed."""
    c = SPEEDS[resolve(speed)]["config"]
    return 2 * float(c["TPEA"]) + float(c["TFLA"])


def max_output_rate(speed, factor: float = DEADTIME_FACTOR) -> float:
    """Peak of the paralyzable throughput curve, 1/(e*tau) in counts/s."""
    import math
    return 1.0 / (math.e * factor * pulse_width_us(speed) * 1e-6)


def predicted_dead_fraction(speed, input_rate_cps: float, factor: float = DEADTIME_FACTOR) -> float:
    """Dead fraction 1 - exp(-ICR * tau) for a paralyzable system with tau = factor*(2*TPEA+TFLA)."""
    import math
    return 1.0 - math.exp(-input_rate_cps * factor * pulse_width_us(speed) * 1e-6)


def choose_speed(input_rate_cps: float, max_dead: float = 0.40, factor: float = DEADTIME_FACTOR) -> int:
    """Slowest (best-resolution) speed whose predicted dead time is <= max_dead; 1 if none is."""
    for s in sorted(SPEEDS, reverse=True):
        if predicted_dead_fraction(s, input_rate_cps, factor) <= max_dead:
            return s
    return SPEED_MIN


def describe() -> str:
    lines = []
    for k, p in SPEEDS.items():
        c = p["config"]
        lines.append(f"speed {k}: TPEA {float(c['TPEA']):5.1f} us  TFLA {c['TFLA']} us  TPFA {c['TPFA']:>3s} ns  "
                     f"THSL {c['THSL']}%  THFA {c['THFA']}  BLRU {c['BLRU']}  "
                     f"max out ~{max_output_rate(k) / 1e3:5.0f} kcps  ({p['source']})")
    return "\n".join(lines)
