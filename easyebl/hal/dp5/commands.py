"""
dp5.commands - catalogue of DP5 ASCII configuration commands and helpers.

The DP5 is configured with text commands ``CMD=VALUE;`` sent in a 0x20 0x02 packet and read
back with ``CMD=?;`` in a 0x20 0x03 packet. This module lists every command found in the Amptek
SDK, with its description, units and limits as given by the SDK (``vcDP5/SCommon.cpp``,
``ConsoleHelper.cpp``, ``AsciiCmdUtilities.cpp``, the DP5Cfg property pages) and the values
used in Amptek's ``SDD Default DP5.txt``.

The DP5 itself validates every value; a rejected command comes back as ACK BAD_PARAMETER with the
offending text, so the limits here are documentation, not enforced by the driver.

Helpers implement the SDK's rules for building configuration strings:
  * device-type filtering (``RemoveCmdByDeviceType``),
  * only one gain style (``GAIN`` or ``GAIA``+``GAIF``),
  * 512-byte packet limit with value shortening and splitting on ``;`` (``SendConfigFileToDpp``),
  * the full-readback command list (``CreateFullReadBackCmd``),
  * parsing of Amptek configuration files (``[DP5 Configuration File]`` section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .protocol import MAX_CONFIG_DATA


@dataclass(frozen=True)
class Command:
    name: str                  # 4-letter mnemonic
    description: str
    unit: str = ""
    limits: str = ""           # as documented by the SDK
    options: tuple = ()        # known enumerated values (not necessarily exhaustive)
    group: str = ""            # shaping / gain / input / mca / preset / aux / scope / mcs / power / misc
    sdd: bool = True           # relevant for an SDD with a reset-type preamp (used by the GUI)
    sdd_default: str = ""      # value in Amptek "SDD Default DP5.txt"


_AUX = ("ICR", "PILEUP", "MCSTB", "ONESH", "DETRES", "MCAEN", "PEAKH", "SCA8",
        "RTDOS", "RTDREJ", "VETO", "LIVE", "STREAM")

COMMANDS: dict[str, Command] = {c.name: c for c in [
    # ---- general / clock ----
    Command("RESC", "Reset configuration to defaults (send first in a full configuration)", "", "",
            ("YES",), "misc", True, "YES"),
    Command("CLCK", "FPGA clock 20 MHz / 80 MHz / AUTO", "MHz", "", ("20", "80", "AUTO"), "shaping", True, "80"),
    # ---- shaping ----
    Command("TPEA", "Peaking time (slow channel)", "us",
            "0.8-102.4 (20 MHz); 0.2-25.6 (80 MHz); 0.2-102.4 (AUTO)", (), "shaping", True, "9.600"),
    Command("TFLA", "Flat-top width (slow channel)", "us", "0-51.2", (), "shaping", True, "0.200"),
    Command("TPFA", "Fast-channel peaking time", "ns", "discrete values, e.g. 50/100/200/400/800/1600/3200",
            (), "shaping", True, "100"),
    Command("CUSP", "Non-trapezoidal (cusp-like) shaping", "%", "-99 to +99", (), "shaping", True, "0"),
    Command("PDMD", "Peak-detect mode", "", "", ("NORM", "MIN"), "shaping", True, "NORM"),
    Command("THSL", "Slow threshold (LLD of the slow channel)", "%", "0-24.9 % of full scale", (),
            "shaping", True, "0.976"),
    Command("THFA", "Fast threshold", "", "0-255.93", (), "shaping", True, "41.43"),
    Command("PURE", "Pile-up rejection on/off", "", "", ("ON", "OFF"), "shaping", True, "ON"),
    Command("RTDE", "Rise-time discrimination on/off", "", "", ("ON", "OFF"), "shaping", True, "OFF"),
    Command("RTDS", "RTD sensitivity", "%", "0-398", (), "shaping", True, "0"),
    Command("RTDT", "RTD threshold", "%", "0-49.9", (), "shaping", True, "0.00"),
    Command("RTDD", "Custom RTD one-shot delay", "", "", (), "shaping", False),
    Command("RTDW", "Custom RTD one-shot width", "", "", (), "shaping", False),
    Command("BLRM", "Baseline restorer mode", "", "", (), "shaping", True, "1"),
    Command("BLRD", "Baseline restorer down correction", "", "0-3", (), "shaping", True, "3"),
    Command("BLRU", "Baseline restorer up correction", "", "0-3", (), "shaping", True, "0"),
    Command("SCTC", "Scintillator time constant (DP5G/TB5 only)", "ns", "", (), "shaping", False),
    # ---- gain / input ----
    Command("GAIN", "Total gain (analog x fine). Do not send together with GAIA/GAIF", "x",
            "0.75-150 (DP5), 0.75-250 (DP5 Rev Dx)", (), "gain", True, "64.998"),
    Command("GAIA", "Coarse (analog) gain index, 1-based (see dp5.gain tables)", "", "1-16 (DP5), 1-24 (Rev Dx)",
            (), "gain", True, "20"),
    Command("GAIF", "Fine gain", "x", "0.750-1.250", (), "gain", True, "0.9313"),
    Command("AINP", "Analog input polarity (sign of the preamp steps)", "", "", ("POS", "NEG"), "input", True, "POS"),
    Command("INOF", "Input offset", "mV", "-2047 to +2047, DEF or AUTO", ("DEF", "AUTO"), "input", True, "DEF"),
    Command("RESL", "Detector reset lockout (blanking after a preamp reset)", "us", "0-65535", (),
            "input", True, "204"),
    Command("PAPZ", "Preamp pole-zero (PX5, DP5 Rev Dx-K/L only)", "us", "34.5-8773", (), "gain", False),
    Command("PAPS", "Preamp supply 8.5 V / 5 V / OFF", "V", "", ("8.5", "5", "OFF", "ON"), "power", True, "ON"),
    # ---- MCA ----
    Command("MCAS", "MCA source", "", "", ("NORM", "MCS", "FAST", "PUR", "RTD"), "mca", True, "NORM"),
    Command("MCAC", "MCA/MCS channels", "", "256/512/1024/2048/4096/8192",
            ("256", "512", "1024", "2048", "4096", "8192"), "mca", True, "2048"),
    Command("MCAE", "MCA/MCS enabled at power-up", "", "", ("ON", "OFF"), "mca", True, "ON"),
    Command("SOFF", "Spectrum offset (+0.25 = +1/4 channel)", "ch", "+/- channels or OFF", ("OFF",), "mca", True, "OFF"),
    Command("TLLD", "Low-level discriminator channel (for preset counts)", "ch", "1-8192 or OFF", ("OFF",),
            "mca", True, "OFF"),
    Command("GATE", "Gate input control", "", "", ("OFF", "HIGH", "LOW"), "mca", True, "OFF"),
    # ---- presets ----
    Command("PRET", "Preset accumulation time", "s", "0-1677721.5 or OFF", ("OFF",), "preset", True, "OFF"),
    Command("PRER", "Preset real time", "s", "0-4294967.29 or OFF", ("OFF",), "preset", True, "OFF"),
    Command("PREC", "Preset counts (within PRCL..PRCH)", "counts", "0-4294967295 or OFF", ("OFF",),
            "preset", True, "OFF"),
    Command("PREL", "Preset live time (MCA8000D only)", "s", "", (), "preset", False),
    Command("PRCL", "Preset-count ROI low channel", "ch", "1-8192", (), "preset", True, "1"),
    Command("PRCH", "Preset-count ROI high channel", "ch", "1-8192", (), "preset", True, "8191"),
    # ---- MCS ----
    Command("MCSL", "MCS low channel (pulse-height window)", "ch", "1-8192", (), "mcs", True, "1"),
    Command("MCSH", "MCS high channel (pulse-height window)", "ch", "1-8192", (), "mcs", True, "8191"),
    Command("MCST", "MCS timebase per channel", "s", "multiples of 0.01, up to 655.35", (), "mcs", True, "0.00"),
    # ---- AUX / GP counter / DAC ----
    Command("AUO1", "AUX_OUT1 selection", "", "", _AUX, "aux", False, "SCA8"),
    Command("AUO2", "AUX_OUT2 selection", "", "", _AUX, "aux", False, "ICR"),
    Command("CON1", "Connector 1 function (not DP5)", "", "", (), "aux", False),
    Command("CON2", "Connector 2 function (not DP5)", "", "", (), "aux", False),
    Command("DACO", "DAC test-output signal select", "", "e.g. SHAPED, FAST, INPUT", ("SHAPED", "FAST", "INPUT"),
            "aux", True, "SHAPED"),
    Command("DACF", "DAC output offset", "mV", "-500 to +499", (), "aux", True, "50"),
    Command("GPED", "General-purpose counter edge", "", "", ("RISING", "FALLING"), "aux", False, "RISING"),
    Command("GPIN", "General-purpose counter input", "", "", (), "aux", False, "AUX1"),
    Command("GPME", "GP counter uses MCA_EN", "", "", ("ON", "OFF"), "aux", False, "ON"),
    Command("GPGA", "GP counter uses GATE", "", "", ("ON", "OFF"), "aux", False, "ON"),
    Command("GPMC", "GP counter cleared with MCA counters", "", "", ("ON", "OFF"), "aux", False, "ON"),
    # ---- SCA ----
    Command("SCAI", "SCA index (selects which SCA SCAL/SCAH/SCAO refer to)", "", "1-16", (), "sca", False),
    Command("SCAL", "SCAx low threshold", "ch", "", (), "sca", False),
    Command("SCAH", "SCAx high threshold", "ch", "", (), "sca", False),
    Command("SCAO", "SCAx output (SCA1-8 only)", "", "", (), "sca", False),
    Command("SCAW", "SCA pulse width", "", "", (), "sca", False),
    # ---- digital scope ----
    Command("SCOE", "Scope trigger edge", "", "", ("RISING", "FALLING"), "scope", True, "FALLING"),
    Command("SCOT", "Scope trigger position", "%", "", (), "scope", True, "12"),
    Command("SCOG", "Digital scope gain", "", "", (), "scope", True, "1"),
    # ---- power / detector (PC5 / PX5) ----
    Command("HVSE", "Detector HV setting (PC5 or PX5 only)", "V", "0 to +/-1500", (), "power", False, "-110"),
    Command("TECS", "TEC set point (PC5 or PX5 only)", "K", "0-299", (), "power", False, "220"),
    Command("BOOT", "Turn PC5 supplies on at power-up", "", "", ("ON", "OFF"), "power", False, "ON"),
    Command("VOLU", "Speaker on/off (PX5 only)", "", "", ("ON", "OFF"), "power", False),
    # ---- test / misc ----
    Command("TPMO", "Test pulser mode", "", "", ("OFF", "ON"), "misc", True, "OFF"),
    Command("ACKE", "ACK / don't ACK packets with errors", "", "", ("ON", "OFF"), "misc", False),
]}

# Order used by the SDK full readback (CreateFullReadBackCmd).
FULL_READBACK_ORDER = [
    "RESC", "CLCK", "TPEA", "GAIF", "GAIN", "RESL", "TFLA", "TPFA", "PURE", "SCTC", "RTDE", "MCAS",
    "MCAC", "SOFF", "AINP", "INOF", "GAIA", "CUSP", "PDMD", "THSL", "TLLD", "THFA", "DACO", "DACF",
    "RTDS", "RTDT", "BLRM", "BLRD", "BLRU", "GATE", "AUO1", "PRET", "PRER", "PREC", "PRCL", "PRCH",
    "HVSE", "TECS", "PAPZ", "PAPS", "SCOE", "SCOT", "SCOG", "MCSL", "MCSH", "MCST", "AUO2", "TPMO",
    "GPED", "GPIN", "GPME", "GPGA", "GPMC", "MCAE", "VOLU", "CON1", "CON2", "BOOT",
]

# Device types (status byte 39)
DPP_DP5, DPP_PX5, DPP_DP5G, DPP_MCA8000D, DPP_TB5, DPP_DP5X = range(6)


def allowed_commands(device_type: int = DPP_DP5, pc5_present: bool = False, has_papz: bool = False) -> set[str]:
    """Commands valid for this device, following ``RemoveCmdByDeviceType`` in the SDK."""
    allowed = set(COMMANDS)
    is_hvse = (device_type != DPP_PX5 and pc5_present) or device_type == DPP_PX5
    is_tecs = (device_type == DPP_DP5 and pc5_present) or device_type in (DPP_PX5, DPP_DP5X)
    rules = {
        "HVSE": is_hvse,
        "TECS": is_tecs,
        # SDK keeps PAPS for every DP5, but FW 6.10 answers PC5_NOT_PRESENT without a PC5
        "PAPS": device_type not in (DPP_DP5G, DPP_TB5) and (device_type != DPP_DP5 or pc5_present),
        "VOLU": device_type == DPP_PX5,
        "CON1": device_type not in (DPP_DP5, DPP_DP5X),
        "CON2": device_type not in (DPP_DP5, DPP_DP5X),
        "INOF": device_type not in (DPP_DP5G, DPP_TB5),
        "SCTC": device_type in (DPP_DP5G, DPP_TB5),
        "BOOT": device_type in (DPP_DP5, DPP_DP5X),
        "GATE": device_type in (DPP_DP5, DPP_DP5X),
        "PAPZ": device_type == DPP_PX5 or has_papz,
        "PREL": device_type == DPP_MCA8000D,
    }
    for k, ok in rules.items():
        if not ok:
            allowed.discard(k)
    return allowed


def full_readback_list(device_type=DPP_DP5, pc5_present=False, has_papz=False) -> list[str]:
    allowed = allowed_commands(device_type, pc5_present, has_papz)
    return [c for c in FULL_READBACK_ORDER if c in allowed]


# --------------------------------------------------------------------------------------
# string helpers
# --------------------------------------------------------------------------------------
def format_value(v) -> str:
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).strip()


def build_config_string(settings) -> str:
    """dict / iterable of (name, value) / "A=1;B=2;" -> "A=1;B=2;" (upper case)."""
    if isinstance(settings, str):
        s = settings.strip()
        if s and not s.endswith(";"):
            s += ";"
        return s.upper()
    items = settings.items() if hasattr(settings, "items") else settings
    return "".join(f"{str(k).strip().upper()}={format_value(v).upper()};" for k, v in items)


def parse_config_string(s: str) -> dict[str, str]:
    """"A=1;B=OFF;" -> {"A": "1", "B": "OFF"} (order preserved)."""
    out: dict[str, str] = {}
    for part in s.replace("\r", "").replace("\n", "").split(";"):
        part = part.strip().strip("\x00")
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out


def remove_commands(cfg: dict[str, str], names: Iterable[str]) -> dict[str, str]:
    names = {n.upper() for n in names}
    return {k: v for k, v in cfg.items() if k not in names}


def apply_gain_style(cfg: dict[str, str], coarse_fine: bool | None = None) -> dict[str, str]:
    """Keep only one gain style. ``None`` = keep GAIN if present, else GAIA/GAIF."""
    if coarse_fine is None:
        coarse_fine = "GAIN" not in cfg
    return remove_commands(cfg, ("GAIN",) if coarse_fine else ("GAIA", "GAIF"))


_SHORTEN = (("US;", ";"), ("OFF;", "OF;"), ("RISING;", "RI;"), ("FALLING;", "FA;"))


def shorten(s: str) -> str:
    """Amptek's value abbreviations used when a configuration exceeds 512 bytes."""
    for a, b in _SHORTEN:
        s = s.replace(a, b)
    return s


def split_config(s: str, limit: int = MAX_CONFIG_DATA) -> list[str]:
    """Split a configuration string into <= ``limit``-byte chunks at ';' boundaries."""
    if len(s) > limit:
        s = shorten(s)
    chunks, cur = [], ""
    for cmd in [c + ";" for c in s.split(";") if c.strip()]:
        if len(cmd) > limit:
            raise ValueError(f"single command longer than {limit} bytes: {cmd[:40]}...")
        if len(cur) + len(cmd) > limit:
            chunks.append(cur)
            cur = ""
        cur += cmd
    if cur:
        chunks.append(cur)
    return chunks


def load_config_file(path: str) -> dict[str, str]:
    """Read an Amptek configuration file (``[DP5 Configuration File]`` section, or bare lines)."""
    cfg: dict[str, str] = {}
    section = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                section = line.strip("[]").strip().lower()
                continue
            if section not in (None, "dp5 configuration file"):
                continue
            cfg.update(parse_config_string(line))
    return cfg


def save_config_file(path: str, cfg: dict[str, str]):
    with open(path, "w") as f:
        f.write("[DP5 Configuration File]\n")
        for k, v in cfg.items():
            f.write(f"{k}={v};\n")
