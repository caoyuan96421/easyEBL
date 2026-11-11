"""EDS data objects: energy calibration, spectra, spectrum maps and electron images.

Pure data + I/O (no hardware, no Qt).  Files:

* spectra: EMSA/MAS ``.msa`` (the vendor-neutral text format every EDS program
  reads) or ``.npz``;
* maps: ``.npz`` (cube, SE image, per-pixel times, calibration, metadata).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import datetime as _dt
import json

import numpy as np

__all__ = ["EnergyCalibration", "Spectrum", "SpectrumMap", "ElectronImage", "load_spectrum", "load_map"]


@dataclass(frozen=True)
class EnergyCalibration:
    """Channel → energy: ``E(k) = offset + gain·k + quad·k²`` (keV) at the *centre* of channel k."""

    offset_kev: float = 0.0
    gain_kev: float = 0.01
    quad_kev: float = 0.0
    name: str = ""

    def __post_init__(self):
        if not self.gain_kev > 0:
            raise ValueError("gain must be positive (keV per channel)")

    def energy(self, channel):
        k = np.asarray(channel, dtype=float)
        e = self.offset_kev + self.gain_kev * k + self.quad_kev * k * k
        return float(e) if np.ndim(e) == 0 else e

    def channel(self, energy_kev):
        """Fractional channel (centre = integer) of *energy_kev*."""
        e = np.asarray(energy_kev, dtype=float)
        if self.quad_kev == 0:
            k = (e - self.offset_kev) / self.gain_kev
        else:
            a, b, c = self.quad_kev, self.gain_kev, self.offset_kev - e
            k = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))) / (2 * a)
        return float(k) if np.ndim(k) == 0 else k

    def energies(self, n: int) -> np.ndarray:
        return self.energy(np.arange(n))

    def channel_width(self, channel=0.0):
        return self.gain_kev + 2 * self.quad_kev * np.asarray(channel, dtype=float)

    def to_dict(self) -> dict:
        return {"offset_kev": self.offset_kev, "gain_kev": self.gain_kev, "quad_kev": self.quad_kev,
                "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "EnergyCalibration":
        return cls(float(d.get("offset_kev", 0.0)), float(d.get("gain_kev", 0.01)), float(d.get("quad_kev", 0.0)),
                   str(d.get("name", "")))


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class Spectrum:
    """One EDS spectrum with its acquisition context."""

    counts: np.ndarray
    calibration: EnergyCalibration = field(default_factory=EnergyCalibration)
    real_time: float = 0.0
    live_time: float = 0.0
    fast_count: int = 0
    slow_count: int = 0
    beam_kv: float = 20.0
    takeoff_deg: float = 35.0
    title: str = ""
    timestamp: str = field(default_factory=_now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.counts = np.asarray(self.counts)
        if self.counts.ndim != 1:
            raise ValueError("spectrum counts must be 1-D")

    @property
    def n_channels(self) -> int:
        return int(self.counts.size)

    @property
    def energies(self) -> np.ndarray:
        return self.calibration.energies(self.n_channels)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    @property
    def icr(self) -> float:
        return self.fast_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def ocr(self) -> float:
        return self.slow_count / self.real_time if self.real_time > 0 else 0.0

    @property
    def dead_fraction(self) -> float:
        return max(0.0, 1.0 - self.slow_count / self.fast_count) if self.fast_count > 0 else 0.0

    def counts_in(self, e_lo: float, e_hi: float) -> int:
        k0 = int(np.ceil(self.calibration.channel(e_lo) - 0.5))
        k1 = int(np.floor(self.calibration.channel(e_hi) + 0.5))
        return int(self.counts[max(0, k0):max(0, min(self.n_channels, k1))].sum())

    def with_calibration(self, cal: EnergyCalibration) -> "Spectrum":
        return replace(self, calibration=cal)

    def __add__(self, other: "Spectrum") -> "Spectrum":
        if other.n_channels != self.n_channels:
            raise ValueError("spectra have different channel counts")
        return replace(self, counts=self.counts + other.counts, real_time=self.real_time + other.real_time,
                       live_time=self.live_time + other.live_time, fast_count=self.fast_count + other.fast_count,
                       slow_count=self.slow_count + other.slow_count, metadata=dict(self.metadata))

    # ------------------------------------------------------------- EMSA/MAS
    def to_msa(self) -> str:
        cal = self.calibration
        hdr = [
            ("FORMAT", "EMSA/MAS Spectral Data File"),
            ("VERSION", "1.0"),
            ("TITLE", self.title or "easyEBL EDS spectrum"),
            ("DATE", self.timestamp[:10]),
            ("TIME", self.timestamp[11:16] if len(self.timestamp) > 11 else ""),
            ("OWNER", "easyEBL"),
            ("NPOINTS", str(self.n_channels)),
            ("NCOLUMNS", "1"),
            ("XUNITS", "keV"),
            ("YUNITS", "counts"),
            ("DATATYPE", "Y"),
            ("XPERCHAN", f"{cal.gain_kev:.8g}"),
            ("OFFSET", f"{cal.offset_kev:.8g}"),
            ("SIGNALTYPE", "EDS"),
            ("BEAMKV  -kV", f"{self.beam_kv:g}"),
            ("ELEVANGLE-dg", f"{self.takeoff_deg:g}"),
            ("LIVETIME  -s", f"{self.live_time:.6g}"),
            ("REALTIME  -s", f"{self.real_time:.6g}"),
            ("#EASYEBL_QUAD", f"{cal.quad_kev:.8g}"),
            ("#EASYEBL_FAST", str(int(self.fast_count))),
            ("#EASYEBL_SLOW", str(int(self.slow_count))),
            ("#EASYEBL_META", json.dumps(_jsonable(self.metadata))),
        ]
        lines = []
        for k, v in hdr:
            key = k if k.startswith("#") else "#" + k
            lines.append(f"{key:<13}: {v}")
        lines.append("#SPECTRUM    : Spectral Data Starts Here")
        lines += [f"{int(c)}," for c in self.counts]
        lines.append("#ENDOFDATA   : ")
        return "\n".join(lines) + "\n"

    def save(self, path) -> None:
        path = str(path)
        if path.lower().endswith(".npz"):
            np.savez_compressed(path, counts=self.counts, header=json.dumps(self._header()))
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.to_msa())

    def _header(self) -> dict:
        return {"calibration": self.calibration.to_dict(), "real_time": self.real_time, "live_time": self.live_time,
                "fast_count": int(self.fast_count), "slow_count": int(self.slow_count), "beam_kv": self.beam_kv,
                "takeoff_deg": self.takeoff_deg, "title": self.title, "timestamp": self.timestamp,
                "metadata": _jsonable(self.metadata)}

    @classmethod
    def from_msa(cls, text: str) -> "Spectrum":
        hdr: dict = {}
        data: list = []
        in_data = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, val = line[1:].partition(":")
                key = key.strip().upper()
                if key == "SPECTRUM":
                    in_data = True
                    continue
                if key == "ENDOFDATA":
                    break
                hdr[key] = val.strip()
                continue
            if in_data:
                for tok in line.replace(";", ",").split(","):
                    tok = tok.strip()
                    if tok:
                        data.append(float(tok))
        ncol = int(float(hdr.get("NCOLUMNS", "1") or 1))
        if hdr.get("DATATYPE", "Y").upper().startswith("XY"):
            data = data[1::2]
        elif ncol > 1:
            pass
        counts = np.asarray(data)
        if np.all(np.mod(counts, 1) == 0):
            counts = counts.astype(np.int64)

        def num(*keys, default=0.0):
            for k in keys:
                for hk, hv in hdr.items():
                    if hk.replace(" ", "").split("-")[0] == k:
                        try:
                            return float(hv)
                        except ValueError:
                            pass
            return default

        xunits = hdr.get("XUNITS", "keV").lower()
        scale = 1e-3 if xunits.startswith("ev") else 1.0
        cal = EnergyCalibration(num("OFFSET") * scale, num("XPERCHAN", default=0.01) * scale,
                                num("EASYEBL_QUAD"))
        meta = {}
        if "EASYEBL_META" in hdr:
            try:
                meta = json.loads(hdr["EASYEBL_META"])
            except ValueError:
                meta = {}
        return cls(counts, cal, real_time=num("REALTIME"), live_time=num("LIVETIME"),
                   fast_count=int(num("EASYEBL_FAST")), slow_count=int(num("EASYEBL_SLOW")),
                   beam_kv=num("BEAMKV", default=20.0), takeoff_deg=num("ELEVANGLE", default=35.0),
                   title=hdr.get("TITLE", ""), timestamp=(hdr.get("DATE", "") + "T" + hdr.get("TIME", "")).strip("T"),
                   metadata=meta)


def load_spectrum(path) -> Spectrum:
    path = str(path)
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as f:
            h = json.loads(str(f["header"]))
            return Spectrum(f["counts"], EnergyCalibration.from_dict(h["calibration"]), h["real_time"],
                            h["live_time"], h["fast_count"], h["slow_count"], h["beam_kv"], h["takeoff_deg"],
                            h["title"], h["timestamp"], h["metadata"])
    with open(path, encoding="utf-8", errors="replace") as fh:
        return Spectrum.from_msa(fh.read())


def _jsonable(d):
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    return d


@dataclass
class ElectronImage:
    """SE (video) image acquired by the EDS module; ``region_um`` = (cx, cy, width, height)."""

    data: np.ndarray  # (ny, nx) volts
    region_um: tuple
    dwell_s: float
    timestamp: str = field(default_factory=_now)
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    def pixel_to_um(self, col: float, row: float) -> tuple:
        cx, cy, w, h = self.region_um
        ny, nx = self.data.shape
        return cx - w / 2 + (col + 0.5) * w / nx, cy + h / 2 - (row + 0.5) * h / ny

    def um_to_pixel(self, x_um: float, y_um: float) -> tuple:
        cx, cy, w, h = self.region_um
        ny, nx = self.data.shape
        return (x_um - (cx - w / 2)) * nx / w - 0.5, ((cy + h / 2) - y_um) * ny / h - 0.5


@dataclass
class SpectrumMap:
    """Spectrum image: one spectrum per pixel (step scan), with SE signal and times per pixel."""

    cube: np.ndarray  # (ny, nx, channels) counts
    calibration: EnergyCalibration
    region_um: tuple  # (cx, cy, width, height)
    dwell_s: float
    real_time: np.ndarray  # (ny, nx) s
    live_time: np.ndarray  # (ny, nx) s
    fast_count: np.ndarray  # (ny, nx)
    se: np.ndarray | None = None  # (ny, nx) mean video V during each pixel
    done: np.ndarray | None = None  # (ny, nx) bool – pixels acquired so far
    beam_kv: float = 20.0
    takeoff_deg: float = 35.0
    timestamp: str = field(default_factory=_now)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, nx: int, ny: int, channels: int, calibration: EnergyCalibration, region_um, dwell_s: float,
              dtype=np.uint16, **kw) -> "SpectrumMap":
        z = lambda dt=float: np.zeros((ny, nx), dt)  # noqa: E731
        return cls(np.zeros((ny, nx, channels), dtype), calibration, tuple(region_um), dwell_s, z(), z(),
                   z(np.int64), np.full((ny, nx), np.nan), np.zeros((ny, nx), bool), **kw)

    @property
    def shape(self) -> tuple:
        return self.cube.shape[:2]

    @property
    def n_channels(self) -> int:
        return int(self.cube.shape[2])

    @property
    def energies(self) -> np.ndarray:
        return self.calibration.energies(self.n_channels)

    def sum_spectrum(self, mask=None) -> Spectrum:
        m = self.done if mask is None else (np.asarray(mask, bool) & self.done)
        counts = self.cube[m].sum(axis=0, dtype=np.int64) if m.any() else np.zeros(self.n_channels, np.int64)
        fast = int(self.fast_count[m].sum())
        rt, lt = float(self.real_time[m].sum()), float(self.live_time[m].sum())
        slow = int(counts.sum())
        return Spectrum(counts, self.calibration, rt, lt, fast, slow, self.beam_kv, self.takeoff_deg,
                        title="map sum" if mask is None else "map region", metadata={"pixels": int(m.sum())})

    def pixel_spectrum(self, row: int, col: int) -> Spectrum:
        c = self.cube[row, col].astype(np.int64)
        return Spectrum(c, self.calibration, float(self.real_time[row, col]), float(self.live_time[row, col]),
                        int(self.fast_count[row, col]), int(c.sum()), self.beam_kv, self.takeoff_deg,
                        title=f"pixel ({col}, {row})")

    def window_image(self, e_lo: float, e_hi: float) -> np.ndarray:
        """Counts per pixel in an energy window (keV)."""
        k0 = max(0, int(np.ceil(self.calibration.channel(e_lo) - 0.5)))
        k1 = min(self.n_channels, int(np.floor(self.calibration.channel(e_hi) + 0.5)))
        if k1 <= k0:
            return np.zeros(self.shape)
        return self.cube[:, :, k0:k1].sum(axis=2, dtype=np.int64).astype(float)

    def pixel_to_um(self, col: float, row: float) -> tuple:
        cx, cy, w, h = self.region_um
        ny, nx = self.shape
        return cx - w / 2 + (col + 0.5) * w / nx, cy + h / 2 - (row + 0.5) * h / ny

    def save(self, path) -> None:
        hdr = {"calibration": self.calibration.to_dict(), "region_um": list(self.region_um), "dwell_s": self.dwell_s,
               "beam_kv": self.beam_kv, "takeoff_deg": self.takeoff_deg, "timestamp": self.timestamp,
               "metadata": _jsonable(self.metadata)}
        np.savez_compressed(str(path), cube=self.cube, real_time=self.real_time, live_time=self.live_time,
                            fast_count=self.fast_count, se=self.se if self.se is not None else np.zeros(0),
                            done=self.done if self.done is not None else np.ones(self.shape, bool),
                            header=json.dumps(hdr))


def load_map(path) -> SpectrumMap:
    with np.load(str(path), allow_pickle=False) as f:
        h = json.loads(str(f["header"]))
        se = f["se"] if f["se"].size else None
        return SpectrumMap(f["cube"], EnergyCalibration.from_dict(h["calibration"]), tuple(h["region_um"]),
                           h["dwell_s"], f["real_time"], f["live_time"], f["fast_count"], se, f["done"],
                           h["beam_kv"], h["takeoff_deg"], h["timestamp"], h["metadata"])
