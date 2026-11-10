"""Saving SEM images without extra dependencies (numpy + zlib only).

* ``.tif``  – 16-bit grayscale, linear in volts over the AI input range
  (raw data, suitable for measurements; the scale is in the JSON sidecar)
* ``.png``  – 8-bit grayscale with the display contrast/brightness applied
* ``.npy``  – float32 volts per channel
* ``.json`` – metadata (pixel size, field, dwell, …)
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import numpy as np

__all__ = ["write_png", "write_tiff16", "to_uint8", "auto_levels", "save_frame"]


def auto_levels(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    """Robust (low, high) display levels from percentiles, ignoring NaN."""
    a = np.asarray(img, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    if a.size > 400_000:  # subsample: percentiles of a large image are stable
        a = a[:: a.size // 200_000]
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if not hi > lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def to_uint8(img: np.ndarray, levels: tuple[float, float]) -> np.ndarray:
    lo, hi = levels
    a = np.asarray(img, dtype=np.float32)
    scale = 255.0 / max(hi - lo, 1e-12)
    out = np.nan_to_num((a - lo) * scale, nan=0.0)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def write_png(path, img: np.ndarray) -> None:
    """Grayscale PNG (uint8 or uint16)."""
    a = np.ascontiguousarray(img)
    if a.dtype == np.uint8:
        depth = 8
        raw = a
    elif a.dtype == np.uint16:
        depth = 16
        raw = a.astype(">u2")
    else:
        raise TypeError("write_png needs uint8 or uint16 data")
    h, w = a.shape
    rows = raw.reshape(h, -1).view(np.uint8).reshape(h, -1)
    data = np.hstack([np.zeros((h, 1), np.uint8), rows]).tobytes()

    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(data, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def write_tiff16(path, img: np.ndarray, description: str = "") -> None:
    """Uncompressed baseline TIFF, 16-bit grayscale, little-endian."""
    a = np.ascontiguousarray(img, dtype="<u2")
    h, w = a.shape
    desc = (description.encode("ascii", "replace") + b"\0") if description else b""
    entries = []
    # header (8) + IFD (2 + n*12 + 4) + description + image
    tags = [
        (256, 4, 1, w),  # ImageWidth
        (257, 4, 1, h),  # ImageLength
        (258, 3, 1, 16),  # BitsPerSample
        (259, 3, 1, 1),  # Compression: none
        (262, 3, 1, 1),  # Photometric: BlackIsZero
        (273, 4, 1, 0),  # StripOffsets (patched)
        (277, 3, 1, 1),  # SamplesPerPixel
        (278, 4, 1, h),  # RowsPerStrip
        (279, 4, 1, a.nbytes),  # StripByteCounts
    ]
    if desc:
        tags.insert(5, (270, 2, len(desc), 0))  # ImageDescription (patched)
    n = len(tags)
    ifd_size = 2 + n * 12 + 4
    desc_off = 8 + ifd_size
    img_off = desc_off + len(desc)
    img_off += img_off % 2
    for tag, typ, count, val in tags:
        if tag == 273:
            val = img_off
        if tag == 270:
            val = desc_off
        if typ == 3 and count == 1:
            entries.append(struct.pack("<HHIHH", tag, typ, count, val, 0))
        else:
            entries.append(struct.pack("<HHII", tag, typ, count, val))
    out = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", n) + b"".join(entries) + struct.pack("<I", 0)
    out += desc
    out += b"\0" * (img_off - len(out))
    Path(path).write_bytes(out + a.tobytes())


def save_frame(frame, path, levels: tuple | None = None, ai_range_v: float = 10.0, formats=("tif", "png", "json")) -> list:
    """Save a :class:`~easyebl.imaging.engine.Frame`; returns the written paths.

    *path* may carry any of the extensions; the stem is used for all formats.
    """
    p = Path(path)
    stem = p.with_suffix("") if p.suffix.lower() in (".tif", ".tiff", ".png", ".npy", ".json") else p
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    meta = frame.metadata()
    meta["ai_range_v"] = ai_range_v
    meta["tif_scale"] = f"V = (value / 65535) * {2 * ai_range_v:g} - {ai_range_v:g}"
    multi = len(frame.images) > 1
    for ch, img in frame.images.items():
        base = Path(f"{stem}_{ch}") if multi else stem
        if "tif" in formats:
            u16 = np.clip((np.nan_to_num(img, nan=-ai_range_v) + ai_range_v) / (2 * ai_range_v) * 65535, 0, 65535)
            write_tiff16(base.with_suffix(".tif"), u16.astype(np.uint16), json.dumps(meta))
            written.append(base.with_suffix(".tif"))
        if "png" in formats:
            lv = levels or auto_levels(img)
            write_png(base.with_suffix(".png"), to_uint8(img, lv))
            meta.setdefault("display_levels_v", {})[ch] = list(lv)
            written.append(base.with_suffix(".png"))
        if "npy" in formats:
            np.save(base.with_suffix(".npy"), np.asarray(img, np.float32))
            written.append(base.with_suffix(".npy"))
    if "json" in formats:
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append(stem.with_suffix(".json"))
    return written
