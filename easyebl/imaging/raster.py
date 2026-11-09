"""Raster-scan SEM imaging with synchronised X/Y, blanker and video.

Line anatomy (unidirectional scan, one row)::

    |<- flyback ->|<- pre-line ->|<------------- pixels ------------->|
     beam returns   lead-in          nx pixels × samples_per_pixel
     (blankable)    (blankable)      (beam on, video sampled)

* **flyback** – beam travels from the end of the previous line to the start
  of the pre-line region (``linear``/``cosine``/``step`` trajectory).
  Blanked by default.
* **pre-line** – lead-in so the deflection settles before the first pixel.
  ``ramp`` mode continues the pixel staircase outside the image (overscan,
  steady-state lag before the first pixel); ``hold`` mode waits at the
  first pixel.  Optionally blanked.
* **pixels** – staircase, one position per pixel held for
  ``samples_per_pixel`` clock ticks.  The first ``settle_samples`` of each
  pixel can be discarded.

In bidirectional (serpentine) mode the flyback becomes a turnaround and odd
rows run in the opposite direction.

All AI samples are acquired on the output clock.  ``lag_comp_s`` shifts the
AI stream (fractional samples allowed) so that video sample ``k + d`` is
attributed to output sample ``k`` – compensating deflection + detector lag.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import threading
import time
from typing import Callable, Iterator, Sequence

import numpy as np

from ..core.calibration import FieldCalibration
from ..core.waveform import OutputChunk, concat_chunks

__all__ = ["RasterConfig", "RasterPlan", "RasterImage", "RasterImager", "LineAssembler"]

_FLYBACK_SHAPES = ("linear", "cosine", "step")
_PRELINE_MODES = ("ramp", "hold")


@dataclass
class RasterConfig:
    """Parameters of one raster scan (all lengths in µm, times in s)."""

    nx: int = 256
    ny: int = 256
    center_um: tuple = (0.0, 0.0)
    width_um: float = 80.0
    height_um: float | None = None  # None -> square pixels
    rotation_rad: float = 0.0
    rate: float = 400e3  # sample clock
    dwell_s: float = 5e-6  # per pixel; quantised to whole samples
    settle_samples: int = 0  # discarded at the start of each pixel
    preline_s: float = 20e-6
    preline_mode: str = "ramp"
    blank_preline: bool = False
    flyback_s: float = 40e-6
    flyback_shape: str = "linear"
    blank_flyback: bool = True
    frame_settle_s: float = 200e-6  # blanked wait at the first line start
    bidirectional: bool = False
    x_direction: int = +1  # direction of the first line (+1: left→right)
    y_direction: str = "down"  # first row scanned: "down" = top row first
    frames: int = 1  # frames averaged
    ai_channels: tuple = ("video",)
    lag_comp_s: float = 0.0  # deflection + detector lag compensation
    ai_latency_samples: float = 1.0  # AI converts on the edge that updates the DAC → sees the previous sample
    line_sync_marker: str | None = None  # timed DO line asserted during pixels
    frame_sync_marker: str | None = None  # asserted during the first pixel line of each frame
    blank_during_pixels: bool = False  # dark/reference scan: beam blanked for the whole line

    def __post_init__(self):
        self.validate()

    # -------------------------------------------------------------- derived
    def validate(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be ≥ 1")
        if self.rate <= 0 or self.dwell_s <= 0:
            raise ValueError("rate and dwell must be positive")
        if self.preline_mode not in _PRELINE_MODES:
            raise ValueError(f"preline_mode must be one of {_PRELINE_MODES}")
        if self.flyback_shape not in _FLYBACK_SHAPES:
            raise ValueError(f"flyback_shape must be one of {_FLYBACK_SHAPES}")
        if self.x_direction not in (1, -1):
            raise ValueError("x_direction must be +1 or -1")
        if self.y_direction not in ("down", "up"):
            raise ValueError("y_direction must be 'down' or 'up'")
        if self.frames < 1:
            raise ValueError("frames must be ≥ 1")
        if not 0 <= self.settle_samples < self.samples_per_pixel:
            raise ValueError("settle_samples must be < samples_per_pixel")
        if self.width_um <= 0 or (self.height_um is not None and self.height_um <= 0):
            raise ValueError("scan size must be positive")

    @property
    def samples_per_pixel(self) -> int:
        return max(1, int(round(self.dwell_s * self.rate)))

    @property
    def actual_dwell_s(self) -> float:
        return self.samples_per_pixel / self.rate

    @property
    def height(self) -> float:
        return self.height_um if self.height_um is not None else self.width_um * self.ny / self.nx

    @property
    def pixel_size_um(self) -> tuple:
        return (self.width_um / self.nx, self.height / self.ny)

    @property
    def lag_comp_samples(self) -> float:
        """Total AI delay in samples (fixed converter latency + lag compensation)."""
        return self.ai_latency_samples + self.lag_comp_s * self.rate

    def n(self, t: float) -> int:
        return max(0, int(round(t * self.rate)))

    def with_(self, **kw) -> "RasterConfig":
        return replace(self, **kw)


@dataclass
class RasterImage:
    """Result of a raster acquisition."""

    data: dict
    config: RasterConfig
    calibration: FieldCalibration | None = None
    line_directions: np.ndarray | None = None
    frame_time_s: float = 0.0
    aborted: bool = False
    rows_completed: int = 0
    timestamp: float = field(default_factory=time.time)

    def image(self, channel: str | None = None) -> np.ndarray:
        if channel is None:
            channel = next(iter(self.data))
        return self.data[channel]

    @property
    def pixel_size_um(self) -> tuple:
        return self.config.pixel_size_um

    @property
    def extent(self) -> tuple:
        """(x0, x1, y0, y1) in field µm, ignoring rotation."""
        cx, cy = self.config.center_um
        w, h = self.config.width_um, self.config.height
        return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

    def pixel_to_field(self, col, row) -> np.ndarray:
        """Field µm of (sub-)pixel coordinates (col, row); row 0 = top."""
        cfg = self.config
        pw, ph = cfg.pixel_size_um
        u = -cfg.width_um / 2 + (np.asarray(col, float) + 0.5) * pw
        v = cfg.height / 2 - (np.asarray(row, float) + 0.5) * ph
        c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
        x = cfg.center_um[0] + c * u - s * v
        y = cfg.center_um[1] + s * u + c * v
        return np.stack([x, y], axis=-1)

    def field_to_pixel(self, xy) -> np.ndarray:
        cfg = self.config
        xy = np.asarray(xy, float)
        dx = xy[..., 0] - cfg.center_um[0]
        dy = xy[..., 1] - cfg.center_um[1]
        c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
        u = c * dx + s * dy
        v = -s * dx + c * dy
        pw, ph = cfg.pixel_size_um
        col = (u + cfg.width_um / 2) / pw - 0.5
        row = (cfg.height / 2 - v) / ph - 0.5
        return np.stack([col, row], axis=-1)


class RasterPlan:
    """Pre-computed sample layout of a raster scan.

    Index conventions: *row* is the image row (0 = top), *line* is the scan
    order index.  ``line_rows[j]`` is the row scanned as line *j*.
    """

    def __init__(self, cfg: RasterConfig, calibration: FieldCalibration):
        cfg.validate()
        self.cfg = cfg
        self.cal = calibration
        self.spp = cfg.samples_per_pixel
        self.n_fb = cfg.n(cfg.flyback_s)
        self.n_pre = cfg.n(cfg.preline_s)
        self.n_pix = cfg.nx * self.spp
        self.line_len = self.n_fb + self.n_pre + self.n_pix
        self.n_frame_settle = max(1, cfg.n(cfg.frame_settle_s))
        d = cfg.lag_comp_samples
        if d < 0:
            raise ValueError("total AI delay (ai_latency_samples + lag_comp_s) must be ≥ 0")
        self.n_tail = int(math.ceil(d)) + 2
        # frame = frame-settle + ny lines (the first line's flyback acts as part of the settle)
        self.frame_len = self.n_frame_settle + cfg.ny * self.line_len
        self.total = cfg.frames * self.frame_len + self.n_tail
        rows = np.arange(cfg.ny)
        self.line_rows = rows if cfg.y_direction == "down" else rows[::-1]
        dirs = np.full(cfg.ny, cfg.x_direction)
        if cfg.bidirectional:
            dirs[1::2] *= -1
        self.line_dirs = dirs
        pw, ph = cfg.pixel_size_um
        self._pw, self._ph = pw, ph
        c, s = math.cos(cfg.rotation_rad), math.sin(cfg.rotation_rad)
        self._rot = np.array([[c, -s], [s, c]])
        # validate that the whole trajectory fits the field
        self._check_extent()

    # ------------------------------------------------------------- geometry
    def _u_of_index(self, i: np.ndarray) -> np.ndarray:
        """Local u (µm) of (possibly fractional / out-of-range) pixel index along +x."""
        return -self.cfg.width_um / 2 + (np.asarray(i, float) + 0.5) * self._pw

    def _v_of_row(self, row) -> np.ndarray:
        return self.cfg.height / 2 - (np.asarray(row, float) + 0.5) * self._ph

    def _to_field(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        uv = np.column_stack([u, v])
        return uv @ self._rot.T + np.asarray(self.cfg.center_um, float)

    def _pre_pixel_index(self, direction: int) -> np.ndarray:
        """Pixel-staircase index (along the scan direction) for pre-line samples."""
        k = np.arange(self.n_pre)  # 0 .. n_pre-1
        if self.cfg.preline_mode == "hold" or self.n_pre == 0:
            idx = np.zeros(self.n_pre)
        else:
            # count back from the first pixel in whole-pixel steps
            back = (self.n_pre - k + self.spp - 1) // self.spp  # 1..ceil(n_pre/spp)
            idx = -back.astype(float)
        return idx

    def _line_local(self, line: int):
        """(u, v, blank, pixel_mask) for one line *excluding* the flyback."""
        cfg = self.cfg
        row = self.line_rows[line]
        d = self.line_dirs[line]
        pre_idx = self._pre_pixel_index(d)
        pix_idx = np.repeat(np.arange(cfg.nx), self.spp).astype(float)
        idx = np.concatenate([pre_idx, pix_idx])
        col = idx if d > 0 else (cfg.nx - 1 - idx)
        u = self._u_of_index(col)
        v = np.full(u.shape, float(self._v_of_row(row)))
        blank = np.concatenate(
            [
                np.full(self.n_pre, cfg.blank_preline or cfg.blank_during_pixels),
                np.full(self.n_pix, cfg.blank_during_pixels),
            ]
        )
        return u, v, blank

    def line_start_um(self, line: int) -> np.ndarray:
        u, v, _ = self._line_local(line)
        if len(u) == 0:  # pragma: no cover
            return self._to_field(np.array([0.0]), np.array([0.0]))[0]
        return self._to_field(u[:1], v[:1])[0]

    def line_end_um(self, line: int) -> np.ndarray:
        u, v, _ = self._line_local(line)
        return self._to_field(u[-1:], v[-1:])[0]

    def _transition(self, p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2))
        t = (np.arange(n) + 1) / n
        shape = self.cfg.flyback_shape
        if shape == "step":
            w = np.ones(n)
        elif shape == "cosine":
            w = 0.5 - 0.5 * np.cos(np.pi * t)
        else:
            w = t
        return p0 + (p1 - p0) * w[:, None]

    def _check_extent(self) -> None:
        cfg = self.cfg
        pts = []
        for line in {0, cfg.ny - 1}:
            for d in (1, -1):
                pre = self._pre_pixel_index(d)
                first = float(pre[0]) if len(pre) else 0.0
                col_first = first if d > 0 else cfg.nx - 1 - first
                col_last = cfg.nx - 1 if d > 0 else 0
                v = float(self._v_of_row(self.line_rows[line]))
                for col in (col_first, col_last):
                    pts.append(self._to_field(np.array([self._u_of_index(col)]), np.array([v]))[0])
        self.cal.um_to_volts(np.array(pts))  # raises OutOfFieldError

    # -------------------------------------------------------------- samples
    def line_start_sample(self, frame: int, line: int) -> int:
        return frame * self.frame_len + self.n_frame_settle + line * self.line_len

    def pixel_sample_starts(self, frame: int, line: int) -> np.ndarray:
        """Absolute sample index of the first sample of each *image column* of ``line``."""
        base = self.line_start_sample(frame, line) + self.n_fb + self.n_pre
        k = np.arange(self.cfg.nx) * self.spp
        if self.line_dirs[line] < 0:
            k = k[::-1]
        return base + k

    @property
    def frame_time_s(self) -> float:
        return self.frame_len / self.cfg.rate

    @property
    def total_time_s(self) -> float:
        return self.total / self.cfg.rate

    @property
    def efficiency(self) -> float:
        return self.cfg.ny * self.n_pix / self.frame_len

    def summary(self) -> dict:
        cfg = self.cfg
        return {
            "samples_per_pixel": self.spp,
            "dwell_s": cfg.actual_dwell_s,
            "line_samples": self.line_len,
            "flyback_samples": self.n_fb,
            "preline_samples": self.n_pre,
            "frame_time_s": self.frame_time_s,
            "total_time_s": self.total_time_s,
            "duty_cycle": self.efficiency,
            "pixel_size_um": cfg.pixel_size_um,
        }

    # -------------------------------------------------------------- streams
    def _frame_chunks(self, frame: int, lines_per_chunk: int, prev_end: np.ndarray | None):
        cfg = self.cfg
        cal = self.cal
        lm = cfg.line_sync_marker
        fm = cfg.frame_sync_marker
        start0 = self.line_start_um(0)
        # frame settle: (blanked) jump to the start of the first line's pre-line region, then wait
        n_fs = self.n_frame_settle
        pos_fs = np.repeat(start0[None, :], n_fs, axis=0)
        if prev_end is not None and n_fs > 1:
            # travel there smoothly using the flyback shape over the first part of the settle
            n_tr = min(n_fs - 1, max(self.n_fb, 1))
            pos_fs[:n_tr] = self._transition(prev_end, start0, n_tr)
        mk = {}
        if lm:
            mk[lm] = np.zeros(n_fs, bool)
        if fm:
            mk[fm] = np.zeros(n_fs, bool)
        v = cal.um_to_volts(pos_fs)
        pending = [OutputChunk(v[:, 0], v[:, 1], np.ones(n_fs, bool), mk)]
        prev = start0
        for line in range(cfg.ny):
            u, vv, blank = self._line_local(line)
            body = self._to_field(u, vv)
            start = body[0]
            if line == 0:
                fb = np.repeat(start[None, :], self.n_fb, axis=0)
            else:
                fb = self._transition(prev, start, self.n_fb)
            pos = np.vstack([fb, body])
            bl = np.concatenate([np.full(self.n_fb, cfg.blank_flyback if line else True), blank])
            volts = cal.um_to_volts(pos, check=False)
            mk = {}
            if lm or fm:
                pix = np.zeros(len(pos), bool)
                pix[self.n_fb + self.n_pre :] = True
                if lm:
                    mk[lm] = pix
                if fm:
                    mk[fm] = pix if line == 0 else np.zeros(len(pos), bool)
            pending.append(OutputChunk(volts[:, 0], volts[:, 1], bl, mk))
            prev = body[-1]
            if len(pending) >= lines_per_chunk:
                yield concat_chunks(pending)
                pending = []
        if pending:
            yield concat_chunks(pending)
        self._last_end = prev

    def chunks(self, lines_per_chunk: int = 8) -> Iterator[OutputChunk]:
        """Generate the complete output stream (all frames + tail)."""
        prev_end = None
        markers = [m for m in (self.cfg.line_sync_marker, self.cfg.frame_sync_marker) if m]
        for f in range(self.cfg.frames):
            yield from self._frame_chunks(f, lines_per_chunk, prev_end)
            prev_end = self._last_end
        # tail: hold blanked at the last position so lag-compensated AI covers the last pixels
        v = self.cal.um_to_volts(prev_end[None, :], check=False)[0]
        yield OutputChunk.constant(self.n_tail, v[0], v[1], True, {m: False for m in markers})

    def full_waveform(self) -> OutputChunk:
        """Whole stream as one chunk (for previews / tests; avoid for huge scans)."""
        return concat_chunks(self.chunks())


class LineAssembler:
    """Turns the AI sample stream into image rows as data arrives."""

    def __init__(self, plan: RasterPlan, channels: Sequence[str]):
        self.plan = plan
        self.channels = tuple(channels)
        cfg = plan.cfg
        self.sums = {c: np.zeros((cfg.ny, cfg.nx)) for c in self.channels}
        self.counts = np.zeros(cfg.ny, int)
        self._buf = np.zeros((len(self.channels), 0))
        self._buf_start = 0
        self._next = (0, 0)  # (frame, line) awaiting data
        self.rows_done = 0
        d = cfg.lag_comp_samples
        self._d_int = int(math.floor(d))
        self._d_frac = d - self._d_int
        k = np.arange(cfg.settle_samples, plan.spp)
        self._offsets = k + self._d_int

    def feed(self, block: np.ndarray, on_row: Callable[[int], None] | None = None) -> None:
        self._buf = np.concatenate([self._buf, block], axis=1) if self._buf.size else block.copy()
        plan = self.plan
        cfg = plan.cfg
        while self._next[0] < cfg.frames:
            f, line = self._next
            starts = plan.pixel_sample_starts(f, line)
            need_end = int(starts.max()) + int(self._offsets.max()) + 2
            if self._buf_start + self._buf.shape[1] < need_end:
                break
            idx = starts[:, None] + self._offsets[None, :] - self._buf_start
            row = plan.line_rows[line]
            for ci, ch in enumerate(self.channels):
                a = self._buf[ci]
                vals = a[idx]
                if self._d_frac > 1e-9:
                    vals = (1 - self._d_frac) * vals + self._d_frac * a[idx + 1]
                self.sums[ch][row] += vals.mean(axis=1)
            self.counts[row] += 1
            self.rows_done += 1
            if on_row is not None:
                on_row(row)
            line += 1
            if line >= cfg.ny:
                f, line = f + 1, 0
            self._next = (f, line)
            # drop consumed samples
            if f < cfg.frames:
                keep_from = int(plan.pixel_sample_starts(f, line).min())
                # never drop samples that have not arrived yet (blocks may end mid-line)
                drop = min(keep_from - self._buf_start, self._buf.shape[1])
                if drop > 0:
                    self._buf = self._buf[:, drop:]
                    self._buf_start += drop

    def images(self) -> dict:
        out = {}
        c = np.maximum(self.counts, 1)[:, None]
        for ch, s in self.sums.items():
            img = s / c
            img[self.counts == 0] = np.nan
            out[ch] = img
        return out


def _row_directions(plan: RasterPlan) -> np.ndarray:
    inv = np.empty(plan.cfg.ny, dtype=int)
    inv[plan.line_rows] = np.arange(plan.cfg.ny)
    return plan.line_dirs[inv]


class RasterImager:
    """High-level raster imaging on a :class:`DAQHandle`."""

    def __init__(self, handle, calibration: FieldCalibration):
        self.handle = handle
        self.calibration = calibration

    def plan(self, cfg: RasterConfig) -> RasterPlan:
        return RasterPlan(cfg, self.calibration)

    def acquire(
        self,
        cfg: RasterConfig,
        on_row: Callable[[int, Callable[[], dict]], None] | None = None,
        timeout: float = 0.0,
        lines_per_chunk: int = 8,
    ) -> RasterImage:
        """Run one acquisition (``cfg.frames`` frames averaged).

        ``on_row(row, get_images)`` is called from the acquisition thread
        whenever a row is completed; ``get_images()`` returns the partial
        images and should only be called when actually needed (it copies the
        frame).  Keep the callback cheap – it runs in the streaming thread.
        Call ``handle.abort()`` from another thread to stop early.
        """
        plan = self.plan(cfg)
        asm = LineAssembler(plan, cfg.ai_channels)

        def consume(block):
            asm.feed(block, (lambda r: on_row(r, asm.images)) if on_row else None)

        res = self.handle.run_stream(
            cfg.rate,
            plan.chunks(lines_per_chunk),
            ai_channels=cfg.ai_channels,
            on_ai=consume,
            timeout=timeout,
        )
        return RasterImage(
            data=asm.images(),
            config=cfg,
            calibration=self.calibration,
            line_directions=_row_directions(plan),
            frame_time_s=plan.frame_time_s,
            aborted=res.aborted,
            rows_completed=asm.rows_done,
        )

    # ------------------------------------------------------------------ live
    def live(
        self,
        cfg: RasterConfig,
        on_frame: Callable[[int, "RasterImage"], None],
        on_row: Callable[[int, Callable[[], dict]], None] | None = None,
        stop: threading.Event | None = None,
        max_frames: int | None = None,
        timeout: float = 0.0,
        lines_per_chunk: int = 8,
        cache_limit_samples: int = 20_000_000,
    ) -> int:
        """Scan frames back-to-back in **one** hardware-timed stream.

        Unlike repeated :meth:`acquire` calls, the DAQ is never stopped between
        frames: the frame waveform is written into the output buffer over and
        over, and the AI stream is cut into frames as it arrives.  This removes
        the per-run task setup (≈0.2 s on USB) and keeps the deflection in a
        steady state.

        ``cfg.frames`` is ignored (each frame is delivered on its own; average
        in the caller).  ``on_frame(index, image)`` is called from the
        streaming thread for every completed frame; ``on_row`` as in
        :meth:`acquire` for the frame in progress.  The stream ends after the
        current frame when ``stop`` is set or ``max_frames`` frames are done;
        ``handle.abort()`` stops it immediately.  Returns the number of
        completed frames.
        """
        one = cfg.with_(frames=1)
        plan = self.plan(one)
        L = plan.frame_len
        span = L + plan.n_tail  # samples one frame's assembler needs (lag tail)
        channels = tuple(one.ai_channels)
        stop = stop or threading.Event()
        dirs = _row_directions(plan)
        state = {"k": 0, "asm": LineAssembler(plan, channels), "fed": 0, "buf": None, "buf0": 0}

        def frames_wanted() -> bool:
            return not stop.is_set() and (max_frames is None or state["k"] < max_frames)

        def gen():
            # first frame starts from the parked position, later frames travel from the previous end
            yield from plan._frame_chunks(0, lines_per_chunk, None)
            steady_end = plan._last_end
            cached = None
            if L <= cache_limit_samples:
                cached = list(plan._frame_chunks(0, lines_per_chunk, steady_end))
            produced = 1
            # keep at least one frame queued beyond the one being delivered
            while frames_wanted() and (max_frames is None or produced < max_frames):
                yield from (cached if cached is not None else plan._frame_chunks(0, lines_per_chunk, steady_end))
                produced += 1
            v = self.calibration.um_to_volts(plan._last_end[None, :], check=False)[0]
            markers = [m for m in (one.line_sync_marker, one.frame_sync_marker) if m]
            yield OutputChunk.constant(plan.n_tail, v[0], v[1], True, {m: False for m in markers})

        def consume(block: np.ndarray):
            st = state
            st["buf"] = block if st["buf"] is None else np.concatenate([st["buf"], block], axis=1)
            while True:
                k = st["k"]
                start = k * L  # absolute index of this frame's first sample
                have_end = st["buf0"] + st["buf"].shape[1]
                lo = start + st["fed"]
                hi = min(have_end, start + span)
                if hi > lo:
                    seg = st["buf"][:, lo - st["buf0"] : hi - st["buf0"]]
                    cb = (lambda r: on_row(r, st["asm"].images)) if on_row else None
                    st["asm"].feed(seg, cb)
                    st["fed"] += hi - lo
                if st["fed"] < span:
                    return
                img = RasterImage(
                    data=st["asm"].images(),
                    config=one,
                    calibration=self.calibration,
                    line_directions=dirs,
                    frame_time_s=plan.frame_time_s,
                    rows_completed=st["asm"].rows_done,
                )
                on_frame(k, img)
                st["k"] = k + 1
                st["asm"] = LineAssembler(plan, channels)
                st["fed"] = 0
                drop = (k + 1) * L - st["buf0"]
                st["buf"] = st["buf"][:, drop:]
                st["buf0"] += drop

        self.handle.run_stream(one.rate, gen(), ai_channels=channels, on_ai=consume, timeout=timeout)
        return state["k"]
