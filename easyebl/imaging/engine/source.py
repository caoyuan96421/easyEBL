"""Fast, line-addressable waveform generation for a :class:`RasterPlan`.

:meth:`RasterPlan.chunks` recomputes every sample's geometry (rotation,
calibration) each frame.  For continuous imaging at 2 MS/s the engine needs
waveforms cheaply and *line by line* (settings can change at any line
boundary), so this module precomputes per-direction line templates in volts
and builds each line with one add:

    volts = M · (u, v) + t      (M, t: rotation ∘ field calibration, affine)
    X_line = M00·u_template + (M01·v_row + t0)
    Y_line = M10·u_template + (M11·v_row + t1)

The output is sample-for-sample identical (to float rounding) to
``RasterPlan._frame_chunks`` – the unit tests check this.
"""
from __future__ import annotations

import numpy as np

from ...core.calibration import OutOfFieldError
from ...core.waveform import OutputChunk
from ..raster import RasterPlan

__all__ = ["FrameSource"]


class FrameSource:
    """Waveform pieces of one frame of *plan*: settle, line blocks, tail."""

    def __init__(self, plan: RasterPlan):
        self.plan = plan
        cfg = self.cfg = plan.cfg
        self.n_fs = plan.n_frame_settle
        self.n_fb = plan.n_fb
        self.n_pre = plan.n_pre
        self.n_pix = plan.n_pix
        self.line_len = plan.line_len
        self.frame_len = plan.frame_len
        self.n_tail = plan.n_tail
        self.ny = cfg.ny
        # affine µm(local u, v) -> volts
        A = plan.cal.um_to_volts_transform
        M = A.linear @ plan._rot
        t = A.linear @ np.asarray(cfg.center_um, float) + A.offset
        self._M, self._t = M, t
        # local u of each body sample (pre-line + pixels) for both scan directions
        pre_idx = plan._pre_pixel_index(+1)
        pix_idx = np.repeat(np.arange(cfg.nx), plan.spp).astype(float)
        idx = np.concatenate([pre_idx, pix_idx])
        self._tmpl = {}
        for d in (1, -1):
            col = idx if d > 0 else (cfg.nx - 1 - idx)
            u = plan._u_of_index(col)
            self._tmpl[d] = (M[0, 0] * u, M[1, 0] * u, u[0], u[-1])
        self._body_blank = np.concatenate(
            [
                np.full(self.n_pre, cfg.blank_preline or cfg.blank_during_pixels),
                np.full(self.n_pix, cfg.blank_during_pixels),
            ]
        )
        rows = plan.line_rows
        self._v = plan._v_of_row(rows).astype(float)  # v of each *line* (scan order)
        self._dirs = plan.line_dirs
        self._ox = M[0, 1] * self._v + t[0]  # per-line X offset
        self._oy = M[1, 1] * self._v + t[1]
        # everything per line is vectorised (no Python loop over lines): this constructor also
        # validates settings in the caller's thread and must not hog the GIL
        fwd = self._dirs > 0
        T, B = self._tmpl[1], self._tmpl[-1]
        u0 = np.where(fwd, T[2], B[2])
        u1 = np.where(fwd, T[3], B[3])
        self.line_start_v = np.column_stack([M[0, 0] * u0 + self._ox, M[1, 0] * u0 + self._oy])
        self.line_end_v = np.column_stack([M[0, 0] * u1 + self._ox, M[1, 0] * u1 + self._oy])
        # exact voltage extremes of every emitted body sample (flybacks/settles interpolate
        # between line ends, so they stay inside these bounds)
        x_lo = np.where(fwd, T[0].min(), B[0].min()) + self._ox
        x_hi = np.where(fwd, T[0].max(), B[0].max()) + self._ox
        y_lo = np.where(fwd, T[1].min(), B[1].min()) + self._oy
        y_hi = np.where(fwd, T[1].max(), B[1].max()) + self._oy
        self.v_max = float(max(np.abs(x_lo).max(), np.abs(x_hi).max(), np.abs(y_lo).max(), np.abs(y_hi).max()))
        self._lm = cfg.line_sync_marker
        self._fm = cfg.frame_sync_marker
        self.markers = tuple(m for m in (self._lm, self._fm) if m)

    def check(self, limit_v: float) -> None:
        """Raise :class:`OutOfFieldError` unless every sample is within ±*limit_v* (no tolerance)."""
        if not self.v_max <= limit_v:
            raise OutOfFieldError(
                f"scan needs {self.v_max:.6g} V, beyond the ±{limit_v:g} V deflection range; "
                "reduce the scan width, beam shift, rotation or pre-line overscan"
            )

    # ------------------------------------------------------------ helpers
    def _weights(self, n: int) -> np.ndarray:
        t = (np.arange(n) + 1) / n
        shape = self.cfg.flyback_shape
        if shape == "step":
            return np.ones(n)
        if shape == "cosine":
            return 0.5 - 0.5 * np.cos(np.pi * t)
        return t

    def _mk(self, n: int) -> dict:
        return {m: np.zeros(n, bool) for m in self.markers}

    # ------------------------------------------------------------- pieces
    def settle(self, prev_end_v=None) -> OutputChunk:
        """Blanked frame settle at the first line start (travels there from *prev_end_v*)."""
        n = self.n_fs
        start = self.line_start_v[0]
        x = np.full(n, start[0])
        y = np.full(n, start[1])
        if prev_end_v is not None and n > 1:
            n_tr = min(n - 1, max(self.n_fb, 1))
            w = self._weights(n_tr)
            p0 = np.asarray(prev_end_v, float)
            x[:n_tr] = p0[0] + (start[0] - p0[0]) * w
            y[:n_tr] = p0[1] + (start[1] - p0[1]) * w
        return OutputChunk(x, y, np.ones(n, bool), self._mk(n))

    def lines(self, j0: int, j1: int, prev_end_v=None) -> OutputChunk:
        """Lines ``j0 … j1-1`` (scan order) as one chunk, each with its flyback.

        *prev_end_v* is where the beam was before line *j0* (ignored for
        ``j0 == 0``, whose flyback holds at the start, blanked).
        """
        L = self.line_len
        k = j1 - j0
        n = k * L
        x = np.empty(n)
        y = np.empty(n)
        b = np.empty(n, bool)
        mk = self._mk(n)
        nfb, cfg = self.n_fb, self.cfg
        w = self._weights(nfb) if nfb else None
        prev = None if prev_end_v is None else np.asarray(prev_end_v, float)
        for i, j in enumerate(range(j0, j1)):
            s = i * L
            d = int(self._dirs[j])
            tx, ty, _, _ = self._tmpl[d]
            start = self.line_start_v[j]
            if nfb:
                if j == 0 or prev is None:
                    x[s : s + nfb] = start[0]
                    y[s : s + nfb] = start[1]
                    b[s : s + nfb] = True
                else:
                    x[s : s + nfb] = prev[0] + (start[0] - prev[0]) * w
                    y[s : s + nfb] = prev[1] + (start[1] - prev[1]) * w
                    b[s : s + nfb] = cfg.blank_flyback
            np.add(tx, self._ox[j], out=x[s + nfb : s + L])
            np.add(ty, self._oy[j], out=y[s + nfb : s + L])
            b[s + nfb : s + L] = self._body_blank
            if self._lm:
                mk[self._lm][s + nfb + self.n_pre : s + L] = True
            if self._fm and j == 0:
                mk[self._fm][s + nfb + self.n_pre : s + L] = True
            prev = self.line_end_v[j]
        return OutputChunk(x, y, b, mk)

    def tail(self, last_v) -> OutputChunk:
        """Blanked hold at *last_v* so lag-compensated AI covers the last pixels."""
        return OutputChunk.constant(self.n_tail, last_v[0], last_v[1], True, {m: False for m in self.markers})

    def frame(self, prev_end_v=None) -> OutputChunk:
        """A whole frame (settle + all lines) – for tests and previews."""
        from ...core.waveform import concat_chunks

        return concat_chunks([self.settle(prev_end_v), self.lines(0, self.ny)])
