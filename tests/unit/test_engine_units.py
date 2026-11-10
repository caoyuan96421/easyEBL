"""Imaging-engine building blocks: fast waveform source, sample ring, stream assembler, export."""
import json

import numpy as np
import pytest

from easyebl.core import AffineTransform, FieldCalibration
from easyebl.core.waveform import concat_chunks
from easyebl.imaging import LineAssembler, RasterConfig, RasterPlan
from easyebl.imaging.engine.assembler import (
    CaptureAccumulator,
    FrameRecord,
    ImageAccumulator,
    SampleRing,
    StreamAssembler,
)
from easyebl.imaging.engine.settings import EngineOptions, speed_by_name, valid_rates
from easyebl.imaging.engine.source import FrameSource
from easyebl.imaging.export import auto_levels, to_uint8, write_png, write_tiff16

CAL = FieldCalibration(100, 10).with_correction(AffineTransform.from_params(sx=1.01, rotation_rad=0.01, tx=0.3))

VARIANTS = [
    {},
    dict(bidirectional=True, rotation_rad=0.2, flyback_shape="cosine", y_direction="up", x_direction=-1,
         preline_mode="hold"),
    dict(flyback_s=0, preline_s=0, line_sync_marker="ls", frame_sync_marker="fs"),
    dict(flyback_shape="step", blank_flyback=False, blank_preline=True, settle_samples=1),
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_frame_source_matches_raster_plan(kw):
    cfg = RasterConfig(nx=17, ny=9, width_um=40, center_um=(3, -2), dwell_s=3e-6, rate=1e6, **kw)
    plan = RasterPlan(cfg, CAL)
    src = FrameSource(plan)
    ref = concat_chunks(list(plan._frame_chunks(0, 4, None)))
    got = concat_chunks([src.settle(None), src.lines(0, 4), src.lines(4, 9, src.line_end_v[3])])
    np.testing.assert_allclose(got.x, ref.x, atol=1e-12)
    np.testing.assert_allclose(got.y, ref.y, atol=1e-12)
    np.testing.assert_array_equal(got.blank, ref.blank)
    assert set(got.markers) == set(ref.markers)
    for k in ref.markers:
        np.testing.assert_array_equal(got.markers[k], ref.markers[k])
    # travelling in from a previous position (second frame of a live stream)
    prev_v = np.array([1.0, 2.0])
    ref2 = concat_chunks(list(plan._frame_chunks(0, 4, CAL.volts_to_um(prev_v[None])[0])))
    got2 = src.frame(prev_v)
    np.testing.assert_allclose(got2.x, ref2.x, atol=1e-12)
    np.testing.assert_allclose(got2.y, ref2.y, atol=1e-12)
    assert len(src.tail(prev_v)) == plan.n_tail


def test_sample_ring_wrap_and_grow():
    r = SampleRing(2, 10)
    data = np.vstack([np.arange(25.0), -np.arange(25.0)])
    for s in range(0, 25, 7):
        r.write(data[:, s : s + 7])
    assert r.end == 25 and r.start == 15
    np.testing.assert_array_equal(r.get(15, 25), data[:, 15:25])
    with pytest.raises(IndexError):
        r.get(14, 20)
    r.ensure(40)
    np.testing.assert_array_equal(r.get(15, 25), data[:, 15:25])
    r.write(data + 100)  # 25 more samples, wrapping inside the grown ring
    assert r.end == 50
    np.testing.assert_array_equal(r.get(30, 50), data[:, 5:25] + 100)
    big = np.ones((2, 100))
    r.write(big)  # larger than the ring: only the newest samples are kept
    assert r.end == 150 and r.start == 110
    np.testing.assert_array_equal(r.get(110, 150), np.ones((2, 40)))


def _run_assembler(cfg, blocks, lag=0.0):
    """Feed a synthetic AI stream (value = absolute sample index) through both assemblers."""
    cfg = cfg.with_(lag_comp_s=lag)
    plan = RasterPlan(cfg, FieldCalibration(100, 10))
    src = FrameSource(plan)
    rows = {}

    def on_line(rec, j, row, vals):
        rows[(rec.index, row)] = vals[0].copy()

    done = []
    asm = StreamAssembler(1, on_line, lambda rec: done.append(rec.index))
    total = 0
    recs = []
    for f in range(2):
        rec = FrameRecord(index=f, start=total, source=src)
        asm.add_record(rec)
        recs.append(rec)
        total += src.frame_len
    total += src.n_tail
    stream = np.arange(float(total))[None, :]
    pos = 0
    for b in blocks:
        asm.feed(stream[:, pos : pos + b])
        pos += b
        if pos >= total:
            break
    if pos < total:
        asm.feed(stream[:, pos:])
    return rows, done, plan


@pytest.mark.parametrize("blocks", [[1] * 10_000 + [10**9], [997] * 200, [10**9]])
@pytest.mark.parametrize("bidir,lag", [(False, 0.0), (True, 3.3e-6)])
def test_stream_assembler_matches_line_assembler(blocks, bidir, lag):
    cfg = RasterConfig(nx=12, ny=7, width_um=30, dwell_s=4e-6, rate=1e6, bidirectional=bidir, settle_samples=1)
    rows, done, plan = _run_assembler(cfg, blocks, lag)
    assert done == [0, 1]
    # reference: the classic LineAssembler on the same stream for one frame
    ref = LineAssembler(plan, ("video",))
    ref.feed(np.arange(float(plan.total + 10))[None, :])
    img = ref.images()["video"]
    for r in range(cfg.ny):
        np.testing.assert_allclose(rows[(0, r)], img[r], atol=1e-9)
        # second frame: same layout shifted by one frame length
        np.testing.assert_allclose(rows[(1, r)], img[r] + plan.frame_len, atol=1e-9)


def test_cut_record_finishes_early():
    cfg = RasterConfig(nx=8, ny=10, width_um=20, dwell_s=2e-6, rate=1e6)
    src = FrameSource(RasterPlan(cfg, FieldCalibration(100, 10)))
    got, done = [], []
    asm = StreamAssembler(1, lambda rec, j, row, v: got.append((rec.index, j)), lambda rec: done.append(rec))
    a = FrameRecord(index=0, start=0, source=src)
    asm.add_record(a)
    a.n_lines = 3  # cut after 3 lines (settings changed)
    b = FrameRecord(index=1, start=a.end, source=src)
    asm.add_record(b)
    asm.feed(np.zeros((1, b.end + src.n_tail)))
    assert [r.index for r in done] == [0, 1] and done[0].cut and not done[1].cut
    assert got[:3] == [(0, 0), (0, 1), (0, 2)] and len(got) == 3 + 10


def test_image_accumulator_running_average_and_reset():
    acc = ImageAccumulator(("video",))
    cfg = RasterConfig(nx=4, ny=2, width_um=10)
    assert acc.ensure(cfg)
    for k in range(10):
        acc.add_row(0, np.full((1, 4), float(k)), n_avg=4)
    # exponential average over ≈4 frames, starting with a plain mean
    assert 5.0 < acc.images["video"][0, 0] < 9.0
    assert np.isnan(acc.images["video"][1]).all()
    assert not acc.ensure(cfg.with_(dwell_s=7e-6))  # speed change keeps the average
    assert acc.ensure(cfg.with_(center_um=(1.0, 0.0)))  # beam shift restarts it…
    assert acc.row_n.sum() == 0 and not np.isnan(acc.images["video"][0, 0])  # …but keeps pixels visible
    ca = CaptureAccumulator(("video",), 2, 4, 3)
    for k in (1.0, 2.0, 6.0):
        ca.add_row(0, np.full((1, 4), k))
    np.testing.assert_allclose(ca.images()["video"][0], 3.0)


def test_presets_and_options():
    assert speed_by_name("slow 2").dwell_s == 10e-6 and speed_by_name("PHOTO1").name == "Photo 1"
    with pytest.raises(KeyError):
        speed_by_name("warp")
    rates = valid_rates(2e6)
    assert rates[0] == 2e6 and all(abs(100e6 / r - round(100e6 / r)) < 1e-9 for r in rates)
    with pytest.raises(ValueError):
        EngineOptions(latency_s=0.001).validate()
    assert EngineOptions().with_(averaging=8).averaging == 8


def test_export_writers(tmp_path):
    img = np.linspace(-1, 1, 60 * 40, dtype=np.float32).reshape(40, 60)
    lv = auto_levels(img)
    u8 = to_uint8(img, lv)
    assert u8.dtype == np.uint8 and u8.min() == 0 and u8.max() == 255
    write_png(tmp_path / "a.png", u8)
    u16 = (np.arange(60 * 40) % 65536).astype(np.uint16).reshape(40, 60)
    write_png(tmp_path / "b.png", u16)
    write_tiff16(tmp_path / "c.tif", u16, json.dumps({"k": 1}))
    raw = (tmp_path / "c.tif").read_bytes()
    assert raw[:4] == b"II*\x00"
    np.testing.assert_array_equal(np.frombuffer(raw[-u16.nbytes :], "<u2").reshape(40, 60), u16)
    qtgui = pytest.importorskip("PySide6.QtGui")
    for name, fmt in (("a.png", "Format_Grayscale8"), ("b.png", "Format_Grayscale16"), ("c.tif", "Format_Grayscale16")):
        q = qtgui.QImage(str(tmp_path / name))
        assert (q.width(), q.height()) == (60, 40) and q.format().name == fmt
