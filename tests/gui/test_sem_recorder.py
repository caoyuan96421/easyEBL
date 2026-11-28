"""SEM video recording: the frame store, the real-time sampler, the AVI / MP4 writers."""
import os
import struct
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtMultimedia  # noqa: E402

from easyebl.imaging.sem import recorder as R  # noqa: E402


def _store(tmp_path, n=25, w=101, h=60):
    s = R.FrameStore(w, h, directory=str(tmp_path))
    for i in range(n):
        a = np.zeros((h, w), np.uint8)
        a[:, :] = (np.arange(w) * 2 + i) % 256
        a[0, 0] = i  # identifies the frame (top-left pixel)
        s.append(a)
    return s


def _chunks(buf, start, end):
    """(fourcc, data offset, size) of the chunks in buf[start:end]."""
    p = start
    while p + 8 <= end:
        fcc, size = buf[p:p + 4], struct.unpack_from("<I", buf, p + 4)[0]
        yield fcc, p + 8, size
        p += 8 + size + (size & 1)


def _probe(path, qapp):
    p = QtMultimedia.QMediaPlayer()
    p.setSource(QtCore.QUrl.fromLocalFile(str(path)))
    t = time.time()
    done = (QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia, QtMultimedia.QMediaPlayer.MediaStatus.InvalidMedia)
    while p.mediaStatus() not in done and time.time() - t < 10:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)
    res = p.metaData().value(QtMultimedia.QMediaMetaData.Key.Resolution)
    return p.mediaStatus(), p.duration(), res


def test_avi_is_uncompressed_8_bit_grey_with_both_indexes(tmp_path, qapp):
    s = _store(tmp_path)
    out = tmp_path / "rec.avi"
    seen = []
    assert R.write_avi(str(out), s, 10.0, progress=lambda d, n: seen.append((d, n)) or True)
    assert seen[-1] == (25, 25)
    b = out.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"AVI " and struct.unpack_from("<I", b, 4)[0] == len(b) - 8
    top = {fcc: (o, n) for fcc, o, n in _chunks(b, 12, len(b))}
    assert set(top) == {b"LIST", b"idx1"} or b"idx1" in top
    # hdrl: 101 × 60, 8 bits, uncompressed, 25 frames at 10 fps
    i = b.index(b"avih") + 8
    us, _, _, _, total = struct.unpack_from("<5I", b, i)
    assert us == 100000 and total == 25 and struct.unpack_from("<II", b, i + 32) == (101, 60)
    i = b.index(b"strh") + 8
    assert b[i:i + 4] == b"vids" and struct.unpack_from("<III", b, i + 20)[:2] == (1000, 10000)
    assert struct.unpack_from("<I", b, i + 32)[0] == 25
    i = b.index(b"strf") + 8
    _, w, h, _, bits, comp, size = struct.unpack_from("<IiiHHII", b, i)
    assert (w, h, bits, comp, size) == (101, 60, 8, 0, 104 * 60)  # rows padded to 4 bytes, bottom-up
    assert b[i + 40 + 4 * 200:i + 40 + 4 * 201] == bytes((200, 200, 200, 0))  # grey palette
    assert struct.unpack_from("<I", b, b.index(b"dmlh") + 8)[0] == 25
    # the frames: 00db chunks inside movi, in order, bottom-up rows
    mv = b.index(b"movi")
    size_movi = struct.unpack_from("<I", b, mv - 4)[0]
    frames = [(o, n) for fcc, o, n in _chunks(b, mv + 4, mv + size_movi) if fcc == b"00db"]
    assert len(frames) == 25 and all(n == 104 * 60 for _, n in frames)
    o, _ = frames[7]
    rows = np.frombuffer(b[o:o + 104 * 60], np.uint8).reshape(60, 104)
    assert rows[-1, 0] == 7 and rows[0, 5] == (5 * 2 + 7) % 256  # top row last; frame 7
    # idx1 points at the chunks (offsets from the 'movi' fourcc); the OpenDML index too
    i = b.index(b"idx1") + 8
    ck, flags, off, n = struct.unpack_from("<4sIII", b, i + 16 * 3)
    assert ck == b"00db" and flags == 0x10 and mv + off + 8 == frames[3][0] and n == 104 * 60
    ix = b.index(b"ix00")
    _, _, _, cnt, _, base = struct.unpack_from("<HBBI4sQ", b, ix + 8)
    assert cnt == 25 and base + struct.unpack_from("<I", b, ix + 32 + 8 * 4)[0] == frames[4][0]
    i = b.index(b"indx") + 8
    assert struct.unpack_from("<I", b, i + 4)[0] == 1
    assert struct.unpack_from("<QII", b, i + 24) == (ix, 32 + 8 * 25, 25)
    # a player reads it
    status, dur, res = _probe(out, qapp)
    assert status == QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia
    assert dur == pytest.approx(2500, abs=150) and res == QtCore.QSize(101, 60)
    s.discard()
    assert not os.path.exists(s.path)


def test_avi_beyond_one_riff_segment_is_opendml(tmp_path, qapp, monkeypatch):
    """Long recordings: RIFF 'AVI ' + RIFF 'AVIX' segments, one ix00 each, in the super index."""
    s = _store(tmp_path, n=25)
    monkeypatch.setattr(R, "_RIFF_LIMIT", 64 * 1024 + 10 * (104 * 60 + 32))  # 10 frames per segment
    out = tmp_path / "long.avi"
    assert R.write_avi(str(out), s, 10.0)
    b = out.read_bytes()
    segs = [(fcc, o) for fcc, o, n in _chunks(b, 0, len(b))]
    assert [fcc for fcc, _ in segs] == [b"RIFF"] * 3 and [b[o:o + 4] for _, o in segs] == [b"AVI ", b"AVIX", b"AVIX"]
    i = b.index(b"indx") + 8
    assert struct.unpack_from("<I", b, i + 4)[0] == 3
    entries = [struct.unpack_from("<QII", b, i + 24 + 16 * k) for k in range(3)]
    assert [e[2] for e in entries] == [10, 10, 5] and all(b[e[0]:e[0] + 4] == b"ix00" for e in entries)
    assert struct.unpack_from("<I", b, b.index(b"avih") + 8 + 16)[0] == 10  # AVI 1.0: the first segment
    assert struct.unpack_from("<I", b, b.index(b"dmlh") + 8)[0] == 25
    status, dur, res = _probe(out, qapp)
    assert status == QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia and dur == pytest.approx(2500, abs=150)
    s.discard()


def test_avi_cancel_removes_the_file(tmp_path):
    s = _store(tmp_path, n=40)
    out = tmp_path / "x.avi"
    assert not R.write_avi(str(out), s, 10.0, progress=lambda d, n: d < 20)
    assert not out.exists()
    s.discard()


def test_mp4_is_h264_at_the_frame_rate(tmp_path, qapp):
    s = _store(tmp_path, n=30)  # odd width: padded to even for H.264
    out = tmp_path / "rec.mp4"
    assert R.write_mp4(str(out), s, 10.0)
    assert out.stat().st_size > 1000 and out.read_bytes()[4:8] == b"ftyp"
    status, dur, res = _probe(out, qapp)
    assert status == QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia
    assert dur == pytest.approx(3000, abs=200) and res == QtCore.QSize(102, 60)
    s.discard()


def test_sampler_keeps_real_time_and_fits_other_sizes(tmp_path, qapp, qtbot):
    imgs = {"cur": None}
    rec = R.SEMRecorder(lambda: imgs["cur"], fps=20, directory=str(tmp_path))
    rec.start()
    qtbot.wait(100)
    assert rec.frames == 0  # nothing shown yet: no frames
    img = QtGui.QImage(64, 48, QtGui.QImage.Format_Grayscale8)
    img.fill(200)
    imgs["cur"] = img
    qtbot.wait(520)
    other = QtGui.QImage(32, 48, QtGui.QImage.Format_Grayscale8)  # another size: fitted in, black around
    other.fill(100)
    imgs["cur"] = other
    qtbot.wait(300)
    store = rec.stop()
    assert not rec.recording and store.width == 64 and store.height == 48
    assert 14 <= store.count <= 20  # ≈ 0.8 s at 20 fps
    fr = list(store.frames())
    assert fr[0][0, 0] == 200 and fr[-1][0, 0] == 0 and fr[-1][24, 32] == 100
    store.discard()
