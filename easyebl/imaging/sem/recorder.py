"""Video recording of the SEM image (the SEM tab's *Record* tool).

While recording, :class:`SEMRecorder` samples the displayed image at a fixed frame rate (real
time: a slow scan shows line by line, a frozen image repeats) into a :class:`FrameStore`, a
temporary file of 8-bit grey frames (long recordings do not fill the memory).  When it stops, the
frames are written as

* **AVI, uncompressed** (:func:`write_avi`): 8-bit grey (a grey palette), OpenDML (AVI 2.0)
  so files beyond 1 GB play; or
* **MP4, H.264** (:func:`write_mp4`): Qt Multimedia's recorder (FFmpeg backend), no extra package.

Every frame has the size of the first one (a frame of another size is fitted into it, black
around).
"""
from __future__ import annotations

import os
import struct
import tempfile
import time
from typing import Callable, Iterator

import numpy as np
from PySide6 import QtCore, QtGui

__all__ = ["FrameStore", "SEMRecorder", "write_avi", "write_mp4", "FORMATS"]

#: save-dialog filter -> extension
FORMATS = {"MP4 video, H.264 (*.mp4)": ".mp4", "AVI video, uncompressed (*.avi)": ".avi"}

Progress = Callable[[int, int], bool]  # (done, total) -> keep going


# ----------------------------------------------------------------------------- frames on disk
class FrameStore:
    """Fixed-size 8-bit grey frames appended to a temporary file."""

    def __init__(self, width: int, height: int, directory: str | None = None):
        self.width, self.height = int(width), int(height)
        fd, self.path = tempfile.mkstemp(prefix="easyebl_rec_", suffix=".gray", dir=directory)
        self._fh = os.fdopen(fd, "w+b")
        self.count = 0

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height

    def append(self, gray: bytes | np.ndarray) -> None:
        b = gray.tobytes() if isinstance(gray, np.ndarray) else gray
        if len(b) != self.frame_bytes:
            raise ValueError(f"frame of {len(b)} bytes, {self.frame_bytes} expected")
        self._fh.write(b)
        self.count += 1

    def frames(self) -> Iterator[np.ndarray]:
        """The frames in order, (height, width) uint8, top row first."""
        self._fh.flush()
        n = self.frame_bytes
        with open(self.path, "rb") as fh:
            for _ in range(self.count):
                yield np.frombuffer(fh.read(n), np.uint8).reshape(self.height, self.width)

    def discard(self) -> None:
        try:
            self._fh.close()
        finally:
            try:
                os.remove(self.path)
            except OSError:
                pass


def gray_frame(img: QtGui.QImage, width: int, height: int) -> np.ndarray:
    """*img* as (height, width) uint8 grey; another size is fitted in (aspect kept, black around)."""
    g = img.convertToFormat(QtGui.QImage.Format_Grayscale8)
    if g.width() != width or g.height() != height:
        s = g.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        out = np.zeros((height, width), np.uint8)
        a = _qimage_gray(s)
        y0, x0 = (height - a.shape[0]) // 2, (width - a.shape[1]) // 2
        out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
        return out
    return _qimage_gray(g)


def _qimage_gray(g: QtGui.QImage) -> np.ndarray:
    w, h, bpl = g.width(), g.height(), g.bytesPerLine()
    a = np.frombuffer(g.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return a[:, :w].copy()


# ----------------------------------------------------------------------------- the sampler
class SEMRecorder(QtCore.QObject):
    """Samples ``grab()`` (a QImage, or None while there is nothing to show) *fps* times a second.
    Frames missed by a late timer are filled with the last image, so the video keeps real time."""

    def __init__(self, grab: Callable[[], QtGui.QImage | None], fps: float = 10.0, directory: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.grab = grab
        self.fps = float(fps)
        self.directory = directory
        self.store: FrameStore | None = None
        self._last: np.ndarray | None = None
        self._last_img = None
        self.t0: float | None = None
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.sample)

    @property
    def recording(self) -> bool:
        return self.timer.isActive()

    @property
    def frames(self) -> int:
        return self.store.count if self.store is not None else 0

    @property
    def elapsed_s(self) -> float:
        return self.frames / self.fps

    def start(self) -> None:
        self.timer.start(max(1, int(round(1000.0 / self.fps))))
        self.sample()

    def sample(self) -> None:
        img = self.grab()
        if img is None and self._last is None:
            return  # nothing shown yet: the video starts with the first image
        if img is not None and img is not self._last_img:
            if self.store is None:
                self.store = FrameStore(img.width(), img.height(), self.directory)
                self.t0 = time.perf_counter()
            self._last = gray_frame(img, self.store.width, self.store.height)
            self._last_img = img
        due = int((time.perf_counter() - self.t0) * self.fps) + 1
        for _ in range(max(1, min(due - self.store.count, int(2 * self.fps)))):  # catch up (≤ 2 s at once)
            self.store.append(self._last)

    def stop(self) -> FrameStore | None:
        """Stop sampling; the frames (None if nothing was recorded)."""
        self.timer.stop()
        store, self.store = self.store, None
        self._last = self._last_img = None
        return store


# ----------------------------------------------------------------------------- AVI
_GRAY_PALETTE = b"".join(bytes((i, i, i, 0)) for i in range(256))
_RIFF_LIMIT = 1 << 30  # one RIFF segment ≤ 1 GiB (the AVI 1.0 limit; AVIX segments follow)
_SUPER_ENTRIES = 256  # room for 256 segments (≈ 256 GB)


def write_avi(path: str, store: FrameStore, fps: float, progress: Progress | None = None) -> bool:
    """Write *store* as an uncompressed 8-bit grey AVI (OpenDML).  False if *progress* cancelled
    (the partial file is removed)."""
    w, h, n = store.width, store.height, store.count
    stride = (w + 3) & ~3
    size = stride * h  # a DIB frame: rows bottom-up, padded to 4 bytes
    scale, rate = 1000, max(1, int(round(fps * 1000)))
    per_seg = max(1, (_RIFF_LIMIT - 64 * 1024) // (size + 8 + 8 + 16))  # frame + ix00 + idx1 entries
    try:
        with open(path, "wb") as f:
            def u32(v):
                return struct.pack("<I", v)

            def patch(pos, fmt, *v):
                here = f.tell()
                f.seek(pos)
                f.write(struct.pack(fmt, *v))
                f.seek(here)

            # --- RIFF 'AVI ' / hdrl
            f.write(b"RIFF" + u32(0) + b"AVI ")
            riff0 = 4
            f.write(b"LIST" + u32(0) + b"hdrl")
            hdrl = f.tell() - 12  # the LIST header
            f.write(b"avih" + u32(56))
            avih = f.tell()
            f.write(struct.pack("<14I", int(round(1e6 / fps)), int(size * fps), 0, 0x10, 0, 0, 1, size + 8,
                                w, h, 0, 0, 0, 0))
            f.write(b"LIST" + u32(0) + b"strl")
            strl = f.tell() - 12
            f.write(b"strh" + u32(56))
            strh = f.tell()
            f.write(b"vids" + b"DIB " + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, scale, rate, 0, n, size + 8,
                                                    0xFFFFFFFF, 0, 0, 0, w, h))
            f.write(b"strf" + u32(40 + 1024))
            f.write(struct.pack("<IiiHHIIiiII", 40, w, h, 1, 8, 0, size, 0, 0, 256, 0) + _GRAY_PALETTE)
            f.write(b"indx" + u32(24 + 16 * _SUPER_ENTRIES))
            indx = f.tell()
            f.write(struct.pack("<HBBI", 4, 0, 0, 0) + b"00db" + bytes(12) + bytes(16 * _SUPER_ENTRIES))
            patch(strl + 4, "<I", f.tell() - strl - 8)
            f.write(b"LIST" + u32(4 + 8 + 248) + b"odml" + b"dmlh" + u32(248))
            dmlh = f.tell()
            f.write(u32(n) + bytes(244))
            patch(hdrl + 4, "<I", f.tell() - hdrl - 8)

            supers = []  # (ix00 offset, ix00 size, frames)
            first_frames = 0
            done = 0
            frames = store.frames()
            seg = 0
            while done < n or seg == 0:
                if seg > 0:
                    f.write(b"RIFF" + u32(0) + b"AVIX")
                    riff = f.tell() - 12
                else:
                    riff = riff0 - 4
                f.write(b"LIST" + u32(0) + b"movi")
                movi = f.tell() - 4  # idx1 offsets are relative to the 'movi' fourcc
                count = min(per_seg, n - done)
                offsets = []
                for _ in range(count):
                    a = next(frames)
                    dib = np.zeros((h, stride), np.uint8)
                    dib[:, :w] = a[::-1]
                    offsets.append(f.tell())
                    f.write(b"00db" + u32(size))
                    f.write(dib.tobytes())
                    done += 1
                    if progress is not None and done % 10 == 0 and not progress(done, n):
                        raise _Cancelled
                # the segment's standard index
                ix = f.tell()
                base = offsets[0] + 8 if offsets else 0
                f.write(b"ix00" + u32(24 + 8 * count) + struct.pack("<HBBI", 2, 0, 1, count) + b"00db"
                        + struct.pack("<QI", base, 0))
                for o in offsets:
                    f.write(struct.pack("<II", o + 8 - base, size))
                supers.append((ix, 32 + 8 * count, count))
                patch(movi - 4, "<I", f.tell() - movi)
                if seg == 0:  # the AVI 1.0 index (players that do not read OpenDML)
                    first_frames = count
                    f.write(b"idx1" + u32(16 * count))
                    for o in offsets:
                        f.write(b"00db" + struct.pack("<III", 0x10, o - movi, size))
                patch(riff + 4, "<I", f.tell() - riff - 8)
                seg += 1
                if seg >= _SUPER_ENTRIES and done < n:
                    raise ValueError("recording too long for one AVI file")
            # headers: frame counts, the super index
            patch(avih + 16, "<I", first_frames)
            patch(dmlh, "<I", n)
            patch(strh + 32, "<I", n)
            patch(indx + 4, "<I", len(supers))
            for k, (o, sz, cnt) in enumerate(supers):
                patch(indx + 24 + 16 * k, "<QII", o, sz, cnt)
        if progress is not None:
            progress(n, n)
        return True
    except _Cancelled:
        _remove(path)
        return False
    except BaseException:
        _remove(path)
        raise


class _Cancelled(Exception):
    pass


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ----------------------------------------------------------------------------- MP4
def write_mp4(path: str, store: FrameStore, fps: float, progress: Progress | None = None,
              timeout_s: float = 600.0) -> bool:
    """Write *store* as H.264 MP4 with Qt Multimedia (runs a local event loop until the file is
    finished).  False if *progress* cancelled (the partial file is removed); RuntimeError if the
    recorder fails."""
    from PySide6 import QtMultimedia as M

    w, h, n = store.width, store.height, store.count
    W, H = w + (w & 1), h + (h & 1)  # H.264 (4:2:0) wants even sizes
    session = M.QMediaCaptureSession()
    inp = M.QVideoFrameInput()
    session.setVideoFrameInput(inp)
    rec = M.QMediaRecorder()
    session.setRecorder(rec)
    fmt = M.QMediaFormat(M.QMediaFormat.FileFormat.MPEG4)
    fmt.setVideoCodec(M.QMediaFormat.VideoCodec.H264)
    rec.setMediaFormat(fmt)
    rec.setQuality(M.QMediaRecorder.Quality.VeryHighQuality)
    rec.setVideoFrameRate(float(fps))
    rec.setVideoResolution(QtCore.QSize(W, H))
    rec.setOutputLocation(QtCore.QUrl.fromLocalFile(os.path.abspath(path)))
    loop = QtCore.QEventLoop()
    st = {"i": 0, "error": None, "cancel": False, "stopping": False}
    source = store.frames()
    pending: list = []  # a frame the full queue refused: sent again first

    def stop():
        if not st["stopping"]:
            st["stopping"] = True
            rec.stop()

    def push():
        while st["i"] < n and not st["stopping"]:
            if pending:
                frame = pending.pop()
            else:
                a = next(source)
                if (W, H) != (w, h):
                    b = np.zeros((H, W), np.uint8)
                    b[:h, :w] = a
                    a = b
                img = QtGui.QImage(a.data, W, H, W, QtGui.QImage.Format_Grayscale8)
                frame = M.QVideoFrame(img.convertToFormat(QtGui.QImage.Format_RGB32))
                frame.setStartTime(int(st["i"] * 1e6 / fps))
                frame.setEndTime(int((st["i"] + 1) * 1e6 / fps))
            if not inp.sendVideoFrame(frame):
                pending.append(frame)
                return  # readyToSendVideoFrame calls again
            st["i"] += 1
            if progress is not None and st["i"] % 10 == 0 and not progress(st["i"], n):
                st["cancel"] = True
                stop()
                return
        stop()

    def on_state(s_):
        if s_ == M.QMediaRecorder.RecorderState.StoppedState and st["stopping"]:
            loop.quit()

    def on_error(err, text=""):
        st["error"] = text or str(err)
        st["stopping"] = True
        loop.quit()

    guard = QtCore.QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(lambda: on_error("timed out"))
    guard.start(int(timeout_s * 1000))
    inp.readyToSendVideoFrame.connect(push)
    rec.recorderStateChanged.connect(on_state)
    rec.errorOccurred.connect(on_error)
    rec.record()
    QtCore.QTimer.singleShot(0, push)
    loop.exec()
    guard.stop()
    if st["error"]:
        _remove(path)
        raise RuntimeError(f"MP4 recording failed: {st['error']}")
    if st["cancel"]:
        _remove(path)
        return False
    if progress is not None:
        progress(n, n)
    return True
