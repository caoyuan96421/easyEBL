"""Cross-process device lock: one easyEBL process per physical DAQ.

Inside one process the :class:`~easyebl.hal.DAQManager` arbitrates (one stream
at a time, claims, the exclusive lease).  Two *processes* (say a SEM window and
an EDS window started separately) cannot see each other's manager, and DAQmx
would happily let both drive the same AO channels.  The NI backend therefore
takes an OS file lock named after the device when it opens; a second process
gets a clear :class:`~easyebl.hal.DeviceBusyError` naming the holder's PID
instead of fighting over the scan coils.  Run SEM and EDS in one process (the
suite launcher) to share the DAQ.

The lock lives in the temp directory and is released by the OS when the
process exits, even after a crash.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading

from .backend import DeviceBusyError

__all__ = ["DeviceLock"]

_LOCK_OFFSET = 4096  # lock a byte past the PID text, so other processes can still read it


_registry: dict = {}  # path -> [file handle, users]: one OS lock per process, shared by its instances
_reg_lock = threading.Lock()


class DeviceLock:
    """OS-level lock of one device name.  Instances in the same process share it."""

    def __init__(self, name: str, directory: str | None = None):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        self.path = os.path.join(directory or tempfile.gettempdir(), f"easyebl-{safe}.lock")
        self.name = name
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def holder_pid(self) -> int | None:
        try:
            with open(self.path, encoding="ascii", errors="ignore") as fh:
                txt = fh.read(64).strip()
            return int(txt.split()[0]) if txt else None
        except (OSError, ValueError, IndexError):
            return None

    def acquire(self) -> None:
        with _reg_lock:
            if self._held:
                return
            ent = _registry.get(self.path)
            if ent is not None:
                ent[1] += 1
                self._held = True
                return
            fh = open(self.path, "a+b")
            try:
                _lock(fh)
            except OSError:
                fh.close()
                pid = self.holder_pid()
                raise DeviceBusyError(
                    f"{self.name} is in use by another easyEBL process" + (f" (PID {pid})" if pid else "")
                    + " – close it, or run SEM and EDS together in one window (easyebl-suite)"
                ) from None
            fh.seek(0)
            fh.truncate()
            fh.write(f"{os.getpid()}\n".encode("ascii"))
            fh.flush()
            _registry[self.path] = [fh, 1]
            self._held = True

    def release(self) -> None:
        with _reg_lock:
            if not self._held:
                return
            self._held = False
            ent = _registry.get(self.path)
            if ent is None:
                return
            ent[1] -= 1
            if ent[1] > 0:
                return
            del _registry[self.path]
            fh = ent[0]
        try:
            fh.seek(0)
            fh.truncate()
            fh.flush()
            _unlock(fh)
        except OSError:
            pass
        finally:
            fh.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


if os.name == "nt":  # pragma: no cover - exercised on Windows
    import msvcrt

    def _lock(fh):
        fh.seek(_LOCK_OFFSET)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh):
        fh.seek(_LOCK_OFFSET)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fh):
        fcntl.lockf(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _LOCK_OFFSET)

    def _unlock(fh):
        fcntl.lockf(fh.fileno(), fcntl.LOCK_UN, 1, _LOCK_OFFSET)
