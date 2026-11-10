"""Connection supervision: notice a lost device, reconnect when it is back.

The :class:`DAQManager` calls :meth:`ReconnectWatchdog.start` when an
operation fails with :class:`DeviceLostError` (USB unplugged, power cycled,
cable glitch).  The watchdog probes the device every ``interval_s`` without
disturbing anything and, once it answers, asks the manager to re-establish
the session (safe state, SEM internal scan, then the previous external-scan
request).  It stops after a successful reconnect or when the manager closes.
"""
from __future__ import annotations

import threading

__all__ = ["ReconnectWatchdog"]


class ReconnectWatchdog:
    def __init__(self, manager, interval_s: float = 0.5):
        self._mgr = manager
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.attempts = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="easyebl-reconnect", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.attempts += 1
            try:
                if self._mgr._try_reconnect():
                    return
            except Exception as e:  # noqa: BLE001 - keep trying; never die silently
                self.last_error = f"{type(e).__name__}: {e}"
            self._stop.wait(self.interval_s)

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)
