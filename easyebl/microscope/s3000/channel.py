"""Carriage-return line channel shared by the SEM and stage clients.

One background thread reads the transport, splits the byte stream on CR
(0x0D) and offers each complete line to the registered handlers in order.
Every transmitted and received line is kept in a timestamped raw log so that
stale replies, malformed lines and unexpected acknowledgements can be
diagnosed later (manual section 10).

The channel also owns the *bus lock*. Both clients take it for a normal
transaction, which guarantees that only one request is outstanding on a
half-duplex port at a time, whether the SEM and stage share the port or not.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

from .errors import TransportError, ValidationError
from .transport import Transport

CR = b"\r"

#: Handler signature: ``handler(text, raw_bytes) -> bool``; return True to consume the line.
LineHandler = Callable[[str, bytes], bool]


@dataclass(frozen=True)
class LogEntry:
    """One raw-log record."""

    timestamp: float
    direction: str  # "TX", "RX" or "NOTE"
    data: bytes
    note: str = ""

    def format(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        ms = int((self.timestamp % 1) * 1000)
        shown = self.data.decode("ascii", errors="backslashreplace").replace("\r", "<CR>").replace("\n", "<LF>")
        tail = f"  # {self.note}" if self.note else ""
        return f"{t}.{ms:03d} {self.direction:<4} {shown}{tail}"


def validate_line_text(text: str) -> bytes:
    """Encode one command line, refusing anything the protocol does not allow.

    The text must be printable ASCII without CR, LF or tabs; the CR terminator
    is appended here and must not be included by the caller.
    """
    if not isinstance(text, str) or not text:
        raise ValidationError("Command text must be a non-empty string")
    if text != text.strip(" "):
        raise ValidationError(f"Leading or trailing spaces are not allowed: {text!r}")
    if "  " in text:
        raise ValidationError(f"Fields must be separated by exactly one space: {text!r}")
    for ch in text:
        if ch in "\r\n\t":
            raise ValidationError("Do not include CR, LF or TAB; the CR terminator is added automatically")
        if not (32 <= ord(ch) < 127):
            raise ValidationError(f"Non-printable or non-ASCII character {ch!r} in {text!r}")
    return text.encode("ascii") + CR


class LineChannel:
    """Assemble CR-terminated lines from a transport and dispatch them."""

    def __init__(self, transport: Transport, *, log_size: int = 20000, poll_interval: float = 0.05):
        self._transport = transport
        self._poll = poll_interval
        self._write_lock = threading.Lock()
        self.bus_lock = threading.Lock()
        self._handlers: list[LineHandler] = []
        self._handlers_lock = threading.Lock()
        self._buf = bytearray()
        self._log: deque[LogEntry] = deque(maxlen=log_size)
        self._log_lock = threading.Lock()
        self._log_listeners: list[Callable[[LogEntry], None]] = []
        self._error_listeners: list[Callable[[Exception], None]] = []
        self.unclaimed: deque[str] = deque(maxlen=1000)
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="s3000-reader", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ handlers
    def add_handler(self, handler: LineHandler, *, first: bool = False) -> None:
        with self._handlers_lock:
            if first:
                self._handlers.insert(0, handler)
            else:
                self._handlers.append(handler)

    def add_log_listener(self, listener: Callable[[LogEntry], None]) -> None:
        self._log_listeners.append(listener)

    def add_error_listener(self, listener: Callable[[Exception], None]) -> None:
        self._error_listeners.append(listener)

    # ------------------------------------------------------------------ log
    def note(self, text: str) -> None:
        """Add a free-text annotation to the raw log."""
        self._append_log(LogEntry(time.time(), "NOTE", b"", text))

    def _append_log(self, entry: LogEntry) -> None:
        with self._log_lock:
            self._log.append(entry)
        for listener in list(self._log_listeners):
            try:
                listener(entry)
            except Exception:  # pragma: no cover - listener bugs must not kill I/O
                pass

    def log_entries(self) -> list[LogEntry]:
        with self._log_lock:
            return list(self._log)

    def dump_log(self, path: str) -> None:
        """Write the raw log to a text file (one entry per line)."""
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.log_entries():
                fh.write(entry.format() + "\n")

    @property
    def partial(self) -> bytes:
        """Bytes received since the last CR (retained, never discarded)."""
        return bytes(self._buf)

    # ------------------------------------------------------------------ I/O
    @property
    def closed(self) -> bool:
        return self._stop.is_set()

    def send_line(self, text: str, *, note: str = "") -> None:
        """Encode, terminate with CR and write one line. Does not wait for a reply."""
        data = validate_line_text(text)
        if self._stop.is_set():
            raise TransportError("Channel is closed" + (f": {self.error}" if self.error else ""))
        with self._write_lock:
            self._transport.write(data)
            self._append_log(LogEntry(time.time(), "TX", data, note))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._transport.read(self._poll)
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.error = exc
                self._append_log(LogEntry(time.time(), "NOTE", b"", f"transport error: {exc}"))
                self._stop.set()
                for listener in list(self._error_listeners):
                    try:
                        listener(exc)
                    except Exception:  # pragma: no cover
                        pass
                break
            if chunk:
                self._feed(chunk)

    def _feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)
        while True:
            idx = self._buf.find(CR)
            if idx < 0:
                return
            raw = bytes(self._buf[: idx + 1])
            del self._buf[: idx + 1]
            self._append_log(LogEntry(time.time(), "RX", raw))
            # The manual documents CR only. Tolerate a stray LF for parsing but keep it in the log.
            text = raw[:-1].decode("ascii", errors="replace").strip("\n")
            if not text:
                continue
            self._dispatch(text, raw)

    def _dispatch(self, text: str, raw: bytes) -> None:
        with self._handlers_lock:
            handlers: Iterable[LineHandler] = list(self._handlers)
        for handler in handlers:
            try:
                if handler(text, raw):
                    return
            except Exception as exc:  # pragma: no cover - defensive
                self._append_log(LogEntry(time.time(), "NOTE", b"", f"handler error: {exc!r}"))
        self.unclaimed.append(text)
        self._append_log(LogEntry(time.time(), "NOTE", b"", f"unclaimed line {text!r}"))

    def close(self) -> None:
        self._stop.set()
        try:
            self._transport.close()
        finally:
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2)
