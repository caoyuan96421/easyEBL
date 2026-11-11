# Architecture and protocol notes

## Package layout

```
driver/
├── dp5/                   the driver package
│   ├── __init__.py        public exports: DP5, Spectrum, Status, COMMANDS, SPEEDS, exceptions
│   ├── device.py          DP5 class: worker thread, thread-safe API, acquisition, speeds, auto speed
│   ├── transport.py       USBTransport (pyusb + libusb-1.0 over WinUSB)
│   ├── protocol.py        packet build/parse, PID tables, ACK codes, exceptions
│   ├── status.py          64-byte status block -> Status dataclass
│   ├── spectrum.py        Spectrum dataclass, spectrum packet decoding, .mca read/write
│   ├── commands.py        catalogue of every ASCII command + config-string helpers
│   ├── presets.py         speed presets 1-5, dead-time model, verbatim SDK SDD files
│   ├── gain.py            coarse gain tables (DP5, DP5 Rev Dx, PX5, DP5-X)
│   └── mock.py            MockTransport: simulated DP5 speaking the real protocol
├── dp5_gui.py             PySide6 GUI (SDD settings only)
├── test_dp5_live.py       hardware smoke test (MCA + Delta)
├── examples/              runnable examples (all accept --mock)
├── tools/gen_docs.py      regenerates docs/api_reference.md and docs/settings_reference.md
├── docs/                  this documentation
└── *.bat                  Windows launchers (GUI, hardware test, GUI test)
```

## Threading model

```
 caller threads (CLI, Qt GUI, scripts)                worker thread "DP5-worker"
 ─────────────────────────────────────                ───────────────────────────
 dp.write_config({...}) ──> queue.put(job, Future) ──> job runs: transport.transact(...)
        blocks on Future.result(timeout) <──────────── Future.set_result / set_exception
                                                       │
 dp.latest() / accumulated() ──> copy under RLock <─── poll every poll_interval while running:
                                                       │   02 03 (MCA) or 02 04 (Delta)
 listener(event, payload)  <────────── called on ──────┘   update latest / accumulated, emit events
```

* **One owner of the USB link.** Only the worker thread calls `transport.transact()`. The DP5 protocol is strictly
  one request, then one response, so access must be serialised. The queue does that.
* **Every public method is thread-safe.** It queues a job and blocks until it completes (default `call_timeout` 10 s).
  Calls made *from* the worker thread, for example inside a listener, run immediately, so a listener can call `dp.get()` without
  deadlocking.
* **Polling happens between jobs.** The worker waits on the queue with a timeout until the next poll is due, so
  configuration changes made during an acquisition are applied between two spectrum reads.
* **Shared state** (`latest`, `accumulated`, cached status/config, mode, interval) is protected by an `RLock`. The getters
  return copies.
* **Listeners run on the worker thread.** Keep them short. In Qt, re-emit a `Signal` (as `dp5_gui.py` does with its
  `Bridge` object) so the work happens on the GUI thread. Exceptions in listeners are logged and swallowed.

### Events

| Event | Payload | When |
|---|---|---|
| `spectrum` | `{"latest": Spectrum, "accumulated": Spectrum}` | after every poll |
| `status` | `Status` | after `status()` |
| `started` | mode string | after `start()` |
| `stopped` | `"user"`, `"preset"` or `"error"` | after `stop()`, a DP5 preset stop, or a poll failure |
| `config` | dict actually sent | after `write_config()` (and so after `apply_speed()`) |
| `error` | exception | when a poll fails (acquisition stops) |

## USB transport

| Item | Value |
|---|---|
| VID / PID | `0x10C4` / `0x842A`, configuration 1, interface 0 |
| Endpoints | bulk OUT `0x02`, bulk IN `0x81`, 64-byte packets |
| Request / response | write one packet, then read with a 32 KiB buffer until the length in the header is satisfied |
| Timeouts | 500 ms (2.5 s for diagnostic data) |
| Zero-length packet | sent after any request whose length is a multiple of 64 bytes (see below) |
| Libraries | `pyusb` + libusb-1.0 from `libusb-package` (talks to WinUSB directly) |

`USBTransport.open()` claims interface 0 and flushes stale IN data. Several DP5s can be opened with
`USBTransport(index=n)`.

**Zero-length packets.** A USB bulk transfer ends with a packet shorter than 64 bytes. A request
that is exactly 64·n bytes long therefore needs an extra zero-length packet (ZLP), or the DP5
keeps waiting for more data and never answers. On the bench this hung speed presets 2 and 3,
whose configuration packets are exactly 128 bytes. libusb doesn't add the ZLP itself (WinUSB only
does with the `SHORT_PACKET_TERMINATE` pipe policy), so `transact()` sends it.
`test_dp5_full.py` section 2d checks 64/128/192/256/512-byte requests.

## Packet format

```
F5 FA | PID1 PID2 | LEN_hi LEN_lo | DATA (LEN bytes) | CHK_hi CHK_lo
CHK = (-sum(all preceding bytes)) & 0xFFFF
```

| Purpose | Request → response |
|---|---|
| Status | `01 01` → `80 01` (64 bytes) |
| Spectrum (+ status) | `02 01` / `02 03` → `81 xx`. Use `02 02` / `02 04` to also clear. Channels = 256 << ((PID2−1)>>1); 3 bytes LE per channel; even PID2 has 64 status bytes appended |
| Write configuration | `20 02` "CMD=VAL;..." (≤ 512 B per packet) → ACK |
| Read configuration | `20 03` "CMD=?;..." → `82 07` "CMD=VAL;..." |
| Enable / disable MCA | `F0 02` / `F0 03` → ACK |
| Autoset fast threshold / input offset | `F0 06` / `F0 05` → ACK |
| Diagnostic data | `03 05` → `82 05` |

The complete PID and ACK tables are in `dp5/protocol.py` and in `../DP5_USB_Protocol_Summary.md`, which
was written from the Amptek SDK.

## Adding another transport (e.g. Ethernet)

`DP5` only needs an object with:

```python
class MyTransport:
    def open(self): ...
    def close(self): ...
    def transact(self, pid1: int, pid2: int, data: bytes = b"", timeout_ms: int | None = None) -> Packet: ...
```

The DP5's Ethernet interface carries the **same packets over UDP port 10001**. A UDP transport would send
`build_packet(...)` as one datagram and collect datagrams until `expected_length()` is reached. Amptek's C# SDK
retries 5 times and uses about 100 ms timeouts. Then use `DP5(MyTransport(...))`; nothing else changes.

## Testing without hardware

`dp5.mock.MockTransport` implements status, spectrum and clear requests, configuration write/readback, ACKs, and the
MCA enable/disable and preset-time behaviour. It produces a Fe-55 (Mn K) spectrum whose
dead time follows the τ ≈ 1.07·(2·TPEA+TFLA) model. Set `mock.reject = {"CUSP"}` to simulate a command that this FPGA
rejects.

```python
from dp5 import DP5
from dp5.mock import MockTransport
with DP5(MockTransport(cps=49000, seed=1)) as dp:
    ...
```

## Regenerating the reference docs

```
python tools/gen_docs.py
```

This rebuilds `docs/api_reference.md` from the docstrings and `docs/settings_reference.md` from `COMMANDS`,
`SPEEDS`, `ALL_SDK_SDD` and the gain tables. Run it after changing any of those.
