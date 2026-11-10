"""Minimal in-memory stand-in for the ``nidaqmx`` package.

Models just enough behaviour to exercise :class:`NIDAQBackend` control flow:
written AO/DO samples are "generated" as soon as the AO task runs, and AI
channels loop back the X deflection (so sample alignment can be checked).
"""
from __future__ import annotations

import enum
import sys
import types

import numpy as np


class FakeDevice:
    def __init__(self, name="Dev1"):
        self.name = name
        self.reset()

    def reset(self):
        self.present = True  # False = USB unplugged: every driver call fails with -201003
        self.unplug_after_writes = None  # unplug when this many AO stream writes have happened
        self.resets = 0
        self.fail_on_write = None  # (k, error_code): raise DaqError on the k-th AO stream write
        self.ao_writes = 0
        self.max_prestart_queue = 0
        self.ao_written = []  # list of (2, n) arrays
        self.do_written = []
        self.static_ao = []
        self.static_aux = []  # (physical channel, volts) of single-channel static AO writes
        self.static_do = []
        self.loopback_row = 0  # which AO channel of the stream the AI input sees (0 = X, 2 = AO2 …)
        self.stream_ao_channels = []
        self.ao_running = False
        self.ai_read = 0
        self.base = 0  # first sample of the running stream
        self._x = np.zeros(1 << 16)
        self.total = 0  # AO samples written over all streams
        self.tasks = []
        self.log = []

    def append_x(self, x):
        n = len(x)
        if self.total + n > len(self._x):
            new = np.zeros(max(2 * len(self._x), self.total + n))
            new[: self.total] = self._x[: self.total]
            self._x = new
        self._x[self.total : self.total + n] = x
        self.total += n

    @property
    def ao_x(self):
        return self._x[: self.total].copy()

    @property
    def do_words(self):
        return np.concatenate(self.do_written) if self.do_written else np.zeros(0, np.uint32)


DEVICE = FakeDevice()


class DaqError(Exception):
    def __init__(self, message, error_code, task_name=""):
        super().__init__(message)
        self.error_code = error_code


def _check():
    if DEVICE.unplug_after_writes is not None and DEVICE.ao_writes >= DEVICE.unplug_after_writes:
        DEVICE.unplug_after_writes = None
        DEVICE.present = False
        DEVICE.ao_running = False
    if not DEVICE.present:
        raise DaqError("Device cannot be accessed. Possible causes: Device is no longer present in the system.",
                       -201003)


class AcquisitionType(enum.Enum):
    CONTINUOUS = 1
    FINITE = 2


class LineGrouping(enum.Enum):
    CHAN_PER_LINE = 1
    CHAN_FOR_ALL_LINES = 2


class RegenerationMode(enum.Enum):
    ALLOW_REGENERATION = 1
    DONT_ALLOW_REGENERATION = 2


class TerminalConfiguration(enum.Enum):
    DIFF = 1
    RSE = 2
    NRSE = 3
    PSEUDO_DIFF = 4


class _Chans:
    def __init__(self, task, kind):
        self.task, self.kind = task, kind

    def add_ao_voltage_chan(self, phys, min_val=-10, max_val=10):
        self.task.channels.append(("ao", phys))

    def add_do_chan(self, phys, line_grouping=None):
        self.task.channels.append(("do", phys))

    def add_ai_voltage_chan(self, phys, terminal_config=None, min_val=-10, max_val=10):
        self.task.channels.append(("ai", phys))


class _Timing:
    def __init__(self, task):
        self.task = task

    def cfg_samp_clk_timing(self, rate, source=None, sample_mode=None, samps_per_chan=None):
        self.task.timing_cfg = dict(rate=rate, source=source, mode=sample_mode, buf=samps_per_chan)
        # X-series: the internal timebase divides 100 MHz
        self.samp_clk_rate = 100e6 / round(100e6 / rate)


class _OutStream:
    def __init__(self, task):
        self.task = task
        self.regen_mode = RegenerationMode.ALLOW_REGENERATION
        self.output_buf_size = None

    @property
    def total_samp_per_chan_generated(self):
        _check()
        return DEVICE.total - DEVICE.base if DEVICE.ao_running else 0


class _InStream:
    def __init__(self, task):
        self.task = task

    @property
    def avail_samp_per_chan(self):
        _check()
        if not DEVICE.ao_running:
            return 0
        return DEVICE.total - DEVICE.ai_read


class Task:
    def __init__(self, name=""):
        self.name = name
        self.channels = []
        self.timing_cfg = None
        self.started = False
        self.ao_channels = _Chans(self, "ao")
        self.do_channels = _Chans(self, "do")
        self.ai_channels = _Chans(self, "ai")
        self.timing = _Timing(self)
        self.out_stream = _OutStream(self)
        self.in_stream = _InStream(self)
        DEVICE.tasks.append(self)

    def kinds(self):
        return {k for k, _ in self.channels}

    def write(self, data, auto_start=False):
        _check()
        if "ao" in self.kinds() and np.isscalar(data):
            phys = [p for k, p in self.channels if k == "ao"][0]
            DEVICE.static_aux.append((phys, float(data)))
            DEVICE.log.append(("static_aux", (phys, float(data))))
        elif "ao" in self.kinds():
            DEVICE.static_ao.append(list(data))
            DEVICE.log.append(("static_ao", tuple(data)))
        else:
            DEVICE.static_do.append(data)
            DEVICE.log.append(("static_do", data))

    def read(self, timeout=1.0):
        _check()
        return 0.0

    def start(self):
        _check()
        self.started = True
        DEVICE.log.append(("start", self.name))
        if "ao" in self.kinds():
            DEVICE.ao_running = True
            DEVICE.base = getattr(self, "_base", DEVICE.total)
            DEVICE.ai_read = DEVICE.base

    def stop(self):
        DEVICE.log.append(("stop", self.name))
        if "ao" in self.kinds():
            DEVICE.ao_running = False

    def close(self):
        DEVICE.log.append(("close", self.name))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class AnalogMultiChannelWriter:
    def __init__(self, out_stream, auto_start=False):
        self.s = out_stream

    def write_many_sample(self, data, timeout=10):
        _check()
        DEVICE.ao_writes += 1
        if DEVICE.fail_on_write and DEVICE.ao_writes == DEVICE.fail_on_write[0]:
            code = DEVICE.fail_on_write[1]
            raise DaqError(f"simulated DAQmx error {code}", code)
        t = self.s.task
        if not hasattr(t, "_base"):
            t._base = DEVICE.total
        n = np.shape(data)[1]
        if not DEVICE.ao_running:
            # a real device would block (and finally time out) on a full, stopped buffer
            queued = DEVICE.total - t._base + n
            DEVICE.max_prestart_queue = max(DEVICE.max_prestart_queue, queued)
            buf = self.s.output_buf_size
            if buf is not None and queued > buf:
                raise DaqError("write timed out: buffer full before start", -200292)
        DEVICE.ao_written.append(np.array(data, copy=True))
        DEVICE.stream_ao_channels = [p for k, p in t.channels if k == "ao"]
        DEVICE.append_x(np.asarray(data)[DEVICE.loopback_row])
        DEVICE.log.append(("ao_write", np.shape(data)[1]))


class DigitalSingleChannelWriter:
    def __init__(self, out_stream, auto_start=False):
        self.s = out_stream

    def write_many_sample_port_uint32(self, data, timeout=10):
        _check()
        DEVICE.do_written.append(np.array(data, dtype=np.uint32, copy=True))


class AnalogMultiChannelReader:
    def __init__(self, in_stream):
        self.s = in_stream

    def read_many_sample(self, data, number_of_samples_per_channel, timeout=1):
        _check()
        assert data.flags["C_CONTIGUOUS"] and data.shape[1] == number_of_samples_per_channel
        n = number_of_samples_per_channel
        seg = DEVICE._x[DEVICE.ai_read : min(DEVICE.total, DEVICE.ai_read + n)]
        data[:, : len(seg)] = seg[None, :]
        DEVICE.ai_read += len(seg)
        return len(seg)


def install(monkeypatch, device_name="Dev1"):
    DEVICE.reset()
    DEVICE.name = device_name
    root = types.ModuleType("nidaqmx")
    root.Task = Task
    constants = types.ModuleType("nidaqmx.constants")
    for c in (AcquisitionType, LineGrouping, RegenerationMode, TerminalConfiguration):
        setattr(constants, c.__name__, c)
    system = types.ModuleType("nidaqmx.system")

    class _Sys:
        @staticmethod
        def local():
            return types.SimpleNamespace(devices=[types.SimpleNamespace(name=DEVICE.name)])

    class _Device:
        def __init__(self, name):
            self.name = name

        def reset_device(self):
            _check()
            DEVICE.resets += 1
            DEVICE.log.append(("reset_device", self.name))

    system.System = _Sys
    system.Device = _Device
    writers = types.ModuleType("nidaqmx.stream_writers")
    writers.AnalogMultiChannelWriter = AnalogMultiChannelWriter
    writers.DigitalSingleChannelWriter = DigitalSingleChannelWriter
    errors = types.ModuleType("nidaqmx.errors")
    errors.DaqError = DaqError
    root.errors = errors
    readers = types.ModuleType("nidaqmx.stream_readers")
    readers.AnalogMultiChannelReader = AnalogMultiChannelReader
    root.constants, root.system, root.stream_writers, root.stream_readers = constants, system, writers, readers
    for name, mod in [("nidaqmx", root), ("nidaqmx.constants", constants), ("nidaqmx.system", system),
                      ("nidaqmx.stream_writers", writers), ("nidaqmx.stream_readers", readers),
                      ("nidaqmx.errors", errors)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return DEVICE
