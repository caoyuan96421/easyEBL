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
        self.ao_written = []  # list of (2, n) arrays
        self.do_written = []
        self.static_ao = []
        self.static_do = []
        self.ao_running = False
        self.ai_read = 0
        self.tasks = []
        self.log = []

    @property
    def ao_x(self):
        return np.concatenate([a[0] for a in self.ao_written]) if self.ao_written else np.zeros(0)

    @property
    def do_words(self):
        return np.concatenate(self.do_written) if self.do_written else np.zeros(0, np.uint32)


DEVICE = FakeDevice()


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


class _OutStream:
    def __init__(self, task):
        self.task = task
        self.regen_mode = RegenerationMode.ALLOW_REGENERATION

    @property
    def total_samp_per_chan_generated(self):
        return DEVICE.ao_x.shape[0] if DEVICE.ao_running else 0


class _InStream:
    def __init__(self, task):
        self.task = task

    @property
    def avail_samp_per_chan(self):
        if not DEVICE.ao_running:
            return 0
        return DEVICE.ao_x.shape[0] - DEVICE.ai_read


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
        if "ao" in self.kinds():
            DEVICE.static_ao.append(list(data))
        else:
            DEVICE.static_do.append(data)

    def start(self):
        self.started = True
        DEVICE.log.append(("start", self.name))
        if "ao" in self.kinds():
            DEVICE.ao_running = True

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
        DEVICE.ao_written.append(np.array(data, copy=True))


class DigitalSingleChannelWriter:
    def __init__(self, out_stream, auto_start=False):
        self.s = out_stream

    def write_many_sample_port_uint32(self, data, timeout=10):
        DEVICE.do_written.append(np.array(data, dtype=np.uint32, copy=True))


class AnalogMultiChannelReader:
    def __init__(self, in_stream):
        self.s = in_stream

    def read_many_sample(self, data, number_of_samples_per_channel, timeout=1):
        x = DEVICE.ao_x
        n = number_of_samples_per_channel
        seg = x[DEVICE.ai_read : DEVICE.ai_read + n]
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

    system.System = _Sys
    writers = types.ModuleType("nidaqmx.stream_writers")
    writers.AnalogMultiChannelWriter = AnalogMultiChannelWriter
    writers.DigitalSingleChannelWriter = DigitalSingleChannelWriter
    readers = types.ModuleType("nidaqmx.stream_readers")
    readers.AnalogMultiChannelReader = AnalogMultiChannelReader
    root.constants, root.system, root.stream_writers, root.stream_readers = constants, system, writers, readers
    for name, mod in [("nidaqmx", root), ("nidaqmx.constants", constants), ("nidaqmx.system", system),
                      ("nidaqmx.stream_writers", writers), ("nidaqmx.stream_readers", readers)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return DEVICE
