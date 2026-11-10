# easyEBL

Prototype control software that turns a conventional SEM (target: Hitachi S‑3000N) into a
simple **e‑beam lithography** and **imaging** tool, driven by an NI‑DAQmx multifunction device
(e.g. USB‑6341):

* 2 × AO → external X/Y scan inputs (DBC port)
* 1 × hardware‑timed DO → beam blanker (plus optional hardware‑timed sync lines)
* AI → SEM video signal, sampled on the **same clock** as the deflection
* software‑timed DO → internal/external scan switch (the HAL's *external control* function)

Everything is written on top of a **hardware abstraction layer (HAL)** and ships with a
physics‑flavoured **SEM simulator**, so all subsystems, tests and GUIs run without hardware.

> ⚠️ **Status: prototype.** The NI‑DAQmx backend is written against `nidaqmx` 1.x and exercised
> against an in‑memory fake, but it has **not yet been validated on hardware**. Before the first
> real run, check every signal on an oscilloscope with the beam off, and make sure the blanker
> is wired **fail‑safe** (undriven = blanked).

## Install

The install scripts create a `.venv` next to the code and install easyEBL into it in editable
mode, with every extra (numpy, scipy, gdstk, PySide6, pyqtgraph, nidaqmx, pytest). They need
Python 3.10–3.13.

**Windows:** double‑click `install.bat`, or run it from a terminal:

```bat
install.bat                      :: create .venv and install everything
install.bat -RunTests            :: also run the test suite
install.bat -Recreate            :: delete and rebuild .venv
install.bat -InstallNIDriver     :: also install the NI-DAQmx driver (admin; DAQ PC only)
install.bat -Python C:\path\to\python.exe   -Extras gui,gds
run_demo.bat                     :: start the demo GUI on the simulator
run_demo.bat --backend nidaqmx --device Dev1
```

If the folder is inside Dropbox, the script marks `.venv` as ignored, so the environment is not
synced. Each PC builds its own `.venv`. Pass `-SyncVenv` to turn this off.

**Linux / macOS:** `./install.sh [--tests] [--recreate] [--extras all,test] [--python python3.12]`

**Manually:** `pip install -e ".[all,test]"`

## Quick start (simulator)

```python
from easyebl.core import FieldCalibration
from easyebl.hal import DAQManager, create_backend
from easyebl.imaging import RasterConfig, RasterImager
from easyebl.alignment import MarkerAligner
from easyebl.litho import LithoWriter, ExposureParameters, read_gds

mgr = DAQManager(create_backend("sim")).open()          # or create_backend("nidaqmx", cfg)
cal = FieldCalibration(field_size_um=100.0, v_full_scale=10.0)

# imaging
img = RasterImager(mgr.get_handle("imaging"), cal).acquire(RasterConfig(nx=512, ny=512, width_um=80))

# marker alignment (design → field transform)
aligner = MarkerAligner(RasterImager(mgr.get_handle("align"), cal), window_um=20)
res = aligner.align([(-35, -35), (35, -35), (35, 35), (-35, 35)], model="similarity")

# GDS lithography with per-polygon dose factors, placed with the alignment result
pattern = read_gds("design.gds")
params = ExposureParameters(beam_current_a=250e-12, area_dose_uC_cm2=300, area_step_um=0.02)
report = LithoWriter(mgr.get_handle("litho"), cal).expose(pattern, params, transform=res.transform)
```

## SEM imaging (USB-6363)

Double-click **`run_sem.bat`** (or `easyebl-sem --backend nidaqmx --device Dev1`, add
`--backend sim` for the simulator). The window looks and works like a commercial SEM screen:

* **Run / Freeze** (F5): Freeze finishes the frame; press again to stop at once. **Single** (F6).
* **Speed**: TV · Fast 1–2 · Slow 1–3 · Photo 1–3 (0.5–160 µs per pixel; F3/F4 step faster or slower). **Scan size**: 320×240 to 2560×1920.
* **Capture** (F9): photo scan at its own size, speed and integration. Saved automatically as a 16‑bit TIFF (raw volts), a PNG, a PNG with the data bar, and JSON metadata.
* **Contrast / Brightness**, **Auto C/B** (Ctrl+A, one-shot or continuous), and running **averaging**.
* **Beam shift**: arrow pad, drag the image, or double-click a point to centre it. It is clamped to the field.
* **Data bar** with the scale bar, field magnification and date. There is also a scan-line marker during slow scans.
* **Magnification, focus/WD, stigmation, alignment** are stubs (`easyebl/imaging/sem/optics.py`) until the SEM's RS‑232 control is implemented.
* **Advanced…**: sample rate, buffer/latency, video input range, flyback, pre-line, settle, lag compensation, field calibration, and save options.
* **External scan** (toolbar) switches the SEM between its own scan and this program (HAL external control).

The same engine can be driven from text commands. Use `run_sem_cli.bat`, or
`python -m easyebl.imaging.cli -c "run; wait frames=5; capture shot frames=2 res=2560x1920 speed=Photo1; quit"`.
You can also start the GUI with `--console` to type commands next to it.

### How the imaging engine uses the hardware

* **One fixed, fast clock.** The 6363 runs at 2 MS/s with one video channel. Dwell is a whole number of clock
  ticks, and every tick is digitised and averaged into its pixel. Slow scans therefore gain SNR, and changing
  the speed never restarts the hardware.
* **One continuous stream with a small buffer.** The output buffer holds only `latency_s` of data (default 100 ms) and
  is topped up in ⅛ pieces. Speed, size, shift and rotation changes are applied at the next line boundary, so the
  hardware never stops. A change reaches the beam after about the buffer depth, instead of after a task restart plus up to 1 s of queued data.
* **Steady frames.** AI is read every ⅛ buffer, so frames are published within milliseconds of their last line.
  Frame times come from the sample clock and are exact.
* **Own thread, published outputs.** The engine (`easyebl/imaging/engine/`) owns the DAQ handle in a high-priority
  scan thread. Commands from the GUI, the console or scripts are validated in the caller's thread and applied by the scan
  thread. Images are published to a latest-frame slot and events go to bounded queues, so no consumer can stall
  scanning or grow memory.
* **Self-healing.** If the host stalls longer than the buffer, the DAQ underflows. The HAL then blanks and parks the beam, and
  the engine doubles the buffer and restarts on its own (at most 5 times in a row).

**Dummy-image test (AO2 → AI0).** The HAL exposes AO2/AO3 as auxiliary outputs **`aux0`/`aux1`**
(`handle.set_aux_output("aux1", 1.5)`). They are dummy channels on boards without them. With
`phantom_output="aux0"` (`--phantom` on `run_sem.bat` / `run_sem_cli.bat`) every scan also drives a
test chart on AO2, computed from the commanded beam position on the same sample clock. Wire AO2 → AI0+ (the default RSE input mode needs nothing else; in DIFF mode also wire AO GND → AI0−),
and the live image shows the chart. Zoom and beam shift move across it, and `--phantom-noise 0.2` adds white noise so
averaging versus scan speed becomes visible. `run_phantom_test.bat` measures fidelity, registration (AI
latency), field of view, beam shift and noise versus dwell against the expected image, and saves a report and PNGs.
Three AO channels limit the 6363 to 1.54 MS/s, which the HAL accounts for.

**Lost USB connection.** If the 6363 is unplugged or switched off, the current call fails with
`DeviceLostError`. Operations then fail at once instead of hanging, and the beam state is reported as *unknown*.
A watchdog probes the device every 0.5 s with a harmless one-sample AI read. When the device answers again, the HAL resets
it, writes the safe state (blanked, parked), selects the SEM's internal scan, and then re-applies external control and
static lines for handles that still hold them. The engine shows **DISCONNECTED**, keeps its mode and settings, and
resumes scanning by itself. A capture in progress fails, and an interrupted single frame is rescanned. The programs also start
without the device and connect when it appears. Simulator tests: `tests/system/test_disconnect_system.py`.
Manual hardware check: `run_unplug_test.bat` (pull and re-plug the cable when prompted).

Hardware robustness tests: `run_hw_imaging_tests.bat` (or `pytest tests/hardware --hw Dev1 -s`). They check
steady fps at 2 MS/s, change latency for each buffer size, start/stop and random aborts, a 20 s settings storm,
GIL stalls shorter and longer than the buffer, a 2560×1920 capture, cancelling a slow photo scan, refused concurrent access,
close while live, and a memory soak. A report goes to `scripts/logs/`.

### Raster debug GUI

Double-click `run_imaging.bat`, or run
`python -m easyebl.imaging.gui --backend nidaqmx --preset usb6363 [--field-um 100]`.

* **External scan control** switches the SEM to DAQ scanning (HAL external control).
  Unchecking it returns the SEM to internal scan.
* **Live** runs one continuous hardware-timed stream. The frame waveform is looped through the
  DAQ output buffer, so frames follow each other with no restart gap. Changing any setting
  while live restarts the stream.
* **Single scan** takes one frame. **Average frames** sets a running average over about N frames.
* **Save image** writes a PNG (contrast-stretched) and an `.npy` file with the raw values.
* `run_hw_test.bat` is a headless check of the board: static lines, single scans and live
  streaming at several rates. It logs to `scripts/logs/console.txt`.

## Demo / debug GUIs

```bash
easyebl-demo                              # all tools as tabs, sharing one DAQ manager
python -m easyebl.hal.gui                 # HAL: handles/claims, static beam, test waveforms
python -m easyebl.imaging.gui             # raster imaging
python -m easyebl.alignment.gui           # automatic marker alignment (sim: misplace specimen)
python -m easyebl.litho.gui_vector        # single polygon (raster fill), lines & dots
python -m easyebl.litho.gui_gds [f.gds]   # GDS lithography with dose mapping
python -m easyebl.litho.gui_dose          # dose / dwell calculator
# add  --backend nidaqmx --device Dev1 [--config daq.json]  for hardware
```

On the simulator, the litho GUIs show the resulting **dose map**, and the alignment GUI
compares the fitted transform with the deliberately misplaced specimen.

## Architecture

```
easyebl/
  core/       AffineTransform, FieldCalibration (µm ↔ V), OutputChunk streams
  hal/        DAQConfig, DAQBackend (abstract), DAQManager + DAQHandle, SimulatedBackend, NIDAQBackend
  imaging/    RasterConfig → RasterPlan (waveforms) → RasterImager (+ LineAssembler for AI)
    engine/   threaded ImagingEngine: settings, source, program, assembler, frames, core
    sem/      SEM GUI: window, view (data bar), panels, advanced dialog, optics stubs
    cli.py    text console; export.py  TIFF/PNG/JSON writers; launch.py  engine from CLI args
  alignment/  templates + NCC matching, transform fitting, MarkerAligner
  litho/      dose maths, geometry, fills, VectorScanCompiler, LithoWriter, GDS import
  gui/        shared widgets (DataclassForm, ImageView, WaveformView) and the launcher
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale (timing model,
handle sharing, lag handling, dose quantisation).

### Key behaviours

| Subsystem | Highlights |
|---|---|
| **HAL** | One hardware‑timed stream at a time per device; per‑module handles; exclusive claims for relays and other resources; automatic **blank + park** after every stream, including errors and aborts; voltage, rate and channel validation; event listeners for GUIs. |
| **Imaging** | Pixel staircase with configurable **pre‑line** (overscan ramp or hold, optionally blanked), **flyback** (linear/cosine/step, blanked by default), frame settle, bidirectional scans, scan rotation, frame averaging, per‑pixel settle discard, fractional **lag compensation**, live rows, sync marker lines for EDS. |
| **Alignment** | Sub‑pixel normalised cross‑correlation; cross/square/image templates; **bidirectional averaging** cancels deflection/video lag (the lag is reported as a health check); translation/rigid/similarity/affine fits; rejection of weak or inconsistent marks. |
| **Vector litho** | Grid‑aligned raster fill (even–odd, so holes and keyhole polygons work), serpentine or unidirectional fill along x or y; single‑pass lines; dots; blanked jumps with distance‑dependent settle; step‑and‑blank mode; **blanker‑delay compensation**; dose **error diffusion**; streaming in chunks; abort. |
| **GDS** | Flattened hierarchy and arrays; layer/datatype filter; dose factor from a GDS **property** (e.g. `"1.25"` or `"dose=1.25"`) or a **datatype/layer → factor table** (dose classes from fractured/PEC'd exports such as BEAMER); paths become polygons. |
| **Dose utility** | Area/line/dot dose ↔ dwell, step from field and points, DAC resolution, clock quantisation, write time, plus warnings. |

## Tests

```bash
pytest                    # everything (unit + subsystem + GUI smoke tests, ~40 s)
pytest tests/unit         # fast unit tests
pytest -m system          # subsystem tests: full pipelines on the simulator
pytest -m gui             # offscreen Qt smoke tests
```

Subsystem tests check physical outcomes on the simulator. For example:
* images match the specimen, and the measured lag shift and its compensation are correct;
* alignment recovers a known transform to well below a pixel, and a unidirectional scan is biased;
* exposed dose is uniform and correct inside polygons, and **zero** outside;
* dose factors from GDS properties or datatypes scale the delivered dose;
* a late blanker leaves dose trails, and compensating for it removes them;
* imaging, alignment and lithography share one device without conflicts.

The NI backend is tested against an in‑memory fake of `nidaqmx`. These tests cover sample
alignment of AO/DO/AI, the clock routing, the start order, blanker polarity, the safe tail and abort.

## Hardware notes

* Default `DAQConfig`: `ao0/ao1` = X/Y (±10 V), `port0/line0` = blanker (active high), AI `ai0`
  = video (single-ended RSE; switch to DIFF in Advanced… or with `--ai-terminal DIFF` when AI0− is wired), `port1/line0` = external control. Save a config with
  `DAQConfig(...).to_json()` and pass it with `--config`.
* **External control** (SEM internal/external scan) is a HAL function. Applications call
  `handle.set_external_control(True/False)` and never see the wires. The wires and their polarity are set
  in the config, e.g. `external_control_lines={"PFI6": True, "PFI7": True}` (the 6363 preset: both HIGH =
  external). `{"PFI6": True, "PFI7": False}` would make PFI7 active-low. A list uses
  `external_control_active_high` (default True) for every wire. The HAL selects **internal** scan when it opens
  the device, when the handle holding external control closes, when the manager closes, and at program exit.
  Switching is refused while a stream runs and always blanks the beam first. A hard crash cannot run cleanup, and
  X‑series lines keep their level until the next start or a device reset. After a reset the lines are
  high‑impedance, so give the relay inputs a **pull-down** so that undriven = internal and the SEM's own controls always work.
* Keep **port0** for hardware‑timed lines only: the DO task drives the whole port, and the
  config validation rejects static lines placed there.
* Choose sample rates that divide 100 MHz (e.g. 400 k or 500 kS/s). Other rates are coerced by
  the device, and easyEBL warns when that happens.
* Fit a hardware **blanker watchdog** (blank unless a heartbeat is present). Software cannot
  guarantee blanking if the host stalls mid‑stream.
* The S‑3000N DBC input is scaled by the SEM magnification. Calibrate `FieldCalibration` for
  each magnification you write at, and match the voltage range to the SEM's own X‑OUT/Y‑OUT signals.
