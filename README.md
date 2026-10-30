# easyEBL

Prototype control software that turns a conventional SEM (target: Hitachi S‑3000N) into a
simple **e‑beam lithography** and **imaging** tool, driven by an NI‑DAQmx multifunction device
(e.g. USB‑6341):

* 2 × AO → external X/Y scan inputs (DBC port)
* 1 × hardware‑timed DO → beam blanker (plus optional hardware‑timed sync lines)
* AI → SEM video signal, sampled on the **same clock** as the deflection
* software‑timed DO → internal/external scan relay

Everything is written on top of a **hardware abstraction layer (HAL)** and ships with a
physics‑flavoured **SEM simulator**, so all subsystems, tests and GUIs run without hardware.

> ⚠️ **Status: prototype.** The NI‑DAQmx backend is written against `nidaqmx` 1.x and exercised
> against an in‑memory fake, but it has **not yet been validated on hardware**. Before the first
> real run, check every signal on an oscilloscope with the beam off, and make sure the blanker
> is wired **fail‑safe** (undriven = blanked).

## Install

```bash
pip install -e ".[all,test]"      # numpy, scipy, gdstk, PySide6, pyqtgraph, nidaqmx, pytest
```

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
  = video (differential), `port1/line0` = scan relay. Save a config with
  `DAQConfig(...).to_json()` and pass it with `--config`.
* Keep **port0** for hardware‑timed lines only: the DO task drives the whole port, and the
  config validation rejects static lines placed there.
* Choose sample rates that divide 100 MHz (e.g. 400 k or 500 kS/s). Other rates are coerced by
  the device, and easyEBL warns when that happens.
* Fit a hardware **blanker watchdog** (blank unless a heartbeat is present). Software cannot
  guarantee blanking if the host stalls mid‑stream.
* The S‑3000N DBC input is scaled by the SEM magnification. Calibrate `FieldCalibration` for
  each magnification you write at, and match the voltage range to the SEM's own X‑OUT/Y‑OUT signals.
