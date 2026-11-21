# easyEBL

Prototype control software that turns a conventional SEM (target: Hitachi S‑3000N) into a
simple **e‑beam lithography** and **imaging** tool, driven by an NI‑DAQmx multifunction device
(e.g. USB‑6341):

* 2 × AO → external X/Y scan inputs (DBC port)
* 1 × hardware‑timed DO → beam blanker (plus optional hardware‑timed sync lines)
* AI → SEM video signal, sampled on the **same clock** as the deflection
* software‑timed DO → internal/external scan switch (the HAL's *external control* function)

It also does **EDS** (energy-dispersive X-ray spectroscopy) with an Amptek **DP5** pulse
processor: spectra, AutoID, standardless quantification, line scans and spectrum maps, in an
Oxford-AZtec-like window that shares the DAQ with SEM imaging (see [EDS](#eds-amptek-dp5)).

The SEM column and the five-axis **stage** are controlled over the S-3000's **RS-232** remote
interface (see [Microscope](#microscope-hitachi-s-3000-rs-232)).

Everything is written on top of a **hardware abstraction layer (HAL)** and ships with a
physics‑flavoured **SEM simulator**, so all subsystems, tests and GUIs run without hardware.

> ⚠️ **Status: prototype.** The NI‑DAQmx backend is written against `nidaqmx` 1.x and exercised
> against an in‑memory fake, but it has **not yet been validated on hardware**. Before the first
> real run, check every signal on an oscilloscope with the beam off, and make sure the blanker
> is wired **fail‑safe** (undriven = blanked).

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                   # the easyebl package and the `easyebl` command
```

`requirements.txt` lists every prerequisite (numpy, scipy, PySide6, pyqtgraph, gdstk, nidaqmx,
pyusb, libusb-package, pyserial, pytest, pytest-qt).  The pyproject extras (`pip install -e ".[all,test]"`)
install the same set.  The hardware also needs the **NI-DAQmx** runtime from NI and, for the DP5,
a libusb/WinUSB driver.

## Running

There is one program:

```bash
easyebl                  # the hardware configured in Hardware → Configure… (saved per user)
easyebl --sim            # this session on simulators only (the saved configuration is kept)
easyebl --open sem,eds   # also open these windows: sem, eds, microscope, vector, gds, dose,
                         # alignment, raster, hal
python -m easyebl        # the same as easyebl
```

It opens a **small menu window** that shows, for the scan DAQ, the EDS pulse processor and the
microscope, whether it is connected (green), simulated (blue), waiting or needing attention
(orange) or failed (red, with the reason), and puts an **icon in the system tray**.

* **Open** starts a subsystem: SEM imaging, EDS, Microscope (column & stage), and under *Tools*
  vector and GDS lithography, the dose calculator, alignment, the raster debug GUI and HAL debug.
  A subsystem whose hardware is missing is greyed out (the tooltip says why).
* **Hardware → Configure…** chooses each device and its address: NI-DAQmx device name or the
  simulator, wiring preset or a DAQConfig JSON, start field, sample clock, buffer latency, video
  input mode; DP5 (USB), its mock device or none; the microscope's serial port (listed), the
  simulator or none, and the baud rate.  *Apply & reconnect* saves it to
  `%APPDATA%\easyEBL\system.json` and reconnects; *All simulators* fills in the simulators.
  **Reconnect all** reopens everything with the same settings.
* **The hardware stays connected in the background.**  All subsystems share the one backend (one
  DAQ manager, one DP5, one serial port), so SEM and EDS share the DAQ through the HAL lease.
  Closing a subsystem window, or the menu window itself, does not disconnect anything.
* **Tray icon:** click it to bring the menu back; right-click it for the menu, any subsystem,
  *Hardware…* and **Shut down easyEBL**, which closes every window and disconnects all hardware.
  The icon shows a coloured dot when a device needs attention.
* Only one easyEBL runs per user: starting it again brings the running menu to the front (and
  opens the windows named by `--open`).

In VS Code select the `.venv` interpreter (*Python: Select Interpreter*) and pick *easyEBL* or
*easyEBL (simulators)* in *Run and Debug* (F5).

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

Open → *SEM imaging*.  The window looks and works like a commercial SEM screen:

* **Run / Freeze** (F5): Freeze finishes the frame; press again to stop at once. **Single** (F6).
* **Speed**: TV · Fast 1–2 · Slow 1–3 · Photo 1–3 (0.5–160 µs per pixel; F3/F4 step faster or slower). **Scan size**: 320×240 to 2560×1920.
* **Capture** (F9): photo scan at its own size, speed and integration. Saved automatically as a 16‑bit TIFF (raw volts), a PNG, a PNG with the data bar, and JSON metadata.
* **Contrast / Brightness**, **Auto C/B** (Ctrl+A, one-shot or continuous), and running **averaging**.
* **Beam shift**: arrow pad, drag the image, or double-click a point to centre it. It is clamped to the field.
* **Data bar** with the scale bar, field magnification and date. There is also a scan-line marker during slow scans.
* **Magnification, focus/WD, stigmation, gun alignment** go to the SEM over RS-232; the
  mouse wheel on the image steps the magnification and the field calibration follows it (see
  [Microscope](#microscope-hitachi-s-3000-rs-232)).  The data bar shows the SEM's kV and WD, and captures
  store the column and stage readbacks in their metadata.  **Microscope…** opens the stage window.
* **Advanced…**: sample rate, buffer/latency, video input range, flyback, pre-line, settle, lag compensation, field calibration, and save options.
* **External scan** (toolbar) switches the SEM between its own scan and this program (HAL external control).

The engine can also be driven by text commands from Python
(`easyebl.imaging.cli.ImagingShell(engine).run_commands("run; wait frames=5; capture shot frames=2")`).

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

**Auxiliary outputs.** The HAL exposes AO2/AO3 as static auxiliary outputs **`aux0`/`aux1`**
(`handle.set_aux_output("aux1", 1.5)`). They are dummy channels on boards without them.

**Lost USB connection.** If the 6363 is unplugged or switched off, the current call fails with
`DeviceLostError`. Operations then fail at once instead of hanging, and the beam state is reported as *unknown*.
A watchdog probes the device every 0.5 s with a harmless one-sample AI read. When the device answers again, the HAL resets
it, writes the safe state (blanked, parked), selects the SEM's internal scan, and then re-applies external control and
static lines for handles that still hold them. The engine shows **DISCONNECTED**, keeps its mode and settings, and
resumes scanning by itself. A capture in progress fails, and an interrupted single frame is rescanned. The programs also start
without the device and connect when it appears. Simulator tests: `tests/system/test_disconnect_system.py`.

### Raster debug GUI

Open → Tools → *Raster imaging (debug)*.

* **External scan control** switches the SEM to DAQ scanning (HAL external control).
  Unchecking it returns the SEM to internal scan.
* **Live** runs one continuous hardware-timed stream. The frame waveform is looped through the
  DAQ output buffer, so frames follow each other with no restart gap. Changing any setting
  while live restarts the stream.
* **Single scan** takes one frame. **Average frames** sets a running average over about N frames.
* **Save image** writes a PNG (contrast-stretched) and an `.npy` file with the raw values.

## EDS (Amptek DP5)

Open → *EDS*.  In *Hardware → Configure…* the pulse processor is the DP5 over USB, or the DP5
driver's mock device, which counts a ⁵⁵Fe-like spectrum (Mn Kα/Kβ) independent of the beam
position (the simulator configuration).

**Workflow (Point & ID / mapping):** acquire an electron image (or take the SEM's latest
frame), then click a point or drag a rectangle/ellipse – the spectrum is acquired there,
elements are identified (AutoID, or pick them on the periodic table) and quantified
(standardless ZA, optional oxygen by stoichiometry).  Draw a map area or a line for spectrum
maps and line scans; maps can be shown as window, fitted (deconvolved) or quantitative
(wt %) maps and clustered into phases.  Spectra save as EMSA `.msa`, maps as `.npz`.
**Tools → Energy calibration** fits channel → keV from the peaks of a standard (Cu L + Cu Kα,
Al + Cu, Mn Kα/Kβ …; auto-assigned or clicked), reports residuals and the detector FWHM, and
stores the result per user.  **Tools → Pulse processor** sets the DP5 shaping preset and
parameters.

**Sharing the DAQ:** every EDS job takes the HAL's *exclusive lease*: the live SEM image pauses
(state `PAUSED`), the SEM is switched to external scan for the job, and imaging resumes by
itself afterwards; a capture or an exposure is never interrupted (the lease waits).  Two
separate programs cannot drive the same NI device (an OS lock gives a clear "in use" error),
which is why there is one easyEBL program for every subsystem.

**How maps are acquired:** the DP5 has no list mode, so maps are step scans.  Every pixel gets
a slot (settle + dwell + ~15 ms) scheduled ahead on the DAQ sample clock; when the beam reaches
it, the MCA counts for a preset real time (`PRER`, 1 ms resolution) and is read out and cleared
while the beam moves on.  A slot the PC reaches too late is redone at the end.  On the
USB-6363 + DP5 the overhead is ≈ 22 ms per pixel (64×48 at 50 ms dwell ≈ 3.6 min).

**DP5 notes (FW 6.10.02):** a preset real time stops counting and sets "preset done" but leaves
"MCA enabled" set (the HAL disables it before read-and-clear).  If the factory fast threshold
(`THFA` 30) misses pulses (ICR < OCR), run *Auto thresholds* with the beam off; the live time is
clamped to the real time meanwhile.

## Microscope (Hitachi S-3000, RS-232)

Open → *Microscope*; the port and baud rate are set in *Hardware → Configure…*.

The SEM text protocol and the stage protocol share **one serial port** (9600 8N1; the SEM relays
the stage lines after `#STAGE ON`, which is sent automatically).  The driver
(`easyebl/microscope/s3000`) is vendored from *hitachi_s3000_control_lib*, which was verified on the
S-3000. On top of it:

* **`MicroscopeLink`** owns the port and runs one request at a time in priority order: SEM settings,
  then stage moves, then reads, then background polling.  A SEM setting waits at most for the
  transaction already on the wire (replies take 0.25–1 s).  After a reply times out the link stops
  until you check the SEM and press **Resync** (nothing is ever resent automatically).
* **`SEMControl`** (column HAL): HV setpoint / on / off, magnification (the 65-step list), working
  distance, coarse/fine focus, stigmator, condenser coarse/fine (raw 0–99), SEM display contrast /
  brightness / scan mode / freeze, detector, AFC / ASF / ABC / AGA.  Readbacks are cached (instant for
  the GUIs), refreshed after each setting and by a slow MAG / HV / WD poll (default 10 s) so changes
  on the SEM panel are seen.  Repeated settings coalesce: ten wheel steps during one transaction
  become one `MAG` with the final value.
* **`StageControl`** (stage HAL): X, Y, Z in µm and R, T in degrees.  Positions are polled at the
  lowest priority, **at most every 20 s** (one `R3` per controller), never while moving.  Moves are
  absolute or relative and wait for the motors' completion events. **STOP** never waits in the queue:
  S0 is sent at once, also while a long SEM command holds the port, and a move that is being sent
  at that moment is halted too.  Negative coordinates stay disabled until their encoding is checked
  on the instrument.
* **Wiring.** Imaging and EDS use the column through `MicroscopeControl`.  The DAQ field follows the
  magnification as `field = K / magnification`.  `K` is learnt from the current field calibration
  the first time and re-learnt when the field is calibrated in *Advanced…*; it is stored in
  `%APPDATA%\easyEBL\microscope.json`.  Zooming keeps the scan's share of the field.  EDS takes the
  SEM's kV as the nominal beam energy, and spectra and images carry the column and stage readbacks.
* The window: HV, magnification, focus / WD, stigmator, condenser and SEM display on the left; on
  the right a big **STOP ALL** (also *Esc*), a position table with read times, absolute / relative
  moves, X/Y/Z/R/T jogs behind an *Enable stage motion* interlock, and the raw TX/RX log.  On the
  real instrument, HV switching, automatic routines and moves ask for confirmation.
* **Beam presets** (*Beam presets…* in the toolbar or *Presets…* in the beam group): any number of
  named settings of beam 1 coarse / fine, stigmator X / Y and contrast / brightness.  *Save current…*
  reads the six values fresh from the SEM; *Recall* (or double-click) sends them, after checking all
  six first.  Names and values can be edited in the table, and rows reordered or deleted.  Presets
  are stored in `microscope.json`.  Only beam 1 is used: the protocol has no command to select
  between the SEM's two beam-current memories.

The serial port is opened once by the easyEBL backend and shared by the SEM, EDS and microscope
windows.  Only the simulator has been used so far; see
*hitachi_s3000_control_lib/docs/commissioning.md* for what is verified on the instrument.

## Lithography and debug tools

Open → Tools: *Vector lithography* (single polygon with raster fill, lines and dots), *GDS
lithography* (dose mapping), *Dose calculator*, *Alignment* (automatic marker alignment; on the
simulator it misplaces the specimen), *Raster imaging (debug)* and *HAL debug* (handles, claims,
static beam, test waveforms).  They use the same DAQ as SEM imaging.

On the simulator, the litho GUIs show the resulting **dose map**, and the alignment GUI
compares the fitted transform with the deliberately misplaced specimen.

## Architecture

```
easyebl/
  core/       AffineTransform, FieldCalibration (µm ↔ V), OutputChunk streams
  hal/        DAQConfig, DAQBackend (abstract), DAQManager + DAQHandle, SimulatedBackend, NIDAQBackend
  imaging/    RasterConfig → RasterPlan (waveforms) → RasterImager (+ LineAssembler for AI)
    engine/   threaded ImagingEngine: settings, source, program, assembler, frames, core
    sem/      SEM GUI: window, view (data bar), panels (column via easyebl.microscope), advanced dialog
    cli.py    text commands (ImagingShell); export.py  TIFF/PNG/JSON writers
  hal/dp5/    vendored Amptek DP5 USB driver (+ mock device); hal/mca.py pulse-processor HAL
  hal/devlock.py  process lock on the NI device
  microscope/ Hitachi S-3000 RS-232: link (shared port, priorities), sem (column HAL), stage (stage HAL),
              coupling (field ↔ magnification), gui; s3000/ vendored driver + simulator
  xray/       X-ray lines/edges/attenuation (xraydb tables) and the generation model
tools/        gen_xray_data.py: regenerates easyebl/xray/data/xray_data.npz (needs xraydb)
tests/        unit/, system/ (simulator pipelines), gui/ (offscreen Qt); data/ real DP5 spectra
  eds/        EDSController (spectra, images, maps), spectrum/map data, analysis/, gui/
  alignment/  templates + NCC matching, transform fitting, MarkerAligner
  litho/      dose maths, geometry, fills, VectorScanCompiler, LithoWriter, GDS import
  gui/        shared widgets (DataclassForm, ImageView, WaveformView) for the tools
  app/        the easyEBL program: SystemConfig, SystemBackend (all hardware), menu window + tray
```

See [`docs/architecture.md`](docs/architecture.md) for the design rationale (timing model,
handle sharing, lag handling, dose quantisation).

### Key behaviours

| Subsystem | Highlights |
|---|---|
| **EDS** | Point/area/live-time spectra, electron image, line scans, scheduled step-scan maps; SNIP background, calibration, AutoID (escape/sum/flank artefacts rejected), non-negative least-squares fit with refined resolution, iterative standardless ZA quant, window/fitted/quant/phase maps. |
| **HAL** | One hardware‑timed stream at a time per device; exclusive **lease** with preemption of the live image; per‑module handles; exclusive claims for relays and other resources; automatic **blank + park** after every stream, including errors and aborts; voltage, rate and channel validation; event listeners for GUIs. |
| **Imaging** | Pixel staircase with configurable **pre‑line** (overscan ramp or hold, optionally blanked), **flyback** (linear/cosine/step, blanked by default), frame settle, bidirectional scans, scan rotation, frame averaging, per‑pixel settle discard, fractional **lag compensation**, live rows, sync marker lines for EDS. |
| **Alignment** | Sub‑pixel normalised cross‑correlation; cross/square/image templates; **bidirectional averaging** cancels deflection/video lag (the lag is reported as a health check); translation/rigid/similarity/affine fits; rejection of weak or inconsistent marks. |
| **Vector litho** | Grid‑aligned raster fill (even–odd, so holes and keyhole polygons work), serpentine or unidirectional fill along x or y; single‑pass lines; dots; blanked jumps with distance‑dependent settle; step‑and‑blank mode; **blanker‑delay compensation**; dose **error diffusion**; streaming in chunks; abort. |
| **GDS** | Flattened hierarchy and arrays; layer/datatype filter; dose factor from a GDS **property** (e.g. `"1.25"` or `"dose=1.25"`) or a **datatype/layer → factor table** (dose classes from fractured/PEC'd exports such as BEAMER); paths become polygons. |
| **Dose utility** | Area/line/dot dose ↔ dwell, step from field and points, DAC resolution, clock quantisation, write time, plus warnings. |

## Tests

```bash
pytest                    # everything (unit + subsystem + GUI smoke tests, ~2 min)
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
