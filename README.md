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
easyebl --open litho     # bring this forward: sem, eds, litho (tabs) or microscope (side bar)
python -m easyebl        # the same as easyebl
```

It opens the **main window** (laid out for a 1920 × 1080 screen, one font size throughout) – the one
window of the program, which runs in the foreground:

* **Centre – tabs SEM, EDS, Lithography**: the SEM image, the EDS electron image and spectrum,
  the lithography stage plot – each with its tools at the top (icon, text under it):
  - *SEM*: Run / Stop, Single, Capture, Record, Full FoV, Autofocus (fine / coarse / with stigmation in its
    menu), Auto stigmation, Auto gun align (the SEM's AGA), **Auto C/B** (the SEM's ABC), Crosshair, Scan
    line, Field inset, Advanced…, Full screen;
  - *EDS*: the electron-image tools Pan, Point, Rectangle, Ellipse, Line, Map area;
  - *Lithography*: Zoom to pattern, Move pattern, Move stage, Color by Field, Color by Dose,
    Writing order, Show dots.
* **Left – the microscope**: *Column* (one HV ON / OFF button, exposure field size and
  magnification, focus and stigmator, beam current, SEM display), *Stage* (STOP ALL AXES,
  position – X / Y / Z on one row, R / T on the next –, jog, and the lithography's **Stage
  positions**: the list on the left, ΔZ, Record WD, Go to in a column on the right) and the **beam
  monitor** (its settings on the left, the screen on the right; the scan monitor described below).
* **Right – the active tab's settings**: SEM scan / capture / image / field of view / column; EDS acquire /
  map / composition with the **Data Manager** under them; lithography chips and exposure schedule.
* **Bottom – three side by side**: **Messages** of every subsystem in one list (the source in its
  colour, warnings orange, errors red); *Calibration* (per field size: magnification, nominal WD,
  WD1, Z_cal, the stored beam and field calibrations), *Alignment* (each chip: Global Align result,
  height) and the **SEM view** (a small live SEM image with Zoom in / out, Run / Stop, Single and
  Auto C/B; drag = beam shift, wheel = zoom, a click answers a marker pick; its data bar shows the
  total magnification and the scale bar.  A marker pick asked by the lithography brings it forward,
  outlined in magenta – click, Confirm or Cancel here or in the SEM tab); the **EDS live spectrum**.
* **Menus**: *File* (the active tab's file actions, e.g. the EDS's Open / Save; **Shut down easyEBL**,
  Ctrl+Q) · *Hardware* (Configure…, Reconnect all) · the active tab's own menus (e.g. the
  Lithography's Schedule, Step, Calibration, View, Stage) · *Panels* (show / hide the side bars, reset
  the layout) · *Help* (About easyEBL).  A tab's keyboard shortcuts (F5, Ctrl+S, …) work only while it
  is shown.
* **Esc stops everything**: every stage axis, a running exposure schedule (and a marker pick), EDS
  acquisitions, a running auto function and the SEM scan.
* The side bars can be moved, floated and resized; the layout is saved in
  `%APPDATA%\easyEBL\mainwindow.ini`.  **Closing the window shuts easyEBL down** (it asks first;
  the hardware is disconnected).
* The status bar has a light for the scan DAQ, the EDS pulse processor and the microscope:
  connected (green), simulated (blue), waiting or needing attention (orange) or failed (red; the
  tooltip says why).

* **Hardware → Configure…** chooses each device and its address: NI-DAQmx device name or the
  simulator, wiring preset or a DAQConfig JSON, scan orientation (invert / swap X/Y), sample clock,
  buffer latency, video input mode; DP5 (USB), its mock device or none; the microscope's serial port (listed), the
  simulator or none, and the baud rate.  *Apply & reconnect* saves it to
  `%APPDATA%\easyEBL\system.json` and reconnects; *All simulators* fills in the simulators.
  **Reconnect all** reopens everything with the same settings.
* **Beam monitor** (the left side bar): the X/Y deflection as an oscilloscope in XY mode with
  persistence, over the full ±10 V field, in the same image coordinates as the other windows (the
  scan orientation's invert X/Y and swap X/Y are undone).  A bright dot always marks where the beam
  is now (solid: beam on, hollow ring: blanked), following the samples the DAQ has actually generated
  and every static move.  It is fed with exactly the samples written to the DAQ;
  blanked samples are not plotted.  Resolution (default 256²) and persistence time (∞ = until
  *Clear*) are saved in `system.json`.  The stream hands samples off through a lock-free ring buffer
  (decimated or skipped when full, never waited for); binning and drawing run in their own thread.
* All subsystems share the one backend (one DAQ manager, one DP5, one serial port), so SEM and
  EDS share the DAQ through the HAL lease.
* Only one easyEBL runs per user: starting it again brings the running window to the front (and
  shows what `--open` names).

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

Every exposure streams with the SEM on **external scan**. The writer claims the switch on its handle for the
stream (the HAL blanks the beam before switching) and releases it afterwards, also when the stream fails. If
the SEM is on external scan already, the switch is left alone. A Lithography schedule also reserves the DAQ
(an exclusive lease) for each field, so the imaging engine cannot scan over the resist meanwhile.

## SEM imaging (USB-6363)

Open → *SEM imaging*.  The window looks and works like a commercial SEM screen:

* **Run / Freeze** (F5): Freeze finishes the frame; press again to stop at once. **Single** (F6).
* **Speed**: TV · Fast 1–2 · Slow 1–3 · Photo 1–3 (0.5–160 µs per pixel; F3/F4 step faster or slower). A
  speed preset also sets the pre-line, flyback and settling times; they are shown in the status bar, not
  adjustable. **Pixels**: 320×240 to 2560×1920, or *Custom…* – any size up to 16384 × 16384 (presets
  4096², 5000², 8000², 10000²; the dialog shows the frame time and the image memory).  The capture size
  has *Custom…* too, for ultra-high-resolution photos.  Frames are assembled and published on a thread of
  their own, so whole-image work (an 8000 × 8000 frame is 256 MB) never holds up the DAQ output buffer;
  partial frames of a large scan are published less often (copying kept under 10 % of the time).
* **Capture** (F9): photo scan at its own size, speed and integration. Saved automatically as a 16‑bit TIFF (raw volts), a PNG, a PNG with the data bar, and JSON metadata.
* **Record**: a video of the image as displayed (the Image contrast / brightness levels, the data bar
  under it; no crosshair or other overlays), in real time at 5, 10 or 25 frames/s (the button's arrow;
  10 by default): a slow scan shows line by line, a frozen image stands still.  While recording the
  button shows **REC m:ss**; the frames go to a temporary file.  Record again stops and asks where to
  save: **MP4** (H.264, Qt Multimedia – no extra package) or **AVI** (uncompressed 8-bit grey, OpenDML,
  so recordings beyond 1 GB play); cancelling asks whether to discard the recording.
* **Image** group: display **contrast / brightness**, **Auto levels** (Ctrl+A, one-shot or continuous: the
  display stretched to the image), and running **averaging**.
* **Auto C/B** (toolbar, also in the small SEM view of the main window) runs the **SEM's own ABC**
  (`ABC ON`).  The SEM needs its own scan for it, so our scan stops first (the engine gives the scan back to
  the SEM); the S-3000 answers `ABC ON` when it has finished – that reply ends the request (at most 240 s) –
  then contrast / brightness are read back, a live scan resumes (else one Single scan is taken) and **Auto
  levels** (the Image group) runs on the first image.  Esc meanwhile: the SEM finishes its ABC, our scan
  stays stopped and Auto levels is skipped.
* **Auto gun align** runs the SEM's AGA the same way (no question first): our scan stops, the SEM has its scan
  until it answers `AGA ON`, then a live scan resumes (no Auto levels).  The column panel's *Auto gun
  alignment* button does the same in the main window.
* **Auto C/B, Auto gun align, Autofocus and Auto stigmation wait in a modal window**: the program is blocked
  until they end, the window shows the progress and closes by itself.  Our autofocus / stigmation have
  *Cancel* (also Esc: the start values are set again); the SEM's own routines (ABC, AGA) cannot be
  interrupted.
* **Field of view** group: **size** (the swept range, down to 0.1 % of the full field: ×1000), **position** (beam shift) and **rotation**.  The
  inset in the lower-left corner shows the scanned region within the full field.
* **Mouse on the image**:
  - **left drag** – the measuring line: its length is shown on the line and, with Δx, Δy and the
    angle, in the status bar; it is kept in field µm, so it stays on its feature while the scan pans
    or zooms.  A click without a drag clears it.
  - **Shift + left drag** – pans the scanned region (beam shift); the image follows the mouse.
  - **Shift + wheel** – zooms the scanned region (the swept range, ×2 per three notches; Ctrl+± /
    Ctrl+0) – the column magnification is never touched; zooming out stops at the full field of view.
  - **wheel** – digital zoom of the image itself about the cursor, down to single pixels (a 5000 ×
    5000 capture can be inspected pixel by pixel); **middle drag** pans the zoomed image.
  - **right click** – *Center with beam* (beam shift to the clicked point), *Center with stage* (the
    stage moves the clicked point to the image centre, through the stage mirror of Hardware →
    Configure…, behind the modal stage dialog), *Clear measurement*, *Reset to Full FoV* (the full
    beam field and no digital zoom).
* **Data bar** with the scale bar, the image magnification (the column's magnification is that of the
  full field of view; zooming in multiplies it by full field / scan width) and date. There is also a
  scan-line marker during slow scans.
* **Magnification, focus/WD, stigmation, gun alignment** go to the SEM over RS-232 (the Column group); the
  field follows the column magnification (see *Field calibration* below).  The data bar shows the SEM's kV and WD, and captures
  store the column and stage readbacks in their metadata.  **Microscope…** opens the stage window.
* **Auto focus / Auto stigma / Focus + stig** (Column group) are easyEBL's own searches
  (`easyebl/imaging/autotune.py`; the SEM's AFC / ASF are not used).  They use small test images
  of the scan field as it is (by default Slow 1 at 256², ≈ 0.35 s; speed and size for each search
  in *Advanced… → Auto focus / stigma*) and a gradient-based sharpness measure.  It is Tenengrad
  (squared Sobel gradients) on a lightly de-noised image, taken in four directions with the
  detector noise's share taken off: the noise is measured in each image.  The harmonic mean of
  the four directions is used, dominated by the most blurred direction, so a line focus scores
  below the round spot.  It is divided by the gross contrast so brightness and contrast cancel.  Each step is
  one `SEMControl` setting confirmed by its readback (≈ 0.5 s on the S-3000) plus one image, and
  background reads are held meanwhile.  The SEM is switched to external scan once for the whole
  run (`ImagingEngine.hold_external_scan()`), not at every test image.
  - **Near the present value first** (focus and stigmator): the start value and two steps either side
    (focus: one focus step, see below; stigmator: 64 counts).  A **peak** there (the sharpest not at an
    end of the five) is refined; a **clear trend** (the sharpest at an end, best / worst ≥ 1.10) is
    followed that way with doubling steps until the sharpness falls – the peak bracketed – at most 16
    steps from the start, then refined.  Refining halves the step around the sharpest value down to
    the **depth of focus** (stigmator: 16 counts, its step), then a parabola through the best three.
    With no peak or trend nearby the whole range is searched as below.  On the simulators the focus
    takes ≈ 14–19 images, the stigmation ≈ 9 per axis.
  - **Depth of focus**: the largest defocus whose blur 2·α·Δz stays below the larger of the live image's
    pixel and the diffraction-limited probe (0.61 λ / α, λ from the accelerating voltage; α the beam
    convergence, 5 mrad), turned into FOCUS F counts with the focus table (the WD calibration) – e.g.
    ≈ 48 counts for a 640-pixel image of a 100 µm field.  FOCUS F is not refined beyond it.
  - **Focus** moves one knob at a time.
    - **Fine** (FOCUS F), the whole range 0–4095: first a grid at 512 counts, then halves the step
      around the sharpest value, two images per halving, down to the depth of focus.  At high magnification the first grid is finer, so it cannot
      step over the focus: at most 4 focus steps, where one step blurs the test image by
      ≈ 2 pixels (e.g. 128 counts at ×10k).  It takes ≈ 27 images, ≈ 23 s up to ≈ ×5k.
    - **Coarse** (FOCUS C) is only for when the fine focus cannot find the focus, and has no time
      limit.  It moves at most 0.1 mm of WD per setting (also on the way back and when restoring)
      and never beyond ±1 mm of WD from its start.  It sweeps one way until the sharpness has
      clearly fallen past its peak, the other way if needed, then fits the peak.
    - **Auto** searches near the present FOCUS F first, then the whole range; if that shows no clear
      optimum, or its peak is at an end, it sets FOCUS F back, runs the coarse sweep, then fine again.
  - **Stigmation**, when nothing is found near the present value: a full-range golden-section search on
    X, then on Y, ending with a parabola.
    It takes ≈ 9–10 steps per axis, ≈ 16 s for both with the default images.
  - **Falling back to the start values:** a result is kept only when it is sharper than the
    start.  Otherwise the start values are set again, as they are on cancel, on an error, on a
    readback that disagrees, with no clear optimum (no structure), or with a peak at the end of
    the range.
  - **Progress** shows under the buttons, with *Cancel*, and the result goes to Messages.
  - **Hooks for other subsystems:** the backend's `AutoTuner` (`bundle.autotune`, shared with
    the SEM window), `SEMWindow.auto_focus()` / `auto_stigmate()` / `focus_and_stigmate()`, and
    the lithography hardware's `auto_focus()` / `auto_stigmate()`.  Each returns an
    `AutoTuneResult`, or a future of one.
  - **Simulators:** the simulated image blurs with the simulated focus and stigmator
    (`microscope/sim_optics.py`).
* **External scan is automatic**: scanning switches the SEM to the DAQ's X/Y, stopping (freeze, single
  frame done, stop) gives the scan back to the SEM.  The status bar shows *EXT SCAN* while it is claimed.
* **Deflection range**: the full field of view is ±`scan_limit_v`; every waveform sample – pre-line
  overshoot, flyback, rotated corners – must stay within ±`ao_limit_v`.  On the 6363 both are ±10 V (DAQmx
  refuses AO samples beyond ±10 V; the DAC itself ends at ±10.1 V), so a scan whose pre-line or rotation
  would leave the range is narrowed to the widest that fits (zoom readout slightly above ×1).
* **Advanced…**: sample rate, buffer/latency, video input range, serpentine scan, pre-line mode, blank
  during pre-line, flyback shape, lag compensation, field calibration, and save options.
* **Field calibration** is global – one value for every magnification: `K`, the field the full
  deflection range sweeps at a hypothetical ×1 (shown also as µm/V).  The field at magnification M is
  `K / M`.  *Advanced… → Calibration* shows `K`, the magnification (from the SEM; entered by hand while
  the SEM does not report one) and the resulting field and µm/V.  To calibrate, draw a measuring line
  across a feature of known size, enter its real length and press *Calibrate*: `K` is scaled by
  real / measured.  `K` and the manual magnification are kept in `%APPDATA%\easyEBL\microscope.json`
  (in memory only with the simulators).  The scan orientation (invert / swap X/Y) is wiring and is set in
  *Hardware → Configure…*.

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

**Static beam and ADC, no stream.** `handle.park(x_v, y_v, blank=False)` holds the beam at a voltage;
`handle.read_ai_mean("video", samples=1000)` reads the video input now (a finite AI-only acquisition,
100 kS/s by default, averaged; `read_ai` returns the samples).  Outputs are untouched; both are refused
while a stream runs.  Point EDS spectra use exactly this: no stream while the DP5 counts.

**Lost USB connection.** If the 6363 is unplugged or switched off, the current call fails with
`DeviceLostError`. Operations then fail at once instead of hanging, and the beam state is reported as *unknown*.
A watchdog probes the device every 0.5 s with a harmless one-sample AI read. When the device answers again, the HAL resets
it, writes the safe state (blanked, parked), selects the SEM's internal scan, and then re-applies external control and
static lines for handles that still hold them. The engine shows **DISCONNECTED**, keeps its mode and settings, and
resumes scanning by itself. A capture in progress fails, and an interrupted single frame is rescanned. The programs also start
without the device and connect when it appears. Simulator tests: `tests/system/test_disconnect_system.py`.

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

**Data Manager** (below the electron image): every point / area spectrum, line scan and map of
the session is an item with a number (and a name if you give it one), kept in memory – a new map
or line scan never replaces an earlier one, and acquiring again at a measured site makes a new
item.  Clicking an item selects it: its outline is drawn thicker on the image with its number in
bold, a map shows its element maps, a line scan its profile.  Clicking a map pixel or a point
of the line profile only *shows* its spectrum; right-click there → **Add to Data Manager** keeps
it as an item.  Right-click an item: **Delete**, **Save**
(EMSA / `.npz`), **Remeasure** (a new item with the same geometry and conditions), **Rename**,
**Properties** (dwell, real/live time, beam energy, magnification, field, region, processor
speed and threshold, timestamp, microscope readbacks).  *Open…* reads spectrum, map and line
files back in (the name, geometry and conditions are stored in the file, so they can be
remeasured).  Unsaved items are marked; closing the window with unsaved items asks to save
(into a folder), discard or cancel.  *Clear all* removes only sites not measured yet.
Element maps are computed **while the map scans** (about once a second, from the pixels
scanned so far; unscanned pixels stay blank) and follow any change of the elements, map type,
binning or oxide mode – during the scan and after it.
**Tools → Energy calibration**: linear offset + slope, editable, or fitted from peaks you mark
in calibration spectra and name on the periodic table (the pairs are kept across spectra).
The DP5 **speed** (1 fastest … 5 best resolution) is a combo box in the Acquire tab, written as
soon as it is chosen; **Tools → Pulse processor** is only for setting raw DP5 parameters by
hand.  The live panel's dead-time bar shows the good range (5–50 %) in green, above it red.

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
(`THFA` 30) misses pulses (ICR < OCR), raise or lower `THFA` in the pulse-processor dialog; the
live time is clamped to the real time meanwhile.

## Microscope (Hitachi S-3000, RS-232)

Open → *Microscope*; the port and baud rate are set in *Hardware → Configure…*.

The SEM text protocol and the stage protocol share **one serial port** (9600 8N1; the SEM relays
the stage lines only after `#STAGE ON`).  The driver
(`easyebl/microscope/s3000`) is vendored from *hitachi_s3000_control_lib*, which was verified on the
S-3000. On top of it:

* **`MicroscopeLink`** owns the port and runs one request at a time in priority order: SEM settings,
  then stage moves, then reads, then background polling.  A SEM setting waits at most for the
  transaction already on the wire (replies take 0.25–1 s).  After a reply times out the link stops
  until you check the SEM and press **Resync** (nothing is ever resent automatically).
* **`SEMControl`** (column HAL): HV setpoint / on / off, magnification (the 65-step list), working
  distance (never read from the SEM: computed from the coarse and fine focus through the S-3000's
  focus table, µm resolution; setting a WD, any value in 4–60 mm, sends the table's `FOCUS C` and
  `FOCUS F`), coarse/fine focus, stigmator, condenser coarse/fine (raw 0–99), SEM display contrast /
  brightness / scan mode / freeze, detector, AFC / ASF / ABC / AGA (AFC / ASF are no longer in the
  GUIs: easyEBL's own auto focus / stigma replace them).  A focus setting is confirmed by one
  readback (written again once if the SEM reports something else, then an error).  A stigmator
  change of more than one step is written once; a one-step change is written twice, because the
  S-3000 misapplies a single one-step write.  Either way it is read back.  On the S-3000 every
  transaction takes ≈ 0.25 s and the readback shows the new value at once (measured 2026-11-25).
  `hold_polling()` pauses the background reads for a sequence of settings.  Readbacks are cached (instant for
  the GUIs), refreshed after each setting and by the background reads so changes
  on the SEM panel are seen.  Repeated settings coalesce: ten wheel steps during one transaction
  become one `MAG` with the final value.
* **`StageControl`** (stage HAL): X, Y, Z in µm and R, T in degrees.  Positions are polled at the
  lowest priority, **at most every 20 s** (one `R3` per controller), never while moving.  Moves are
  absolute or relative and wait for the motors' completion events. **STOP** never waits in the queue:
  S0 is sent at once, also while a long SEM command holds the port, and a move that is being sent
  at that moment is halted too.  Negative coordinates stay disabled until their encoding is checked
  on the instrument.
* **Who drives the stage.** The **SEM PC** has it between requests (`#STAGE OFF`: its trackball
  works).  Each position read sends `#STAGE ON`, the query and `#STAGE OFF`.  A **move** is one
  exclusive job on the port: `#STAGE ON`, the legs (backlash clearing included), the wait until
  every axis has stopped, the wait until every commanded axis *reads* within tolerance of its
  setpoint (X / Y / Z 0.5 µm, R / T 0.01°: X / Y keep correcting on their encoders for seconds
  after reporting stopped – up to 30 s, then the move fails; Z / R / T do not correct, so one left
  off fails at once), `S2` (motors free – the worm gears hold the stage)
  and `#STAGE OFF` – also when it fails or is stopped.  Meanwhile SEM settings wait in line and go
  out after the move, the background reads pause, and a second move is refused (`StageBusy`);
  only STOP gets through.  Subsystems that move the stage on their own show the modal wait dialog
  of `easyebl.microscope.movewait` (`run_blocking_move`, `StageMoveDialog`, with STOP).
* **Wiring.** Imaging and EDS use the column through `MicroscopeControl`.  `FieldScale`
  (`microscope/coupling.py`) keeps the DAQ field at `K / magnification` (the global field calibration,
  see the SEM window); zooming keeps the scan's share of the field.  EDS takes the
  SEM's kV as the nominal beam energy, and spectra and images carry the column and stage readbacks.
* The window: HV, the lithography's **exposure field size** (see *Lithography*), magnification,
  focus / WD, stigmator, condenser and SEM display on the left.  On
  the right is a compact stage panel:
  - **STOP ALL** (also *Esc*);
  - the line showing who drives the stage;
  - a digital readout of X, Y, Z, R and T (double-click a value to move that axis to / by a value;
    a value turns orange while its axis moves, red on a limit or fault);
  - the X/Y/Z/R/T jogs, with the step sizes and the X/Y approach beside them;
  - the lithography's **Stage positions** (the same list and buttons as the Lithography window's
    Stage Positions tab: X, Y, ΔZ, *Record WD*; see *Lithography*);
  - the raw TX/RX log.

  On the real instrument, HV switching, the automatic routines and moves ask for confirmation.  The
  SEM's own routine left in the window is **Auto gun alignment** (`AGA ON`, in the beam current
  group; also in the SEM toolbar).  HV is one button: *HV ON* while off.  The voltage box has no Set button: while HV is on it shows
  the SEM's voltage and Enter (or leaving the box) sets a new one; while HV is off a change is only
  kept, and HV ON switches on at it (at start: **30 kV** unless HV is already on), *HV OFF* while on; the
  emission current is shown in µA.  Auto focus and stigma are in the SEM imaging window.  The SEM's own scan mode is read back (`R SCAN`: answered by the S-3000
  although its manual does not list it) and shown in the display group.
* **Beam presets** (*Beam presets…* in the toolbar or *Presets…* in the beam group): any number of
  named settings of beam 1 coarse / fine, stigmator X / Y and contrast / brightness.  *Save current…*
  reads the six values fresh from the SEM; *Recall* (or double-click) sends them, after checking all
  six first.  Names and values can be edited in the table, and rows reordered or deleted.  Presets
  are stored in `microscope.json`.  Only beam 1 is used: the protocol has no command to select
  between the SEM's two beam-current memories.

The serial port is opened once by the easyEBL backend and shared by the SEM, EDS and microscope
windows.  Only the simulator has been used so far; see
*hitachi_s3000_control_lib/docs/commissioning.md* for what is verified on the instrument.

## Lithography

*Open → Lithography* is the e-beam lithography module.  Its window plots the **stage
coordinates (SCS, µm)**, drawn the same way round as the SEM image (*View → Show as the SEM
image*).  It shows:

- every chip: its CCS origin and x / y axes (nominal, or as measured by a Global Align);
- every field of every Exposure: its field of view (dashed), its pattern where the chip map puts
  it, and the selected step's fields drawn thicker (in the same colours); the whole GDS – every layer, also shapes in no
  field such as markers placed in the top cell itself – is drawn gray underneath, so what the
  Exposure does not write stays gray;
- with the toolbar's drawing modes: the patterns **coloured by field** or **by dose factor** (the
  viridis colour map from the lowest to the highest factor, its **colour bar** with labelled ticks on
  the plot's right edge; dose 0 gray), the **writing order** (the
  fields numbered in turn, ▲ the first, and dotted lines through the shapes of each field in the
  order they are written) and the **exposure points** themselves (*Show dots*: once the plot is zoomed in
  to 50 × 50 µm or less, only the points in the region shown – computed again as it moves; above
  400 000 points every n-th is drawn);
- the alignment markers (Global: magenta, Local: cyan);
- the named stage positions, and the stage with its current field of view: the full deflection
  field (green) and, when the SEM imaging window scans a reduced or shifted area, that area (red).

Right click on the plot (or *Stage → Move stage…*): **Move stage…** to the clicked position, to a
Global / Local marker of a chosen chip, or to a stage position – in focus (a chip: Z_cal and its
WD2 there; a stage position: Z_cal + ΔZ and WD1; off: Z and the focus stay).

The status bar shows the selected chip's alignment, the field size selected in the Microscope
window (against the schedule's), the WD now and WD1, the beam and field calibrations of the field
size, and the last field's map.

### Field sizes and focus (WD1, WD2, ΔZ)

The stage Z is not accurate enough, and there is no height sensor, to focus with Z, and the WD
(from the focus) is nominal.  So the stage Z stays at the calibration sample's **Z_cal** on every
chip and the **focus** does the rest:

1. **Field size** (*Calibration → Field sizes…*: name, magnification, nominal WD; to start with
   100 µm ×1200 WD 7 mm, 250 µm ×500 WD 11 mm, 500 µm ×250 WD 24 mm).  After loading the sample,
   select one in the **Microscope window**: it sets the magnification, moves the stage Z to
   1000 × the nominal WD (µm), and sets the focus to the **WD1 recorded for that field size**
   (the nominal WD when there is none yet).  Tune either by hand.  A schedule runs only at the
   field size selected there.
2. **WD1**: go to the *Calibration sample* (Stage positions), focus and stigmate at a high
   magnification, then **Record WD**: WD1 and the stage Z (Z_cal).  Both are kept **per field
   size**, with the field size settings; every ΔZ is against the selected field size's Z_cal.
   *Go to XYZ* moves to Z_cal + ΔZ and sets the focus to WD1.
3. **WD2**: go to the chip, focus on the corner / first alignment mark with the stage still at
   Z_cal, and record it in the chip's *Height…* dialog – one WD2 for the chip, or WD2x at ≥ 3
   points for a plane (a least-squares plane in chip coordinates; each field's WD from it at its
   centre).  A height is refused at another Z than Z_cal, and a schedule with a chip height
   recorded at another Z (Z_cal changed since) does not run.
4. A WD2 / WD2x more than 1 mm from WD1 is warned about.

During a run the stage moves in X and Y only; after it reaches a field centre the focus is set to
the chip's WD there (WD2, or the plane).  Stage positions store **ΔZ** (µm, relative to Z_cal;
positive moves the stage away from the gun).  Replacing the calibration sample (*Replace with
current*) at another Z changes every other ΔZ so their Z stays; *Record WD* changes only the
calibration sample.  A Z move to a stage position larger than *largest Z move* (Schedule settings,
default 2 mm) or a Z below 0 fails before the stage moves.

### Coordinates

| frame | meaning |
|---|---|
| GDS | an Exposure's GDS file, top-level cell coordinates (µm) |
| CCS | chip coordinates: GDS + the Exposure's offset (no rotation), axes as in the SEM image |
| FCS | field coordinates: CCS − the field centre |
| SCS | stage coordinates (encoder read-back) |
| BCS | true µm of the nominal beam position along the scan axes, from the field centre |
| XCS | scanner coordinates: deflection volts ÷ (V/µm) of the global field calibration (field = K / mag) |

| step | map | from |
|---|---|---|
| CCS → SCS | `scs = G(ccs) = A_g·ccs + t_g` | nominal: `A_g = M` (stage mirror), `t_g` = chip origin; Global Align: 2 markers rotation + scale + shift, 3–4 affine; kept for every later Exposure on the chip |
| field target | `s_f = G(c_f)`; `ŝ` = its read-back | |
| FCS → BCS | `bcs = B·(A_g·fcs + G(c_f) − ŝ + δ_f)` | `B`: the field calibration's rotation (with `M`); `δ_f`: the field's Local marker (SCS) |
| BCS → XCS | `xcs = L⁻¹·bcs·(1 − k·∣bcs∣²)`, the one-step inverse of `bcs = D(L·xcs)` | beam calibration: `L` = X / Y scale and shear, `k` pincushion |
| XCS → volts | `V = xcs · 2V_fs / field` | the global field calibration (SEM window) |

Marker picking runs the chain backwards with the same maps (the clicked XCS → BCS → SCS), so
within a field `L` and `B` cancel between measuring and writing.  Without alignment the
pattern is written with the beam calibration's scale / shear / distortion and the field
calibration's rotation; with alignment the scale, shear, rotation and shift come from the
alignment, the distortion from the beam calibration, and the field calibration places the
later fields.

The stage mirror is a hardware setting: *Hardware → Configure… → Devices → Microscope*
(S-3000: mirror X on, Y off).

### Calibrations

Every schedule starts with two steps that cannot be deleted, moved or disabled: **Column
setting** (the schedule's field size – the magnification; the focus goes to WD1 –, beam preset,
nominal current) and **Beam calibration**.  *Calibration → Run the fixed steps only* (Ctrl+K)
runs just these two.  Later Column settings change the beam preset / current only.

* **Beam calibration**: the stage goes to the calibration sample (its X, Y, Z_cal), the focus to
  WD1.  With *confirm the focus by hand first* (default on) the column zooms to ×10k (settable),
  the SEM window scans live, and a prompt waits – it does not block the other windows: focus and
  stigmate (it should be very close), then *Continue* (or *Abort*); the WD read back is WD1 from
  then on (saved for the field size); the stage Z must not change.  This is to be replaced by auto
  focus / stigmation.  Back at the step's magnification one image of the calibration grid (1 µm
  square) is taken over the whole field at ≤ 0.1 µm pixels (5000 × 5000 for 500 µm).  The grid's
  two lattice frequencies are demodulated and their phases fitted for the X / Y scale, shear and
  pincushion `k` (the grid's rotation and shift are fitted and dropped).  A window shows the
  image, the numbers, the fit residual and a map of the beam position error over the field;
  *Accept* keeps it, saved in `litho.json`, for the **field size** (used at any WD on the chips).
  Every accepted one is stored with its field size (the last one per field size and
  magnification).  With *use a stored calibration* the step takes the one stored for the
  schedule's field size instead of measuring (and measures when none is); the step's line and
  editor show which one (its date and numbers).  One stored without a field size (an older file)
  is not used.  The one-step inverse leaves an error of about 3·k²·r⁵ (≈ 40 nm at the field edge for
  0.8 µm of distortion there).
* **Field calibration** is not a step: *Calibration → Field calibration…* measures it outside a
  schedule, at the field size selected now, at a chosen stage position with a structured feature
  (Z_cal + ΔZ, the focus at WD1): a reference image, then one at each of a 3 × 3 grid (or 5-point
  cross) of stage shifts of ±30 % of the field; the images are de-noised (Gaussian, 2.5 px) and
  flattened (a wide blur subtracted), matched by partially whitened phase correlation (sub-pixel;
  quality: the peak-to-sidelobe ratio), and an image far off the others is left out of the fit;
  the shifts, converted to BCS through the beam calibration, give `B`.  Only its **stage ↔ beam
  rotation** is used; its scale (stage encoder against the grid) is a check.  A window shows the
  progress and the result; *Accept* keeps it – **one per field size, across sessions** (in
  `litho.json`) – and the schedules use it automatically; *Discard* keeps the one in force.  Each
  image is matched for every mirroring, so a wrong mirror setting shows as a mismatch, pointing to
  the hardware setting.  Without one for the field size the nominal rotation is used (a warning).

A fit that is refused (rms over the limit, weak matches, …) is shown with **Use anyway** next to
*Discard*: an override – the refused calibration is used (logged as such); *Discard* fails the
step.  When nothing could be fitted at all (no grid found) there is nothing to use.

### Stage

- X and Y always end travelling in + by at least the *X/Y approach* distance (default 50 µm;
  Microscope window, Jog), to clear the backlash.  Every move then waits until each moved axis
  reads within tolerance of its setpoint (X / Y / Z 0.5 µm, R / T 0.01°).
- Every stage move of the lithography (schedule, calibrations, *Go to*) shows the modal *Stage
  moving* dialog (live positions, STOP); a move already in progress is waited for first, and
  Abort stops the stage.
- During a run, each move is checked against the freshly read position: a step fails when the
  stage is farther than the *position tolerance* (Stage settings…, default 0.5 µm).

### Tabs

* **Stage Positions**: named positions X, Y and **ΔZ**: *Add current stage*, *Add…*, *Replace
  with current* (the selected one becomes the stage now), **Record WD** (WD1 of the field size and
  Z_cal on the calibration sample), *Go to XYZ* (X, Y, Z_cal + ΔZ, the focus to WD1), *Go to XY*,
  *Delete*.  *Calibration sample* and *Faraday cup* are always in the list (replaced, not
  deleted).  A position without ΔZ (from an older file, which stored H or Z) is shown red.  The
  same list, with the same buttons, is in the Microscope window; both follow every change.
  Saved in `%APPDATA%\easyEBL\litho.json` (`EASYEBL_LITHO_SETTINGS` overrides) with the field
  sizes, WD1 per field size and the calibrations.
* **Chips**: name, the nominal stage position of the CCS origin, the alignment, the height, a
  note; *Add chip* (at the stage), *Origin = stage*, *Move…* (a reference point – the origin,
  the first field centre, the bounding-box centre or Global marker 1 – to the stage, a click or
  a typed position), *Reset alignment*, *Height…* (focus kept, one WD2 or a WD2x map: *Record
  here* at Z_cal, *Go to* a point – Z_cal and its WD; the WD range over the chip's fields).
* **Exposure Schedule**, three step types:
  - *Column setting*: the field size (the first one: the schedule's), a beam preset of the
    microscope module (or *Keep as is*), an optional nominal current for the dose readout.
  - *Beam calibration*: on the calibration sample, with its settings.
  - *Exposure*: a chip and a whole GDS file (top cell, layers, offset of the GDS origin in CCS); choosing the
    file – also the same one again – reads it from disk, and **⟳** next to it reloads it after it changed.
    Each top-level cell (the references placed by the top cell, array elements included) is one
    field; its centre is the cell's bounding-box centre, or (checkbox) the cell's origin.
    Dwell per point and point step (GDS dose factors multiply the dwell).  **The dose factor comes
    from the datatype, decided layer by layer** (no setting; the editor shows it per layer): a layer
    whose shapes all have datatype 0 is written at the plain dose (factor 1); a layer with other
    datatypes carries the dose in them – factor = datatype / 1000 (2/500 → 0.500, 2/1000 → 1.000,
    2/1250 → 1.250); datatype 0 on such a layer is dose 0 (not written, a warning).  With layers
    1 and 2 selected, 1/0 and 2/500, 2/1000 in the file: layer 1 at the plain dose, layer 2 at 0.5×
    and 1×.  Markers are CCS points: **Global Align** 2–4 markers – inside one field's field of view:
    that field is aligned and written first, then the others in cell-name order; in no field's but
    within one field size: a dummy **Field 0** centred on them (drawn and labelled on the plot) is
    aligned first and nothing written there, then every field in cell-name order (with its Local
    marker, or no further alignment); wider than a field: an error, the schedule does not run.
    Cell-name order compares numbers as numbers (Field_2 before Field_10).  They are measured twice (1 → 2 → 1 → 2), the second pass predicted from the
    first, and the two passes must agree within the *alignment tolerance* (default 0.5 µm; the
    mean is used) – else *Repeat*, *Continue anyway* (an override) or *Abort*; **Local
    markers**, assigned automatically to fields whose field of view holds them, at most one per
    field, so that as many fields as possible get one (then the nearest centres) – the table's
    *field* column shows each marker's field (— : not used); its buttons: *Add manually* (a row to type X, Y
    in), *Pick & Add*, *Pick & Replace* (the selected marker moved to a click) and *Remove* (the
    selected rows).  Wrong assignments are warnings.  Without a GDS file the step only aligns.
    **Markers are picked on the plot**: *Pick* (one Global marker), *Pick all on the plot…*
    (marker 1, 2, … in turn) and, for Local markers, *Pick & Add* (each click adds one, Esc
    ends) and *Pick & Replace* (one click).  A click snaps to the nearest corner or centre of any polygon of the Exposure's
    GDS – every layer, exposed or not, also shapes in the top cell itself – within 12 screen
    pixels (ringed in yellow while the mouse moves; nothing near: nothing taken); Shift-click
    takes the clicked point itself; Esc or a right click ends picking.

  *Validate* (and Run) warns about: patterns outside the field of view, markers in no field or
  two in one, Global markers not in one field, unknown chips, beam presets or field sizes, the
  Microscope window at another field size, missing beam or field calibrations, no WD1, chips
  without a usable height or recorded at another Z, WD2 more than 1 mm from WD1.  A run is
  refused (with the reasons) at another field size than the Microscope window's, without WD1, or
  with a chip height recorded at another Z than Z_cal.  Schedules are JSON files (`*.ebls.json`,
  version 5: the field size, the Beam calibration settings, the chips' heights) with the chips;
  GDS paths are stored relative to the file.  Version 3 and 4 files open with their WD, heights
  and Field calibration steps dropped and the magnification turned into the default field size
  with it (none if no default has it).
* **Run** starts by putting the SEM in RUN mode on Slow scan 1 (left so afterwards), then runs
  every step from the first.  When it finishes, the stage goes back to the calibration sample
  (Z_cal, the focus at WD1); after a failure or Abort it stays.  **Run from selected** skips the
  fixed steps and starts at the selected step – only when the column is at the field size's
  magnification, the stage at Z_cal and a beam calibration of the field size is stored;
  otherwise it is refused with the reason.
* **Running** an Exposure, field by field: the stage goes to the field's target (X, Y) and the
  focus to the chip's WD there; for each marker the SEM imaging window zooms its scan onto the
  predicted position (beam shift and scan width) (magenta cross).  **Each click centres the scan
  on the clicked point** (beam shift, clamped at the field edge), so the marker comes under the
  green centre cross; click again to refine, as often as you like.  The bar above the image shows
  the scan centre, its offset from the prediction and the clicks; **Confirm** (Enter) takes the
  scan centre as the marker position, *Reset* goes back to the prediction, *Cancel* (Esc) aborts
  the run.  Then the SEM scan stops and the pattern is written through the chain.  A pattern
  outside the field is skipped or aborts, as set in *Schedule settings*.
* **Demo files**: `python -m easyebl.litho.demo` writes them to `examples/litho/`:
  `multifield_demo.gds` (3 × 3 fields of 80 µm on a 100 µm pitch; layer 1 the pattern, its dose in
  the datatype – 1000, the rings 1200 –, layer 2 alignment crosses at (±45, ±45) µm), `multifield_demo.ebls.json` (field size 250 µm, chip at
  stage (5000, 5000) µm, the fixed steps – record the calibration sample and its WD1 –, one
  Exposure of layer 1 with Global Align in the first field and a Local marker in every other) and
  `dummy_demo.ebls.json` (the GDS without alignment, then an alignment-only Exposure).

Not implemented yet: automatic marker finding (markers are clicked), autofocus for WD1 / WD2
(focused by hand) and a measured beam current.  On the simulators the SEM image follows
the simulated stage through a hidden stage → beam map (0.35° and +0.6 %), so a field calibration has something to find.

## Architecture

```
easyebl/
  core/       AffineTransform, FieldCalibration (µm ↔ V), OutputChunk streams
  hal/        DAQConfig, DAQBackend (abstract), DAQManager + DAQHandle, SimulatedBackend, NIDAQBackend
  imaging/    RasterConfig → RasterPlan (waveforms) → RasterImager (+ LineAssembler for AI);
              autotune.py: auto focus / stigmation (AutoTuner, sharpness)
    engine/   threaded ImagingEngine: settings, source, program, assembler, frames, core
    sem/      SEM GUI: window, view (data bar), panels (column via easyebl.microscope), advanced dialog
    cli.py    text commands (ImagingShell); export.py  TIFF/PNG/JSON writers
  hal/dp5/    vendored Amptek DP5 USB driver (+ mock device); hal/mca.py pulse-processor HAL
  hal/devlock.py  process lock on the NI device
  microscope/ Hitachi S-3000 RS-232: link (shared port, priorities), sem (column HAL), stage (stage HAL),
              coupling (field ↔ magnification), sim_optics (simulated focus / stigmator blur),
              gui; s3000/ vendored driver + simulator
  xray/       X-ray lines/edges/attenuation (xraydb tables) and the generation model
tools/        gen_xray_data.py: regenerates easyebl/xray/data/xray_data.npz (needs xraydb)
tests/        unit/, system/ (simulator pipelines), gui/ (offscreen Qt); data/ real DP5 spectra
  eds/        EDSController (spectra, images, maps), spectrum/map data, analysis/, gui/
  alignment/  templates + NCC matching, transform fitting, MarkerAligner
  litho/      dose maths, geometry, fills, VectorScanCompiler, LithoWriter, GDS import;
              coords (SCS/WCS/BCS), schedule, fracture (importer), runner, settings, demo; module/ (window)
  gui/        the DAQ output (scan) monitor widget
  app/        the easyEBL program: SystemConfig, SystemBackend (all hardware), main window, hardware dialog
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
| **GDS** | Flattened hierarchy and arrays; layer/datatype filter; dose factor from the **datatype, per layer** (datatype 0 only: uniform; else datatype / 1000; written back the same way); paths become polygons. |
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
* dose factors from GDS datatypes scale the delivered dose;
* a late blanker leaves dose trails, and compensating for it removes them;
* imaging, alignment and lithography share one device without conflicts.

The NI backend is tested against an in‑memory fake of `nidaqmx`. These tests cover sample
alignment of AO/DO/AI, the clock routing, the start order, blanker polarity, the safe tail and abort.

**Hardware tests** (`tests/hardware`, marker `hardware`) talk to the real DP5 (USB) and S-3000
(the serial port saved in the easyEBL configuration, else COM4).  They run with every `pytest`
and skip by themselves when a device is not connected, switched off or in use (close easyEBL
first).  They check communication and settings, not spectra: DP5 identity, slow threshold,
shaping preset, preset real time, counting and clearing, the EDS live monitor; SEM readbacks,
magnification, WD, focus, stigmator, condenser, contrast / brightness, freeze, beam presets,
priorities, coalescing, stage reads, small X / Y moves and STOP.  Every changed setting and the
X / Y position are restored.  They never set or switch the high voltage, run automatic routines,
touch the DP5 detector bias or cooler, or move Z / R / T.

```bash
pytest tests/hardware                         # on the instrument
EASYEBL_HW_NO_STAGE=1 pytest tests/hardware   # without stage moves
EASYEBL_NO_HARDWARE_TESTS=1 pytest            # never touch the instrument
EASYEBL_HW_DRY_RUN=1 pytest tests/hardware    # check the tests on the mock DP5 / S-3000 simulator
```

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
* The S‑3000N DBC input is scaled by the SEM magnification.  easyEBL assumes field = K / magnification
  with one `K`; check the calibration at the magnifications you write at (the SEM's own magnification
  steps may not be exactly proportional), and match the voltage range to the SEM's X‑OUT/Y‑OUT signals.
