# easyEBL architecture notes

## 1. Coordinates and calibration

* All geometry is in **µm field coordinates**: the origin is at the field centre, +x points right and +y points up.
* `FieldCalibration` maps µm to volts with `nominal scaling ∘ correction`. The *correction* is an
  `AffineTransform` that holds the measured field distortion (scale, rotation, offset).
* Marker alignment produces a **design → field** transform. `LithoWriter` applies it to the
  exposure *points* after filling, so the fill grid stays tied to the design.
* `OutOfFieldError` is raised **before** streaming if anything maps outside ±full scale. This
  includes imaging overscan from pre‑line ramps.

## 2. Timing model

All hardware‑timed operations are `Iterable[OutputChunk]`. Each chunk carries X, Y, blank and
optional marker lines, one value per tick of a single sample clock. The NI backend works as follows:

1. The AO task (X, Y) generates the sample clock.
2. The DO task (blanker plus markers, the whole of port0) and the AI task use `/DevN/ao/SampleClock`.
3. The slave tasks start before AO. Output sample *k*, DO sample *k* and AI sample *k* therefore
   share one clock edge.
4. Regeneration is disabled. The output buffer is set explicitly to at least `min_buffer_s` of
   data, input is re‑chunked to a quarter of the buffer, and AI is drained between writes.
5. Every stream, including an **abort**, ends with a clocked safe tail. The tail first blanks
   while holding the last position (`tail_hold_s`, which covers blanker latency), then parks.
   If an error occurs, AO (the clock master) is stopped first and the tasks are closed. The
   static safe state is then written **blank first, then move**.
6. The device coerces the sample clock to 100 MHz/N. The actual rate is read back, and a
   warning is issued if it differs from the request. Use rates such as 400 k or 500 kS/s.
7. **AI latency**: AI sample *k* is converted on the same edge that updates the DAC, so it sees
   the beam position from *before* update *k*. Imaging therefore uses `ai_latency_samples = 1`
   by default, and `lag_comp_s` handles the remaining deflection and detector lag. The
   simulator models this latency too.

Remaining hardware risk: if the host falls behind (buffer underflow), AO stops and the blank
line freezes in its last state until the error surfaces. Keep streaming callbacks light. A
hardware watchdog on the blanker (blank unless a heartbeat is present) is strongly recommended.

## 3. HAL: sharing one device

* There is one `DAQManager` per device. Each subsystem or GUI gets its own `DAQHandle`.
* `run_stream` takes the device operation lock. Other callers fail with `DeviceBusyError` or wait
  (`timeout`), and the manager reports the `active_owner`.
* Static lines such as the scan relay need an **exclusive claim**. Arbitrary named resources
  (`"stage"`, `"sem_serial"`, …) use the same mechanism, so future subsystems coordinate the same way.
* **Safe state**: after every stream (normal, aborted or error) the manager blanks and parks the
  beam. Switching a static line blanks first.
* Listeners receive events (`stream_started`, `claim`, …). Exceptions in listeners are swallowed,
  so a GUI can never break hardware control.
* Callbacks (`on_ai`, `progress`, listeners) run in the streaming thread while the device lock is
  held. GUIs forward data through queued Qt signals and never block inside a callback.
* NaN or infinite samples are rejected, as are out‑of‑range voltages. If the safe state cannot be
  written, the manager records the beam state as *unknown*, never as blanked.

## 4. Raster imaging

Each line is laid out as `flyback | pre-line | nx × samples_per_pixel`. The pixels form a staircase.

* **Pre‑line** has two modes. `ramp` continues the staircase outside the image, so the deflection
  reaches steady‑state lag before the first pixel. `hold` waits at the first pixel.
* **Flyback** is blanked by default. Bidirectional scans turn it into a turnaround.
* **Lag compensation** shifts the AI stream by `lag_comp_s` (fractional samples are allowed). The
  `LineAssembler` processes rows as soon as their samples (plus the lag) have arrived, so partial
  images are available live.

## 5. Alignment and lag

A fast raster image is shifted by `v·τ` along the fast axis, while lithography exposes
quasi‑statically. `MarkerAligner` therefore images every mark twice, scanning +x and −x, and
averages the two positions. This cancels the shift to first order. The half‑difference is
reported as the measured lag. The subsystem tests demonstrate both the bias of single‑direction
scans and the correction.

## 6. Vector scan

`build_runs` turns each element into runs of points:

* **Polygons** use a raster fill on a global grid, so adjacent shapes tile without gaps or overlaps.
* **Lines** use equal sub‑segments.
* **Dots** are single points.

`VectorScanCompiler` then emits the sample stream:

* **Between runs**: blank‑lead → blanked jump → settle (base + per‑µm, capped).
* **Within a run**: each point is held for its dwell. Step‑and‑blank mode is optional.
* **Dose quantisation**: `diffuse` (the default) carries the rounding remainder along each run, so
  the integrated dose is exact even at 3–4 samples per point. `round` quantises each point
  independently. Requests are clamped to `min_samples_per_point` *before* diffusion, and
  clamped (overdosed) points are counted. `CompileStats` reports the per‑point and total errors.
* **Blanker latency**: `blanker_delay_s` delays the XY stream relative to the blank stream.

## 7. Simulator

`SimulatedBackend` closes the loop:

* volts → true position, including a true distortion;
* first‑order deflection lag;
* blanker delay and leakage;
* video from a `SampleModel`, with detector delay, bandwidth and noise;
* a `DoseMap` that accumulates `I·dt` for every unblanked sample at the *actual* beam position.

Filter and delay states persist across chunks, so splitting a stream into chunks does not change
the result (this is tested).

## 8. Not yet implemented

* EDS (Amptek DP5) integration. Hooks exist: timed marker lines (`line_sync_marker`,
  `frame_sync_marker`) for hardware sequential buffering or list‑mode sync.
* Hitachi RS‑232 remote control (magnification, focus, stage) and multi‑field writing with
  stage moves and per‑field marks.
* A calibration‑grid workflow (distortion and centre shift per magnification step).
* Proximity‑effect correction (use BEAMER and import the dose classes).
