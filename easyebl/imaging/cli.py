"""Command-line control of the imaging engine.

Interactive::

    python -m easyebl.imaging.cli --backend nidaqmx --device Dev1
    sem> run
    sem> speed Slow 1
    sem> shift 2 -1
    sem> capture photos/sample1 frames=2 res=2560x1920 speed=Photo1
    sem> quit

One-shot / scripted (commands separated by ``;`` or from a file)::

    python -m easyebl.imaging.cli -c "run; wait 3; capture shot.tif; quit"
    python -m easyebl.imaging.cli --script session.txt

The same :class:`ImagingShell` also runs inside the SEM GUI (``--console``),
controlling the very engine the GUI shows; every command goes through the
engine's thread-safe API.
"""
from __future__ import annotations

import argparse
import cmd
import shlex
import sys
import threading
import time

from .engine import RESOLUTIONS, SCAN_SPEEDS, EngineState, ImagingEngine, speed_by_name
from .export import save_frame

__all__ = ["ImagingShell", "main", "start_console"]


def _kv(args: list[str]) -> tuple[list[str], dict]:
    pos, kw = [], {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            kw[k.strip().lower()] = v.strip()
        else:
            pos.append(a)
    return pos, kw


def _res(text: str) -> tuple[int, int]:
    w, h = text.lower().replace("×", "x").split("x")
    return int(w), int(h)


class ImagingShell(cmd.Cmd):
    intro = "easyEBL SEM imaging console – 'help' lists commands, 'quit' leaves."
    prompt = "sem> "

    def __init__(self, engine: ImagingEngine, stdin=None, stdout=None, close_on_quit: bool = True):
        super().__init__(stdin=stdin, stdout=stdout)
        if stdin is not None:
            self.use_rawinput = False
        self.engine = engine
        self.close_on_quit = close_on_quit
        self.failed = False

    # ---------------------------------------------------------------- plumbing
    def say(self, msg: str = "") -> None:
        self.stdout.write(msg + "\n")
        self.stdout.flush()

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except (ValueError, KeyError, RuntimeError, TimeoutError) as e:
            self.failed = True
            self.say(f"error: {e}")
        except Exception as e:  # noqa: BLE001
            self.failed = True
            self.say(f"error: {type(e).__name__}: {e}")
        return False

    def emptyline(self):
        return False

    def default(self, line):
        self.failed = True
        self.say(f"unknown command: {line.split()[0]!r} (try 'help')")

    def run_commands(self, text: str) -> bool:
        """Run ';'/newline separated commands; returns True if 'quit' was reached."""
        for part in text.replace("\n", ";").split(";"):
            part = part.strip()
            if not part or part.startswith("#"):
                continue
            self.say(self.prompt + part)
            if self.onecmd(part):
                return True
        return False

    # ---------------------------------------------------------------- scanning
    def do_run(self, arg):
        """run – live scanning"""
        self.engine.run()

    def do_freeze(self, arg):
        """freeze [now] – stop at the end of the frame (or immediately)"""
        self.engine.freeze(immediate=arg.strip().lower() in ("now", "!", "immediate"))

    def do_single(self, arg):
        """single – one frame, then freeze"""
        self.engine.single()

    def do_stop(self, arg):
        """stop – stop immediately (beam blanked and parked)"""
        self.engine.stop()

    def do_speed(self, arg):
        """speed [name | dwell_µs] – scan speed (no argument: list presets)"""
        a = arg.strip()
        if not a:
            cur = self.engine.config.actual_dwell_s
            for s in SCAN_SPEEDS:
                mark = "*" if abs(s.dwell_s - cur) < 1e-12 else " "
                self.say(f" {mark} {s.label():<22} frame {self.engine.frame_time_s(self.engine.config.with_(dwell_s=s.dwell_s)):.3f} s")
            return
        try:
            sp = speed_by_name(a)
            self.engine.set_speed(sp)
        except KeyError:
            self.engine.set_speed(float(a) * 1e-6)
        c = self.engine.config
        self.say(f"dwell {c.actual_dwell_s * 1e6:g} µs ({c.samples_per_pixel} samples/pixel), "
                 f"frame {self.engine.frame_time_s():.3f} s")

    def do_res(self, arg):
        """res [WxH] – scan resolution (no argument: list presets)"""
        a = arg.strip()
        if not a:
            c = self.engine.config
            self.say("presets: " + ", ".join(f"{w}x{h}" for w, h in RESOLUTIONS) + f"   (now {c.nx}x{c.ny})")
            return
        self.engine.set_resolution(*_res(a))
        self.say(f"frame {self.engine.frame_time_s():.3f} s")

    def do_width(self, arg):
        """width <µm> – scan field width (clamped to the deflection field)"""
        c = self.engine.set_scan_width(float(arg))
        self.say(f"width {c.width_um:g} µm, pixel {c.pixel_size_um[0] * 1e3:.2f} nm")

    def do_shift(self, arg):
        """shift <dx_µm> <dy_µm> | shift reset – move the beam (scan centre)"""
        a = arg.split()
        if a and a[0].lower() in ("reset", "0", "home"):
            c = self.engine.reset_beam_shift()
        else:
            c = self.engine.shift_beam(float(a[0]), float(a[1]))
        self.say(f"beam shift {c.center_um[0]:+.3f}, {c.center_um[1]:+.3f} µm")

    def do_center(self, arg):
        """center <x_µm> <y_µm> – absolute beam shift"""
        x, y = (float(v) for v in arg.split())
        c = self.engine.set_beam_shift(x, y)
        self.say(f"beam shift {c.center_um[0]:+.3f}, {c.center_um[1]:+.3f} µm")

    def do_rotate(self, arg):
        """rotate <deg> – scan rotation"""
        self.engine.set_rotation_deg(float(arg))

    def do_bidir(self, arg):
        """bidir on|off – serpentine scan"""
        self.engine.update(bidirectional=arg.strip().lower() in ("1", "on", "true", "yes"))

    def do_avg(self, arg):
        """avg <N> – running average over N frames (1 = off)"""
        self.engine.set_options(averaging=int(arg))

    def do_rate(self, arg):
        """rate <S/s> – sample clock (restarts the stream)"""
        self.engine.set_options(rate=float(arg))

    def do_latency(self, arg):
        """latency <ms> – output buffer depth (restarts the stream)"""
        self.engine.set_options(latency_s=float(arg) / 1e3)

    def do_airange(self, arg):
        """airange <V> – AI input range ±V (restarts the stream)"""
        self.engine.set_options(ai_range_v=float(arg))

    def do_aimode(self, arg):
        """aimode [DIFF|RSE|NRSE|PSEUDO_DIFF] – video input mode (restarts scanning)"""
        a = arg.strip()
        if not a:
            self.say(f"video input mode: {self.engine.handle.ai_terminal}")
            return
        self.say(f"video input mode {self.engine.set_ai_terminal(a)}")

    def do_set(self, arg):
        """set field=value … – any RasterConfig field, e.g. set flyback_s=60e-6 lag_comp_s=1e-6"""
        _, kw = _kv(shlex.split(arg))
        cur = self.engine.config
        typed = {}
        for k, v in kw.items():
            if not hasattr(cur, k):
                raise ValueError(f"unknown setting '{k}'")
            old = getattr(cur, k)
            if isinstance(old, bool):
                typed[k] = v.lower() in ("1", "on", "true", "yes")
            elif isinstance(old, int):
                typed[k] = int(v)
            elif isinstance(old, float):
                typed[k] = float(v)
            elif isinstance(old, tuple):
                typed[k] = tuple(float(x) for x in v.split(","))
            else:
                typed[k] = v
        self.engine.update(**typed)

    # ---------------------------------------------------------------- results
    def do_capture(self, arg):
        """capture [path] [frames=N] [res=WxH] [speed=name] [timeout=s] – photo scan, saved if a path is given"""
        pos, kw = _kv(shlex.split(arg))
        opts = {}
        if "res" in kw:
            opts["nx"], opts["ny"] = _res(kw["res"])
        if "speed" in kw:
            opts["speed"] = kw["speed"]
        frames = int(kw.get("frames", 1))
        fut = self.engine.capture(frames=frames, **opts)
        cfg = self.engine.config
        est = self.engine.frame_time_s() * frames
        self.say(f"capturing … (≈ {est:.1f} s at live settings)")
        timeout = float(kw.get("timeout", max(60.0, 5 * est + 30)))
        frame = fut.result(timeout=timeout)
        self.say(f"captured {frame.config.nx}x{frame.config.ny}, {frame.frames_averaged} frame(s)")
        if pos:
            self._save(frame, pos[0])
        del cfg

    def do_save(self, arg):
        """save <path> – save the latest image (tif + png + json)"""
        f = self.engine.latest()
        if f is None:
            raise RuntimeError("no image yet")
        self._save(f, arg.strip() or time.strftime("sem_%Y%m%d_%H%M%S"))

    def _save(self, frame, path):
        rng = self.engine.options.ai_range_v or self.engine.daq_config.ai_range_v
        files = save_frame(frame, path, ai_range_v=rng)
        self.say("saved " + ", ".join(str(f) for f in files))

    def do_wait(self, arg):
        """wait [frames=N | seconds] – wait for N complete frames, or for a number of seconds"""
        pos, kw = _kv(arg.split())
        if "frames" in kw or not pos:
            n = int(kw.get("frames", 1))
            timeout = float(kw.get("timeout", 60))
            seq = self.engine.frames.seq
            for _ in range(n):
                f = self.engine.wait_frame(seq, timeout=timeout, complete=True)
                if f is None:
                    raise TimeoutError("no frame within the timeout")
                seq = f.seq
            self.say(f"{n} frame(s) – {self.engine.status().summary()}")
        else:
            time.sleep(float(pos[0]))

    def do_status(self, arg):
        """status – engine and stream health"""
        s = self.engine.status()
        self.say(s.summary())
        self.say(f"  frames {s.frames_completed}, frame time {s.frame_time_s:.4f} s, display {s.fps_display:.2f} fps")
        self.say(f"  streams {s.streams}, underruns {s.underruns}, recoveries {s.recoveries}, "
                 f"AI backlog max {s.ai_backlog_max}")
        if s.change_latency_s is not None:
            self.say(f"  last settings change → first new line: {s.change_latency_s * 1e3:.1f} ms")
        if s.start_latency_s is not None:
            self.say(f"  last start → first sample: {s.start_latency_s * 1e3:.1f} ms")
        if s.last_error:
            self.say(f"  last error: {s.last_error}")
        if s.saturation:
            self.say(f"  video saturation: {s.saturation:.1%} of pixels at the input limit")

    def do_settings(self, arg):
        """settings – current scan settings"""
        c = self.engine.config
        o = self.engine.options
        self.say(f"{c.nx}x{c.ny}, width {c.width_um:g} µm, centre {c.center_um}, rotation {c.rotation_rad:g} rad")
        self.say(f"dwell {c.actual_dwell_s * 1e6:g} µs @ {c.rate:g} S/s ({c.samples_per_pixel} samples/pixel), "
                 f"bidirectional {c.bidirectional}")
        self.say(f"flyback {c.flyback_s * 1e6:g} µs ({c.flyback_shape}), pre-line {c.preline_s * 1e6:g} µs "
                 f"({c.preline_mode}), lag comp {c.lag_comp_s * 1e6:g} µs, AI latency {c.ai_latency_samples:g}")
        self.say(f"averaging {o.averaging}, latency {o.latency_s * 1e3:g} ms, AI range {o.ai_range_v or 'config'}")

    def do_external(self, arg):
        """external on|off – SEM scan source: this program (on) or the SEM's own scan (off)"""
        a = arg.strip().lower()
        if not a:
            self.say(f"external scan: {self.engine.handle.manager.external_control}")
            return
        on = a in ("1", "on", "true", "yes")
        was = self.engine.is_scanning
        if was:
            self.engine.stop()
            self.engine.wait_idle(5.0)
        self.engine.handle.set_external_control(on)
        if not on:
            self.engine.handle.release("external_control")
        self.say("external scan ON" if on else "external scan OFF (SEM internal scan)")
        if was and on:
            self.engine.run()

    def do_aux(self, arg):
        """aux [aux0|aux1 <V>] – set an auxiliary analog output (no argument: show them)"""
        a = arg.split()
        h = self.engine.handle
        if not a:
            for name, (phys, v) in h.aux_outputs.items():
                self.say(f"  {name}: {v:+.4f} V  ({phys or 'dummy – no channel on this board'})")
            return
        if self.engine.is_scanning:
            raise RuntimeError("aux outputs cannot change while scanning – freeze first")
        h.set_aux_output(a[0], float(a[1]))
        self.say(f"{a[0]} = {float(a[1]):+.4f} V")

    def do_events(self, arg):
        """events – recent engine events"""
        for ev in list(self.engine.events.history)[-20:]:
            self.say(time.strftime("%H:%M:%S ", time.localtime(ev.t)) + str(ev))

    def do_quit(self, arg):
        """quit – stop scanning and leave"""
        if self.close_on_quit:
            self.engine.close()
        return True

    do_exit = do_quit
    do_EOF = do_quit


def start_console(engine: ImagingEngine, on_quit=None) -> threading.Thread:
    """Run an :class:`ImagingShell` on stdin/stdout in a daemon thread (for the GUI)."""

    def loop():
        sh = ImagingShell(engine, close_on_quit=False)
        try:
            sh.cmdloop()
        finally:
            if on_quit is not None:
                on_quit()

    t = threading.Thread(target=loop, name="easyebl-console", daemon=True)
    t.start()
    return t


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    from .launch import add_hardware_args, open_engine

    p = add_hardware_args(argparse.ArgumentParser(description="easyEBL SEM imaging console"))
    p.add_argument("-c", "--commands", default=None, help="commands separated by ';' (then exit)")
    p.add_argument("--script", default=None, help="file with one command per line (then exit)")
    p.add_argument("--external", "--relay", dest="external", action="store_true",
                   help="switch the SEM to external scan (HAL external control) while the console runs")
    args = p.parse_args(argv)
    mgr, engine = open_engine(args)
    relay = None
    try:
        if args.external:
            relay = mgr.get_handle("external_scan")
            relay.set_external_control(True)
        sh = ImagingShell(engine)
        if args.commands or args.script:
            text = args.commands or open(args.script, encoding="utf-8").read()
            sh.run_commands(text)
            return 1 if sh.failed else 0
        sh.cmdloop()
        return 0
    finally:
        engine.close()
        if relay is not None:
            relay.close()  # returns the SEM to its internal scan
        mgr.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
