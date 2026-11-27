"""Auto focus / stigmation (easyebl.imaging.autotune): the sharpness measure, the options and the
plumbing.  The searches themselves are verified on the instrument, not on the simulators."""
import time

import numpy as np
import pytest

from easyebl.hal.simulated import default_demo_sample
from easyebl.imaging.autotune import AutoTuner, fit_log_parabola, image_time_s, noise_sigma, sharpness


# ------------------------------------------------------------------ sharpness measure
def _blur(img, sa, sb, ang=0.0):
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    u = fx * np.cos(ang) + fy * np.sin(ang)
    v = -fx * np.sin(ang) + fy * np.cos(ang)
    g = np.exp(-2 * np.pi ** 2 * ((sa * u) ** 2 + (sb * v) ** 2))
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape)


def _scene():
    """256² of the demo specimen at 0.2 µm pixels."""
    return default_demo_sample().image[::2, ::2][100:356, 100:356]


def test_gradient_sharpness_falls_with_blur_and_ignores_gain_and_noise():
    img = _scene()
    s = [sharpness(_blur(img, b, b)) for b in (0.0, 1.0, 2.0, 4.0)]
    assert s[0] > s[1] > s[2] > s[3] > 0
    assert sharpness(3.0 * img + 7.0) == pytest.approx(sharpness(img), rel=1e-6)  # gain / offset cancel
    rng = np.random.default_rng(1)
    for sigma in (0.02, 0.05):  # image rms ≈ 0.1
        noisy = [sharpness(_blur(img, b, b) + rng.normal(0, sigma, img.shape)) for b in (0.0, 1.0, 2.0, 4.0)]
        assert noisy[0] > noisy[1] > noisy[2] > noisy[3], sigma
    assert noise_sigma(img + rng.normal(0, 0.05, img.shape)) == pytest.approx(0.05, rel=0.05)


def test_noise_alone_has_no_detail():
    rng = np.random.default_rng(2)
    assert all(sharpness(rng.normal(0, 1, (256, 256))) == 0.0 for _ in range(5))
    assert sharpness(np.full((256, 256), 0.3)) == 0.0


def test_a_line_focus_scores_below_the_round_spot_between():
    """Astigmatism: sharp one way and blurred the other is worse than the circle of least confusion."""
    img = _scene()
    assert sharpness(_blur(img, 2.0, 2.0)) > sharpness(_blur(img, 0.3, 4.0, 0.4))


def test_log_parabola_vertex_of_a_gaussian():
    xs = np.array([-30.0, -5.0, 10.0, 40.0])
    ys = 3.0 * np.exp(-((xs - 7.0) ** 2) / (2 * 25.0 ** 2))
    assert fit_log_parabola(xs, ys) == pytest.approx(7.0, abs=1e-6)
    assert fit_log_parabola([0, 1, 2], [1.0, 0.5, 1.0]) is None


# ------------------------------------------------------------------ options and plumbing
class _Eng:
    state = None
    rate = 1e6


def test_options_and_test_image_time():
    from easyebl.microscope.control import StubMicroscope

    t = AutoTuner(_Eng(), StubMicroscope())
    o = t.options
    assert (o.focus_speed, o.focus_px, o.stig_speed, o.stig_px) == ("Slow 1", 256, "Slow 1", 256)
    assert (o.coarse_step_mm, o.coarse_range_mm, o.fine_initial_step) == (0.1, 1.0, 512)
    with pytest.raises(ValueError):
        t.set_options(focus_speed="Warp 9")
    with pytest.raises(ValueError):
        t.set_options(stig_px=8)
    t.set_options(stig_speed="Slow 2", stig_px=384)
    assert t.options.stig_speed == "Slow 2" and t.options.stig_px == 384
    # Slow 1 (5 µs) at 256²: ≈ 0.35 s per image
    assert image_time_s("Slow 1", 256, 1e6) == pytest.approx(256 * (256 * 5e-6 + 60e-6), rel=0.05)
    assert t.image_time_s("stig") > t.image_time_s("focus")
    t.close()


def test_not_connected_reports_without_changing_anything():
    from easyebl.microscope.control import StubMicroscope

    r = AutoTuner(_Eng(), StubMicroscope()).run_focus()
    assert not r.ok and "not connected" in r.message


def test_the_engine_holds_the_sem_on_external_scan_between_captures(tmp_path, monkeypatch):
    """hold_external_scan(): a run of test images claims the SEM once, releases it at the end."""
    from easyebl.app.backend import SystemBackend
    from easyebl.app.config import SystemConfig

    b = SystemBackend(SystemConfig(), eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    try:
        eng = b.engine
        eng.stop()
        time.sleep(0.6)
        calls = []
        h = eng.handle
        real_set, real_rel = h.set_external_control, h.release

        def on(v):
            calls.append(("on", v))
            return real_set(v)

        def rel(what):
            if what == "external_control":
                calls.append(("release",))
            return real_rel(what)

        monkeypatch.setattr(h, "set_external_control", on)
        monkeypatch.setattr(h, "release", rel)
        with eng.hold_external_scan():
            for _ in range(3):
                eng.capture(frames=1, nx=64, ny=64).result(20)
                time.sleep(0.3)  # stopped between images: still claimed
            assert calls == [("on", True)] and eng.external_scan_held
        deadline = time.time() + 3
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert calls == [("on", True), ("release",)], calls
    finally:
        b.close()


@pytest.fixture
def sim_tuner(tmp_path):
    """The simulators: the SEM image blurs with the simulated focus and stigmator (best: FOCUS C
    10670, FOCUS F 2300, stigmator X 2400, Y 1808)."""
    from easyebl.app.backend import SystemBackend
    from easyebl.app.config import SystemConfig

    b = SystemBackend(SystemConfig(), eds_settings_path=str(tmp_path / "eds.json"), microscope_delay_s=0.01).start()
    sem = b.microscope.sem
    deadline = time.time() + 10
    while not isinstance(sem.state.get("focus_fine"), int) and time.time() < deadline:
        time.sleep(0.05)
    b.engine.update(nx=640, ny=480)
    sem.set_stigmator("X", 2400).result(5)
    sem.set_stigmator("Y", 1808).result(5)
    sem.set_focus("C", 10670).result(5)
    events = []
    b.microscope.autotune.add_listener(lambda e: events.append(e.get("text", "")))
    yield b, sem, b.microscope.autotune, events
    b.close()


def test_the_depth_of_focus_sets_the_last_focus_step(sim_tuner):
    """Refining stops at the largest acceptable defocus: the blur 2·α·Δz below the larger of the live
    image's pixel and the diffraction-limited probe (λ of the kV), in FOCUS F counts (focus table)."""
    b, sem, t, _ = sim_tuner
    tol = t.focus_tolerance_counts()
    assert 1.0 <= tol < t.focus_step_counts() / 0.08155
    b.engine.update(nx=2560, ny=1920)  # smaller pixels: a smaller tolerance
    assert t.focus_tolerance_counts() < tol


@pytest.mark.parametrize("start", [2450, 1900, 600])
def test_auto_focus_searches_near_the_present_value_first(sim_tuner, start):
    b, sem, t, events = sim_tuner
    sem.set_focus("F", start).result(5)
    r = t.run_focus("auto")
    assert r.ok, r.message
    assert abs(r.final["focus_fine"] - 2300) <= max(2 * t.focus_tolerance_counts(), 30)
    assert not any("whole range" in e for e in events)  # a peak or a trend near the start
    assert r.steps <= 22  # the full-range grid alone is 9 images, then the halvings to 1 count


def test_auto_stigmation_searches_near_the_present_value_first(sim_tuner):
    b, sem, t, events = sim_tuner
    sem.set_focus("F", 2300).result(5)
    sem.set_stigmator("X", 2448).result(5)
    sem.set_stigmator("Y", 1728).result(5)
    r = t.run_stigmate()
    assert r.ok, r.message
    assert abs(r.final["stig_x"] - 2400) <= 32 and abs(r.final["stig_y"] - 1808) <= 32
    assert r.final["stig_x"] % 16 == 0 and r.final["stig_y"] % 16 == 0
    assert not any("whole range" in e for e in events)


def test_no_trend_near_the_start_falls_back_to_the_whole_range(sim_tuner):
    """Far out of focus (flat sharpness near the start): the whole FOCUS F range is searched."""
    b, sem, t, events = sim_tuner
    t.set_options(local_contrast=50.0)  # nothing counts as a clear trend
    sem.set_focus("F", 2450).result(5)
    r = t.run_focus("fine")
    assert r.ok and any("whole range" in e for e in events)
    assert abs(r.final["focus_fine"] - 2300) <= max(2 * t.focus_tolerance_counts(), 30)
