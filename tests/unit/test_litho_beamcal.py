"""Beam calibration (scanner → beam map) from synthetic grid images."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from easyebl.litho.beamcal import (BeamCalConfig, BeamCalError, BeamCalibration, fit_grid,
                                   measure_beam_calibration, synthetic_grid)

K_TEST = 0.8 / 50.0 ** 3  # 0.8 µm radial distortion at 50 µm
TRUE = dict(sx=1.003, sy=0.997, shear=0.002, k=K_TEST, rotation_rad=0.3, shift_um=(0.23, -0.31))


def _check(cal, sx, sy, shear, k, tol_s=2e-5, tol_sh=5e-5, tol_k=0.02):
    assert cal.sx == pytest.approx(sx, abs=tol_s)
    assert cal.sy == pytest.approx(sy, abs=tol_s)
    assert cal.shear == pytest.approx(shear, abs=tol_sh)
    if k == 0:
        assert abs(cal.k) < 1e-8
    else:
        assert cal.k == pytest.approx(k, rel=tol_k)


def test_identity_grid():
    img, px = synthetic_grid(1000, 100.0)
    cal = fit_grid(img, px)
    _check(cal, 1.0, 1.0, 0.0, 0.0)
    assert abs(cal.rotation_mrad) < 0.01
    assert cal.rms_um < 0.002
    assert cal.field_um == pytest.approx(100.0)
    assert cal.pixels == (1000, 1000)
    assert 0.3 < cal.used <= 1.0


def test_recovers_scale_shear_distortion_and_nuisance():
    img, px = synthetic_grid(1000, 100.0, **TRUE)
    cal = fit_grid(img, px)
    _check(cal, TRUE["sx"], TRUE["sy"], TRUE["shear"], TRUE["k"])
    assert cal.rotation_mrad == pytest.approx(300.0, abs=0.05)
    # the shift is known modulo the lattice: compare in the grid frame
    t = TRUE["rotation_rad"]
    rot = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])  # R(−θ)
    d = rot @ (np.array(cal.shift_um) - np.array(TRUE["shift_um"]))
    assert np.abs(d - np.round(d)).max() < 1e-3
    assert cal.rms_um < 0.005


def test_noisy_squares():
    img, px = synthetic_grid(1000, 100.0, kind="squares", noise=0.3, seed=3,
                             **{**TRUE, "rotation_rad": -0.2})
    cal = fit_grid(img, px)
    _check(cal, TRUE["sx"], TRUE["sy"], TRUE["shear"], TRUE["k"], tol_s=5e-5, tol_sh=1e-4, tol_k=0.03)
    assert cal.rotation_mrad == pytest.approx(-200.0, abs=0.2)


def test_larger_image_tighter():
    img, px = synthetic_grid(2000, 200.0, sx=0.9995, sy=1.0012, shear=-0.0015, k=0.5 / 100.0 ** 3,
                             rotation_rad=0.6, noise=0.1)
    cal = fit_grid(img, px)
    _check(cal, 0.9995, 1.0012, -0.0015, 0.5 / 100.0 ** 3, tol_s=1e-5, tol_sh=2e-5, tol_k=0.01)
    assert cal.rotation_mrad == pytest.approx(600.0, abs=0.05)


def test_large_rotation_is_modulo_90_degrees():
    img, px = synthetic_grid(800, 100.0, rotation_rad=math.radians(70))
    cal = fit_grid(img, px)
    _check(cal, 1.0, 1.0, 0.0, 0.0)
    assert cal.rotation_mrad == pytest.approx(math.radians(70 - 90) * 1e3, abs=0.05)


def test_blank_and_noise_images_are_refused():
    rng = np.random.default_rng(1)
    with pytest.raises(BeamCalError):
        fit_grid(rng.standard_normal((800, 800)).astype(np.float32), 0.125)
    with pytest.raises(BeamCalError):
        fit_grid(np.zeros((800, 800), np.float32), 0.125)


def test_poor_fit_carries_result():
    img, px = synthetic_grid(800, 100.0, noise=1.0)
    with pytest.raises(BeamCalError) as e:
        fit_grid(img, px, max_rms_um=1e-6)
    assert isinstance(e.value.result, BeamCalibration)
    assert e.value.result.sx == pytest.approx(1.0, abs=1e-3)


def test_forward_inverse_round_trip():
    cal = BeamCalibration(sx=1.003, sy=0.997, shear=0.002, k=K_TEST, field_um=100.0)
    ang = np.linspace(0, 2 * np.pi, 37)
    for r in (1.0, 10.0, 50.0):
        x = r * np.column_stack([np.cos(ang), np.sin(ang)])
        back = cal.to_xcs(cal.to_bcs(x))
        err = np.linalg.norm(back - x, axis=1).max()
        rb = np.linalg.norm(cal.to_bcs(x), axis=1).max()
        assert err <= 3.5 * K_TEST ** 2 * rb ** 5 + 1e-12  # the one-step inverse: ≈ 3k²r⁵
    assert cal.to_bcs([0.0, 0.0]).shape == (2,)
    np.testing.assert_allclose(cal.linear_to_xcs(cal.linear_to_bcs([3.0, -4.0])), [3.0, -4.0])
    assert cal.distortion_at(50.0) == pytest.approx(0.8)
    assert cal.edge_distortion_um() == pytest.approx(K_TEST * (100 / math.sqrt(2)) ** 3)
    assert "ppm" in cal.describe()


def test_identity():
    cal = BeamCalibration.identity()
    assert cal.is_identity
    np.testing.assert_array_equal(cal.L, np.eye(2))
    x = np.array([[12.0, -7.0], [0.5, 3.0]])
    np.testing.assert_array_equal(cal.to_bcs(x), x)
    np.testing.assert_array_equal(cal.to_xcs(x), x)
    assert not BeamCalibration(k=1e-9).is_identity


class _Cond:
    def __init__(self, m):
        self.m = m

    def to_dict(self):
        return {"m": self.m}

    @classmethod
    def from_dict(cls, d):
        return cls(d["m"])


def test_dict_round_trip():
    cal = BeamCalibration(sx=1.001, sy=0.999, shear=1e-4, k=2e-7, rotation_mrad=12.0, shift_um=(0.1, 0.2),
                          rms_um=0.003, pitch_um=1.0, field_um=500.0, pixels=(5000, 5000), condition=_Cond(800),
                          position="grid", name="x800")
    d = json.loads(json.dumps(cal.to_dict()))
    back = BeamCalibration.from_dict(d, condition_from_dict=_Cond.from_dict)
    assert back.condition.m == 800
    for f in ("sx", "sy", "shear", "k", "shift_um", "pixels", "created", "position", "name", "field_um"):
        assert getattr(back, f) == getattr(cal, f)
    assert BeamCalibration.from_dict(d).condition == {"m": 800}
    cfg = BeamCalConfig(pitch_um=2.0, pixels=3000, position="cal")
    assert BeamCalConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


class _FakeHw:
    def __init__(self, **kw):
        self.kw = kw
        self.calls = []

    def field_um(self):
        return 100.0

    def capture_image(self, width_um, px, frames):
        self.calls.append((width_um, px, frames))
        return synthetic_grid(px, width_um, **self.kw)


def test_measure_beam_calibration():
    hw = _FakeHw(sx=1.002, sy=0.999, k=K_TEST)
    logs, images, checks = [], [], []
    cfg = BeamCalConfig(pixels=1000, frames=2, position="grid")
    cal = measure_beam_calibration(hw, cfg, condition="cond", log=logs.append, check=lambda: checks.append(1),
                                   on_image=lambda img, px: images.append(img.shape))
    assert hw.calls == [(100.0, 1000, 2)]
    assert images == [(1000, 1000)]
    assert cal.condition == "cond" and cal.position == "grid"
    _check(cal, 1.002, 0.999, 0.0, K_TEST)
    assert len(checks) >= 2 and any("beam calibration" in s for s in logs)


def test_measure_aborts_via_check():
    def stop():
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        measure_beam_calibration(_FakeHw(), BeamCalConfig(pixels=500), None, log=None, check=stop)
