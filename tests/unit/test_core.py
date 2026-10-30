import math

import numpy as np
import pytest

from easyebl.core import (
    AffineTransform,
    FieldCalibration,
    OutOfFieldError,
    OutputChunk,
    chunk_arrays,
    concat_chunks,
    rechunk,
    total_samples,
)


# ------------------------------------------------------------------ transforms
class TestAffineTransform:
    def test_identity_and_apply_shapes(self):
        t = AffineTransform.identity()
        assert t.is_identity()
        np.testing.assert_allclose(t.apply([1.0, 2.0]), [1.0, 2.0])
        assert t.apply(np.zeros((5, 2))).shape == (5, 2)

    def test_compose_order(self):
        tr = AffineTransform.translation(1, 0)
        rot = AffineTransform.rotation(math.pi / 2)
        # rot first, then translation
        np.testing.assert_allclose((tr @ rot).apply([1, 0]), [1, 1], atol=1e-12)
        # translation first, then rot
        np.testing.assert_allclose((rot @ tr).apply([1, 0]), [0, 2], atol=1e-12)

    def test_inverse_roundtrip(self):
        t = AffineTransform.from_params(3, -2, 0.1, 1.01, 0.98, 0.02)
        p = np.random.default_rng(0).normal(size=(10, 2))
        np.testing.assert_allclose(t.inverse().apply(t.apply(p)), p, atol=1e-12)

    @pytest.mark.parametrize("params", [dict(tx=1, ty=2, rotation_rad=0.01, sx=1.002, sy=0.997, shear=0.0),
                                        dict(tx=-5, ty=0, rotation_rad=-0.3, sx=2, sy=2, shear=0.0)])
    def test_decompose_roundtrip(self, params):
        d = AffineTransform.from_params(**params).decompose()
        for k, v in params.items():
            assert d[k] == pytest.approx(v, abs=1e-9)

    def test_immutable_and_validation(self):
        t = AffineTransform.identity()
        with pytest.raises(ValueError):
            t.matrix[0, 0] = 2
        with pytest.raises(ValueError):
            AffineTransform(np.eye(2))
        with pytest.raises(ValueError):
            AffineTransform(np.ones((3, 3)))

    def test_list_roundtrip(self):
        t = AffineTransform.from_params(1, 2, 0.3)
        assert AffineTransform.from_list(t.to_list()) == t


# ----------------------------------------------------------------- calibration
class TestFieldCalibration:
    def test_nominal_scaling(self):
        cal = FieldCalibration(100.0, 10.0)
        np.testing.assert_allclose(cal.um_to_volts([[50, -50], [0, 0]]), [[10, -10], [0, 0]])
        assert cal.volts_per_um == pytest.approx(0.2)

    def test_roundtrip_with_correction_and_inversion(self):
        cal = FieldCalibration(80.0, 5.0, AffineTransform.from_params(0.5, -0.2, 0.01, 1.01, 0.99), invert_y=True)
        p = np.random.default_rng(1).uniform(-30, 30, (20, 2))
        np.testing.assert_allclose(cal.volts_to_um(cal.um_to_volts(p)), p, atol=1e-10)
        v = FieldCalibration(80.0, 5.0, invert_y=True).um_to_volts([[0, 10]])
        assert v[0, 1] < 0

    def test_out_of_field(self):
        cal = FieldCalibration(100.0, 10.0)
        with pytest.raises(OutOfFieldError):
            cal.um_to_volts([[0, 0], [50.1, 0]])
        cal.um_to_volts([[50.1, 0]], check=False)
        np.testing.assert_array_equal(cal.contains([[0, 0], [49, 49], [51, 0]]), [True, True, False])
        np.testing.assert_array_equal(cal.contains([[49, 0]], margin_um=2), [False])

    def test_additional_transform_prepends(self):
        cal = FieldCalibration(100.0, 10.0)
        t = AffineTransform.translation(1.0, 0.0)
        c2 = cal.with_additional_transform(t)
        np.testing.assert_allclose(c2.um_to_volts([[0, 0]]), [[0.2, 0]])

    def test_json_and_validation(self):
        cal = FieldCalibration(123.0, 7.0, AffineTransform.from_params(1, 2, 0.1), invert_x=True, name="x")
        c2 = FieldCalibration.from_json(cal.to_json())
        assert c2.field_size_um == 123.0 and c2.invert_x and c2.correction == cal.correction
        with pytest.raises(ValueError):
            FieldCalibration(0.0)
        with pytest.raises(ValueError):
            FieldCalibration(10.0, -1)

    def test_dac_step(self):
        assert FieldCalibration(100.0, 10.0).dac_step_um(16, 10.0) == pytest.approx(100.0 / 65536)


# -------------------------------------------------------------------- waveform
class TestWaveform:
    def _chunk(self, n, start=0, marker=False):
        x = np.arange(start, start + n, dtype=float)
        mk = {"sync": x.astype(int) % 2 == 0} if marker else {}
        return OutputChunk(x, -x, x.astype(int) % 3 == 0, mk)

    def test_validation(self):
        with pytest.raises(ValueError):
            OutputChunk(np.zeros(3), np.zeros(2), np.zeros(3, bool))
        with pytest.raises(ValueError):
            OutputChunk(np.zeros(3), np.zeros(3), np.zeros(3, bool), {"m": np.zeros(2, bool)})

    def test_constant_and_slice(self):
        c = OutputChunk.constant(5, 1.0, -2.0, True, {"m": False})
        assert len(c) == 5 and c.blank.all() and not c.markers["m"].any()
        assert c.max_abs_voltage() == 2.0
        assert len(c.slice(1, 3)) == 2

    def test_concat_fills_missing_markers(self):
        a = self._chunk(3, marker=True)
        b = self._chunk(2, 3)
        c = concat_chunks([a, OutputChunk.empty(), b])
        assert len(c) == 5
        np.testing.assert_array_equal(c.markers["sync"], [True, False, True, False, False])

    @pytest.mark.parametrize("size", [1, 3, 7, 100])
    def test_rechunk_preserves_data(self, size):
        chunks = [self._chunk(n, s, marker=True) for n, s in [(5, 0), (0, 5), (11, 5), (2, 16)]]
        out = list(rechunk(chunks, size))
        assert all(len(c) == size for c in out[:-1]) and 0 < len(out[-1]) <= size
        merged = concat_chunks(out)
        np.testing.assert_array_equal(merged.x, np.arange(18))
        np.testing.assert_array_equal(merged.markers["sync"], np.arange(18) % 2 == 0)
        assert total_samples(out) == 18

    def test_chunk_arrays(self):
        x = np.arange(10.0)
        out = list(chunk_arrays(x, x, x > 5, 4))
        assert [len(c) for c in out] == [4, 4, 2]
        with pytest.raises(ValueError):
            list(rechunk([], 0))
