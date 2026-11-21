"""Beam energy from the spectrum (Duane–Hunt limit of the continuum)."""
import pytest

from easyebl.eds import EnergyCalibration, Spectrum
from easyebl.eds.analysis import duane_hunt, estimate_beam_kv

from . import eds_synth as S


@pytest.fixture(autouse=True)
def wide_range(monkeypatch):
    # 4096 channels × 10.2 eV = 0–41.8 keV: the end of the continuum lies inside the MCA range
    monkeypatch.setattr(S, "CHANNELS", 4096)
    S._rate_shape.cache_clear()
    yield
    S._rate_shape.cache_clear()


@pytest.mark.parametrize("comp", [{"Cu": 1.0}, {"Si": 1.0}, {"Au": 1.0}, {"Al": 0.529, "O": 0.471}])
@pytest.mark.parametrize("kv", [5.0, 10.0, 20.0, 30.0])
def test_beam_energy_from_the_continuum(comp, kv):
    r = duane_hunt(S.spectrum_of(comp, seconds=5.0, beam_kv=kv, seed=1))
    assert r is not None
    assert r.kv == pytest.approx(kv, abs=max(0.5, 3 * r.sigma_kv))


def test_not_measurable_returns_none():
    # too few counts
    assert estimate_beam_kv(S.spectrum_of({"Cu": 1.0}, seconds=0.05, beam_kv=20.0)) is None
    # the end of the continuum beyond the MCA range (0–10 keV at 20 kV)
    sp = S.spectrum_of({"Cu": 1.0}, seconds=5.0, beam_kv=20.0)
    cut = Spectrum(sp.counts[:980], EnergyCalibration(sp.calibration.offset_kev, sp.calibration.gain_kev),
                   sp.real_time, sp.live_time, beam_kv=20.0)
    assert estimate_beam_kv(cut) is None
