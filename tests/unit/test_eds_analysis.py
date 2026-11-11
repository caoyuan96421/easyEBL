"""EDS analysis: background, calibration, AutoID, fitting, quantification, maps, files."""
import numpy as np
import pytest

from easyebl.eds import EnergyCalibration, Spectrum, load_spectrum
from easyebl.eds import analysis as A

from .eds_synth import spectrum_of, synthetic_map, true_calibration

STAINLESS = {"Fe": 0.70, "Cr": 0.19, "Ni": 0.09, "Mn": 0.02}


@pytest.fixture(scope="module")
def steel():
    return spectrum_of(STAINLESS, seconds=15)


@pytest.fixture(scope="module")
def cu():
    return spectrum_of({"Cu": 1.0}, seconds=8)


def test_energy_calibration_roundtrip():
    c = EnergyCalibration(-0.01, 0.0101, 2e-8)
    k = np.array([0.0, 100.5, 1500.0])
    np.testing.assert_allclose(c.channel(c.energy(k)), k, atol=1e-9)
    assert EnergyCalibration.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        EnergyCalibration(0, 0)


def test_snip_background_follows_the_continuum(cu):
    bg = A.snip_background(cu.counts, cu.calibration)
    e = cu.energies
    cont = (e > 2.5) & (e < 6.0)  # line-free for Cu
    ratio = bg[cont].sum() / cu.counts[cont].sum()
    assert 0.8 < ratio < 1.05
    pk = (e > 7.95) & (e < 8.15)
    assert bg[pk].sum() < 0.1 * cu.counts[pk].sum()


def test_peak_search_and_two_point_calibration(cu):
    wrong = cu.with_calibration(EnergyCalibration(0.06, 0.0097))
    peaks = A.find_peaks(wrong)
    assert len(peaks) >= 2
    pairs = A.auto_assign(wrong, A.standard_lines("Cu (L + Kα)"))
    assert [lab for lab, *_ in pairs] == ["Cu La1", "Cu Ka1"]
    fit = A.fit_calibration([p.channel for *_, p in pairs], [e for _, e, _ in pairs])
    true = true_calibration()
    for e in (1.0, 5.0, 8.0, 15.0):
        k = true.channel(e)
        assert abs(fit.calibration.energy(k) - e) < 0.006  # < 6 eV over the range
    # resolution from the Cu Kα width
    ka = min(peaks, key=lambda p: abs(wrong.calibration.energy(p.channel) - 7.78))
    fwhm_kev = ka.fwhm_ch * fit.calibration.gain_kev
    mn, fano = A.fit_resolution([(8.04, fwhm_kev)])
    assert 120 < mn < 150 and fano == pytest.approx(0.115)


def test_one_point_and_quadratic_fits():
    f1 = A.fit_calibration([800.0], [8.0])
    assert f1.calibration.gain_kev == pytest.approx(0.01) and f1.calibration.offset_kev == 0
    ch = np.array([50.0, 300.0, 800.0, 1500.0])
    true = EnergyCalibration(-0.02, 0.01, 3e-7)
    f2 = A.fit_calibration(ch, true.energy(ch), order=2)
    assert f2.rms_ev < 1e-6 and f2.calibration.quad_kev == pytest.approx(3e-7, rel=1e-6)


@pytest.mark.parametrize("comp", [STAINLESS, {"Ti": 0.6, "O": 0.4}, {"Na": 0.39, "Cl": 0.61},
                                  {"Cu": 0.5, "Au": 0.5}])
def test_autoid_and_quant(comp):
    sp = spectrum_of(comp, seconds=15)
    found = {i.element for i in A.identify_elements(sp)}
    assert found == set(comp), found
    fit = A.fit_spectrum(sp, sorted(found))
    assert fit.chi2_reduced < 2.0
    q = A.quantify(fit, sp.beam_kv, oxygen_by_stoichiometry="O" in comp)
    wt = q.wt()
    assert sum(wt.values()) == pytest.approx(100.0)
    for el, w in comp.items():
        assert wt[el] == pytest.approx(100 * w, abs=max(1.5, 0.08 * 100 * w)), q.table()


def test_autoid_short_acquisition_finds_majors():
    sp = spectrum_of(STAINLESS, seconds=0.5)
    found = {i.element for i in A.identify_elements(sp)}
    assert {"Fe", "Cr"} <= found <= set(STAINLESS)


def test_quant_oxide_mode_and_table():
    sp = spectrum_of({"Al": 0.529, "O": 0.471}, seconds=10)  # Al2O3
    fit = A.fit_spectrum(sp, ["Al", "O"])
    q = A.quantify(fit, sp.beam_kv, oxygen_by_stoichiometry=True)
    al = [r for r in q.rows if r.element == "Al"][0]
    assert al.oxide == "Al2O3" and al.oxide_wt == pytest.approx(100.0, abs=0.01)
    assert q.wt()["Al"] == pytest.approx(52.93, abs=0.05)  # stoichiometric by construction
    assert "Al2O3" in q.table()


def test_escape_peak_model():
    f = A.escape_fraction(np.array([1.0, 2.0, 5.9, 8.0, 15.0]))
    assert f[0] == 0 and f[1] > f[2] > f[3] > f[4] > 0
    assert 0.003 < f[2] < 0.02  # Mn Kα: ~1 %


def test_maps_fitted_quant_and_phases():
    lab = np.zeros((10, 12), int)
    lab[:, 6:] = 1
    lab[3:7, 2:5] = 2
    comps = [{"Fe": 1.0}, {"Cu": 1.0}, {"Ti": 0.6, "O": 0.4}]
    sm = synthetic_map(lab, comps, dwell_s=0.05)
    wm = {m.label: m.data for m in A.window_maps(sm, ["Fe", "Cu", "Ti"])}
    assert wm["Fe Kα"][lab == 0].mean() > 10 * max(wm["Fe Kα"][lab == 1].mean(), 1)
    fm, fit = A.fitted_maps(sm, ["Fe", "Cu", "Ti", "O"])
    fmd = {m.label: m.data for m in fm}
    assert fmd["Cu K"][lab == 1].mean() > 20 * abs(fmd["Cu K"][lab == 0].mean())
    assert fmd["Ti K"][lab == 2].mean() > 20 * abs(fmd["Ti K"][lab != 2].mean())
    qm, q = A.quant_maps(sm, ["Fe", "Cu", "Ti"])
    qd = {m.label: m.data for m in qm}
    assert np.nanmean(qd["Fe"][lab == 0]) > 85 and np.nanmean(qd["Cu"][lab == 1]) > 85
    assert np.nanmean(qd["Ti"][lab == 2]) > 85
    qo, _ = A.quant_maps(sm, ["Fe", "Cu", "Ti"], oxygen_by_stoichiometry=True)
    qod = {m.label: m.data for m in qo}
    assert np.nanmean(qod["Ti"][lab == 2]) == pytest.approx(59.9, abs=4)  # TiO2 (O by stoichiometry)
    ph = A.phase_map([m for m in fm if m.label in ("Fe K", "Cu K", "Ti K")], k=3)
    # each true phase maps to one cluster (up to relabelling)
    for t in range(3):
        vals, counts = np.unique(ph.labels[lab == t], return_counts=True)
        assert counts.max() >= 0.9 * (lab == t).sum()
    b = A.bin_map(sm, 2)
    assert b.shape == (5, 6) and b.cube.sum() == sm.cube[:10, :12].sum()


def test_msa_roundtrip_and_foreign_header(tmp_path, steel):
    p = tmp_path / "s.msa"
    steel.save(p)
    s2 = load_spectrum(p)
    assert np.array_equal(s2.counts, steel.counts) and s2.real_time == pytest.approx(steel.real_time)
    assert s2.calibration.offset_kev == pytest.approx(steel.calibration.offset_kev)
    # a minimal EMSA file from another program (eV units, XY data)
    txt = "\n".join(["#FORMAT : EMSA/MAS Spectral Data File", "#NPOINTS : 3", "#XUNITS : eV", "#XPERCHAN : 10.0",
                     "#OFFSET : -20.0", "#DATATYPE : XY", "#SPECTRUM :", "-20, 5", "-10, 7", "0, 9", "#ENDOFDATA :"])
    s3 = Spectrum.from_msa(txt)
    assert list(s3.counts) == [5, 7, 9] and s3.calibration.gain_kev == pytest.approx(0.01)
    assert s3.calibration.offset_kev == pytest.approx(-0.02)


# real DP5 spectra of the EDS phantom (USB-6363 AO3 → DP5, 2026-11-11, scripts/hw_eds_test.py)
_DP5 = {"3_cu_calibrated.msa": {"Cu"}, "4_0_Si_substrate.msa": {"Si"}, "4_1_Al_pad.msa": {"Al"},
        "4_2_Cu_grid.msa": {"Cu"}, "4_3_stainless.msa": {"Fe", "Cr", "Ni"}, "4_4_TiO2.msa": {"Ti", "O"},
        "4_5_NaCl.msa": {"Na", "Cl"}, "4_6_Au.msa": {"Au"}}


@pytest.mark.parametrize("name", sorted(_DP5))
def test_autoid_on_real_dp5_spectra(name):
    from pathlib import Path

    from easyebl.xray.generation import Detector

    sp = load_spectrum(Path(__file__).resolve().parents[1] / "data" / "dp5_phantom" / name)
    found = {i.element for i in A.identify_elements(sp, detector=Detector(fwhm_mn_ev=156))}
    assert found == _DP5[name]
