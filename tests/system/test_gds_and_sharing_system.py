"""GDS → exposure, and several subsystems sharing one DAQ manager."""
import threading
import time

import numpy as np
import pytest

from easyebl.alignment import MarkerAligner
from easyebl.hal import DeviceBusyError, SimParams, make_demo_sample
from easyebl.imaging import RasterConfig, RasterImager
from easyebl.litho import (
    ExposureParameters,
    GDSDoseMapping,
    LithoWriter,
    Pattern,
    VectorScanSettings,
    read_gds,
    rectangle,
    write_gds,
)

pytestmark = pytest.mark.system
gdstk = pytest.importorskip("gdstk")

I = 0.25e-9


def dose_in(sim, poly, shrink=2):
    from scipy import ndimage

    xs, ys = sim.dose.coords()
    X, Y = np.meshgrid(xs, ys)
    inside = poly.contains(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    core = ndimage.binary_erosion(inside, iterations=shrink)
    return float(np.median(sim.dose.dose_uC_cm2()[core]))


@pytest.mark.parametrize("mode", ["property", "datatype"])
def test_beamer_like_gds_doses(make_manager, calibration, tmp_path, mode):
    # three dose classes, carried both as properties and as datatypes
    pads = [rectangle(-30 + 22 * k, -5, -18 + 22 * k, 5, dose_factor=f, layer=1) for k, f in enumerate([0.8, 1.0, 1.4])]
    f = tmp_path / "pec.gds"
    write_gds(Pattern(pads), f, datatype_from_dose={0.8: 10, 1.0: 11, 1.4: 12})
    mapping = GDSDoseMapping("property") if mode == "property" else GDSDoseMapping(
        "datatype", table={10: 0.8, 11: 1.0, 12: 1.4})
    pat = read_gds(f, dose=mapping)
    assert sorted(pat.dose_factors()) == [0.8, 1.0, 1.4]

    m = make_manager(SimParams(beam_current_a=I, lag_tau_s=3e-6, dose_pixel_um=0.1))
    w = LithoWriter(m.get_handle("gds"), calibration, VectorScanSettings(rate=500e3))
    rep = w.expose(pat, ExposureParameters(beam_current_a=I, area_dose_uC_cm2=300, area_step_um=0.05))
    assert not rep.aborted
    for p in pat:
        assert dose_in(m.backend, p) == pytest.approx(300 * p.dose_factor, rel=0.03)


def test_demo_gds_generator_roundtrip(tmp_path, make_manager, calibration):
    from easyebl.litho.gui_gds import make_demo_gds

    path = make_demo_gds(str(tmp_path / "demo.gds"))
    pat = read_gds(path)
    assert len(pat) == 9 and set(np.round(pat.dose_factors(), 3)) == {1.0, 1.25, 1.6}
    assert all(calibration.contains(np.vstack([e.points for e in pat])))


def test_imaging_alignment_litho_share_one_device(make_manager, calibration):
    nominal = np.array([(-35.0, -35.0), (35.0, -35.0), (35.0, 35.0), (-35.0, 35.0)])
    m = make_manager(SimParams(beam_current_a=I, noise_sigma=0.03), sample=make_demo_sample(marks=nominal + [0.5, 0.3]))
    imager = RasterImager(m.get_handle("imaging"), calibration)
    aligner = MarkerAligner(RasterImager(m.get_handle("alignment"), calibration),
                            RasterConfig(nx=96, ny=96, dwell_s=4e-6, preline_s=30e-6), window_um=20)
    writer = LithoWriter(m.get_handle("litho"), calibration, VectorScanSettings(rate=500e3))
    assert len(m.handles()) == 3

    res = aligner.align(nominal, "translation")
    np.testing.assert_allclose(res.transform.offset, [0.5, 0.3], atol=0.05)

    # while lithography streams (real-time), imaging is refused, then succeeds when it waits
    m.backend.params.realtime = True
    params = ExposureParameters(beam_current_a=I, area_step_um=0.05)
    done = {}
    t = threading.Thread(target=lambda: done.setdefault(
        "rep", writer.expose(Pattern([rectangle(-2, -2, 2, 2)]), params, transform=res.transform)))
    t.start()
    deadline = time.time() + 2
    while not m.is_busy and time.time() < deadline:
        time.sleep(0.001)
    assert m.active_owner == "litho"
    small = RasterConfig(nx=32, ny=32, width_um=10)
    with pytest.raises(DeviceBusyError):
        imager.acquire(small)
    img = imager.acquire(small, timeout=10)  # waits for the exposure to finish
    t.join(10)
    assert not done["rep"].aborted and not img.aborted
    assert m.active_owner is None and m.beam_state.blank
