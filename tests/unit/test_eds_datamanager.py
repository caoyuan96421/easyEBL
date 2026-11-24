"""EDS Data Manager model: numbering, names, dirty tracking, remeasure bookkeeping, file round trips."""
import numpy as np
import pytest

from easyebl.eds.datamanager import LINE, MAP, SPECTRUM, DataItem, DataManager, default_filename, flatten, open_file
from easyebl.eds.spectrum import EnergyCalibration, Spectrum, SpectrumMap


def _spectrum(seed=0, n=256):
    rng = np.random.default_rng(seed)
    return Spectrum(rng.poisson(20, n), EnergyCalibration(0.01, 0.02), real_time=1.5, live_time=1.2,
                    fast_count=7000, slow_count=5000, beam_kv=15.0, metadata={"at_um": [1.0, 2.0]})


def _map(nx=4, ny=3, channels=64, linescan=False):
    sm = SpectrumMap.empty(nx, ny, channels, EnergyCalibration(0.0, 0.03), (1.0, -2.0, 20.0, 15.0), 0.01,
                           metadata={"type": "linescan" if linescan else "map", "field_um": 100.0})
    sm.cube[..., 5] = np.arange(nx * ny).reshape(ny, nx)
    sm.done[:] = True
    sm.se[:] = 0.5
    sm.real_time[:] = 0.01
    return sm


def _events(dm):
    log = []
    dm.listeners.append(lambda ev, it: log.append((ev, it.number)))
    return log


def test_add_numbers_and_names():
    dm = DataManager()
    log = _events(dm)
    a = dm.new(SPECTRUM, "point", (0.1, 0.2))
    b = dm.new(MAP, "map", (0, 0, 0.5, 0.5))
    c = dm.new(LINE, "line", (-0.1, 0, 0.1, 0))
    assert [i.number for i in dm] == [1, 2, 3] and len(dm) == 3
    assert (a.title, b.title, c.title) == ("Spectrum 1", "Map 2", "Line 3")
    assert dm.of_kind(MAP) == [b] and dm.get(3) is c and dm.get(9) is None
    assert log == [("added", 1), ("added", 2), ("added", 3)]
    # items without data are never dirty
    assert not dm.dirty and a.status() == "not acquired"
    # a deleted number is not reused
    dm.remove(b)
    assert dm.new(SPECTRUM, "field", (0, 0, 1, 1)).number == 4
    assert ("removed", 2) in log
    with pytest.raises(ValueError):
        DataItem(0, MAP, "point", (0, 0))
    with pytest.raises(ValueError):
        DataItem(0, "image", "point", (0, 0))


def test_describe_uses_the_present_field():
    it = DataItem(1, SPECTRUM, "point", (0.1, -0.2))
    assert it.describe(100.0) == "point (10.00, -20.00) µm"
    assert it.describe(10.0) == "point (1.00, -2.00) µm"
    assert "5.00 × 2.50 µm" in DataItem(2, MAP, "map", (0, 0, 0.5, 0.25)).describe(10.0)
    assert DataItem(3, SPECTRUM, "field", (0, 0, 1, 1)).describe(10.0) == "whole image"


def test_dirty_tracking_and_rename():
    dm = DataManager()
    it = dm.new(SPECTRUM, "point", (0, 0))
    dm.rename(it, "no data")  # renaming an empty site: nothing to save
    assert it.title == "no data" and not dm.dirty
    dm.set_data(it, _spectrum(), final=False)  # partial: still acquiring
    assert it.acquiring and not it.dirty and it.status() == "acquiring"
    dm.set_data(it, _spectrum())
    assert it.dirty and dm.dirty_items() == [it] and it.status() == "unsaved"
    it.dirty = False
    dm.rename(it, "  Grain A ")
    assert it.name == "Grain A" and it.dirty and it.data.title == "Grain A"
    assert it.data.metadata["data_manager"]["name"] == "Grain A"
    it.dirty = False
    dm.rename(it, "Grain A")  # unchanged: stays clean
    assert not it.dirty
    dm.rename(it, "")  # back to the default label
    assert it.title == "Spectrum 1" and it.dirty


def test_remeasure_bookkeeping():
    dm = DataManager()
    it = dm.new(MAP, "map", (0.1, 0, 0.4, 0.3), conditions={"request": {"nx": 8, "ny": 6, "dwell_s": 0.005},
                                                            "field_um": 50.0})
    dm.set_data(it, _map())
    dm.rename(it, "area")
    new = dm.remeasure(it)
    assert new.number == 2 and new.remeasure_of == 1 and new.kind == MAP and new.params == it.params
    assert new.conditions == it.conditions and new.conditions is not it.conditions
    new.conditions["request"]["nx"] = 99  # a deep copy
    assert it.conditions["request"]["nx"] == 8
    assert new.data is None and not new.dirty and new.name == "area (re)"
    assert it.data is not None and it.dirty  # the original is kept as it was
    dm.set_data(new, _map())
    dm.finish(new)
    assert new.dirty and not new.acquiring and len(dm.dirty_items()) == 2
    # nothing recorded (an opened foreign spectrum, a map pixel): cannot be remeasured
    plain = dm.add(DataItem(0, SPECTRUM, "spot", data=_spectrum()))
    assert not plain.can_remeasure
    with pytest.raises(ValueError):
        dm.remeasure(plain)


def test_spectrum_save_open_round_trip(tmp_path):
    dm = DataManager()
    it = dm.new(SPECTRUM, "ellipse", (0.1, 0.2, 0.3, 0.2),
                conditions={"request": {"time_s": 2.0, "live_time": True}, "field_um": 80.0,
                            "processor": {"speed": 3, "min_energy_kev": 0.22}})
    dm.set_data(it, _spectrum())
    dm.rename(it, "inclusion")
    path = dm.save(it, str(tmp_path / default_filename(it)))
    assert path.endswith("001_inclusion.msa") and not it.dirty and it.path == path and not dm.dirty
    for p in (path, str(tmp_path / "x.npz")):
        if p.endswith(".npz"):
            dm.save(it, p)
        dm2 = DataManager()
        back = dm2.open(p)
        assert back.number == 1 and back.kind == SPECTRUM and not back.dirty and back.path == p
        assert back.name == "inclusion" and back.shape == "ellipse" and back.params == pytest.approx(it.params)
        assert back.conditions["request"] == {"time_s": 2.0, "live_time": True} and back.can_remeasure
        assert np.array_equal(back.data.counts, it.data.counts)
        assert back.data.live_time == pytest.approx(1.2) and back.data.beam_kv == pytest.approx(15.0)


def test_map_and_line_save_open_round_trip(tmp_path):
    dm = DataManager()
    m = dm.new(MAP, "map", (0.01, -0.02, 0.2, 0.15), conditions={"request": {"nx": 4, "ny": 3, "dwell_s": 0.01}})
    dm.set_data(m, _map())
    ln = dm.new(LINE, "line", (-0.1, 0.0, 0.1, 0.0), conditions={"request": {"n": 4, "dwell_s": 0.01}})
    dm.set_data(ln, _map(4, 1, linescan=True))
    paths = dm.save_all(str(tmp_path))
    assert len(paths) == 2 and all(p.endswith(".npz") for p in paths) and not dm.dirty
    bm, bl = open_file(paths[0]), open_file(paths[1])
    assert (bm.kind, bm.shape, bl.kind, bl.shape) == (MAP, "map", LINE, "line")
    assert bm.params == pytest.approx(m.params) and bl.params == pytest.approx(ln.params)
    assert np.array_equal(bm.data.cube, m.data.cube) and bm.data.shape == (3, 4)
    assert bm.conditions["request"]["nx"] == 4 and bl.can_remeasure
    # a map saved without the data manager (older file): geometry from the region and the field
    raw = _map()
    raw.save(str(tmp_path / "old.npz"))
    old = open_file(str(tmp_path / "old.npz"))
    assert old.kind == MAP and old.params == pytest.approx((0.01, -0.02, 0.2, 0.15)) and not old.can_remeasure
    # a map item saved under a name without .npz gets it
    m.dirty = True
    p = dm.save(m, str(tmp_path / "plain"))
    assert p.endswith("plain.npz")


def test_save_all_reuses_the_file_and_skips_empty(tmp_path):
    dm = DataManager()
    a = dm.new(SPECTRUM, "point", (0, 0))
    dm.set_data(a, _spectrum())
    p = dm.save(a, str(tmp_path / "mine.msa"))
    dm.rename(a, "renamed")
    dm.new(SPECTRUM, "point", (0.1, 0))  # no data
    assert dm.save_all(str(tmp_path / "elsewhere")) == [p]  # back to its own file
    assert not dm.dirty and open_file(p).name == "renamed"


def test_properties_list_the_conditions():
    it = DataItem(5, SPECTRUM, "point", (0.1, 0.2), name="P",
                  conditions={"request": {"time_s": 1.0, "live_time": False}, "field_um": 100.0,
                              "magnification": 1000, "processor": {"speed": 2, "min_energy_kev": 0.22}})
    it.data = _spectrum()
    rows = dict(flatten(it.properties()))
    assert rows["Item › number"] == "5" and rows["Item › name"] == "P"
    assert rows["Spectrum › real time (s)"] == "1.5" and rows["Spectrum › live time (s)"] == "1.2"
    assert rows["Conditions › processor › speed"] == "2" and rows["Conditions › magnification"] == "1000"
    assert rows["Conditions › request › time_s"] == "1" and rows["Metadata › at_um"] == "1, 2"
    m = DataItem(6, MAP, "map", (0, 0, 1, 1))
    m.data = _map()
    rows = dict(flatten(m.properties()))
    assert rows["Map › pixels"] == "4 × 3" and rows["Map › dwell (ms)"] == "10"


def test_open_foreign_msa(tmp_path):
    sp = _spectrum()
    sp.title = "from another program"
    sp.metadata = {}
    text = "\n".join(ln for ln in sp.to_msa().splitlines() if not ln.startswith("#EASYEBL_META"))
    (tmp_path / "f.msa").write_text(text, encoding="utf-8")
    it = open_file(str(tmp_path / "f.msa"))
    assert it.kind == SPECTRUM and it.shape == "spot" and it.name == "from another program"
    assert it.source == "f.msa" and not it.can_remeasure
