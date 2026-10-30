import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

from easyebl.litho import GDSDoseMapping, GDSError, Pattern, gds_info, parse_dose_value, read_gds, rectangle, write_gds


@pytest.fixture
def hier_gds(tmp_path):
    """Two-level hierarchy with an array reference, properties, datatypes and a path."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    sub = lib.new_cell("SUB")
    p = gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=3)
    p.set_gds_property(1, "1.25")
    sub.add(p)
    q = gdstk.rectangle((2, 0), (3, 1), layer=1, datatype=4)  # no property
    sub.add(q)
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(sub, (10, 0), columns=3, rows=1, spacing=(5, 0)))
    r = gdstk.rectangle((-5, -5), (-4, -4), layer=2, datatype=0)
    r.set_gds_property(7, "dose=0.8")
    top.add(r)
    top.add(gdstk.FlexPath([(0, -10), (10, -10)], 0.5, layer=3))
    path = tmp_path / "hier.gds"
    lib.write_gds(str(path))
    return path


def test_parse_dose_value():
    assert parse_dose_value(b"1.25\x00") == 1.25
    assert parse_dose_value("dose=0.8") == 0.8
    assert parse_dose_value("D 1e-1") == 0.1
    assert parse_dose_value("none") is None and parse_dose_value(None) is None


def test_info(hier_gds):
    info = gds_info(hier_gds)
    assert info["top"] == ["TOP"] and set(info["cells"]) == {"SUB", "TOP"}
    assert {(1, 3), (1, 4), (2, 0), (3, 0)} <= set(info["layers"])


def test_flatten_and_property_doses(hier_gds):
    st = {}
    pat = read_gds(hier_gds, dose=GDSDoseMapping("auto", table={4: 1.5}), stats=st)
    polys = pat.by_kind("area")
    assert len(polys) == 3 * 2 + 1 + 1  # 3 array copies × 2, square, path
    l13 = [p for p in polys if (p.layer, p.datatype) == (1, 3)]
    assert len(l13) == 3 and all(p.dose_factor == 1.25 for p in l13)
    assert sorted(p.bbox[0] for p in l13) == [10, 15, 20]
    assert all(p.dose_factor == 1.5 for p in polys if p.datatype == 4)  # datatype table fallback
    assert [p.dose_factor for p in polys if p.layer == 2] == [0.8]
    assert st == {"property": 4, "datatype": 3, "default": 1}
    path_poly = [p for p in polys if p.layer == 3][0]
    assert path_poly.area == pytest.approx(10 * 0.5)


def test_modes_and_filters(hier_gds):
    pat = read_gds(hier_gds, dose=GDSDoseMapping("property", property_attr=7, default=2.0))
    assert sorted({p.dose_factor for p in pat}) == [0.8, 2.0]
    pat = read_gds(hier_gds, layers=[(1, 3)], dose=GDSDoseMapping("datatype", table={3: 1.1}))
    assert len(pat) == 3 and all(p.dose_factor == 1.1 for p in pat)
    pat = read_gds(hier_gds, layers=[2], dose=GDSDoseMapping("layer", table={2: 3.0}))
    assert [p.dose_factor for p in pat] == [3.0]
    pat = read_gds(hier_gds, dose=GDSDoseMapping("none", default=1.0, value_scale=5))
    assert set(pat.dose_factors()) == {1.0}
    pat = read_gds(hier_gds, layers=[(1, 3)], dose=GDSDoseMapping("property", value_scale=0.01))
    np.testing.assert_allclose(pat.dose_factors(), 0.0125)
    sub = read_gds(hier_gds, cell="SUB")
    assert len(sub) == 2
    with pytest.raises(GDSError):
        read_gds(hier_gds, cell="NOPE")
    with pytest.raises(ValueError):
        GDSDoseMapping("magic")


def test_multiple_tops_need_cell(tmp_path):
    lib = gdstk.Library(unit=1e-6)
    lib.new_cell("A").add(gdstk.rectangle((0, 0), (1, 1)))
    lib.new_cell("B").add(gdstk.rectangle((0, 0), (1, 1)))
    f = tmp_path / "two.gds"
    lib.write_gds(str(f))
    with pytest.raises(GDSError, match="top cells"):
        read_gds(f)
    assert len(read_gds(f, cell="B")) == 1


def test_write_roundtrip(tmp_path):
    pat = Pattern([rectangle(0, 0, 1, 1, dose_factor=1.3, layer=5), rectangle(2, 0, 3, 1, dose_factor=0.7)])
    f = tmp_path / "w.gds"
    write_gds(pat, f, datatype_from_dose={1.3: 1, 0.7: 2})
    back = read_gds(f)
    assert [p.dose_factor for p in back] == [1.3, 0.7]
    assert [(p.layer, p.datatype) for p in back] == [(5, 1), (0, 2)]
    back2 = read_gds(f, dose=GDSDoseMapping("datatype", table={1: 9.0, 2: 8.0}))
    assert [p.dose_factor for p in back2] == [9.0, 8.0]
    np.testing.assert_allclose(back[0].bbox, (0, 0, 1, 1))
