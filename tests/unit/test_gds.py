import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

from easyebl.litho import GDSError, Pattern, gds_info, read_gds, rectangle, write_gds
from easyebl.litho.gds import dose_summary, dose_text, dose_warnings, load_library


@pytest.fixture
def hier_gds(tmp_path):
    """Two-level hierarchy with an array reference, dose datatypes and a path: layer 1 dose-encoded
    (1/1250, 1/500 in SUB, 1/0 in TOP: dose 0), layer 2 and 3 uniform (datatype 0)."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    sub = lib.new_cell("SUB")
    sub.add(gdstk.rectangle((0, 0), (1, 2), layer=1, datatype=1250))
    sub.add(gdstk.rectangle((2, 0), (3, 1), layer=1, datatype=500))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(sub, (10, 0), columns=3, rows=1, spacing=(5, 0)))
    top.add(gdstk.rectangle((-5, -5), (-4, -4), layer=2, datatype=0))
    top.add(gdstk.rectangle((-8, -5), (-7, -4), layer=1, datatype=0))  # datatype 0 on an encoded layer
    top.add(gdstk.FlexPath([(0, -10), (10, -10)], 0.5, layer=3))
    path = tmp_path / "hier.gds"
    lib.write_gds(str(path))
    return path


def test_info(hier_gds):
    info = gds_info(hier_gds)
    assert info["top"] == ["TOP"] and set(info["cells"]) == {"SUB", "TOP"}
    assert {(1, 1250), (1, 500), (1, 0), (2, 0), (3, 0)} <= set(info["layers"])


def test_flatten_and_the_dose_per_layer(hier_gds):
    """Layer by layer: datatype 0 only → uniform (1); else datatype / 1000, datatype 0 → 0."""
    st = {}
    pat = read_gds(hier_gds, stats=st)
    polys = pat.by_kind("area")
    assert len(polys) == 3 * 2 + 1 + 1 + 1  # 3 array copies × 2, two squares, path
    l1 = [p for p in polys if (p.layer, p.datatype) == (1, 1250)]
    assert len(l1) == 3 and all(p.dose_factor == 1.25 for p in l1)
    assert sorted(p.bbox[0] for p in l1) == [10, 15, 20]
    assert all(p.dose_factor == 0.5 for p in polys if p.datatype == 500)
    assert [p.dose_factor for p in polys if (p.layer, p.datatype) == (1, 0)] == [0.0]
    assert [p.dose_factor for p in polys if p.layer in (2, 3)] == [1.0, 1.0]
    assert st == {"datatype": 7, "uniform": 2}
    path_poly = [p for p in polys if p.layer == 3][0]
    assert path_poly.area == pytest.approx(10 * 0.5)
    # a cell alone: the file decides the encoded layers (SUB has no datatype 0, but no matter)
    sub = read_gds(hier_gds, cell="SUB")
    assert len(sub) == 2 and sorted(p.dose_factor for p in sub) == [0.5, 1.25]


def test_the_dose_summary(hier_gds):
    lib = load_library(str(hier_gds))
    s = dose_summary(lib)
    assert s[1] == {"levels": [0.0, 0.5, 1.25], "uniform": False, "zero": 1}
    assert s[2] == {"levels": [1.0], "uniform": True, "zero": 0}
    assert dose_warnings(s) == ["layer 1: 1 shape(s) with datatype 0 among dose-encoded datatypes → dose 0, "
                                "not written"]
    text = dose_text(s)
    assert "layer 1: dose factor 0.500 … 1.250 (2 levels); 1 shape(s) datatype 0 → dose 0 ⚠" in text
    assert "layer 2: uniform (datatype 0)" in text
    assert dose_summary(lib, [2]) == {2: s[2]} and not dose_warnings(dose_summary(lib, [(1, 500)]))


def test_filters(hier_gds):
    pat = read_gds(hier_gds, layers=[(1, 1250)])
    assert len(pat) == 3 and all(p.dose_factor == 1.25 for p in pat)
    pat = read_gds(hier_gds, layers=[2])
    assert [p.dose_factor for p in pat] == [1.0]
    with pytest.raises(GDSError):
        read_gds(hier_gds, cell="NOPE")


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
    """The dose factor is written as the datatype (× 1000): it reads back the same."""
    pat = Pattern([rectangle(0, 0, 1, 1, dose_factor=1.3, layer=5), rectangle(2, 0, 3, 1, dose_factor=0.7)])
    f = tmp_path / "w.gds"
    write_gds(pat, f)
    back = read_gds(f)
    assert [p.dose_factor for p in back] == [1.3, 0.7]
    assert [(p.layer, p.datatype) for p in back] == [(5, 1300), (0, 700)]
    np.testing.assert_allclose(back[0].bbox, (0, 0, 1, 1))
    with pytest.raises(GDSError, match="datatype"):
        write_gds(Pattern([rectangle(0, 0, 1, 1, dose_factor=70.0)]), tmp_path / "x.gds")
