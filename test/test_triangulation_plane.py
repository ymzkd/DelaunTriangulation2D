import pytest

from mesh3 import Plane, Vertex3, Edge3, Facet3, Sphere
from geometric_trait3 import Vector3


def test_plane_initialize():
    origin = Vertex3(0.0, 0.0, 0.0)
    pt_x = Vertex3(1.0, 0.0, 0.0)
    pt_y = Vertex3(0.0, 1.0, 0.0)

    pln = Plane(origin, pt_x, pt_y)

    assert pytest.approx(pln.ex.length()) == 1.0
    assert pytest.approx(pln.ey.length()) == 1.0
    assert pytest.approx(pln.ey * pln.ex) == 0.0

    expected_ez = Vector3(0.0, 0.0, 1.0)
    assert pytest.approx(expected_ez * pln.ez) == 1.0


def test_plane_edge():
    origin = Vertex3(0, 0, 0)
    pt_x = Vertex3(1.0, 0.0, 0.0)
    pt_y = Vertex3(0.0, 1.0, 0.0)
    e = Edge3(pt_x, pt_y)
    pln = Plane(origin, pt_x, pt_y)

    assert e.ispt_rightside(Vertex3(2.0, 2.0, 0.0), pln)
    assert not(e.ispt_rightside(Vertex3(0.1, 0.1, 0.0), pln))


def test_plane_facet():
    origin = Vertex3(0, 0, 0)
    pt_x = Vertex3(1.0, 0.0, 0.0)
    pt_y = Vertex3(0.0, 1.0, 0.0)
    f = Facet3(origin, pt_x, pt_y)

    expected_area = 0.5
    expected_normal = Vertex3(0.0, 0.0, 1.0)

    assert pytest.approx(f.area()) == expected_area
    assert pytest.approx(f.normal() * expected_normal) == 1.0


def test_plane_diametric_ball():
    origin = Vertex3(0, 0, 0)
    pt_x = Vertex3(1.0, 0.0, 0.0)
    pt_y = Vertex3(0.0, 1.0, 0.0)
    f = Facet3(origin, pt_x, pt_y)
    sphere = f.diametric_ball()

    expected_in1 = Vertex3(0.709997, 0.637645, 0.478907)
    expected_in2 = Vertex3(0.709997, 0.111057, 0.478907)

    expected_out1 = Vertex3(0.709997, 0.637645, 0.748269)
    expected_out2 = Vertex3(0.709997, 0.111057, 0.676096)

    assert sphere.isinside(expected_in1)
    assert sphere.isinside(expected_in2)
    assert not sphere.isinside(expected_out1)
    assert not sphere.isinside(expected_out2)

