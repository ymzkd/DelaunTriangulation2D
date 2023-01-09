import pytest

from geometric_trait3 import Plane, Point3, Triangle, Vector3


def test_vector_division():
    v1 = Vector3(2.0, 3.0, 4.0) / 2.0
    v2 = Vector3(2.0, 3.0, 4.0) / 2

    vt1 = Vector3(1.0, 1.5, 2.0)

    assert pytest.approx(v1.x) == vt1.x
    assert pytest.approx(v1.y) == vt1.y
    assert pytest.approx(v1.z) == vt1.z

    assert pytest.approx(v2.x) == vt1.x
    assert pytest.approx(v2.y) == vt1.y
    assert pytest.approx(v2.z) == vt1.z


def test_plane_side():
    origin = Point3(0.0, 0.0, 0.0)
    pt_x = Point3(1.0, 0.0, 0.0)
    pt_y = Point3(0.0, 1.0, 0.0)

    pln = Plane(origin, pt_x, pt_y)

    point_upside = Point3(0.0, 0.0, 1.0)
    point_downside = Point3(0.0, 0.0, -1.0)

    assert pln.signed_distance(point_upside) > 0.0
    assert pln.signed_distance(point_downside) < 0.0


def test_triangle_hash():
    pt1 = Point3(0.0, 0.0, 0.0)
    pt2 = Point3(1.0, 0.0, 0.0)
    pt3 = Point3(0.0, 1.0, 0.0)
    t1 = Triangle(pt1, pt2, pt3)
    t1a = Triangle(pt2, pt3, pt1)

    assert t1.__hash__() == t1a.__hash__()