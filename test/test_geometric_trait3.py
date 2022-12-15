import pytest

from geometric_trait3 import Plane, Point, Triangle


def test_plane_side():
    origin = Point(0.0, 0.0, 0.0)
    pt_x = Point(1.0, 0.0, 0.0)
    pt_y = Point(0.0, 1.0, 0.0)

    pln = Plane(origin, pt_x, pt_y)

    point_upside = Point(0.0, 0.0, 1.0)
    point_downside = Point(0.0, 0.0, -1.0)

    assert pln.signed_distance(point_upside) > 0.0
    assert pln.signed_distance(point_downside) < 0.0


def test_triangle_hash():
    pt1 = Point(0.0, 0.0, 0.0)
    pt2 = Point(1.0, 0.0, 0.0)
    pt3 = Point(0.0, 1.0, 0.0)
    t1 = Triangle(pt1, pt2, pt3)
    t1a = Triangle(pt2, pt3, pt1)

    assert t1.__hash__() == t1a.__hash__()