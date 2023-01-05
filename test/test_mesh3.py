import math

import pytest

from mesh3 import Facet, Vertex, Segment, Edge, TetCell, Polygon, Vector

def test_segment_equality():
    v0 = Vertex(0.0, 0.0, 0.0)
    v1 = Vertex(1.0, 0.0, 0.0)

    s0 = Segment(v0, v1)
    s1 = Segment(v1, v0)
    s3 = Segment(v0, v1)

    assert s0 == s1
    assert s0 == s3


def test_edge_equality():
    v0 = Vertex(0.0, 0.0, 0.0)
    v1 = Vertex(1.0, 0.0, 0.0)

    e0 = Edge(v0, v1)
    e1 = Edge(v1, v0)
    e2 = Edge(v0, v1)

    assert not e0 == e1
    assert e0 == e2


def test_isinball():
    v0 = Vertex(math.inf, math.inf, math.inf, infinite=True)
    v1 = Vertex(0.6, 0.0, 0.0)
    v2 = Vertex(0.6, 0.4, 0.45)
    v3 = Vertex(0.0, 0.4, 0.45)

    t1 = TetCell(v0, v1, v2, v3)

    vu = Vertex(0.0, 0.0, 0.0)
    assert not t1.is_incircumsphere(vu)


def test_infinite_incircumsphere():
    vg = Vertex(math.inf, math.inf, math.inf, infinite=True)
    v1 = Vertex(0.0, 0.0, 0.0)
    v2 = Vertex(1.0, 0.0, 0.0)
    v3 = Vertex(0.0, 1.0, 0.0)

    t1 = TetCell(vg, v1, v2, v3)

    vt1 = Vertex(0.1, 0.1, 0.0)
    vt2 = Vertex(-0.1, -0.1, 0.0)
    assert t1.is_incircumsphere(vt1)
    assert not t1.is_incircumsphere(vt2)


def test_polygon_area():
    vertices = [
        Vertex(4.504194, -0.536649, 0.0),
        Vertex(3.046386, -1.229997, 0.0),
        Vertex(2.441928, 0.174477, 0.0),
        Vertex(1.197458, 0.361147, 0.0),
        Vertex(1.36635, 1.97007, 0.0),
        Vertex(3.508618, 1.641174, 0.0),
        Vertex(3.337926, 0.337933, 0.0)]
    poly = Polygon(vertices)
    area_vec = poly._area_vector()
    normal_vec = poly.normal

    expected_area = 5.053641
    expected_vector = Vector(0.0, 0.0, -5.053641)

    assert pytest.approx(poly.area) == expected_area
    assert pytest.approx(area_vec.x) == expected_vector.x
    assert pytest.approx(area_vec.y) == expected_vector.y
    assert pytest.approx(area_vec.z) == expected_vector.z
    assert pytest.approx(normal_vec.length()) == 1.0
