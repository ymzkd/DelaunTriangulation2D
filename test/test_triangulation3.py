import pytest

import math
from triangulation3 import Vertex, TetCell
from geometric_trait3 import Plane

def test_tetrahedron_insphere():
    gv = Vertex(math.inf, math.inf, math.inf, infinite=True)

    pt1 = Vertex(0.0, 0.0, 0.0)
    pt2 = Vertex(1.0, 0.0, 0.0)
    pt3 = Vertex(0.0, 1.0, 0.0)
    pt4 = Vertex(0.0, 0.0, 1.0)

    t1 = TetCell(pt4, pt1, pt2, pt3)
    gt1 = TetCell(gv, pt1, pt2, pt3)

    test_in1 = Vertex(0.5, 0.5, 0.5)
    test_out1 = Vertex(0.5, 0.5, -0.5)

    assert t1.is_incircumsphere(test_in1)
    assert not t1.is_incircumsphere(test_out1)
    assert gt1.is_incircumsphere(test_in1)
    assert not gt1.is_incircumsphere(test_out1)


# def test_facet():
#     v1 = Vertex(0.0, 0.0, 0.0)
#     v2 = Vertex(1.0, 0.0, 0.0)
#     v3 = Vertex(0.0, 1.0, 0.0)
#
#     p1 = Plane(v1, v2, v3)
#
#     vt1 = Vertex(0.0, 0.0, 1.0)
#     vt2 = Vertex(0.0, 0.0, -1.0)
#     p1.signed_distance(vt1)
#     p1.signed_distance(vt2)

def test_outersphere():
    v1 = Vertex(0, 0, 1)
    v2 = Vertex(0, 0, 0)
    v3 = Vertex(1, 0, 0)
    v4 = Vertex(0, 1, 0)

    tet = TetCell(v1, v2, v3, v4, mesh=None)
    sphere = tet.outer_sphere()

    vt1 = Vertex(5, 5, 5)
    vt2 = Vertex(0.5, 0.5, 0.5)
    vt3 = Vertex(-0.5, -0.5, -0.5)
    assert sphere.isinside(vt1) == False
    assert sphere.isinside(vt2) == True
    assert sphere.isinside(vt3) == False


def test_tet_orientation():
    v1 = Vertex(0, 0, 1)
    v2 = Vertex(0, 0, 0)
    v3 = Vertex(1, 0, 0)
    v4 = Vertex(0, 1, 0)
    tet = TetCell(v1, v2, v3, v4, mesh=None)

    assert tet.orient() > 0.0

    v1 = Vertex(0, 0, -1)
    v2 = Vertex(0, 0, 0)
    v3 = Vertex(1, 0, 0)
    v4 = Vertex(0, 1, 0)
    tet = TetCell(v1, v2, v3, v4, mesh=None)

    assert tet.orient() < 0.0


def test_tet_adjacent():
    v1 = Vertex(0, 0, 1)
    v2 = Vertex(0, 0, 0)
    v3 = Vertex(1, 0, 0)
    v4 = Vertex(0, 1, 0)
    v5 = Vertex(0, 0, -1)
    tet1 = TetCell(v1, v2, v3, v4, mesh=None)
    tet2 = TetCell(v3, v2, v1, v5, mesh=None)

    assert tet1.facets[3] == tet2.facets[3].opposite()
    assert tet1.facets[3].opposite() == tet2.facets[3]
