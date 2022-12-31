import math

import pytest

from mesh3 import Facet, Vertex, Segment, Edge, TetCell


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


def test_facet_parametrization():
    v0 = Vertex(0.0, 0.0, 0.0)
    v1 = Vertex(1.0, 0.0, 0.0)
    v2 = Vertex(0.0, 1.0, 0.0)

    f1 = Facet(v0, v1, v2)


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

