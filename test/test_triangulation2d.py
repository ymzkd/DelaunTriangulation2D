import math

import pytest

from triangulation2 import Facet, Vertex


def test_triangle_orientation():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(1.0, 1.0)

    t1 = Facet(v0, v1, v2)
    t2 = Facet(v0, v2, v1)
    assert t1.orientation() > 0.0
    assert t2.orientation() < 0.0


def test_triangle_ccw():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(1.0, 1.0)

    t1 = Facet(v0, v1, v2)
    t2 = Facet(v0, v2, v1)
    assert t1.is_ccw() == True
    assert t2.is_ccw() == False


def test_infinite_triangle_incircle_condition():
    v0 = Vertex(math.inf, math.inf, infinite=True)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(0.0, 0.0)

    t1 = Facet(v0, v1, v2)

    vt1 = Vertex(0.5, 0.0)
    assert t1.is_incircumcircle(vt1)
