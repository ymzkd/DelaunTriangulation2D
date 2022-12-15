import pytest

import math
from triangulation3 import Vertex, TetCell


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
