import pytest

from mesh2 import Facet, Vertex, Segment, Polyloop, Mesh


def test_segment_encroach():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(2.0, 0.0)

    s0 = Segment(v0, v2)
    s1 = Segment(v0, v1)
    s2 = Segment(v1, v2)
    s0.children = [s1, s2]

    tgt_v1 = Vertex(1.5, 0.01) # encroach s2
    tgt_v2 = Vertex(1.5, 10.01) # Outside
    tgt_v3 = Vertex(0.5, 0.01) # encroach s1

    assert s0.vertex_encroached(tgt_v1) == s2
    assert s0.vertex_encroached(tgt_v2) is None
    assert s0.vertex_encroached(tgt_v3) == s1


def test_polyloop_area():
    vertices = [
        Vertex(4.504194, -0.536649),
        Vertex(3.046386, -1.229997),
        Vertex(2.441928, 0.174477),
        Vertex(1.197458, 0.361147),
        Vertex(1.36635, 1.97007),
        Vertex(3.508618, 1.641174),
        Vertex(3.337926, 0.337933)]

    poly = Polyloop(vertices)

    assert pytest.approx(poly.area) == -5.05364092


def test_segment_contains():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(2.0, 0.0)
    v3 = Vertex(3.0, 0.0)

    s0 = Segment(v0, v3)
    s10 = Segment(v0, v2)
    s11 = Segment(v2, v3)
    s0.children = [s10, s11]
    s20 = Segment(v0, v1)
    s21 = Segment(v1, v2)
    s22 = Segment(v2, v3)
    s10.children = [s20, s21]
    s11.children = [s22]

    tgt_s1 = Segment(v0, v3)  # On root
    tgt_s2 = Segment(v0, v2)  # On mid
    tgt_s3 = Segment(v1, v2)  # On last
    tgt_s4 = Segment(v1, v3)  # expected fail

    assert tgt_s1 in s0
    assert tgt_s2 in s0
    assert tgt_s3 in s0
    assert tgt_s4 not in s0


def test_segment_flatten():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(2.0, 0.0)
    v3 = Vertex(3.0, 0.0)

    s0 = Segment(v0, v3)
    s10 = Segment(v0, v2)
    s11 = Segment(v2, v3)
    s0.children = [s10, s11]
    s20 = Segment(v0, v1)
    s21 = Segment(v1, v2)
    s22 = Segment(v2, v3)
    s10.children = [s20, s21]
    s11.children = [s22]

    expected_segments = [s20, s21, s22]

    assert all([ts == es for ts, es in zip(s0.flatten_child(), expected_segments)])


def test_segment_cluster():
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(4.0, 1.0)
    v2 = Vertex(4.0, 4.0)
    v3 = Vertex(2.0, 4.0)
    v4 = Vertex(-4.0, 1.0)
    v5 = Vertex(-5.0, -1.0)
    v6 = Vertex(-1.0, -3.0)
    v7 = Vertex(3.0, -1.0)
    seg1 = Segment(v0, v1)
    seg2 = Segment(v0, v2)
    seg3 = Segment(v0, v3)
    seg4 = Segment(v0, v4)
    seg5 = Segment(v0, v5)
    seg6 = Segment(v0, v6)
    seg7 = Segment(v0, v7)

    seg = seg2
    segments = [seg1, seg2, seg3, seg4, seg5, seg6, seg7]

    cluster = Mesh.cluster_from_segments(seg, v0, segments)
    assert len(cluster) == 4
