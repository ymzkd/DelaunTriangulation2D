import pytest

import triangulation3d as tr3

# dummy_v1 = tr3.Vertex3(0,0,1)
# dummy_v2 = tr3.Vertex3(0,0,0)
# dummy_v3 = tr3.Vertex3(1,0,0)
# dummy_v4 = tr3.Vertex3(0,1,0)
#
# dummy_mesh = tr3.Triangulation3([dummy_v1, dummy_v2, dummy_v3, dummy_v4])

def test_outersphere():
    v1 = tr3.Vertex3(0,0,1)
    v2 = tr3.Vertex3(0,0,0)
    v3 = tr3.Vertex3(1,0,0)
    v4 = tr3.Vertex3(0,1,0)

    tet = tr3.Tetrahedron(v1, v2, v3, v4, mesh=None)
    sphere = tet.outer_sphere()

    vt1 = tr3.Vertex3(5,5,5)
    vt2 = tr3.Vertex3(0.5,0.5,0.5)
    vt3 = tr3.Vertex3(-0.5,-0.5,-0.5)
    assert sphere.isinside(vt1) == False
    assert sphere.isinside(vt2) == True
    assert sphere.isinside(vt3) == False


def test_tet_orientation():
    v1 = tr3.Vertex3(0,0,1)
    v2 = tr3.Vertex3(0,0,0)
    v3 = tr3.Vertex3(1,0,0)
    v4 = tr3.Vertex3(0,1,0)
    tet = tr3.Tetrahedron(v1, v2, v3, v4, mesh=None)

    assert tet.orient() > 0.0

    v1 = tr3.Vertex3(0, 0, -1)
    v2 = tr3.Vertex3(0, 0, 0)
    v3 = tr3.Vertex3(1, 0, 0)
    v4 = tr3.Vertex3(0, 1, 0)
    tet = tr3.Tetrahedron(v1, v2, v3, v4, mesh=None)

    assert tet.orient() < 0.0


def test_tet_adjacent():
    v1 = tr3.Vertex3(0,0,1)
    v2 = tr3.Vertex3(0,0,0)
    v3 = tr3.Vertex3(1,0,0)
    v4 = tr3.Vertex3(0,1,0)
    v5 = tr3.Vertex3(0,0,-1)
    tet1 = tr3.Tetrahedron(v1, v2, v3, v4, mesh=None)
    tet2 = tr3.Tetrahedron(v3, v2, v1, v5, mesh=None)

    assert tet1.facets[3] == tet2.facets[3].opposite()
    assert tet1.facets[3].opposite() == tet2.facets[3]
