from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

v1 = tr.Vertex(0.1, 0.1)
v2 = tr.Vertex(0.9, 0.1)
v3 = tr.Vertex(0.5, 0.4)
v4 = tr.Vertex(0.8, 0.8)
v5 = tr.Vertex(0.1, 0.8)
poly = tr.Polyloop([v1, v2, v3, v4, v5])

# メッシュ生成
mesh, tri, vert = tr.Triangulation.createMesh(poly, 1.5)

dh.plot_triangles(mesh.triangles, segments=mesh.segments, circles=[si.diametric_ball() for si in mesh.segments])
dh.plot_mesh(mesh)
