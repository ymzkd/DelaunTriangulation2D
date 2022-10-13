from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

v1 = tr.Vertex(-1.5, -1.0)
v1a = tr.Vertex(-0.2, -1.0)
v1b = tr.Vertex(0.0, -0.9)
v1c = tr.Vertex(0.1, -1.0)
v2 = tr.Vertex(1.5, -1.2)
v3 = tr.Vertex(1.5, 1.0)
v4 = tr.Vertex(-1.5, 1.0)
poly = tr.Polyloop([v1, v1a, v1b, v1c, v2, v3, v4])

# メッシュ生成
mesh, tri, vert = tr.Triangulation.createMesh(poly, 1.45, 20)

# v = tr.Vertex(0.0, 0.0)
# mesh.add_vertex(v)

dh.plot_triangles(mesh.triangles, segments=mesh.segments, circles=[si.diametric_ball() for si in mesh.segments])
dh.plot_mesh(mesh)

for tri_i in mesh.triangles:
    if tri_i.is_seditious():
        print("find seditious triangle!!")