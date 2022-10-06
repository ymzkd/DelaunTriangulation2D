from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

v1 = tr.Vertex(-1.0, -1.0)
v2 = tr.Vertex(10.0, -1.2)

# x座標値14はエッジ分割時の鋭角対応で突破できる。
# x座標値16はコーナー三角形の対応が必要になりそう。
v3 = tr.Vertex(16.0, 1.0)
v4 = tr.Vertex(-1.0, 1.0)
poly = tr.Polyloop([v1, v2, v3, v4])

# メッシュ生成
mesh, tri, vert = tr.Triangulation.createMesh(poly, 1.45, 20)

# v = tr.Vertex(0.0, 0.0)
# mesh.add_vertex(v)

# dh.plot_triangles(mesh.triangles, segments=mesh.segments, circles=[si.diametric_ball() for si in mesh.segments])
dh.plot_mesh(mesh)
