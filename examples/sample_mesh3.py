import mesh3
import numpy as np
import random

import examples.develop_helper as dh

# L型のPLCサンプル

v0 = mesh3.Vertex(0.0, 0.0, 0.0)
v1 = mesh3.Vertex(0.6, 0.0, 0.0)
v2 = mesh3.Vertex(0.6, 0.0, 0.45)
v3 = mesh3.Vertex(0.35, 0.0, 0.45)
v4 = mesh3.Vertex(0.35, 0.0, 0.8)
v5 = mesh3.Vertex(0.0, 0.0, 0.8)

v6 = mesh3.Vertex(0.0, 0.4, 0.0)
v7 = mesh3.Vertex(0.6, 0.4, 0.0)
v8 = mesh3.Vertex(0.6, 0.4, 0.45)
v9 = mesh3.Vertex(0.35, 0.4, 0.45)
v10 = mesh3.Vertex(0.35, 0.4, 0.8)
v11 = mesh3.Vertex(0.0, 0.4, 0.8)

pl1 = mesh3.Polygon([v0, v1, v2, v3, v4, v5])
pl2 = mesh3.Polygon([v5, v4, v10, v11])
pl3 = mesh3.Polygon([v4, v3, v9, v10])
pl4 = mesh3.Polygon([v3, v2, v8, v9])
pl5 = mesh3.Polygon([v2, v1, v7, v8])
pl6 = mesh3.Polygon([v1, v0, v6, v7])
pl7 = mesh3.Polygon([v0, v5, v11, v6])
pl8 = mesh3.Polygon([v7, v6, v11, v10, v9, v8])

vertices = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
polygons = [pl1, pl2, pl3, pl4, pl5, pl6, pl7, pl8]

trig = mesh3.Mesh(vertices, polygons)
dh.plot_mesh2(trig)
