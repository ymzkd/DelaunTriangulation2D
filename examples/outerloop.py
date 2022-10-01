from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

vertices = []
for i in range(50):
    vertices.append(tr.Vertex(np.random.rand(), np.random.rand()))

testseg = tr.Segment(*sample(vertices, 2))
v1 = tr.Vertex(0.1, 0.1)
v2 = tr.Vertex(0.8, 0.1)
v3 = tr.Vertex(0.5, 0.4)
v4 = tr.Vertex(0.8, 0.8)
v5 = tr.Vertex(0.1, 0.8)
poly = tr.Polyloop([v1, v2, v3, v4, v5])
vertices.append(v1)
vertices.append(v2)
vertices.append(v3)
vertices.append(v4)
vertices.append(v5)
tess = tr.Triangulation(vertices, outerloops=[poly])

# dh.plot_triangulation(segments=poly.edges(), triangulate=tess)
dh.plot_triangles(tess.extract_loopinside(poly), segments=poly.edges())
