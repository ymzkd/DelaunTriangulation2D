from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

vertices = []
for i in range(50):
    vertices.append(tr.Vertex(np.random.rand(), np.random.rand()))

tess = tr.Triangulation(vertices)

dh.plot_triangles(tess.finite_triangles())
