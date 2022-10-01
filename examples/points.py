from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

vertices = []
for i in range(10):
    vertices.append(tr.Vertex(np.random.rand(), np.random.rand()))

tess = tr.Triangulation(vertices)

# dh.plot_triangles(tess.finite_triangles())
print("before refinement")
print(dh.triangulation_statics(tess))
dh.plot_mesh(tess)

print("after refinement")
tess.refinement()
print(dh.triangulation_statics(tess))
tess.refinement()
print(dh.triangulation_statics(tess))
tess.refinement()
print(dh.triangulation_statics(tess))
tess.refinement()
print(dh.triangulation_statics(tess))

# dh.plot_triangles(tess.finite_triangles())
dh.plot_mesh(tess)

