import random
from random import sample

import triangulation as tr
import numpy as np

import examples.develop_helper as dh

vertices = []
for i in range(50):
    vertices.append(tr.Vertex(np.random.rand(), np.random.rand()))

testseg = tr.Segment(*sample(vertices, 2))
tess = tr.Triangulation(vertices)

subsegments =  tess.insert_segment(testseg)

dh.plot_triangles(tess.finite_triangles(),
                  segments=subsegments,
                  circles=[si.diametric_ball() for si in subsegments])

dh.plot_mesh(tess)