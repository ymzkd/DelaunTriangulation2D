import triangulation2 as tr
import numpy as np

import examples.develop_helper as dh

point_cloud = np.random.random((100, 2))
vertices = [tr.Vertex(i[0], i[1]) for i in point_cloud]

trig = tr.Triangulation(vertices)
dh.plot_triangulation(trig.finite_triangles())
