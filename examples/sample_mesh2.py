import mesh2
import numpy as np
import random

import examples.develop_helper as dh

seed = random.randint(0, 99999999)
print(f"random seed: {seed}")
np.random.seed(seed)
# np.random.seed(5905689)

v1 = mesh2.Vertex(-0.3, 0.0)
v2 = mesh2.Vertex(1.0, 0.0)
v3 = mesh2.Vertex(1.0, 1.0)
v4 = mesh2.Vertex(-0.3, 1.0)
v5 = mesh2.Vertex(-0.3, 0.6)
v6 = mesh2.Vertex(0.0, 0.6)
v7 = mesh2.Vertex(0.0, 0.4)
v8 = mesh2.Vertex(-0.3, 0.4)

v9 = mesh2.Vertex(0.2, 0.2)
v10 = mesh2.Vertex(0.2, 0.8)
v11 = mesh2.Vertex(0.6, 0.8)
v12 = mesh2.Vertex(0.6, 0.2)

# v13 = mesh2.Vertex(0.8, 0.7)  # y=0.7ならOK,それ以上の鋭角がダメそう
v13 = mesh2.Vertex(0.8, 0.8)  # このケースNG
v14 = mesh2.Vertex(0.7, 0.2)
v15 = mesh2.Vertex(0.9, 0.2)

poly = mesh2.Polyloop([v1, v2, v3, v4, v5, v6, v7, v8])
poly_in = mesh2.Polyloop([v9, v10, v11, v12])

# point_cloud = np.random.random((5, 2))
vertices = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]

seg1 = mesh2.Segment(v13, v14)
seg2 = mesh2.Segment(v13, v15)

# vertices += [mesh2.Vertex(i[0], i[1]) for i in point_cloud]

trig = mesh2.Mesh(vertices, poly, 1.5, innerloops=[poly_in], segments=[seg1, seg2])
dh.plot_mesh2(trig)
