import mesh2
import numpy as np
import random

import examples.develop_helper as dh

seed = random.randint(0, 99999999)
print(f"random seed: {seed}")
np.random.seed(seed)
# np.random.seed(39135982)

v1 = mesh2.Vertex(-0.3, 0.0)
v2 = mesh2.Vertex(1.0, 0.0)
v3 = mesh2.Vertex(1.0, 1.0)
v4 = mesh2.Vertex(-0.3, 1.0)
v5 = mesh2.Vertex(-0.3, 0.6)
v6 = mesh2.Vertex(0.0, 0.6)
v7 = mesh2.Vertex(0.0, 0.4)
v8 = mesh2.Vertex(-0.3, 0.4)

poly = mesh2.Polyloop([v1, v2, v3, v4, v5, v6, v7, v8])

point_cloud = np.random.random((5, 2))
vertices = [v1, v2, v3, v4, v5, v6, v7, v8]
vertices += [mesh2.Vertex(i[0], i[1]) for i in point_cloud]

trig = mesh2.Mesh(vertices, poly, 1.5)
dh.plot_mesh2(trig)
