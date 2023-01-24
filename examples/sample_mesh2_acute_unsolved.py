import mesh2
import numpy as np
import random

import examples.develop_helper as dh

seed = random.randint(0, 99999999)
print(f"random seed: {seed}")
np.random.seed(seed)
# np.random.seed(5905689)

v1 = mesh2.Vertex(1.0, 1.0)
v2 = mesh2.Vertex(-1.0, 1.0)
v3 = mesh2.Vertex(-1.0, -1.0)
v4 = mesh2.Vertex(1.0, -1.0)

poly = mesh2.Polyloop([v1, v2, v3, v4])

vs0 = mesh2.Vertex(0.0, 0.0)

# Case2
vs1 = mesh2.Vertex(0.5, 0.0)
vs2 = mesh2.Vertex(0.51, 0.03)
segments = [
    mesh2.Segment(vs0, vs1),
    mesh2.Segment(vs0, vs2),
]
vertices = [v1, v2, v3, v4, vs0, vs1, vs2]

trig = mesh2.Mesh(vertices, poly, 1.5, segments=segments)
dh.plot_mesh2(trig)
