import mesh3
import numpy as np
import random

import examples.develop_helper as dh

# import sys
# sys.setrecursionlimit(10000)
# import warnings
# warnings.resetwarnings()
# warnings.simplefilter('error')

seed = random.randint(0, 99999999)
print(f"random seed: {seed}")
# random.seed(seed)
random.seed(82040843)
# random.seed(90935726)

# Cube型のPLCサンプル

v0 = mesh3.Vertex3(0.0, 0.0, 0.0)
v1 = mesh3.Vertex3(0.6, 0.0, 0.0)
v2 = mesh3.Vertex3(0.6, 0.0, 0.45)
v3 = mesh3.Vertex3(0.0, 0.0, 0.45)

v4 = mesh3.Vertex3(0.0, 0.4, 0.0)
v5 = mesh3.Vertex3(0.6, 0.4, 0.0)
v6 = mesh3.Vertex3(0.6, 0.4, 0.45)
v7 = mesh3.Vertex3(0.0, 0.4, 0.45)

pl1 = mesh3.Polygon3([v0, v1, v2, v3])
pl2 = mesh3.Polygon3([v3, v0, v4, v7])
pl3 = mesh3.Polygon3([v4, v0, v1, v5])
pl4 = mesh3.Polygon3([v5, v1, v2, v6])
pl5 = mesh3.Polygon3([v6, v2, v3, v7])
pl6 = mesh3.Polygon3([v4, v5, v6, v7])

vertices = [v0, v1, v2, v3, v4, v5, v6, v7]
polygons = [pl1, pl2, pl3, pl4, pl5, pl6]
random.shuffle(vertices)

trig = mesh3.Mesh3(vertices, polygons)
dh.plot_triangulation3(trig)
