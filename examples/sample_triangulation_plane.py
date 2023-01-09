"""TriangulationPLCクラスのサンプル

TriangulationPLCはPLC一面の三角形分割を行うクラスだが、Mesh3と組み合わせて使うことが意図されており、
TriangulationPLC単体では入力PLCの線要素のencroachを解消する能力はないので注意.
"""
import numpy as np

import mesh3 as trp
import examples.develop_helper as dh

generate_seed = np.random.randint(0, 100000)
print("seed: ", generate_seed)
np.random.seed(generate_seed)
# np.random.seed(41069)

# Random plane
random_plane_points = np.random.random((3, 3))
base_points = [trp.Vertex3(*p) for p in random_plane_points]
pln = trp.Plane(*base_points)

# Create vertices
points = [
    [-0.756842, 0.937157, 0],
    [-1.735611, -1.226436, 0],
    [0.977467, -2.582975, 0],
    [3.553174, -1.964806, 0],
    [3.123889, 1.229071, 0]]

vertices = [trp.Vertex3(*pi) for pi in points]

# Transform vertices
vertices = [vi.transform(pln.transmat()) for vi in vertices]
poly = trp.Polygon3(vertices)

# Execute & Visualize
trig = trp.TriangulationPLC(poly)
trig.mark_inout()
dh.plot_triangulationPLC(trig)
