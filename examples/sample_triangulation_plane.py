"""
Triangulation Planeクラスで、`乱数に基づくランダム入力によるサンプル
"""

import mesh3 as trp
import numpy as np

import examples.develop_helper as dh

generate_seed = np.random.randint(0, 100000)
print("seed: ", generate_seed)
np.random.seed(generate_seed)
# np.random.seed(41069)

# Create vertices
point_cloud = np.random.random((10, 3))
vertices = [trp.Vertex(i[0], i[1], 0.0) for i in point_cloud]

# Random base plane
random_plane_points = np.random.random((3, 3))
base_points = [trp.Vertex(*p) for p in random_plane_points]
pln = trp.Plane(*base_points)

# Transform vertices
vertices = [vi.transform(pln.transmat()) for vi in vertices]

# Execute & Visualize
trig = trp.TriangulationPlane(vertices)
dh.plot_triangulation_plane(trig)
