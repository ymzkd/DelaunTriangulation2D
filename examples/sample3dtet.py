import numpy as np
import triangulation3d as tr3

vertices = []
for i in range(10):
    vertices.append(tr3.Vertex3(np.random.rand(), np.random.rand(), np.random.rand()))

tet = tr3.Triangulation3.superTetrafromVertices(vertices)