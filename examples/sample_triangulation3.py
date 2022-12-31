from random import sample
import random
import triangulation3 as tr3
import numpy as np

import examples.develop_helper as dh

seed = random.randint(0, 99999999)
print(f"random seed: {seed}")
np.random.seed(seed)

point_cloud = np.random.random((100, 3))
vertices = [tr3.Vertex(i[0], i[1], i[2]) for i in point_cloud]

trig = tr3.Triangulation3(vertices)
dh.plot_triangulation3(trig)
