from random import sample

import triangulation3d as tr3
import numpy as np

import examples.develop_helper as dh

point_cloud = np.random.random((1000, 3))
vertices = [tr3.Vertex3(i[0], i[1], i[2]) for i in point_cloud]

trig = tr3.Triangulation3(vertices)
dh.plot_triangulation3(trig)
