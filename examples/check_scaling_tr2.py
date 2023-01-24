import triangulation2 as tr
import numpy as np
import time

import examples.develop_helper as dh

start_time = time.time()
point_cloud = np.random.random((10, 2))
vertices = [tr.Vertex(i[0], i[1]) for i in point_cloud]
trig = tr.Triangulation(vertices)
# dh.plot_triangulation2d(trig)
delta1 = start_time - time.time()
print(f"time: 10 sampling, {delta1} sec")

start_time = time.time()
point_cloud = np.random.random((100, 2))
vertices = [tr.Vertex(i[0], i[1]) for i in point_cloud]
trig = tr.Triangulation(vertices)
# dh.plot_triangulation2d(trig)
delta1 = start_time - time.time()
print(f"time: 100 sampling, {delta1} sec")

start_time = time.time()
point_cloud = np.random.random((1000, 2))
vertices = [tr.Vertex(i[0], i[1]) for i in point_cloud]
trig = tr.Triangulation(vertices)
# dh.plot_triangulation2d(trig)
delta1 = start_time - time.time()
print(f"time: 1000 sampling, {delta1} sec")

start_time = time.time()
point_cloud = np.random.random((10000, 2))
vertices = [tr.Vertex(i[0], i[1]) for i in point_cloud]
trig = tr.Triangulation(vertices)
# dh.plot_triangulation2d(trig)
delta1 = start_time - time.time()
print(f"time: 10000 sampling, {delta1} sec")

