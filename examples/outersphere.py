import triangulation3d as tr3

v1 = tr3.Vertex3(0, 0, 1)
v2 = tr3.Vertex3(0, 0, 0)
v3 = tr3.Vertex3(1, 0, 0)
v4 = tr3.Vertex3(0, 1, 0)

tet = tr3.Tetrahedron(v1, v2, v3, v4)
center = tet.outer_sphere_center()
sphere = tet.outer_sphere()

d1 = center.distance(v1)
d2 = center.distance(v2)
d3 = center.distance(v3)
d4 = center.distance(v4)
print(d1, sphere.radius)
print(d2, sphere.radius)
print(d3, sphere.radius)
print(d4, sphere.radius)