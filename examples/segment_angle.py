import triangulation as tr

v1 = tr.Vertex(0, 0)
v2 = tr.Vertex(1, 1)
v3 = tr.Vertex(0, 1)

s1 = tr.Segment(v1, v2)
s2 = tr.Segment(v1, v3)

print("angle: ", s1.angle_pivot(s2, v1))