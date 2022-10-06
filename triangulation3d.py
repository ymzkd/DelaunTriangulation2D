from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

TOLERANCE = 1.0e-12

class Vertex3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class Tetrahedron:
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3
    v4: Vertex3

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3, v4: Vertex3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4


class Triangulation3:
    vertices: List[Vertex3]
    triangles: List[Triangle3]

    def __init__(self, vertices: List[Vertex3]):
        self.vertices = vertices

        # Create Bounding Box

    @staticmethod
    def superTetrafromVertices(vertices: List[Vertex3], size_fac=2.0) -> Tetrahedron:
        v0 = vertices[0]
        xmin = v0.x; xmax = v0.x
        ymin = v0.y; ymax = v0.y
        zmin = v0.z; zmax = v0.z
        for vi in vertices[1:]:
            xmin = min(xmin, vi.x)
            xmax = max(xmax, vi.x)
            ymin = min(ymin, vi.y)
            ymax = max(ymax, vi.y)
            zmin = min(zmin, vi.z)
            zmax = max(zmax, vi.z)

        center = Vertex3((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5)
        rad = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2) * size_fac * 0.5

        l = 4.0 * math.sqrt(2/3) * rad
        v1 = Vertex3(-l / 2 + center.x, -l / 2 / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z)
        v2 = Vertex3(l / 2 + center.x, -l / 2 / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z)
        v3 = Vertex3(center.x, l / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z)
        v4 = Vertex3(center.x, center.y, l * 3 / 4 * math.sqrt(2 / 3) + center.z)

        return Tetrahedron(v1, v2, v3, v4)


class Sphere:
    center: Vertex3
    radius: float

    def __init__(self, center: Vertex3, radius: float):
        self.center = center
        self.radius = radius
