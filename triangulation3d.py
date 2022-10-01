from __future__ import annotations

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

class Triangle3:
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3


class Triangulation3:
    vertices: List[Vertex3]
    triangles: List[Triangle3]