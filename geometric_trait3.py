from __future__ import annotations
from typing import Generic, TypeVar, List

import math

import numpy as np

PointTrait = TypeVar("PointTrait")
FacetTrait = TypeVar("FacetTrait")


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def toarray(self):
        return np.array([self.x, self.y, self.z])

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __str__(self):
        return f'x: {self.x}, y: {self.y}, z: {self.z}'

    def outer_product(self, other: Vector):
        """
        外積
        """
        new_vec = Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        return new_vec


class Point(Vector):
    """メッシュ頂点クラス

    Attributes:
        x, y, z (float): 頂点座標
    """

    def __init__(self, x: float, y: float, z: float):
        super(Point, self).__init__(x, y, z)

    def distance(self, other: Point) -> float:
        """他の頂点までの距離

        Args:
            other (Point): 計算対象の頂点

        Returns:
            float: 距離
        """
        return (other - self).length()

    def __hash__(self):
        return hash((self.x, self.y, self.z))


class Line(Generic[PointTrait]):
    """2つの節点を結ぶ線分
    Attributes:
        v1 (PointTrait): 節点1
        v2 (PointTrait): 節点2
    """
    def __init__(self, v1: PointTrait, v2: PointTrait):
        self.v1 = v1
        self.v2 = v2

    def length(self) -> float:
        return self.v1.distance(self.v2)

    def direction(self) -> Vector:
        return self.v2 - self.v1

    def __hash__(self):
        return hash((self.v1, self.v2))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return '\n  '.join(['Line: '] + [f'v1: {self.v1}', f'v2: {self.v2}'])


class Sphere:
    center: Point
    radius: float

    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    def isinside(self, v: Point) -> bool:
        d1 = self.center.distance(v)
        return d1 <= self.radius


class Plane:
    """
    平面。というか局所座標系になりつつある。
    """

    def __init__(self, origin: Point, pt_x: Point, pt_y: Point):
        ex = pt_x - origin
        ex *= 1.0 / ex.length()
        ey = pt_y - origin
        ey = ey - (ey * ex) * ey
        ey *= 1.0 / ey.length()

        self.ex = ex
        self.ey = ey
        self.ez = ex.outer_product(ey)
        self.origin = origin

    def transmat(self):
        """
        指定した原点と平面上をx-y平面とした座標系による同次座標系変換マトリックス 4x4
        Returns:
            4x4の同次座標系変換マトリクス
        """
        mat = np.zeros([4, 4])
        mat[0, 0:3] = self.ex.toarray()
        mat[1, 0:3] = self.ey.toarray()
        mat[2, 0:3] = self.ez.toarray()
        mat[3, 3] = 1.0
        mat[0:3, 3] = self.origin.toarray()
        return mat

    def signed_distance(self, pt: Point) -> float:
        """平面から入力点ptまでの符号付き距離

        Args:
            pt (Point): 距離を計測する基準点

        Returns:
            float: 平面から点までの符号付き距離、法線方向が正で逆方向が負
        """
        v = pt - self.origin
        return self.ez * v


class Triangle(Generic[PointTrait]):
    def __init__(self, v1: PointTrait, v2: PointTrait, v3: PointTrait):
        self.vertices = [v1, v2, v3]

    def __str__(self):
        return '\n  '.join(['Triangle: '] + [f'v{i}: ({vi})' for i, vi in enumerate(self.vertices)])

    @property
    def v1(self):
        return self.vertices[0]

    @v1.setter
    def v1(self, value: PointTrait):
        self.vertices[0] = value

    @property
    def v2(self):
        return self.vertices[1]

    @v2.setter
    def v2(self, value: PointTrait):
        self.vertices[1] = value

    @property
    def v3(self):
        return self.vertices[2]

    @v3.setter
    def v3(self, value: PointTrait):
        self.vertices[2] = value

    def __hash__(self):
        """
        節点の並び順には依存するが、どこが先頭であるかには依存しないハッシュ値を返す。
        e.g. v1->v2->v3のハッシュ値とv2->v3->v1のハッシュ値は同じ

        Notes:
            上記説明は内部的には常にハッシュ値が最小の頂点を先頭として並び変えてハッシュ計算
            を行うことによって実装している。
        """
        hashes = [hash(self.v1), hash(self.v2), hash(self.v3)]
        offset = hashes.index(min(hashes))
        hashset = tuple(hashes[(i + offset) % 3] for i in range(3))
        return hash(hashset)

    def __eq__(self, other):
        return hash(self) == hash(other)


class Tetrahedron(Generic[PointTrait, FacetTrait]):
    v1: PointTrait
    v2: PointTrait
    v3: PointTrait
    v4: PointTrait
    vertices: List[PointTrait]
    facets: List[FacetTrait]

    def __init__(self, v1: PointTrait, v2: PointTrait, v3: PointTrait, v4: PointTrait):
        self.vertices = [v1, v2, v3, v4]

    @property
    def v1(self) -> PointTrait:
        return self.vertices[0]

    @v1.setter
    def v1(self, value: PointTrait):
        self.vertices[0] = value

    @property
    def v2(self) -> PointTrait:
        return self.vertices[1]

    @v2.setter
    def v2(self, value: PointTrait):
        self.vertices[1] = value

    @property
    def v3(self) -> PointTrait:
        return self.vertices[2]

    @v3.setter
    def v3(self, value: PointTrait):
        self.vertices[2] = value

    @property
    def v4(self) -> PointTrait:
        return self.vertices[3]

    @v4.setter
    def v4(self, value: PointTrait):
        self.vertices[3] = value

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    def outer_sphere(self) -> Sphere:
        v12 = self.v1 - self.v2
        v13 = self.v1 - self.v3
        v34 = self.v3 - self.v4
        mA = np.vstack([v12.toarray(), v13.toarray(), v34.toarray()])

        vb = 0.5 * np.array(
            [self.v1.length() ** 2 - self.v2.length() ** 2,
             self.v1.length() ** 2 - self.v3.length() ** 2,
             self.v3.length() ** 2 - self.v4.length() ** 2]
        )
        center = np.linalg.solve(mA, vb)
        center = Point(center[0], center[1], center[2])
        rad = np.linalg.norm(self.v1.toarray() - center.toarray())
        return Sphere(center, rad)

    def orient(self) -> float:
        va = self.v1 - self.v2
        vb = self.v1 - self.v3
        vc = self.v1 - self.v4
        return det3x3(va, vb, vc)


def det3x3(v1: Vector, v2: Vector, v3: Vector):
    """
    行列式の計算
    | v1[0] v2[0] v3[0] |
    | v1[1] v2[1] v3[1] |
    | v1[2] v2[2] v3[2] |

    Parameters
    ----------
    v1, v2, v3: Vector3d

    """
    return v1.x * v2.y * v3.z + v1.y * v2.z * v3.x + v1.z * v2.x * v3.y \
           - v1.x * v2.z * v3.y - v1.y * v2.x * v3.z - v1.z * v2.y * v3.x