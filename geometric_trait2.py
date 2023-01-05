from __future__ import annotations
from typing import Generic, TypeVar

import math

import numpy as np

PointTrait = TypeVar("PointTrait")


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def toarray(self):
        return np.array([self.x, self.y])

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            return Vector(self.x * other, self.y * other)

    def __rmul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.x / other, self.y / other)
        else:
            raise ValueError(f'Only division by integer or float is allowed.({other})')

    @classmethod
    def zero(cls) -> Vector:
        return Vector(0.0, 0.0)

    def outer_product(self, other: Vector):
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def __str__(self):
        return f'x: {self.x}, y: {self.y}'


class Point(Vector):
    """メッシュ頂点クラス

    Attributes:
        x, y (float): 頂点座標
    """

    def __init__(self, x: float, y: float):
        super(Point, self).__init__(x, y)

    def distance(self, other: Point) -> float:
        """他の頂点までの距離

        Args:
            other (Point): 計算対象の頂点

        Returns:
            float: 距離
        """
        pt1 = self.toarray()
        pt2 = other.toarray()
        return math.sqrt(np.sum((pt2 - pt1) ** 2))

    def __hash__(self):
        return hash((self.x, self.y))


class Line(Generic[PointTrait]):
    """2つの節点を結ぶ線分
    Attributes:
        v1 (PointTrait): 節点1
        v2 (PointTrait): 節点2
    """
    def __init__(self, v1: PointTrait, v2: PointTrait):
        self.v1 = v1
        self.v2 = v2

    def ispt_rightside(self, pt) -> bool:
        """2つの節点を通る直線に対して入力点が右側にあるか判定

        Args:
            pt (ndarray[2]): 入力点座標を表す2次元のnumpy配列

        Returns:
            bool: 右側の場合True,左側の場合False
        """
        v10 = pt - self.v1.toarray()
        v12 = self.v2.toarray() - self.v1.toarray()

        return np.cross(v10, v12) > 0

    def is_cross(self, other: Line) -> bool:
        """2つの線分が交差しているかを判定

        Args:
            other (Segment): 比較対象線分

        Returns:
            bool: 交差していればTrue
        """
        # this -> other
        j1 = self.ispt_rightside(other.v1.toarray()) != self.ispt_rightside(other.v2.toarray())
        # other -> this
        j2 = other.ispt_rightside(self.v1.toarray()) != other.ispt_rightside(self.v2.toarray())
        return j1 and j2

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


class Circle:
    """中心と半径で定義された円

    Attributes:
        center (PointTrait): それぞれ中心のx座標とy座標
        rad (float): 円の半径
    """
    def __init__(self, center: Point, rad: float) -> None:
        self.center = center
        self.rad = rad

    def ispoint_inside(self, pt) -> bool:
        return (pt[0] - self.center.x)**2 + (pt[1] - self.center.y)**2 <= self.rad**2


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

    def area(self):
        v12 = np.array([self.v2.x - self.v1.x, self.v2.y - self.v1.y])
        v13 = np.array([self.v3.x - self.v1.x, self.v3.y - self.v1.y])
        return np.cross(v12, v13) * 0.5

    def outer_circle(self) -> Circle:
        """三角形の外接円を求める。

        Returns:
            Circle: 三角形の外接円
        """
        x1 = self.v1.x
        y1 = self.v1.y
        x2 = self.v2.x
        y2 = self.v2.y
        x3 = self.v3.x
        y3 = self.v3.y
        c = 2 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        cx = ((y3 - y1) * (x2**2 - x1**2 + y2**2 - y1**2) +
              (y1 - y2) * (x3**2 - x1**2 + y3**2 - y1**2)) / c
        cy = ((x1 - x3) * (x2**2 - x1**2 + y2**2 - y1**2) +
              (x2 - x1) * (x3**2 - x1**2 + y3**2 - y1**2)) / c
        r = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        return Circle(Point(cx, cy), r)

    def orientation(self) -> float:
        """
        節点順がCCWであれば正、CWであれば負
        """
        mat = np.array([[self.v1.x - self.v3.x, self.v1.y - self.v3.y],
                        [self.v2.x - self.v3.x, self.v2.y - self.v3.y]])
        return np.linalg.det(mat)

    def is_ccw(self) -> bool:
        """
        節点順がCCWであればTrue、CWであればFalse
        """
        return self.orientation() > 0.0

    def fix_orientation(self):
        """節点順がCCWでなかった場合に節点順をCCWに修正"""
        if not(self.is_ccw()):
            self.v2, self.v3 = self.v3, self.v2
