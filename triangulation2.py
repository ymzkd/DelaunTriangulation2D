from __future__ import annotations

import math
# from enum import Enum
from math import sqrt
from typing import List, Tuple, Union

import numpy as np

from geometric_trait2 import Point, Circle, Triangle, Line

TOLERANCE = 1.0e-12


class Vertex(Point):
    """メッシュ頂点クラス

    Attributes:
        x, y (float): 頂点座標値、infinite節点の場合はそれぞれ、infと設定される。
        infinite (bool): infinite節点であるときにTrue、それ以外の場合にはFalse

    TODO:
        meshというAttributeが必要かどうか検討
    """
    def __init__(self, x: float, y: float, mesh: Triangulation = None, infinite: bool = False):
        super(Vertex, self).__init__(x, y)
        self.mesh = mesh
        self.infinite = infinite


class Edge(Line[Vertex]):
    """メッシュFacetのエッジクラス

    Edgeは頂点v1とv2から成る、メッシュのFacetの各辺に対応する。

    Attributes:
        v1, v2 (Vertex): それぞれエッジの始点と終点に対応する節点

    TODO:
        meshというAttributeが必要かどうか検討
    """
    def __init__(self, v1: Vertex, v2: Vertex, mesh: Triangulation = None):
        super(Edge, self).__init__(v1, v2)
        self.mesh = mesh

    def opposite(self) -> Edge:
        """このEdgeの逆向きのEdgeを取得する.

        Returns:
            Edge: このEdgeの逆向きのEdge
        """
        return Edge(self.v2, self.v1, self.mesh)


class CircumCircle(Circle):
    """中心と半径で定義された円

    Attributes:
        cx, cy (float): それぞれ中心のx座標とy座標
        r (float): 円の半径
    """
    def __init__(self, center: Point, rad: float) -> None:
        super(CircumCircle, self).__init__(center, rad)

    @staticmethod
    def create_from_edge(seg: Edge) -> CircumCircle:
        pt1 = seg.v1
        pt2 = seg.v2
        r = pt1.distance(pt2) / 2
        return CircumCircle(Point((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2), r)


class Facet(Triangle[Vertex]):
    """三角形メッシュFacetクラス

    Attributes:
        vertices (List[Vertex]): メッシュ頂点の配列[v1, v2, v3]
        v1, v2, v3 (Vertex) : メッシュ頂点
    """
    vertices: List[Vertex]
    mesh: Triangulation

    def __init__(self, v1, v2, v3, mesh: Triangulation = None):
        super(Facet, self).__init__(v1, v2, v3)
        self.mesh = mesh

    def get_edge(self, idx: int) -> Edge:
        if idx == 0:
            return Edge(self.v1, self.v2, self.mesh)
        elif idx == 1:
            return Edge(self.v2, self.v3, self.mesh)
        else:
            return Edge(self.v3, self.v1, self.mesh)

    def is_incircumcircle(self, v: Vertex) -> bool:
        """
        節点vがこの三角形の外接円の内部に位置するかを判定

        節点vがこの三角形の外接円の内部に位置するかを判定。三角形がGhostTriangleの場合は
        finite edgeの半平面(左側・CCW)に節点が位置すれば外接円内部と判定する。

        Args:
            v: テスト対象の節点

        Returns:
            節点vがこの三角形の内部に位置する場合はTrue, その他の場合はFalse
        """
        if self.is_infinite():
            # 節点が半平面に位置するか？
            vid_inf = [v.infinite for v in self.vertices].index(True)
            seg = self.get_edge((vid_inf + 1) % 3)
            return not(seg.ispt_rightside(v.toarray()))
        else:
            # 三角形外接円に節点が含まれるか？
            cir = self.outer_circle()
            return cir.ispoint_inside(v.toarray())

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    def edge_radius_ratio(self) -> float:
        rad = self.outer_circle().rad
        edge_length = [self.get_edge(i).length() for i in range(3)]
        return rad / np.min(edge_length)


class Triangulation:
    """三角形メッシュ生成・データの格納

    Attributes:
        vertices (List[Vertex]): 頂点のリスト
        triangles (List[Triangles]): 生成された三角形のリスト
        edge_triangle_table (Dict[Edge,Triangle]): エッジをキーとしてそれに隣接する三角形をデータとした関係テーブル
    """
    def __init__(self, vertices: List[Vertex]):
        self.edge_triangle_table = {}
        self.triangles = []
        self.vertices = []

        if len(vertices) < 3:
            print("3つ以上の節点入力が必要")

        gv = Vertex(math.inf, math.inf, infinite=True)
        v1 = vertices.pop()
        v2 = vertices.pop()
        v3 = vertices.pop()

        t1 = Facet(v1, v2, v3, mesh=self)
        t1.fix_orientation()
        gt1 = Facet(gv, t1.v2, t1.v1, mesh=self)
        gt2 = Facet(gv, t1.v3, t1.v2, mesh=self)
        gt3 = Facet(gv, t1.v1, t1.v3, mesh=self)

        self.vertices = [gv, v1, v2, v3]
        self.add_triangle(t1)
        self.add_triangle(gt1)
        self.add_triangle(gt2)
        self.add_triangle(gt3)

        for vi in vertices:
            self.add_vertex(vi)

    def locate(self, v: Vertex) -> Facet:
        """任意の点を外接円内部に含む三角形を探索する。
        もし、既存のTriangulationの外側に点が挿入された場合は、
        半平面に挿入節点を含むGhost Triangleを返す。

        Args:
            v (Vertex): 頂点

        Returns:
            Facet: 入力節点を含む三角形
        """
        for tri in self.triangles:
            if tri.is_incircumcircle(v):
                return tri
        else:
            raise Exception(f"No triangles include point coordinate: [{v.x},{v.y}]")

    def dig_cavity(self, u: Vertex, edge: Edge):
        """
        Triangulation3内で追加節点uを四面体の外接球に含む四面体群を探索する。
        Args:
            u(Vertex): 追加節点
            edge(Segment): 探索起点となる四面体面のFacetDescriptor

        Notes:
            'Delaunay Mesh Generation' Chapter3.4
        """
        base_edge = edge.opposite()
        try:
            tri = self.edge_triangle_table[base_edge]
        except KeyError:
            # step2
            # if tri.is_infinite():
                # tet_new = Tetrahedron(u, tri.v1, tri.v2, tri.v3, self)
                # self.add_triangle()
            return

        if tri.is_incircumcircle(u):
            edges = [tri.get_edge(i) for i in range(3)]
            self.remove_triangle(tri)
            for edge in edges:
                # 探索始点エッジはスキップ
                if edge == base_edge:
                    continue
                self.dig_cavity(u, edge)
        else:
            self.add_triangle(Facet(u, edge.v1, edge.v2, mesh=self))

    def add_vertex(self, v: Vertex) -> None:
        """頂点vを挿入

        Args:
            v (Vertex): 挿入節点
        """
        tri = self.locate(v)
        self.vertices.append(v)

        edges = [tri.get_edge(i) for i in range(3)]
        self.remove_triangle(tri)
        for edge in edges:
            self.dig_cavity(v, edge)

    def remove_triangle(self, tri: Facet):
        _ = self.edge_triangle_table.pop(tri.get_edge(0))
        _ = self.edge_triangle_table.pop(tri.get_edge(1))
        _ = self.edge_triangle_table.pop(tri.get_edge(2))
        self.triangles.remove(tri)

    def add_triangle(self, tri: Facet):
        e1 = tri.get_edge(0)
        e2 = tri.get_edge(1)
        e3 = tri.get_edge(2)
        self.edge_triangle_table[e1] = tri
        self.edge_triangle_table[e2] = tri
        self.edge_triangle_table[e3] = tri
        self.triangles.append(tri)

    def finite_triangles(self) -> List[Facet]:
        """包絡三角形を除いた三角形を得る。

        Returns:
            List[Facet]: 包絡三角形を除いた三角形のリスト
        """
        triangles = []
        for tri in self.triangles:
            if tri.is_infinite():
                continue
            triangles.append(tri)

        return triangles
