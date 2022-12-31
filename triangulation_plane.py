from __future__ import annotations
from typing import List, Dict

import math
import numpy as np

from geometric_trait3 import Point, Plane, Vector, Line, Triangle, Sphere


class Vertex(Point):
    infinite: bool

    def __init__(self, x: float, y: float, z: float, infinite: bool = False):
        super().__init__(x, y, z)
        self.infinite = infinite

    def transform(self, transmat) -> Vertex:
        xyzw = np.ones([4])
        xyzw[0:3] = self.toarray()
        xyzw = transmat @ xyzw
        return Vertex(*xyzw[:3], infinite=self.infinite)

    @staticmethod
    def ghost_vertex() -> Vertex:
        return Vertex(math.inf, math.inf, math.inf, infinite=True)


class Edge(Line):
    def __init__(self, v1: Vertex, v2: Vertex, mesh: TriangulationPlane = None):
        super(Edge, self).__init__(v1, v2)
        if mesh:
            self.plane = mesh.plane
        self.mesh = mesh

    def opposite(self) -> Edge:
        return Edge(self.v2, self.v1, self.mesh)

    def ispt_rightside(self, pt) -> bool:
        """

        Args:
            pt: 節点座標を表す三次元のベクトル

        Returns:
            TriangulationPlaneの平面上で入力点がこのエッジの右側にあるかどうか？
        """
        vav = pt - self.v1.toarray()
        vab = self.v2.toarray() - self.v1.toarray()
        return (self.plane.ez.toarray() @ np.cross(vav, vab)) >= 0.0

    def is_infinite(self) -> bool:
        return any((self.v1.infinite, self.v2.infinite))


class Facet(Triangle):

    def __init__(self, v1, v2, v3, mesh: TriangulationPlane = None):
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
        """節点vがこの三角形の外接円の内部に位置するか
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
            sph = self.diametric_ball()
            return sph.isinside(v)

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])


class TriangulationPlane:
    triangles: List[Facet]
    edge_triangle_table: Dict[Edge, Facet]

    def __init__(self, vertices: List[Vertex]):
        self.vertices = vertices
        self.edge_triangle_table = {}
        self.triangles = []

        gv = Vertex.ghost_vertex()
        v1 = vertices.pop(0)
        v2 = vertices.pop(0)
        v3 = vertices.pop(0)

        self.plane = Plane(v1, v2, v3)
        t1 = Facet(v1, v2, v3, mesh=self)
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
            raise Exception(f"No triangles include point coordinate: [{v.x},{v.y},{v.z}]")

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
            for seg in edges:
                if seg == base_edge:
                    continue
                self.dig_cavity(u, seg)
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
