from __future__ import annotations

import math
from typing import List, Union, Dict

import numpy as np

from geometric_trait3 import Point3, Triangle, Plane, Tetrahedron, Sphere, Line

TOLERANCE = 1.0e-12


class Vertex(Point3):
    x: float
    y: float
    z: float
    infinite: bool

    def __init__(self, x: float, y: float, z: float, infinite: bool = False):
        super(Vertex, self).__init__(x, y, z)
        self.infinite = infinite


class Facet(Triangle):
    """
    Triangulation3dにおいて四面体の各面をCCWに格納する。
    テーブルのキーとして四面体の面と四面体の隣接関係を関連づけるために用いる。
    """
    v1: Vertex
    v2: Vertex
    v3: Vertex

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex):
        super(Facet, self).__init__(v1, v2, v3)

    def opposite(self) -> Facet:
        return Facet(self.v3, self.v2, self.v1)

    def is_infinite(self) -> bool:
        return any([self.v1.infinite, self.v2.infinite, self.v3.infinite])

    # def plane(self) -> Plane:
    #     return Plane(self.v1, self.v2, self.v3)

    def point_at(self, v0: Vertex) -> List[float]:
        vec02 = self.v2 - v0
        vec03 = self.v3 - v0
        t1 = np.cross(vec02.toarray(), vec03.toarray())

        vec03 = self.v3 - v0
        vec01 = self.v1 - v0
        t2 = np.cross(vec03.toarray(), vec01.toarray())

        vec01 = self.v1 - v0
        vec02 = self.v2 - v0
        t3 = np.cross(vec01.toarray(), vec02.toarray())

        vec12 = self.v2 - self.v1
        vec13 = self.v3 - self.v1
        t0 = np.cross(vec12.toarray(),vec13.toarray())

        # inv_t0 = 1.0 / t0
        t0_norm = np.linalg.norm(t0)
        l1 = np.dot(t1, t0) / t0_norm
        l2 = np.dot(t2, t0) / t0_norm
        l3 = np.dot(t3, t0) / t0_norm

        return [l1, l2, l3]

    def diametric_ball(self) -> Sphere:
        pln = self.plane()
        v = np.array(
            ((self.v1 * self.v1 - self.v2 * self.v2) * 0.5, (self.v1 * self.v1 - self.v3 * self.v3) * 0.5, pln.origin * pln.ez))
        mat = np.vstack(((self.v1 - self.v2).toarray(), (self.v1 - self.v3).toarray(), pln.ez.toarray()))
        cent = Point3(*np.linalg.solve(mat, v))
        rad = (self.v1 - cent).length()
        return Sphere(cent, rad)


class TetCell(Tetrahedron[Vertex, Facet]):
    v1: Vertex
    v2: Vertex
    v3: Vertex
    v4: Vertex
    vertices: List[Vertex]
    facets: List[Facet]

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex, v4: Vertex,
                 mesh: Triangulation3 = None):
        super(TetCell, self).__init__(v1, v2, v3, v4)
        # self.vertices = [v1, v2, v3, v4]
        self.mesh = mesh
        f1 = Facet(v2, v4, v3)
        f2 = Facet(v1, v3, v4)
        f3 = Facet(v1, v4, v2)
        f4 = Facet(v1, v2, v3)
        self.facets = [f1, f2, f3, f4]

    # def validate(self):
    #     if self.is_infinite():
    #         return
    #     for i, fi in enumerate(self.facets):
    #         if fi.plane().signed_distance(self.vertices[i]) > 0.0:
    #             raise ValueError("facet reverse")
    #
    #     if self.orient() < 0.0:
    #         raise ValueError("tetrahedron reverse")

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    def is_incircumsphere(self, v: Vertex) -> bool:
        if self.is_infinite():
            return self.infinite_is_incircumsphere(v)
        else:
            sph = self.outer_sphere()
            return sph.isinside(v)

    def infinite_is_incircumsphere(self, v: Vertex, delta=1.0e-06) -> bool:
        """Ghost TetrahedronのOpenCircum ball内外判定

        Args:
            v (Vertex): テスト対象節点
            delta:

        Returns:
            bool: テスト対象節点がOpenCircumBall内部であればTrue,その他の場合はFalse
        Notes:
            'Delaunay Mesh Generation' Chapter 5.2 (P107)
        """
        for fi in self.facets:
            if fi.is_infinite():
                continue
            # facetは外側向き
            dist = fi.plane().signed_distance(v)
            if dist < (0.0 - delta):  # TODO: ここも開集合とするために微小数値を引く必要ありそう
                return True
            else:
                ball = fi.diametric_ball()
                return abs(dist) < delta and ball.isinside(v)

    def is_inside(self, v: Vertex) -> bool:
        if self.is_infinite():
            for fi in self.facets:
                if fi.is_infinite():
                    continue
                return fi.plane().signed_distance(v) <= 0.0
        else:
            judges = [fi.plane().signed_distance(v) <= 0.0 for fi in self.facets]
            return all(judges)

    def fix_orientation(self):
        if self.orient() < 0.0:
            self.vertices[:3], self.vertices[3] = self.vertices[1:], self.vertices[0]
            f1 = Facet(self.v2, self.v4, self.v3)
            f2 = Facet(self.v1, self.v3, self.v4)
            f3 = Facet(self.v1, self.v4, self.v2)
            f4 = Facet(self.v1, self.v2, self.v3)
            self.facets = [f1, f2, f3, f4]


class Triangulation3:
    vertices: List[Vertex]
    tetrahedrons: List[TetCell]
    face_adjacent_table: Dict[Facet, TetCell]

    def __init__(self, vertices: List[Vertex]):
        self.face_adjacent_table = {}
        self.tetrahedrons = []

        if len(vertices) < 4:
            print("4つ以上の節点入力が必要")

        # Create initial tetrahedron
        gv = Vertex(math.inf, math.inf, math.inf, infinite=True)
        v1 = vertices.pop(0)
        v2 = vertices.pop(0)
        v3 = vertices.pop(0)
        v4 = vertices.pop(0)
        t1 = TetCell(v1, v2, v3, v4, mesh=self)
        t1.fix_orientation()

        gt1 = TetCell(gv, t1.v1, t1.v2, t1.v3, mesh=self)
        gt2 = TetCell(gv, t1.v1, t1.v3, t1.v4, mesh=self)
        gt3 = TetCell(gv, t1.v2, t1.v1, t1.v4, mesh=self)
        gt4 = TetCell(gv, t1.v2, t1.v4, t1.v3, mesh=self)

        self.vertices = [gv, v1, v2, v3, v4]
        self.add_tetrahedron(t1)
        self.add_tetrahedron(gt1)
        self.add_tetrahedron(gt2)
        self.add_tetrahedron(gt3)
        self.add_tetrahedron(gt4)

        for i, vi in enumerate(vertices):
            # self.check_triangles()
            self.add_vertex(vi)

    def check_triangles(self):
        for ti in self.tetrahedrons:
            for fi in ti.facets:
                if fi not in self.face_adjacent_table:
                    print("Facet not existing")
                if fi.opposite() not in self.face_adjacent_table:
                    print("Facet opposite not existing")
            if not ti.is_infinite():
                if ti.orient() < 0.0:
                    print("Orientation is reverse")

    def finite_tetrahedrons(self) -> List[TetCell]:
        return [ti for ti in self.tetrahedrons if not ti.is_infinite()]

    def finite_vertices(self) -> List[Vertex]:
        return [vi for vi in self.vertices if not vi.infinite]

    def remove_tetrahedron(self, tet: TetCell):
        _ = self.face_adjacent_table.pop(tet.facets[0])
        _ = self.face_adjacent_table.pop(tet.facets[1])
        _ = self.face_adjacent_table.pop(tet.facets[2])
        _ = self.face_adjacent_table.pop(tet.facets[3])
        self.tetrahedrons.remove(tet)

    def add_tetrahedron(self, tet: TetCell):
        count_num = len(self.face_adjacent_table)
        self.face_adjacent_table[tet.facets[0]] = tet
        self.face_adjacent_table[tet.facets[1]] = tet
        self.face_adjacent_table[tet.facets[2]] = tet
        self.face_adjacent_table[tet.facets[3]] = tet
        if (len(self.face_adjacent_table) - count_num) != 4:
            raise ValueError("Table add failed.")
        self.tetrahedrons.append(tet)

    def getIncludeTet(self, pt: Vertex) -> Union[TetCell, None]:
        for tet_i in self.tetrahedrons:
            # if tet_i.is_incircumsphere(pt):
            if tet_i.is_inside(pt):
                return tet_i
        else:
            return None

    def add_vertex(self, v: Vertex):
        """
        節点をTriangulation3に追加する。
        Args:
            v(Vertex): 追加節点
        """
        tet = self.getIncludeTet(v)
        self.vertices.append(v)

        self.remove_tetrahedron(tet)

        for fi in tet.facets:
            self.dig_cavity(v, fi)

    def dig_cavity(self, u: Vertex, f_tgt: Facet):
        """
        Triangulation3内で追加節点uを四面体の外接球に含む四面体群を探索する。
        Args:
            u(Vertex): 追加節点
            f_tgt(Facet): 探索起点となる四面体面のFacetDescriptor

        Notes:
            'Delaunay Mesh Generation' Chapter5.2
        """
        # step1
        f_base = f_tgt.opposite()
        try:
            tet = self.face_adjacent_table[f_base]
        except KeyError:
            # step2
            # Triangle is already deleted
            return

        if tet.is_incircumsphere(u):
            # step3
            self.remove_tetrahedron(tet)
            for fi in tet.facets:
                # 探索始点Facetはスキップ
                if fi == f_base:
                    continue
                self.dig_cavity(u, fi)
        else:
            # step4
            new_tet = TetCell(u, f_tgt.v2, f_tgt.v1, f_tgt.v3, self)
            self.add_tetrahedron(new_tet)
