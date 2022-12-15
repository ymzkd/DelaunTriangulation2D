from __future__ import annotations

import math
from typing import List, Union, Dict

from geometric_trait3 import Point, Triangle, Plane, Tetrahedron

TOLERANCE = 1.0e-12


class Vertex(Point):
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

    def plane(self) -> Plane:
        return Plane(self.v1, self.v2, self.v3)


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

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    def is_incircumsphere(self, v: Vertex) -> bool:
        if self.is_infinite():
            for fi in self.facets:
                if fi.is_infinite():
                    continue
                if fi.plane().signed_distance(v) < 0.0:
                    return True
            return False
        else:
            sph = self.outer_sphere()
            return sph.isinside(v)

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
        v1 = vertices.pop()
        v2 = vertices.pop()
        v3 = vertices.pop()
        v4 = vertices.pop()
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
        count_before = len(self.face_adjacent_table)
        _ = self.face_adjacent_table.pop(tet.facets[0])
        _ = self.face_adjacent_table.pop(tet.facets[1])
        _ = self.face_adjacent_table.pop(tet.facets[2])
        _ = self.face_adjacent_table.pop(tet.facets[3])
        if count_before - len(self.face_adjacent_table) != 4:
            print("table remove is failed")
        self.tetrahedrons.remove(tet)

    def add_tetrahedron(self, tet: TetCell):
        count_before = len(self.face_adjacent_table)
        self.face_adjacent_table[tet.facets[0]] = tet
        self.face_adjacent_table[tet.facets[1]] = tet
        self.face_adjacent_table[tet.facets[2]] = tet
        self.face_adjacent_table[tet.facets[3]] = tet
        if len(self.face_adjacent_table) - count_before != 4:
            print("table add is failed")
        self.tetrahedrons.append(tet)

    def getIncludeTet(self, pt: Vertex) -> Union[TetCell, None]:
        for tet_i in self.tetrahedrons:
            if tet_i.is_incircumsphere(pt):
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
            # if f_tgt.is_infinite():
            #     tet_new = Tetrahedron(u, f_tgt.v1, f_tgt.v2, f_tgt.v3, self)
            #     self.add_tetrahedron(tet_new)
            # self.tetrahedrons.append(tet_new)
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
            self.add_tetrahedron(TetCell(u, f_tgt.v2, f_tgt.v1, f_tgt.v3, self))
