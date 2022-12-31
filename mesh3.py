from __future__ import annotations

import math
from typing import List, Union, Dict, Set

import numpy as np

from geometric_trait3 import Point, Triangle, Plane, Tetrahedron, Line, Sphere
import triangulation_plane as tp

TOLERANCE = 1.0e-12


class Vertex(Point):
    x: float
    y: float
    z: float
    infinite: bool

    def __init__(self, x: float, y: float, z: float, infinite: bool = False):
        super(Vertex, self).__init__(x, y, z)
        self.infinite = infinite


class Edge(Line[Vertex]):
    def __init__(self, v1: Vertex, v2: Vertex, mesh: Mesh = None):
        super(Edge, self).__init__(v1, v2)
        self.mesh = mesh


class Segment(Edge):
    children: List[Segment]

    def __init__(self, v1: Vertex, v2: Vertex, mesh: Mesh = None):
        super(Segment, self).__init__(v1, v2, mesh)
        self.children = []

    def __contains__(self, item: Segment):
        if item == self:
            return True
        elif self.children:
            for si in self.children:
                if si.__contains__(item):
                    return True
        else:
            return False

    @property
    def midpt(self) -> Vertex:
        x = (self.v1.x + self.v2.x) * 0.5
        y = (self.v1.y + self.v2.y) * 0.5
        z = (self.v1.z + self.v2.z) * 0.5
        return Vertex(x, y, z)

    def diametric_ball(self) -> Sphere:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Sphere(self.midpt, self.length() * 0.5)

    def vertex_encroached(self, v: Vertex) -> Segment | None:
        """
        頂点vがこの線分の直径円の中に位置するかどうか判定
        Args:
            v (Vertex): 判定対称の頂点

        Returns:
            (Segment | None) 頂点vがこの線分の直径円の中に位置するか。
        """
        if v is self.v1 or v is self.v2:
            return None

        ball = self.diametric_ball()
        if ball.isinside(v):
            if self.children:
                for seg_i in self.children:
                    if s := seg_i.vertex_encroached(v):
                        return s
            else:
                return self
        else:
            return None

    def flatten_child(self) -> List[Segment]:
        """この線分のchildを展開して下層の線分をリストにまとめる。

        Returns:
            List[Segment]: この線分以下の樹構造のリスト
        """
        children = []
        if self.children:
            for child in self.children:
                children += child.flatten_child()
        else:
            children.append(self)
        return children

    def __hash__(self):
        return hash(frozenset((self.v1, self.v2)))


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


class TetCell(Tetrahedron[Vertex, Facet]):
    v1: Vertex
    v2: Vertex
    v3: Vertex
    v4: Vertex
    vertices: List[Vertex]
    facets: List[Facet]

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex, v4: Vertex,
                 mesh: Mesh = None, initialize=False):
        super(TetCell, self).__init__(v1, v2, v3, v4)
        self.mesh = mesh
        f1 = Facet(v2, v4, v3)
        f2 = Facet(v1, v3, v4)
        f3 = Facet(v1, v4, v2)
        f4 = Facet(v1, v2, v3)
        self.facets = [f1, f2, f3, f4]
        if not initialize:
            if not self.is_infinite():
                if abs(self.volume()) < 1.0e-6:
                    raise ValueError("volume is too small.")

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
            return self.infinite_is_incircumsphere(v)
        else:
            judges = [fi.plane().signed_distance(v) < 0.0 for fi in self.facets]
            return all(judges)

    def fix_orientation(self):
        if self.orient() < 0.0:
            self.vertices[:3], self.vertices[3] = self.vertices[1:], self.vertices[0]
            f1 = Facet(self.v2, self.v4, self.v3)
            f2 = Facet(self.v1, self.v3, self.v4)
            f3 = Facet(self.v1, self.v4, self.v2)
            f4 = Facet(self.v1, self.v2, self.v3)
            self.facets = [f1, f2, f3, f4]


class Polygon:
    def __init__(self, vertices: List[Vertex]):
        self.vertices = vertices

    @property
    def segments(self) -> Set[Segment]:
        vertex_num = len(self.vertices)
        return set(Segment(self.vertices[i], self.vertices[(i+1)%vertex_num]) for i in range(vertex_num))


class Mesh:
    """PLCの四面体分割
    DelTetPLC 'Tetrahedral meshing of PLCs'によるPLCの四面体分割

    Notes:
            'Delaunay Mesh Generation' page 175
    """
    vertices: List[Vertex]
    tetrahedrons: List[TetCell]
    face_adjacent_table: Dict[Facet, TetCell]
    segment_plc_table: Dict[Segment, List[tp.TriangulationPlane]]
    plc_triangulations: List[tp.TriangulationPlane]

    def __init__(self, vertices: List[Vertex], polygons: List[Polygon]):

        # Step1 Initialize dates
        self.face_adjacent_table = {}
        self.tetrahedrons = []

        # Step2a Compute Del(S)
        self.triangulate(vertices)

        # Step2b Compute Del(S cap g)
        self.plc_triangulations = []
        self.segment_plc_table = {}
        for pi in polygons:
            plc_triangulate = tp.TriangulationPlane(pi.vertices)
            self.plc_triangulations.append(plc_triangulate)
            for si in pi.segments:
                if si in self.segment_plc_table:
                    self.segment_plc_table[si].append(plc_triangulate)
                else:
                    self.segment_plc_table[si] = [plc_triangulate]

        # Step3 resolve segment encroach
        self.resolve_segment_encroach()

        # Step4 resolve subpolygon encroach

        # Step5 refine tetrahedron

    def triangulate(self, vertices: List[Vertex]):
        if len(vertices) < 4:
            print("4つ以上の節点入力が必要")

        # Create initial tetrahedron
        gv = Vertex(math.inf, math.inf, math.inf, infinite=True)
        v1 = vertices.pop()
        v2 = vertices.pop()
        v3 = vertices.pop()

        v4 = None; t1 = None
        for v in vertices:
            v4 = v
            t1 = TetCell(v1, v2, v3, v4, mesh=self, initialize=True)
            t1.fix_orientation()
            if t1.volume() > 1.0e-6:
                break
        vertices.remove(v4)

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
        del self.face_adjacent_table[tet.facets[0]]
        del self.face_adjacent_table[tet.facets[1]]
        del self.face_adjacent_table[tet.facets[2]]
        del self.face_adjacent_table[tet.facets[3]]
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

    def split_segment(self, seg: Segment) -> List[Segment]:
        # 線分自体を2つに分割する。
        mid_pt = seg.midpt
        seg1 = Segment(seg.v1, mid_pt, mesh=self)
        seg2 = Segment(mid_pt, seg.v2, mesh=self)

        # 対応するPLC面の局所PlaneTriangulationに点を挿入
        for seg_i, plcs in self.segment_plc_table.items():
            if seg in seg_i:
                for plc in plcs:
                    plc.add_vertex(mid_pt)

        # 全体のTriangulationに点を挿入
        self.add_vertex(mid_pt)

        return [seg1, seg2]

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
            new_tet = TetCell(u, f_tgt.v2, f_tgt.v1, f_tgt.v3, self)
            self.add_tetrahedron(new_tet)

    def resolve_segment_encroach(self):
        while s := self.pick_segment_encroach():
            segments = self.split_segment(s)
            s.children = segments

    def pick_segment_encroach(self) -> Segment | None:
        for vi in self.finite_vertices():
            for seg_i in self.segment_plc_table:
                if vi is seg_i.v1 or vi is seg_i.v2:
                    continue
                if s := seg_i.vertex_encroached(vi):
                    return s
        return None

    def resolve_face_encroach(self):
        while f := self.pick_face_encroach():
            # segmentのencroachが優先なので、facetの重心がsegmentをencroachしていないか調べる
            # segmentをencroachしていたらsegmentのencroachを解消して再度ループ
            # それ以外の場合はfacetのencroachを以下で解消
                # encroachが見つかったfaceの属する平面Triangulationにfacetの重心を追加
                # facetの重心を全体のTriangulationにも追加
            pass

    # def pick_face_encroach(self) -> Facet | None:
    #     for vi in self.finite_vertices():
    #         for tri_i in self.plc_triangulations:
    #             if fc_i := tri_i.encroached_facet(vi):
    #                 return fc_i
    #     return None
