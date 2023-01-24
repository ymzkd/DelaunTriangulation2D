from __future__ import annotations

import copy
from enum import Enum
import math
from typing import List, Union, Dict

import numpy as np

from geometric_trait3 import Point3, Triangle, Plane, Tetrahedron, Line, Sphere, Vector3

TOLERANCE = 1.0e-12


class FacetLocationType(Enum):
    """このFacetがメッシュの領域内部にあるか、外部に属するかの属性
    """
    undefined = 0
    inside = 1
    outside = 2


class TetLocationType(Enum):
    """このTetrahedronがメッシュの領域内部にあるか、外部に属するかの属性
    """
    undefined = 0
    inside = 1
    outside = 2


class Vertex3(Point3):
    x: float
    y: float
    z: float
    infinite: bool

    def __init__(self, x: float, y: float, z: float, infinite: bool = False):
        super(Vertex3, self).__init__(x, y, z)
        self.infinite = infinite

    def transform(self, transmat) -> Vertex3:
        xyzw = np.ones([4])
        xyzw[0:3] = self.toarray()
        xyzw = transmat @ xyzw
        return Vertex3(*xyzw[:3], infinite=self.infinite)

    @staticmethod
    def ghost_vertex() -> Vertex3:
        return Vertex3(math.inf, math.inf, math.inf, infinite=True)


class Edge3(Line[Vertex3]):
    def __init__(self, v1: Vertex3, v2: Vertex3, mesh: Mesh3 = None):
        super(Edge3, self).__init__(v1, v2)
        self.mesh = mesh

    def opposite(self) -> Edge3:
        return Edge3(self.v2, self.v1, self.mesh)

    def ispt_rightside(self, pt: Point3, pln: Plane) -> bool:
        """入力点が入力平面上でこのエッジの右半開空間に位置するか

        Args:
            pt (Point3): 節点座標を表す三次元のベクトル
            pln (Plane): このエッジが配置される平面

        Returns:
            TriangulationPlaneの平面上で入力点がこのエッジの右側にあるかどうか？
        """
        vav = pt.toarray() - self.v1.toarray()
        vab = self.v2.toarray() - self.v1.toarray()
        return (pln.ez.toarray() @ np.cross(vav, vab)) > 1.0e-6

    def distance_inplane(self, pt: Point3, pln: Plane) -> float:
        """入力点の入力平面上でのこのエッジとの符号付き距離

        Args:
            pt (Point3): 節点座標を表す三次元のベクトル
            pln (Plane): このエッジが配置される平面

        Returns:
            float: 入力点の入力平面上でのこのエッジとの距離
        """
        vav = pt.toarray() - self.v1.toarray()
        vab = self.v2.toarray() - self.v1.toarray()
        area = (pln.ez.toarray() @ np.cross(vav, vab))
        return area / np.linalg.norm(vab)

    def is_infinite(self) -> bool:
        return any((self.v1.infinite, self.v2.infinite))

    @property
    def midpt(self) -> Vertex3:
        x = (self.v1.x + self.v2.x) * 0.5
        y = (self.v1.y + self.v2.y) * 0.5
        z = (self.v1.z + self.v2.z) * 0.5
        return Vertex3(x, y, z)

    def diametric_ball(self) -> Sphere:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Sphere(self.midpt, self.length() * 0.5)

    def to_segment(self) -> Segment3:
        return Segment3(self.v1, self.v2, self.mesh)


class Segment3(Edge3):
    children: List[Segment3]

    def __init__(self, v1: Vertex3, v2: Vertex3, mesh: Mesh3 = None):
        super(Segment3, self).__init__(v1, v2, mesh)
        self.children = []

    def __contains__(self, item: Segment3):
        if item == self:
            return True
        elif self.children:
            for si in self.children:
                if si.__contains__(item):
                    return True
        else:
            return False

    def opposite(self) -> Segment3:
        return Segment3(self.v2, self.v1, self.mesh)

    @property
    def midpt(self) -> Vertex3:
        x = (self.v1.x + self.v2.x) * 0.5
        y = (self.v1.y + self.v2.y) * 0.5
        z = (self.v1.z + self.v2.z) * 0.5
        return Vertex3(x, y, z)

    def diametric_ball(self) -> Sphere:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Sphere(self.midpt, self.length() * 0.5)

    def vertex_encroached(self, v: Vertex3) -> Segment3 | None:
        """
        頂点vがこの線分の直径円の中に位置するかどうか判定
        Args:
            v (Vertex3): 判定対称の頂点

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

    def flatten_child(self, reverse: bool = False) -> List[Segment3]:
        """この線分のchildを展開して下層の線分をリストにまとめる。

        Args:
            reverse(bool): すべての線分を反転させた結果をリストとして返す。
        Returns:
            List[Segment3]: この線分以下の樹構造のリスト
        """
        children = []
        if self.children:
            for child in self.children:
                children += child.flatten_child(reverse)
        else:
            if reverse:
                children.append(self.opposite())
            else:
                children.append(self)
        return children

    def __hash__(self):
        return hash(frozenset((self.v1, self.v2)))

    def to_edge(self) -> Edge3:
        """SegmentをEdgeに変換

        Returns:
            Edge3: Segmentに対応するEdge

        Note:
            mesh3ではSegmentは節点順に依存しないハッシュ値を生成するため
            節点順を考慮したハッシュ値を生成するためにはEdgeに変換する必要がある。
        """
        return Edge3(self.v1, self.v2, self.mesh)


class Facet3(Triangle):
    """
    Triangulation3dにおいて四面体の各面をCCWに格納する。
    テーブルのキーとして四面体の面と四面体の隣接関係を関連づけるために用いる。
    """
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3
    location: FacetLocationType

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3, location: FacetLocationType = FacetLocationType.undefined,
                 mesh: Union[TriangulationPLC, Mesh3, None] = None):
        super(Facet3, self).__init__(v1, v2, v3)
        self.mesh = mesh
        self.location = location
        if (not self.is_infinite()) and self.area() < 1.0e-6:
            raise ValueError(f"Facet area is too small {self}")

    def opposite(self) -> Facet3:
        return Facet3(self.v3, self.v2, self.v1, mesh=self.mesh)

    def is_infinite(self) -> bool:
        return any([self.v1.infinite, self.v2.infinite, self.v3.infinite])

    def point_at(self, v0: Vertex3) -> List[float]:
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

    def get_edge(self, idx: int) -> Edge3:
        if idx == 0:
            return Edge3(self.v1, self.v2, self.mesh)
        elif idx == 1:
            return Edge3(self.v2, self.v3, self.mesh)
        else:
            return Edge3(self.v3, self.v1, self.mesh)

    def is_incircumcircle(self, v: Vertex3) -> bool:
        """節点vがこの三角形の外接円の内部に位置するか
            節点vがこの三角形の外接円の内部に位置するかを判定。三角形がGhostTriangleの場合は
            finite edgeの半平面(左側・CCW)に節点が位置すれば外接円内部と判定する。

        Args:
            v: テスト対象の節点

        Returns:
            節点vがこの三角形の内部に位置する場合はTrue, その他の場合はFalse
        """
        if self.is_infinite():
            return self.infinite_is_incircumball(v)
            # # 節点が半平面に位置するか？
            # vid_inf = [v.infinite for v in self.vertices].index(True)
            # seg = self.get_edge((vid_inf + 1) % 3)
            # return not(seg.ispt_rightside(v, self.mesh.plane))
        else:
            # 三角形外接円に節点が含まれるか？
            sph = self.diametric_ball()
            return sph.isinside(v)

    def infinite_is_incircumball(self, v: Vertex3, delta=1.0e-06) -> bool:
        # 節点が半平面に位置するか？
        vid_inf = [v.infinite for v in self.vertices].index(True)
        edge = self.get_edge((vid_inf + 1) % 3)

        dist = edge.distance_inplane(v, self.mesh.plane)
        if dist < -delta:
            # 内側開集合
            return True
        else:
            ball = edge.diametric_ball()
            return abs(dist) < delta and ball.isinside(v, delta)

        # dist = fi.plane().signed_distance(v)
        # if dist < (0.0 - delta):  # TODO: ここも開集合とするために微小数値を引く必要ありそう
        #     return True
        # else:
        #     ball = fi.diametric_ball()
        #     return abs(dist) < delta and ball.isinside(v)
        # return not (seg.ispt_rightside(v, self.mesh.plane))


class TetCell3(Tetrahedron[Vertex3, Facet3]):
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3
    v4: Vertex3
    vertices: List[Vertex3]
    facets: List[Facet3]
    location: TetLocationType

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3, v4: Vertex3,
                 mesh: Mesh3 = None):
        super(TetCell3, self).__init__(v1, v2, v3, v4)
        self.mesh = mesh
        f1 = Facet3(v2, v4, v3, mesh=self.mesh)
        f2 = Facet3(v1, v3, v4, mesh=self.mesh)
        f3 = Facet3(v1, v4, v2, mesh=self.mesh)
        f4 = Facet3(v1, v2, v3, mesh=self.mesh)
        self.facets = [f1, f2, f3, f4]

        self.location = TetLocationType.undefined
        if not self.is_infinite() and abs(self.volume()) < 1.0e-6:
            raise ValueError("volume is too small.")

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    def is_incircumsphere(self, v: Vertex3) -> bool:
        if self.is_infinite():
            return self.infinite_is_incircumsphere(v)
        else:
            sph = self.outer_sphere()
            return sph.isinside(v)

    def infinite_is_incircumsphere(self, v: Vertex3, delta=1.0e-06) -> bool:
        """Ghost TetrahedronのOpenCircum ball内外判定

        Args:
            v (Vertex3): テスト対象節点
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

    def is_inside(self, v: Vertex3) -> bool:
        if self.is_infinite():
            return self.infinite_is_incircumsphere(v)
        else:
            judges = [fi.plane().signed_distance(v) < 0.0 for fi in self.facets]
            return all(judges)

    def fix_orientation(self):
        if self.orient() < 0.0:
            self.vertices[:3], self.vertices[3] = self.vertices[1:], self.vertices[0]
            f1 = Facet3(self.v2, self.v4, self.v3, mesh=self.mesh)
            f2 = Facet3(self.v1, self.v3, self.v4, mesh=self.mesh)
            f3 = Facet3(self.v1, self.v4, self.v2, mesh=self.mesh)
            f4 = Facet3(self.v1, self.v2, self.v3, mesh=self.mesh)
            self.facets = [f1, f2, f3, f4]


class Polygon3:
    """三次元上の多角形ポリゴン
    n個の節点のリストにより、三次元上の多角形ポリゴンを表す。
    """
    segments: List[Segment3]

    def __init__(self, vertices: List[Vertex3]):
        self.vertices = vertices
        vertex_num = len(self.vertices)
        self.segments = [Segment3(self.vertices[i], self.vertices[(i + 1) % vertex_num]) for i in range(vertex_num)]

    @property
    def count(self) -> int:
        return len(self.vertices)

    def _area_vector(self) -> Vector3:
        sum_vec = Vector3.zero()
        for i in range(self.count):
            vi = self.vertices[i]
            vi1 = self.vertices[(i + 1) % self.count]
            sum_vec += vi.outer_product(vi1)
        else:
            sum_vec *= 0.5
        return sum_vec

    @property
    def area(self) -> float:
        return self._area_vector() .length()

    @property
    def normal(self) -> Vector3:
        return self._area_vector() / self.area

    def __contains__(self, item: Segment3) -> bool:
        for seg_i in self.segments:
            if item in seg_i:
                return True
        return False


class Mesh3:
    """PLCの四面体分割
    DelTetPLC "Tetrahedral meshing of PLCs"によるPLCの四面体分割

    Notes:
            'Delaunay Mesh Generation' page 175
    """
    vertices: List[Vertex3]
    tetrahedrons: List[TetCell3]
    face_adjacent_table: Dict[Facet3, TetCell3]
    segment_plc_table: Dict[Segment3, List[TriangulationPLC]]
    plc_triangulations: List[TriangulationPLC]

    def __init__(self, vertices: List[Vertex3], polygons: List[Polygon3], p: float):
        """

        Args:
            vertices:
            polygons:
            p(float): Facetの外接球半径-最短エッジ比の最大値を指定, 2.0以上でアルゴリズム上終了することが保証されている。
        """

        # Step1 Initialize dates
        self.face_adjacent_table = {}
        self.tetrahedrons = []
        self.p = p

        # Step2a Compute Del(S)
        self.triangulate(vertices)

        # Step2b Compute Del(S cap g)
        self.plc_triangulations = []
        self.segment_plc_table = {}
        for pi in polygons:
            plc_triangulate = TriangulationPLC(pi)
            self.plc_triangulations.append(plc_triangulate)
            for i, si in enumerate(pi.segments):
                if si in self.segment_plc_table.keys():
                    # TriangulationPLCの対応するSegmentを置き換え
                    keys = list(self.segment_plc_table.keys())
                    idx = keys.index(si)
                    plc_triangulate.outerloop.segments[i] = keys[idx]
                    self.segment_plc_table[si].append(plc_triangulate)
                else:
                    self.segment_plc_table[si] = [plc_triangulate]

        # Step3 resolve segment encroach
        self.resolve_segment_encroach()

        # Step4 resolve subpolygon encroach
        self.resolve_face_encroach()
        for tri in self.plc_triangulations:
            tri.mark_inout()
        self.mark_inout()

        # Step5 refine tetrahedron
        self.refine_tetra()

    def triangulate(self, vertices: List[Vertex3]):
        insert_vertices = copy.copy(vertices)
        if len(insert_vertices) < 4:
            print("4つ以上の節点入力が必要")

        # Create initial tetrahedron
        gv = Vertex3(math.inf, math.inf, math.inf, infinite=True)
        v1 = insert_vertices.pop()
        v2 = insert_vertices.pop()
        v3 = insert_vertices.pop()

        v4 = None; t1 = None
        for v in insert_vertices:
            v4 = v
            t1 = TetCell3(v1, v2, v3, v4, mesh=self)
            t1.fix_orientation()
            if t1.volume() > 1.0e-6:
                break
        insert_vertices.remove(v4)

        gt1 = TetCell3(gv, t1.v1, t1.v2, t1.v3, mesh=self)
        gt2 = TetCell3(gv, t1.v1, t1.v3, t1.v4, mesh=self)
        gt3 = TetCell3(gv, t1.v2, t1.v1, t1.v4, mesh=self)
        gt4 = TetCell3(gv, t1.v2, t1.v4, t1.v3, mesh=self)

        self.vertices = [gv, v1, v2, v3, v4]
        self.add_tetrahedron(t1)
        self.add_tetrahedron(gt1)
        self.add_tetrahedron(gt2)
        self.add_tetrahedron(gt3)
        self.add_tetrahedron(gt4)

        for i, vi in enumerate(insert_vertices):
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

    def finite_tetrahedrons(self) -> List[TetCell3]:
        return [ti for ti in self.tetrahedrons if not ti.is_infinite()]

    def finite_vertices(self) -> List[Vertex3]:
        return [vi for vi in self.vertices if not vi.infinite]

    def remove_tetrahedron(self, tet: TetCell3):
        del self.face_adjacent_table[tet.facets[0]]
        del self.face_adjacent_table[tet.facets[1]]
        del self.face_adjacent_table[tet.facets[2]]
        del self.face_adjacent_table[tet.facets[3]]
        self.tetrahedrons.remove(tet)

    def add_tetrahedron(self, tet: TetCell3):
        count_before = len(self.face_adjacent_table)
        self.face_adjacent_table[tet.facets[0]] = tet
        self.face_adjacent_table[tet.facets[1]] = tet
        self.face_adjacent_table[tet.facets[2]] = tet
        self.face_adjacent_table[tet.facets[3]] = tet
        if len(self.face_adjacent_table) - count_before != 4:
            print("table add is failed")
        self.tetrahedrons.append(tet)

    def split_segment(self, seg: Segment3) -> List[Segment3]:
        # 線分自体を2つに分割する。
        mid_pt = seg.midpt
        seg1 = Segment3(seg.v1, mid_pt, mesh=self)
        seg2 = Segment3(mid_pt, seg.v2, mesh=self)

        # 対応するPLC面の局所PlaneTriangulationに点を挿入
        for seg_i, plcs in self.segment_plc_table.items():
            if seg in seg_i:
                for plc in plcs:
                    plc.add_vertex(mid_pt)

        # 全体のTriangulationに点を挿入
        self.add_vertex(mid_pt)

        return [seg1, seg2]

    # def split_tetra(self):
    #     for t in self.tetrahedrons:
    #         if t.edge_radius_ratio() > self.p:
    #             # step1
    #             sph = t.outer_sphere()
    #             if self.segment_encroach()

    def getIncludeTet(self, pt: Vertex3) -> TetCell3:
        for tet_i in self.tetrahedrons:
            # if tet_i.is_incircumsphere(pt):
            if tet_i.is_inside(pt):
                return tet_i
        else:
            raise ValueError(f"Cant find {pt} included tetrahedron.")

    def add_vertex(self, v: Vertex3):
        """
        節点をTriangulation3に追加する。
        Args:
            v(Vertex3): 追加節点
        """
        tet = self.getIncludeTet(v)
        self.vertices.append(v)

        self.remove_tetrahedron(tet)

        for fi in tet.facets:
            self.dig_cavity(v, fi)

    def dig_cavity(self, u: Vertex3, f_tgt: Facet3):
        """
        Triangulation3内で追加節点uを四面体の外接球に含む四面体群を探索する。
        Args:
            u(Vertex3): 追加節点
            f_tgt(Facet3): 探索起点となる四面体面のFacetDescriptor

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
            new_tet = TetCell3(u, f_tgt.v2, f_tgt.v1, f_tgt.v3, self)
            self.add_tetrahedron(new_tet)

    def resolve_segment_encroach(self):
        while s := self.pick_segment_encroach():
            segments = self.split_segment(s)
            s.children = segments

    def pick_segment_encroach(self) -> Segment3 | None:
        for vi in self.finite_vertices():
            if seg := self.segment_encroach(vi):
                return seg
        return None

    def segment_encroach(self, v: Vertex3) -> Segment3 | None:
        for seg_i in self.segment_plc_table:
            if v is seg_i.v1 or v is seg_i.v2:
                continue
            if s := seg_i.vertex_encroached(v):
                return s
        return None

    def resolve_face_encroach(self):
        """PLCへのencroachを逐次的に解消する。

        Notes:
            この関数が呼ばれる前にsegmentのencroachを解消しておく必要がある。
        """
        while f := self.pick_face_encroach():
            # segmentのencroachが優先なので、facetの重心がsegmentをencroachしていないか調べる
            new_pt = f.diametric_ball().center
            trig_pln = f.mesh
            vertex = Vertex3(new_pt.x, new_pt.y, new_pt.z)
            # それ以外の場合はfacetのencroachを以下で解消
            # encroachが見つかったfaceの属する平面Triangulationにfacetの重心を追加
            trig_pln.add_vertex(vertex)
            # facetの重心を全体のTriangulationにも追加
            self.add_vertex(vertex)

            self.resolve_segment_encroach()

    def refine_tetra(self):
        """クライテリアを満たさない四面体を改良するために節点を追加

        Notes:
            この関数が呼ばれる前にsegment,PLCのencroachを解消しておく必要がある。
        """
        while t := self.pick_bad_tetra():
            # step1
            sph = t.outer_sphere()
            self.add_vertex(Vertex3(sph.center.x, sph.center.y, sph.center.z))

            self.resolve_segment_encroach()
            self.resolve_face_encroach()
            for tri in self.plc_triangulations:
                tri.mark_inout()
            self.mark_inout()

    def pick_face_encroach(self) -> Facet3 | None:
        for vi in self.finite_vertices():
            for tri_i in self.plc_triangulations:
                if fc_i := tri_i.encroached_facet(vi):
                    return fc_i
        return None

    def pick_bad_tetra(self) -> TetCell3 | None:
        for ti in self.tetrahedrons:
            if ti.location != TetLocationType.inside:
                continue
            if ti.edge_radius_ratio() > self.p:
                return ti
        return None

    def mark_inout(self):
        # finite->inside, infinite->outside
        for cell in self.tetrahedrons:
            if cell.is_infinite():
                cell.location = TetLocationType.outside
            else:
                cell.location = TetLocationType.inside

        # PLCs外側を処理
        for ti in self.tetrahedrons:
            if not ti.is_infinite():
                continue

            finite_facet = None
            for fi in ti.facets:
                if fi.is_infinite():
                    continue
                else:
                    finite_facet = fi
                    break

            self._mark_outside(finite_facet)

    def _mark_outside(self, f: Facet3):
        ti = self.face_adjacent_table[f.opposite()]
        if ti.location == TetLocationType.outside:
            return

        plc = self.face_attached_plc(f)
        if plc:
            return
        else:
            ti.location = TetLocationType.outside
            for fi in ti.facets:
                if fi == f:
                    continue
                self._mark_outside(fi)

    def face_attached_plc(self, f: Facet3):
        for plc in self.plc_triangulations:
            if plc.face_attached(f):
                return plc
        return None

    def mark_inout_save(self):
        # finite->inside, infinite->outside
        for cell in self.tetrahedrons:
            if cell.is_infinite():
                cell.location = TetLocationType.outside
            else:
                cell.location = TetLocationType.inside

        # PLCs外側を処理
        sub_facets = []
        for plc in self.plc_triangulations:
            sub_facets.extend(plc.inside_triangles())
        for f in sub_facets:
            cell = self.face_adjacent_table[f]
            print(cell)
            # self._mark_outside(f)

    def _mark_outside_save(self, f: Facet3):
        f_neigh = f.opposite()
        if self._facet_contains(f_neigh):
            print('facet contains')
            return

        t_neigh = self.face_adjacent_table[f_neigh]
        print('find facet')
        if t_neigh.location == TetLocationType.outside:
            return

        t_neigh.location = TetLocationType.outside
        idx = t_neigh.facets.index(f_neigh)
        f1 = t_neigh.facets[(idx + 1) % 4]
        f2 = t_neigh.facets[(idx + 2) % 4]
        f3 = t_neigh.facets[(idx + 3) % 4]
        self._mark_outside(f1)
        self._mark_outside(f2)
        self._mark_outside(f3)

    def _facet_contains(self, f: Facet3) -> bool:
        for plc in self.plc_triangulations:
            if f in plc.inside_triangles():
                return True
        return False


class TriangulationPLC:
    triangles: List[Facet3]
    edge_triangle_table: Dict[Edge3, Facet3]
    outerloop: Polygon3

    def __init__(self, outerloop: Polygon3):
        self.outerloop = outerloop
        self.edge_triangle_table = {}
        self.triangles = []

        insert_vertices = copy.copy(outerloop.vertices)
        gv = Vertex3.ghost_vertex()
        v1 = insert_vertices.pop(0)
        v2 = insert_vertices.pop(0)
        v3 = insert_vertices.pop(0)

        self.plane = Plane(v1, v2, v3)
        t1 = Facet3(v1, v2, v3, mesh=self)
        gt1 = Facet3(gv, t1.v2, t1.v1, mesh=self)
        gt2 = Facet3(gv, t1.v3, t1.v2, mesh=self)
        gt3 = Facet3(gv, t1.v1, t1.v3, mesh=self)
        self.vertices = [gv, v1, v2, v3]
        self.add_triangle(t1)
        self.add_triangle(gt1)
        self.add_triangle(gt2)
        self.add_triangle(gt3)

        for vi in insert_vertices:
            self.add_vertex(vi)

    @property
    def finite_vertices(self):
        return [v for v in self.vertices if not v.infinite]

    def remove_triangle(self, tri: Facet3):
        _ = self.edge_triangle_table.pop(tri.get_edge(0))
        _ = self.edge_triangle_table.pop(tri.get_edge(1))
        _ = self.edge_triangle_table.pop(tri.get_edge(2))
        self.triangles.remove(tri)

    def add_triangle(self, tri: Facet3):
        e1 = tri.get_edge(0)
        e2 = tri.get_edge(1)
        e3 = tri.get_edge(2)
        self.edge_triangle_table[e1] = tri
        self.edge_triangle_table[e2] = tri
        self.edge_triangle_table[e3] = tri
        self.triangles.append(tri)

    def locate(self, v: Vertex3) -> Facet3:
        """任意の点を外接円内部に含む三角形を探索する。
        もし、既存のTriangulationの外側に点が挿入された場合は、
        半平面に挿入節点を含むGhost Triangleを返す。

        Args:
            v (Vertex3): 頂点

        Returns:
            Facet3: 入力節点を含む三角形
        """
        for tri in self.triangles:
            if tri.is_incircumcircle(v):
                return tri
        else:
            raise Exception(f"No triangles include point coordinate: [{v.x},{v.y},{v.z}]")

    def dig_cavity(self, u: Vertex3, edge: Edge3):
        """
        Triangulation3内で追加節点uを四面体の外接球に含む四面体群を探索する。
        Args:
            u(Vertex3): 追加節点
            edge(Segment3): 探索起点となる四面体面のFacetDescriptor

        Notes:
            'Delaunay Mesh Generation' Chapter3.4
        """
        base_edge = edge.opposite()
        try:
            tri = self.edge_triangle_table[base_edge]
        except KeyError:
            # step2
            # Triangle is already deleted
            return

        if tri.is_incircumcircle(u):
            edges = [tri.get_edge(i) for i in range(3)]
            self.remove_triangle(tri)
            for seg in edges:
                if seg == base_edge:
                    continue
                self.dig_cavity(u, seg)
        else:
            self.add_triangle(Facet3(u, edge.v1, edge.v2, mesh=self))

    def add_vertex(self, v: Vertex3) -> None:
        """頂点vを挿入

        Args:
            v (Vertex3): 挿入節点
        """
        tri = self.locate(v)
        self.vertices.append(v)
        edges = [tri.get_edge(i) for i in range(3)]
        self.remove_triangle(tri)
        for edge in edges:
            self.dig_cavity(v, edge)

    def finite_triangles(self) -> List[Facet3]:
        """infinite三角形を除いた三角形のリストを得る。

        Returns:
            List[Facet3]: infinite三角形を除いた三角形のリスト
        """
        return [f for f in self.triangles if not f.is_infinite()]

    def inside_triangles(self) -> List[Facet3]:
        """内部三角形を除いた三角形のリストを得る。

        Returns:
            List[Facet3]: 内部三角形を除いた三角形のリスト
        """
        return [f for f in self.triangles if f.location == FacetLocationType.inside]

    def encroached_facet(self, v: Vertex3) -> Facet3 | None:
        for fi in self.finite_triangles():
            if any([vi == v for vi in fi.vertices]):
                return None
            sph = fi.diametric_ball()
            if sph.isinside(v):
                return fi
        return None

    def mark_inout(self):
        # 最初にfinite->inside, infinite->outsideに初期化
        for f in self.triangles:
            if f.is_infinite():
                f.location = FacetLocationType.outside
            else:
                f.location = FacetLocationType.inside

        # loopを処理
        sub_segments = []
        for seg, v in zip(self.outerloop.segments, self.outerloop.vertices):
            sub_segments.extend(seg.flatten_child(seg.v1 != v))
        for seg in sub_segments:
            self._mark_outside(seg.to_edge(), self.outerloop)

    def _mark_outside(self, e: Edge3, loop: Polygon3):
        # 隣接Facetは境界の内側
        e_neigh = e.opposite()
        # if e_neigh.to_segment() in loop:  # ここでsegmentの向きが考慮できないことが効いてくる？　　
        #     return

        f_neigh = self.edge_triangle_table[e_neigh]
        # 隣接Facetは調査済み
        if f_neigh.location == FacetLocationType.outside:
            return

        f_neigh.location = FacetLocationType.outside
        idx = f_neigh.vertices.index(e_neigh.v2)
        e1 = f_neigh.get_edge(idx)
        e2 = f_neigh.get_edge((idx + 1) % 3)
        if e1.to_segment() not in loop:
            self._mark_outside(e1, loop)
        if e2.to_segment() not in loop:
            self._mark_outside(e2, loop)

    def face_attached(self, f: Facet3):
        """入力Facet fがこのPLCに含まれるかを判定,
        ただし、Facetの向きを問わず、同じVertex-Facet接続関係であるかどうかは問わない

        Notes:
            本線方向や、入力Facetの重心に対するlocate等を行えば方向や領域内外を考慮した判定も可能と思われる。
        """
        return all(vi in self.vertices for vi in f.vertices)

