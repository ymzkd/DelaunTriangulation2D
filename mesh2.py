from __future__ import annotations

import math
from enum import Enum
from typing import List, Union, Dict, Tuple

import numpy as np

from geometric_trait2 import Point, Line, Circle, Triangle, Vector

TOLERANCE = 1.0e-12


class TriangleLocationType(Enum):
    """このFacetがメッシュの領域内部にあるか、外部に属するかの属性
    """
    undefined = 0
    inside = 1
    outside = 2


class SegmentPosition(Enum):
    """Segmentの端部を指定する識別子
    """
    iend = 1
    jend = 2
    both = 3


class VertexSource(Enum):
    """頂点の属性情報
    """
    fixed = 1
    auto = 2


class Vertex(Point):
    """メッシュ頂点クラス

    Attributes:
        x, y (float): 頂点座標
        mesh (Mesh): この頂点が属するMesh
        infinite (bool): この頂点がGhostVertex,すなわち無限遠であるならばTrue,そうでない場合はFalse
                         Meshに含まれるGhostVertexは1つのみである点に注意
        source (VertexSource): この頂点の属性データ。ユーザー指定、アルゴリズム自動生成などの属性を指定
        incident_facet (Facet): この節点を含む任意の代表Facet

    TODO:
        meshというAttributeが必要かどうか検討,Triangulation後にちゃんとセットされていないと思われる。
    """
    incident_facet: Facet | None

    def __init__(self, x, y, mesh: Mesh = None, infinite: bool = False,
                 source: VertexSource = VertexSource.fixed):
        super(Vertex, self).__init__(x, y)
        self.mesh = mesh
        self.infinite = infinite
        self.source = source
        self.incident_facet = None


class Edge(Line[Vertex]):

    def __init__(self, v1: Vertex, v2: Vertex, mesh: Mesh = None):
        super(Edge, self).__init__(v1, v2)
        self.mesh = mesh

    def opposite(self) -> Edge:
        return Edge(self.v2, self.v1, self.mesh)

    @property
    def midpt(self) -> Vertex:
        x = (self.v1.x + self.v2.x) * 0.5
        y = (self.v1.y + self.v2.y) * 0.5
        return Vertex(x, y, mesh=self.mesh)

    def diametric_ball(self) -> Circle:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Circle(self.midpt, self.length() * 0.5)


class Segment(Edge):
    """2つの節点を結ぶ線分
    Attributes:
        v1 (Vertex): 節点1
        v2 (Vertex): 節点2
    """
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
        return Vertex(x, y, mesh=self.mesh)

    def diametric_ball(self) -> Circle:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Circle(self.midpt, self.length() * 0.5)

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

        cir = self.diametric_ball()
        # if cir.ispoint_inside([v.x, v.y]):
        if cir.ispoint_inside(v):
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

    def pick_adjacent_vertex(self, other: Segment):
        """
        入力線分はこの線分のどの頂点において接触しているか？あるいは接触していない場合はNoneを返す。
        Args:
            other (Segment): 確認する対象線分

        Returns (None | Vertex2):

        """
        if self.v1 == other.v1 or self.v1 == other.v2:
            return self.v1
        elif self.v2 == other.v1 or self.v2 == other.v2:
            return self.v2
        else:
            return None

    def adjacent_segments(self, pos: SegmentPosition):
        """この線分に接続する線分のリストを生成

        この線分に接続する線分のリストを生成する。この線分自体は含まず、
        取得される線分はSegmentツリーの親である。

        Args:
            pos (SegmentPosition): この線分のどちら側の節点の接続線分を抽出するか？

        Returns:
            List[Segment]: この線分に接続する線分のリスト
        """
        if pos == SegmentPosition.iend:
            tgt_vertex = self.v1
        else:  # pos == SegmentPosition.jend
            tgt_vertex = self.v2

        # どうやってこの線分自身の親線分をはじくか？
        segments = []
        for si in self.mesh.segments_all:
            adj = (si.v1 == tgt_vertex or si.v2 == tgt_vertex)
            if adj and (self not in si):
                segments.append(si)
        return segments

    def adjacent_min_angle(self, pos: SegmentPosition) -> float:
        """
        線分のi端/j端に交差する線分との最小角度を求める。
        Args:
            pos (SegmentPosition): i端/j端を指定する指定子

        Returns:
            float: 位置に応じた最小交差角度で、交差がない場合はPiを返す。

        """
        segments = self.adjacent_segments(pos)
        if len(segments) == 0:
            return math.pi

        pivot = self.v1 if pos == SegmentPosition.iend else self.v2
        angles = [self.angle_segment(si, pivot) for si in segments]
        return min(angles)

    def direction_from_pivot(self, pivot: Vertex):
        """
        線分の方向ベクトルを指定した一端から他の一端へのベクトルとして求める。
        Args:
            pivot (Vertex): 線分の一端の節点でpivotからの方向ベクトルを求める。

        Returns:
            np.array: 方向ベクトル
        """
        if self.v1 == pivot:
            return np.array([self.v2.x - self.v1.x, self.v2.y - self.v1.y])
        elif self.v2 == pivot:
            return np.array([self.v1.x - self.v2.x, self.v1.y - self.v2.y])
        else:
            raise ValueError("argument 'pivot' must be equal to either node of the segment.")

    def angle_segment(self, other: Segment, pivot: Vertex = None) -> float:
        """
        一端を共有する他の線分との交差角度
        Args:
            other: 交差する線分
            pivot: 交差する節点、入力がない場合はメソッド内部で調べる

        Returns:
            線分同士の交差角度[rad: 0～pi]
        """
        if pivot is None:
            pivot = self.pick_adjacent_vertex(other)

        v1 = self.direction_from_pivot(pivot)
        v2 = other.direction_from_pivot(pivot)
        norm_product = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm_product <= -1.0:
            return math.pi
        else:
            return math.acos(norm_product)

    def encroach_type(self) -> Union[SegmentPosition, None]:
        """
        この線分が他の線分といずれかの端部で鋭角を成すか、両端に鋭角を持つか、それ以外であるか
        Returns:
            それぞれの場合におけるSegmentPosition
        """
        border_angle = math.pi / 2
        check_i = self.adjacent_min_angle(SegmentPosition.iend) < border_angle
        check_j = self.adjacent_min_angle(SegmentPosition.jend) < border_angle
        if check_i and check_j:
            return SegmentPosition.both
        elif check_i:
            return SegmentPosition.iend
        elif check_j:
            return SegmentPosition.jend
        else:
            return None

    def incident_facets(self) -> tuple[Facet | None, Facet | None]:
        """この線分の両側に位置するFacetを取得

        Returns:
            Facet: この線分の両側に位置するFacetを線分方向の左,右の順番で格納。
                   線分側面にFacetが存在しない場合はNoneを返す。
        TODO:
            利用を意図していた内外判定部分では利用しなくなったので必要かを確認
        """
        facets = self.mesh.incident_faces(self.v1)
        left, right = None, None
        for fi in facets:
            if self.v2 in fi.vertices:
                idx1 = fi.vertices.index(self.v1)
                if fi.vertices[(idx1 + 1) % 3] == self.v2:
                    left = fi  # right side
                else:
                    right = fi  # left side
        return left, right


class Facet(Triangle[Vertex]):
    """一つの三角形メッシュ面

    Attributes:
        v1, v2, v3 (Vertex) : メッシュ頂点
        mesh (Mesh): この頂点が属するMesh
        locationtype (TriangleLocationType): このFacetがメッシュの領域内部にあるか、外部に属するかの属性
    """
    vertices: List[Vertex]
    locked: bool

    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex,
                 mesh: Mesh = None, locationtype = TriangleLocationType.undefined, force_create: bool=False):
        super(Facet, self).__init__(v1, v2, v3)

        if not(self.is_infinite() or force_create):
            if abs(self.area()) < 0.0000001:
                raise ValueError(f"Triangle area is too small.[{self.area()}]\n{self}")

        # Update vertex incident facet
        v1.incident_facet = self
        v2.incident_facet = self
        v3.incident_facet = self

        self.mesh = mesh
        self.locationtype = locationtype
        self.locked = False

    @property
    def edges(self) -> List[Edge]:
        return [Edge(self.vertices[i], self.vertices[(i+1)%3]) for i in range(3)]

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
            return self.__infinite_is_incircumcircle(v, 0.000001)
        else:
            # 三角形外接円に節点が含まれるか？
            cir = self.outer_circle()
            return cir.ispoint_inside(v)
            # return cir.ispoint_inside(v.toarray())

    def __infinite_is_incircumcircle(self, v: Vertex, delta=0.0001) -> bool:
        """infinite triangle(ghost triangle)に対して節点vがその外接円内部に位置するか？

        infinite triangle(ghost triangle)に対して節点vがその外接円内部に位置する場合はTrue,
        そうでない場合はFalseを返す。ただし、外接円内部になりやすいように微小許容幅をとることも
        考慮し、delta >= 0 とした微小値を設定することができる。

        Args:
            v (Vertex): 外接円内部判定対象の節点
            delta (float): 外接円内部になりやすいような許容幅. delta>=0

        Returns:
            bool: 節点vが三角形の外接円内部の場合はTrue, そうでない場合はFalse
        """

        vid_inf = [v.infinite for v in self.vertices].index(True)
        edge = self.get_edge((vid_inf + 1) % 3)

        dist = edge.direction().outer_product(v - edge.v1) * 0.5 / edge.length()  # 内が正
        if dist > delta:
            # 内側開集合
            return True
        else:
            ball = edge.diametric_ball()
            return abs(dist) < delta and ball.ispoint_inside(v, delta)

    def is_infinite(self) -> bool:
        return any([vi.infinite for vi in self.vertices])

    # def orientation(self) -> float:
    #     """
    #     節点順がCCWであれば正、CWであれば負
    #     """
    #     mat = np.array([[self.v1.x - self.v3.x, self.v1.y - self.v3.y],
    #                     [self.v2.x - self.v3.x, self.v2.y - self.v3.y]])
    #     return np.linalg.det(mat)

    def fix_orientation(self):
        """節点順がCCWでなかった場合に節点順をCCWに修正"""
        if not(self.is_ccw()):
            self.v2, self.v3 = self.v3, self.v2

    def edge_radius_ratio(self) -> float:
        """半径-エッジ比の計算
        この三角形の半径-エッジ比を計算する。

        Returns:
            float: この三角形の半径-エッジ比

        Notes:
            'Delaunay Mesh Generation' Chapter1 Definition 1.21
            半径をr, 三角形の最短エッジ長さをlminとして半径-エッジ比は,r/lminとして得られ、
            最小が正三角形の場合の1/sqrt(3)=0.577...である。
        """
        rad = self.outer_circle().rad
        edge_length = []
        for i in range(3):
            edge = self.get_edge(i)
            pt0, pt1 = edge.v1.toarray(), edge.v2.toarray()
            edge_length.append(np.linalg.norm(pt1 - pt0))
        return rad / np.min(edge_length)

    def corner_angle(self, idx: int) -> float:
        """
        入力idに基づいた三角形コーナーの角度(rad)を求める。
        Args:
            idx: 0～2のコーナーを示すインデックス

        Returns:
            コーナーの角度(rad)
        """
        v0 = self.vertices[idx]
        v1 = self.vertices[(idx + 1) % 3]
        v2 = self.vertices[(idx + 2) % 3]
        vec1 = np.array([v1.x - v0.x, v1.y - v0.y])
        vec2 = np.array([v2.x - v0.x, v2.y - v0.y])
        cos_sita = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return math.acos(cos_sita)

    def is_seditious(self) -> bool:
        corner_angles = [self.corner_angle(i) for i in range(3)]
        min_idx = np.argmin(corner_angles)
        min_angle = corner_angles[min_idx]

        # 角度が鋭角かどうか。
        if min_angle > math.pi / 3:
            return False

        v0 = self.vertices[min_idx]
        v1 = self.vertices[(min_idx + 1) % 3]
        v2 = self.vertices[(min_idx + 2) % 3]

        segments1 = self.mesh.adjacent_segments(v1)
        segments2 = self.mesh.adjacent_segments(v2)

        # seditious edgeの両端が線分の中間点に位置するか。
        if len(segments1) != 2 or len(segments2) != 2:
            return False

        # 交差線分が等分割され、同じ線分が細分化されたものであるか。
        s1, s2 = segments1
        length_diff = abs(s1.length() - s2.length())
        angle_diff = abs(s1.angle_segment(s2) - math.pi)
        if angle_diff > TOLERANCE:
            return False
        # TODO: 長さについての条件は上手くいかないこともある。
        # if length_diff > TOLERANCE or angle_diff > TOLERANCE:
        #     return False
        s1, s2 = segments2
        length_diff = abs(s1.length() - s2.length())
        angle_diff = abs(s1.angle_segment(s2) - math.pi)
        if angle_diff > TOLERANCE:
            return False
        # TODO: 長さについての条件は上手くいかないこともある。
        # if length_diff > TOLERANCE or angle_diff > TOLERANCE:
        #     return False

        if abs(v0.distance(v1) - v0.distance(v2)) > TOLERANCE:
            return False

        return True


class Polyloop:
    """頂点が時計回りに格納された多角形ループ

    Attributes:
        vertices (List[Vertex]): ループの頂点リスト
        segments (List[Segment]): ループのSegmentリスト

    """
    vertices: List[Vertex]
    segments: List[Segment]

    def __init__(self, vertices: List[Vertex], mesh=None):
        self.vertices = vertices
        self.mesh = mesh

        v_num = len(vertices)
        self.segments = []
        for i in range(v_num):
            self.segments.append(Segment(vertices[i], vertices[(i+1)%v_num], mesh))

    def __contains__(self, item):
        for seg_i in self.segments:
            if item in seg_i:
                return True
        return False

    @property
    def count(self) -> int:
        return len(self.vertices)

    @property
    def area(self) -> float:
        """符号付き面積
        符号付き面積計算の結果を返す。符号はCCWの場合に正、CWの場合に負となる。
        Returns:
            float: 符号付き面積
        """
        sum_cross = 0.0
        for i in range(self.count):
            vi = self.vertices[i]
            vi1 = self.vertices[(i + 1) % self.count]
            sum_cross += vi.outer_product(vi1)
        return sum_cross * 0.5


class Mesh:
    """三角形メッシュ生成・データの格納

    Attributes:
        outerloop (Polyloop): 分割に挿入されたPolyloopのリスト
        vertices (List[Vertex]): 頂点のリスト
        triangles (List[Triangles]): 生成された三角形のリスト
        edge_triangle_table (Dict[Segment,Triangle]): エッジをキーとしてそれに隣接する三角形をデータとした関係テーブル
    """
    edge_triangle_table: Dict[Edge, Facet]
    triangles: List[Facet]
    vertices: List[Vertex]
    outerloop: Polyloop
    innerloops: List[Polyloop]
    segments: List[Segment]
    segments_all: List[Segment]

    def __init__(self, vertices: List[Vertex], outerloop: Polyloop, p: float, innerloops: List[Polyloop] = [],
                 segments: List[Segment] = [], maxiter: int = 500):
        """

        Args:
            vertices:
            outerloop:
            p(float): Facetの外接円半径-最短エッジ比の最大値を指定, sqrt(2)以上でアルゴリズム上終了することが保証されている。
            innerloops:
            segments:
            maxiter:
        """
        # Step1 Initialize dates
        self.edge_triangle_table = {}
        self.triangles = []
        self.vertices = []
        self.outerloop = outerloop
        self.innerloops = innerloops
        self.segments = [seg for seg in segments]
        self.segments_all = [seg for seg in self.outerloop.segments]
        for loop in innerloops:
            self.segments_all.extend(loop.segments)

        self.segments_all.extend(segments)

        for seg_i in self.segments_all:
            seg_i.mesh = self

        # step2 Compute Del S
        self.triangulate(vertices)

        # step3 Resolve segment encroachment
        self.resolve_encroach()
        self.mark_inout()

        # step4 Subdivide triangles
        count_iter = 0
        while tri := self.pick_poor_triangle(p):
            count_iter += 1
            if count_iter > maxiter:
                print("iteration reach max iteration.")
                break
            if tri:
                self.split_triangle2(tri)
                # self.split_triangle(tri)
            else:
                break

    def triangulate(self, vertices: List[Vertex]):
        if len(vertices) < 3:
            print("3つ以上の節点入力が必要")

        gv = Vertex(math.inf, math.inf, infinite=True, source=VertexSource.auto)
        v1 = vertices.pop()
        v2 = vertices.pop()
        # v3 = vertices.pop()

        v3 = None; t1 = None
        for v in vertices:
            v3 = v
            t1 = Facet(v1, v2, v3, mesh=self, force_create=True)
            if abs(t1.area()) > 1.0e-6:
                t1.fix_orientation()
                vertices.remove(v3)
                break

        gt1 = Facet(gv, t1.v2, t1.v1)
        gt2 = Facet(gv, t1.v3, t1.v2)
        gt3 = Facet(gv, t1.v1, t1.v3)

        self.vertices = [gv, v1, v2, v3]
        self.add_triangle(t1)
        self.add_triangle(gt1)
        self.add_triangle(gt2)
        self.add_triangle(gt3)

        for vi in vertices:
            self.add_vertex(vi)

    def pick_encroach(self) -> Segment | None:
        """MeshからencroachしているSegmentを1つ抽出

        Meshに含まれるencroachしている線分と頂点を調査し、
        encroachしているがあれば線分を返し、なければNoneを返す

        Returns:
            Segment|None: Meshにencroachしているedgeが含まれる場合はそれを返し、
            その他の場合はNoneを返す。
        """
        for vi in self.finite_vertices():
            for seg_i in self.segments_all:
                if vi is seg_i.v1 or vi is seg_i.v2:
                    continue
                if s := seg_i.vertex_encroached(vi):
                    return s
        return None

    def resolve_encroach(self):
        while e := self.pick_encroach():
            segments = self.split_segment(e)
            e.children = segments

    def pick_poor_triangle(self, re_rate: float) -> Facet | None:
        """
        指定したradius-edge比を満たしていない不良な三角形を抽出
        Args:
            re_rate: 不良と判断するradius-edge比のしきい値

        Returns:
            不良な三角形、見つからなかった場合はNoneを返す
        """
        for tri_i in self.finite_triangles():
            if tri_i.is_seditious():
                # print("find seditious")
                continue
            if tri_i.locked:
                continue
            if tri_i.locationtype == TriangleLocationType.inside and tri_i.edge_radius_ratio() > re_rate:
                return tri_i
        return None

    def split_triangle(self, tri: Facet):
        c = tri.outer_circle()
        v = Vertex(c.center.x, c.center.y, mesh=self, source=VertexSource.auto)

        # 追加点にencroachが見つかった場合はsegmentを処理
        for seg_i in self.segments_all:
            if s := seg_i.vertex_encroached(v):
                segments = self.split_segment(s)
                s.children = segments
                self.resolve_encroach()
                self.mark_inout()
                return

        # FacetのCircumCenterを挿入
        self.add_vertex(v)
        # self.resolve_encroach() # FIX: vはencroachしていないので不要かと
        self.mark_inout()

    def split_triangle2(self, tri: Facet):
        cluster = tri.outer_circle()
        v = Vertex(cluster.center.x, cluster.center.y, mesh=self, source=VertexSource.auto)

        # 追加点にencroachが見つかった場合はsegmentを処理
        for seg_i in self.segments_all:
            if s := seg_i.vertex_encroached(v):
                # Better Algorithmの判定を追加
                # 1. 両端がsegmentの交点か？
                et = s.encroach_type()
                if et is None:
                    # 中間線分なので分割
                    segments = self.split_segment(s)
                    s.children = segments
                    self.resolve_encroach()
                    self.mark_inout()
                    return
                if et == SegmentPosition.both:
                    # 分割ただしバランスを見た分割
                    segments = self.split_segment(s)
                    s.children = segments
                    self.resolve_encroach()
                    self.mark_inout()
                    return
                # 2. segment clusterを取得
                if et == SegmentPosition.iend:
                    v = s.v1
                elif et == SegmentPosition.jend:
                    v = s.v2
                if cluster := self.segment_cluster(s, v):
                    if len(cluster) == 1:
                        segments = self.split_segment(s)
                        s.children = segments
                        self.resolve_encroach()
                        self.mark_inout()
                        return
                    shells = self.cshell_segments(cluster)
                    if any([i != shells[0] for i in shells]):
                        # clusterの長さが異なる場合は分割
                        segments = self.split_segment(s)
                        s.children = segments
                        self.resolve_encroach()
                        self.mark_inout()
                        return

                angles = []
                # 3. insert radiusとr_minを比較
                for i in range(1, len(cluster)):
                    s1 = cluster[i - 1]
                    s2 = cluster[i]
                    angles.append(s1.angle_segment(s2, v))
                min_angle = min(angles)
                r_min = 2.0 * math.sin(min_angle*0.5) * s.length()
                r_ins = min([e.length() for e in tri.edges])
                if r_min >= r_ins:
                    print("r_min >= r_ins")
                    segments = self.split_segment(s)
                    s.children = segments
                    self.resolve_encroach()
                    self.mark_inout()
                    return

                # 挿入の取りやめ
                tri.locked = True
                return

        # FacetのCircumCenterを挿入
        self.add_vertex(v)
        # self.resolve_encroach() # FIX: vはencroachしていないので不要かと
        self.mark_inout()

    def adjacent_segments(self, v: Vertex) -> List[Segment]:
        """
        節点vに接続される線分を抽出
        Args:
            v: 対象節点

        Returns:
            vに接続される線分のリスト
        """
        segments = []
        for si in self.segments_all:
            if si.v1 == v or si.v2 == v:
                segments.append(si)
        return segments

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
            edge(Edge): 探索起点となる四面体面のFacetDescriptor

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
            for e in edges:
                if e == base_edge:
                    continue
                self.dig_cavity(u, e)
        else:
            self.add_triangle(Facet(u, edge.v1, edge.v2, mesh=self))

    def segment_cluster(self, s: Segment, v: Vertex) -> List[Segment]:
        incident_seg = []
        for si in self.segments:
            for sj in si.flatten_child():
                if sj.v1 == v or sj.v2 == v:
                    incident_seg.append(sj)
        return self.cluster_from_segments(s, v, incident_seg)

    @staticmethod
    # Input Segments, TGTSegment
    def cluster_from_segments(seg, v, segments):
        fixdir = lambda x: math.pi * 2 + x if x < 0.0 else x

        # collect angles
        angles = []
        for si in segments:
            d = si.direction_from_pivot(v)
            angles.append(fixdir(math.atan2(d[1], d[0])))

        # sorted by angle
        result = [i for _, i in sorted(zip(angles, segments))]
        iseg = result.index(seg)
        angles.sort()

        # collect clusters
        cluster = [seg]
        segment_num = len(result)

        delta_angles = [angles[i] - angles[i - 1] for i in range(1, segment_num)]
        delta_angles.append(angles[0] + math.pi * 2 - angles[-1])

        for i in range(segment_num):
            i1 = (i + iseg) % segment_num
            i2 = (i + iseg + 1) % segment_num
            diff = delta_angles[i1]
            if diff < math.pi / 3:
                cluster.append(result[i2])
            else:
                last_ind = i
                break
        else:
            last_ind = segment_num

        for i in range(segment_num - last_ind):
            i1 = (iseg - i)
            i2 = (iseg - i - 1)
            diff = delta_angles[i2]
            if diff < math.pi / 3:
                cluster.insert(0, result[i2])
            else:
                break

        return cluster

    def add_vertex(self, v: Vertex) -> None:
        """頂点vを挿入

        Args:
            v (Vertex): 挿入節点
        """
        tri = self.locate(v)
        self.vertices.append(v)

        edges = [tri.get_edge(i) for i in range(3)]
        self.remove_triangle(tri)
        # print('Search Start Triangle: ', tri)
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

    @staticmethod
    def cshell(l: float) -> int:
        return int(math.floor(math.log(l) / math.log(2.0)))

    @staticmethod
    def cshell_segments(segments: List[Segment]) -> List[int]:
        return [Mesh.cshell(si.length()) for si in segments]

    def split_segment(self, seg: Segment) -> List[Segment]:
        """encroachしている線分を分割

        encroachしている線分を線分同士の接続性や角度関係を考慮して分割

        Args:
            seg (Segment): 分割対象線分

        Returns:
            List[Segment]: 分割された線分

        Notes:

        """
        acute_pos = seg.encroach_type()
        axis = np.array([seg.v2.x - seg.v1.x, seg.v2.y - seg.v1.y])
        axis /= np.linalg.norm(axis)

        # 両端が鋭角の場合の線分分割
        if acute_pos == SegmentPosition.both:
            # i1 = self.cshell(seg.length() / 2)
            # int(math.floor(math.log(seg.length() / 2) / math.log(2.0)))
            dl1 = 2.0 ** self.cshell(seg.length() / 2)
            vec = axis * dl1
            pt1 = Vertex(seg.v1.x + vec[0], seg.v1.y + vec[1], self, source=VertexSource.auto)

            length2 = seg.length() * 4 / 5 - dl1
            # i2 = self.cshell(length2)
            # int(math.floor(math.log(length2) / math.log(2.0)))
            dl2 = 2.0 ** self.cshell(length2)
            vec = -axis * dl2
            pt2 = Vertex(seg.v2.x + vec[0], seg.v2.y + vec[1], self, source=VertexSource.auto)

            self.add_vertex(pt1)
            self.add_vertex(pt2)
            seg1 = Segment(seg.v1, pt1, mesh=self)
            seg2 = Segment(pt1, pt2, mesh=self)
            seg3 = Segment(pt2, seg.v2, mesh=self)
            return [seg1, seg2, seg3]

        # 一端が鋭角の場合の線分分割
        # i = self.cshell(seg.length() / 1.5)
        # int(math.floor(math.log(seg.length() / 1.5) / math.log(2.0)))
        dl = 2.0**self.cshell(seg.length() / 1.5)
        if acute_pos is None:
            mid_pt = seg.midpt
        elif acute_pos == SegmentPosition.iend:
            vec = axis * dl
            mid_pt = Vertex(seg.v1.x + vec[0], seg.v1.y + vec[1], self, source=VertexSource.auto)
        else:  # acute_pos == SegmentPosition.jend
            vec = -axis * dl
            mid_pt = Vertex(seg.v2.x + vec[0], seg.v2.y + vec[1], self, source=VertexSource.auto)

        # その他の場合の線分分割
        self.add_vertex(mid_pt)
        seg1 = Segment(seg.v1, mid_pt, mesh=self)
        seg2 = Segment(mid_pt, seg.v2, mesh=self)
        return [seg1, seg2]

    def finite_triangles(self) -> List[Facet]:
        """包絡三角形を除いた三角形を得る。

        Returns:
            List[Facet]: 包絡三角形を除いた三角形のリスト
        """
        return [tri for tri in self.triangles if not tri.is_infinite()]

    def inside_triangles(self) -> List[Facet]:
        return [tri for tri in self.triangles
                if tri.locationtype == TriangleLocationType.inside]

    def finite_vertices(self) -> List[Vertex]:
        return [vi for vi in self.vertices if not vi.infinite]

    def mark_inout(self):
        """このやり方のmark_inoutならばいろんなループに対応できるかも"""

        # 最初にfinite->inside, infinite->outsideに初期化
        for tri in self.triangles:
            if tri.is_infinite():
                tri.locationtype = TriangleLocationType.outside
            else:
                tri.locationtype = TriangleLocationType.inside

        # outer loopを処理
        sub_segments = []
        for seg in self.outerloop.segments:
            sub_segments.extend(seg.flatten_child())
        for seg in sub_segments:
            self._mark_outside(seg, self.outerloop)

        # inner loopを処理
        for loop in self.innerloops:
            sub_segments = []
            for seg in loop.segments:
                sub_segments.extend(seg.flatten_child())
            for seg in sub_segments:
                self._mark_outside(seg, loop)

    def _mark_outside(self, e: Edge | Segment, loop: Polyloop):
        # 隣接Facetは境界の内側
        e_neigh = e.opposite()
        if e_neigh in loop:
            return

        f_neigh = self.edge_triangle_table[e_neigh]
        # 隣接Facetは調査済み
        if f_neigh.locationtype == TriangleLocationType.outside:
            return

        f_neigh.locationtype = TriangleLocationType.outside
        idx = f_neigh.vertices.index(e_neigh.v2)
        e1 = f_neigh.get_edge(idx)
        e2 = f_neigh.get_edge((idx + 1) % 3)
        self._mark_outside(e1, loop)
        self._mark_outside(e2, loop)

    def incident_faces(self, v: Vertex) -> List[Facet]:
        """節点v周りのFacetをCCWで取得

        Args:
            v(Vertex): 入力節点

        Returns:
            List[Facet]: 節点v周りのFacetのリスト
        """
        # circular search incident facet
        fi = v.incident_facet
        faces = [fi]
        while True:
            edge_idx = fi.vertices.index(v)
            fi = self.edge_triangle_table[fi.get_edge(edge_idx).opposite()]
            if fi == faces[0]:
                break
            faces.append(fi)

        return faces

