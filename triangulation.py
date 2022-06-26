from __future__ import annotations

from math import sqrt
from typing import List, Tuple

import numpy as np

TOLERANCE = 1.0e-12


class Segment:
    """2つの節点を結ぶ線分
    Attributes:
        v1 (Vertex): 節点1
        v2 (Vertex): 節点2
    """
    def __init__(self, v1: Vertex, v2: Vertex, mesh: Triangulation = None):
        self.v1 = v1
        self.v2 = v2
        self.mesh = mesh

    @property
    def midpt(self):
        x = (self.v1.x + self.v2.x) * 0.5
        y = (self.v1.y + self.v2.y) * 0.5
        return Vertex(x, y, mesh=self.mesh)

    def ispt_rightside(self, pt) -> bool:
        """2つの節点を通る直線に対して入力点が右側にあるか判定

        Args:
            pt (ndarray[2]): 入力点座標を表す2次元のnumpy配列

        Returns:
            bool: 右側の場合True,左側の場合False
        """
        v10 = pt - self.v1.point
        v12 = self.v2.point - self.v1.point

        return np.cross(v10, v12) > 0.0

    def is_cross(self, other: Segment) -> bool:
        """2つの線分が交差しているかを判定

        Args:
            other (Segment): 比較対象線分

        Returns:
            bool: 交差していればTrue
        """
        # this -> other
        j1 = self.ispt_rightside(other.v1.point) != self.ispt_rightside(other.v2.point)
        # other -> this
        j2 = other.ispt_rightside(self.v1.point) != other.ispt_rightside(self.v2.point)
        return j1 and j2

    def diametric_ball(self) -> Circle:
        """線分を直径とした円を生成

        Returns:
            Circle: 線分を直径とした円
        """
        return Circle.create_from_segment(self)

    def vertex_encroached(self, v: Vertex) -> bool:
        """
        頂点vがこの線分の直径円の中に位置するかどうか判定
        Args:
            v (Vertex): 判定対称の頂点

        Returns:
            (bool) 頂点vがこの線分の直径円の中に位置するか。
        """
        cir = self.diametric_ball()
        if cir.ispoint_inside([v.x, v.y]):
            return True

        return False

    def encroached(self) -> bool:
        """
        線分が三角形分割のいずれかの節点とenchroachの関係にあるかを判定
        具体的には線分を直径とした円の中にいずれかの頂点が含まれるかによる。
        Returns:
            bool: 線分を直径とした円の中に頂点が含まれるか。

        Notes:
            FIXME: encroachの探索が全探索なので隣接関係から探したほうが良さそう。
        """
        for vi in self.mesh.vertices:
            if (vi is self.v1 or vi is self.v2):
                continue
            if self.vertex_encroached(vi):
                return True

        return False

    def split_segment(self) -> List[Segment]:
        seg_que = [self]
        segments = []
        while(seg_que):
            seg = seg_que.pop()
            if seg.encroached():
                midpt = seg.midpt
                seg1 = Segment(seg.v1, midpt, mesh=self.mesh)
                seg2 = Segment(midpt, seg.v2, mesh=self.mesh)
                self.mesh.add_vertex(midpt)
                seg_que.append(seg1)
                seg_que.append(seg2)
            else:
                segments.append(seg)
        return segments

    def length(self) -> float:
        return self.v1.distance_to_vertex(self.v2)

class Circle:
    """中心と半径で定義された円

    Attributes:
        cx, cy (float): それぞれ中心のx座標とy座標
        r (float): 円の半径
    """
    def __init__(self, cx, cy, r) -> None:
        self.cx = cx
        self.cy = cy
        self.r = r

    def ispoint_inside(self, pt) -> bool:
        return (pt[0] - self.cx)**2 + (pt[1] - self.cy)**2 <= self.r**2

    @staticmethod
    def create_from_segment(seg:Segment):
        pt1 = seg.v1
        pt2 = seg.v2
        r = pt1.distance_to_vertex(pt2) / 2
        return Circle((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2, r)


class Vertex:
    """メッシュ頂点クラス

    Attributes:
        x, y (float): 頂点座標
        sample_triangle (Triangle): 隣接関係探索用にこの頂点が属している代表三角形を一つ格納

        point (ndarray[2]): x,y座標で構成された2次元配列
    """
    sample_triangle: Triangle

    def __init__(self, x, y, mesh: Triangulation = None, infinite: bool = False):
        self.x = x
        self.y = y
        self.sample_triangle = None
        self.mesh = mesh
        self.infinite = infinite

    @property
    def point(self):
        return np.array([self.x, self.y])

    def distance_to_vertex(self, other: Vertex) -> float:
        """他の頂点までの距離

        Args:
            other (Vertex): 計算対象の頂点

        Returns:
            float: 距離
        """
        pt1 = self.point
        pt2 = other.point
        return sqrt(np.sum((pt2 - pt1) ** 2))

    def incident_triangles(self) -> List[Triangle]:
        """この頂点の三角形の頂点を時計回りに取得

        Returns:
            List[Triangle]: この頂点の属する三角形のリスト
        Note:
            三角形が時計回りに一周取得できる必要があるので、
            スーパー三角形の頂点等で実行するとエラーが生じる。
        """
        triangles = []
        curr_tri = self.sample_triangle
        while True:
            triangles.append(curr_tri)
            iv = curr_tri.vertices.index(self)
            curr_tri = curr_tri.neighs[(iv+2) % 3]
            if curr_tri == self.sample_triangle:
                break

        return triangles


class Triangle:
    """一つの三角形メッシュ面

    Attributes:
        v1, v2, v3 (Vertex) : メッシュ頂点
        n1, n2, n3 (Triangle): 隣接三角形
        vertices (List[Vertex]): メッシュ頂点の配列[v1, v2, v3]を返すproperty。
        neighs (List[Triangle]): メッシュ隣接要素の配列[n1, n2, n3]を返すproperty。

    """
    def __init__(self, v1, v2, v3, n1=None, n2=None, n3=None):
        self.vertices = [v1, v2, v3]
        self.neighs = [n1, n2, n3]

    @property
    def v1(self):
        return self.vertices[0]

    @v1.setter
    def v1(self, value: Triangle):
        self.vertices[0] = value

    @property
    def v2(self):
        return self.vertices[1]

    @v2.setter
    def v2(self, value: Triangle):
        self.vertices[1] = value

    @property
    def v3(self):
        return self.vertices[2]

    @v3.setter
    def v3(self, value: Triangle):
        self.vertices[2] = value

    @property
    def n1(self):
        return self.neighs[0]

    @n1.setter
    def n1(self, value: Triangle):
        self.neighs[0] = value

    @property
    def n2(self):
        return self.neighs[1]

    @n2.setter
    def n2(self, value: Triangle):
        self.neighs[1] = value

    @property
    def n3(self):
        return self.neighs[2]

    @n3.setter
    def n3(self, value: Triangle):
        self.neighs[2] = value

    def _get_edge_ends(self, id: int):
        """
        0, 1, 2というインデックスを入力して対応する辺の始点と終点を得る。
        Args:
            id: 三角形の各エッジに対応するインデックス(0～2)

        Returns:
            [i番目エッジ始点, i番目エッジ終点]: [Point, Point]
        """
        if id == 0:
            return self.v1.point, self.v2.point
        elif id == 1:
            return self.v2.point, self.v3.point
        else:
            return self.v3.point, self.v1.point

    def ispoint_inside(self, pt) -> bool:
        """入力点が三角形の内部に位置しているかを確認

        Args:
            pt (ndarray[2]): 検証したい節点座標に対応する配列

        Returns:
            bool: 入力点が三角形内部の場合True,外部の場合False
        """
        for i in range(3):
            pt0, pt1 = self._get_edge_ends(i)
            v = pt - pt0
            v01 = pt1 - pt0
            cr = np.cross(v01, v)
            if cr < 0:
                return False
        return True

    def divide_triangle(self, v: Vertex) -> List[Triangle]:
        """三角形を頂点vで分割
        三角形を三角形内部に位置する頂点vで分割する。
        Args:
            v (Vertex): 三角形を分割する頂点

        Returns:
            List[Triangle]: 頂点vを基準に分割された3つの三角形
        """
        t1 = Triangle(v, self.v1, self.v2)
        t2 = Triangle(v, self.v2, self.v3)
        t3 = Triangle(v, self.v3, self.v1)

        t1.n1 = t3
        t1.n2 = self.n1
        t1.n3 = t2
        v.sample_triangle = t1

        t2.n1 = t1
        t2.n2 = self.n2
        t2.n3 = t3

        t3.n1 = t2
        t3.n2 = self.n3
        t3.n3 = t1

        return [t1, t2, t3]

    def split_edge(self, ie: int, v: Vertex) -> List[Triangle]:
        """
        三角形のie番目のエッジ上の節点vによりエッジを分割
        Args:
            ie: 分割するエッジのインデックス
            v: 追加頂点
        Returns:
            分割した4つの三角形
        """
        other = self.neighs[ie]
        ie_opp = self.mirror_index(ie)
        v1 = self.vertices[(ie + 2)%3]
        v2 = self.vertices[ie]
        v3 = other.vertices[(ie_opp + 2)%3]
        v4 = self.vertices[(ie + 1)%3]

        t1 = Triangle(v, v1, v2)
        t2 = Triangle(v, v2, v3)
        t3 = Triangle(v, v3, v4)
        t4 = Triangle(v, v4, v1)

        v.sample_triangle = t1
        v1.sample_triangle = t1
        v2.sample_triangle = t2
        v3.sample_triangle = t3
        v3.sample_triangle = v4

        n1 = self.neighs[(ie + 2)%3]
        n2 = other.neighs[(ie_opp + 1)%3]
        n3 = other.neighs[(ie_opp + 2)%3]
        n4 = self.neighs[(ie + 1)%3]

        t1.neighs = [t4, n1, t2]
        t2.neighs = [t1, n2, t3]
        t3.neighs = [t2, n3, t4]
        t4.neighs = [t3, n4, t1]

        if n1:
            fi = self.mirror_index((ie+2)%3)
            n1.neighs[fi] = t1
        if n2:
            fi = other.mirror_index((ie_opp+1)%3)
            n2.neighs[fi] = t2
        if n3:
            fi = other.mirror_index((ie_opp+2)%3)
            n3.neighs[fi] = t3
        if n4:
            fi = self.mirror_index((ie+1)%3)
            n4.neighs[fi] = t4

        return [t1, t2, t3, t4], other

    def mirror_index(self, fi: int) -> int:
        """この三角形のfi番目の隣接三角形においてこの三角形は何番目に格納されているか

        Args:
            fi (int): 隣接三角形を指定
        Returns:
            int: fi番目隣接三角形においてこの三角形は何番目か。
        """
        other = self.neighs[fi]
        return other.neighs.index(self)

    def is_delaun(self, fi: int) -> bool:
        """隣接三角形とのデローニ特性の充足を確認

        Args:
            fi (int): この三角形の何番目の隣接三角形を指定するか？

        Returns:
            bool: この三角形とfi番目の隣接三角形はdelaunay特性を満たすか
        """
        oppose_id = self.mirror_index(fi)
        v_oppose = self.neighs[fi].vertices[(oppose_id + 2) % 3]

        cir = self.outer_circle()

        return not(cir.ispoint_inside(v_oppose.point))

    def is_infinite(self) -> bool:
         return any([vi.infinite for vi in self.vertices])

    def flip(self, fi: int):
        """fi番目隣接三角形とエッジをFlipする。

        Args:
            fi (int): どの三角形とエッジをフリップするかをインデックスで指定

        Returns:
            [[n1, 1], [n2, 2]]: フリップした結果,連鎖的にデローニ特性を検証する必要がある隣接要素とインデックス
        TODO: Docstring説明不足なので確認
        """
        fi2 = self.mirror_index(fi)
        t_oppos = self.neighs[fi]
        v1 = self.vertices[(fi + 1) % 3]
        v2 = self.vertices[(fi + 2) % 3]
        v3 = self.vertices[fi]
        v4 = t_oppos.vertices[(fi2 + 2) % 3]
        n1 = self.neighs[(fi + 1) % 3]
        n2 = self.neighs[(fi + 2) % 3]
        n3 = t_oppos.neighs[(fi2 + 1) % 3]
        n4 = t_oppos.neighs[(fi2 + 2) % 3]

        # Update neighbor's neighborship
        if self.neighs[(fi + 2) % 3] is not None:
            fn2 = self.mirror_index((fi + 2) % 3)
            self.neighs[(fi + 2) % 3].neighs[fn2] = t_oppos

        if t_oppos.neighs[(fi2 + 2) % 3] is not None:
            fn4 = t_oppos.mirror_index((fi2 + 2) % 3)
            t_oppos.neighs[(fi2 + 2) % 3].neighs[fn4] = self

        # Update Triangles
        self.v1 = v2
        self.v2 = v4
        self.v3 = v1
        self.n1 = t_oppos
        self.n2 = n4
        self.n3 = n1

        t_oppos.v1 = v4
        t_oppos.v2 = v2
        t_oppos.v3 = v3
        t_oppos.n1 = self
        t_oppos.n2 = n2
        t_oppos.n3 = n3

        # Update vertices sampling triangle
        v1.sample_triangle = self
        v3.sample_triangle = t_oppos

        # DEBUG
        if self.area() < TOLERANCE:
            print("self area is zero(flip)", self.area())
        if t_oppos.area() < TOLERANCE:
            print("t_oppos area is zero(flip)", t_oppos.area())

        return [self, 1], [t_oppos, 1], [self, 2], [t_oppos, 2]

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
        r = sqrt((x1 - cx)**2 + (y1 - cy)**2)
        return Circle(cx, cy, r)

    def area(self):
        v12 = np.array([self.v2.x - self.v1.x, self.v2.y - self.v1.y])
        v13 = np.array([self.v3.x - self.v1.x, self.v3.y - self.v1.y])
        return np.cross(v12, v13) * 0.5

    def edge_radius_ratio(self) -> float:
        rad = self.outer_circle().r
        edge_length = []
        for i in range(3):
            pt0, pt1 = self._get_edge_ends(0)
            edge_length.append(np.linalg.norm(pt1 - pt0))
        return np.min(edge_length) / rad


class Polyloop:
    """頂点が時計回りに格納された多角形ループ

    Attributes:
        vertices (List[Vertex]): ループの頂点リスト
        neighs (List[Triangles]): ループ外側に位置する隣接三角形
    """
    def __init__(self, vertices: List[Vertex], neighs=None, mesh=None):
        self.vertices = vertices
        self.mesh = mesh
        if neighs is None:
            self.neighs = [None] * len(vertices)
        else:
            self.neighs = neighs

    @staticmethod
    def from_triangles(triangles: List[Triangle]) -> Polyloop:
        """Create Closed Polyline from triangle strip

        Args:
            triangles (List[Triangle]): strip triangles

        Returns:
            Polyloop: closed polyline
        """
        iv = triangles[0].neighs.index(triangles[1])
        v1 = triangles[0].vertices[(iv + 2) % 3]
        v2 = triangles[0].vertices[iv]
        neighs = [triangles[0].neighs[(iv + 2) % 3]]
        vertices = [v1, v2]
        for i, tri in enumerate(triangles[1:]):
            tri_prev = triangles[i]

            if i+2 >= len(triangles):
                iv = tri.neighs.index(tri_prev)
                vertices.append(tri.vertices[(iv + 2) % 3])
                neighs.append(tri.neighs[(iv + 1) % 3])
                break

            tri_next = triangles[i+2]
            iv = tri.neighs.index(tri_prev)
            if tri.neighs[(iv + 1) % 3] is not tri_next:
                neighs.append(tri.neighs[(iv + 1) % 3])
                vertices.append(tri.vertices[(iv + 2) % 3])
        neighs.append(None)
        return Polyloop(vertices, neighs=neighs)

    def triangulate(self) -> List[Triangle]:
        """このループの内部をデローニ三角形分割

        Returns:
            List[Triangle]: 生成された三角形のリスト。
        """
        vertices_stack = self.vertices[:]  # List Copy
        for vi in vertices_stack:
            vi.sample_triangle = None

        triangles = []
        while len(vertices_stack) > 2:
            for i, vi in enumerate(vertices_stack):
                v1 = vertices_stack[i]
                v2 = vertices_stack[i+1]
                v3 = vertices_stack[i+2]
                seg = Segment(v1, v2)
                if seg.ispt_rightside(v3.point):
                    continue
                else:
                    break

            tri = Triangle(v1, v2, v3)
            if v1.sample_triangle:
                tri.n1 = v1.sample_triangle
                v1.sample_triangle.n3 = tri
            if v2.sample_triangle:
                tri.n2 = v2.sample_triangle
                v2.sample_triangle.n3 = tri
            triangles.append(tri)
            vertices_stack.remove(v2)
            v1.sample_triangle = tri

        face_stack = [[ti, 2] for ti in triangles]
        while face_stack:
            ti, fi = face_stack.pop()
            if ti.neighs[fi] is None:
                continue
            if ti.is_delaun(fi):
                continue
            face_stack.extend(ti.flip(fi))

        # 境界の隣接関係を修復
        for tri in triangles:
            for i, vi in enumerate(tri.vertices):
                if tri.neighs[i] is not None:
                    continue
                id = self.vertices.index(vi)
                ni = self.neighs[id]
                if ni is None:
                    self.vertices[id].sample_triangle = tri
                    continue
                ni.neighs[(ni.vertices.index(vi) + 2) % 3] = tri
                tri.neighs[i] = ni

        return triangles

    def edges(self) -> List[Segment]:
        """PolyloopのすべてのエッジをSegmentのリストとする。

        Returns:
            List[Segment]: PolyloopのそれぞれのエッジをSegmentに変換したリスト。
        """
        vert_num = len(self.vertices)
        segments = []
        for i in range(vert_num):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1) % vert_num]
            segments.append(Segment(v1, v2, self.mesh))
        return segments


class Triangulation:
    """三角形メッシュ生成・データの格納

    Attributes:
        outerloops (List[Polyloop]): 分割に挿入されたPolyloopのリスト
        vertices (List[Vertex]): 頂点のリスト
        triangles (List[Triangles]): 生成された三角形のリスト
    """
    def __init__(self, vertices: List[Vertex], segments: List[Segment]=[], outerloops: List[Polyloop]=[]):
        self.segments = segments
        self.outerloops = outerloops

        super_tri = self.bounding_triangle_from_points(vertices, size_fac=2.0)
        self.triangles = [super_tri]
        super_tri.v1.sample_triangle = super_tri
        super_tri.v1.mesh = self
        super_tri.v2.sample_triangle = super_tri
        super_tri.v2.mesh = self
        super_tri.v3.sample_triangle = super_tri
        super_tri.v3.mesh = self
        super_tri.v1.infinite = True
        super_tri.v2.infinite = True
        super_tri.v3.infinite = True
        self.vertices = [super_tri.v1, super_tri.v2, super_tri.v3]
        for vi in vertices:
            self.add_vertex(vi)
        for si in segments:
            self.insert_edge(si)
        for li in outerloops:
            self.insert_loop(li)

    @staticmethod
    def createMesh(poly: Polyloop, p: float):
        segments = poly.edges()
        # step2
        mesh = Triangulation(poly.vertices)
        mesh.segments = segments

        criteria_no = True
        while(criteria_no):
            # step3
            print("subdivide segment")
            for seg_i in mesh.segments:
                subsegments = mesh.insert_segment(seg_i)
                mesh.segments.remove(seg_i)
                # DEBUG: Segment Length
                for i in segments:
                    if i.length() < TOLERANCE:
                        raise Exception("Segment Length Zero")

                mesh.segments.extend(subsegments)
            print(f'segment count: {len(mesh.segments)}')

            # step4
            # check criteria
            print("check angle criteria")
            subdiv_triangles = []
            for tri in mesh.triangles:
                if tri.is_infinite():
                    continue
                if tri.edge_radius_ratio() > p:
                    subdiv_triangles.append(tri)
            criteria_no = (len(subdiv_triangles) > 0)

            # execute subdivide
            print(f'execute subdivide, {len(subdiv_triangles)}')
            for tri in subdiv_triangles:
            # while(subdiv_triangles):
                tri = subdiv_triangles.pop()
                cir = tri.outer_circle()
                v = Vertex(cir.cx, cir.cy, mesh)

                split_segment = False
                for seg_i in mesh.segments:
                    if seg_i.vertex_encroached(v):
                        # print("v is encroached")
                        subsegments = mesh.split_segment(seg_i)
                        mesh.segments.remove(seg_i)
                        mesh.segments.extend(subsegments)
                        split_segment = True
                        break

                if split_segment:
                    continue

                mesh.add_vertex(v, tri)

        return mesh, None, None

    def locate(self, v: Vertex) -> Triangle:
        """任意の点を含む三角形を探索する。

        Args:
            v (Vertex): 頂点

        Returns:
            Triangle: 入力節点を含む三角形
        """
        for tri in self.triangles:
            if tri.ispoint_inside(v.point):
                return tri
        else:
            raise Exception("No triangles include points.")

    def add_vertex(self, v: Vertex, checktri=None) -> None:
        """頂点vを挿入

        Args:
            v (Vertex): 挿入節点
        """
        self.mesh = self # TODO: 意味をチェック
        tri = self.locate(v)
        self.vertices.append(v)
        div_triangles = tri.divide_triangle(v)
        # 分割方法の分岐
        tri_areas = [ti.area() for ti in div_triangles]
        min_id = np.argmin(tri_areas)
        if tri_areas[min_id] < TOLERANCE:
            # print("DIVIDE EDGE")
            # エッジを分割
            # 隣接三角形の隣接関係を更新
            div_triangles, other = tri.split_edge(min_id, v)

            for ti in div_triangles:
                # DEBUG: Edge Splited Triangles Check
                if ti.area() < TOLERANCE:
                    print("self area is zero(DIVIDE EDGE)", ti.area())
                    raise Exception("Edge Splited Triangles Area Zero")

            self.triangles.remove(other)
        else:
            # print("DIVIDE TRIANGLE")
            # 三角形を分割
            # 隣接三角形の隣接関係を更新
            if tri.n1:
                fi = tri.mirror_index(0)
                tri.n1.neighs[fi] = div_triangles[0]

            if tri.n2:
                fi = tri.mirror_index(1)
                tri.n2.neighs[fi] = div_triangles[1]

            if tri.n3:
                fi = tri.mirror_index(2)
                tri.n3.neighs[fi] = div_triangles[2]
            tri.v1.sample_triangle = div_triangles[0]
            tri.v2.sample_triangle = div_triangles[1]
            tri.v3.sample_triangle = div_triangles[2]

            for ti in div_triangles:
                # DEBUG
                if ti.area() < TOLERANCE:
                    print("self area is zero(DIVIDE TRIANGLE)", ti.area())

        self.triangles.extend(div_triangles)
        self.triangles.remove(tri)

        face_stack = [[ti, 1] for ti in div_triangles]
        while face_stack:
            ti, fi = face_stack.pop()
            # print("Area(add_vertex): ", ti.area())
            if ti.neighs[fi] is None:
                continue
            if ti.is_delaun(fi):
                continue

            face_stack.extend(ti.flip(fi))

    @staticmethod
    def bounding_triangle_from_points(vertices: List[Vertex], size_fac=2.0) -> Triangle:
        """頂点座標からスーパー三角形を生成

        Args:
            vertices (List[Vertex]): 包絡正三角形に考慮する頂点リスト
            size_fac (float, optional): 包絡正三角形のバウンディングボックスに対する大きさ Defaults to 2.0.

        Returns:
            Triangle: _description_
        """
        px = [pti.x for pti in vertices]
        py = [pti.y for pti in vertices]

        x_min = np.min(px); x_max = np.max(px)
        x_width = x_max - x_min
        y_min = np.min(py); y_max = np.max(py)
        y_width = y_max - y_min

        wrap_length = np.max([x_width, y_width])
        cx = x_min + 0.5 * x_width; cy = y_min + 0.5 * y_width

        baselength = wrap_length * size_fac
        dx = baselength * (1/sqrt(3.0) + 0.5)
        dy = baselength * (sqrt(3.0)*0.5 + 0.5)

        v1 = Vertex(cx - dx, cy - baselength * 0.5)
        v2 = Vertex(cx + dx, cy - baselength * 0.5)
        v3 = Vertex(cx, cy + dy)

        return Triangle(v1, v2, v3)

    def extract_incident_triangles(self, seg: Segment) -> List[Triangle]:
        """既存三角形分割からエッジを挿入した時に交差する三角形を抽出

        Args:
            seg (Segment): 検証する線分

        Raises:
            Exception: 交差する三角形のエッジが見つからなかったとき
                       (すでに三角形のエッジとして検証対象の線分が存在している可能性あり)

        Returns:
            List[Triangle]: 交差する三角形のリスト
        """
        incident_triangles = []
        for tri in seg.v1.incident_triangles():
            iv = tri.vertices.index(seg.v1)
            testseg = Segment(tri.vertices[(iv+1)%3], tri.vertices[(iv+2)%3])
            if testseg.is_cross(seg):
                prev_tri = tri
                test_tri = tri.neighs[(iv+1)%3]
                incident_triangles = [prev_tri, test_tri]

        # 頂点が線分と共通でないかチェック
        # TODO: ここでtest_triがassignされていないエラーがたまに起きる
        while (seg.v2 not in test_tri.vertices):
            # 次の三角形を探す
            ip = test_tri.neighs.index(prev_tri)
            for i in range(2):
                v1 = test_tri.vertices[(ip+i+1)%3]
                v2 = test_tri.vertices[(ip+i+2) % 3]
                testseg = Segment(v1, v2)
                if (testseg.is_cross(seg)):
                    prev_tri, test_tri = test_tri, test_tri.neighs[(
                        ip+i+1) % 3]
                    break
            else:
                raise Exception("Not Found Cross Triangle Edge")

            incident_triangles.append(test_tri)

        return incident_triangles

    def refinement(self, p: float = 1.5):
        import copy
        que = copy.copy(self.triangles)
        while(que):
            tri = que.pop()
            if tri.is_infinite():
                continue
            if tri.edge_radius_ratio() > p:
                cir = tri.outer_circle()
                v = Vertex(cir.cx, cir.cy, self)
                self.add_vertex(v)

    def insert_segment(self, seg: Segment) -> List[Segment]:
        seg.mesh = self
        seg_que = [seg]
        segments = []
        while(seg_que):
            seg_i = seg_que.pop()
            if seg_i.encroached():
                print("segment splited (insert_segment)")
                mid_pt = seg_i.midpt
                self.add_vertex(mid_pt)

                seg_que.append(Segment(seg_i.v1, mid_pt, mesh=self))
                seg_que.append(Segment(mid_pt, seg_i.v2, mesh=self))
            else:
                segments.append(seg_i)
        return segments

    def split_segment(self, seg: Segment) -> List[Segment]:
        mid_pt = seg.midpt
        self.add_vertex(mid_pt)
        seg1 = Segment(seg.v1, mid_pt, mesh=self)
        seg2 = Segment(mid_pt, seg.v2, mesh=self)
        return [seg1, seg2]

    def insert_edge(self, seg: Segment):
        """segmentを三角形分割に挿入

        Args:
            seg (Segment): Triangulationに挿入する線分
        """
        seg.mesh = self
        triangles = self.extract_incident_triangles(seg)
        loop1 = Polyloop.from_triangles(triangles)
        loop2 = Polyloop.from_triangles(triangles[::-1])
        for ti in triangles:
            self.triangles.remove(ti)
        triangles1 = loop1.triangulate()
        for ti in triangles1:
            self.triangles.append(ti)
        triangles2 = loop2.triangulate()
        for ti in triangles2:
            self.triangles.append(ti)

        # FIXME: かなり強引なのでなにかうまい対処法を見つける。
        # Loop1の境界三角形を見つけて処理
        for tri in triangles1:
            found_boundary = False
            for i in range(3):
                v1 = tri.vertices[i]
                v2 = tri.vertices[(i + 1)%3]
                if (seg.v2 == v1 and seg.v1 == v2):
                    tri1 = tri
                    ni1 = i
                    found_boundary = True
                    break
            if found_boundary:
                break
        # Loop2の境界三角形を見つけて処理
        for tri in triangles2:
            found_boundary = False
            for i in range(3):
                v1 = tri.vertices[i]
                v2 = tri.vertices[(i + 1)%3]
                if (seg.v1 == v1 and seg.v2 == v2):
                    tri2 = tri
                    ni2 = i
                    found_boundary = True
                    break
            if found_boundary:
                break
        tri1.neighs[ni1] = tri2
        tri2.neighs[ni2] = tri1

        # Experimental 
        seg.right_triangle = tri1
        seg.left_triangle = tri2

    def insert_loop(self, loop: Polyloop):
        """loopを三角形分割に挿入

        Args:
            loop (Polyloop): 挿入するPolyloop
        """
        loop.mesh = self
        for i, ei in enumerate(loop.edges()):
            self.insert_edge(ei)
            loop.neighs[i] = ei.right_triangle

    def extract_loopinside(self, loop: Polyloop) -> List[Triangle]:
        """loopの内側の三角形を抽出してリストを返す。

        Args:
            loop (Polyloop): 抽出する基準となるPolyloop

        Returns:
            List[Triangle]: 抽出された三角形のリスト
        """
        self.triangles, triangles = self.triangles[:], self.triangles
        remove_triangles = self.extract_loopoutside(loop)
        for ti in remove_triangles:
            triangles.remove(ti)
        return triangles

    def extract_loopoutside(self, loop: Polyloop) -> List[Triangle]:
        """loopの外側の三角形を抽出してリストを返す。

        Args:
            loop (Polyloop): 抽出する基準となるPolyloop

        Returns:
            List[Triangle]: 抽出された三角形のリスト
        """
        collect_triangles = [loop.neighs[0]]
        check_stack = [loop.neighs[0]]

        while(check_stack):
            tgt = check_stack.pop()
            for i, ti in enumerate(tgt.neighs):
                if ti is None:
                    continue
                if ti in collect_triangles:
                    continue
                v1 = tgt.vertices[i]
                v2 = tgt.vertices[(i+1) % 3]
                if v1 in loop.vertices and v2 in loop.vertices:
                    continue
                collect_triangles.append(ti)
                check_stack.append(ti)
        return collect_triangles

    def finite_triangles(self) -> List[Triangle]:
        """包絡三角形を除いた三角形を得る。

        Returns:
            List[Triangle]: 包絡三角形を除いた三角形のリスト
        """
        triangles = []
        for tri in self.triangles:
            if any([self.vertices[0] in tri.vertices,
                    self.vertices[1] in tri.vertices,
                    self.vertices[2] in tri.vertices]):
                continue
            triangles.append(tri)

        return triangles
