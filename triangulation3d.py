from __future__ import annotations

import math
from typing import List, Union, Dict

import numpy as np

TOLERANCE = 1.0e-12


class Vertex3:
    x: float
    y: float
    z: float
    infinite: bool

    def __init__(self, x: float, y: float, z: float, infinite: bool = False):
        self.x = x
        self.y = y
        self.z = z
        self.infinite = infinite

    def __add__(self, other):
        return Vertex3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vertex3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vertex3(-self.x, -self.y, -self.z)

    def __mul__(self, other: Vertex3):
        if isinstance(other, Vertex3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vertex3(self.x * other, self.y * other, self.z * other)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def outer_product(self, other: Vertex3):
        """
        外積
        """
        new_vec = Vertex3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        return new_vec

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def toarray(self):
        return np.array([self.x, self.y, self.z])

    def distance(self, other: Vertex3) -> float:
        return (self - other).length()


class FacetDescriptor:
    """
    Triangulation3dにおいて四面体の各面をCCWに格納する。
    テーブルのキーとして四面体の面と四面体の隣接関係を関連づけるために用いる。
    """
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

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

    def opposite(self) -> FacetDescriptor:
        return FacetDescriptor(self.v3, self.v2, self.v1)

    def is_infinite(self) -> bool:
        return all([self.v1.infinite, self.v2.infinite, self.v3.infinite])


class Tetrahedron:
    v1: Vertex3
    v2: Vertex3
    v3: Vertex3
    v4: Vertex3
    vertices: List[Vertex3]
    facets: List[FacetDescriptor]

    def __init__(self, v1: Vertex3, v2: Vertex3, v3: Vertex3, v4: Vertex3,
                 mesh: Triangulation3):
        # n1: Tetrahedron = None, n2: Tetrahedron = None,
        # n3: Tetrahedron = None, n4: Tetrahedron = None
        self.vertices = [v1, v2, v3, v4]
        self.mesh = mesh
        # self.neighs = [n1, n2, n3, n4]

        f1 = FacetDescriptor(v2, v3, v4)
        f2 = FacetDescriptor(v1, v4, v3)
        f3 = FacetDescriptor(v1, v2, v4)
        f4 = FacetDescriptor(v1, v3, v2)
        self.facets = [f1, f2, f3, f4]
        if self.mesh:
            # Debug: mesh指定なしのTetrahedron生成を有効化
            self.mesh.face_adjacent_table[f1] = self
            self.mesh.face_adjacent_table[f2] = self
            self.mesh.face_adjacent_table[f3] = self
            self.mesh.face_adjacent_table[f4] = self

        if self.orient() < 0.0:
            print("Orientation is Reverse")

    # def __del__(self):
    #     print('Tetrahedron Delete')
    #     if self.mesh:
    #         # Debug: mesh指定なしのTetrahedron生成を有効化


    @property
    def v1(self) -> Vertex3:
        return self.vertices[0]

    @v1.setter
    def v1(self, value: Vertex3):
        self.vertices[0] = value

    @property
    def v2(self) -> Vertex3:
        return self.vertices[1]

    @v2.setter
    def v2(self, value: Vertex3):
        self.vertices[1] = value

    @property
    def v3(self) -> Vertex3:
        return self.vertices[2]

    @v3.setter
    def v3(self, value: Vertex3):
        self.vertices[2] = value

    @property
    def v4(self) -> Vertex3:
        return self.vertices[3]

    @v4.setter
    def v4(self, value: Vertex3):
        self.vertices[3] = value

    @property
    def neighs(self) -> List[Tetrahedron]:
        list_neighs = []
        for fi in self.facets:
            try:
                list_neighs.append(self.mesh.face_adjacent_table[fi])
            except KeyError:
                list_neighs.append(None)
        return list_neighs

    @property
    def n1(self) -> Tetrahedron:
        return self.neighs[0]

    @property
    def n2(self) -> Tetrahedron:
        return self.neighs[1]

    @property
    def n3(self) -> Tetrahedron:
        return self.neighs[2]

    @property
    def n4(self) -> Tetrahedron:
        return self.neighs[3]

    def judge_inside(self, pt: Vertex3):
        """
        pt: Point3d
            内部の判定を行う節点
        """
        triangles = [
            [self.v2, self.v3, self.v4],
            [self.v1, self.v4, self.v3],
            [self.v1, self.v2, self.v4],
            [self.v1, self.v3, self.v2]
        ]

        for tri in triangles:
            judge = sideFromPlane(*tri, pt)
            if not judge:
                return False
        else:
            return True

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
        center = Vertex3(center[0], center[1], center[2])
        rad = np.linalg.norm(self.v1.toarray() - center.toarray())
        return Sphere(center, rad)

    def mirror_index(self, fi: int) -> int:
        """この四面体のfi番目の隣接四面体においてこの四面体は何番目に格納されているか

        Args:
            fi (int): 対象の隣接四面体のインデックス

        Returns:
            int: fi番目の隣接四面体においてこの四面体が何番目に登録されているか

        Notes:
            fi番目の隣接四面体が存在しない場合のチェックを行わないので代入前にチェックが必要
        """
        other = self.neighs[fi]
        return other.neighs.index(self)

    def orient(self) -> float:
        va = self.v1 - self.v2
        vb = self.v1 - self.v3
        vc = self.v1 - self.v4
        return det3x3(va, vb, vc)


class Triangulation3:
    vertices: List[Vertex3]
    tetrahedrons: List[Tetrahedron]

    face_adjacent_table: Dict[FacetDescriptor, Tetrahedron]

    def __init__(self, vertices: List[Vertex3]):
        self.vertices = vertices
        self.face_adjacent_table = {}

        # Create Bounding Box
        init_tet = self.super_tetrahedron()
        self.tetrahedrons = [init_tet]

        for i, vi in enumerate(vertices):
            self.add_vertex(vi)

        self.vertices.extend([init_tet.v1, init_tet.v2, init_tet.v3, init_tet.v4])

    def super_tetrahedron(self, size_fac=2.0) -> Tetrahedron:
        v0 = self.vertices[0]
        xmin = v0.x; xmax = v0.x
        ymin = v0.y; ymax = v0.y
        zmin = v0.z; zmax = v0.z
        for vi in self.vertices[1:]:
            xmin = min(xmin, vi.x)
            xmax = max(xmax, vi.x)
            ymin = min(ymin, vi.y)
            ymax = max(ymax, vi.y)
            zmin = min(zmin, vi.z)
            zmax = max(zmax, vi.z)

        center = Vertex3((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5)
        rad = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2) * size_fac * 0.5

        l = 4.0 * math.sqrt(2/3) * rad
        v1 = Vertex3(center.x, center.y, l * 3 / 4 * math.sqrt(2 / 3) + center.z, infinite=True)
        v2 = Vertex3(-l / 2 + center.x, -l / 2 / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z, infinite=True)
        v3 = Vertex3(l / 2 + center.x, -l / 2 / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z, infinite=True)
        v4 = Vertex3(center.x, l / math.sqrt(3) + center.y, -l / 4 * math.sqrt(2 / 3) + center.z, infinite=True)

        return Tetrahedron(v1, v2, v3, v4, mesh=self)

    def getIncludeTet(self, pt: Vertex3) -> Union[Tetrahedron, None]:
        for tet_i in self.tetrahedrons:
            if tet_i.judge_inside(pt):
                return tet_i
        else:
            return None

    def add_vertex(self, v: Vertex3):
        """
        節点をTriangulation3に追加する。
        Args:
            v(Vertex3): 追加節点
        """
        tet = self.getIncludeTet(v)

        search_facets = tet.facets
        _ = self.face_adjacent_table.pop(tet.facets[0])
        _ = self.face_adjacent_table.pop(tet.facets[1])
        _ = self.face_adjacent_table.pop(tet.facets[2])
        _ = self.face_adjacent_table.pop(tet.facets[3])
        self.tetrahedrons.remove(tet)

        if len(self.tetrahedrons)==0:
            t1 = Tetrahedron(v, tet.v2, tet.v3, tet.v4, self)
            t2 = Tetrahedron(v, tet.v1, tet.v3, tet.v2, self)
            t3 = Tetrahedron(v, tet.v1, tet.v4, tet.v3, self)
            t4 = Tetrahedron(v, tet.v1, tet.v2, tet.v4, self)
            self.tetrahedrons.append(t1)
            self.tetrahedrons.append(t2)
            self.tetrahedrons.append(t3)
            self.tetrahedrons.append(t4)
            return

        for fi in search_facets:
            self.dig_cavity(v, fi)

    def dig_cavity(self, u: Vertex3, f_tgt: FacetDescriptor):
        """
        Triangulation3内で追加節点uを四面体の外接球に含む四面体群を探索する。
        Args:
            u(Vertex3): 追加節点
            f_tgt(FacetDescriptor): 探索起点となる四面体面のFacetDescriptor

        Notes:
            'Delaunay Mesh Generation' Chapter5.2
        """
        # step1
        f_base = f_tgt.opposite()
        try:
            tet = self.face_adjacent_table[f_base]
        except KeyError:
            # step2
            if f_tgt.is_infinite():
                tet_new = Tetrahedron(u, f_tgt.v1, f_tgt.v2, f_tgt.v3, self)
                self.tetrahedrons.append(tet_new)
            return

        if tet.outer_sphere().isinside(u):
            # step3
            search_facets = tet.facets
            _ = self.face_adjacent_table.pop(tet.facets[0])
            _ = self.face_adjacent_table.pop(tet.facets[1])
            _ = self.face_adjacent_table.pop(tet.facets[2])
            _ = self.face_adjacent_table.pop(tet.facets[3])
            self.tetrahedrons.remove(tet)

            for fi in search_facets:
                if fi == f_base:
                    continue
                self.dig_cavity(u, fi)
        else:
            # step4
            tet_new = Tetrahedron(u, f_tgt.v1, f_tgt.v2, f_tgt.v3, self)
            self.tetrahedrons.append(tet_new)


class Sphere:
    center: Vertex3
    radius: float

    def __init__(self, center: Vertex3, radius: float):
        self.center = center
        self.radius = radius

    def isinside(self, v: Vertex3) -> bool:
        d1 = self.center.distance(v)
        return d1 <= self.radius


def det3x3(v1: Vertex3, v2: Vertex3, v3: Vertex3):
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


def sideFromPlane(pt1: Vertex3, pt2: Vertex3, pt3: Vertex3, tgt_pt: Vertex3) -> bool:
    """[summary]

    :param pt1: 平面を定義する節点1
    :param pt2: 平面を定義する節点2
    :param pt3: 平面を定義する節点3
    :param tgt_pt: 平面に対する位置関係を判定する節点

    :return: 入力節点が平面の上側であればTrue, 下側であればFalse
    """
    vec1 = pt2 - pt1
    vec2 = pt3 - pt1
    normal = vec1.outer_product(vec2)

    coef_d = - pt1 * normal
    judge = normal * tgt_pt + coef_d  # 平面方程式の判別式

    return judge > 0.0