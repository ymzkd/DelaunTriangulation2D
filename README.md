# Delaunay Triangulation and Mesh
- 2次元の点群のデローニ分割
- 2次元の線要素によって定義された領域の三角形メッシュ分割
- 3次元の点群のデローニ分割
- 3次元のPLCsによって定義された領域の四面体メッシュ分割

## 使い方
###  Install
```
pip install .
```

### Uninstall
```
pip uninstall triangulation
```

## TODO
### 改良事項
- 検索スピードが遅いのでどうにかする。(今は全探索)
- requirement.txtを開発時と本番用を用意したがsetupで適宜に読めるようにしておく。
- encroachの探索、Meshの領域内外判定等が効率が悪いように思う。

### 修正事項
- triangulation2dにおいてVertexのメンバーにTriangulation自身が含まれるが、実際は上手く設定できていないので機能していなさそう。要調査・修正

## Reference
1. [Mesh generation for domains with small angles, Shewchuk et.al.](https://dl.acm.org/doi/10.1145/336154.336163)
2. [Delaunay Mesh Generation, Shewchuk, Cheng et.al.](https://dl.acm.org/doi/10.5555/2422925)