# Delaunay Triangulation 2D
制約として線分やループを挿入できる2次元のデローニ三角形分割を行う。

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