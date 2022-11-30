from typing import List
from random import choice

import numpy as np

import triangulation as tr
import triangulation3d as tr3


def plot_triangulation(triangles: List[tr.Triangle]=[], segments=[], triangulate=None):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 6))

    def motion(event):
        try:
            motion.patch_inside.set(facecolor='white')
        except:
            motion.patch_inside = None

        try:
            for i in motion.patch_neighs:
                i.set(facecolor='white')
            motion.patch_neighs = []
        except:
            motion.patch_neighs = []

        x = event.xdata
        y = event.ydata
        tri = triangulate.locate(tr.Vertex(x, y))

        # Located Triangle
        points = [[vi.x, vi.y] for vi in tri.vertices]
        motion.patch_inside = patches.Polygon(
            xy=points, closed=True, facecolor="lightblue", edgecolor='black')
        ax.add_patch(motion.patch_inside)

        # Neigher Triangle
        for i, tri_i in enumerate(tri.neighs):
            if tri_i is None:
                continue
            points = [[vi.x, vi.y] for vi in tri_i.vertices]
            patch = patches.Polygon(
                xy=points, closed=True, facecolor='green', edgecolor='black')
            motion.patch_neighs.append(patch)
            ax.add_patch(patch)

        plt.draw()

    for tri_i in triangles:
        points = [[vi.x, vi.y] for vi in tri_i.vertices]
        patch = patches.Polygon(
            xy=points, closed=True, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(patch)
    
    segs = np.zeros((len(segments), 2, 2))
    for i, si in enumerate(segments):
        segs[i, 0, 0] = si.v1.x
        segs[i, 0, 1] = si.v1.y
        segs[i, 1, 0] = si.v2.x
        segs[i, 1, 1] = si.v2.y
    
    if triangulate:
        for tri_i in triangulate.triangles:
            points = [[vi.x, vi.y] for vi in tri_i.vertices]
            patch = patches.Polygon(
                xy=points, closed=True, facecolor='white', edgecolor='black')
            # patchs.append(patch)
            ax.add_patch(patch)
        plt.connect('motion_notify_event', motion)
        
    line_segments = LineCollection(segs, linewidths=(2), linestyle='solid', edgecolor='red', alpha=0.5)
    ax.add_collection(line_segments)

    ax.autoscale()
    ax.set_aspect('equal')
    plt.show()


def plot_triangles(triangles: List[tr.Triangle] = [], segments=[], circles=[], vertices=[]):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.collections import PatchCollection
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Triangles
    triangle_patches = []
    triangle_evaluations = []
    for tri_i in triangles:
        points = [[vi.x, vi.y] for vi in tri_i.vertices]
        patch = patches.Polygon(
            xy=points, closed=True, facecolor='lightgreen', edgecolor='black')
        triangle_patches.append(patch)
        # triangle_evaluations.append(tri_i.edge_radius_ratio())

    pc = PatchCollection(triangle_patches, facecolor='lightgreen', edgecolor='black')
    colors = 100 * np.random.rand(len(triangle_patches))
    pc.set_array(np.array(colors))
    # pc.set_array(triangle_evaluations)
    ax.add_collection(pc)

    # Plot Segments
    segs = np.zeros((len(segments), 2, 2))
    for i, si in enumerate(segments):
        segs[i, 0, 0] = si.v1.x
        segs[i, 0, 1] = si.v1.y
        segs[i, 1, 0] = si.v2.x
        segs[i, 1, 1] = si.v2.y
    line_segments = LineCollection(segs, linewidths=(
        2), linestyle='solid', edgecolor='red', alpha=0.5)
    ax.add_collection(line_segments)

    # Plot Circle
    for ci in circles:
        patch = patches.Circle((ci.cx, ci.cy), radius=ci.r, fill=False, edgecolor='blue')
        ax.add_patch(patch)

    # Plot Vertices
    x_coords = []
    y_coords = []
    for vi in vertices:
        x_coords.append(vi.x)
        y_coords.append(vi.y)
    ax.scatter(x_coords, y_coords)

    ax.autoscale()
    ax.set_aspect('equal')
    # plt.colorbar(pc)
    plt.show()


def plot_mesh(mesh: tr.Triangulation):
    import pyvista as pv
    # mesh points
    points = []
    for vi in mesh.vertices:
        points.append([vi.x, vi.y, 0.0])
    vertices = np.array(points)

    # mesh faces
    faces = []
    rad_edge_ratios = []

    # for ti in mesh.triangles:
    for ti in mesh.extract_inside():
        if ti.is_infinite():
            continue
        i1 = mesh.vertices.index(ti.v1)
        i2 = mesh.vertices.index(ti.v2)
        i3 = mesh.vertices.index(ti.v3)
        faces.append([3,i1,i2,i3])
        rad_edge_ratios.append(ti.edge_radius_ratio())
    faces = np.hstack(faces)

    surf = pv.PolyData(vertices, faces)
    shrunk = surf.shrink(0.9)
    # plot each face with a different color
    # surf.plot(cpos=[-1, 1, 0.5], show_edges=True, color=True)
    shrunk.plot(cpos=[[0,0,1],[0,0,0],[0,1,0]], scalars=np.array(rad_edge_ratios), show_edges=True, color=True, line_width=2.5, zoom='tight')

def plot_triangulation3(tri3: tr3.Triangulation3):
    import numpy as np
    import pyvista as pv
    # from pyvista import CellType # CellType見つからない

    # pdata = pv.PolyData(tri3.vertices)
    # pdata['orig_sphere'] = np.arange(100)

    # create many spheres from the point cloud
    # sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    # pc = pdata.glyph(scale=False, geom=sphere, orient=False)

    vertices = np.array([[vi.x, vi.y, vi.z] for vi in tri3.vertices])

    cells = []
    cells2 = []
    count_finitecell = 0
    for ti in tri3.tetrahedrons:

        i1 = tri3.vertices.index(ti.v1)
        i2 = tri3.vertices.index(ti.v2)
        i3 = tri3.vertices.index(ti.v3)
        i4 = tri3.vertices.index(ti.v4)

        if ti.is_infinite():
            cells2.append([4, i1, i2, i3, i4])
        else:
            cells.append([4, i1, i2, i3, i4])
            count_finitecell += 1

    if len(cells) >= 1:
        cells = np.hstack(cells)
    else:
        cells = np.array(cells)
    if len(cells2) >= 1:
        cells2 = np.hstack(cells2)
    else:
        cells2 = np.array(cells2)

    # each cell is a HEXAHEDRON
    celltypes = np.empty(count_finitecell, dtype=np.uint8)
    celltypes[:] = 10
    grid1 = pv.UnstructuredGrid(cells, celltypes, vertices)

    # each cell is a HEXAHEDRON
    celltypes = np.empty(len(tri3.tetrahedrons) - count_finitecell, dtype=np.uint8)
    celltypes[:] = 10
    grid2 = pv.UnstructuredGrid(cells2, celltypes, vertices)

    p = pv.Plotter()
    p.set_background("lightgray")
    p.add_mesh(grid1, show_edges=True, color='red')
    p.add_mesh(grid2, show_edges=True, opacity=0.6)

    # exploded = grid.explode()
    _ = p.show()
    # p = pv.Plotter()
    # p.add_mesh(tet)
    # p.add_mesh(pc, cmap='coolwarm')
    # p.enable_shadows()
    # p.show()
    # pc.plot(cmap='Reds')

def triangulation_statics(tess: tr.Triangulation):
    evaluates = []
    for i in tess.triangles:
        if i.is_infinite():
            continue
        evaluates.append(i.edge_radius_ratio())

    evaluates = np.array(evaluates)
    mean = evaluates.mean()
    max = evaluates.max()
    min = evaluates.min()
    std = evaluates.std()
    return {'mean': mean, 'max': max, 'min': min, 'std': std}

def check_neigh_all(tess: tr.Triangulation):
    for tri_i in tess.triangles:
        for ni in tri_i.neighs:
            if ni is None:
                continue

            if not any(tri_i == tri_n  for tri_n in ni.neighs):
                print("Found Neighership Error")
