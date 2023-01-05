from typing import List
from random import choice

import numpy as np
import pyvista

import mesh2
import triangulation2 as tr2
import triangulation3 as tr3
import mesh3


def plot_triangulation(triangles: List[tr2.Facet]=[], segments=[], triangulate=None):
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
        tri = triangulate.locate(tr2.Vertex(x, y))

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

    # Plot Triangle
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


def plot_triangulation2d(tr: tr2.Triangulation):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Triangle
    for tri_i in tr.finite_triangles():
        points = [[vi.x, vi.y] for vi in tri_i.vertices]
        patch = patches.Polygon(
            xy=points, closed=True, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(patch)

    # Plot Voronoi
    for vi in tr.vertices:
        triangles = tr.incident_faces(vi)
        if any(tri.is_infinite() for tri in triangles):
            continue

        vertices = [tri.outer_circle().center for tri in triangles]
        points = [[vi.x, vi.y] for vi in vertices]
        patch = patches.Polygon(
            xy=points, closed=True, edgecolor='blue', fill=False)
        ax.add_patch(patch)

    # segs = np.zeros((len(tr. segments), 2, 2))
    # for i, si in enumerate(segments):
    #     segs[i, 0, 0] = si.v1.x
    #     segs[i, 0, 1] = si.v1.y
    #     segs[i, 1, 0] = si.v2.x
    #     segs[i, 1, 1] = si.v2.y
    #
    # if triangulate:
    #     for tri_i in triangulate.triangles:
    #         points = [[vi.x, vi.y] for vi in tri_i.vertices]
    #         patch = patches.Polygon(
    #             xy=points, closed=True, facecolor='white', edgecolor='black')
    #         # patchs.append(patch)
    #         ax.add_patch(patch)
    #     plt.connect('motion_notify_event', motion)

    # line_segments = LineCollection(segs, linewidths=(2), linestyle='solid', edgecolor='red', alpha=0.5)
    # ax.add_collection(line_segments)

    ax.autoscale()
    ax.set_aspect('equal')
    plt.show()


def plot_triangles(triangles: List[tr2.Facet] = [], segments=[], circles=[], vertices=[]):
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


def plot_mesh2(mesh: mesh2.Mesh):
    import pyvista as pv

    plotter = pv.Plotter()

    # Plot Vertices
    fixed_points = []
    auto_points = []
    mesh_vertices = mesh.finite_vertices()
    for vi in mesh_vertices:
        if vi.source == mesh2.VertexSource.fixed:
            fixed_points.append([vi.x, vi.y, 0.0])
        else:
            auto_points.append([vi.x, vi.y, 0.0])
    if fixed_points:
        plotter.add_mesh(pv.PointSet(np.array(fixed_points)), color='red', label='Fixed Point')
    if auto_points:
        plotter.add_mesh(pv.PointSet(np.array(auto_points)), color='blue', label='Auto Point')

    # Plot Meshes
    points = []
    for vi in mesh_vertices:
        points.append([vi.x, vi.y, 0.0])
    vertices = np.array(points)
    # mesh faces
    faces = []
    rad_edge_ratios = []
    for ti in mesh.inside_triangles():
        i1 = mesh_vertices.index(ti.v1)
        i2 = mesh_vertices.index(ti.v2)
        i3 = mesh_vertices.index(ti.v3)
        faces.append([3, i1, i2, i3])
        rad_edge_ratios.append(ti.edge_radius_ratio())
    faces = np.hstack(faces)
    surf = pv.PolyData(vertices, faces)
    shrunk = surf.shrink(0.95)
    sargs = dict(
        title_font_size=20, label_font_size=16)
    plotter.add_mesh(shrunk, scalars=np.array(rad_edge_ratios),
                     scalar_bar_args=sargs,
                     show_edges=True, color=True, line_width=2.5, label="Domain")

    # Plot Segments
    for seg in mesh.segments:
        outer_points = [[vi.x, vi.y, 0.0] for vi in [seg.v1, seg.v2]]
        outer_points = np.array(outer_points)
        line = pv.lines_from_points(outer_points)
        plotter.add_mesh(line, color='lime', line_width=4, label="Segment")

    # Plot OuterLoop
    outer_points = [[vi.x, vi.y, 0.0] for vi in mesh.outerloop.vertices]
    outer_points = np.array(outer_points)
    lines = pv.lines_from_points(outer_points, close=True)
    plotter.add_mesh(lines, color='cyan', line_width=4, label="OuterLoop")

    # Plot Innerloops
    for loop in mesh.innerloops:
        loop_points = [[vi.x, vi.y, 0.0] for vi in loop.vertices]
        loop_points = np.array(loop_points)
        lines = pv.lines_from_points(loop_points, close=True)
        plotter.add_mesh(lines, color='magenta', line_width=4, label="InnerLoop")

    # Plot Show
    plotter.add_legend(size=(0.1,0.1))
    plotter.show()


def plot_triangulation_plane(mesh: mesh3.TriangulationPlane):
    import pyvista as pv
    # mesh points
    points = []
    for vi in mesh.vertices:
        points.append([vi.x, vi.y, vi.z])
    vertices = np.array(points)

    # mesh faces
    faces = []
    rad_edge_ratios = []

    # for ti in mesh.triangles:
    for ti in mesh.finite_triangles():
        if ti.is_infinite():
            continue
        i1 = mesh.vertices.index(ti.v1)
        i2 = mesh.vertices.index(ti.v2)
        i3 = mesh.vertices.index(ti.v3)
        faces.append([3,i1,i2,i3])
        # rad_edge_ratios.append(ti.edge_radius_ratio())
    faces = np.hstack(faces)
    p = pv.Plotter()

    # Plot Vertex
    vertex_coords = []
    for vi in mesh.vertices:
        if vi.infinite: continue
        vertex_coords.append(vi.toarray())
    p.add_mesh(pyvista.PointSet(vertex_coords))

    # p.set_background("lightgray")
    # p.add_mesh(grid1, show_edges=True, color='red')
    # p.add_mesh(grid2, show_edges=True, opacity=0.6)

    # for tri in mesh.triangles:
    #     if tri.is_infinite():
    #         for i in range(3):
    #             edge = tri.get_edge(i)
    #             if edge.is_infinite():
    #                 continue
    #             vec = (edge.v2 - edge.v1).toarray()
    #             arrow = pv.Arrow(start=edge.v1.toarray(), direction=vec, scale='auto')
    #             p.add_mesh(arrow)
                # print("arrow added")

    surf = pv.PolyData(vertices, faces)
    shrunk = surf.shrink(0.98)
    # plot each face with a different color
    # surf.plot(cpos=[-1, 1, 0.5], show_edges=True, color=True)
    # shrunk.plot(cpos=[[0,0,1],[0,0,0],[0,1,0]], show_edges=True, color=True, line_width=2.5, zoom='tight')
    # shrunk.plot(cpos=[[0,0,1],[0,0,0],[0,1,0]], scalars=np.array(rad_edge_ratios), show_edges=True, color=True, line_width=2.5, zoom='tight')
    p.add_mesh(shrunk, show_edges=True, color=True, line_width=2.5)
    # p.add_mesh(surf_cavity, show_edges=True, color=True, line_width=2.5)
    # p.add_mesh(surf_cavity, show_edges=True, color='red', line_width=2.5)
    _ = p.show()
    # _ = p.show(cpos=[[0,0,1],[0,0,0],[0,1,0]], zoom='tight')

def plot_triangulation3(tri3: tr3.Triangulation3):
    import numpy as np
    import pyvista as pv
    # from pyvista import CellType # CellType見つからない

    # pdata = pv.PolyData(tri3.vertices)
    # pdata['orig_sphere'] = np.arange(100)

    # create many spheres from the point cloud
    # sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    # pc = pdata.glyph(scale=False, geom=sphere, orient=False)

    vertices = tri3.finite_vertices()
    points = np.array([[vi.x, vi.y, vi.z] for vi in vertices])
    cells = []
    # cells2 = []
    rad_edge_ratios = []
    count_finitecell = 0
    for ti in tri3.finite_tetrahedrons():

        i1 = vertices.index(ti.v1)
        i2 = vertices.index(ti.v2)
        i3 = vertices.index(ti.v3)
        i4 = vertices.index(ti.v4)
        cells.append([4, i1, i2, i3, i4])
        rad_edge_ratios.append(ti.edge_radius_ratio())
        count_finitecell += 1

    cells = np.hstack(cells)
    # if len(cells) >= 1:
    #     pass
    # else:
    #     cells = np.array(cells)
    # if len(cells2) >= 1:
    #     cells2 = np.hstack(cells2)
    # else:
    #     cells2 = np.array(cells2)

    # each cell is a HEXAHEDRON
    celltypes = np.empty(count_finitecell, dtype=np.uint8)
    celltypes[:] = 10
    grid1 = pv.UnstructuredGrid(cells, celltypes, points)

    # each cell is a HEXAHEDRON
    # celltypes = np.empty(len(tri3.tetrahedrons) - count_finitecell, dtype=np.uint8)
    # celltypes[:] = 10
    # grid2 = pv.UnstructuredGrid(cells2, celltypes, points)

    p = pv.Plotter()
    p.set_background("lightgray")
    p.add_mesh(grid1, scalars=np.array(rad_edge_ratios), show_edges=True, color='red')
    # p.add_mesh(grid2, show_edges=True, opacity=0.6)

    # Add Plane Widget
    # def hello(normal, origin):
    #     print("normal, origin: ", normal, origin)
    #
    # p.add_plane_widget(hello, origin=(0,0,0), bounds=(-10,10,-10,10,-10,10))
    _ = p.show()

    # p = pv.Plotter()
    # p.add_mesh(tet)
    # p.add_mesh(pc, cmap='coolwarm')
    # p.enable_shadows()
    # p.show()
    # pc.plot(cmap='Reds')

def triangulation_statics(tess: tr2.Triangulation):
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

def check_neigh_all(tess: tr2.Triangulation):
    for tri_i in tess.triangles:
        for ni in tri_i.neighs:
            if ni is None:
                continue

            if not any(tri_i == tri_n  for tri_n in ni.neighs):
                print("Found Neighership Error")
