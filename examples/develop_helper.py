from typing import List
from random import choice

import numpy as np

import triangulation as tr


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


def plot_triangles(triangles: List[tr.Triangle] = [], segments=[], circles=[]):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from matplotlib.collections import PatchCollection
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Triangles
    triangle_patches = []
    for tri_i in triangles:
        points = [[vi.x, vi.y] for vi in tri_i.vertices]
        patch = patches.Polygon(
            xy=points, closed=True, facecolor='lightgreen', edgecolor='black')
        triangle_patches.append(patch)

    pc = PatchCollection(triangle_patches, facecolor='lightgreen', edgecolor='black')
    colors = 100 * np.random.rand(len(triangle_patches))
    pc.set_array(np.array(colors))
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

    ax.autoscale()
    ax.set_aspect('equal')
    plt.show()

def check_neigh_all(tess: tr.Triangulation):
    for tri_i in tess.triangles:
        for ni in tri_i.neighs:
            if ni is None:
                continue

            if not any(tri_i == tri_n  for tri_n in ni.neighs):
                print("Found Neighership Error")
