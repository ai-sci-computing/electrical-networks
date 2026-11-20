"""Opening images: the potential landscape of a triangle of buses and a network of clusters as nodes.

figures/problems_teaser.png      The potential U of the lossless model on a triangle of three buses with unit
                                 susceptances, reference angle fixed, U(t1, t2) = -cos t1 - cos t2 - cos(t1 - t2)
                                 - P1 t1 - P2 t2 (Bergen & Hill 1981, Sec. IV; Mehta et al. 2015, eq. 2), over three
                                 periods of the one angle and two of the other, as a shaded relief with level lines:
                                 a tilted egg crate whose wells (dots) are the stable operating points and whose
                                 passes (crosses) are the saddles. The same potential and injections as the right panel of figures/landscape.png.
figures/aggregation_teaser.png   A planar network of five clusters of buses and the network whose nodes are the
                                 clusters, branches summed in parallel (the rigid aggregate, Chow & Kokotovic 1985,
                                 eq. 4.7), branch width proportional to the summed susceptance.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, box

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
DPI = 200

# =====================================================================================
# 1. The potential landscape of the triangle
# =====================================================================================
P1, P2 = 0.3, -0.3
t1 = np.linspace(-np.pi, 5 * np.pi, 900)                 # three periods of the one angle, two of the other
t2 = np.linspace(-np.pi, 3 * np.pi, 600)
T1, T2 = np.meshgrid(t1, t2)
U = -np.cos(T1) - np.cos(T2) - np.cos(T1 - T2) - P1 * T1 - P2 * T2


def grad(x):
    return np.array([np.sin(x[0]) + np.sin(x[0] - x[1]) - P1, np.sin(x[1]) - np.sin(x[0] - x[1]) - P2])


def hess(x):
    c12 = np.cos(x[0] - x[1])
    return np.array([[np.cos(x[0]) + c12, -c12], [-c12, np.cos(x[1]) + c12]])


# the critical points of one period by Newton's method from a grid of starts, classified by the Hessian's index
crit = []
for a0 in np.linspace(-np.pi, np.pi, 9):
    for b0 in np.linspace(-np.pi, np.pi, 9):
        x = np.array([a0, b0])
        for _ in range(60):
            try:
                x = x - np.linalg.solve(hess(x), grad(x))
            except np.linalg.LinAlgError:
                break
        x = (x + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(grad(x)) < 1e-10 and not any(np.linalg.norm((x - c + np.pi) % (2 * np.pi) - np.pi) < 1e-6 for c in crit):
            crit.append(x)
index = [int(np.sum(np.linalg.eigvalsh(hess(c)) < 0)) for c in crit]
assert sorted(index) == [0, 1, 1, 1, 2, 2]             # one minimum, three saddles, two maxima, as in figures/landscape.png

# the washboard is tilted, so the color shows U with the tilt removed and the relief carries the tilt
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ls = LightSource(azdeg=315, altdeg=40)
periodic = -np.cos(T1) - np.cos(T2) - np.cos(T1 - T2)
rgb = ls.shade_rgb(plt.cm.cividis((periodic - periodic.min()) / np.ptp(periodic))[..., :3], U, vert_exag=0.9, blend_mode="soft",
                   dx=t1[1] - t1[0], dy=t2[1] - t2[0])
ax.imshow(rgb, extent=(t1[0], t1[-1], t2[0], t2[-1]), origin="lower", interpolation="bilinear")
ax.contour(T1, T2, U, levels=28, colors="white", linewidths=0.45, alpha=0.55, linestyles="solid")
for c, k in zip(crit, index):
    for n1 in range(-1, 4):
        for n2 in range(-1, 3):
            x, y = c[0] + 2 * np.pi * n1, c[1] + 2 * np.pi * n2
            if t1[0] < x < t1[-1] and t2[0] < y < t2[-1]:
                if k == 0:
                    ax.scatter(x, y, s=42, color="white", edgecolor="0.1", lw=0.9, zorder=3)
                elif k == 1:
                    ax.scatter(x, y, s=34, marker="x", color="white", lw=1.3, zorder=3)
ax.set_xlim(t1[0], t1[-1]); ax.set_ylim(t2[0], t2[-1])
ax.set_aspect("equal"); ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig(OUT / "problems_teaser.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# =====================================================================================
# 2. Clusters as nodes
# =====================================================================================
rng = np.random.default_rng(7)
centers = np.array([[0.0, 0.0], [2.3, 0.5], [1.1, 2.0], [3.3, 2.3], [-0.6, 2.4]])
pts = np.vstack([c + 0.52 * rng.standard_normal((13, 2)) for c in centers])
# clusters = the cells of the centroids (a few steps of Lloyd's iteration), so that the convex hulls of two clusters
# cannot overlap: each hull lies in its own Voronoi cell, and the cells are convex and disjoint
cent0 = centers.copy()
for _ in range(50):
    label = np.argmin(((pts[:, None, :] - cent0[None, :, :]) ** 2).sum(-1), axis=1)
    cent0 = np.array([pts[label == m].mean(0) for m in range(5)])
assert (label == np.argmin(((pts[:, None, :] - cent0[None, :, :]) ** 2).sum(-1), axis=1)).all()
tri = Delaunay(pts)
edges = {tuple(sorted((s[a], s[b]))) for s in tri.simplices for a, b in ((0, 1), (1, 2), (0, 2))}
edges = [e for e in edges if np.linalg.norm(pts[e[0]] - pts[e[1]]) < 1.15]          # a planar grid without long chords
b = np.array([1.0 / np.linalg.norm(pts[i] - pts[j]) for i, j in edges])             # susceptance as reciprocal length
# the first five slots, in their fixed order, of a categorical palette validated for color-vision deficiencies
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#2b2b2a"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.3))
for m in range(5):
    # the shaded region: the hull with a rounded margin, cut back to the cluster's own cell less a gap
    hull = Polygon(pts[label == m][ConvexHull(pts[label == m]).vertices]).buffer(0.13)
    cell = box(-10, -10, 10, 10)
    for n in range(5):
        if n != m:
            mid, d = (cent0[m] + cent0[n]) / 2, (cent0[n] - cent0[m]) / np.linalg.norm(cent0[n] - cent0[m])
            far, side = mid - 30 * d, 30 * np.array([-d[1], d[0]])
            cell = cell.intersection(Polygon([mid + side, mid - side, far - side, far + side]))
    region = hull.intersection(cell.buffer(-0.035))
    ax.fill(*np.asarray(region.exterior.coords).T, color=COLORS[m], alpha=0.22, lw=0, zorder=0)
for (i, j), w in zip(edges, b):
    inside = label[i] == label[j]
    ax.plot(*pts[[i, j]].T, color=COLORS[label[i]] if inside else INK, lw=0.8 if inside else 1.6, alpha=0.65 if inside else 1.0,
            solid_capstyle="round", zorder=1)
ax.scatter(*pts.T, s=20, c=[COLORS[m] for m in label], edgecolor="white", lw=0.7, zorder=2)

cent = np.array([pts[label == m].mean(0) for m in range(5)])
Wc = np.zeros((5, 5))
for (i, j), w in zip(edges, b):
    if label[i] != label[j]:
        Wc[label[i], label[j]] += w; Wc[label[j], label[i]] += w
for m in range(5):
    for n in range(m + 1, 5):
        if Wc[m, n] > 0:
            bx.plot(*cent[[m, n]].T, color=INK, lw=0.9 * Wc[m, n], solid_capstyle="round", zorder=1)
bx.scatter(*cent.T, s=460, c=COLORS, edgecolor="white", lw=2.0, zorder=2)
for a in (ax, bx):
    a.set_xlim(pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4); a.set_ylim(pts[:, 1].min() - 0.4, pts[:, 1].max() + 0.4)
    a.set_aspect("equal"); a.axis("off")
fig.tight_layout(w_pad=0.5)
fig.savefig(OUT / "aggregation_teaser.png", dpi=DPI, facecolor="white")
plt.close(fig)
print("wrote", OUT / "problems_teaser.png", "and", OUT / "aggregation_teaser.png")
