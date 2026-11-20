"""Illustrations of clusters as nodes.

figures/three_reductions.png   Two triangles with branch susceptance w = 1 joined by a bridge of susceptance 1,
                               each triangle a cluster, reduced three ways: point masses on the two buses away
                               from the bridge (the Kron reduction, Dorfler & Bullo 2013, eq. 2.1), uniform
                               distributions (the complete aggregate, Chow & Kokotovic 1985, eqs. 4.3, 4.4),
                               shorted clusters (the rigid aggregate, eq. 4.7; the effective resistance between
                               two sets, Lyons & Peres 2016, Exercise 2.5). Closed forms 1 + 4/(3w), 1 + 4/(9w), 1,
                               derived in tests/test_reweight.py.
figures/injection_rules.png    The same two triangles with the buses 0 and 5 kept. Left, the columns of the
                               accompanying matrix D = -L_ab L_bb^{-1} (Dorfler & Bullo 2013, Lemma 2.1(3)), the
                               distribution factors of Ward 1949, equal to the probabilities that the random walk
                               from a bus reaches bus 0 before bus 5 (Doyle & Snell 2000, Sec. 1.3.2): 6/7, 5/7,
                               2/7, 1/7, closed form in tests/test_kron.py. Right, the sum over each cluster.
figures/sweep_cut.png          Two groups of six buses joined by two branches, with a thirteenth bus hanging on one
                               branch, all susceptances one. Left, the Fiedler vector of the normalized problem
                               L v = lambda D v and the sweep cut (von Luxburg 2007, Secs. 3, 5; Bandeira, Singer &
                               Spielman 2013, Thm. 1.1). Right, the conductance of every side of the sweep with the
                               two bounds of Cheeger's inequality. The smallest cut, 1, severs the hanging bus and
                               has conductance 1; the sweep cut has cut 2 and conductance 2/22 = 1/11.
figures/reductions_converge.png  The zoned network of tests/test_reweight.py (five clusters of four buses, each a
                               complete graph with susceptance `internal`, six interconnections of two lines):
                               the distances between all ten pairs of clusters in the Kron reduction onto one
                               bus per cluster and in the rigid aggregate, relative to the complete aggregate
                               with uniform distributions, against the internal susceptance. The three agree in
                               the limit of weakly connected areas (Chow & Kokotovic 1985, Sec. II).
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import accompanying_matrix, kron_reduce
from gridnet.reweight import zonal_operator
from gridnet.spectral import conductance, cut_weight, fiedler, sweep_cut

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
(HERE / "results").mkdir(exist_ok=True)
_log = open(HERE / "results" / "aggregation_figure.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/aggregation_figure.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()
DPI = 200
BLUE, ORANGE, GREEN = "tab:blue", "tab:orange", "tab:green"


def resistance(Lr, a, b):
    """Effective resistance between nodes a and b of a symmetric operator with the constants as kernel."""
    return effective_resistance(Lr)[a, b]


# =====================================================================================
# 1. Two triangles, three reductions
# =====================================================================================
W = 1.0
EDGES = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
L = laplacian(6, EDGES, [W] * 6 + [1.0])
zones = np.array([0, 0, 0, 1, 1, 1])
U = np.eye(2)[zones]
d_kron = resistance(kron_reduce(L, [0, 5]), 0, 1)
d_complete = resistance(zonal_operator(L, zones), 0, 1)
d_rigid = resistance(U.T @ L @ U, 0, 1)
assert np.allclose([d_kron, d_complete, d_rigid], [1 + 4 / (3 * W), 1 + 4 / (9 * W), 1.0])

h = np.sqrt(3) / 2
XY = np.array([[0, 0], [0.5, h], [1, 0], [2, 0], [2.5, h], [3, 0]]) - np.array([0, h / 3])
XY[[0, 1, 2]] = np.array([[0, h], [0, -h], [h * 1.3, 0]]) * 0.62           # left triangle, apex toward the bridge
XY[[3, 4, 5]] = np.array([[-h * 1.3, 0], [0, h], [0, -h]]) * 0.62 + [2.6, 0]
XY[[0, 1, 2]] += [0.0, 0]
XY[:, 0] += 0.2
# the buses away from the bridge that carry the point masses are 0 and 5 (any of 0, 1 and of 4, 5 by symmetry)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
panels = [
    ("point masses\nthe Kron reduction", d_kron, BLUE),
    ("uniform distributions\nthe complete aggregate", d_complete, GREEN),
    ("shorted clusters\nthe rigid aggregate", d_rigid, ORANGE),
]
for k, (ax, (title, d, color)) in enumerate(zip(axes, panels)):
    if k == 2:                                           # shorted: the cluster is one node, drawn as a filled patch
        for tri in ([0, 1, 2], [3, 4, 5]):
            ax.fill(*XY[tri].T, color=color, alpha=0.35, lw=0)
    for (i, j) in EDGES:
        bridge = (i, j) == (2, 3)
        ax.plot(*XY[[i, j]].T, color="0.25" if bridge or k < 2 else color, lw=2.2 if bridge else 1.4, zorder=1)
    # where the unit enters and leaves; with the clusters shorted the least-energy pattern is the whole unit
    # at the two ends of the bridge (Thomson's principle), whose resistance 1 is the rigid distance
    share = {0: [1, 0, 0, 0, 0, 1], 1: [1 / 3] * 6, 2: [0, 0, 1, 1, 0, 0]}[k]
    # a filled disk has an area proportional to the fraction it carries (the whole unit 220 square points);
    # a bus that carries nothing is a small open circle, whose size means nothing
    for n, (x, y) in enumerate(XY):
        ax.scatter(x, y, s=220 * share[n] if share[n] > 0 else 30, color=color if share[n] > 0 else "white",
                   edgecolor="0.25", lw=1.0, zorder=2)
    ax.set_title(title, fontsize=8.5)
    ax.text(XY[:, 0].mean(), -0.95, f"distance {d:.2f}", ha="center", fontsize=10, color=color, weight="bold")
    ax.set_xlim(XY[:, 0].min() - 0.35, XY[:, 0].max() + 0.35); ax.set_ylim(-1.05, 0.8)
    ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "three_reductions.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# =====================================================================================
# 1b. Where the injections go: Ward's distribution factors against the sum over each cluster
# =====================================================================================
KEEP = [0, 5]
D = accompanying_matrix(L, KEEP)                          # 2 x 6, the identity on the kept buses included
assert np.allclose(D[0], [1, 6 / 7, 5 / 7, 2 / 7, 1 / 7, 0]) and np.allclose(D.sum(0), 1)
T = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=float)      # every bus wholly to its own cluster's node
FRACTION = {1.0: "", 6 / 7: "6/7", 5 / 7: "5/7", 2 / 7: "2/7", 1 / 7: "1/7", 0.0: ""}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, rule, title in zip(axes, (D, T), ("Ward's distribution factors", "the sum over each cluster")):
    for (i, j) in EDGES:
        ax.plot(*XY[[i, j]].T, color="0.25", lw=2.2 if (i, j) == (2, 3) else 1.4, zorder=1)
    for n, (x, y) in enumerate(XY):
        kept = n in KEEP and rule is D                    # buses are kept only in the Kron reduction
        r = 0.2 if kept else 0.15
        f = rule[0, n]                                    # the fraction of bus n's injection that goes to bus 0
        ax.add_patch(Circle((x, y), r, color=ORANGE, zorder=2, lw=0))
        if f > 0:
            ax.add_patch(Wedge((x, y), r, 90, 90 + 360 * f, color=BLUE, zorder=3, lw=0))
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="0.15", lw=2.4 if kept else 1.0, zorder=4))
        label = next((t for v, t in FRACTION.items() if abs(v - f) < 1e-9), f"{f:.2f}")
        if label:
            ax.text(x, y + (0.3 if y >= 0 else -0.3), label, ha="center", va="center", fontsize=8.5, color=BLUE)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(XY[:, 0].min() - 0.45, XY[:, 0].max() + 0.45); ax.set_ylim(-0.95, 0.95)
    ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "injection_rules.png", dpi=DPI)
plt.close(fig)


# =====================================================================================
# 1c. Cut, conductance and the sweep cut
# =====================================================================================
def ring(o):
    return [(o + i, o + (i + 1) % 6) for i in range(6)]


CE = ring(0) + [(0, 2), (0, 3), (1, 4), (3, 5)] + ring(6) + [(6, 8), (6, 9), (7, 10), (9, 11)] + [(2, 11), (3, 10)] + [(0, 12)]
LC = laplacian(13, CE, [1.0] * len(CE))
lam2, v = fiedler(LC, normalized=True)
v = v if v[0] < 0 else -v
side, h_sweep = sweep_cut(LC, v)
order = np.argsort(v)
sweep = []
for k in range(1, 13):
    m = np.zeros(13, bool); m[order[:k]] = True
    sweep.append((cut_weight(LC, m), conductance(LC, m)))
sweep = np.array(sweep)
assert np.isclose(h_sweep, 1 / 11) and np.isclose(cut_weight(LC, side), 2.0)
assert np.isclose(sweep[0, 0], 1.0) and np.isclose(sweep[0, 1], 1.0)          # the hanging bus: smallest cut, conductance 1
assert 0.5 * lam2 <= h_sweep <= np.sqrt(2 * lam2)

ang = np.deg2rad(-30 + np.arange(6) * 60)                # buses 2 and 3 face the other group
hexagon = np.c_[np.cos(ang), np.sin(ang)]
CXY = np.zeros((13, 2))
CXY[[2, 3, 4, 5, 0, 1]] = hexagon + [-1.8, 0]
CXY[[11, 10, 9, 8, 7, 6]] = hexagon * [-1, 1] + [1.8, 0]  # the mirror image, bus 11 opposite bus 2 and 10 opposite 3
CXY[12] = CXY[0] + 1.0 * hexagon[4]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
vmax = abs(v).max()
for (i, j) in CE:
    severed = side[i] != side[j]
    ax.plot(*CXY[[i, j]].T, color="tab:red" if severed else "0.35", lw=2.4 if severed else 1.3,
            ls=(0, (3, 1.5)) if severed else "-", zorder=1)
sc = ax.scatter(*CXY.T, c=v, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=120, edgecolor="0.2", lw=0.9, zorder=2)
ax.text(0, 1.2, "sweep cut\ncut 2, conductance 1/11", ha="center", va="bottom", fontsize=8.5, color="tab:red")
ax.text(CXY[12][0] - 0.2, CXY[12][1] - 0.3, "smallest cut\ncut 1, conductance 1", ha="left", va="top", fontsize=8.5, color="0.3")
cb = fig.colorbar(sc, cax=ax.inset_axes([0.58, 0.04, 0.38, 0.035]), orientation="horizontal")
cb.set_label("entry of the Fiedler vector", fontsize=8); cb.ax.tick_params(labelsize=7.5)
ax.set_xlim(CXY[:, 0].min() - 0.5, CXY[:, 0].max() + 0.5); ax.set_ylim(-2.3, 1.9)
ax.set_aspect("equal"); ax.axis("off")

ks = np.arange(1, 13)
bx.plot(ks, sweep[:, 1], color=BLUE, lw=1.6, marker="o", ms=4)
kmin = int(np.argmin(sweep[:, 1]))
bx.scatter([ks[kmin]], [sweep[kmin, 1]], s=70, facecolor="none", edgecolor="tab:red", lw=1.8, zorder=3)
for yb, txt in ((0.5 * lam2, r"$\lambda_2/2$"), (np.sqrt(2 * lam2), r"$\sqrt{2\lambda_2}$")):
    bx.axhline(yb, color="0.45", lw=1.0, ls="--")
    bx.text(12.3, yb * 1.06, txt, fontsize=8.5, color="0.3", ha="right", va="bottom")
bx.set_yscale("log")
bx.set_xlabel("buses on the side, in Fiedler order", fontsize=8.5)
bx.set_ylabel("conductance of the side", fontsize=8.5)
bx.set_xticks([1, 4, 7, 10, 12]); bx.tick_params(labelsize=8.5)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "sweep_cut.png", dpi=DPI)
plt.close(fig)
print(f"sweep cut: lambda_2 = {lam2:.4f}, bounds {0.5 * lam2:.4f} and {np.sqrt(2 * lam2):.4f}, conductance {h_sweep:.4f}")


# =====================================================================================
# 2. The reductions converge as the internal branches strengthen
# =====================================================================================
def zoned_network(internal, seed=0):
    """The construction of tests/test_reweight.py::_zoned_network: five clusters of four buses, every cluster a
    complete graph with susceptance `internal`, a ring of clusters plus one chord, each interconnection made of
    two lines between different bus pairs with susceptances drawn uniformly from [0.5, 2]."""
    rng = np.random.default_rng(seed)
    edges, b = [], []
    for zn in range(5):
        for i, j in combinations(range(4), 2):
            edges.append((4 * zn + i, 4 * zn + j)); b.append(internal)
    for za, zc in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        for (i, j) in ((0, 1), (2, 3)):
            edges.append((4 * za + i, 4 * zc + j)); b.append(rng.uniform(0.5, 2.0))
    return edges, np.array(b), np.repeat(np.arange(5), 4)


strengths = np.logspace(-0.5, 3, 36)
pairs = list(combinations(range(5), 2))
ratio = {"kron": [], "rigid": []}
for s in strengths:
    edges, b, z = zoned_network(s)
    Lf = laplacian(20, edges, b)
    Uz = np.eye(5)[z]
    ops = {"complete": zonal_operator(Lf, z), "kron": kron_reduce(Lf, [0, 4, 8, 12, 16]), "rigid": Uz.T @ Lf @ Uz}
    ref = np.array([resistance(ops["complete"], a, c) for a, c in pairs])
    for key, values in ratio.items():
        values.append(np.array([resistance(ops[key], a, c) for a, c in pairs]) / ref)
ratio = {k: np.array(v) for k, v in ratio.items()}
assert (ratio["rigid"] <= 1 + 1e-12).all()                  # the rigid aggregate is below every complete aggregate
assert abs(ratio["kron"][-1] - 1).max() < 0.01 and abs(ratio["rigid"][-1] - 1).max() < 0.01

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for key, color, label in (("kron", BLUE, "Kron reduction onto one bus per cluster"), ("rigid", ORANGE, "rigid aggregate")):
    r = ratio[key]
    ax.fill_between(strengths, r.min(1), r.max(1), color=color, alpha=0.22, lw=0)
    ax.plot(strengths, np.median(r, 1), color=color, lw=1.8)
    ax.text(strengths[9], np.median(r, 1)[9] + (0.13 if key == "kron" else -0.13), label, color=color, fontsize=8.5,
            va="center")
ax.axhline(1, color=GREEN, lw=1.4)
ax.text(strengths[-1], 1.03, "complete aggregate", color=GREEN, fontsize=8.5, ha="right", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("susceptance of the branches inside the clusters\n(between the clusters 0.5 to 2)", fontsize=8.5)
ax.set_ylabel("distance between two clusters\nrelative to the complete aggregate", fontsize=8.5)
ax.tick_params(labelsize=8.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "reductions_converge.png", dpi=DPI)
plt.close(fig)

print(f"two triangles, w = {W}: Kron {d_kron:.4f}, complete {d_complete:.4f}, rigid {d_rigid:.4f}")
for w in (1.0, 100.0):                                   # "for w = 100 they agree within 1.3 percent"
    Lw = laplacian(6, EDGES, [w] * 6 + [1.0])
    d = [resistance(kron_reduce(Lw, [0, 5]), 0, 1), resistance(zonal_operator(Lw, zones), 0, 1), resistance(U.T @ Lw @ U, 0, 1)]
    print(f"two triangles, w = {w:g}: Kron {d[0]:.4f}, complete {d[1]:.4f}, rigid {d[2]:.4f}, largest over smallest {max(d) / min(d):.4f}")
for s_ in (1.0, 10.0, 100.0):
    k = int(np.argmin(abs(strengths - s_)))
    print(f"internal {strengths[k]:7.2f}: Kron/complete median {np.median(ratio['kron'][k]):.3f} "
          f"[{ratio['kron'][k].min():.3f}, {ratio['kron'][k].max():.3f}], rigid/complete median "
          f"{np.median(ratio['rigid'][k]):.3f} [{ratio['rigid'][k].min():.3f}, {ratio['rigid'][k].max():.3f}]")
