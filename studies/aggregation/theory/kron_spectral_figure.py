"""Explanatory figure for Kron reduction and spectral partitioning.

(a) Star-mesh transform: eliminating the center of a star (Kron reduction, Dorfler &
    Bullo 2013, eq. 2.1) yields a complete graph with weights b_i b_j / sum b; effective
    resistances between the remaining nodes are unchanged (Thm. 3.9).
(b) IEEE 14 Kron-reduced to its five generator buses (1, 2, 3, 6, 8): the induced dense
    graph with edge width proportional to the reduced coupling (Thm. 3.4).
(c) IEEE 14 Fiedler vector of the susceptance Laplacian L(1/x) as node color, and the
    conductance sweep cut (von Luxburg 2007, Sec. 5; Bandeira et al. 2013, Thm. 1.1)
    as a dashed boundary; the value lies between the Cheeger bounds.
Run:  python studies/aggregation/theory/kron_spectral_figure.py   (needs data/matpower/case14.m)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gridnet.graph import laplacian
from gridnet.kron import kron_reduce, laplacian_to_graph
from gridnet.network import read_matpower_case
from gridnet.spectral import cheeger_bounds, fiedler, sweep_cut

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figures" / "kron_spectral.png"
OUT.parent.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})

# (a) star-mesh -----------------------------------------------------------------------
b = np.array([1.0, 2.0, 3.0, 4.0])
Lstar = laplacian(5, [(0, i) for i in range(1, 5)], b)
Lmesh = kron_reduce(Lstar, keep=[1, 2, 3, 4])
mesh_edges, mesh_w, _ = laplacian_to_graph(Lmesh)
ax = axes[0]
ang = np.pi / 2 + 2 * np.pi * np.arange(4) / 4
leaf = {i + 1: (np.cos(a), np.sin(a)) for i, a in enumerate(ang)}
# star on the left, mesh on the right
for (i, (x, y)) in leaf.items():
    ax.plot([-1.6, -1.6 + x], [0, y], color="0.3", lw=1 + b[i - 1])
    ax.text(-1.6 + 1.25 * x, 1.25 * y, f"$b_{i}={b[i - 1]:.0f}$", ha="center", va="center", fontsize=8)
for (i, j), w in zip(mesh_edges, mesh_w):
    (x0, y0), (x1, y1) = leaf[i + 1], leaf[j + 1]
    ax.plot([1.6 + x0, 1.6 + x1], [y0, y1], color="tab:blue", lw=1 + 3 * w)
    ax.text(1.6 + (x0 + x1) / 2, (y0 + y1) / 2, f"{w:.2f}", fontsize=7, ha="center", va="center",
            bbox={"fc": "white", "ec": "none", "pad": 0.5})
for cx, pts in [(-1.6, {**leaf, 0: (0, 0)}), (1.6, leaf)]:
    for i, (x, y) in pts.items():
        ax.scatter([cx + x], [y], s=260, c="tab:orange" if i == 0 else "white", edgecolors="black", zorder=3)
        ax.text(cx + x, y, str(i), ha="center", va="center", fontsize=8, zorder=4)
ax.annotate("", xy=(0.45, -1.35), xytext=(-0.45, -1.35), arrowprops={"arrowstyle": "-|>", "lw": 2})
ax.text(0, -1.2, "eliminate 0", ha="center", fontsize=9)
ax.set_xlim(-3.1, 3.1); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.set_axis_off()
ax.set_title("(a) star-mesh: Kron reduction of a star\nweights $b_ib_j/\\sum b$, resistances preserved")

# (b) IEEE 14 reduced to generator buses ----------------------------------------------
net = read_matpower_case(ROOT / "data" / "matpower" / "case14.m")
L = laplacian(net.n_bus, net.edges, 1 / net.x)
gen_ext = [1, 2, 3, 6, 8]
keep = [int(np.flatnonzero(net.bus_id == g)[0]) for g in gen_ext]
Lr = kron_reduce(L, keep)
red_edges, red_w, _ = laplacian_to_graph(Lr)
G = nx.Graph(); G.add_nodes_from(range(net.n_bus)); G.add_edges_from(net.edges)
pos = nx.kamada_kawai_layout(G)
ax = axes[1]
nx.draw_networkx_edges(G, pos, edge_color="0.85", ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[i for i in range(net.n_bus) if i not in keep], node_size=120, node_color="0.9", edgecolors="0.6", ax=ax)
for (a_, b_), w in zip(red_edges, red_w):
    i, j = keep[a_], keep[b_]
    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="tab:blue", lw=0.5 + 2.5 * w / red_w.max(), alpha=0.8)
nx.draw_networkx_nodes(G, pos, nodelist=keep, node_size=300, node_color="tab:orange", edgecolors="black", ax=ax)
nx.draw_networkx_labels(G, pos, labels={i: str(net.bus_id[i]) for i in keep}, font_size=8, ax=ax)
ax.set_title(f"(b) IEEE 14 Kron-reduced to generator buses\n{len(red_edges)} reduced couplings (width ∝ weight)")
ax.set_axis_off()

# (c) Fiedler vector and sweep cut ---------------------------------------------------
lam, v = fiedler(L, normalized=True)
S, h = sweep_cut(L, v, objective="conductance")
lo, hi = cheeger_bounds(L)
ax = axes[2]
cut_edges = [e for e in net.edges if S[e[0]] != S[e[1]]]
nx.draw_networkx_edges(G, pos, edge_color="0.5", ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="tab:red", width=2.5, style="dashed", ax=ax)
nodes = nx.draw_networkx_nodes(G, pos, node_size=300, node_color=v, cmap=plt.cm.coolwarm, edgecolors="black", ax=ax)
nx.draw_networkx_labels(G, pos, labels={i: str(net.bus_id[i]) for i in range(net.n_bus)}, font_size=7, ax=ax)
fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.02, label="Fiedler vector (normalized)")
ax.set_title(f"(c) IEEE 14: Fiedler vector of $L(1/x)$ and sweep cut (dashed)\n$\\lambda_2/2={lo:.3f} \\leq h_S={h:.3f} \\leq \\sqrt{{2\\lambda_2}}={hi:.3f}$")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
