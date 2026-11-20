"""Explanatory figure for the cycle basis: spanning tree, fundamental cycles, winding number.

Left: IEEE 14 with a breadth-first spanning tree (solid) and its chords (dashed); the
seven chords give seven fundamental cycles (Kavitha et al. 2009, Sec. 2), two of them
shaded. Right: a 6-node ring with angles advancing by 60 degrees per node drawn as
arrows on the unit circle; the wrapped differences sum to -2 pi, winding number -1
(Manik et al. 2017, Def. 6; Delabays et al. 2016, Sec. II).
Run:  python studies/aggregation/theory/cycles_figure.py   (needs data/matpower/case14.m)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gridnet.cycles import fundamental_cycle_basis, spanning_forest, winding_numbers
from gridnet.network import read_matpower_case

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figures" / "cycle_basis.png"
OUT.parent.mkdir(exist_ok=True)

net = read_matpower_case(ROOT / "data" / "matpower" / "case14.m")
edges = net.edges
tree, chords = spanning_forest(net.n_bus, edges)
C = fundamental_cycle_basis(net.n_bus, edges)

G = nx.Graph()
G.add_nodes_from(range(net.n_bus))
G.add_edges_from(edges)
pos = nx.kamada_kawai_layout(G)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})

# shade two fundamental cycles as polygons through the node positions of their support
for k, color in [(0, "tab:blue"), (3, "tab:red")]:
    support = np.flatnonzero(C[:, k])
    H = nx.Graph([edges[l] for l in support])
    cyc = nx.cycle_basis(H)[0]
    poly = np.array([pos[v] for v in cyc])
    ax1.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.15, zorder=0)
    nx.draw_networkx_edges(G, pos, edgelist=[edges[l] for l in support], width=3.5, edge_color=color, alpha=0.6, ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=[edges[l] for l in tree], width=1.6, edge_color="0.2", ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=[edges[l] for l in chords], width=1.6, edge_color="0.2", style="dashed", ax=ax1)
nx.draw_networkx_nodes(G, pos, node_size=260, node_color="white", edgecolors="black", ax=ax1)
nx.draw_networkx_labels(G, pos, labels={i: str(net.bus_id[i]) for i in range(net.n_bus)}, font_size=8, ax=ax1)
ax1.set_title(f"IEEE 14: BFS spanning tree (solid), {len(chords)} chords (dashed),\ntwo of the {C.shape[1]} fundamental cycles shaded")
ax1.set_axis_off()

# ring with winding number
n = 6
ring = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
theta = 2 * np.pi * np.arange(n) / n
q = winding_numbers(theta, n, ring)[0]
t = np.linspace(0, 2 * np.pi, 200)
ax2.plot(np.cos(t), np.sin(t), color="0.7", lw=1)
for i in range(n):
    ax2.annotate("", xy=(np.cos(theta[i]), np.sin(theta[i])), xytext=(0, 0),
                 arrowprops={"arrowstyle": "->", "lw": 2, "color": plt.cm.viridis(i / n)})
    ax2.text(1.18 * np.cos(theta[i]), 1.18 * np.sin(theta[i]), rf"$\theta_{i}$", ha="center", va="center", fontsize=10)
ax2.set_aspect("equal")
ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
ax2.set_axis_off()
ax2.set_title(f"ring of {n}: $\\theta_i = 2\\pi i/{n}$; each difference $\\theta_i-\\theta_{{i+1}} = -\\pi/3$\nsum $= -2\\pi$, winding number $q = {q}$")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
