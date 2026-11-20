"""Figure for the conventions part: the IEEE 14-bus network and its admittance matrix.

Left: the graph of case14 with edge width proportional to the DC coupling b = 1/(x tau)
(MATPOWER manual 8.0, eq. 3.24); transformers (tau != 1) drawn in orange, the one bus
with a shunt capacitor marked. Right: |Y_bus| (eq. 3.13) on a log color scale, showing
the Laplacian-like sparsity pattern with the three transformer couplings.
Run:  python studies/foundations/setup/ieee14_figure.py   (needs data/matpower/case14.m)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LogNorm

from gridnet.admittance import bus_admittance, dc_model
from gridnet.network import read_matpower_case

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figures" / "ieee14_admittance.png"
OUT.parent.mkdir(exist_ok=True)

net = read_matpower_case(ROOT / "data" / "matpower" / "case14.m")
Y = bus_admittance(net)
_, Bf, _, _ = dc_model(net)
b = np.abs(Bf).max(axis=1)                     # b_l = 1/(x_l tau_l) per branch

G = nx.Graph()
G.add_nodes_from(range(net.n_bus))
for l, (i, j) in enumerate(net.edges):
    G.add_edge(i, j, b=b[l], trafo=net.tap[l] != 1.0)
pos = nx.kamada_kawai_layout(G, weight=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
widths = 0.4 + 4.0 * b / b.max()
colors = ["tab:orange" if net.tap[l] != 1.0 else "0.35" for l in range(net.n_branch)]
nx.draw_networkx_edges(G, pos, edgelist=net.edges, width=widths, edge_color=colors, ax=ax1)
node_colors = ["tab:green" if net.bs[i] != 0 else "white" for i in range(net.n_bus)]
nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="black", ax=ax1)
nx.draw_networkx_labels(G, pos, labels={i: str(net.bus_id[i]) for i in range(net.n_bus)}, font_size=8, ax=ax1)
ax1.set_title("IEEE 14: edge width ∝ 1/(xτ); transformers orange; shunt bus green")
ax1.set_axis_off()

im = ax2.imshow(np.abs(Y), norm=LogNorm(vmin=np.abs(Y[Y != 0]).min(), vmax=np.abs(Y).max()), cmap="viridis")
ax2.set_title(r"$|Y_{bus}|$ (log scale)")
ax2.set_xticks(range(net.n_bus)); ax2.set_xticklabels(net.bus_id, fontsize=7)
ax2.set_yticks(range(net.n_bus)); ax2.set_yticklabels(net.bus_id, fontsize=7)
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
