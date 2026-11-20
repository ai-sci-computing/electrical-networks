"""A first look at the data: what is in the PyPSA-Eur OSM base network?

Reads the five CSV files of the PyPSA-Eur OSM network archive (version 0.7, the input of
the `base_network` rule; Hoersch et al. 2018 describe the model, the OSM-based network
build is documented in the PyPSA-Eur repository) and reports what the spectral argument
depends on: the distribution of nominal voltages, the number of transformers and where
their ratings come from, whether line ratings and impedances are set, and how the
alternating-current graph decomposes into synchronous areas.

Data: data/raw/pypsa-eur-osm-0.7/{buses,lines,links,converters,transformers}.csv from
https://data.pypsa.org/workflows/eur/osm/0.7/ (not committed; see README).
Run:  python studies/aggregation/european_grid/kill_point.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
FIG = Path(__file__).parent / "figures" / "osm_base_network.png"
FIG.parent.mkdir(exist_ok=True)

# geometry strings are single-quoted in these files


def read(name):
    return pd.read_csv(DATA / f"{name}.csv", quotechar="'")


buses, lines, trafos = read("buses"), read("lines"), read("transformers")
links, converters = read("links"), read("converters")
ac = buses[buses.dc.astype(str).isin(["f", "False"])]

# --- tables ---------------------------------------------------------------------------
print(f"buses {len(buses)} (AC {len(ac)}, DC {len(buses) - len(ac)}), countries {buses.country.nunique()}")
print("AC bus nominal voltages [kV]:")
print(ac.voltage.value_counts().sort_index().to_string())
by_v = lines.groupby("voltage").agg(n=("line_id", "size"), km=("length", lambda s: s.sum() / 1e3),
                                    circuits=("circuits", "sum"), type=("type", "first"))
print("lines by voltage:"); print(by_v.round(0).to_string())
print(f"lines with missing s_nom / x: {lines.s_nom.isna().sum()} / {lines.x.isna().sum()}")
print(f"transformers {len(trafos)}, s_nom from data (min/median/max) "
      f"{trafos.s_nom.min():.0f}/{trafos.s_nom.median():.0f}/{trafos.s_nom.max():.0f} MVA; x is not in the data")
pairs = pd.Series([f"{int(min(a, b))}-{int(max(a, b))}" for a, b in zip(trafos.voltage_bus0, trafos.voltage_bus1)])
print("transformer voltage pairs:"); print(pairs.value_counts().to_string())
print(f"DC links {len(links)}, converters {len(converters)}")

G = nx.Graph()
G.add_nodes_from(ac.bus_id)
G.add_edges_from(zip(lines.bus0, lines.bus1))
G.add_edges_from(zip(trafos.bus0, trafos.bus1))
comps = sorted(nx.connected_components(G), key=len, reverse=True)
sizes = [len(c) for c in comps]
print(f"AC graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(comps)} components, sizes {sizes}")
print(f"cycle rank (all components): {G.number_of_edges() - G.number_of_nodes() + len(comps)}")
for c in comps[:4]:
    cs = ac[ac.bus_id.isin(c)].country.value_counts()
    print(f"  component of {len(c)} buses: {', '.join(f'{k} {v}' for k, v in cs.head(6).items())}")

# --- figure ---------------------------------------------------------------------------
xy = buses.set_index("bus_id")[["x", "y"]]
vclass = pd.cut(lines.voltage, [0, 250, 350, 450, 800], labels=["220-236 kV", "254-330 kV", "380-420 kV", "500-750 kV"])
colors = {"220-236 kV": "tab:green", "254-330 kV": "tab:olive", "380-420 kV": "tab:red", "500-750 kV": "tab:purple"}
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
axm = fig.add_subplot(gs[:, 0])
for cls, col in colors.items():
    sel = lines[vclass == cls]
    segs = np.stack([xy.loc[sel.bus0].values, xy.loc[sel.bus1].values], axis=1)
    axm.add_collection(LineCollection(segs, colors=col, linewidths=0.5 if cls != "500-750 kV" else 1.2, label=cls))
axm.scatter(xy.loc[trafos.bus0].x, xy.loc[trafos.bus0].y, s=4, c="black", label=f"transformers ({len(trafos)})", zorder=3)
axm.set_xlim(-11, 42); axm.set_ylim(34, 72); axm.set_aspect(EUROPE_STRETCH)
axm.legend(loc="upper left", fontsize=8, frameon=False, markerscale=3)
axm.set_title(f"PyPSA-Eur OSM base network v0.7: {len(ac)} AC buses, {len(lines)} lines")
axm.set_xlabel("longitude"); axm.set_ylabel("latitude")

ax1 = fig.add_subplot(gs[0, 1])
vc = ac.voltage.value_counts().sort_index()
ax1.bar([str(v) for v in vc.index], vc.values, color="tab:blue")
ax1.set_title("AC buses per nominal voltage [kV]"); ax1.tick_params(axis="x", labelsize=8)
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(range(len(sizes)), sizes, color="tab:gray")
ax2.set_yscale("log"); ax2.set_xticks(range(len(sizes)))
ax2.set_title(f"connected components of the AC graph ({len(sizes)}): sizes")
for i, s in enumerate(sizes):
    ax2.text(i, s, str(s), ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print(f"wrote {FIG}")
