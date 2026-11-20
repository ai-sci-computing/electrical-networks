"""K-way partitions within countries, electrical versus geographic criterion.

For each country with enough buses in the continental component, the induced subgraph
(largest connected part) is partitioned in two ways with the same k:
  * electrical: normalized spectral clustering of the susceptance Laplacian L(1/x)
    (von Luxburg 2007, Sec. 4, the Shi & Malik 2000 relaxation of the normalized cut);
  * geographic: k-means on bus coordinates (the criterion of PyPSA-Eur's default
    `cluster_network`, Hoersch & Brown 2017, minus its load weighting and its prior
    simplification).
Each partition is scored under both criteria: the normalized cut Ncut (electrical) and the
k-means inertia in km^2 (geographic). Agreement by adjusted Rand index, normalized mutual
information (scikit-learn) and variation of information. Sensitivity of the electrical
partition to the transformer reactance x_T by ARI.
Run:  python studies/aggregation/european_grid/country_partitions.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from gridnet.graph import laplacian
from gridnet.pypsa_io import planar_km, read_osm_archive
from gridnet.spectral import ncut, spectral_clustering, variation_of_information

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
K_TOTAL = 100          # clusters distributed over countries in proportion to bus count
MIN_BUSES = 150
SEED = 0


def country_subgraph(net, info, country, trafo_x_net=None):
    buses = info["buses"]
    sel = np.flatnonzero(buses.country.values == country)
    inside = set(sel.tolist())
    br = [l for l, (i, j) in enumerate(net.edges) if i in inside and j in inside]
    G = nx.Graph()
    G.add_nodes_from(sel.tolist())
    G.add_edges_from((net.edges[l][0], net.edges[l][1]) for l in br)
    comp = max(nx.connected_components(G), key=len)
    nodes = sorted(comp)
    idx = {b: k for k, b in enumerate(nodes)}
    br = [l for l in br if net.edges[l][0] in comp]
    edges = [(idx[net.edges[l][0]], idx[net.edges[l][1]]) for l in br]
    x = net.x[br]
    xy = buses.iloc[nodes][["x", "y"]].values
    return nodes, edges, x, xy, len(sel) - len(nodes)


def inertia(P, labels):
    return float(sum(((P[labels == c] - P[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels)))


net, info = read_osm_archive(ARCHIVE, component=0, trafo_x=0.1)
nets_x = {xT: read_osm_archive(ARCHIVE, component=0, trafo_x=xT)[0] for xT in (0.05, 0.2)}
counts = info["buses"].country.value_counts()
countries = [c for c, n in counts.items() if n >= MIN_BUSES]
N = counts[countries].sum()

rows, maps = [], {}
for c in countries:
    nodes, edges, x, xy, dropped = country_subgraph(net, info, c)
    k = max(2, round(K_TOTAL * len(nodes) / N))
    L = laplacian(len(nodes), edges, 1 / x)
    lab_e = spectral_clustering(L, k, normalized=True, seed=SEED)
    P = planar_km(xy)
    lab_g = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(P)
    row = {
        "country": c, "buses": len(nodes), "dropped": dropped, "branches": len(edges), "k": k,
        "ncut_electrical": ncut(L, lab_e), "ncut_geographic": ncut(L, lab_g),
        "inertia_electrical_km2": inertia(P, lab_e), "inertia_geographic_km2": inertia(P, lab_g),
        "ari": adjusted_rand_score(lab_e, lab_g), "nmi": normalized_mutual_info_score(lab_e, lab_g),
        "vi": variation_of_information(lab_e, lab_g),
        "cut_branches_electrical": int(sum(lab_e[i] != lab_e[j] for i, j in edges)),
        "cut_branches_geographic": int(sum(lab_g[i] != lab_g[j] for i, j in edges)),
    }
    for xT, netx in nets_x.items():
        _, _, xx, _, _ = country_subgraph(netx, info, c)
        labx = spectral_clustering(laplacian(len(nodes), edges, 1 / xx), k, normalized=True, seed=SEED)
        row[f"ari_electrical_xT{xT}"] = adjusted_rand_score(lab_e, labx)
    rows.append(row)
    maps[c] = (xy, edges, lab_e, lab_g, k)
    print(f"{c}: n={len(nodes)} (dropped {dropped}) k={k}  Ncut e/g = {row['ncut_electrical']:.3f}/{row['ncut_geographic']:.3f}  "
          f"inertia e/g = {row['inertia_electrical_km2']:.2e}/{row['inertia_geographic_km2']:.2e}  ARI={row['ari']:.2f} NMI={row['nmi']:.2f}  "
          f"ARI(xT=0.05,0.2)={row['ari_electrical_xT0.05']:.2f},{row['ari_electrical_xT0.2']:.2f}")
(RESULTS / "country_partitions.json").write_text(json.dumps(rows, indent=1))

# --- figure: three countries, electrical (top) vs geographic (bottom) ----------------
COUNTRY = {"DE": "Germany", "FR": "France", "IT": "Italy"}
show = [c for c in ("DE", "FR", "IT") if c in maps]
score = {r["country"]: r for r in rows}
fig, axes = plt.subplots(2, len(show), figsize=(5 * len(show), 9.5))
# one color per part: tab20 without its two greys and tab20b, 38 colors for at most 25 parts (France)
PALETTE = np.vstack([plt.get_cmap("tab20")(np.r_[0:14, 16:20]), plt.get_cmap("tab20b")(np.arange(20))])
for col, c in enumerate(show):
    xy, edges, lab_e, lab_g, k = maps[c]
    for row_i, (lab, name) in enumerate([(lab_e, "spectral clustering"), (lab_g, "$k$-means on the coordinates")]):
        ax = axes[row_i, col]
        segs = np.stack([xy[[i for i, _ in edges]], xy[[j for _, j in edges]]], axis=1)
        cut = np.array([lab[i] != lab[j] for i, j in edges])
        ax.add_collection(LineCollection(segs[~cut], colors="0.8", linewidths=0.5))
        ax.add_collection(LineCollection(segs[cut], colors="black", linewidths=1.2))
        assert lab.max() < len(PALETTE)
        ax.scatter(xy[:, 0], xy[:, 1], c=PALETTE[lab], s=9, zorder=3)
        ax.set_aspect(EUROPE_STRETCH)
        nc = score[c]["ncut_electrical" if row_i == 0 else "ncut_geographic"]
        ax.set_title(f"{COUNTRY.get(c, c)}, {k} parts, {name}\nnormalized cut {nc:.2f}")
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIGDIR / "country_partitions.png", dpi=150, bbox_inches="tight")
print("wrote", FIGDIR / "country_partitions.png")
