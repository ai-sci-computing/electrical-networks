"""PyPSA-Eur's own clustering against the electrical partition, per country.

PyPSA-Eur (pinned commit, run `gridnet`, 128 clusters, k-means) produces two bus maps:
base bus -> simplified bus (`simplify_network`, Hoersch et al. 2018) and simplified bus ->
cluster (`cluster_network`, load-weighted k-means within countries, Hoersch & Brown 2017).
Their composition assigns every base-network bus a cluster; base bus names are the OSM
archive ids, so the assignment transfers to the continental component (continental_laplacian.py).

For each country, with k_c the number of clusters PyPSA-Eur gave it: PyPSA-Eur's
partition of the country's buses versus normalized spectral clustering of the country
subgraph with the same k_c (von Luxburg 2007, Sec. 4; Shi & Malik 2000), plus the
geographic k-means baseline of country_partitions.py. Scores: Ncut on the country Laplacian L(1/x) and
k-means inertia in km^2; agreement by ARI, NMI, VI (Hubert & Arabie 1985; Meila 2007).
Run:  python studies/aggregation/european_grid/pypsa_eur_comparison.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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
PYPSA_EUR = ROOT / "data" / "raw" / "pypsa-eur-gridnet"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 0

simp = pd.read_csv(PYPSA_EUR / "busmap_simplify_network.csv", index_col=0)["busmap"]
clus = pd.read_csv(PYPSA_EUR / "busmap_cluster_network.csv", index_col=0)["busmap"]
base_to_cluster = simp.map(clus)                       # composition base -> simplified -> cluster
print(f"PyPSA-Eur: {len(simp)} base buses -> {simp.nunique()} simplified -> {clus.nunique()} clusters; "
      f"unmapped base buses: {int(base_to_cluster.isna().sum())}")

net, info = read_osm_archive(ARCHIVE, component=0, trafo_x=0.1)
buses = info["buses"].copy()
buses["cluster"] = buses.bus_id.map(base_to_cluster)
print(f"continental component: {len(buses)} buses, {int(buses.cluster.notna().sum())} with a PyPSA-Eur cluster")


def inertia(P, labels):
    return float(sum(((P[labels == c] - P[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels)))


rows, maps = [], {}
for c in sorted(buses.country.unique()):
    sel = np.flatnonzero((buses.country.values == c) & buses.cluster.notna().values)
    k = buses.cluster.iloc[sel].nunique()
    if len(sel) < 100 or k < 2:
        continue
    inside = set(sel.tolist())
    br = [l for l, (i, j) in enumerate(net.edges) if i in inside and j in inside]
    G = nx.Graph(); G.add_nodes_from(sel.tolist()); G.add_edges_from(net.edges[l] for l in br)
    comp = max(nx.connected_components(G), key=len)
    nodes = sorted(comp); idx = {b: t for t, b in enumerate(nodes)}
    br = [l for l in br if net.edges[l][0] in comp]
    edges = [(idx[net.edges[l][0]], idx[net.edges[l][1]]) for l in br]
    L = laplacian(len(nodes), edges, 1 / net.x[br])
    xy = buses.iloc[nodes][["x", "y"]].values
    P = planar_km(xy)
    lab_p = pd.factorize(buses.cluster.iloc[nodes])[0]
    k = len(np.unique(lab_p))
    lab_e = spectral_clustering(L, k, normalized=True, seed=SEED)
    lab_g = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(P)
    row = {"country": c, "buses": len(nodes), "k": k,
           "ncut_pypsa": ncut(L, lab_p), "ncut_electrical": ncut(L, lab_e), "ncut_geographic": ncut(L, lab_g),
           "inertia_pypsa": inertia(P, lab_p), "inertia_electrical": inertia(P, lab_e), "inertia_geographic": inertia(P, lab_g),
           "ari_pypsa_electrical": adjusted_rand_score(lab_p, lab_e), "nmi_pypsa_electrical": normalized_mutual_info_score(lab_p, lab_e),
           "vi_pypsa_electrical": variation_of_information(lab_p, lab_e),
           "ari_pypsa_geographic": adjusted_rand_score(lab_p, lab_g),
           # plain k-means against the electrical partition, the comparison of the figure (its
           # agreement with PyPSA-Eur above says nothing about that)
           "ari_geographic_electrical": adjusted_rand_score(lab_g, lab_e),
           "cut_pypsa": int(sum(lab_p[i] != lab_p[j] for i, j in edges)), "cut_electrical": int(sum(lab_e[i] != lab_e[j] for i, j in edges))}
    rows.append(row); maps[c] = (xy, edges, lab_p, lab_e, k)
    print(f"{c}: n={len(nodes)} k={k}  Ncut pypsa/elec/geo = {row['ncut_pypsa']:.3f}/{row['ncut_electrical']:.3f}/{row['ncut_geographic']:.3f}  "
          f"inertia = {row['inertia_pypsa']:.2e}/{row['inertia_electrical']:.2e}/{row['inertia_geographic']:.2e}  "
          f"ARI(p,e)={row['ari_pypsa_electrical']:.2f} ARI(p,g)={row['ari_pypsa_geographic']:.2f} ARI(g,e)={row['ari_geographic_electrical']:.2f}")
(RESULTS / "pypsa_eur_comparison.json").write_text(json.dumps(rows, indent=1))

COUNTRY = {"DE": "Germany", "FR": "France", "IT": "Italy"}
show = [c for c in ("DE", "FR", "IT") if c in maps]
score = {r["country"]: r for r in rows}
fig, axes = plt.subplots(2, len(show), figsize=(5 * len(show), 9.5))
for col, c in enumerate(show):
    xy, edges, lab_p, lab_e, k = maps[c]
    for r_i, (lab, name) in enumerate([(lab_p, "PyPSA-Eur"), (lab_e, "spectral clustering")]):
        ax = axes[r_i, col]
        segs = np.stack([xy[[i for i, _ in edges]], xy[[j for _, j in edges]]], axis=1)
        cut = np.array([lab[i] != lab[j] for i, j in edges])
        ax.add_collection(LineCollection(segs[~cut], colors="0.8", linewidths=0.5))
        ax.add_collection(LineCollection(segs[cut], colors="black", linewidths=1.2))
        ax.scatter(xy[:, 0], xy[:, 1], c=lab, cmap="tab20", s=9, zorder=3)
        ax.set_aspect(EUROPE_STRETCH)
        nc = score[c]["ncut_pypsa" if r_i == 0 else "ncut_electrical"]
        ax.set_title(f"{COUNTRY.get(c, c)}, {k} clusters, {name}\nnormalized cut {nc:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIGDIR / "pypsa_eur_comparison.png", dpi=150, bbox_inches="tight")
print("wrote", FIGDIR / "pypsa_eur_comparison.png")
