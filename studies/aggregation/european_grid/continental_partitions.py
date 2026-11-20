"""Spectral k-way partitions of the continental component, borders ignored, against the countries.

country_partitions.py and pypsa_eur_comparison.py partition inside countries because PyPSA-Eur never
lets a cluster cross a border.
This script asks the prior question: does the electrical structure of the continental grid follow the
borders at all? Normalized spectral clustering (von Luxburg 2007, Sec. 4; Shi & Malik 2000) of the
5,577-bus component into k parts for a range of k, compared with the country labels by the adjusted
Rand index (Hubert & Arabie 1985) and the normalized mutual information, and at k = 96 with
PyPSA-Eur's own clusters of the component. At k equal to the number of countries, the
purity of each country (share of its buses in its majority part) says which countries are electrical
units and which are not. The recursive two-way cuts (Shi & Malik 2000, Sec. 3.2), the weakest cut first,
give the sequence in which the continent comes apart after the Pyrenees (the first sweep cut of
continental_laplacian.py). Sensitivity to the transformer reactance as in continental_laplacian.py and
country_partitions.py.

Data: data/raw/pypsa-eur-osm-0.7 and data/raw/pypsa-eur-gridnet bus maps; Natural Earth borders in
data/raw/naturalearth (scripts/fetch_data.py --only naturalearth).
Run: python studies/aggregation/european_grid/continental_partitions.py
"""

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from gridnet.graph import laplacian
from gridnet.pypsa_io import read_osm_archive
from gridnet.spectral import ncut, recursive_bipartition, spectral_embedding

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
PYPSA_EUR = ROOT / "data" / "raw" / "pypsa-eur-gridnet"
NE = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.shp"
FIGDIR = Path(__file__).parent / "figures"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 0
KS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 96]
K_MAX = max(KS)

t0 = time.time()
net, info = read_osm_archive(ARCHIVE, component=0, trafo_x=0.1)
buses = info["buses"].reset_index(drop=True)
country = buses.country.values
n_countries = len(set(country))
big = buses.country.value_counts()
print(f"component: {len(buses)} buses, {n_countries} countries, {int((big >= 20).sum())} with at least 20 buses")
L = laplacian(net.n_bus, net.edges, 1 / net.x)
X = spectral_embedding(L, K_MAX, normalized=True)
print(f"embedding of {K_MAX} eigenvectors in {time.time()-t0:.0f} s")

simp = pd.read_csv(PYPSA_EUR / "busmap_simplify_network.csv", index_col=0)["busmap"]
clus = pd.read_csv(PYPSA_EUR / "busmap_cluster_network.csv", index_col=0)["busmap"]
pypsa_cluster = buses.bus_id.map(simp.map(clus)).values
has_cluster = pd.notna(pypsa_cluster)

def replay(history, k, n):
    """Labels of the recursive partition after k-1 splits, from the history of a deeper run."""
    labels = np.zeros(n, dtype=np.int64)
    for h in history[: k - 1]:
        labels[h["new_side"]] = h["new"]
    return labels


hist = []
recursive_bipartition(L, K_MAX, normalized=True, history=hist)
print(f"recursive hierarchy to {K_MAX} parts in {time.time()-t0:.0f} s")
rows, parts = [], {}
for k in sorted(set(KS + [n_countries])):
    rec = replay(hist, k, len(buses))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X[:, :k])
    parts[k] = rec
    row = {"k": k, "ncut_recursive": ncut(L, rec), "ncut_kmeans": ncut(L, km), "ari_recursive_kmeans": adjusted_rand_score(rec, km),
               "ari_country": adjusted_rand_score(country, rec), "nmi_country": normalized_mutual_info_score(country, rec),
               "ari_country_kmeans": adjusted_rand_score(country, km)}
    if k == 96:
        row["ari_pypsa"] = adjusted_rand_score(pypsa_cluster[has_cluster], rec[has_cluster])
        row["ari_pypsa_kmeans"] = adjusted_rand_score(pypsa_cluster[has_cluster], km[has_cluster])
    rows.append(row)
    print(f"k={k:3d}  Ncut rec {row['ncut_recursive']:.4f} km {row['ncut_kmeans']:.4f}  ARI rec/km {row['ari_recursive_kmeans']:.2f}  "
          f"ARI vs countries rec {row['ari_country']:.3f} km {row['ari_country_kmeans']:.3f}"
          + (f"  vs PyPSA-Eur rec {row['ari_pypsa']:.3f} km {row['ari_pypsa_kmeans']:.3f}" if k == 96 else ""))
scores = pd.DataFrame(rows).sort_values("k"); scores.to_csv(RESULTS / "continental_partitions_scores.csv", index=False)

# purity at k = number of countries: which countries are electrical units
kc = n_countries
lab = parts[kc]
purity = []
for c, cnt in big.items():
    idx = country == c
    vc = pd.Series(lab[idx]).value_counts()
    purity.append({"country": c, "buses": int(cnt), "majority_part": int(vc.index[0]), "share_in_majority": float(vc.iloc[0] / cnt),
                       "parts_touched": int((vc / cnt >= 0.05).sum())})
purity = pd.DataFrame(purity); purity.to_csv(RESULTS / "continental_partitions_country_purity.csv", index=False)
# and the converse: how many countries share a part
part_countries = pd.Series(country).groupby(lab).apply(lambda s: (s.value_counts() / len(s) >= 0.1).sum())
print(f"k = {kc} (one per country): country purity median {purity.share_in_majority.median():.2f}, "
      f"{int((purity.share_in_majority >= 0.9).sum())} of {len(purity)} countries with 90 % of buses in one part; "
      f"parts holding buses of 2+ countries (each at least 10 %): {int((part_countries >= 2).sum())} of {kc}")

# recursive cuts: the order in which the continent comes apart
seq = []
for step, h in enumerate(hist[:7], 1):
    comp_new = pd.Series(country[h["new_side"]]).value_counts()
    comp_rest = pd.Series(country[h["parent_side"]]).value_counts()
    seq.append({"step": step, "conductance": h["conductance"],
                    "split_off": {"buses": int(h["new_side"].size), "countries": {c: int(n) for c, n in comp_new.head(4).items()}},
                    "remaining": {"buses": int(h["parent_side"].size), "countries": {c: int(n) for c, n in comp_rest.head(4).items()}}})
    print(f"cut {step}: h = {h['conductance']:.1e}, splits off {h['new_side'].size} buses { {c: int(n) for c, n in comp_new.head(3).items()} } "
          f"from {h['parent_side'].size} buses { {c: int(n) for c, n in comp_rest.head(3).items()} }")
(RESULTS / "continental_partitions_recursive.json").write_text(json.dumps(seq, indent=1))

# sensitivity to the transformer reactance at k = number of countries
sens = []
for xT in (0.05, 0.2):
    net2, _ = read_osm_archive(ARCHIVE, component=0, trafo_x=xT)
    L2 = laplacian(net2.n_bus, net2.edges, 1 / net2.x)
    lab2 = recursive_bipartition(L2, kc, normalized=True)
    sens.append({"x_T": xT, "ari_vs_default": adjusted_rand_score(lab, lab2), "ari_country": adjusted_rand_score(country, lab2)})
    print(f"x_T={xT}: ARI vs default partition {sens[-1]['ari_vs_default']:.3f}, vs countries {sens[-1]['ari_country']:.3f}")
pd.DataFrame(sens).to_csv(RESULTS / "continental_partitions_sensitivity.csv", index=False)

# --- figure: the partition at k = number of countries over the borders, and ARI against k ------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
land = gpd.read_file(NE).cx[-11:41, 34:71]
land.plot(ax=ax, facecolor="#f2f0eb", edgecolor="0.35", linewidth=0.5, zorder=0)
xy = buses[["x", "y"]].values
rng = np.random.default_rng(3)
colors = np.vstack([plt.get_cmap("tab20")(np.r_[0:14, 16:20]), plt.get_cmap("tab20b")(np.arange(20))])  # tab20 without its two greys
colors = colors[rng.permutation(len(colors))][: kc + 1]
segs = [[xy[i], xy[j]] for i, j in net.edges]
cut = [lab[i] != lab[j] for i, j in net.edges]
ax.add_collection(LineCollection([s for s, c in zip(segs, cut) if not c], colors="0.7", linewidths=0.3, zorder=1))
ax.scatter(xy[:, 0], xy[:, 1], c=colors[lab % len(colors)], s=3, lw=0, zorder=2)
ax.add_collection(LineCollection([s for s, c in zip(segs, cut) if c], colors="black", linewidths=0.8, zorder=3))
ax.set_xlim(-10.5, 40.5); ax.set_ylim(35, 60); ax.set_aspect(EUROPE_STRETCH); ax.axis("off")
# no in-plot title, the caption names the panel
bx.plot(scores.k, scores.ari_country, "o-", color="tab:blue", ms=3, lw=1.2, label="hierarchy against the countries")
bx.plot(scores.k, scores.ari_country_kmeans, "s--", color="tab:blue", ms=2.5, lw=0.9, alpha=0.6, label="spectral clustering against the countries")
r96 = scores.loc[scores.k == 96].iloc[0]
bx.scatter([96], [r96.ari_pypsa], color="tab:red", s=25, zorder=3, label="hierarchy against PyPSA-Eur's clusters")
bx.axvline(kc, color="0.6", lw=0.8, ls=":"); bx.text(kc + 1, 0.02, f"k = {kc}, one per country", fontsize=7, color="0.4")
bx.set_xscale("log"); bx.set_xlabel("number of parts"); bx.set_ylabel("adjusted Rand index")
bx.set_ylim(0, 1); bx.legend(fontsize=7, frameon=False, loc="upper right")
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
fig.tight_layout()
FIGDIR.mkdir(exist_ok=True)
fig.savefig(FIGDIR / "continental_partitions.png", dpi=200, bbox_inches="tight")
print("wrote", FIGDIR / "continental_partitions.png", f"total {time.time()-t0:.0f} s")
