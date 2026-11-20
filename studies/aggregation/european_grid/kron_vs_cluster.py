"""What PyPSA-Eur's aggregation does to the electrical distances, measured against the Kron reduction.

The Kron reduction of the continental base network onto one representative bus per PyPSA-Eur
cluster is the exact reduction: it preserves the response seen from the kept buses and in
particular every effective resistance between them (Dorfler & Bullo 2013, Thm. 3.9), at the
price of a complete reduced graph (Thm. 3.4). PyPSA-Eur's `simplify_network` and
`cluster_network` (Hoersch et al. 2018; Hoersch & Brown 2017) instead produce a sparse network
of 128 cluster buses whose line reactances are re-derived from a uniform 380 kV line type,
lengths and capacities. Both describe the same clusters, so their couplings and effective
resistances can be compared pair by pair.

Compared quantities, on the 100 MVA base of gridnet.pypsa_io:
  * adjacent cluster pairs (lines of the clustered network): the aggregated susceptance
    1/x_cluster against the Kron coupling W_ij of the same pair;
  * all cluster pairs: effective resistance in the clustered network against the exact one;
  * the share of the total Kron coupling that sits on pairs the clustered network does not
    connect at all, the fill-in an aggregation discards;
  * the Kron-adjacent graph: the exact Kron couplings kept on the clustered pairs only, the
    rest dropped, against the exact effective resistances, to see how much of the aggregation's
    error is the parallel rule and how much the missing fill-in;
  * the quotient graph of the base network under the cluster map, zero resistance inside a
    cluster and the base branches between two clusters added in parallel, against the exact
    resistances: what the aggregation rule alone would give;
  * PyPSA's length convention: `aggregatelines` (pypsa/clustering/spatial.py) sets the length of
    an aggregated line to the great-circle distance between the two cluster buses times a factor
    and rescales the reactance by the ratio of that length to the original; the clustered
    lines' lengths and reactances against those of the simplified lines they replace;
  * the closing check: `aggregatelines` gives an aggregated line the reactance
    x_rule = L / sum_k (l_k / x_k) in per unit (each replaced line k, of length l_k and reactance x_k,
    stretched by L / l_k and all of them in parallel; the voltage factor of the code cancels in per
    unit), so the length convention alone predicts x_rule / x_parallel = L sum_k (1/x_k) / sum_k (l_k/x_k).
    Dividing that predicted stretch out of every clustered line must land at the quotient graph's
    numbers if the aggregated network is the quotient graph stretched by the length convention and
    nothing else; whatever else changes the reactance (the line type re-derived from length and
    num_parallel) shows as observed / predicted != 1. Dividing out the observed ratio x_ratio instead
    rebuilds the parallel sum of the simplified lines by construction, the quotient graph of the
    simplified network, and tests nothing; it is kept as that quotient graph.
Two choices of representative (nearest to the cluster centroid, largest weighted degree) and
three transformer reactances (0.05, 0.1, 0.2 p.u.; PyPSA-Eur's default is 0.1) give the sensitivities.

Verification inside the script: the effective resistances of the reduced matrix equal those of
the full network between the same buses, computed independently by a grounded solve.

Data: data/raw/pypsa-eur-osm-0.7 (archive), data/raw/pypsa-eur-gridnet (bus maps and
networks/clustered.nc of the 128-cluster run).  Run: python studies/aggregation/european_grid/kron_vs_cluster.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _clustered import BASE_MVA, PYPSA_EUR, kept_buses, on_clusters, read_clustered
from matplotlib.collections import LineCollection

from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import laplacian_from_couplings, reduced_couplings
from gridnet.pypsa_io import great_circle_km

FIGDIR = Path(__file__).parent / "figures"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# --- the clustered network (_clustered.py, shared with sparse_reductions.py) -----------------
import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

clustered = read_clustered()
nc, centroids, clus = clustered["nc"], clustered["centroids"], clustered["clus"]
x100 = nc.lines.x_pu.values * BASE_MVA                 # PyPSA's per unit is on 1 MVA


# --- PyPSA's length convention, from the simplified network the clustering step consumed --------
ns = pypsa.Network(str(PYPSA_EUR / "networks" / "simplified.nc"))
ns.calculate_dependent_values()
s2c = ns.buses.index.to_series().map(clus)
# per pair of clusters: PyPSA-Eur joins 18 pairs by two or three clustered lines, each replacing part of the
# simplified lines between the pair, so the comparison is between the parallel sums over the whole pair
a, b = ns.lines.bus0.map(s2c), ns.lines.bus1.map(s2c)
simplified_pair = pd.Series([tuple(sorted(t)) for t in zip(a, b)], index=ns.lines.index)
clustered_pair = pd.Series([tuple(sorted(t)) for t in zip(nc.lines.bus0, nc.lines.bus1)], index=nc.lines.index)
conv = []
for pair, names in clustered_pair.groupby(clustered_pair).groups.items():
    cl = nc.lines.loc[names]
    sl = ns.lines[simplified_pair == pair]
    d = great_circle_km(centroids[pair[0]], centroids[pair[1]])
    if len(sl) and d > 0:
        L = float(cl.length.mean())                                                     # the same for every line of the pair
        conv.append({"bus0": pair[0], "bus1": pair[1], "lines": " ".join(map(str, names)), "n_clustered": len(cl),
                         "length_clustered": L, "centroid_distance": d, "n_replaced": len(sl),
                         "length_ratio": L / sl.length.mean(), "x_ratio": (1 / sl.x_pu).sum() / (1 / cl.x_pu).sum(),         # parallel x of the pair's clustered lines over that of the replaced ones
                         "x_rule_ratio": L * (1 / sl.x_pu).sum() / (sl.length / sl.x_pu).sum()})       # the length convention alone
conv = pd.DataFrame(conv); conv.to_csv(RESULTS / "kron_vs_cluster_length_convention.csv", index=False)
print(f"length convention: clustered length / centroid distance = {conv.eval('length_clustered / centroid_distance').median():.3f} (median); "
      f"clustered length / mean replaced length quartiles {conv.length_ratio.quantile(.25):.2f}, {conv.length_ratio.median():.2f}, {conv.length_ratio.quantile(.75):.2f}; "
      f"clustered x / parallel x of replaced lines quartiles {conv.x_ratio.quantile(.25):.2f}, {conv.x_ratio.median():.2f}, {conv.x_ratio.quantile(.75):.2f}; "
      f"observed / predicted by the length rule quartiles {*(conv.x_ratio / conv.x_rule_ratio).quantile([.25, .5, .75]).round(3),}")


def grounded_effective_resistance(L, nodes):
    """R_ij between the given nodes from the full Laplacian, by grounding nodes[0]: with
    G = L[-g,-g]^{-1}, R_ij = G_ii + G_jj - 2 G_ij and R_ig = G_ii (Dorfler & Bullo 2013, eq. 1.3
    written with the grounded inverse)."""
    g = nodes[0]
    others = np.array([i for i in range(L.shape[0]) if i != g])
    pos = {int(i): k for k, i in enumerate(others)}
    cols = np.array([pos[int(i)] for i in nodes[1:]])
    E = np.zeros((others.size, cols.size)); E[cols, np.arange(cols.size)] = 1.0
    X = np.linalg.solve(L[np.ix_(others, others)], E)     # potentials for unit injections
    G = X[cols, :]                                          # k-1 by k-1 block of the grounded inverse
    k = len(nodes)
    R = np.zeros((k, k))
    d = np.diag(G)
    R[1:, 1:] = d[:, None] + d[None, :] - 2 * G
    R[0, 1:] = R[1:, 0] = d
    return R


summary = []
pairs_out = []
t0 = time.time()
for xT in (0.05, 0.1, 0.2):
    # clustered AC graph among the clusters present in the component, its largest component, and the base network
    setup = on_clusters(clustered, xT)
    net, labels, clusters, cidx, Lc, Bc, edges, L = (setup[k] for k in ("net", "labels", "clusters", "cidx", "Lc", "Bc", "edges", "L"))
    comp = set(clusters)
    Rc = effective_resistance(Lc)
    adjacent = Bc > 0
    # quotient graph of the base network under the cluster map (the aggregation rule alone)
    qe, qw = [], []
    for (i, j), bw in zip(edges, 1 / net.x):
        li, lj = labels[i], labels[j]
        if isinstance(li, str) and isinstance(lj, str) and li != lj and li in cidx and lj in cidx:
            qe.append((cidx[li], cidx[lj])); qw.append(bw)
    Rq = effective_resistance(laplacian(len(clusters), qe, np.array(qw)))
    # the clustered network with a stretch divided out of every line: the observed one (the quotient graph of the
    # simplified network, by construction) and the one the length convention predicts (the closing check)
    def unstretched(column, comp=comp, clusters=clusters, cidx=cidx):
        stretch = pd.Series({str(n): r for names, r in zip(conv.lines, conv[column]) for n in names.split()})
        xs = pd.Series(x100, index=nc.lines.index.astype(str)) / stretch.reindex(nc.lines.index.astype(str)).fillna(1.0)
        su = pd.DataFrame({"bus0": nc.lines.bus0.values, "bus1": nc.lines.bus1.values, "b": (1 / xs).values})
        su = su[su.bus0.isin(comp) & su.bus1.isin(comp)]
        return effective_resistance(laplacian(len(clusters), [(cidx[a], cidx[b]) for a, b in zip(su.bus0, su.bus1)], su.b.values))
    Rs, Ru = unstretched("x_ratio"), unstretched("x_rule_ratio")
    for mode in ("nearest", "degree"):
        keep = kept_buses(clustered, setup, mode).tolist()
        W, R = reduced_couplings(L, keep)
        if xT == 0.1 and mode == "nearest":
            Rchk = grounded_effective_resistance(L, keep)
            err = np.max(np.abs(R - Rchk)) / np.max(Rchk)
            print(f"Thm 3.9 check: max relative deviation of reduced vs full effective resistances {err:.1e}")
            assert err < 1e-8
        iu = np.triu_indices(len(clusters), 1)
        ratio_b = Bc[adjacent & np.triu(np.ones_like(adjacent), 1)] / W[adjacent & np.triu(np.ones_like(adjacent), 1)]
        ratio_R = Rc[iu] / R[iu]
        offgrid = W[iu][~adjacent[iu]].sum() / W[iu].sum()
        Radj = effective_resistance(laplacian_from_couplings(W, mask=adjacent))
        ratio_adj = Radj[iu] / R[iu]
        ratio_q = Rq[iu] / R[iu]
        ratio_u = Ru[iu] / R[iu]
        ratio_s = Rs[iu] / R[iu]
        summary.append({"x_T": xT, "representative": mode, "clusters": len(clusters), "lines": int(adjacent[iu].sum()),
                            "b_ratio_median": float(np.median(ratio_b)), "b_ratio_q25": float(np.quantile(ratio_b, .25)),
                            "b_ratio_q75": float(np.quantile(ratio_b, .75)),
                            "R_ratio_median": float(np.median(ratio_R)), "R_ratio_q25": float(np.quantile(ratio_R, .25)),
                            "R_ratio_q75": float(np.quantile(ratio_R, .75)),
                            "R_log_corr": float(np.corrcoef(np.log(Rc[iu]), np.log(R[iu]))[0, 1]),
                            "kron_coupling_off_lines": float(offgrid),
                            "R_adj_ratio_median": float(np.median(ratio_adj)), "R_adj_ratio_q25": float(np.quantile(ratio_adj, .25)),
                            "R_adj_ratio_q75": float(np.quantile(ratio_adj, .75)),
                            "R_adj_log_corr": float(np.corrcoef(np.log(Radj[iu]), np.log(R[iu]))[0, 1]),
                            "R_quotient_ratio_median": float(np.median(ratio_q)), "R_quotient_ratio_q25": float(np.quantile(ratio_q, .25)),
                            "R_quotient_ratio_q75": float(np.quantile(ratio_q, .75)),
                            "R_simplified_quotient_ratio_median": float(np.median(ratio_s)),
                            "R_unstretched_ratio_median": float(np.median(ratio_u))})
        if xT == 0.1 and mode == "nearest":
            for i, j in zip(*np.where(adjacent & np.triu(np.ones_like(adjacent), 1))):
                pairs_out.append({"a": clusters[i], "b": clusters[j], "b_cluster": Bc[i, j], "w_kron": W[i, j], "R_cluster": Rc[i, j], "R_kron": R[i, j]})
            main = {"clusters": clusters, "Rc": Rc, "R": R, "Bc": Bc, "W": W, "adjacent": adjacent}
    print(f"x_T={xT}: {len(clusters)} clusters, {int(adjacent[iu].sum())} clustered lines, {time.time()-t0:.0f} s")

summ = pd.DataFrame(summary); summ.to_csv(RESULTS / "kron_vs_cluster_summary.csv", index=False)
pd.DataFrame(pairs_out).to_csv(RESULTS / "kron_vs_cluster_pairs.csv", index=False)
print(summ.round(3).to_string(index=False))
# the length convention on the pairs of the comparison, the clustered lines inside the continental component
cont = {tuple(sorted(p)) for p in zip(pd.DataFrame(pairs_out).a, pd.DataFrame(pairs_out).b)}
sub = conv[[tuple(sorted(p)) in cont for p in zip(conv.bus0, conv.bus1)]]
for label, c in (("all pairs of the run", conv), ("the pairs of the comparison", sub)):
    print(f"length convention, {label} ({len(c)}): clustered length / mean replaced length quartiles "
          f"{c.length_ratio.quantile(.25):.2f}, {c.length_ratio.median():.2f}, {c.length_ratio.quantile(.75):.2f}; clustered x / parallel x of the "
          f"replaced lines quartiles {c.x_ratio.quantile(.25):.2f}, {c.x_ratio.median():.2f}, {c.x_ratio.quantile(.75):.2f}; its inverse, the factor "
          f"on the susceptance, quartiles {(1 / c.x_ratio).quantile(.25):.2f}, {(1 / c.x_ratio).median():.2f}, {(1 / c.x_ratio).quantile(.75):.2f}")

# --- figure --------------------------------------------------------------------------------
clusters, Rc, R, Bc, W, adjacent = (main[k] for k in ("clusters", "Rc", "R", "Bc", "W", "adjacent"))
iu = np.triu_indices(len(clusters), 1)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
ax.scatter(R[iu], Rc[iu], s=5, color="0.55", alpha=0.5, lw=0, label="all pairs")
adj = adjacent[iu]
ax.scatter(R[iu][adj], Rc[iu][adj], s=9, color="tab:blue", lw=0, label="pairs joined by a line")
lim = (min(R[iu].min(), Rc[iu].min()) * 0.8, max(R[iu].max(), Rc[iu].max()) * 1.2)
ax.plot(lim, lim, color="0.3", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("effective resistance, Kron reduction [p.u.]"); ax.set_ylabel("effective resistance, clustered network [p.u.]")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax = axes[1]
xy = np.array([centroids[c] for c in clusters])
segs, vals = [], []
for i, j in zip(*np.where(adjacent & np.triu(np.ones_like(adjacent), 1))):
    segs.append([xy[i], xy[j]]); vals.append(np.log10(Bc[i, j] / W[i, j]))
lc = LineCollection(segs, cmap="RdBu_r", norm=plt.Normalize(-1, 1), linewidths=1.6)
lc.set_array(np.array(vals)); ax.add_collection(lc)
ax.scatter(xy[:, 0], xy[:, 1], s=6, color="0.2", zorder=3)
ax.set_xlim(-10.5, 30.5); ax.set_ylim(35, 58); ax.set_aspect(EUROPE_STRETCH); ax.axis("off")
cb = fig.colorbar(lc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("log10 of clustered over Kron coupling", fontsize=8); cb.ax.tick_params(labelsize=7)
fig.tight_layout()
FIGDIR.mkdir(exist_ok=True)
fig.savefig(FIGDIR / "kron_vs_cluster.png", dpi=200, bbox_inches="tight")
print("wrote", FIGDIR / "kron_vs_cluster.png")
