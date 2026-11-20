"""PyPSA-Eur's clustered network against the complete aggregate: the figure and the sensitivity to x_T.

Completes matched_reference.py with two items:

1. the comparison at transformer reactances 0.05, 0.1 and 0.2 p.u. The clustered network is PyPSA-Eur's own and does
   not change; the base network, and with it the complete aggregate (Chow & Kokotovic 1985, eqs. (4.3), (4.4)), the
   Kron reduction (Dorfler & Bullo 2013, eq. (2.1)) and the rigid aggregate (eq. (4.7)), are rebuilt for each value.
2. figures/matched_reference.png, the successor of figures/kron_vs_cluster.png (kron_vs_cluster.py, left as it is): left, the effective resistances of the clustered network against those of the
   complete aggregate with demand as distribution (the Kron reduction onto the representative buses in gray for
   comparison); right, the median ratio against the number of buses that carry a cluster's distribution, from
   results/matched_reference_concentration.csv.

Numerics: every reference resistance is (p_a - p_c)^T L^+ (p_a - p_c), computed with the grounded inverse, which is
exact for zero-sum vectors. The rigid aggregate is built from the branches between clusters directly, not as U^T L U,
whose zero eigenvalue is polluted by the cancellation of the internal susceptances (found 2026-09-19 on a small
example, studies/aggregation/theory/aggregation_figure.py); the two constructions are compared below.

Run: .venv/bin/python studies/aggregation/european_grid/matched_reference_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import laplacian_from_couplings
from gridnet.pypsa_io import read_osm_archive

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
ARCHIVE = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
RESULTS, FIGDIR = HERE / "results", HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY = "tab:blue", "tab:orange", "0.6"

z = np.load(RESULTS / "kron128.npz")
nb, Bc, keep = int(z["n_base"]), z["Bc"], z["keep"]
k = len(Bc)
iu = np.triu_indices(k, 1)
adjacent = (Bc > 0)[iu]
R_net = effective_resistance(laplacian_from_couplings(Bc))

# the demand of matched_reference.py: mean hourly demand of the simplified buses, shared equally among the base buses
# merged into one simplified bus
demand = xr.open_dataset(ROOT / "data" / "raw" / "pypsa-eur-gridnet" / "electricity_demand_simplified.nc")["electricity demand (MW)"].mean("time").to_series()
simplified = pd.Series(z["simplified_bus"])
load0 = (simplified.map(demand).fillna(0.0) / simplified.map(simplified.value_counts()).fillna(1.0)).values


def references(xT):
    """Resistances between the clusters for the base network with transformer reactance xT: complete aggregate with
    demand, complete aggregate with uniform distributions, Kron reduction onto the representatives, rigid aggregate."""
    net, _ = read_osm_archive(ARCHIVE, component=0, trafo_x=xT)
    i, j, b = np.asarray(net.from_bus), np.asarray(net.to_bus), 1 / np.asarray(net.x)
    L = laplacian(nb, list(zip(i.tolist(), j.tolist())), b)
    if xT == 0.1:                                       # the cache's bus order (it stores parallel branches merged)
        assert np.allclose(-L[z["base_i"], z["base_j"]], z["base_b"]) and np.count_nonzero(np.triu(L, 1)) == len(z["base_b"])
    zone = z["zone"].copy(); lost = np.flatnonzero(zone < 0)       # buses PyPSA-Eur's map lacks go to the electrically nearest zone
    X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
    dd = dijkstra(X, directed=False, indices=lost); dd[:, lost] = np.inf
    zone[lost] = zone[np.argmin(dd, axis=1)]
    load = load0.copy()
    for zn in np.flatnonzero(np.bincount(zone, weights=load, minlength=k) == 0):
        load[zone == zn] = 1.0
    G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])

    def resistance(w):
        P = np.zeros((nb, k)); P[np.arange(nb), zone] = w; P /= P.sum(0, keepdims=True)
        M = P.T @ G @ P; d = np.diag(M)
        return (d[:, None] + d[None, :] - 2 * M)[iu]

    onehot = np.zeros(nb); onehot[keep] = 1.0
    cross = zone[i] != zone[j]
    rigid = laplacian(k, list(zip(zone[i][cross].tolist(), zone[j][cross].tolist())), b[cross])
    U = np.zeros((nb, k)); U[np.arange(nb), zone] = 1.0
    other = effective_resistance(U.T @ L @ U)[iu]
    R_rigid = effective_resistance(rigid)[iu]
    print(f"   x_T = {xT}: rigid aggregate built from the branches between clusters against U^T L U, largest relative "
          f"deviation of a resistance {np.abs(other / R_rigid - 1).max():.1e}")
    return {"demand": resistance(load), "uniform": resistance(np.ones(nb)), "kron": resistance(onehot), "rigid": R_rigid}


print("1. PyPSA-Eur's clustered network and the rigid aggregate over the references, by transformer reactance")
rows, main = [], None
for xT in (0.05, 0.1, 0.2):
    ref = references(xT)
    if xT == 0.1:
        main = ref
    for name in ("demand", "uniform", "kron"):
        q = R_net[iu] / ref[name]
        rows.append({"x_T": xT, "network": "PyPSA-Eur", "reference": name, "q25": np.quantile(q, .25), "median": np.median(q),
                         "q75": np.quantile(q, .75), "median_adjacent": np.median(q[adjacent]),
                         "log_correlation": np.corrcoef(np.log(R_net[iu]), np.log(ref[name]))[0, 1]})
    q = ref["rigid"] / ref["demand"]
    rows.append({"x_T": xT, "network": "rigid aggregate", "reference": "demand", "q25": np.quantile(q, .25), "median": np.median(q),
                     "q75": np.quantile(q, .75), "median_adjacent": np.median(q[adjacent]),
                     "log_correlation": np.corrcoef(np.log(ref["rigid"]), np.log(ref["demand"]))[0, 1]})
out = pd.DataFrame(rows); out.to_csv(RESULTS / "matched_reference_sensitivity.csv", index=False)
print(out.round(3).to_string(index=False))

# --- the figure ---------------------------------------------------------------------------------------------------
conc = pd.read_csv(RESULTS / "matched_reference_concentration.csv")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.4))
y = R_net[iu]
ax.scatter(main["kron"], y, s=3, color=GRAY, alpha=0.35, lw=0, rasterized=True)
ax.scatter(main["demand"][~adjacent], y[~adjacent], s=3, color=ORANGE, alpha=0.35, lw=0, rasterized=True)
ax.scatter(main["demand"][adjacent], y[adjacent], s=7, color=BLUE, lw=0, zorder=3)
lo, hi = min(y.min(), main["demand"].min()) * 0.7, max(y.max(), main["kron"].max()) * 1.3
ax.plot([lo, hi], [lo, hi], color="0.2", lw=0.9, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("exact effective resistance between two clusters (p.u.)", fontsize=8.5)
ax.set_ylabel("effective resistance in PyPSA-Eur's network (p.u.)", fontsize=8.5)
ax.text(0.04, 0.95, "against the complete aggregate\nclusters joined by a line\nother pairs", transform=ax.transAxes, fontsize=8,
        va="top", color="0.2")
for yy, c in ((0.885, BLUE), (0.83, ORANGE)):
    ax.scatter([0.6], [yy], transform=ax.transAxes, s=14, color=c, lw=0, clip_on=False)
ax.text(0.96, 0.05, "against the Kron reduction\nonto one bus per cluster", transform=ax.transAxes, fontsize=8, ha="right", color="0.45")

rand = conc[conc.buses_per_cluster.astype(str).str.fullmatch(r"\d+")].copy()
m = rand.buses_per_cluster.astype(int).values
bx.fill_between(m, rand.lowest, rand.highest, color=BLUE, alpha=0.22, lw=0)
bx.plot(m, rand.median_ratio, color=BLUE, lw=1.6, marker="o", ms=4)
top = conc[conc.buses_per_cluster.astype(str).str.startswith("demand,")]
mt = top.buses_per_cluster.str.extract(r"(\d+)")[0].astype(int).values
bx.plot(mt, top.median_ratio, color=ORANGE, lw=0, marker="s", ms=5)
full = float(conc.loc[conc.buses_per_cluster == "all, demand", "median_ratio"].iloc[0])
bx.axhline(full, color="0.3", lw=1.0, ls="--")
bx.text(20, full + 0.012, f"all buses, by demand, {full:.2f}", fontsize=8, ha="right", color="0.3")
bx.text(2.6, 0.575, "the buses of largest demand", fontsize=8, color=ORANGE)
bx.text(4.2, 0.665, "buses drawn at random", fontsize=8, color=BLUE)
bx.set_xscale("log"); bx.set_xticks(m); bx.set_xticklabels([str(v) for v in m]); bx.minorticks_off()
bx.set_ylim(0.45, 0.9)
bx.set_xlabel("buses that carry a cluster's distribution", fontsize=8.5)
bx.set_ylabel("median ratio of PyPSA-Eur's resistances\nto the exact ones", fontsize=8.5)
for a in (ax, bx):
    a.tick_params(labelsize=8)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR / "matched_reference.png", dpi=200)
print(f"figure written, {k} clusters, {int(adjacent.sum())} lines")
