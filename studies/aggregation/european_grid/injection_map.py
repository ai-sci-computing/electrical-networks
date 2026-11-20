"""Where the injections go: Ward's distribution of the eliminated injections against the clustering's rule.

A clustering puts the whole injection of a bus on the node of its own cluster, the 0/1 matrix T. The exact
reduction onto one representative bus per cluster distributes it by the accompanying matrix of the Kron
reduction, D = -L_ab L_bb^{-1} on the eliminated buses (Dorfler & Bullo 2013, Sec. I, Lemma 2.1(3); Ward 1949's
distribution factors; Tinney & Powell 1977, eq. (4)): nonnegative, columns summing to one, entry (a, x) the
probability that a random walk from bus x reaches representative a first (Doyle & Snell, Sec. 1.3.2). D needs no
operating point. This script measures how far T is from D on the European base network (5,577 buses, 96 clusters
of PyPSA-Eur in the main synchronous component), and makes the same comparison in the second convention of a
reduction, the complete aggregate of Chow & Kokotovic 1985, where a zone's injection is its sum and the angle its mean.

Predictions, written before the first run (2026-09-19): since the clusters are electrically as wide as the gaps
between them, the median share of a bus's injection that D gives to its own cluster's
representative is around one half, buses near a cluster boundary give most to neighbors, and a typical bus feeds
three or four representatives noticeably.

Run: .venv/bin/python studies/aggregation/european_grid/injection_map.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gridnet.kron import accompanying_matrix, kron_reduce

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
z = np.load(RESULTS / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
keep, xy, Bc = z["keep"], z["bus_xy"], z["Bc"]
k = len(keep)
A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).toarray()
L = np.diag(A.sum(1)) - A
zone = z["zone"].copy(); lost = np.flatnonzero(zone < 0)
X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
dd = dijkstra(X, directed=False, indices=lost); dd[:, lost] = np.inf
zone[lost] = zone[np.argmin(dd, axis=1)]

D = accompanying_matrix(L, keep)
assert np.all(D > -1e-10) and np.allclose(D.sum(0), 1.0)          # Lemma 2.1(3)
T = np.zeros((k, nb)); T[zone, np.arange(nb)] = 1.0

own = D[zone, np.arange(nb)]
adjacent = (Bc > 0) | np.eye(k, dtype=bool)
own_and_adjacent = np.array([D[adjacent[zone[x]], x].sum() for x in range(nb)])
recipients = 1.0 / np.sum(D ** 2, axis=0)                          # effective number of representatives fed by a bus
noticeable = np.sum(D > 0.05, axis=0)
interior = np.ones(nb, dtype=bool); interior[keep] = False
q = lambda v: np.quantile(v[interior], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
print(f"{nb} buses, {k} clusters; quantiles 5, 25, 50, 75, 95 % over the {int(interior.sum())} eliminated buses")
print("share of a bus's injection that reaches its own cluster's representative:", q(own), " mean", own[interior].mean().round(3))
print("share that reaches its own or an adjacent cluster's representative:     ", q(own_and_adjacent))
print("effective number of representatives fed, 1 / sum_a D_ax^2:              ", q(recipients))
print("number of representatives receiving more than 5 %:                      ", q(noticeable.astype(float)))
print(f"buses whose own representative receives less than half: {np.mean(own[interior] < 0.5):.3f}; less than a quarter: {np.mean(own[interior] < 0.25):.3f}; "
      f"buses for which another cluster's representative receives more than their own: {np.mean(np.argmax(D, axis=0)[interior] != zone[interior]):.3f}")
labeled = interior & (z["zone"] >= 0)                              # without the buses PyPSA-Eur's cluster map lacks (Ukraine, Moldova), assigned here by proximity
print(f"restricted to the {int(labeled.sum())} buses of PyPSA-Eur's cluster map: own share quartiles {np.quantile(own[labeled], [.25, .5, .75]).round(3)}, below one half {np.mean(own[labeled] < 0.5):.3f}")
print(f"relative Frobenius distance ||D - T|| / ||T|| = {np.linalg.norm(D - T) / np.linalg.norm(T):.3f}  (the relative error of the cluster injections for uncorrelated bus injections of equal variance)")

# the same question in the second convention: zone means of the angles as response, the injections of a zone summed
# (exact if they are spread evenly over the zone) against the true response to the individual bus injections
G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])       # grounded inverse; differs from L^+ only by constants per column and row
P = T.T / T.sum(1)[None, :]
Pi_n = np.eye(nb) - 1.0 / nb; Pi_k = np.eye(k) - 1.0 / k
true_response = Pi_k @ (P.T @ G) @ Pi_n                           # zone means of the angles for balanced bus injections
aggregated = Pi_k @ (P.T @ G @ P) @ T @ Pi_n                      # the same with every zone's injection spread evenly
print(f"complete aggregate: relative Frobenius error of the zone-mean angles when a zone's injections are replaced by their sum, spread evenly: "
      f"{np.linalg.norm(true_response - aggregated) / np.linalg.norm(true_response):.3f}")
kron_response = Pi_k @ np.linalg.pinv(kron_reduce(L, keep)) @ Pi_k   # angles of the representatives per reduced injection
exact_rep = kron_response @ D @ Pi_n; with_T = kron_response @ T @ Pi_n
print(f"Kron reduction: relative Frobenius error of the representatives' angles when D is replaced by T: {np.linalg.norm(exact_rep - with_T) / np.linalg.norm(exact_rep):.3f}")

pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "zone": zone, "own_share": own, "own_and_adjacent": own_and_adjacent, "recipients": recipients}).to_csv(RESULTS / "injection_map.csv", index=False)

# figure style: column width 7.2 in, no titles inside the plot, direct labels; the caption carries the description
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
sc = ax.scatter(xy[interior, 0], xy[interior, 1], c=own[interior], s=1.6, cmap="viridis", vmin=0, vmax=1, lw=0, rasterized=True)
ax.scatter(xy[keep, 0], xy[keep, 1], s=9, facecolor="none", edgecolor="k", lw=0.6)
ax.set_aspect(EUROPE_STRETCH); ax.axis("off")
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.6)
cb.set_label("fraction of a bus's injection that reaches\nthe kept bus of its own cluster", fontsize=8); cb.ax.tick_params(labelsize=7.5)
ax = axes[1]
ax.hist(own[interior], bins=40, color="tab:blue", alpha=0.85)
med_own = float(np.median(own[interior]))
ax.axvline(med_own, color="k", lw=1, ls="--"); ax.axvline(1.0, color="tab:red", lw=1.4)
top = ax.get_ylim()[1]
ax.text(med_own + 0.02, top * 0.97, f"median {med_own:.2f}", fontsize=8, va="top")
ax.text(0.98, top * 0.6, "the sum over\neach cluster", fontsize=8, color="tab:red", ha="right", va="top")
ax.set_xlabel("fraction that reaches the own kept bus", fontsize=8.5); ax.set_ylabel("buses", fontsize=8.5); ax.tick_params(labelsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "figures" / "injection_map.png", dpi=200)
print("wrote", HERE / "figures" / "injection_map.png")
