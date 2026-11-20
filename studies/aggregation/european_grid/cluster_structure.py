"""Is the European grid well-clustered? Three measurements behind the disagreement between its reductions.

1. Effective resistance against geographic distance in the European base network (5,577 buses). In a
   homogeneous two-dimensional medium the resistance between two points grows with the logarithm of
   their distance, the Green's function of the plane Laplacian; on a graph of large minimal degree it
   even approaches 1/d_i + 1/d_j, a purely local quantity. Which of these the grid follows decides
   whether a patch of the grid can be electrically small against its distance to the next patch.
2. How well separated the parts of a partition are, for PyPSA-Eur's geographic clusters, for k-means on the
   coordinates and for spectral clustering of the susceptance Laplacian, at k = 10, 30, 96: the mean effective
   resistance between two buses of the same part against the mean between buses of adjacent parts.

3. The gap Upsilon(k) = lambda_{k+1} / rho(k) by which Peng, Sun & Zanetti 2015 define a well-clustered graph.

Sources: Barooah & Hespanha 2009, Thm. 1 (resistance grows linearly, logarithmically, or stays bounded in graphs that
are sparse and dense in one, two, three dimensions); Cserti 2000, eq. (33) (the square lattice, R = (ln r + gamma + ln(8)/2) / pi);
von Luxburg, Radl & Hein 2014 (the local limit 1/d_i + 1/d_j); Peng, Sun & Zanetti 2015, eqs. (1.2), (1.3). Run: .venv/bin/python studies/aggregation/european_grid/cluster_structure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gridnet.pypsa_io import great_circle_km, planar_km
from gridnet.spectral import spectral_clustering

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
_log = open(RESULTS / "cluster_structure.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/cluster_structure.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()
z = np.load(RESULTS / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
xy = z["bus_xy"]
A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).toarray()
L = np.diag(A.sum(1)) - A
G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])       # grounded inverse; R_ij = G_ii + G_jj - 2 G_ij
d = np.diag(G).copy()
deg = np.diag(L).copy()


# --- 1. resistance against distance ------------------------------------------------------------------
rng = np.random.default_rng(0)
a, c = rng.integers(0, nb, 400000), rng.integers(0, nb, 400000)
ok = a != c; a, c = a[ok], c[ok]
R = d[a] + d[c] - 2 * G[a, c]
dist = great_circle_km(xy[a], xy[c])
local = 1 / deg[a] + 1 / deg[c]
sel = dist > 5
bins = np.geomspace(10, 3500, 26)
mid = np.sqrt(bins[1:] * bins[:-1])
which = np.digitize(dist, bins)
med = np.array([np.median(R[which == k]) if np.sum(which == k) > 50 else np.nan for k in range(1, len(bins))])
q25 = np.array([np.quantile(R[which == k], .25) if np.sum(which == k) > 50 else np.nan for k in range(1, len(bins))])
q75 = np.array([np.quantile(R[which == k], .75) if np.sum(which == k) > 50 else np.nan for k in range(1, len(bins))])
good = np.isfinite(med)


def r2(x, y):
    p = np.polyfit(x, y, 1); res = y - np.polyval(p, x)
    return p, 1 - res.var() / y.var()


fits = {"R = a + b log(distance)": r2(np.log(mid[good]), med[good]), "R = a + b distance": r2(mid[good], med[good]),
        "log R = a + b log(distance)": r2(np.log(mid[good]), np.log(med[good]))}
print("median effective resistance against distance, binned (10 km to 3500 km):")
for name, (p, score) in fits.items():
    print(f"  {name:30s} slope {p[0]:.4g}, intercept {p[1]:.4g}, R^2 = {score:.4f}")
for lo, hi in ((10, 500), (500, 3500)):
    part = good & (mid >= lo) & (mid <= hi)
    (pl, sl), (pp, sp), (pn, sn) = r2(np.log(mid[part]), med[part]), r2(np.log(mid[part]), np.log(med[part])), r2(mid[part], med[part])
    print(f"  {lo:4d} to {hi:4d} km ({int(part.sum())} bins): logarithmic R^2 = {sl:.4f} (slope {pl[0]:.4g} per e-fold); power law exponent {pp[0]:.3f}, R^2 = {sp:.4f}; linear R^2 = {sn:.4f}")
print(f"  median R at 50 km {np.interp(50, mid[good], med[good]):.4f}, 200 km {np.interp(200, mid[good], med[good]):.4f}, 800 km {np.interp(800, mid[good], med[good]):.4f}, 3000 km {np.interp(3000, mid[good], med[good]):.4f} p.u.")
print(f"  the local quantity 1/d_i + 1/d_j: median {np.median(local):.4f} p.u.; correlation of R with it over all pairs {np.corrcoef(R, local)[0, 1]:.3f}, "
      f"median R / (1/d_i + 1/d_j) = {np.median(R / local):.2f}")

# --- 2. separation of the parts of a partition ---------------------------------------------------------
zone96 = z["zone"].copy(); lost = np.flatnonzero(zone96 < 0)
X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
dd = dijkstra(X, directed=False, indices=lost); dd[:, lost] = np.inf
zone96[lost] = zone96[np.argmin(dd, axis=1)]


def separation(labels):
    """The median over parts of the mean effective resistance between two buses of one part, the median over pairs
    of adjacent parts (parts joined by a branch) of the mean between a bus of the one and a bus of the other, the
    ratio of the two medians, and the severed fraction of the susceptance."""
    labels = np.asarray(labels); k = labels.max() + 1
    members = [np.flatnonzero(labels == m) for m in range(k)]
    inside = []
    for idx in members:
        if len(idx) > 1:
            Rz = d[idx][:, None] + d[idx][None, :] - 2 * G[np.ix_(idx, idx)]
            inside.append(Rz[np.triu_indices(len(idx), 1)].mean())
    pairs = {(min(labels[p], labels[q]), max(labels[p], labels[q])) for p, q in zip(i, j) if labels[p] != labels[q]}
    between = [np.mean(d[members[m]][:, None] + d[members[n]][None, :] - 2 * G[np.ix_(members[m], members[n])]) for m, n in pairs]
    cut = sum(w for p, q, w in zip(i, j, b) if labels[p] != labels[q]) / b.sum()
    return np.median(inside), np.median(between), np.median(inside) / np.median(between), cut


rows = []
for k in (10, 30, 96):
    _, geo = kmeans2(planar_km(xy), k, minit="++", seed=1)
    _, geo = np.unique(geo, return_inverse=True)
    spec = spectral_clustering(L, k, seed=1)
    for name, lab in (("k-means on coordinates", geo), ("spectral clustering", spec)):
        ins, bet, ratio, cut = separation(lab)
        rows.append({"k": int(lab.max() + 1), "partition": name, "R_inside": ins, "R_between_adjacent": bet, "ratio": ratio, "share_of_susceptance_cut": cut})
ins, bet, ratio, cut = separation(zone96)
rows.append({"k": 96, "partition": "PyPSA-Eur clusters", "R_inside": ins, "R_between_adjacent": bet, "ratio": ratio, "share_of_susceptance_cut": cut})
lam = np.linalg.eigvalsh(L)[:130]
out = pd.DataFrame(rows); out.to_csv(RESULTS / "cluster_structure.csv", index=False)
print(out.round(4).to_string(index=False))
print("eigenvalue ratios lambda_{k+1} / lambda_k of the susceptance Laplacian (a gap marks k well-separated parts):",
      {k: round(float(lam[k] / lam[k - 1]), 3) for k in (2, 3, 5, 10, 30, 96)}, "largest ratio for k in 2..120:", int(np.argmax(lam[2:121] / lam[1:120]) + 2))

# figure style: column width 7.2 in, no titles inside the plot, direct labels; the caption carries the description
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
ax = axes[0]
ax.fill_between(mid[good], q25[good], q75[good], color="tab:blue", alpha=0.18, lw=0)
ax.plot(mid[good], med[good], "o-", color="tab:blue", ms=3, lw=1.5)
core = good & (mid <= 500); pc = np.polyfit(np.log(mid[core]), med[core], 1)
ax.plot(mid[good], np.polyval(pc, np.log(mid[good])), "--", color="0.3", lw=1)
ax.axhline(np.median(local), color="tab:red", lw=1, ls=":")
ax.text(0.04, 0.93, "median and quartiles", transform=ax.transAxes, fontsize=8, color="tab:blue")
ax.text(mid[good][-1], np.polyval(pc, np.log(mid[good][-1])) - 0.011, "logarithm fitted\nup to 500 km", fontsize=8, color="0.3", ha="right", va="top")
ax.text(mid[good][-1], np.median(local) + 0.002, "$1/d_i + 1/d_j$", fontsize=8, color="tab:red", ha="right", va="bottom")
ax.set_xscale("log"); ax.set_xlabel("geographic distance (km)", fontsize=8.5); ax.set_ylabel("effective resistance (p.u.)", fontsize=8.5)
ax = axes[1]
for name, color, marker in (("k-means on coordinates", "tab:orange", "s"), ("spectral clustering", "tab:green", "^"), ("PyPSA-Eur clusters", "tab:red", "*")):
    dsel = out[out.partition == name]
    ax.plot(dsel.k, dsel.ratio, marker=marker, color=color, lw=1.5, ms=9 if marker == "*" else 5)
    last = dsel.iloc[1] if len(dsel) > 1 else dsel.iloc[-1]          # label at the middle point, clear of the lines
    if len(dsel) > 1:
        ax.text(last.k, last.ratio + (0.025 if "means" in name else -0.025), name, fontsize=8, color=color, ha="center", va="bottom" if "means" in name else "top")
    else:
        ax.text(last.k * 0.92, last.ratio + 0.004, name, fontsize=8, color=color, ha="right", va="bottom")
ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax.set_ylim(0.5, 1.02)
ax.set_xscale("log"); ax.set_xticks([10, 30, 96]); ax.set_xticklabels(["10", "30", "96"]); ax.minorticks_off()
ax.set_xlabel("number of parts", fontsize=8.5); ax.set_ylabel("resistance inside a part\nover resistance between adjacent parts", fontsize=8.5)
for a_ in axes:
    a_.tick_params(labelsize=8)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "figures" / "cluster_structure.png", dpi=200)
print("wrote", HERE / "figures" / "cluster_structure.png")

# --- 3. the gap of Peng, Sun & Zanetti 2015, eq. (1.3): Upsilon(k) = lambda_{k+1} / rho(k) --------------
# lambda: eigenvalues of the normalized Laplacian; rho(k) is at most the largest conductance of the parts of any
# k-way partition, which bounds Upsilon(k) from below; the higher-order Cheeger inequality rho(k) >= lambda_k / 2
# (Lee, Oveis Gharan & Trevisan 2014, quoted as their (1.2)) bounds it from above.
from gridnet.spectral import conductance, normalized_laplacian

lam_n = np.linalg.eigvalsh(normalized_laplacian(L, "sym"))
print("well-clustered? Upsilon(k) = lambda_{k+1} / rho(k) with rho(k) estimated by the largest conductance of the parts:")
for k in (2, 3, 10, 30, 96):
    lab = spectral_clustering(L, k, seed=1)
    phi = max(conductance(L, lab == m) for m in range(k))
    print(f"  k = {k:3d}: lambda_k = {lam_n[k - 1]:.5f}, lambda_k+1 = {lam_n[k]:.5f}; largest conductance of the parts {phi:.5f}; "
          f"{lam_n[k] / phi:.3f} <= Upsilon(k) <= 2 lambda_k+1 / lambda_k = {2 * lam_n[k] / lam_n[k - 1]:.3f} (rho(k) >= lambda_k / 2, their (1.2)); "
          f"their structure theorem asks for Upsilon = Omega(k^2), k^2 = {k**2}")
# at two parts the first cut of the recursive bisection (Shi & Malik 2000, Sec. 3.2), the Pyrenees cut,
# has a smaller conductance than the k-means partition and gives the better lower bound ("between 0.5 and 4.6")
from gridnet.spectral import recursive_bipartition

lab = recursive_bipartition(L, 2, normalized=True)
phi = max(conductance(L, lab == m) for m in range(2))
print(f"  k =   2, the first cut of the recursive bisection: parts of {np.bincount(lab).tolist()} buses, largest conductance {phi:.5f}; "
      f"{lam_n[2] / phi:.3f} <= Upsilon(2) <= {2 * lam_n[2] / lam_n[1]:.3f}")

# --- 4. the node and area parameters of Chow & Kokotovic 1985, Sec. II-A --------------------------------
from gridnet.spectral import area_parameters

print("Chow & Kokotovic 1985: node parameter d = c^E / c^I, area parameter delta = gamma^E / (m c^I); both should be << 1")
base_edges = list(zip(i.tolist(), j.tolist()))
area_rows = []
for k in (10, 30, 96):
    _, geo = kmeans2(planar_km(xy), k, minit="++", seed=1); _, geo = np.unique(geo, return_inverse=True)
    for name, lab in (("k-means on coordinates", geo), ("spectral clustering", spectral_clustering(L, k, seed=1))):
        area_rows.append(dict(k=int(lab.max() + 1), partition=name, **area_parameters(base_edges, lab)))
area_rows.append(dict(k=96, partition="PyPSA-Eur clusters", **area_parameters(base_edges, zone96)))
areas = pd.DataFrame(area_rows); areas.to_csv(RESULTS / "cluster_structure_area_parameters.csv", index=False)
print(areas.round(4).to_string(index=False))

# --- 5. the rigid aggregate against the slow modes, as in Chow & Kokotovic 1985, Sec. V and Table VII --------
# Their model: x'' = K x with unit inertias, K the negative Laplacian; the rigid aggregate M_a y'' = K_a y has the area
# sizes as inertias and the summed external connections as K_a (eq. (4.7)); modes are square roots of eigenvalues. On
# the WSCC system (2,161 nodes, 9 areas, all connections set to 1) the aggregate's modes are 174 % too fast on average
# and the compensated aggregate multiplies K_a by c = 1 / (1 + 1.74)^2 = 0.133 (eq. (4.9)).
print("rigid aggregate against the slow modes (Chow & Kokotovic 1985, Table VII: WSCC, 9 areas, mean error 174 %, c = 0.133):")
L_unit = np.diag((A > 0).sum(1).astype(float)) - (A > 0).astype(float)
agg_rows = []
for weights, Lw in (("unit connections", L_unit), ("susceptances", L)):
    lam_full = np.linalg.eigvalsh(Lw)
    for k, name, lab in [(kk, "spectral clustering", spectral_clustering(Lw, kk, seed=1)) for kk in (9, 30, 96)] + [(96, "PyPSA-Eur clusters", zone96)]:
        U = np.zeros((nb, k)); U[np.arange(nb), lab] = 1.0
        Ka, Ma = U.T @ Lw @ U, U.sum(0)
        agg = np.sort(np.linalg.eigvals(Ka / Ma[:, None]).real)[1:]          # nonzero eigenvalues of M_a^{-1} K_a
        slow = lam_full[1:k]
        err = np.mean(np.sqrt(agg) / np.sqrt(slow) - 1.0)
        agg_rows.append({"weights": weights, "k": k, "partition": name, "mean_mode_error_percent": 100 * err, "compensation_c": 1 / (1 + err) ** 2,
                             "d_average": area_parameters(base_edges, lab)["d_average"]})
agg = pd.DataFrame(agg_rows); agg.to_csv(RESULTS / "cluster_structure_rigid_aggregate.csv", index=False)
print(agg.round(3).to_string(index=False))
