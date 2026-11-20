"""PyPSA-Eur's clustered network against the reference that matches a clustering, and its length convention as a
compensation of the rigid aggregate.

kron_vs_cluster.py holds the clustered network against the Kron reduction onto one representative bus per cluster.
A clustering, which sums the injections of a cluster on its node, belongs to the other
convention: the node stands for the mean over the cluster's buses, and the exact reduction in that convention is the
complete aggregate of Chow & Kokotovic 1985, eqs. (4.3), (4.4) (gridnet.reweight.zonal_operator). The two are members of
one family, indexed by how a zone's injection is distributed over its buses (participation factors, Cheng & Overbye 2005,
p. 1869): all weight on one bus gives the Kron reduction, PyPSA-Eur's own weights are the demands it sums. This script

1. repeats the comparison of effective resistances of kron_vs_cluster.py against both references, for PyPSA-Eur's
   clustered network and for the rigid aggregate (the summed susceptances, their eq. (4.7));
2. states PyPSA-Eur's centroid-length convention (kron_vs_cluster.py: aggregated reactances a median 3.4 times the
   parallel combination of the replaced lines) in Chow & Kokotovic's terms: the error of the aggregate's modes against
   the slow modes of the base network (their Table VII) and the compensation factor c of their eq. (4.9).

Run: .venv/bin/python studies/aggregation/european_grid/matched_reference.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gridnet.graph import effective_resistance
from gridnet.kron import laplacian_from_couplings
from gridnet.reweight import zonal_operator

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
_log = open(RESULTS / "matched_reference.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/matched_reference.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()
z = np.load(RESULTS / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
Bc, W = z["Bc"], z["W"]
k = len(Bc)
A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).toarray()
L = np.diag(A.sum(1)) - A
zone = z["zone"].copy(); lost = np.flatnonzero(zone < 0)
X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
dd = dijkstra(X, directed=False, indices=lost); dd[:, lost] = np.inf
zone[lost] = zone[np.argmin(dd, axis=1)]
U = np.zeros((nb, k)); U[np.arange(nb), zone] = 1.0
sizes = U.sum(0)

def resistance(Lz):
    G = np.linalg.pinv(Lz); d = np.diag(G)
    return d[:, None] + d[None, :] - 2 * G


# the weights that describe PyPSA-Eur: it puts the summed demand of a cluster on its node, so a base bus takes part in
# its zone's injection in proportion to its demand. Mean hourly demand per bus of the simplified network (the file
# PyPSA-Eur's clustering consumes), carried to the base buses through busmap_simplify_network, shared equally among the
# base buses merged into one simplified bus; base buses without demand get weight zero.
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
demand = xr.open_dataset(ROOT / "data" / "raw" / "pypsa-eur-gridnet" / "electricity_demand_simplified.nc")["electricity demand (MW)"].mean("time").to_series()
simplified = pd.Series(z["simplified_bus"])
share = simplified.map(demand).fillna(0.0) / simplified.map(simplified.value_counts()).fillna(1.0)
load = share.values.copy()
for zn in np.flatnonzero(np.bincount(zone, weights=load, minlength=k) == 0):   # a zone without recorded demand: equal weights
    load[zone == zn] = 1.0
print(f"demand carried to the base network: {load.sum() / 1e3:.1f} GW mean, on {int((share > 0).sum())} of {nb} buses; largest share of one bus in its zone, median over zones: "
      f"{np.median([load[zone == zn].max() / load[zone == zn].sum() for zn in range(k)]):.2f}")

onehot = np.zeros(nb); onehot[z["keep"]] = 1.0
references = {"Kron reduction onto representative buses": effective_resistance(laplacian_from_couplings(W)),
              "complete aggregate, equal weights": resistance(zonal_operator(L, zone)),
              "complete aggregate, weights = demand": resistance(zonal_operator(L, zone, weights=load))}
pairs = np.triu_indices(k, 1)
check = np.abs(resistance(zonal_operator(L, zone, weights=onehot))[pairs] / references["Kron reduction onto representative buses"][pairs] - 1).max()
print(f"check: the complete aggregate with all weight on the representative buses is the Kron reduction, largest relative deviation of a resistance {check:.1e}")
networks = {"PyPSA-Eur's clustered network": laplacian_from_couplings(Bc), "rigid aggregate (summed susceptances)": U.T @ L @ U}
iu = np.triu_indices(k, 1); adjacent = (Bc > 0)[iu]

print(f"1. effective resistances, {k} clusters, all pairs. The references relative to the Kron reduction, quartiles:")
for ref_name in list(references)[1:]:
    r = references[ref_name][iu] / references["Kron reduction onto representative buses"][iu]
    print(f"   {ref_name:40s} {np.quantile(r, [.25, .5, .75]).round(3)}, adjacent pairs median {np.median(r[adjacent]):.3f}")
rows = []
for name, La in networks.items():
    R = effective_resistance(La)
    for ref_name, ref in references.items():
        q = R[iu] / ref[iu]
        rows.append({"network": name, "reference": ref_name, "q25": np.quantile(q, .25), "median": np.median(q), "q75": np.quantile(q, .75),
                         "median_adjacent": np.median(q[adjacent]), "log_correlation": np.corrcoef(np.log(R[iu]), np.log(ref[iu]))[0, 1]})
out = pd.DataFrame(rows); out.to_csv(RESULTS / "matched_reference.csv", index=False)
print(out.round(3).to_string(index=False))

print("\n2. modes of the aggregates against the slow modes of the base network (Chow & Kokotovic 1985, Table VII: WSCC, 9 areas, 174 %, c = 0.133)")
slow = np.linalg.eigvalsh(L)[1:k]
modes = {name: np.sort(np.linalg.eigvalsh(La / np.sqrt(np.outer(sizes, sizes))))[1:] for name, La in networks.items()}
for name, agg in modes.items():
    err = np.sqrt(agg / slow) - 1
    print(f"   {name:40s} mean mode error {100 * err.mean():6.1f} %, median {100 * np.median(err):6.1f} %, ten slowest modes {100 * err[:10].mean():6.1f} %, c = {1 / (1 + err.mean()) ** 2:.3f}, for the ten slowest {1 / (1 + err[:10].mean()) ** 2:.3f}")
factor = np.median(modes["PyPSA-Eur's clustered network"] / modes["rigid aggregate (summed susceptances)"])
stretch = pd.read_csv(RESULTS / "kron_vs_cluster_length_convention.csv").x_ratio.median()      # per pair of clusters, kron_vs_cluster.py
print(f"   PyPSA-Eur's network acts as the rigid aggregate with its connections multiplied by {factor:.3f} (median over the modes); "
      f"one over the median stretch of the length convention, 1 / {stretch:.2f} = {1 / stretch:.3f}")

# --- 3. how the comparison depends on the concentration of a cluster's weights ---------------------------------
# A node also carries generation, which sits at fewer buses than demand. Equal weight on m random buses per cluster,
# from the Kron end (m = 1) to the complete aggregate with equal weights (all buses).
print("\n3. PyPSA-Eur's network over the reference (median of the resistance ratios), by the number of buses that carry a cluster's weight")
G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])             # grounded inverse, exact for balanced vectors such as p_a - p_c
R_net = effective_resistance(networks["PyPSA-Eur's clustered network"])


def reference_resistance(w):
    P = np.zeros((nb, k)); P[np.arange(nb), zone] = w; P /= P.sum(0, keepdims=True)
    M = P.T @ G @ P; d = np.diag(M)
    return d[:, None] + d[None, :] - 2 * M


rng = np.random.default_rng(0)
conc = []
for m in (1, 2, 3, 5, 10, 20):
    meds = []
    for _ in range(8):
        w = np.zeros(nb)
        for zn in range(k):
            idx = np.flatnonzero(zone == zn); w[rng.choice(idx, size=min(m, len(idx)), replace=False)] = 1.0
        meds.append(float(np.median(R_net[iu] / reference_resistance(w)[iu])))
    conc.append({"buses_per_cluster": m, "median_ratio": np.median(meds), "lowest": min(meds), "highest": max(meds)})
conc.append({"buses_per_cluster": "all, equal", "median_ratio": float(np.median(R_net[iu] / reference_resistance(np.ones(nb))[iu]))})
conc.append({"buses_per_cluster": "all, demand", "median_ratio": float(np.median(R_net[iu] / reference_resistance(load)[iu]))})
for m in (1, 3, 5):
    w = np.zeros(nb)
    for zn in range(k):
        idx = np.flatnonzero(zone == zn); top = idx[np.argsort(load[idx])[::-1][:m]]; w[top] = load[top]
    conc.append({"buses_per_cluster": f"demand, {m} largest", "median_ratio": float(np.median(R_net[iu] / reference_resistance(w)[iu]))})
conc = pd.DataFrame(conc); conc.to_csv(RESULTS / "matched_reference_concentration.csv", index=False)
print(conc.round(3).to_string(index=False))
eff = [1 / np.sum((load[zone == zn] / load[zone == zn].sum()) ** 2) for zn in range(k)]
print(f"effective number of buses carrying a cluster's demand, 1 / sum of squared shares: median {np.median(eff):.1f}, quartiles {np.quantile(eff, .25):.1f} to {np.quantile(eff, .75):.1f}; median cluster size {int(np.median(np.bincount(zone)))}")

# "Leaving out the pairs that involve these two clusters moves the median below by less than one percent":
# the clusters that take the buses of Ukraine and Moldova, which PyPSA-Eur's run leaves out
buses = pd.read_csv(ROOT / "data" / "raw" / "pypsa-eur-osm-0.7" / "buses.csv", quotechar="'", usecols=["bus_id", "country"])
country = pd.Series(z["bus_id"]).map(buses.set_index("bus_id")["country"]).values
east = np.unique(zone[lost[np.isin(country[lost], ["UA", "MD"])]])
keep_pair = ~np.isin(iu[0], east) & ~np.isin(iu[1], east)
q = R_net[iu] / references["complete aggregate, weights = demand"][iu]
print(f"buses of Ukraine and Moldova: {int(np.isin(country, ['UA', 'MD']).sum())}, all without a cluster in PyPSA-Eur's run: {bool(np.all(z['zone'][np.isin(country, ['UA', 'MD'])] < 0))}; "
      f"added to clusters {east.tolist()}; median ratio against demand weights {np.median(q):.4f} over all {len(q)} pairs, "
      f"{np.median(q[keep_pair]):.4f} over the {int(keep_pair.sum())} pairs without these clusters, change {100 * (np.median(q[keep_pair]) / np.median(q) - 1):+.2f} %")
