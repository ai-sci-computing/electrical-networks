"""The Iberian cut as a distance: the reciprocal of the cut against the distance between the two sides as clusters.

For the two sides of the first sweep cut of the continental component (continental_laplacian.py) the rigid aggregate
is one branch whose susceptance is the cut, so its distance is 1 / cut (Chow & Kokotovic 1985, eq. (4.7); Lyons & Peres
2016, Exercise 2.5). The distance with a distribution p on the Iberian side and q on the other is E(p - q) =
(p - q)^T L^+ (p - q) (the complete aggregate, eqs. (4.3), (4.4)), which expands into the mean resistance between the
sides less half the mean resistance inside each (Sejdinovic et al. 2013, Definition 5). The difference between the two
numbers is the part of the separation that lies inside the two sides and not in the border.

Run: .venv/bin/python studies/aggregation/european_grid/pyrenees_distance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix

from gridnet.spectral import cut_weight, fiedler, sweep_cut

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
z = np.load(HERE / "results" / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).toarray()
L = np.diag(A.sum(1)) - A

_, v = fiedler(L, normalized=True)
S, h = sweep_cut(L, v, objective="conductance")
S = S if S.sum() < nb / 2 else ~S
cut = cut_weight(L, S)
print(f"Iberian side {int(S.sum())} buses, cut {cut:.3f} p.u. over {int(((S[i] != S[j])).sum())} branches, conductance {h:.2e}")
assert int(S.sum()) == 1223

G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])          # grounded inverse, exact on zero-sum vectors
demand = xr.open_dataset(ROOT / "data" / "raw" / "pypsa-eur-gridnet" / "electricity_demand_simplified.nc")["electricity demand (MW)"].mean("time").to_series()
simplified = pd.Series(z["simplified_bus"])
load = (simplified.map(demand).fillna(0.0) / simplified.map(simplified.value_counts()).fillna(1.0)).values

rows = []
for name, w in (("uniform", np.ones(nb)), ("demand", load)):
    p = np.where(S, w, 0.0); p /= p.sum()
    q = np.where(~S, w, 0.0); q /= q.sum()
    between = 2 * (-(p @ G @ q)) + p @ np.diag(G) + q @ np.diag(G)     # sum p_i q_j R_ij with R_ij = G_ii + G_jj - 2 G_ij
    inside_p = 2 * (p @ np.diag(G)) - 2 * (p @ G @ p)                  # sum p_i p_i' R_ii'
    inside_q = 2 * (q @ np.diag(G)) - 2 * (q @ G @ q)
    E = (p - q) @ G @ (p - q)
    assert np.isclose(E, between - inside_p / 2 - inside_q / 2)
    rows.append({"distribution": name, "distance": E, "mean_between": between, "mean_inside_iberia": inside_p, "mean_inside_rest": inside_q,
                     "rigid": 1 / cut, "rigid_over_distance": (1 / cut) / E})
out = pd.DataFrame(rows); out.to_csv(HERE / "results" / "pyrenees_distance.csv", index=False)
print(out.round(5).to_string(index=False))
