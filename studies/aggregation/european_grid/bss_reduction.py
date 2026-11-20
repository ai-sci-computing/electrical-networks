"""The deterministic sparsifier of Batson, Spielman & Srivastava 2012 on the exact reduction of the continental grid.

gridnet.sparsify.bss_sparsify (Thm. 1.1 and Sec. 3.2 of arXiv:0808.0163v3) applied to the Kron reduction onto 96
buses (results/kron128.npz), for numbers of edges d (n - 1) that span the range of results/sparse_reductions.csv.
For each d: edges, the spectral error against the exact reduction, the guarantee 2 sqrt(d)/(d + 1) of the theorem in
the two-sided form, and the quantiles of the effective resistances relative to the exact ones.
Writes results/bss_reduction.csv.   Run: .venv/bin/python studies/aggregation/european_grid/bss_reduction.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridnet.graph import effective_resistance
from gridnet.kron import laplacian_from_couplings
from gridnet.pypsa_io import great_circle_km
from gridnet.sparsify import bss_sparsify, spectral_error

HERE = Path(__file__).parent
z = np.load(HERE / "results" / "kron128.npz")
L = laplacian_from_couplings(z["W"])
n = len(L); iu = np.triu_indices(n, 1)
R = effective_resistance(L)[iu]
rows = []
for d in (1.5, 2.1, 3.0, 4.7, 6.5, 10.0, 14.3, 20.0, 30.0, 50.0, 80.0, 120.0):
    t0 = time.time()
    H = bss_sparsify(L, d)
    q = effective_resistance(H)[iu] / R
    # independent check of the error: the extreme generalized eigenvalues of (H, L) on the complement of the constants
    from scipy.linalg import eigh
    P = np.eye(n)[:, 1:] - 1.0 / n
    mu = eigh(P.T @ H @ P, P.T @ L @ P, eigvals_only=True)
    assert abs(max(abs(mu - 1)) - spectral_error(L, H)) < 1e-8 and mu[-1] / mu[0] <= (d + 1 + 2 * np.sqrt(d)) / (d + 1 - 2 * np.sqrt(d))
    rows.append({"d": d, "edges": int(np.count_nonzero(np.triu(H, 1))), "eps": spectral_error(L, H), "guarantee": 2 * np.sqrt(d) / (d + 1),
                     "R_q05": np.quantile(q, .05), "R_median": np.median(q), "R_q95": np.quantile(q, .95), "seconds": time.time() - t0})
out = pd.DataFrame(rows); out.to_csv(HERE / "results" / "bss_reduction.csv", index=False)
print(out.round(3).to_string(index=False))

# --- how long the edges are: the selection against a sampled network (the run of median error among ten seeds) ------
from gridnet.sparsify import sparsify

xy = z["bus_xy"][z["keep"]]


runs = sorted((spectral_error(L, sparsify(L, 1200, seed=s)), s) for s in range(10))
seed = runs[len(runs) // 2][1]
for name, H in (("selection, d = 3", bss_sparsify(L, 3.0)), (f"sampling, q = 1200, seed {seed}", sparsify(L, 1200, seed=seed))):
    i, j = np.nonzero(np.triu(-H, 1) > 0)
    length = np.array([great_circle_km(xy[a], xy[b]) for a, b in zip(i, j)])
    print(f"{name}: {len(i)} edges, spectral error {spectral_error(L, H):.2f}, {int((z['Bc'][i, j] > 0).sum())} of them join neighboring clusters, "
          f"{int((length > 800).sum())} longer than 800 km, longest {length.max():.0f} km")
