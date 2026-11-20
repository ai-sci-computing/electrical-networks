"""Loukas's local variation coarsening as his code does it against the same algorithm as his paper defines it.

third_party/coarsening_paper.py is his coarsening_utils.py with five switchable changes (its docstring lists them
with the places in Loukas 2019): copy (no writing into the caller's eigenpairs), A (the later-level weights
B (B^T L B)^{+1/2}), B (the recursion with the averaging matrix), norm (the spectral norm in the local variation
cost), connected (pruned candidate sets split into connected pieces). This script switches them on one after the
other and reports what each changes, on his Minnesota example (his Table 1) and on the European base network at
96 parts and at milder reductions. With "copy" alone the patched module
must give exactly what coarsening.py gives with the installed package, which is checked.

Measures as in coarsening.py: the mean relative error of the first ten eigenvalues (Loukas's Table 1, the sum over
i = 2..10 divided by 10) of his matrix C L C^T and of the uniform quotient of the partition
(gridnet.spectral.coarsened_eigenvalues), and the restricted spectral error eps_10.

Run:  python studies/aggregation/european_grid/loukas_paper_check.py
Writes results/loukas_paper_check.log and results/loukas_paper_check.csv.
"""

import sys
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.sparse.linalg

for _name, _value in (("int", int), ("float", float)):            # aliases removed in NumPy 1.24, used by his code
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if not hasattr(sp, "newaxis"):
    sp.newaxis = np.newaxis

import pandas as pd
import pygsp
from graph_coarsening.coarsening_utils import coarsen as coarsen_installed
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent / "third_party"))
import coarsening_paper as cp

from gridnet.spectral import coarsened_eigenvalues, restricted_spectral_error

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
log = open(RESULTS / "loukas_paper_check.log", "w")
say = lambda *a: (print(*a), print(*a, file=log), log.flush())
K = 10
STEPS = [("his code (copies)", {"copy"}), ("+ A", {"copy", "A"}), ("+ A, B", {"copy", "A", "B"}),
         ("+ A, B, norm", {"copy", "A", "B", "norm"}), ("+ A, B, norm, connected", {"copy", "A", "B", "norm", "connected"})]


def his_matrix_error(C, L_sparse, lam):
    Lc = (C @ L_sparse @ C.T).tocsc(); Lc = (Lc + Lc.T) / 2
    lc = np.sort(sp.sparse.linalg.eigsh(Lc, k=K, sigma=-1e-3 * Lc.diagonal().max(), which="LM", return_eigenvectors=False))
    return float(np.abs(lc[1:] / lam[1:K] - 1).sum() / K)


def quotient_error(L, labels, lam):
    lc = coarsened_eigenvalues(L, labels, k=K, seed=0) if sp.sparse.issparse(L) else coarsened_eigenvalues(L, labels)
    return float(np.abs(lc[1:K] / lam[1:K] - 1).sum() / K)


def connected_parts(W, labels):
    """How many parts of the partition induce a disconnected subgraph (Definition 2.5 b asks for none)."""
    bad = 0
    for p in np.unique(labels):
        idx = np.flatnonzero(labels == p)
        if len(idx) > 1 and sp.sparse.csgraph.connected_components(W[idx][:, idx], directed=False)[0] > 1:
            bad += 1
    return bad


rows = []
# --- Minnesota, his Table 1 ----------------------------------------------------------------------------------
M = pygsp.graphs.Minnesota()
Gm = pygsp.graphs.Graph(M.W.astype(float), coords=M.coords)
offset = 2 * max(Gm.dw)
lk, Uk = sp.sparse.linalg.eigsh(offset * sp.sparse.eye(Gm.N, format="csc") - Gm.L, k=K, which="LM", tol=1e-6,
                                v0=np.random.default_rng(0).standard_normal(Gm.N))
lk, Uk = (offset - lk)[::-1].copy(), Uk[:, ::-1].copy()
Wm = sp.sparse.csr_matrix(Gm.W)
say(f"Minnesota, {Gm.N} vertices; published (Table 1): edges 0.088, 0.432, 4.579; neighborhoods 0.078, 0.310, 1.866")
for method in ("variation_edges", "variation_neighborhoods"):
    for r in (0.3, 0.5, 0.7):
        C_inst = coarsen_installed(Gm, K=K, r=r, max_levels=10, method=method, algorithm="greedy", Uk=Uk.copy(), lk=lk.copy())[0]
        for label, fixes in STEPS:
            cp.FIXES = set(fixes)
            C = cp.coarsen(Gm, K=K, r=r, max_levels=10, method=method, algorithm="greedy", Uk=Uk, lk=lk)[0]
            if fixes == {"copy"}:
                assert (abs(C - C_inst)).max() < 1e-12, "the patched module with copy alone differs from the installed package"
            lab = np.asarray(abs(C).argmax(axis=0)).ravel()
            rows.append({"graph": "Minnesota", "method": method, "reduction": r, "step": label, "parts": C.shape[0],
                             "error_his_matrix": his_matrix_error(C, Gm.L, lk), "error_quotient": quotient_error(Gm.L, lab, lk),
                             "disconnected_parts": connected_parts(Wm, lab)})
            q = rows[-1]
            say(f"  {method:24s} {int(100 * r)} % {label:26s}: his matrix {q['error_his_matrix']:.3f}, quotient {q['error_quotient']:.3f}, "
                f"{q['parts']} parts, {q['disconnected_parts']} disconnected")

# --- the European base network ---------------------------------------------------------------------------------
z = np.load(RESULTS / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
W = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
L = (sp.sparse.diags(np.asarray(W.sum(1)).ravel()) - W).toarray()
lam, V = np.linalg.eigh(L)
Uk_e, lk_e = V[:, :K].copy(), lam[:K].copy()
G = pygsp.graphs.Graph(W, coords=z["bus_xy"])
say(f"\nEuropean base network, {nb} buses (published: neighborhoods 96 parts, error 13.7, eps_10 7.5; edges stops at 273, 17.6, 8.5)")
for method, r in (("variation_neighborhoods", 1 - 96 / nb), ("variation_edges", 1 - 96 / nb), ("variation_neighborhoods", 0.5), ("variation_neighborhoods", 0.7)):
    for label, fixes in STEPS:
        cp.FIXES = set(fixes)
        C = cp.coarsen(G, K=K, r=r, max_levels=20, method=method, algorithm="greedy", Uk=Uk_e, lk=lk_e)[0]
        lab = np.asarray(abs(C).argmax(axis=0)).ravel()
        rows.append({"graph": "Europe", "method": method, "reduction": round(r, 4), "step": label, "parts": C.shape[0],
                         "largest_part": int(np.bincount(lab).max()), "error_his_matrix": his_matrix_error(C, sp.sparse.csc_matrix(L), lam),
                         "error_quotient": quotient_error(L, lab, lam), "eps_10": restricted_spectral_error(L, lab, K),
                         "disconnected_parts": connected_parts(W, lab)})
        q = rows[-1]
        say(f"  {method:24s} r = {r:.3f} {label:26s}: {q['parts']} parts (largest {q['largest_part']}), quotient error {q['error_quotient']:.2f}, "
            f"his matrix {q['error_his_matrix']:.2f}, eps_10 {q['eps_10']:.2f}, {q['disconnected_parts']} disconnected")
pd.DataFrame(rows).to_csv(RESULTS / "loukas_paper_check.csv", index=False)
