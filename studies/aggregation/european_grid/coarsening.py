"""Coarsening with spectral guarantees (Loukas 2019) on the European base network, against other partitions.

Loukas's coarsened Laplacian for a partition is the quotient graph, L_c = P^-+ L P^+ = U^T L U (his Prop. 2.2), the
rigid aggregate of Chow & Kokotovic 1985. What he adds is a measure, the constant eps_k of restricted spectral
approximation on the span of the first k eigenvectors (his Def. 3.1), algorithms that choose the contraction sets to
keep it small (local variation, his Sec. 4), and guarantees in terms of it (Thms. 3.3-3.5).

1. His public code (github.com/loukasa/graph-coarsening, Apache 2.0, installed from source with
   `pip install git+https://github.com/loukasa/graph-coarsening@1d7aad02ef29`, plus pygsp and sortedcontainers) reproduces his
   Table 1 for the Minnesota road network here. It was written for NumPy < 1.24 and PyGSP 0.5; three aliases that newer
   NumPy and SciPy removed are restored below, which does not touch the algorithm. Its evaluation function computes the
   coarse eigenvalues with eigsh(which='SM', tol=1e-3), which in about one call in ten misses the zero eigenvalue and
   compares all others one index off; the eigenvalues here come from gridnet.spectral.coarsened_eigenvalues
   (shift-invert), and the failure rate of his call is printed once.
2. The European base network, 5,577 buses, coarsened to about 96 nodes by his two local variation algorithms as his
   paper defines them (third_party/coarsening_paper.py with all five changes that make it follow the paper: no writing
   into the eigenpairs, the later-level weights B (B^T L B)^{+1/2}, the averaging matrix in their recursion, the
   spectral norm in the cost, connected contraction sets; his unchanged code on the same
   grid in loukas_paper_check.py), and the same measures for PyPSA-Eur's clusters, k-means on the coordinates and
   spectral clustering.

Prediction written before the first run (2026-09-19): eps_k is close to one for every partition (pairs on a ring
already give 0.9); his algorithms give the smallest eigenvalue errors, which are still of order one at a reduction
of 98 % (his Table 1 has errors of 2 to 9 at 70 %).

Run: .venv/bin/python studies/aggregation/european_grid/coarsening.py
"""

from pathlib import Path

import numpy as np
import scipy as sp

for _name, _value in (("int", int), ("float", float)):            # aliases removed in NumPy 1.24, used by graph_coarsening
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if not hasattr(sp, "newaxis"):                                     # removed from SciPy, used by graph_coarsening
    sp.newaxis = np.newaxis

import sys as _sys

import pandas as pd
import pygsp
from graph_coarsening.coarsening_utils import coarsen, coarsening_quality
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

_sys.path.insert(0, str(Path(__file__).parent / "third_party"))
import coarsening_paper

from gridnet.pypsa_io import planar_km
from gridnet.spectral import (
    area_parameters,
    coarsened_eigenvalues,
    restricted_spectral_error,
    spectral_clustering,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
_log = open(RESULTS / "coarsening.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/coarsening.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()
K = 10

# --- 1. his Table 1, Minnesota, k = 10 -------------------------------------------------------------------------
M = pygsp.graphs.Minnesota()
Gm = pygsp.graphs.Graph(M.W.astype(float), coords=M.coords)       # as his graph_lib.real('minnesota'), with PyGSP 0.6's argument name
offset = 2 * max(Gm.dw)
lk, Uk = sp.sparse.linalg.eigsh(offset * sp.sparse.eye(Gm.N, format="csc") - Gm.L, k=K, which="LM", tol=1e-6,
                                v0=np.random.default_rng(0).standard_normal(Gm.N))   # a fixed start: ARPACK's own is random per process
lk, Uk = (offset - lk)[::-1], Uk[:, ::-1]
lk.setflags(write=False); Uk.setflags(write=False)                # his functions write into their inputs; every call gets a copy
print(f"Minnesota, {Gm.N} vertices, {Gm.Ne} edges; mean relative error of the first {K} eigenvalues at reductions of 30, 50, 70 %")
def eigenvalue_error(lam_c, lam_ref):
    """His Table 1: (1/k) sum_i |lambda_i - lambda~_i| / lambda_i over the first k eigenvalues, the first term zero."""
    return float(np.abs(lam_c[1:len(lam_ref)] / lam_ref[1:] - 1).sum() / len(lam_ref))


def eigenvalues_of_his_matrix(C, L_sparse, k):
    """The k smallest eigenvalues of C L C^T for his (multilevel) coarsening matrix C, by shift-invert."""
    Lc = (C @ L_sparse @ C.T).tocsc(); Lc = (Lc + Lc.T) / 2
    return np.sort(sp.sparse.linalg.eigsh(Lc, k=k, sigma=-1e-3 * Lc.diagonal().max(), which="LM", return_eigenvectors=False))


print("  columns: published | his multilevel matrix C, eigenvalues by shift-invert | uniform quotient of the same partition | levels, largest ratio of weights within a row of C")
for method, published in (("variation_edges", [0.088, 0.432, 4.579]), ("variation_neighborhoods", [0.078, 0.310, 1.866])):
    for r, pub in zip((0.3, 0.5, 0.7), published):
        # his coarsen sets lk[lk < 1e-10] = 1 in the array it is given (coarsening_utils.py, level 1), so a shared lk
        # reaches every later call with the zero eigenvalue read as one; each call gets its own copy
        C, _, Call, _ = coarsen(Gm, K=K, r=r, max_levels=10, method=method, algorithm="greedy", Uk=Uk.copy(), lk=lk.copy())
        labels = np.asarray(abs(C).argmax(axis=0)).ravel()
        rows_C = C.tocsr(); spread = max(rows_C[q].data.max() / rows_C[q].data.min() for q in range(rows_C.shape[0]))
        print(f"  {method:24s} {int(100 * r)} %: {pub:.3f} | {eigenvalue_error(eigenvalues_of_his_matrix(C, Gm.L, K), lk):.3f} | "
              f"{eigenvalue_error(coarsened_eigenvalues(Gm.L, labels, k=K, seed=0), lk):.3f} | {len(Call)}, {spread:.2f}")
        if method == "variation_neighborhoods" and r == 0.5:
            reference = eigenvalue_error(eigenvalues_of_his_matrix(C, Gm.L, K), lk)
            calls = [float(np.nanmean(coarsening_quality(Gm, C, kmax=K, Uk=Uk.copy(), lk=lk.copy())["error_eigenvalue"])) for _ in range(20)]
off_calls = [c for c in calls if abs(c - reference) > 1e-2]
# the reading that matches his table: the six calls in the order of his table with one shared lk, as the first version
# of this script made them; after the first call the zero eigenvalue is read as one (see above)
shared_lk, shared = lk.copy(), []
for method in ("variation_edges", "variation_neighborhoods"):
    for r in (0.3, 0.5, 0.7):
        C, _, _, _ = coarsen(Gm, K=K, r=r, max_levels=10, method=method, algorithm="greedy", Uk=Uk.copy(), lk=shared_lk)
        shared.append(eigenvalue_error(eigenvalues_of_his_matrix(C, Gm.L, K), lk))
print("  with one lk shared by the six calls in this order (his array overwritten after the first): " + ", ".join(f"{e:.3f}" for e in shared))
# which reading is his method: Loukas 2019, Sec. 4.1, Remark, B_0 = U_k Lambda_k^{+1/2}, the pseudoinverse square root, so
# the column of the zero eigenvalue is zero. With lk[0] overwritten to one the constant vector enters B_0, and at the
# levels after the first, where his code inverts the eigenvalues of B^T L B that are not exactly zero, it is kept with
# a small eigenvalue and a large weight. The two smallest |eigenvalues| of B^T L B at levels 2 and 3, edges at 70 %:
import graph_coarsening.coarsening_utils as _cu

_seen, _eig = [], np.linalg.eig
def _eig_spy(Mx):
    d, V = _eig(Mx); _seen.append(np.sort(np.abs(d))[:2]); return d, V
_cu.np.linalg.eig = _eig_spy
for label, lk_in in (("each call its own copy", lk.copy()), ("lk[0] overwritten to one", np.r_[1.0, lk[1:]])):
    _seen.clear()
    C, *_ = coarsen(Gm, K=K, r=0.7, max_levels=10, method="variation_edges", algorithm="greedy", Uk=Uk.copy(), lk=lk_in)
    print(f"  {label:26s}: error {eigenvalue_error(eigenvalues_of_his_matrix(C, Gm.L, K), lk):.3f}; two smallest |eigenvalues| of B^T L B "
          f"at the later levels {[np.round(x, 3).tolist() for x in _seen]}")
_cu.np.linalg.eig = _eig
print(f"  his own evaluation function (eigsh which='SM', tol=1e-3) on the 50 % coarsening, 20 calls: {len(off_calls)} differ from {reference:.3f}"
      + (f", e.g. {off_calls[0]:.3f}: the zero eigenvalue is missed and the others are compared one index off" if off_calls else ""))

# --- 2. the European base network ------------------------------------------------------------------------------
z = np.load(RESULTS / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"]); xy = z["bus_xy"]
W = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
L = (sp.sparse.diags(np.asarray(W.sum(1)).ravel()) - W).toarray()
lam, V = np.linalg.eigh(L)
# The first K eigenpairs handed to his coarsen, as for Minnesota above. Without them his first level calls
# eigsh(2 d_max I - L, which='LM', tol=1e-5) from an unseeded ARPACK start: here 2 d_max = 70,686 against
# lambda_10 = 0.67, so the tolerance allows errors of the size of the eigenvalues themselves; in three calls
# (2026-10-24) the zero eigenvalue came out as 1.6e-7, 1.2e-7 and 3.4e-4, never below his mask of 1e-10, so the
# constant vector entered with the weight lambda^(-1/2), lambda_7 to lambda_10 were 30 to 150 % off, and every
# process contracted another partition (np.random.seed does not reach ARPACK). With the exact pairs the
# partitions are identical from process to process.
Uk, lk = V[:, :K], lam[:K].copy()
zone96 = z["zone"].copy(); lost = np.flatnonzero(zone96 < 0)
X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
dd = dijkstra(X, directed=False, indices=lost); dd[:, lost] = np.inf
zone96[lost] = zone96[np.argmin(dd, axis=1)]

partitions = {"PyPSA-Eur clusters": zone96}
_, geo = kmeans2(planar_km(xy), 96, minit="++", seed=1); partitions["k-means on coordinates"] = np.unique(geo, return_inverse=True)[1]
partitions["spectral clustering"] = spectral_clustering(L, 96, seed=1)
G = pygsp.graphs.Graph(W, coords=xy)
his_matrices = {}
for method in ("variation_neighborhoods", "variation_edges"):
    np.random.seed(0)          # seeded 2026-09-27 in the belief that his local variation draws random numbers; the variation came from ARPACK (above)
    coarsening_paper.FIXES = {"copy", "A", "B", "norm", "connected"}
    C, Gc, *_ = coarsening_paper.coarsen(G, K=K, r=1 - 96 / nb, max_levels=20, method=method, algorithm="greedy", Uk=Uk, lk=lk.copy())
    partitions[f"Loukas, local variation ({method.split('_')[1]})"] = np.asarray(abs(C).argmax(axis=0)).ravel()
    his_matrices[f"Loukas, local variation ({method.split('_')[1]})"] = C

# milder reductions, the range of his paper: his neighborhood algorithm against k-means on the coordinates with as many parts
mild = []
for r in (0.5, 0.7, 0.9, 0.95):
    np.random.seed(0)
    coarsening_paper.FIXES = {"copy", "A", "B", "norm", "connected"}
    C, *_ = coarsening_paper.coarsen(G, K=K, r=r, max_levels=20, method="variation_neighborhoods", algorithm="greedy", Uk=Uk, lk=lk.copy())
    lab = np.asarray(abs(C).argmax(axis=0)).ravel(); n_parts = int(lab.max()) + 1
    _, geo_r = kmeans2(planar_km(xy), n_parts, minit="++", seed=1); geo_r = np.unique(geo_r, return_inverse=True)[1]
    for name, l in (("Loukas, local variation (neighborhoods)", lab), ("k-means on coordinates", geo_r)):
        lc = coarsened_eigenvalues(L, l)
        mild.append({"reduction": r, "partition": name, "parts": int(l.max()) + 1, "largest_part": int(np.bincount(l).max()),
                         # Loukas's Table 1: the sum over the first K eigenvalues divided by K, the zero eigenvalue's term being zero (as eigenvalue_error above)
                         "eigenvalue_error_10": float(np.abs(lc[1:K] / lam[1:K] - 1).sum() / K), "eps_10": restricted_spectral_error(L, l, K),
                         "eigenvalue_error_10_his_matrix": (float(np.abs(eigenvalues_of_his_matrix(C, sp.sparse.csc_matrix(L), K)[1:] / lam[1:K] - 1).sum() / K) if name.startswith("Loukas") else np.nan)})
mild = pd.DataFrame(mild); mild.to_csv(RESULTS / "coarsening_mild.csv", index=False)
print("\nEuropean base network at milder reductions:")
print(mild.round(3).to_string(index=False))

rows = []
base_edges = list(zip(i.tolist(), j.tolist()))
for name, labels in partitions.items():
    k_parts = int(labels.max()) + 1
    lam_c = coarsened_eigenvalues(L, labels)
    sizes = np.bincount(labels)
    rows.append({"partition": name, "parts": k_parts, "largest_part": int(sizes.max()), "smallest_part": int(sizes.min()),
                     "eps_2": restricted_spectral_error(L, labels, 2), "eps_10": restricted_spectral_error(L, labels, K),
                     "eigenvalue_error_10": float(np.abs(lam_c[1:K] / lam[1:K] - 1).sum() / K),
                     "lambda2_ratio": float(lam_c[1] / lam[1]), "d_average": area_parameters(base_edges, labels)["d_average"],
                     "eigenvalue_error_10_his_matrix": (float(np.abs(eigenvalues_of_his_matrix(his_matrices[name], sp.sparse.csc_matrix(L), K)[1:] / lam[1:K] - 1).sum() / K) if name in his_matrices else np.nan)})
out = pd.DataFrame(rows); out.to_csv(RESULTS / "coarsening.csv", index=False)
print(f"\nEuropean base network, {nb} buses, first {K} eigenvalues; eigenvalue error = (1/{K}) sum_(i = 2..{K}) |lambda~_i / lambda_i - 1|")
print(out.round(3).to_string(index=False))
