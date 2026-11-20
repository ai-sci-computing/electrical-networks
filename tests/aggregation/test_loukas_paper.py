"""Tests for studies/aggregation/european_grid/third_party/coarsening_paper.py, Loukas's coarsening code with the
changes that make it follow his paper (Loukas 2019, JMLR 20; audit of 2026-10-29, deviations 1 to
5). Each change is checked against the paper's definition computed independently, and his unchanged code against
the same definition where it deviates. Needs his package (the "coarsening" extra)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph

pytest.importorskip("graph_coarsening")
pygsp = pytest.importorskip("pygsp")
for _name, _value in (("int", int), ("float", float)):                  # aliases his code still uses
    if not hasattr(np, _name):
        setattr(np, _name, _value)
if not hasattr(sp, "newaxis"):
    sp.newaxis = np.newaxis

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies" / "aggregation" / "european_grid" / "third_party"))
import coarsening_paper as cp
from graph_coarsening.coarsening_utils import coarsen as coarsen_installed
from graph_coarsening.coarsening_utils import coarsen_matrix, get_coarsening_matrix


def ring(n, w=None):
    w = np.ones(n) if w is None else np.asarray(w, float)
    i = np.arange(n); j = (i + 1) % n
    return sp.sparse.csr_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n))


def laplacian(W):
    return (sp.sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W).toarray()


def first_eigenpairs(L, k):
    lam, U = np.linalg.eigh(L)
    return lam[:k].copy(), U[:, :k].copy()


def test_fix_copy_leaves_the_eigenpairs_untouched_and_his_code_does_not():
    """Deviation 1: his coarsen sets lk[lk < 1e-10] = 1 in the caller's array; with the fix the arrays are unchanged and
    the coarsening is the same as his with a fresh copy."""
    W = ring(12, 1 + np.arange(12) % 3)
    G = pygsp.graphs.Graph(W)
    lk, Uk = first_eigenpairs(laplacian(W), 4)
    lk_before = lk.copy()
    cp.FIXES = {"copy"}
    C = cp.coarsen(G, K=4, r=0.5, method="variation_edges", Uk=Uk, lk=lk)[0]
    assert np.array_equal(lk, lk_before)
    C_inst = coarsen_installed(G, K=4, r=0.5, method="variation_edges", Uk=Uk.copy(), lk=lk.copy())[0]
    assert abs(C - C_inst).max() < 1e-12
    shared = lk.copy()
    coarsen_installed(G, K=4, r=0.5, method="variation_edges", Uk=Uk.copy(), lk=shared)
    assert shared[0] == 1.0 and lk_before[0] < 1e-10                      # his code overwrote the zero eigenvalue


def test_averaging_matrix_keeps_the_laplacian_norm_of_piecewise_constant_vectors():
    """Deviation 3, Sec. 2.4 and Scheme 1: with the averaging P (entries 1/|V^(r)|) and the coarse Laplacian
    P^-+ L P^+ of summed weights, x_c = P x has ||x_c||_{L_c} = ||x||_L for every x constant on the contraction
    sets, which the proof of Prop. 4.1 uses at each level. His normalized matrix (entries 1/sqrt|V^(r)|) does not."""
    W = ring(9, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    G = pygsp.graphs.Graph(W)
    parts = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
    iC = get_coarsening_matrix(G, parts)
    P = cp.averaging_matrix(iC)
    np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1)
    np.testing.assert_allclose(sorted(set(np.round(P.data, 12))), [1 / 4, 1 / 3, 1 / 2])
    Lc = laplacian(sp.sparse.csr_matrix(coarsen_matrix(W, iC)))
    L = laplacian(W)
    x = np.zeros(9)
    for value, idx in zip((0.3, -1.2, 2.0), parts):
        x[idx] = value
    xc = P @ x
    np.testing.assert_allclose(xc @ Lc @ xc, x @ L @ x)
    xn = iC @ x
    assert abs(xn @ Lc @ xn - x @ L @ x) > 0.1                               # his normalized matrix


def test_later_level_weights_are_the_pseudoinverse_square_root():
    """Deviation 2, Sec. 4.1 and the proof of Prop. 4.1: A = B (B^T L B)^{+1/2} spans the same space as B and is
    L-orthonormal on it, A^T L A the orthogonal projection onto the range of B^T L B. His B D^{-1/2} V is not."""
    rng = np.random.default_rng(0)
    L = laplacian(ring(10, rng.uniform(0.5, 2, 10)))
    B = rng.standard_normal((10, 4))
    A = cp.later_level_weights(B, L)
    Q = A.T @ L @ A
    np.testing.assert_allclose(Q @ Q, Q, atol=1e-10)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(Q)), np.round(np.sort(np.linalg.eigvalsh(Q))), atol=1e-10)
    assert np.linalg.matrix_rank(np.c_[A, B], tol=1e-8) == np.linalg.matrix_rank(B, tol=1e-8)
    d, V = np.linalg.eig(B.T @ L @ B)
    A_his = B @ np.diag(d.real ** -0.5) @ V.real
    Q_his = A_his.T @ L @ A_his
    assert np.abs(Q_his @ Q_his - Q_his).max() > 1e-3


def test_local_variation_cost_against_its_definition():
    """Deviation 4, eq. 5 and 6: cost(C) = ||Pi_C^perp A||_{L_C}^2 / (|C| - 1) with the N x N Laplacian of W_C (the
    weights inside C, twice those leaving it, zero elsewhere) and Pi_C^perp x = x - mean over C on C, zero off C;
    the squared L_C-norm of a matrix is the spectral norm of A^T Pi L_C Pi A. His code takes the Frobenius norm."""
    rng = np.random.default_rng(1)
    n = 8
    W = ring(n, rng.uniform(0.5, 2, n)).tolil()
    W[0, 4] = W[4, 0] = 1.5
    W = W.tocsr()
    A = rng.standard_normal((n, 3))
    for C in ([0, 1], [0, 1, 7], [3, 4, 5, 0]):
        inC = np.isin(np.arange(n), C)
        Wd = W.toarray()
        WC = np.where(np.outer(inC, inC), Wd, 0) + 2 * np.where(np.outer(inC, ~inC) | np.outer(~inC, inC), Wd, 0)
        LC = np.diag(WC.sum(1)) - WC
        Pi = np.zeros((n, n))
        Pi[np.ix_(C, C)] = np.eye(len(C)) - 1 / len(C)
        expected = np.linalg.norm(A.T @ Pi @ LC @ Pi @ A, 2) / (len(C) - 1)
        np.testing.assert_allclose(cp.local_variation_cost(W, A, C), expected, rtol=1e-12)
        frob = np.linalg.norm(A.T @ Pi @ LC @ Pi @ A, "fro") / (len(C) - 1)
        np.testing.assert_allclose(cp.local_variation_cost(W, A, C, spectral=False), frob, rtol=1e-12)


def test_fix_connected_gives_connected_contraction_sets():
    """Deviation 5, Def. 2.5 b: every contraction set induces a connected subgraph. On the Minnesota road network at a
    reduction of 70 % by neighborhoods his code returns disconnected sets, the fixed code none."""
    M = pygsp.graphs.Minnesota()
    G = pygsp.graphs.Graph(M.W.astype(float), coords=M.coords)
    W = sp.sparse.csr_matrix(G.W)
    lk, Uk = sp.sparse.linalg.eigsh(2 * max(G.dw) * sp.sparse.eye(G.N, format="csc") - G.L, k=10, which="LM", tol=1e-6,
                                    v0=np.random.default_rng(0).standard_normal(G.N))
    lk, Uk = (2 * max(G.dw) - lk)[::-1].copy(), Uk[:, ::-1].copy()

    def disconnected(C):
        lab = np.asarray(abs(C).argmax(axis=0)).ravel()
        return sum(sp.sparse.csgraph.connected_components(W[idx][:, idx], directed=False)[0] > 1
                   for idx in (np.flatnonzero(lab == p) for p in np.unique(lab)) if len(idx) > 1)

    cp.FIXES = {"copy"}
    assert disconnected(cp.coarsen(G, K=10, r=0.7, method="variation_neighborhoods", Uk=Uk, lk=lk)[0]) > 0
    cp.FIXES = {"copy", "A", "B", "norm", "connected"}
    assert disconnected(cp.coarsen(G, K=10, r=0.7, method="variation_neighborhoods", Uk=Uk, lk=lk)[0]) == 0
