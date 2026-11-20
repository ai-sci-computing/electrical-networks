"""Tests for gridnet.sparsify: spectral sparsification by effective-resistance sampling.

Reference: Spielman & Srivastava, SIAM J. Comput. 40(6) (2011), arXiv:0803.0929:
  the algorithm H = Sparsify(G, q)                                        Sec. 1.1
  guarantee (1 - eps) x^T L x <= x^T L~ x <= (1 + eps) x^T L x            Thm. 1, eq. (2)
  eigenvalues within factors 1 -+ eps by Courant-Fischer                  after Thm. 1
  the sampled Laplacian is unbiased, E w~_e = w_e                         Sec. 3
Closed forms: the sampling weights w_e R_e are 1 on every edge of a tree and 2/n on every edge
of the complete graph with unit weights (R_e = 2/n), and they sum to n - c (Foster's theorem,
tr(L^+ L) = n - c).

Approximate Schur complement: Durfee, Kyng, Peebles, Rao & Sachdeva, STOC 2017, arXiv:1611.07451,
Algorithm 3 (ApproxSchur) and Thm. 2.2, built on CliqueSample of Kyng & Sachdeva, FOCS 2016,
arXiv:1605.02353, Algorithm 2. Closed forms for the expectation of one elimination: the star-mesh
transform w_i w_j / sum(w) and the series rule (both from the Schur complement, `gridnet.kron`).
"""

import numpy as np
import pytest

from gridnet.graph import laplacian
from gridnet.kron import kron_reduce
from gridnet.sparsify import (
    approx_schur,
    bss_sparsify,
    leverage_scores,
    sparsify,
    spectral_error,
    spectral_error_best_scale,
)


def complete(n, w=1.0):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return laplacian(n, edges, [w] * len(edges)), edges


def test_leverage_scores_on_a_tree_are_one():
    L = laplacian(5, [(0, 1), (1, 2), (1, 3), (3, 4)], [2.0, 0.5, 3.0, 1.0])
    edges, scores = leverage_scores(L)
    assert len(edges) == 4 and np.allclose(scores, 1.0)


def test_leverage_scores_on_the_complete_graph_and_fosters_sum():
    n = 7
    L, _ = complete(n, w=3.0)
    edges, scores = leverage_scores(L)
    assert len(edges) == n * (n - 1) // 2
    assert np.allclose(scores, 2.0 / n)                          # independent of the common weight
    assert scores.sum() == pytest.approx(n - 1)
    rng = np.random.default_rng(0)                               # Foster on a random weighted graph, two components
    A = np.triu(rng.uniform(0.1, 2.0, (6, 6)) * (rng.random((6, 6)) < 0.6), 1)
    A[0, 1] = 1.0                                                # at least one edge
    big = np.zeros((9, 9)); big[:6, :6] = A; big[6, 7] = big[7, 8] = 1.5
    big = big + big.T
    Lb = np.diag(big.sum(1)) - big
    n_comp = int(np.sum(np.linalg.eigvalsh(Lb) < 1e-10))
    assert leverage_scores(Lb)[1].sum() == pytest.approx(9 - n_comp)


def test_spectral_error_closed_forms():
    L, _ = complete(6)
    assert spectral_error(L, 1.2 * L) == pytest.approx(0.2)
    assert spectral_error(L, 0.7 * L) == pytest.approx(0.3)
    path = laplacian(3, [(0, 1), (1, 2)], [1.0, 1.0])
    broken = laplacian(3, [(0, 1)], [1.0])                       # a lost bridge: x^T L~ x = 0 on a vector with x^T L x > 0
    assert spectral_error(path, broken) == pytest.approx(1.0)


def test_spectral_error_best_scale_closed_forms():
    # a common factor is removed entirely: c L against L has error 0 at the factor 1/c
    L, _ = complete(6)
    eps, c = spectral_error_best_scale(L, 2.0 * L)
    assert eps == pytest.approx(0.0, abs=1e-12) and c == pytest.approx(0.5)
    # triangle with unit weights against the path that lacks one branch, weights one: on the complement of the
    # constants mu = 1 (the direction orthogonal to the missing branch's) and mu = 1/3, so kappa = 3, the error at the
    # best factor is (3 - 1)/(3 + 1) = 1/2 at c = 2/(1 + 1/3) = 3/2, the SDP optimum on this triangle
    tri = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, 1.0])
    path = laplacian(3, [(0, 1), (1, 2)], [1.0, 1.0])
    eps, c = spectral_error_best_scale(tri, path)
    assert eps == pytest.approx(0.5) and c == pytest.approx(1.5)
    assert spectral_error(tri, c * path) == pytest.approx(eps)
    # a network that falls apart stays at one, whatever the factor
    broken = laplacian(3, [(0, 1)], [1.0])
    assert spectral_error_best_scale(path, broken)[0] == pytest.approx(1.0)


def test_spectral_error_best_scale_is_the_minimum_over_factors():
    rng = np.random.default_rng(4)
    n = 7
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    L = laplacian(n, edges, rng.uniform(0.5, 2.0, len(edges)))
    La = laplacian(n, edges, rng.uniform(0.1, 5.0, len(edges)))
    eps, _c = spectral_error_best_scale(L, La)
    grid = np.linspace(0.01, 3.0, 30001)
    assert eps == pytest.approx(min(spectral_error(L, t * La) for t in grid), abs=1e-3)
    assert eps <= spectral_error(L, La) + 1e-12


def test_sparsify_keeps_a_tree_exactly_in_expectation_and_is_unbiased():
    """E L~ = L (Sec. 3). On a tree all p_e are equal, so the sampled weights are w_e n_e (n-1)/q
    with multinomial counts n_e; the mean over many runs approaches L at the rate 1/sqrt(runs)."""
    L = laplacian(4, [(0, 1), (1, 2), (2, 3)], [1.0, 2.0, 4.0])
    one = sparsify(L, q=30, seed=1)
    assert np.allclose(one.sum(1), 0.0) and np.allclose(one, one.T)
    off = -one[[0, 1, 2], [1, 2, 3]] / np.array([1.0, 2.0, 4.0]) * 30 / 3
    assert np.allclose(off, np.round(off)) and off.sum() == pytest.approx(30)   # integer counts that sum to q
    mean = np.mean([sparsify(L, q=30, seed=s) for s in range(600)], axis=0)
    assert np.allclose(mean, L, atol=0.15)


def test_sparsify_the_complete_graph():
    """K_60 has 1770 edges. With q = 4000 samples the sparsifier has fewer edges, its error eps is
    well below one, and every eigenvalue lies within the factors 1 -+ eps (Thm. 1 and the remark
    after it)."""
    n = 60
    L, edges = complete(n)
    H = sparsify(L, q=4000, seed=0)
    kept = int(np.count_nonzero(np.triu(H, 1)))
    assert kept < len(edges)
    eps = spectral_error(L, H)
    assert 0.0 < eps < 0.5
    lam, lam_h = np.linalg.eigvalsh(L)[1:], np.linalg.eigvalsh(H)[1:]
    assert np.all(lam_h >= (1 - eps) * lam - 1e-9) and np.all(lam_h <= (1 + eps) * lam + 1e-9)
    assert spectral_error(L, sparsify(L, q=40000, seed=0)) < eps          # more samples, smaller error


# --- approximate Schur complement ----------------------------------------------------------------

def test_approx_schur_of_a_star_is_the_star_mesh_transform_in_expectation():
    """Eliminating the center of a star is one CliqueSample; its expectation is the exact clique
    w_i w_j / sum(w) (Kyng & Sachdeva 2016, Sec. 3: E sum_i Y_i = C_v(S))."""
    w = np.array([1.0, 2.0, 3.0, 4.0])
    L = laplacian(5, [(0, k) for k in range(1, 5)], w)
    exact = kron_reduce(L, keep=[1, 2, 3, 4])
    runs = [approx_schur(L, keep=[1, 2, 3, 4], copies=1, seed=s) for s in range(4000)]
    for H in runs[:20]:
        assert H.shape == (4, 4) and np.allclose(H.sum(1), 0.0) and np.allclose(H, H.T) and np.all(H - np.diag(np.diag(H)) <= 1e-15)
    assert np.allclose(np.mean(runs, axis=0), exact, atol=0.06)


def test_approx_schur_of_a_path_gives_multiples_of_the_series_conductance():
    """Interior vertex of degree two with `copies` multi-edges per line: every successful sample has
    weight (w1/c)(w2/c) / ((w1 + w2)/c), the series conductance over c, so the result is a multiple
    k/c of the series conductance with 0 <= k <= 2c and mean 1."""
    b1, b2, c = 2.0, 3.0, 4
    L = laplacian(3, [(0, 1), (1, 2)], [b1, b2])
    series = b1 * b2 / (b1 + b2)
    ks = np.array([-approx_schur(L, keep=[0, 2], copies=c, seed=s, leverage=False)[0, 1] / series * c for s in range(2000)])
    assert np.allclose(ks, np.round(ks)) and ks.min() >= 0 and ks.max() <= 2 * c
    assert ks.mean() == pytest.approx(c, rel=0.05)


def test_approx_schur_converges_to_the_kron_reduction():
    """Thm. 2.2 of Durfee et al. 2017: the error against the exact Schur complement falls with the
    number of copies (rho ~ eps^-2). 8 x 8 grid, keep the 16 nodes of a coarse 4 x 4 subgrid."""
    n = 8
    idx = lambda i, j: i * n + j
    edges = [(idx(i, j), idx(i, j + 1)) for i in range(n) for j in range(n - 1)] + [(idx(i, j), idx(i + 1, j)) for i in range(n - 1) for j in range(n)]
    rng = np.random.default_rng(3)
    L = laplacian(n * n, edges, rng.uniform(0.5, 2.0, len(edges)))
    keep = [idx(i, j) for i in range(0, n, 2) for j in range(0, n, 2)]
    exact = kron_reduce(L, keep)
    err = {c: np.median([spectral_error(exact, approx_schur(L, keep, copies=c, seed=s)) for s in range(12)]) for c in (5, 50, 500)}
    assert err[5] > err[50] > err[500]
    assert err[500] < 0.25
    from scipy.sparse import csr_matrix  # sparse input with supplied scores, same seed, same result
    dense = approx_schur(L, keep, copies=50, seed=7)
    sparse = approx_schur(csr_matrix(L), keep, copies=50, seed=7, scores=leverage_scores(L)[1])
    assert np.allclose(dense, sparse)
    with pytest.raises(ValueError):
        approx_schur(L, keep=[0, 0, 1], copies=5)


# --- the deterministic construction of Batson, Spielman & Srivastava 2012 (arXiv:0808.0163v3) ---------------
# Theorem 1.1: for d > 1 every weighted graph on n vertices contains a reweighted subgraph with at most
# ceil(d (n - 1)) edges and x^T L x <= x^T L_H x <= kappa x^T L x, kappa = (d + 1 + 2 sqrt d)/(d + 1 - 2 sqrt d).
# Scaled by 2/(1 + kappa) this is a two-sided spectral error eps = (kappa - 1)/(kappa + 1) = 2 sqrt(d)/(d + 1).

def _kappa(d):
    return (d + 1 + 2 * np.sqrt(d)) / (d + 1 - 2 * np.sqrt(d))


def _relative_spectrum(L, H):
    """Nonzero eigenvalues of L^{+/2} H L^{+/2} on the range of L."""
    w, V = np.linalg.eigh(L)
    keep = w > 1e-9 * w.max()
    S = V[:, keep] / np.sqrt(w[keep])
    return np.linalg.eigvalsh(S.T @ H @ S)


@pytest.mark.parametrize("d", [2.0, 4.0, 9.0])
def test_bss_meets_the_guarantee_of_theorem_1_1_on_the_complete_graph(d):
    from gridnet.sparsify import bss_sparsify
    n = 30
    L, _ = complete(n)
    H = bss_sparsify(L, d)
    assert np.count_nonzero(np.triu(H, 1)) <= int(np.ceil(d * (n - 1)))
    mu = _relative_spectrum(L, H)
    assert mu.min() > 0 and mu.max() / mu.min() <= _kappa(d) * (1 + 1e-9)
    assert spectral_error(L, H) <= 2 * np.sqrt(d) / (d + 1) + 1e-9           # H is returned in the two-sided scaling


def test_bss_meets_the_guarantee_on_a_random_weighted_graph_and_returns_a_subgraph():
    from gridnet.sparsify import bss_sparsify
    rng = np.random.default_rng(5)
    n = 40
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
    edges += [(i, i + 1) for i in range(n - 1) if (i, i + 1) not in edges]        # connected
    L = laplacian(n, edges, rng.uniform(0.1, 10.0, len(edges)))
    H = bss_sparsify(L, 3.0)
    assert np.allclose(H, H.T) and np.allclose(H.sum(1), 0) and (np.triu(H, 1) <= 1e-12).all()   # a Laplacian
    assert not np.any((np.triu(H, 1) < 0) & (np.triu(L, 1) == 0))                 # only edges of the given graph
    assert np.count_nonzero(np.triu(H, 1)) <= int(np.ceil(3.0 * (n - 1)))
    mu = _relative_spectrum(L, H)
    assert mu.max() / mu.min() <= _kappa(3.0) * (1 + 1e-9)


def test_bss_keeps_every_edge_of_a_tree():
    """Truth known: every edge of a tree is a bridge, so a sparsifier with a finite condition number must keep all
    n - 1 of them, whatever d; the bound of the theorem still has to hold."""
    from gridnet.sparsify import bss_sparsify
    rng = np.random.default_rng(2)
    n = 25
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    L = laplacian(n, edges, rng.uniform(0.5, 2.0, n - 1))
    H = bss_sparsify(L, 2.0)
    assert np.count_nonzero(np.triu(H, 1)) == n - 1
    mu = _relative_spectrum(L, H)
    assert mu.max() / mu.min() <= _kappa(2.0) * (1 + 1e-9)


def test_bss_potentials_never_increase():
    """Conditions (c) and (d) of the proof (Sec. 3.2): with the barriers shifted at every step the upper and the
    lower potential do not increase, and no eigenvalue crosses a barrier."""
    from gridnet.sparsify import bss_sparsify
    L, _ = complete(20)
    history = []
    bss_sparsify(L, 4.0, history=history)
    up = np.array([h["upper_potential"] for h in history]); lo = np.array([h["lower_potential"] for h in history])
    assert (np.diff(up) <= 1e-9).all() and (np.diff(lo) <= 1e-9).all()
    assert all(h["lower_barrier"] < h["lambda_min"] and h["lambda_max"] < h["upper_barrier"] for h in history)


def test_ring_with_a_chord_separates_cut_from_spectral_approximation():
    """Spielman & Teng 2011 (arXiv:0808.4134), Sec. 5.3, Example 3. Vertices (u, i), u in Z_n, i = 1..k; complete
    bipartite graphs between consecutive groups; G has one more edge, a chord from (0, 1) to (n/2, 1). Their
    statements, checked here for n = 16, k = 2: the minimum cut of G is 2k, so removing the chord changes no cut by
    more than the factor 1 + 1/(2k), attained by isolating an end of the chord; for x(u, i) = min(u, n - u) the
    quadratic forms are n k^2 without the chord and n k^2 + (n/2)^2 with it, a factor 1 + n/(4 k^2)."""
    import networkx as nx
    n, k = 16, 2
    idx = lambda u, i: (u % n) * k + i
    ring = [(idx(u, i), idx(u + 1, j)) for u in range(n) for i in range(k) for j in range(k)]
    chord = (idx(0, 0), idx(n // 2, 0))
    Lt = laplacian(n * k, ring, [1.0] * len(ring))
    L = laplacian(n * k, ring + [chord], [1.0] * (len(ring) + 1))
    x = np.repeat([min(u, n - u) for u in range(n)], k).astype(float)
    assert x @ Lt @ x == pytest.approx(n * k**2) and x @ L @ x == pytest.approx(n * k**2 + (n / 2) ** 2)
    G = nx.Graph(); G.add_edges_from(ring + [chord], weight=1.0)
    assert nx.stoer_wagner(G)[0] == pytest.approx(2 * k)
    end = np.zeros(n * k); end[chord[0]] = 1.0                       # the cut that isolates one end of the chord
    assert (end @ L @ end) / (end @ Lt @ end) == pytest.approx(1 + 1 / (2 * k))
    assert spectral_error(L, Lt) >= 1 - 1 / (1 + n / (4 * k**2)) - 1e-12     # the form of x alone forces this error


def test_the_spectral_comparisons_reject_an_indefinite_reference():
    """eq. (2) compares two quadratic forms through L^{+/2}, which exists only for L positive
    semidefinite. A negative weight (a series capacitor) makes L indefinite; the triangle with weights
    1, 1, -0.8 has the eigenvalues -0.6, 0, 3 (closed form in tests/test_graph.py). Dropping the negative
    eigenvalue, as a pseudoinverse cut at zero would, compares on a subspace and returns a number that
    means nothing, so the functions refuse."""
    L = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, -0.8])
    for f in (lambda: spectral_error(L, L), lambda: spectral_error_best_scale(L, L),
              lambda: bss_sparsify(L, 2.0)):
        with pytest.raises(ValueError, match="positive semidefinite|negative weight"):   # bss_sparsify meets the weight first
            f()


def test_a_negative_weight_is_refused_not_dropped():
    """Sampling by w_e R_e (Spielman & Srivastava 2011), the barrier construction and the elimination
    of Kyng & Sachdeva assume nonnegative weights. A Laplacian with a negative weight (a series
    capacitor) is refused; before, the edge was dropped silently and another graph sparsified."""
    from gridnet.graph import laplacian
    from gridnet.sparsify import approx_schur, bss_sparsify, leverage_scores

    L = laplacian(4, [(0, 1), (1, 2), (2, 3), (0, 2)], [1.0, 1.0, 1.0, -0.2])
    for call in (lambda: leverage_scores(L), lambda: bss_sparsify(L, 2.0), lambda: approx_schur(L, [0, 3], copies=1.0)):
        with pytest.raises(ValueError):
            call()


def test_sparsify_draws_in_proportion_to_the_leverage_scores():
    """Spielman & Srivastava 2011, Sec. 1.1: edge e is drawn with probability p_e proportional to w_e R_e. On a graph
    whose scores are far from uniform (a triangle of strong branches with a pendant path) the number of draws of
    every edge, read back from its weight w_e c_e / (q p_e), stays within five standard deviations of q p_e,
    which uniform probabilities would miss by more than twenty (scores 2/3 on the triangle, one on the path)."""
    n = 6
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)]
    w = np.array([10.0, 10.0, 10.0, 1.0, 2.0, 3.0])
    L = laplacian(n, edges, w)
    ed, scores = leverage_scores(L)
    p = scores / scores.sum()
    q = 200_000
    H = sparsify(L, q, seed=0)
    wts = -L[ed[:, 0], ed[:, 1]]
    counts = -H[ed[:, 0], ed[:, 1]] * q * p / wts
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-6)            # whole numbers of draws
    sd = np.sqrt(q * p * (1 - p))
    assert np.all(np.abs(counts - q * p) < 5 * sd)
    uniform = q / len(ed)
    assert np.max(np.abs(counts - uniform) / np.sqrt(uniform)) > 20               # 2/3 on the triangle, 1 on the path


def test_the_spectral_error_is_infinite_when_the_approximation_joins_components():
    """Eq. (2) of Spielman & Srivastava 2011 is asked of every x, also of those in the kernel of
    L. L two separate edges (0, 1) and (2, 3) of weight 1, L~ the same with a bridge (1, 2) of
    weight 5: x = (1, 1, -1, -1) is constant on both components of L, so x^T L x = 0, while
    x^T L~ x = 5 (1 - (-1))^2 = 20, and no finite eps gives 20 <= (1 + eps) 0. On the range of L
    alone the two forms compare as if the bridge were harmless. The error is infinite, at every
    common factor; the reverse case, L~ splitting a component of L, keeps the error one."""
    L = laplacian(4, [(0, 1), (2, 3)], [1.0, 1.0])
    joined = laplacian(4, [(0, 1), (2, 3), (1, 2)], [1.0, 1.0, 5.0])
    x = np.array([1.0, 1.0, -1.0, -1.0])
    assert x @ L @ x == 0.0 and x @ joined @ x == 20.0
    assert spectral_error(L, joined) == np.inf
    eps, c = spectral_error_best_scale(L, joined)
    assert eps == np.inf and np.isnan(c)
    assert spectral_error(joined, L) == pytest.approx(1.0)      # a component split: mu = 0
    assert spectral_error(L, 1.3 * L) == pytest.approx(0.3)     # the same kernel: finite as before
