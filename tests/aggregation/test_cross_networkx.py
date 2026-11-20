"""The graph and spectral functions against networkx, and the Kron reduction against its theorems.

Cross-implementation: `effective_resistance` against networkx's
resistance distance, `fiedler` against its algebraic connectivity (combinatorial and normalized),
`normalized_laplacian`, `conductance` and the two-way `ncut` against networkx; `spectral_embedding` against
its defining eigenproblem. Kron reduction: Doerfler & Bullo 2013, Lemma 3.3 (eliminating the interior buses
one at a time gives the reduction of eliminating them at once, the quotient property, statement 2; the
reduction of a Laplacian is a Laplacian, statement 1) and Thm. 3.7, statement 1 (a connected graph reduces
to a connected one, the closure of irreducibility), and Anderson & Trapp 1975, Thm. 1 (the Schur
complement is the largest X with [X 0; 0 0] below L in the semidefinite order, the shorted operator).
"""

import networkx as nx
import numpy as np
import pytest

from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import kron_reduce
from gridnet.spectral import conductance, fiedler, ncut, normalized_laplacian, spectral_embedding


def weighted_graph(seed=0, n=12, extra=10):
    """A connected weighted graph: a ring plus random chords, weights in [0.5, 2]."""
    rng = np.random.default_rng(seed)
    edges = {(i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i) for i in range(n)}
    while len(edges) < n + extra:
        i, j = sorted(rng.choice(n, 2, replace=False))
        edges.add((int(i), int(j)))
    edges = sorted(edges)
    w = rng.uniform(0.5, 2.0, len(edges))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), wk in zip(edges, w):
        G.add_edge(i, j, weight=float(wk))
    return n, edges, w, G


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_effective_resistance_against_networkx(seed):
    n, edges, w, G = weighted_graph(seed)
    R = effective_resistance(laplacian(n, edges, w))
    for i, j in [(0, 5), (3, 11), (2, 7)]:
        # networkx reads the weight as a conductance when invert_weight=False
        np.testing.assert_allclose(R[i, j], nx.resistance_distance(G, i, j, weight="weight", invert_weight=False), rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 1])
def test_fiedler_and_normalized_laplacian_against_networkx(seed):
    n, edges, w, G = weighted_graph(seed)
    L = laplacian(n, edges, w)
    lam, v = fiedler(L)
    np.testing.assert_allclose(lam, nx.algebraic_connectivity(G, weight="weight", method="tracemin_lu", tol=1e-12), rtol=1e-8)
    np.testing.assert_allclose(L @ v, lam * v, atol=1e-10)
    lam_n, _ = fiedler(L, normalized=True)
    np.testing.assert_allclose(lam_n, nx.algebraic_connectivity(G, weight="weight", normalized=True, method="tracemin_lu", tol=1e-12), rtol=1e-8)
    np.testing.assert_allclose(normalized_laplacian(L, "sym"), nx.normalized_laplacian_matrix(G, nodelist=range(n), weight="weight").toarray(), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_conductance_and_ncut_against_networkx(seed):
    n, edges, w, G = weighted_graph(seed)
    L = laplacian(n, edges, w)
    S = np.zeros(n, dtype=bool); S[: n // 2 - 1] = True
    A, B = set(np.flatnonzero(S).tolist()), set(np.flatnonzero(~S).tolist())
    np.testing.assert_allclose(conductance(L, S), nx.conductance(G, A, B, weight="weight"), rtol=1e-12)
    np.testing.assert_allclose(ncut(L, S.astype(int)), nx.normalized_cut_size(G, A, B, weight="weight"), rtol=1e-12)


def test_spectral_embedding_solves_the_generalized_eigenproblem():
    """von Luxburg 2007, Sec. 4 and Prop. 3: the normalized embedding holds the k lowest generalized eigenvectors of
    (L, D), D-orthonormal, the first constant; the unnormalized one the k lowest eigenvectors of L."""
    n, edges, w, _ = weighted_graph(3)
    L = laplacian(n, edges, w); D = np.diag(np.diag(L))
    k = 4
    X = spectral_embedding(L, k, normalized=True)
    lam = np.diag(X.T @ L @ X)
    np.testing.assert_allclose(L @ X, D @ X * lam, atol=1e-10)
    np.testing.assert_allclose(X.T @ D @ X, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(lam, np.sort(np.linalg.eigvalsh(normalized_laplacian(L, "sym")))[:k], atol=1e-10)
    assert np.ptp(X[:, 0]) < 1e-10
    U = spectral_embedding(L, k, normalized=False)
    np.testing.assert_allclose(np.diag(U.T @ L @ U), np.linalg.eigvalsh(L)[:k], atol=1e-10)


def test_kron_reduction_quotient_property_and_connectivity():
    """Doerfler & Bullo 2013, Lemma 3.3: eliminating the interior buses one at a time gives the one-step reduction
    (statement 2, the quotient property) and the reduction of a Laplacian is a Laplacian (statement 1: zero row
    sums, nonpositive off-diagonal entries); Thm. 3.7, statement 1: the reduction of a connected graph is
    connected (lambda_2 > 0)."""
    n, edges, w, _ = weighted_graph(4, n=14, extra=8)
    L = laplacian(n, edges, w)
    keep = [0, 3, 6, 9, 12]
    once = kron_reduce(L, keep)
    stepwise, current = L.copy(), list(range(n))
    for b in [i for i in range(n) if i not in keep]:
        rest = [i for i in current if i != b]
        stepwise = kron_reduce(stepwise, [current.index(i) for i in rest])
        current = rest
    np.testing.assert_allclose(stepwise, once, atol=1e-12)
    np.testing.assert_allclose(once.sum(axis=1), 0, atol=1e-12)
    assert np.all(once[~np.eye(len(keep), dtype=bool)] <= 1e-15)
    assert np.linalg.eigvalsh(once)[1] > 1e-6


def test_schur_complement_is_the_shorted_operator():
    """Anderson & Trapp 1975, Thm. 1: S = L/L_bb is the largest X with [X 0; 0 0] <= L. So L - [S 0; 0 0] is
    positive semidefinite, and for X = S + t u u^T with any t > 0 and u off the kernel of S it is not."""
    n, edges, w, _ = weighted_graph(5)
    L = laplacian(n, edges, w)
    keep = [1, 4, 7, 10]
    order = keep + [i for i in range(n) if i not in keep]
    Lp = L[np.ix_(order, order)]
    S = kron_reduce(L, keep)
    k = len(keep)

    def embed(X):
        E = np.zeros((n, n)); E[:k, :k] = X
        return E

    assert np.linalg.eigvalsh(Lp - embed(S)).min() > -1e-10
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.standard_normal(k); u -= u.mean()                     # off the kernel (the constants) of S
        assert np.linalg.eigvalsh(Lp - embed(S + 1e-3 * np.outer(u, u))).min() < -1e-8
