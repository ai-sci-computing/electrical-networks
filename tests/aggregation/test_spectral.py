"""Tests for gridnet.spectral: Fiedler vector, normalized Laplacians, sweep cut, Cheeger.

References
  unnormalized / normalized Laplacians and their properties:
      von Luxburg, Statistics and Computing 17 (2007), Secs. 3.1-3.3, Props. 1-4.
  RatioCut and Ncut and their spectral relaxations: von Luxburg 2007, Secs. 5.1-5.3.
  Cheeger constant, Cheeger inequality lambda_2/2 <= h_G <= sqrt(2 lambda_2) for the
  normalized Laplacian, constructive via the sweep over the eigenvector:
      Bandeira, Singer & Spielman, SIAM J. Matrix Anal. Appl. 34(4) (2013), Sec. 1.2, Thm. 1.1.
  weights for power grids (admittance = inverse reactance):
      Sanchez-Garcia et al., IEEE Trans. Power Systems 29(5) (2014), Sec. II-B.
Closed forms: lambda_2 of path, cycle, complete graph; the path's Fiedler vector; two
cliques joined by a bridge.
"""

import numpy as np
import pytest

from gridnet.graph import laplacian
from gridnet.spectral import (
    cheeger_bounds,
    conductance,
    cut_weight,
    fiedler,
    normalized_laplacian,
    ratio_cut,
    spectral_clustering,
    sweep_cut,
    variation_of_information,
)


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n):
    return path_edges(n) + [(n - 1, 0)]


def complete_edges(n, offset=0):
    return [(i + offset, j + offset) for i in range(n) for j in range(i + 1, n)]


def two_cliques(m):
    """K_m and K_m joined by one bridge (m-1, m)."""
    return 2 * m, complete_edges(m) + complete_edges(m, m) + [(m - 1, m)]


# --- algebraic connectivity and Fiedler vector ------------------------------------

@pytest.mark.parametrize("n", [4, 7, 12])
def test_lambda2_closed_forms(n):
    assert np.isclose(fiedler(laplacian(n, path_edges(n)))[0], 2 - 2 * np.cos(np.pi / n))
    assert np.isclose(fiedler(laplacian(n, cycle_edges(n)))[0], 2 - 2 * np.cos(2 * np.pi / n))
    assert np.isclose(fiedler(laplacian(n, complete_edges(n)))[0], n)


def test_path_fiedler_vector_is_a_cosine():
    n = 9
    _, v = fiedler(laplacian(n, path_edges(n)))
    u = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    u /= np.linalg.norm(u)
    assert np.isclose(abs(v @ u), 1.0)                       # equal up to sign


def test_disconnected_graph_has_lambda2_zero_and_fiedler_is_indicator():
    L = laplacian(6, path_edges(3) + [(3, 4), (4, 5)])
    lam, v = fiedler(L)
    assert np.isclose(lam, 0, atol=1e-12)
    assert np.allclose(L @ v, 0, atol=1e-12)


def test_two_cliques_fiedler_sign_pattern_and_sweep_cut_find_the_bridge():
    m = 5
    n, edges = two_cliques(m)
    L = laplacian(n, edges)
    _, v = fiedler(L)
    assert np.all(np.sign(v[:m]) == np.sign(v[0])) and np.all(np.sign(v[m:]) == -np.sign(v[0]))
    S, val = sweep_cut(L, v, objective="conductance")
    assert set(np.flatnonzero(S)) in ({*range(m)}, {*range(m, n)})
    vol = m * (m - 1) + 1                                    # degrees inside K_m plus the bridge end
    assert np.isclose(val, 1 / vol)
    S2, val2 = sweep_cut(L, v, objective="ratio")
    assert set(np.flatnonzero(S2)) in ({*range(m)}, {*range(m, n)})
    assert np.isclose(val2, 1 / m + 1 / m)


# --- cut functionals ----------------------------------------------------------------

def test_cut_weight_is_the_quadratic_form_of_the_indicator():
    rng = np.random.default_rng(0)
    n = 8
    edges = complete_edges(n)
    w = rng.uniform(0.1, 1, len(edges))
    L = laplacian(n, edges, w)
    S = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    x = S.astype(float)
    assert np.isclose(cut_weight(L, S), x @ L @ x)
    volS, volSc = L.diagonal()[S].sum(), L.diagonal()[~S].sum()
    assert np.isclose(conductance(L, S), cut_weight(L, S) / min(volS, volSc))
    assert np.isclose(ratio_cut(L, S), cut_weight(L, S) * (1 / S.sum() + 1 / (~S).sum()))


# --- normalized Laplacians ------------------------------------------------------------

def test_normalized_laplacians_share_eigenvalues_and_are_psd():
    rng = np.random.default_rng(1)
    n = 7
    edges = complete_edges(n)
    L = laplacian(n, edges, rng.uniform(0.1, 1, len(edges)))
    Ls = normalized_laplacian(L, kind="sym")
    Lrw = normalized_laplacian(L, kind="rw")
    ws = np.linalg.eigvalsh(Ls)
    wrw = np.sort(np.linalg.eigvals(Lrw).real)
    np.testing.assert_allclose(ws, wrw, atol=1e-10)         # von Luxburg Prop. 3(2)
    assert ws[0] > -1e-12 and np.isclose(ws[0], 0, atol=1e-12)
    assert ws[-1] <= 2 + 1e-12
    np.testing.assert_allclose(Ls, Ls.T)


def test_normalized_fiedler_solves_generalized_problem():
    rng = np.random.default_rng(2)
    n = 8
    edges = complete_edges(n)
    L = laplacian(n, edges, rng.uniform(0.1, 1, len(edges)))
    lam, v = fiedler(L, normalized=True)
    D = np.diag(L.diagonal())
    np.testing.assert_allclose(L @ v, lam * D @ v, atol=1e-10)   # L v = lambda D v (Prop. 3(3))


# --- Cheeger inequality ----------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cheeger_inequality_is_constructive(seed):
    rng = np.random.default_rng(seed)
    n = 14
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35] + path_edges(n)
    L = laplacian(n, edges, rng.uniform(0.2, 2, len(edges)))
    lo, hi = cheeger_bounds(L)
    lam, v = fiedler(L, normalized=True)
    _, h = sweep_cut(L, v, objective="conductance")
    assert lo <= h <= hi + 1e-12                            # Bandeira et al. 2013, Thm. 1.1
    assert np.isclose(lo, lam / 2) and np.isclose(hi, np.sqrt(2 * lam))


# --- k-way clustering and partition comparison ----------------------------------------

def test_spectral_clustering_recovers_three_cliques():
    pytest.importorskip("sklearn")
    m = 6
    edges = complete_edges(m) + complete_edges(m, m) + complete_edges(m, 2 * m) + [(m - 1, m), (2 * m - 1, 2 * m)]
    L = laplacian(3 * m, edges)
    labels = spectral_clustering(L, k=3, seed=0)
    truth = np.repeat([0, 1, 2], m)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(truth, labels) == 1.0
    assert np.isclose(variation_of_information(truth, labels), 0, atol=1e-12)


def test_variation_of_information_closed_form():
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.zeros(6, dtype=int)
    assert np.isclose(variation_of_information(a, b), 1.0)         # H(a) + H(b) - 2 I = 1 bit (Meila 2007, eq. 19)
    assert np.isclose(variation_of_information(a, a), 0)
    assert np.isclose(variation_of_information(a, b), variation_of_information(b, a))


def test_fiedler_subset_solver_agrees_with_dense():
    rng = np.random.default_rng(5)
    n = 40
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2] + path_edges(n)
    L = laplacian(n, edges, rng.uniform(0.1, 1, len(edges)))
    lam, v = fiedler(L)
    w, U = np.linalg.eigh(L)
    assert np.isclose(lam, w[1]) and np.isclose(abs(v @ U[:, 1]), 1.0)
    lam_n, v_n = fiedler(L, normalized=True)
    D = np.diag(L.diagonal())
    np.testing.assert_allclose(L @ v_n, lam_n * D @ v_n, atol=1e-9)


def test_ncut_of_labels_two_cliques():
    """Ncut(A_1..A_k) = sum_i cut(A_i, complement)/vol(A_i) (von Luxburg 2007, Sec. 5.1)."""
    from gridnet.spectral import ncut
    m = 5
    n, edges = two_cliques(m)
    L = laplacian(n, edges)
    labels = np.repeat([0, 1], m)
    vol = m * (m - 1) + 1
    assert np.isclose(ncut(L, labels), 2 * (1 / vol))
    # a 3-way split of one clique adds cuts inside it: strictly larger
    labels3 = labels.copy(); labels3[:2] = 2
    assert ncut(L, labels3) > ncut(L, labels)


def test_adjusted_rand_index_matches_hubert_arabie_formula():
    """ARI = (Index - E[Index]) / (Max - E[Index]) with the hypergeometric expectation
    (Hubert & Arabie 1985, eq. (4) and the corrected Rand index of eq. (5) that follows it), computed
    by hand from the contingency table and compared with scikit-learn."""
    from math import comb

    adjusted_rand_score = pytest.importorskip("sklearn.metrics").adjusted_rand_score
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    table = np.zeros((3, 3), dtype=int)
    for i, j in zip(a, b):
        table[i, j] += 1
    n = a.size
    index = sum(comb(int(x), 2) for x in table.ravel())
    ra = sum(comb(int(x), 2) for x in table.sum(axis=1))
    rb = sum(comb(int(x), 2) for x in table.sum(axis=0))
    expected = ra * rb / comb(n, 2)
    maximum = (ra + rb) / 2
    ari = (index - expected) / (maximum - expected)
    assert np.isclose(adjusted_rand_score(a, b), ari)


def test_variation_of_information_is_a_metric_and_bounded():
    """Meila 2007: Property 1 (metric: symmetry, triangle inequality, zero iff equal) and
    Property 3 (VI <= log n) on random labelings."""
    rng = np.random.default_rng(7)
    n = 30
    a, b, c = (rng.integers(0, 4, n) for _ in range(3))
    vab, vbc, vac = (variation_of_information(x, y) for x, y in ((a, b), (b, c), (a, c)))
    assert np.isclose(vab, variation_of_information(b, a))
    assert vac <= vab + vbc + 1e-12
    assert max(vab, vbc, vac) <= np.log2(n) + 1e-12
    assert variation_of_information(a, a) == 0


def test_recursive_bipartition_splits_the_weakest_part_first():
    adjusted_rand_score = pytest.importorskip("sklearn.metrics").adjusted_rand_score

    from gridnet.spectral import recursive_bipartition

    # three cliques of size 6 in a chain A - B - C, bridges of weight 0.1 between them
    m = 6
    edges, w = [], []
    for c in range(3):
        for i in range(m):
            for j in range(i + 1, m):
                edges.append((c * m + i, c * m + j)); w.append(1.0)
    edges += [(m - 1, m), (2 * m - 1, 2 * m)]; w += [0.1, 0.1]
    L = laplacian(3 * m, edges, w)
    truth = np.repeat([0, 1, 2], m)
    labels2 = recursive_bipartition(L, 2)
    assert len(set(labels2.tolist())) == 2
    # the first cut is one of the two bridges: no clique is split
    for c in range(3):
        assert len(set(labels2[c * m:(c + 1) * m].tolist())) == 1
    hist = []
    labels3 = recursive_bipartition(L, 3, history=hist)
    assert adjusted_rand_score(truth, labels3) == 1.0
    assert len(hist) == 2 and all(h['conductance'] > 0 for h in hist)
    assert set(hist[0]['new_side'].tolist()) | set(hist[0]['parent_side'].tolist()) == set(range(3 * m))
    with pytest.raises(ValueError):
        recursive_bipartition(L, 1)


def test_area_parameters_of_chow_and_kokotovic():
    """Chow & Kokotovic 1985, Sec. II: with c^I the smallest number of internal and c^E the largest number of
    external connections of a node, the node parameter is d = c^E / c^I (eq. (2.3)); with gamma^E the largest
    number of external connections of an area and m the smallest number of nodes of an area, the area parameter
    is delta = gamma^E / (m c^I). Their Fig. 1, two areas of five nodes, every node with at least three internal
    and at most one external connection, two connections between the areas: d = 0.333, delta = 0.133. Their
    Fig. 2, a path of eight nodes in two areas of four: d = 1, the requirement d << 1 violated."""
    from gridnet.spectral import area_parameters

    edges = _chow_kokotovic_fig_1()
    p = area_parameters(edges, np.repeat([0, 1], 5))
    assert p["d"] == pytest.approx(1 / 3) and p["delta"] == pytest.approx(2 / 15)
    path = area_parameters([(k, k + 1) for k in range(7)], np.repeat([0, 1], 4))
    assert path["d"] == pytest.approx(1.0) and path["delta"] == pytest.approx(0.25)
    # the variant for large networks: mean external over mean internal connections per node, two ties touch four
    # nodes, 0.4 per node; each area has 9 internal connections, 3.6 per node (3.2 with the graph this test
    # had until 2026-10-24, two edges short per area)
    assert p["d_average"] == pytest.approx(0.4 / 3.6)


def test_slow_and_fast_eigenvalues_of_chow_and_kokotovic_tables_1_to_4():
    """Chow & Kokotovic 1985, Tables I to IV (printed p. 719), unit weights and masses. The 10-node network
    of Fig. 1: slow eigenvalue exact 0.61, complete aggregate 0.63, rigid aggregate 0.80; fast eigenvalues
    3.34, 5.00, 5.00, 6.16 and 3.00, 5.00, 5.00, 6.90. The longitudinal network of Fig. 2, a path of eight
    in two areas of four: 0.152, 0.182, 0.500, fast 0.586, 1.235, 2.000, 2.765, 3.414, 3.848. The
    aggregates carry the area sizes as masses; the complete aggregate is `zonal_operator` with uniform
    distributions, the rigid one chi^T L chi."""
    import scipy.linalg

    from gridnet.reweight import zonal_operator

    def slow(L, zones):
        M = np.diag(np.bincount(zones).astype(float))
        chi = np.eye(M.shape[0])[zones]
        return (np.linalg.eigvalsh(L)[1], scipy.linalg.eigh(zonal_operator(L, zones), M, eigvals_only=True)[1],
                scipy.linalg.eigh(chi.T @ L @ chi, M, eigvals_only=True)[1])

    E = _chow_kokotovic_fig_1()
    L = laplacian(10, E, np.ones(len(E)))
    np.testing.assert_allclose(slow(L, np.repeat([0, 1], 5)), [0.61, 0.63, 0.80], atol=5e-3)
    np.testing.assert_allclose(np.linalg.eigvalsh(L)[2:], [3.00, 3.34, 5.00, 5.00, 5.00, 5.00, 6.16, 6.90], atol=5e-3)
    P = laplacian(8, [(k, k + 1) for k in range(7)], np.ones(7))
    np.testing.assert_allclose(slow(P, np.repeat([0, 1], 4)), [0.152, 0.182, 0.500], atol=5e-4)
    np.testing.assert_allclose(np.linalg.eigvalsh(P)[2:], [0.586, 1.235, 2.000, 2.765, 3.414, 3.848], atol=5e-4)


def _chow_kokotovic_fig_1():
    """The 10-node network of Chow & Kokotovic 1985, Fig. 1, from the matrices printed on p. 719: each
    area K5 less one edge, (0, 2) in area 1 (K1^I) and (6, 7) in area 2 (K2^I), the ties (2, 5) and (4, 9)
    (K12^E)."""
    one = [(a, b) for a in range(5) for b in range(a + 1, 5) if (a, b) != (0, 2)]
    two = [(a + 5, b + 5) for a in range(5) for b in range(a + 1, 5) if (a, b) != (1, 2)]
    return one + two + [(2, 5), (4, 9)]


# --- coarsening: restricted spectral approximation (Loukas 2019) --------------------------------------

def _two_cliques():
    """K4 x K2: two 4-cliques joined by a perfect matching, unit weights; spectrum {0, 2, 4, 4, 4, 6, 6, 6}."""
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    edges = edges + [(a + 4, b + 4) for a, b in edges] + [(a, a + 4) for a in range(4)]
    return laplacian(8, edges, np.ones(len(edges)))


def test_restricted_spectral_error_closed_forms():
    """Loukas 2019, Def. 3.1: L_c and L are (R, eps)-similar if ||x - Pi x||_L <= eps ||x||_L on R, Pi = P^+ P the
    averaging over the contraction sets; R_k is the span of the first k eigenvectors. For K4 x K2 contracted to
    its two cliques the second eigenvector is constant on the cliques, eps_2 = 0, and the next ones average to zero
    on each clique, Pi x = 0 and eps_3 = 1. Without contraction eps = 0 for every k."""
    from gridnet.spectral import restricted_spectral_error

    L = _two_cliques()
    assert np.allclose(np.linalg.eigvalsh(L), [0, 2, 4, 4, 4, 6, 6, 6])
    labels = np.repeat([0, 1], 4)
    assert restricted_spectral_error(L, labels, 2) == pytest.approx(0.0, abs=1e-10)
    assert restricted_spectral_error(L, labels, 3) == pytest.approx(1.0)
    assert restricted_spectral_error(L, np.arange(8), 6) == pytest.approx(0.0, abs=1e-10)


def test_coarsened_eigenvalues_and_the_bound_of_theorem_3_3():
    """The coarsened Laplacian of Loukas 2019, Prop. 2.2, L_c = P^-+ L P^+, is U^T L U for averaging coarsening
    matrices; its eigenvalues relative to the set sizes, those of D^{-1/2} U^T L U D^{-1/2}, are the ones his
    Table 1 compares with the original (and the squared modes of Chow & Kokotovic's rigid aggregate). For K4 x K2
    on its cliques: U^T L U = [[4, -4], [-4, 4]], sizes 4, eigenvalues 0 and 2, the first two of L exactly. Thm.
    3.3 on a weighted ring contracted in pairs: lambda_k <= lambda~_k <= (1 + eps_k)^2 / (1 - eps_k^2 lambda_k /
    lambda_2) lambda_k with gamma_1 = gamma_2 = 1 for this normalization and equal set sizes, whenever eps_k^2 <
    lambda_2 / lambda_k."""
    from gridnet.spectral import coarsened_eigenvalues, restricted_spectral_error

    assert coarsened_eigenvalues(_two_cliques(), np.repeat([0, 1], 4)) == pytest.approx([0.0, 2.0], abs=1e-12)
    rng = np.random.default_rng(0)
    n = 24
    L = laplacian(n, [(i, (i + 1) % n) for i in range(n)], rng.uniform(0.8, 1.2, n))
    labels = np.arange(n) // 2
    lam, lam_c = np.linalg.eigvalsh(L), coarsened_eigenvalues(L, labels)
    eps = restricted_spectral_error(L, labels, 2)
    assert eps ** 2 < 1.0                                        # the theorem's condition eps_k^2 < lambda_2 / lambda_k at k = 2
    assert lam[1] - 1e-12 <= lam_c[1] <= (1 + eps) ** 2 / (1 - eps ** 2) * lam[1] + 1e-12
    # even this mildest coarsening has eps_2 near one: removing the pair averages from a smooth vector leaves a
    # sawtooth whose differences along the ring are as large as the vector's own, and the coarse eigenvalue is
    # nearly twice the original (0.119 against 0.065), the stiffening of the rigid aggregate
    assert 0.85 < eps < 1.0 and 1.7 < lam_c[1] / lam[1] < 2.0
    for k in (3, 4):                                             # beyond k = 2 the condition fails and the theorem is silent
        assert restricted_spectral_error(L, labels, k) ** 2 > lam[1] / lam[k - 1]


def test_coarsened_eigenvalues_sparse_path_matches_dense_and_never_loses_the_zero_mode():
    """The k smallest coarse eigenvalues by shift-invert Lanczos against the dense computation, on a 30 x 30 grid
    contracted in 2 x 2 blocks, from 25 random starting vectors. ARPACK's which='SM' at a loose tolerance, the call
    in the public code of Loukas 2019, returns eigenvalues 2..k+1 in about one run in ten because the zero mode
    has not converged, and every error is then computed against the wrong partner (found 2026-09-19)."""
    from scipy.sparse import csr_matrix

    from gridnet.spectral import coarsened_eigenvalues

    n = 30
    idx = lambda a, b: a * n + b
    edges = [(idx(a, b), idx(a, b + 1)) for a in range(n) for b in range(n - 1)] + [(idx(a, b), idx(a + 1, b)) for a in range(n - 1) for b in range(n)]
    rng = np.random.default_rng(0)
    L = laplacian(n * n, edges, rng.uniform(0.5, 2.0, len(edges)))
    labels = np.array([(a // 2) * (n // 2) + b // 2 for a in range(n) for b in range(n)])
    dense = coarsened_eigenvalues(L, labels)[:10]
    assert abs(dense[0]) < 1e-10
    for seed in range(25):
        sparse = coarsened_eigenvalues(csr_matrix(L), labels, k=10, seed=seed)
        assert np.allclose(sparse, dense, rtol=1e-8, atol=1e-10)


def test_sweep_cut_prefers_the_balanced_side_to_the_smallest_cut():
    """Two groups of six buses (a ring with four
    chords each, 10 branches) joined by two branches, a thirteenth bus hanging on one branch at bus 0, all
    weights one. By counting: the smallest cut severs the hanging bus, cut 1, volume 1, conductance 1; the
    side made of the first group and the hanging bus has cut 2 and volume 2*10 + 1*2 (its own branches
    twice, the hanging branch twice) + 2 (the severed ones once) = 22 + 2 = 24 against 2*10 + 2 = 22 on the
    other side, hence conductance 2/22 = 1/11. The sweep cut must return it, within Cheeger's bounds."""
    from gridnet.graph import laplacian
    from gridnet.spectral import conductance, cut_weight, fiedler, sweep_cut
    ring = lambda o: [(o + i, o + (i + 1) % 6) for i in range(6)]
    E = ring(0) + [(0, 2), (0, 3), (1, 4), (3, 5)] + ring(6) + [(6, 8), (6, 9), (7, 10), (9, 11)] + [(2, 11), (3, 10), (0, 12)]
    L = laplacian(13, E, [1.0] * len(E))
    hanging = np.zeros(13, bool); hanging[12] = True
    assert cut_weight(L, hanging) == 1.0 and conductance(L, hanging) == 1.0
    lam, v = fiedler(L, normalized=True)
    side, h = sweep_cut(L, v)
    assert h == pytest.approx(1 / 11) and cut_weight(L, side) == pytest.approx(2.0)
    assert set(np.flatnonzero(side)) in ({0, 1, 2, 3, 4, 5, 12}, {6, 7, 8, 9, 10, 11})
    assert 0.5 * lam <= h <= np.sqrt(2 * lam)


def test_recursive_bipartition_splits_a_disconnected_part_into_its_components_first():
    """A part that falls apart has a cut of weight zero, conductance 0, and is split along its components
    before any connected part. An isolated node has degree zero and no normalized Laplacian (von Luxburg 2007,
    Sec. 3.2), so the split cannot come from the Fiedler vector; here the path 0-1-2 and the isolated node 3."""
    from gridnet.spectral import recursive_bipartition

    hist = []
    labels = recursive_bipartition(laplacian(4, [(0, 1), (1, 2)]), 2, history=hist)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1])
    assert hist[0]["conductance"] == 0.0
    # three components of two nodes: the component of the part's first node stays, the rest splits off
    three = laplacian(6, [(0, 1), (2, 3), (4, 5)])
    np.testing.assert_array_equal(recursive_bipartition(three, 3), [0, 0, 1, 1, 2, 2])
    for normalized in (True, False):
        np.testing.assert_array_equal(recursive_bipartition(laplacian(4, [(0, 1), (1, 2)]), 3, normalized=normalized)[3], 1)


def test_partition_labels_must_be_consecutive_from_zero():
    """The coarsening functions index their parts by the labels 0..k-1; an unused label would be an empty
    part (a zero column of the indicator matrix, a division by its size zero), so it is refused."""
    from gridnet.spectral import area_parameters, coarsened_eigenvalues, restricted_spectral_error

    L = _two_cliques()
    gap = np.array([0, 0, 0, 0, 2, 2, 2, 2])
    for f in (lambda: coarsened_eigenvalues(L, gap), lambda: coarsened_eigenvalues(L, gap, k=2),
              lambda: restricted_spectral_error(L, gap, 2),
              lambda: area_parameters([(0, 1), (1, 2), (2, 3), (3, 4)], [0, 0, 2, 2, 2])):
        with pytest.raises(ValueError, match="labels"):
            f()


def test_restricted_spectral_error_needs_a_connected_graph():
    """eps_k divides by sqrt(lambda_2), ..., sqrt(lambda_k) (Loukas 2019, Def. 3.1, x = V Lambda^{-1/2} y); a
    disconnected graph has lambda_2 = 0 and the constant is undefined."""
    from gridnet.spectral import restricted_spectral_error

    L = laplacian(4, [(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="connected"):
        restricted_spectral_error(L, [0, 0, 1, 1], 2)


def test_loukas_table_1_on_minnesota():
    """Loukas 2019, Table 1, with his public code (graph_coarsening at 1d7aad02ef29, the "coarsening" extra): the mean
    relative error of the first ten eigenvalues on the Minnesota road network at reductions of 30, 50 and 70 %, local
    variation by edges and by neighborhoods, the coarse eigenvalues of C L C^T by shift-invert. Each call gets its own
    copy of the eigenpairs, since his coarsen writes into them. Published 0.088, 0.432, 4.579 and 0.078, 0.310, 1.866;
    the 50 % edge entry and both 70 % entries differ (the first deviation of the audit), the others hold to the printed digits."""
    pytest.importorskip("graph_coarsening")
    pygsp = pytest.importorskip("pygsp")
    import scipy as sp
    import scipy.sparse.linalg
    for name, value in (("int", int), ("float", float)):                  # aliases his code still uses
        if not hasattr(np, name):
            setattr(np, name, value)
    if not hasattr(sp, "newaxis"):
        sp.newaxis = np.newaxis
    from graph_coarsening.coarsening_utils import coarsen

    K = 10
    M = pygsp.graphs.Minnesota()
    G = pygsp.graphs.Graph(M.W.astype(float), coords=M.coords)
    offset = 2 * max(G.dw)
    lk, Uk = sp.sparse.linalg.eigsh(offset * sp.sparse.eye(G.N, format="csc") - G.L, k=K, which="LM", tol=1e-6,
                                    v0=np.random.default_rng(0).standard_normal(G.N))
    lk, Uk = (offset - lk)[::-1], Uk[:, ::-1]

    def error(C):
        Lc = (C @ G.L @ C.T).tocsc(); Lc = (Lc + Lc.T) / 2
        lc = np.sort(sp.sparse.linalg.eigsh(Lc, k=K, sigma=-1e-3 * Lc.diagonal().max(), which="LM", return_eigenvectors=False))
        return float(np.abs(lc[1:] / lk[1:] - 1).sum() / K)

    found = {(method, r): error(coarsen(G, K=K, r=r, max_levels=10, method=method, algorithm="greedy", Uk=Uk.copy(), lk=lk.copy())[0])
             for method in ("variation_edges", "variation_neighborhoods") for r in (0.3, 0.5, 0.7)}
    published = {("variation_edges", 0.3): 0.088, ("variation_neighborhoods", 0.3): 0.078, ("variation_neighborhoods", 0.5): 0.310}
    for key, value in published.items():
        assert round(found[key], 3) == value, (key, found[key])
    # the deviating entries, as found (first deviation): 0.412, 3.638 and 1.578 against the printed 0.432, 4.579 and 1.866
    np.testing.assert_allclose([found[("variation_edges", 0.5)], found[("variation_edges", 0.7)], found[("variation_neighborhoods", 0.7)]],
                               [0.412, 3.638, 1.578], atol=1e-3)


def test_normalized_laplacian_refuses_a_loopy_laplacian():
    """The degrees are read from the diagonal, which counts a shunt as degree; a matrix whose rows do not sum to
    zero is refused rather than normalized with the wrong degrees (code/01 first #11)."""
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    normalized_laplacian(L)
    with pytest.raises(ValueError, match="loop-less"):
        normalized_laplacian(L + np.diag([0.3, 0.0, 0.0]))
