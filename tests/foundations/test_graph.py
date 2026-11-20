"""Tests for gridnet.graph: oriented incidence matrix and weighted Laplacian.

Analytical references (closed-form spectra of the unit-weight Laplacian):
  path P_n:      lambda_k = 2 - 2 cos(pi k / n),      k = 0, ..., n-1
  cycle C_n:     lambda_k = 2 - 2 cos(2 pi k / n),    k = 0, ..., n-1
  complete K_n:  0 once, n with multiplicity n-1
These are standard; see e.g. Brouwer & Haemers, Spectra of Graphs (2012), Secs. 1.4.1 (complete), 1.4.3 (cycle), 1.4.4 (path).
The Laplacian factorization L = A W A^T with A the oriented incidence matrix is
von Luxburg (2007), Sec. 3.1, Prop. 1 (there written as the quadratic form).
"""

import numpy as np
import pytest

from gridnet.graph import effective_resistance, incidence_matrix, laplacian, laplacian_pseudoinverse


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n):
    return path_edges(n) + [(n - 1, 0)]


def complete_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# --- incidence matrix -------------------------------------------------------

def test_incidence_shape_and_entries():
    # edge 0: 0 -> 1, edge 1: 1 -> 2 on three nodes
    A = incidence_matrix(3, [(0, 1), (1, 2)])
    assert A.shape == (3, 2)
    # convention: +1 at the tail (from-node), -1 at the head (to-node)
    np.testing.assert_array_equal(A, [[1, 0], [-1, 1], [0, -1]])


def test_incidence_columns_sum_to_zero():
    A = incidence_matrix(6, cycle_edges(6))
    np.testing.assert_array_equal(A.sum(axis=0), np.zeros(6))


@pytest.mark.parametrize("edges,components", [
    (path_edges(5), 1),
    (cycle_edges(5), 1),
    (path_edges(3) + [(3, 4)], 2),   # two components on 5 nodes
])
def test_incidence_rank_is_nodes_minus_components(edges, components):
    A = incidence_matrix(5, edges)
    assert np.linalg.matrix_rank(A) == 5 - components


def test_incidence_rejects_self_loop_and_bad_index():
    with pytest.raises(ValueError):
        incidence_matrix(3, [(1, 1)])
    with pytest.raises(ValueError):
        incidence_matrix(3, [(0, 3)])


# --- Laplacian ---------------------------------------------------------------

def test_laplacian_equals_A_W_At():
    edges = cycle_edges(5)
    A = incidence_matrix(5, edges)
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    L = laplacian(5, edges, w)
    np.testing.assert_allclose(L, A @ np.diag(w) @ A.T)


def test_laplacian_is_symmetric_psd_with_constant_kernel():
    edges = complete_edges(4)
    L = laplacian(4, edges, np.arange(1, 7, dtype=float))
    np.testing.assert_allclose(L, L.T)
    assert np.all(np.linalg.eigvalsh(L) > -1e-12)
    np.testing.assert_allclose(L @ np.ones(4), 0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 5, 12])
def test_path_spectrum(n):
    L = laplacian(n, path_edges(n))
    expected = 2 - 2 * np.cos(np.pi * np.arange(n) / n)
    np.testing.assert_allclose(np.linalg.eigvalsh(L), np.sort(expected), atol=1e-12)


@pytest.mark.parametrize("n", [3, 6, 11])
def test_cycle_spectrum(n):
    L = laplacian(n, cycle_edges(n))
    expected = 2 - 2 * np.cos(2 * np.pi * np.arange(n) / n)
    np.testing.assert_allclose(np.linalg.eigvalsh(L), np.sort(expected), atol=1e-12)


@pytest.mark.parametrize("n", [2, 4, 9])
def test_complete_spectrum(n):
    L = laplacian(n, complete_edges(n))
    expected = np.array([0.0] + [float(n)] * (n - 1))
    np.testing.assert_allclose(np.linalg.eigvalsh(L), expected, atol=1e-12)


def test_laplacian_default_weights_are_one():
    edges = path_edges(4)
    np.testing.assert_allclose(laplacian(4, edges), laplacian(4, edges, np.ones(3)))


def test_laplacian_rejects_wrong_weight_length():
    with pytest.raises(ValueError):
        laplacian(4, path_edges(4), np.ones(2))


# --- Laplacian pseudoinverse and effective resistance --------------------------
# Effective resistance R_ij = (e_i - e_j)^T L^+ (e_i - e_j): Dorfler & Bullo, IEEE Trans. Circuits Syst. I
# 60(1) (2013), eq. (1.3) and Remark 1.1. Closed forms: path of unit edges R_0,n-1 = n-1
# (series); ring of n unit edges R_ij = d(n-d)/n for graph distance d (two parallel paths
# of d and n-d resistors); complete graph K_n: R_ij = 2/n.


def test_pseudoinverse_is_inverse_on_the_orthogonal_complement_of_ones():
    L = laplacian(5, complete_edges(5), np.arange(1, 11, dtype=float))
    Lp = laplacian_pseudoinverse(L)
    P = np.eye(5) - np.ones((5, 5)) / 5           # projector onto 1^perp
    np.testing.assert_allclose(L @ Lp, P, atol=1e-12)
    np.testing.assert_allclose(Lp @ np.ones(5), 0, atol=1e-12)


def test_effective_resistance_path_ring_complete():
    n = 6
    R = effective_resistance(laplacian(n, path_edges(n)))
    assert np.isclose(R[0, n - 1], n - 1)
    assert np.isclose(R[1, 3], 2)
    R = effective_resistance(laplacian(n, cycle_edges(n)))
    for d in range(1, n):
        assert np.isclose(R[0, d], d * (n - d) / n)
    R = effective_resistance(laplacian(n, complete_edges(n)))
    off = R[~np.eye(n, dtype=bool)]
    np.testing.assert_allclose(off, 2 / n)
    np.testing.assert_allclose(np.diag(R), 0, atol=1e-14)


def test_effective_resistance_scales_inversely_with_weight():
    n, edges = 4, complete_edges(4)
    R1 = effective_resistance(laplacian(n, edges))
    R3 = effective_resistance(laplacian(n, edges, 3 * np.ones(6)))
    np.testing.assert_allclose(R3, R1 / 3)


def test_pseudoinverse_of_a_loopy_laplacian_is_its_inverse():
    """A loopy Laplacian, a Laplacian plus a nonnegative diagonal of shunts (Dorfler & Bullo 2013,
    Sec. II), is invertible as soon as one shunt is positive on every component; its constants are not a
    kernel, and the Moore-Penrose inverse is the inverse. One component with a shunt, one without.
    The shunt is recognized only when the row sums exceed 1e-6 of the smallest nonzero eigenvalue on the
    zero-sum vectors; below that they are taken for the rounding of a reduction formed by cancellation, the
    constants are removed, and the result is not the inverse (off by 1e8 at a shunt of 1e-8)."""
    L = laplacian(5, [(0, 1), (1, 2), (0, 2), (3, 4)], [1.0, 2.0, 0.5, 3.0])
    Q = L + np.diag([0.4, 0.0, 0.0, 0.0, 0.0])
    P = laplacian_pseudoinverse(Q)
    np.testing.assert_allclose(P[:3, :3], np.linalg.inv(Q[:3, :3]), atol=1e-12)
    np.testing.assert_allclose(P, np.linalg.pinv(Q), atol=1e-12)


def test_effective_resistance_is_infinite_across_components():
    """R_ij = L^+_ii + L^+_jj - 2 L^+_ij (Dorfler & Bullo 2013, eq. (1.3)) is the potential difference
    for a unit current from i to j, which exists only if i and j are connected; across components no
    current can flow and the resistance is infinite. The pseudoinverse formula alone returns finite
    numbers there, 0 between two isolated nodes (found 2026-09-18 on a disconnected sparsifier)."""
    from gridnet.graph import effective_resistance, laplacian

    L = laplacian(5, [(0, 1), (1, 2)], [2.0, 2.0])               # path 0-1-2, nodes 3 and 4 isolated
    R = effective_resistance(L)
    assert R[0, 1] == pytest.approx(0.5) and R[0, 2] == pytest.approx(1.0)
    assert np.all(np.isinf(R[:3, 3:])) and np.isinf(R[3, 4]) and np.all(np.diag(R) == 0.0)
    assert np.allclose(R, R.T, equal_nan=False)


# Closed form for a Laplacian with one negative weight (a series capacitor, x < 0, as in MATPOWER's
# case300 and case9241pegase): the triangle with w_01 = w_12 = 1 and w_02 = -0.8. Its nonzero eigenvalues
# are the roots of lam^2 - 2 (a + b + c) lam + 3 (ab + bc + ca) (a, b, c the three weights), here
# lam^2 - 2.4 lam - 1.8, so lam = 3 and lam = -0.6: L is indefinite. Effective resistances by the series
# and parallel rules, which hold for any nonzero weights: between 0 and 2 the direct edge (conductance
# -0.8) in parallel with the two-edge path (conductance 1/2), R_02 = 1/(-0.3) = -10/3; between 0 and 1 the
# edge (1) in parallel with the path 0-2-1 (resistance -1.25 + 1 = -0.25, conductance -4), R_01 = -1/3.
INDEFINITE_TRIANGLE = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, -0.8])


def test_pseudoinverse_of_an_indefinite_laplacian_is_the_moore_penrose_inverse():
    L = INDEFINITE_TRIANGLE
    np.testing.assert_allclose(np.linalg.eigvalsh(L), [-0.6, 0.0, 3.0], atol=1e-12)
    Lp = laplacian_pseudoinverse(L)
    np.testing.assert_allclose(Lp, np.linalg.pinv(L), atol=1e-12)
    np.testing.assert_allclose(L @ Lp @ L, L, atol=1e-12)                 # the Penrose conditions
    np.testing.assert_allclose(Lp @ L @ Lp, Lp, atol=1e-12)


def test_effective_resistance_with_a_negative_weight_follows_the_series_and_parallel_rules():
    R = effective_resistance(INDEFINITE_TRIANGLE)
    assert R[0, 2] == pytest.approx(-10 / 3) and R[0, 1] == pytest.approx(-1 / 3)
    assert R[1, 2] == pytest.approx(-1 / 3)                                  # exchanging nodes 0 and 2 maps L to itself


@pytest.mark.data("matpower")
def test_pseudoinverse_of_case300_with_its_series_capacitor():
    """MATPOWER's case300 has one branch with x = -0.3697, a series capacitor, and its Bbus has a negative
    eigenvalue; the pseudoinverse must still satisfy the Penrose conditions and agree with numpy's SVD-based
    pinv."""
    from pathlib import Path

    from gridnet.admittance import dc_model
    from gridnet.network import read_matpower_case

    net = read_matpower_case(Path(__file__).resolve().parents[2] / "data" / "matpower" / "case300.m")
    B = dc_model(net)[0]
    assert np.linalg.eigvalsh(B)[0] < -1.0
    P = laplacian_pseudoinverse(B)
    scale = np.abs(B).max()
    np.testing.assert_allclose(B @ P @ B, B, atol=1e-9 * scale)
    np.testing.assert_allclose(P, np.linalg.pinv(B), atol=1e-9 * np.abs(P).max())


# --- rerouting distance (Strake, Kaiser, Basiri, Ronellenfitsch & Witthaut 2019, Definition 2, Appendix D) ---

def test_rerouting_distance_on_a_ring_is_the_ring():
    """On a ring every cycle through two edges is the ring itself."""
    from gridnet.graph import rerouting_distances

    edges = [(i, (i + 1) % 6) for i in range(6)]
    for disjoint in ("vertex", "edge"):
        d = rerouting_distances(6, edges, 0, disjoint=disjoint)
        assert d[0] == 0 and np.all(d[1:] == 6)
        w = np.arange(1.0, 7.0)
        np.testing.assert_allclose(rerouting_distances(6, edges, 2, weights=w, disjoint=disjoint)[[0, 5]], [21.0, 21.0])


def test_rerouting_distance_by_hand_on_a_square_with_a_diagonal_and_a_pendant():
    """Square 0-1-2-3 with the diagonal 0-2 and the pendant edge 3-4. Edge 0-1 (index 0): with 1-2 the
    triangle 0-1-2 (3), with 2-3 and 3-0 the square (4), with the diagonal the triangle (3); the pendant
    lies on no cycle (inf). Two parallel copies of an edge form a cycle of length 2."""
    from gridnet.graph import rerouting_distances

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]
    for disjoint in ("vertex", "edge"):
        np.testing.assert_array_equal(rerouting_distances(5, edges, 0, disjoint=disjoint), [0, 3, 4, 4, 3, np.inf])
        np.testing.assert_array_equal(rerouting_distances(5, edges, 5, disjoint=disjoint), [np.inf] * 5 + [0])
        np.testing.assert_array_equal(rerouting_distances(5, edges + [(1, 0)], 0, disjoint=disjoint)[[1, 6]], [3, 2])


def test_rerouting_distance_vertex_against_edge_disjoint_on_a_figure_eight():
    """Two triangles 0-1-2 and 2-3-4 that share only vertex 2. Edges 0-1 and 3-4 lie on no common simple
    cycle (Definition 2: inf), but the closed walk 0-1-2-3-4-2-0 repeats only a vertex, so two edge-disjoint
    paths exist (Lemma 4's mapping: 6)."""
    from gridnet.graph import rerouting_distances

    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
    assert rerouting_distances(5, edges, 0, disjoint="vertex")[4] == np.inf
    assert rerouting_distances(5, edges, 0, disjoint="edge")[4] == 6


@pytest.mark.parametrize("seed", range(6))
def test_rerouting_distance_against_every_simple_cycle(seed):
    """Definition 2 directly: the shortest simple cycle containing both edges, found by enumerating every
    simple cycle of a small random graph with random positive weights; and the metric of Lemma 3
    (symmetry, triangle inequality) for both variants."""
    import networkx as nx

    from gridnet.graph import rerouting_distances

    rng = np.random.default_rng(seed)
    G = nx.gnm_random_graph(8, 13, seed=seed)
    edges = list(G.edges()); m = len(edges); w = rng.uniform(0.5, 2.0, m)
    index = {frozenset(e): k for k, e in enumerate(edges)}
    best = np.full((m, m), np.inf); np.fill_diagonal(best, 0.0)
    for cycle in nx.simple_cycles(G):
        if len(cycle) < 3:
            continue
        ks = [index[frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))] for i in range(len(cycle))]
        length = w[ks].sum()
        for a in ks:
            for c in ks:
                if a != c:
                    best[a, c] = min(best[a, c], length)
    D = {v: np.array([rerouting_distances(8, edges, k, weights=w, disjoint=v) for k in range(m)]) for v in ("vertex", "edge")}
    np.testing.assert_allclose(D["vertex"], best)
    for d in D.values():
        np.testing.assert_allclose(d, d.T)
        assert np.all(D["edge"] <= D["vertex"] + 1e-12)
        for b in range(m):
            assert np.all(d <= d[:, [b]] + d[[b], :] + 1e-9)


def test_rerouting_distances_refuses_self_loops_and_indices_out_of_range():
    """As `incidence_matrix` does (code/01 second M16)."""
    from gridnet.graph import rerouting_distances

    with pytest.raises(ValueError, match="self-loop"):
        rerouting_distances(3, [(0, 1), (1, 2), (2, 2)], 0)
    with pytest.raises(ValueError, match="outside"):
        rerouting_distances(3, [(0, 1), (1, 2), (2, 5)], 0)


def test_a_connected_laplacian_too_badly_conditioned_to_resolve_is_refused():
    """A connected graph with positive weights has exactly one zero eigenvalue, so an eigenvalue
    below the cutoff m eps max|lambda| on the zero-sum vectors is no kernel there: it is a
    genuine lambda_2 that the matrix no longer determines (the weak weight is lost in the
    rounding of the diagonal entries it was added to). Treated as zero it would glue nothing
    and give a small finite resistance across the weak branch. A path of 2,000 buses with
    weights 1e3 and one of 1e-7: the resistance from end to end is the series sum,
    1,999/1e3 + 1e7, and the code returned 0.666. Two cliques of 50 nodes with weights 1e4
    joined by a bridge of 1e-8: about 1e8 across, the code returned 3.9e-6. Both are refused
    now. With a negative weight a further kernel can be real, and the cutoff stays: the
    triangle with weights 1, 1 and -1/2 has rank one."""
    from gridnet.graph import laplacian_pseudoinverse, range_whitening

    n = 2000
    w = np.full(n - 1, 1e3); w[n // 2] = 1e-7
    path = laplacian(n, [(k, k + 1) for k in range(n - 1)], w)
    edges = [(i, j) for i in range(50) for j in range(i)] + [(50 + i, 50 + j) for i in range(50) for j in range(i)] + [(0, 50)]
    cliques = laplacian(100, edges, [1e4] * (len(edges) - 1) + [1e-8])
    for L in (path, cliques):
        for f in (effective_resistance, laplacian_pseudoinverse, range_whitening):
            with pytest.raises(ValueError, match="too badly conditioned"):
                f(L)
    # still resolved: the same path with the weak weight at 1e-3 (condition number about 1e12 on 2,000 buses)
    w[n // 2] = 1e-3
    R = effective_resistance(laplacian(n, [(k, k + 1) for k in range(n - 1)], w))
    assert R[0, -1] == pytest.approx(np.sum(1 / w), rel=1e-6)
    # a negative weight: series 1 and 1 give 1/2, in parallel with -1/2 the triangle carries no current between its ends
    tri = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, -0.5])
    assert np.linalg.matrix_rank(laplacian_pseudoinverse(tri), tol=1e-9) == 1


def test_rounding_in_the_row_sums_is_not_taken_for_a_shunt():
    """The row sums of a Laplacian built from its weights are rounding, of the size of eps
    times the weights of the row. On a graph with weights from 1e-4 to 1e7 they reach 1e-9,
    above 1e-6 times a smallest eigenvalue of 1e-4: measured against the spectrum alone the
    matrix passed for a loopy Laplacian, its constants were not removed, and L^+ 1 was 0.4
    where it is zero. A component counts as loopy only when a row sum also exceeds the rounding
    of the sum of its own entries. Over 300 random graphs of 30 buses (a path plus 20 branches,
    weights 10^U(-4, 7)): L^+ 1 = 0 to rounding and the Penrose condition L L^+ L = L on every
    one (the second to the rounding of its products); a true shunt of 1e-3 at one bus is still seen (L then is regular and L^+ its inverse)."""
    from gridnet.graph import laplacian_pseudoinverse

    rng = np.random.default_rng(1)
    hard = 0
    for _ in range(300):
        n = 30
        edges = [(i, i + 1) for i in range(n - 1)] + [tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(20)]
        L = laplacian(n, edges, 10 ** rng.uniform(-4, 7, len(edges)))
        hard += np.abs(L.sum(axis=1)).max() > 1e-6 * np.linalg.eigvalsh(L)[1]
        Lp = laplacian_pseudoinverse(L)
        assert np.abs(Lp.sum(axis=1)).max() <= 1e-10 * np.abs(Lp).max()
        # Penrose's L L^+ L = L to the rounding of the two products, eps |L|^2 |L^+| (the condition number times eps |L|)
        assert np.abs(L @ Lp @ L - L).max() <= 100 * np.finfo(float).eps * np.abs(L).max() ** 2 * np.abs(Lp).max()
    assert hard >= 5                                            # the cases the spectrum rule misread
    L = laplacian(4, [(0, 1), (1, 2), (2, 3)], [1.0, 2.0, 3.0])
    L[0, 0] += 1e-3
    np.testing.assert_allclose(laplacian_pseudoinverse(L) @ L, np.eye(4), atol=1e-9)
