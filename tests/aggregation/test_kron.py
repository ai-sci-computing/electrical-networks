"""Tests for gridnet.kron: Kron reduction (Schur complement) of Laplacian-like matrices.

Reference: Dorfler & Bullo, IEEE Trans. Circuits Syst. I 60(1) (2013):
  definition Q_red = Q/Q(alpha, alpha) = Q[a,a] - Q[a,b] Q[b,b]^{-1} Q[b,a]   eq. (2.1)
  reduced matrix is again a (loopy) Laplacian                             Lemma 2.1
  one-step reduction equals iterative single-node reduction               Lemma 3.3
  eigenvalue interlacing                                                  Thm. 3.5
  effective resistance among boundary nodes is invariant                  Thm. 3.9
Closed forms: series reduction of a path (two conductances in series) and the star-mesh
transform (a star with center conductances b_i becomes a complete graph with weights
b_i b_j / sum b), both elementary consequences of eq. (2.1).
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.admittance import bus_admittance
from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import kron_reduce, laplacian_to_graph

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


def test_series_reduction_of_a_path():
    b1, b2 = 2.0, 3.0
    L = laplacian(3, [(0, 1), (1, 2)], [b1, b2])
    Lr = kron_reduce(L, keep=[0, 2])
    w = b1 * b2 / (b1 + b2)
    np.testing.assert_allclose(Lr, w * np.array([[1, -1], [-1, 1]]))


def test_star_mesh_transform():
    b = np.array([1.0, 2.0, 3.0, 4.0])
    edges = [(0, i) for i in range(1, 5)]                  # center 0, leaves 1..4
    L = laplacian(5, edges, b)
    Lr = kron_reduce(L, keep=[1, 2, 3, 4])
    expected = -np.outer(b, b) / b.sum()
    np.fill_diagonal(expected, 0)
    np.fill_diagonal(expected, -expected.sum(axis=1))
    np.testing.assert_allclose(Lr, expected)


@pytest.mark.data("matpower")
def test_reduced_matrix_is_a_laplacian_lemma_2_1():
    from gridnet.network import read_matpower_case
    net = read_matpower_case(CASES / "case14.m")
    L = laplacian(net.n_bus, net.edges, 1 / net.x)
    keep = [0, 1, 2, 5, 7]
    Lr = kron_reduce(L, keep)
    np.testing.assert_allclose(Lr, Lr.T, atol=1e-12)
    np.testing.assert_allclose(Lr.sum(axis=1), 0, atol=1e-12)          # loop-less; true by construction since 5b6c318
    # (kron_reduce sets the diagonal from the off-diagonal), kept as a guard; the symmetry and the sign are the content
    off = Lr[~np.eye(len(keep), dtype=bool)]
    assert np.all(off <= 1e-12)                                          # nonnegative weights
    edges, weights, shunts = laplacian_to_graph(Lr)
    assert np.all(weights > 0) and np.allclose(shunts, 0, atol=1e-12)
    np.testing.assert_allclose(laplacian(len(keep), edges, weights), Lr, atol=1e-12)


def test_effective_resistance_invariance_thm_3_9():
    rng = np.random.default_rng(0)
    n = 9
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5] + [(i, i + 1) for i in range(n - 1)]
    L = laplacian(n, edges, rng.uniform(0.5, 2, len(edges)))
    keep = [0, 3, 4, 8]
    R = effective_resistance(L)
    Rr = effective_resistance(kron_reduce(L, keep))
    np.testing.assert_allclose(Rr, R[np.ix_(keep, keep)], atol=1e-12)


def test_iterative_equals_one_step_lemma_3_3():
    rng = np.random.default_rng(1)
    n = 7
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    L = laplacian(n, edges, rng.uniform(0.1, 1, len(edges)))
    keep = [0, 1, 2]
    one_step = kron_reduce(L, keep)
    M = L.copy()
    for interior in [6, 5, 4, 3]:                       # eliminate one node at a time
        M = kron_reduce(M, [i for i in range(M.shape[0]) if i != interior])
    np.testing.assert_allclose(M, one_step, atol=1e-12)


def test_interlacing_thm_3_5():
    rng = np.random.default_rng(2)
    n = 8
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    L = laplacian(n, edges, rng.uniform(0.1, 1, len(edges)))
    keep = [0, 2, 4, 6, 7]
    lam = np.linalg.eigvalsh(L)
    lam_r = np.linalg.eigvalsh(kron_reduce(L, keep))
    k = len(keep)
    for r in range(k):
        assert lam[r] - 1e-12 <= lam_r[r] <= lam[r + n - k] + 1e-12


def test_loopy_laplacian_with_shunt_at_eliminated_node():
    b1, b2, g = 2.0, 3.0, 0.5
    Q = laplacian(3, [(0, 1), (1, 2)], [b1, b2]) + np.diag([0, g, 0])
    Qr = kron_reduce(Q, [0, 2])
    s = b1 + b2 + g
    expected = np.array([[b1 - b1**2 / s, -b1 * b2 / s], [-b1 * b2 / s, b2 - b2**2 / s]])
    np.testing.assert_allclose(Qr, expected)
    _, _, shunts = laplacian_to_graph(Qr)
    assert np.all(shunts > 0)                                            # the shunt spreads to both ends


def test_singular_interior_raises():
    # interior node 2 is disconnected from the boundary {0, 1}: Q(beta, beta) singular
    L = laplacian(4, [(0, 1), (2, 3)])
    with pytest.raises(ValueError):
        kron_reduce(L, keep=[0, 1])


@pytest.mark.data("matpower")
def test_complex_admittance_reduction_is_the_dirichlet_to_neumann_map():
    """For zero interior current injections, boundary voltages from Y and Y_red agree."""
    from gridnet.network import read_matpower_case
    net = read_matpower_case(CASES / "case14.m")
    Y = bus_admittance(net)                                  # nonsingular thanks to shunts/charging
    keep = np.array([0, 1, 2, 5, 7])
    Yr = kron_reduce(Y, keep)
    rng = np.random.default_rng(3)
    I = np.zeros(net.n_bus, dtype=complex)
    I[keep] = rng.normal(size=5) + 1j * rng.normal(size=5)
    V = np.linalg.solve(Y, I)
    Vr = np.linalg.solve(Yr, I[keep])
    np.testing.assert_allclose(Vr, V[keep], atol=1e-12)


# --- helpers for comparing a reduction with an aggregation -------------------------------


def test_pick_representatives_nearest_to_centroid_and_by_degree():
    from gridnet.kron import pick_representatives

    xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    labels = np.array(["a", "a", "b", "b", "b"])
    centroids = {"a": (0.9, 0.0), "b": (11.9, 0.0)}
    assert pick_representatives(xy, labels, centroids, mode="nearest") == {"a": 1, "b": 4}
    degrees = np.array([5.0, 1.0, 1.0, 7.0, 2.0])
    assert pick_representatives(xy, labels, centroids, mode="degree", degrees=degrees) == {"a": 0, "b": 3}
    with pytest.raises(ValueError):
        pick_representatives(xy, labels, centroids, mode="degree")


def test_reduced_couplings_triangle_matches_star_mesh_and_thm_3_9():
    from gridnet.kron import reduced_couplings

    b01, b02, b12 = 1.0, 2.0, 4.0
    L = laplacian(3, [(0, 1), (0, 2), (1, 2)], [b01, b02, b12])
    W, R = reduced_couplings(L, keep=[0, 1])
    # eliminating node 2 adds its two edges in series to the direct coupling
    w = b01 + b02 * b12 / (b02 + b12)
    np.testing.assert_allclose(W, [[0, w], [w, 0]])
    np.testing.assert_allclose(R, effective_resistance(L)[:2, :2], atol=1e-12)
    np.testing.assert_allclose(R[0, 1], 1 / w)


def test_laplacian_from_couplings_with_and_without_mask():
    from gridnet.kron import laplacian_from_couplings, reduced_couplings

    b = np.array([1.0, 2.0, 3.0, 4.0])
    L = laplacian(5, [(0, i) for i in range(1, 5)], b)
    W, R = reduced_couplings(L, keep=[1, 2, 3, 4])
    # all couplings kept: the Kron-reduced Laplacian itself, and the exact resistances
    Lr = laplacian_from_couplings(W)
    np.testing.assert_allclose(Lr, kron_reduce(L, [1, 2, 3, 4]), atol=1e-12)
    np.testing.assert_allclose(effective_resistance(Lr), R, atol=1e-12)
    # only a ring of the four kept: dropped couplings raise every resistance
    ring = np.zeros((4, 4), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ring[i, j] = ring[j, i] = True
    Ls = laplacian_from_couplings(W, mask=ring)
    assert np.count_nonzero(Ls) == 4 + 8
    Rs = effective_resistance(Ls)
    iu = np.triu_indices(4, 1)
    assert np.all(Rs[iu] > R[iu])


# --- the accompanying matrix: where the eliminated injections go ------------------------------------

def test_accompanying_matrix_of_a_path_splits_by_conductance():
    """Path 0-1-2 with conductances b1, b2, node 1 eliminated: its injection goes to the ends in the
    proportion b1 : b2 (current division), from Q_ac = -Q[a,b] Q[b,b]^{-1} (Dorfler & Bullo 2013, Sec. I
    and Lemma 2.1(3); Ward 1949's distribution factors; Tinney & Powell 1977, eq. (4))."""
    from gridnet.kron import accompanying_matrix

    b1, b2 = 2.0, 3.0
    D = accompanying_matrix(laplacian(3, [(0, 1), (1, 2)], [b1, b2]), keep=[0, 2])
    assert D.shape == (2, 3)
    assert np.allclose(D, [[1.0, b1 / (b1 + b2), 0.0], [0.0, b2 / (b1 + b2), 1.0]])


def test_accompanying_matrix_is_column_stochastic_and_makes_the_reduction_exact():
    """Lemma 2.1(3): nonnegative, and column stochastic for a Laplacian without shunts. With the injections
    I_a + Q_ac I_b the Kron-reduced network reproduces the potentials of the kept nodes for any balanced
    injection (Dorfler & Bullo 2013, Sec. I)."""
    from gridnet.kron import accompanying_matrix

    rng = np.random.default_rng(0)
    n = 12
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 5), (2, 9), (3, 7), (4, 10)]
    L = laplacian(n, edges, rng.uniform(0.5, 2.0, len(edges)))
    keep = [1, 4, 8, 11]
    D = accompanying_matrix(L, keep)
    assert np.all(D >= -1e-14) and np.allclose(D.sum(0), 1.0)
    assert np.allclose(D[:, keep], np.eye(4))
    p = rng.normal(size=n); p -= p.mean()
    theta = np.linalg.pinv(L) @ p
    theta_red = np.linalg.pinv(kron_reduce(L, keep)) @ (D @ p)
    assert np.allclose(theta[keep] - theta[keep].mean(), theta_red - theta_red.mean(), atol=1e-10)


def test_accompanying_matrix_holds_the_hitting_probabilities_of_the_random_walk():
    """Doyle & Snell 2000, Sec. 1.3.2: the voltage with boundary value 1 at a and 0 at the other kept nodes is the
    probability that the walk with transition probabilities w_xy / w_x, started at x, reaches a first. For an
    absorbing chain these are B = (I - P_bb)^{-1} P_ba, and column x of the accompanying matrix is row x of B."""
    from gridnet.kron import accompanying_matrix

    rng = np.random.default_rng(1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (2, 5), (5, 4)]
    L = laplacian(6, edges, rng.uniform(0.5, 2.0, len(edges)))
    keep, rest = [0, 3], [1, 2, 4, 5]
    P = -L / np.diag(L)[:, None]; np.fill_diagonal(P, 0.0)
    B = np.linalg.solve(np.eye(4) - P[np.ix_(rest, rest)], P[np.ix_(rest, keep)])
    assert np.allclose(accompanying_matrix(L, keep)[:, rest], B.T)


def test_accompanying_matrix_reproduces_doyle_and_snell_example():
    """Doyle & Snell 2000, Sec. 1.3.2, Figs. 16-18: nodes a, b, c, d with conductances a-c 1, c-d 2, c-b 1, d-a 1,
    d-b 2 (their transition probabilities: from c, 1/4 to a and 1/2 to d; from d, 1/5 to a and 2/5 to c). With
    v_a = 1 and v_b = 0 they find v_c = 7/16 and v_d = 3/8, "also the probability, starting at c, of reaching a
    before b". These are the entries of the row of a in the accompanying matrix."""
    from gridnet.kron import accompanying_matrix

    a, b, c, d = 0, 1, 2, 3
    L = laplacian(4, [(a, c), (c, d), (c, b), (d, a), (d, b)], [1.0, 2.0, 1.0, 1.0, 2.0])
    D = accompanying_matrix(L, keep=[a, b])
    assert D[0, c] == pytest.approx(7 / 16) and D[0, d] == pytest.approx(3 / 8)
    assert D[1, c] == pytest.approx(9 / 16) and D[1, d] == pytest.approx(5 / 8)


def test_accompanying_matrix_of_two_triangles_joined_by_a_bridge_in_closed_form():
    """Where the injection of a bus goes when the buses between are eliminated. Triangles 0-1-2 and
    3-4-5, bridge 2-3, all susceptances one, kept buses 0 and 5. The fraction h of a bus's injection that
    reaches bus 0 is harmonic at the eliminated buses with h(0) = 1, h(5) = 0 (Doyle & Snell 2000, Sec. 1.3.2):
    h1 = (1 + h2)/2, h2 = (1 + h1 + h3)/3, h3 = (h2 + h4)/3, h4 = h3/2, solved by 6/7, 5/7, 2/7, 1/7."""
    from gridnet.kron import accompanying_matrix
    L = laplacian(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)], [1.0] * 7)
    D = accompanying_matrix(L, [0, 5])[:, [1, 2, 3, 4]]          # the full matrix carries the identity on the kept buses
    assert np.allclose(D[0], [6 / 7, 5 / 7, 2 / 7, 1 / 7])
    assert np.allclose(D.sum(0), 1.0)


@pytest.mark.parametrize("small, large", [(1e-9, 1e9), (1e-4, 1e8)])
def test_kron_reduction_of_a_path_with_extreme_weights_is_the_series_rule(small, large):
    """A path 0 - 1 - 2 reduced onto its ends is one branch with the series susceptance
    small * large / (small + large). Whether the interior block is singular is a property of the
    graph (an interior component without a branch to the kept nodes); a numerical rank refused this
    valid reduction at weights 1e18 apart."""
    from gridnet.graph import laplacian

    L = laplacian(3, [(0, 1), (1, 2)], [small, large])
    K = kron_reduce(L, [0, 2])
    series = small * large / (small + large)
    np.testing.assert_allclose(K, series * np.array([[1.0, -1.0], [-1.0, 1.0]]), rtol=1e-10)


def test_kron_reduction_refuses_an_interior_component_cut_off_from_the_boundary():
    from gridnet.graph import laplacian

    L = laplacian(4, [(0, 1), (2, 3)], [1.0, 1.0])            # buses 2 and 3 connect to nothing kept
    with pytest.raises(ValueError):
        kron_reduce(L, [0, 1])


def test_reduced_couplings_refuses_a_loopy_laplacian():
    """Its couplings and resistances are those of a loop-less L (rows summing to zero); a shunt on the diagonal
    would be read as coupling and the resistances taken from the wrong matrix (code/01 first #16)."""
    from gridnet.kron import reduced_couplings

    L = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    reduced_couplings(L, [0, 1])
    with pytest.raises(ValueError, match="loop-less"):
        reduced_couplings(L + np.diag([0.0, 0.0, 0.5]), [0, 1])


def test_accompanying_matrix_refuses_an_interior_component_without_a_kept_node():
    """A triangle {2, 3, 4} cut off from the kept nodes {0, 1}: its injections reach no kept
    node, the interior block is singular, and the column sums one of Dorfler & Bullo 2013,
    Lemma 2.1(3), cannot hold. `kron_reduce` refuses this input, and so does the accompanying
    matrix (it returned zero columns for the three nodes, their injections dropped)."""
    from gridnet.kron import accompanying_matrix
    L = laplacian(5, [(0, 1), (2, 3), (3, 4), (2, 4)], [1.0, 0.1, 0.7, 0.3])
    with pytest.raises(ValueError, match="not connected to the boundary"):
        kron_reduce(L, [0, 1])
    with pytest.raises(ValueError, match="not connected to the boundary"):
        accompanying_matrix(L, [0, 1])


# --- the pseudoinverse and the range whitening of `gridnet.graph` on reductions formed by cancellation ---


@pytest.mark.parametrize("internal", [1e2, 1e3, 1e4])
def test_effective_resistance_of_a_kron_reduction_formed_by_cancellation(internal):
    """Kron reduction keeps the effective resistances between the kept nodes (Dorfler & Bullo 2013,
    Thm. 3.9). Five clusters of four buses, each a complete graph of susceptance `internal`, joined in a
    ring with one chord by pairs of lines of susceptance 0.5 to 2, reduced onto one bus per cluster: the
    reduced matrix is formed by cancellation of the large internal susceptances, and its computed zero
    eigenvalue (-5e-14 at internal = 100) exceeds a cutoff relative to the largest eigenvalue, so that a
    pseudoinverse with such a cutoff inverts it and moves the resistances by 0.2 percent."""
    from itertools import combinations

    from gridnet.kron import kron_reduce

    rng = np.random.default_rng(0)
    edges, b = [], []
    for z in range(5):
        for i, j in combinations(range(4), 2):
            edges.append((4 * z + i, 4 * z + j)); b.append(internal)
    for za, zc in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        for i, j in ((0, 1), (2, 3)):
            edges.append((4 * za + i, 4 * zc + j)); b.append(rng.uniform(0.5, 2.0))
    L = laplacian(20, edges, b)
    keep = [0, 4, 8, 12, 16]
    R_full = effective_resistance(L)[np.ix_(keep, keep)]
    R_kron = effective_resistance(kron_reduce(L, keep))
    np.testing.assert_allclose(R_kron, R_full, rtol=1e-10)


@pytest.mark.parametrize("internal", [1e2, 1e4, 1e6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_range_whitening_of_a_kron_reduction_formed_by_cancellation(internal, seed):
    """H = V_r Lambda_r^{-1/2} must whiten L on its range, H^T L H = I, and the spectral error of a
    matrix against itself is zero. On the Kron reductions of the five-cluster network the computed
    zero eigenvalue exceeds a cutoff relative to the largest eigenvalue (the defect fixed in
    `laplacian_pseudoinverse`), and with that cutoff the whitening raised or kept the kernel."""
    from itertools import combinations

    from gridnet.graph import range_whitening
    from gridnet.kron import kron_reduce
    from gridnet.sparsify import spectral_error

    rng = np.random.default_rng(seed)
    edges, b = [], []
    for z in range(5):
        for i, j in combinations(range(4), 2):
            edges.append((4 * z + i, 4 * z + j)); b.append(internal)
    for za, zc in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        for i, j in ((0, 1), (2, 3)):
            edges.append((4 * za + i, 4 * zc + j)); b.append(rng.uniform(0.5, 2.0))
    K = kron_reduce(laplacian(20, edges, b), [0, 4, 8, 12, 16])
    H = range_whitening(K)
    assert H.shape == (5, 4)
    np.testing.assert_allclose(H.T @ K @ H, np.eye(4), atol=1e-9)
    assert spectral_error(K, K) < 1e-9


def _clusters(internal, seed):
    """The five-cluster network of the two tests above, with the internal susceptances spread by a factor of
    four around `internal` (so that their row sums do not cancel exactly in binary arithmetic)."""
    from itertools import combinations

    rng = np.random.default_rng(seed)
    edges, b = [], []
    for z in range(5):
        for i, j in combinations(range(4), 2):
            edges.append((4 * z + i, 4 * z + j)); b.append(internal * rng.uniform(0.5, 2.0))
    for za, zc in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        for i, j in ((0, 1), (2, 3)):
            edges.append((4 * za + i, 4 * zc + j)); b.append(rng.uniform(0.5, 2.0))
    return edges, np.array(b)


@pytest.mark.parametrize("internal", [1e6, 1e10, 1e12])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kron_reduction_of_a_laplacian_has_rows_summing_to_zero(internal, seed):
    """The Kron reduction of a Laplacian is a Laplacian (Dorfler & Bullo 2013, Lemma 2.1): its rows sum to
    zero. In floating point the row sums of L are not zero but rounding, of the order of eps times the
    largest weight at a node; taken as shunts and carried through the reduction they reached 1e-4 of the
    largest reduced entry at internal = 1e12 (audit of 2026-11-20, A2). Row sums within their own
    rounding are zero."""
    edges, b = _clusters(internal, seed)
    K = kron_reduce(laplacian(20, edges, b), [0, 4, 8, 12, 16])
    assert np.abs(K.sum(axis=1)).max() <= 5 * np.finfo(float).eps * np.abs(K).sum(axis=1).max()


@pytest.mark.parametrize("internal", [1e10, 1e11])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pseudoinverse_of_a_kron_reduction_formed_by_cancellation_at_large_weight_ratios(internal, seed):
    """At internal susceptances 1e10 times the ties the rounding of the row sums of L, carried into the
    reduction, exceeded 1e-6 of its lambda_2, and `laplacian_pseudoinverse` took the reduced Laplacian for
    a loopy one and inverted that rounding (audit of 2026-11-20, A1 and A2). Reference: the pseudoinverse
    of the exact Schur complement of the float weights, in 50 digits (mpmath), as Pi G Pi with G the
    inverse grounded at the first kept node."""
    mp = pytest.importorskip("mpmath")
    from gridnet.graph import laplacian_pseudoinverse

    mp.mp.dps = 50
    edges, b = _clusters(internal, seed)
    n, keep = 20, [0, 4, 8, 12, 16]
    L = mp.zeros(n, n)
    for (i, j), w in zip(edges, b):
        w = mp.mpf(float(w)); L[i, j] -= w; L[j, i] -= w; L[i, i] += w; L[j, j] += w
    beta = [i for i in range(n) if i not in keep]
    def sub(r, c):
        return mp.matrix([[L[i, j] for j in c] for i in r])
    K = sub(keep, keep) - sub(keep, beta) * sub(beta, beta) ** -1 * sub(beta, keep)
    G = mp.zeros(5, 5)
    Gi = mp.matrix([[K[i, j] for j in range(1, 5)] for i in range(1, 5)]) ** -1
    for i in range(1, 5):
        for j in range(1, 5):
            G[i, j] = Gi[i - 1, j - 1]
    Pi = mp.eye(5) - mp.ones(5, 5) / 5
    ref = np.array((Pi * G * Pi).tolist(), dtype=float)
    P = laplacian_pseudoinverse(kron_reduce(laplacian(n, edges, b), keep))
    assert np.abs(P - ref).max() <= 1e-4 * np.abs(ref).max()
