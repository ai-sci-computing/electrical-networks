"""Tests for gridnet.cycles: spanning forest, fundamental cycle basis, winding numbers.

References
  cycle space, dimension nu = m - n + c, strictly fundamental basis from a spanning
  tree, integrality (chord rows form an identity):
      Kavitha et al., Computer Science Review 3(4) (2009), Sec. 2 and Thm. 2.3, Sec. 3.
  winding number of a cycle: (1/2 pi) sum of angle differences taken in (-pi, pi]:
      Manik, Timme & Witthaut, Chaos 27 (2017), Def. 6, eq. (27);
      Delabays, Coletta & Jacquod, J. Math. Phys. 57 (2016), Sec. II.
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.cycles import fundamental_cycle_basis, spanning_forest, winding_numbers
from gridnet.graph import incidence_matrix

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n):
    return path_edges(n) + [(n - 1, 0)]


def complete_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# --- spanning forest ------------------------------------------------------------

def test_spanning_forest_of_tree_is_everything():
    tree, chords = spanning_forest(5, path_edges(5))
    np.testing.assert_array_equal(np.sort(tree), np.arange(4))
    assert chords.size == 0


def test_spanning_forest_sizes():
    for n, edges, c in [(6, cycle_edges(6), 1), (5, complete_edges(5), 1), (6, path_edges(3) + [(3, 4), (4, 5), (5, 3)], 2)]:
        tree, chords = spanning_forest(n, edges)
        assert tree.size == n - c
        assert chords.size == len(edges) - (n - c)
        assert set(tree) | set(chords) == set(range(len(edges)))
        assert not set(tree) & set(chords)


def test_spanning_forest_tree_edges_are_acyclic():
    n, edges = 5, complete_edges(5)
    tree, _ = spanning_forest(n, edges)
    A = incidence_matrix(n, [edges[i] for i in tree])
    assert np.linalg.matrix_rank(A) == A.shape[1]      # columns independent <=> no cycle


# --- fundamental cycle basis -----------------------------------------------------

def test_tree_has_empty_basis():
    C = fundamental_cycle_basis(5, path_edges(5))
    assert C.shape == (4, 0)


def test_ring_basis_is_one_all_plus_one_column():
    n = 6
    edges = cycle_edges(n)                     # oriented consistently i -> i+1, (n-1) -> 0
    C = fundamental_cycle_basis(n, edges)
    assert C.shape == (n, 1)
    np.testing.assert_array_equal(np.abs(C[:, 0]), 1)
    # the chord gets +1 and the tree path is traversed along its orientation: all +1
    np.testing.assert_array_equal(C[:, 0], np.ones(n))


@pytest.mark.parametrize("n,edges", [(6, cycle_edges(6)), (5, complete_edges(5)), (7, complete_edges(7))])
def test_cycles_lie_in_the_kernel_of_A(n, edges):
    A = incidence_matrix(n, edges)
    C = fundamental_cycle_basis(n, edges)
    np.testing.assert_array_equal(A @ C, 0)


@pytest.mark.parametrize("n,edges,nu", [
    (5, complete_edges(5), 6), (7, complete_edges(7), 15), (6, cycle_edges(6), 1),
    (6, path_edges(3) + [(3, 4), (4, 5), (5, 3)], 1),    # two components: nu = m - n + c
])
def test_dimension_is_m_minus_n_plus_c(n, edges, nu):
    C = fundamental_cycle_basis(n, edges)
    assert C.shape == (len(edges), nu)
    assert np.linalg.matrix_rank(C) == nu


def test_entries_are_signs_and_chord_rows_form_identity():
    n, edges = 6, complete_edges(6)
    C = fundamental_cycle_basis(n, edges)
    assert set(np.unique(C)) <= {-1.0, 0.0, 1.0}
    _, chords = spanning_forest(n, edges)
    np.testing.assert_array_equal(C[chords, :], np.eye(len(chords)))   # integral basis


@pytest.mark.data("matpower")
def test_ieee14_has_seven_fundamental_cycles():
    from gridnet.network import read_matpower_case
    net = read_matpower_case(CASES / "case14.m")
    C = fundamental_cycle_basis(net.n_bus, net.edges)
    assert C.shape == (20, 20 - 14 + 1)
    np.testing.assert_array_equal(incidence_matrix(net.n_bus, net.edges) @ C, 0)


def test_reversing_an_edge_flips_its_row_sign():
    n, edges = 5, complete_edges(5)
    C1 = fundamental_cycle_basis(n, edges)
    edges2 = list(edges)
    edges2[3] = (edges[3][1], edges[3][0])
    C2 = fundamental_cycle_basis(n, edges2)
    # same tree/chord choice (chosen by edge index), so only row 3 changes sign
    D = C1.copy(); D[3] *= -1
    np.testing.assert_array_equal(C2, D)


# --- winding numbers --------------------------------------------------------------

def test_ring_winding_numbers():
    n = 6
    edges = cycle_edges(n)
    theta = 2 * np.pi * np.arange(n) / n           # angles advance by 60 deg along the ring
    # differences theta_i - theta_{i+1} = -pi/3 on every edge (chord wraps to -pi/3): sum -2 pi
    np.testing.assert_array_equal(winding_numbers(theta, n, edges), [-1])
    np.testing.assert_array_equal(winding_numbers(-theta, n, edges), [1])
    np.testing.assert_array_equal(winding_numbers(2 * theta, n, edges), [-2])
    np.testing.assert_array_equal(winding_numbers(np.zeros(n), n, edges), [0])


def test_winding_numbers_are_gauge_invariant():
    n = 8
    edges = cycle_edges(n) + [(0, 4)]                 # ring plus a chord: nu = 2
    rng = np.random.default_rng(0)
    theta = rng.uniform(-3, 3, n)
    q = winding_numbers(theta, n, edges)
    np.testing.assert_array_equal(winding_numbers(theta + 1.7, n, edges), q)     # global shift
    shifted = theta.copy(); shifted[3] += 2 * np.pi                                # angles live on the circle
    np.testing.assert_array_equal(winding_numbers(shifted, n, edges), q)


def test_small_angles_have_zero_winding():
    n, edges = 6, complete_edges(6)
    rng = np.random.default_rng(1)
    theta = rng.uniform(-0.2, 0.2, n)                 # all differences far inside (-pi, pi]
    np.testing.assert_array_equal(winding_numbers(theta, n, edges), 0)


def test_winding_numbers_accept_precomputed_basis_and_are_integers():
    n, edges = 5, complete_edges(5)
    C = fundamental_cycle_basis(n, edges)
    q = winding_numbers(np.linspace(0, 4, n), n, edges, basis=C)
    assert q.dtype.kind == "i" and q.shape == (C.shape[1],)


@pytest.mark.parametrize("edge", [(0, 3), (0, -1), (1, 1)])
def test_cycle_functions_refuse_an_edge_outside_the_nodes_or_a_self_loop(edge):
    """As `gridnet.graph.incidence_matrix` and `laplacian` do: an index n raised IndexError, an index -1 was
    read as the last node without a word, and a self-loop became a cycle of its own (audit of 2026-11-20, A5)."""
    from gridnet.cycles import fundamental_cycle_basis, spanning_forest

    edges = [(0, 1), (1, 2), edge]
    with pytest.raises(ValueError):
        spanning_forest(3, edges)
    with pytest.raises(ValueError):
        fundamental_cycle_basis(3, edges)
