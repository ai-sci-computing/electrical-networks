"""Spanning forest, fundamental cycle basis, and winding numbers.

Mathematical setting
--------------------
With the oriented incidence matrix A (n x m) of `gridnet.graph`, the cycle space of the
graph is ker A acting on edge vectors: the vectors z in R^m with A z = 0 are the
"circulations", flows around closed paths that satisfy conservation at every node. Its
dimension is nu = m - n + c with c the number of connected components (Kavitha et al.
2009, Thm. 2.3; the graph has no 2-cells, so this is the first Betti number of the
graph as a 1-complex). The image of A^T (edge differences of node potentials) is the
orthogonal complement, the cut space.

A *strictly fundamental* cycle basis (Kavitha et al. 2009, Sec. 2) comes from a spanning
forest T: every non-tree edge ("chord") e = (u, v) closes exactly one cycle with the
tree path from v back to u. Writing each such cycle as a signed edge vector with entries
in {-1, 0, +1} (+1 where the cycle traverses an edge along its orientation) gives the
cycle matrix C (m x nu) with A C = 0. Restricted to the chord rows, C is the identity, so
the basis is integral (Kavitha et al. 2009, Sec. 3): every integer circulation is an
integer combination of the columns. That is what makes winding numbers integers.

Winding numbers (Manik, Timme & Witthaut 2017, Def. 6, eq. 27; Delabays, Coletta &
Jacquod 2016, Sec. II). Node angles theta live on the circle, so only the differences
eta_l = theta_i - theta_j taken in [-pi, pi] are physical (np.angle returns either end
at +-pi). Around a cycle z the
differences must sum to a multiple of 2 pi; that multiple,
    q_z = (1 / 2 pi) sum_l z_l eta_l  in  Z,
is the winding number of the angle configuration on that cycle. The vector of winding
numbers over a cycle basis labels the topological sector of a power flow solution; in a
Kuramoto-type or lossless power flow model, solutions in different sectors differ by
loop flows (Delabays et al. 2016, Sec. I; Manik et al. 2017, Thm. 7).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridnet.graph import _check_edges, incidence_matrix

Edge = tuple[int, int]


def spanning_forest(n_nodes: int, edges: Sequence[Edge]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Edge indices of a breadth-first spanning forest and of its complement (chords).

    Breadth-first search from the lowest-numbered unvisited node of each component;
    the first edge that reaches a node becomes its tree edge. Parallel edges: the first
    by index is the tree edge, later copies are chords. The choice is deterministic in
    the edge order so that tests can reason about it.

    Returns (tree, chords): index arrays into `edges`, disjoint, together all edges.
    """
    in_tree, _, _ = _bfs_forest(n_nodes, edges)
    idx = np.arange(len(edges))
    return idx[in_tree], idx[~in_tree]


def _bfs_forest(
    n_nodes: int, edges: Sequence[Edge]
) -> tuple[NDArray[np.bool_], list[tuple[int, int, int] | None], NDArray[np.int64]]:
    """The breadth-first search behind `spanning_forest`, with its parent pointers.

    Returns (in_tree, parent, depth): in_tree[l] marks the tree edges; parent[v] = (u, l, s)
    with tree edge l joining u and v and s = +1 if l is oriented u -> v, -1 if v -> u
    (None at a root); depth[v] is the number of tree edges from v to its root. Raises ValueError
    for an index outside 0 .. n_nodes-1 or a self-loop, as `gridnet.graph.incidence_matrix` does.
    """
    _check_edges(n_nodes, edges)
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n_nodes)]
    for l, (i, j) in enumerate(edges):
        adj[i].append((j, l, +1))
        adj[j].append((i, l, -1))
    in_tree = np.zeros(len(edges), dtype=bool)
    parent: list[tuple[int, int, int] | None] = [None] * n_nodes
    depth = np.full(n_nodes, -1)
    for root in range(n_nodes):
        if depth[root] >= 0:
            continue
        depth[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, l, s in adj[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = (u, l, s)
                    in_tree[l] = True
                    queue.append(v)
    return in_tree, parent, depth


def fundamental_cycle_basis(n_nodes: int, edges: Sequence[Edge]) -> NDArray[np.float64]:
    """Signed cycle matrix C (m x nu) of the strictly fundamental basis of a BFS forest.

    Column k belongs to chord e_k = (u, v) (in the order returned by `spanning_forest`):
    entry +1 at e_k, and along the unique tree path from v to u entry +1 on edges
    traversed with their orientation, -1 against it, 0 elsewhere. Then A C = 0 and
    C[chords, :] is the identity (Kavitha et al. 2009, Secs. 2-3).

    Complexity O(nu * n) from walking tree paths through parent pointers; fine for the
    networks in this project (nu = 7 for IEEE 14, a few thousand for PyPSA-Eur).
    """
    m = len(edges)
    # one breadth-first search gives the forest and its parent pointers (see _bfs_forest)
    in_tree, parent, depth = _bfs_forest(n_nodes, edges)
    chords = np.flatnonzero(~in_tree)
    C = np.zeros((m, len(chords)))
    for k, l in enumerate(chords):
        u, v = edges[l]
        C[l, k] = 1.0
        # walk from v and from u up to their lowest common ancestor; going up from a node
        # w to parent p along tree edge oriented p -> w (s = +1) means traversing it
        # against its orientation when we move w -> p. The cycle runs u -> v (chord),
        # then v -> ... -> lca -> ... -> u. So from v upward: entry -s; from u upward: +s.
        a, b = v, u
        while a != b:
            if depth[a] >= depth[b]:
                p, le, s = parent[a]
                C[le, k] = -s
                a = p
            else:
                p, le, s = parent[b]
                C[le, k] = s
                b = p
    return C


def winding_numbers(
    theta: ArrayLike,
    n_nodes: int,
    edges: Sequence[Edge],
    basis: NDArray | None = None,
) -> NDArray[np.int64]:
    """Winding number of the angle configuration `theta` on each basis cycle.

    eta = A^T theta wrapped to [-pi, pi] by np.angle (the physical angle differences, Delabays et
    al. 2016, Sec. II), then q = C^T eta / (2 pi), which is an integer vector because
    every cycle sum of wrapped differences is a multiple of 2 pi (Manik et al. 2017,
    Def. 6, eq. 27). Invariant under theta -> theta + const and under adding 2 pi to any
    single angle.

    Parameters
    ----------
    basis : cycle matrix from `fundamental_cycle_basis`; computed if omitted.
    """
    theta = np.asarray(theta, dtype=np.float64)
    A = incidence_matrix(n_nodes, edges)
    C = fundamental_cycle_basis(n_nodes, edges) if basis is None else basis
    eta = A.T @ theta
    eta = np.angle(np.exp(1j * eta))             # wrap to [-pi, pi]; an exact half turn may land on either end
    q = C.T @ eta / (2 * np.pi)
    q_int = np.rint(q)
    if not np.allclose(q, q_int, atol=1e-9):
        raise ValueError("cycle sums are not multiples of 2 pi; the basis is not a cycle basis")
    return q_int.astype(np.int64)
