"""Kron reduction: eliminating interior nodes of a network by a Schur complement.

Reference: Dorfler & Bullo, "Kron Reduction of Graphs with Applications to Electrical
Networks", IEEE Trans. Circuits Syst. I 60(1) (2013). Equation and theorem numbers
below refer to that paper.

Mathematical setting
--------------------
Let Q be a symmetric matrix indexed by the nodes, of the form Q = L + diag(shunts) with
L a weighted Laplacian and shunts >= 0 ("loopy Laplacian" in the paper's terminology;
"loop-less" when the shunts vanish).
The sign of the shunts depends on the convention: with Q = -Im Y_bus (the
susceptance Laplacian, couplings 1/x > 0) an inductive shunt is positive, but the
charging of lines and capacitor banks raises Im Y and enters as a negative shunt, so a
grid with charging is in general not a loopy Laplacian in the paper's sense, and its
Lemma 2.1 and Thm. 3.9 are not guaranteed for it. `kron_reduce` does not check the
hypothesis (it also reduces the complex Y_bus); `reduced_couplings` requires the
loop-less case and checks it. Partition the nodes into boundary nodes alpha
(kept) and interior nodes beta (eliminated). The Kron-reduced matrix is the Schur
complement (eq. 2.1)

    Q_red = Q[alpha, alpha] - Q[alpha, beta] Q[beta, beta]^{-1} Q[beta, alpha].

It is the Dirichlet-to-Neumann map of the interior: if V = Q^{-1} I with zero injections
at interior nodes, then I[alpha] = Q_red V[alpha]. Consequences proven in the paper:

* Q_red is again a loopy Laplacian, so it defines a graph on the boundary nodes with
  nonnegative weights and shunts (Lemma 2.1). Weights appear between any two boundary
  nodes joined by a path through the interior (Thm. 3.4), so reduction densifies.
* Eliminating nodes one at a time gives the same result as eliminating them at once
  (Lemma 3.3), so reduction is a well-defined operation on graphs.
* The eigenvalues interlace: lambda_r(Q) <= lambda_r(Q_red) <= lambda_{r+n-|alpha|}(Q)
  (Thm. 3.5).
* Effective resistances between boundary nodes are unchanged (Thm. 3.9). This is the
  sense in which Kron reduction is the physically exact reduction of a resistive
  network, and the reason it serves as the reference reduction against which
  aggregation methods are judged.

Electrical reading
------------------
Q may also be the complex bus admittance matrix Y_bus: eliminating all buses without
current injection ("passive" buses) gives the reduced admittance matrix seen from the
generator buses, the classical use in transient stability (the "reduced network" of the
swing equation model). The formula is identical; only the Laplacian-specific
statements need real symmetric input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gridnet.graph import effective_resistance


def _boundary(Q: NDArray, keep: Sequence[int]) -> tuple[NDArray, NDArray]:
    """The kept (boundary) indices alpha, validated, and the eliminated (interior) indices beta, sorted."""
    n = Q.shape[0]
    alpha = np.asarray(keep, dtype=int)
    if alpha.ndim != 1 or len(set(alpha.tolist())) != alpha.size or np.any(alpha < 0) or np.any(alpha >= n):
        raise ValueError("keep must be distinct node indices within range")
    return alpha, np.setdiff1d(np.arange(n), alpha)


def _check_interior(Q: NDArray, alpha: NDArray, beta: NDArray) -> None:
    """Raise ValueError when the interior block Q(beta, beta) is singular by the pattern of Q: a
    connected component of the interior with no edge to a boundary node and no shunt (see
    `kron_reduce`)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    Qbb = Q[np.ix_(beta, beta)]
    pattern = Qbb != 0
    np.fill_diagonal(pattern, False)
    _, comp = connected_components(csr_matrix(pattern), directed=False)
    linked = np.any(Q[np.ix_(beta, alpha)] != 0, axis=1)
    shunted = np.abs(Q.sum(axis=1)[beta]) > 1e-12 * np.abs(Q).max()
    for c in np.unique(comp):
        part = comp == c
        if not linked[part].any() and not shunted[part].any():
            raise ValueError("interior block is singular: an interior component is not connected to the boundary")


def kron_reduce(Q: NDArray, keep: Sequence[int]) -> NDArray:
    """Schur complement Q / Q(beta, beta) onto the boundary nodes `keep` (eq. 2.1).

    Works for real or complex square matrices. Raises ValueError if the interior block
    is singular, which for a loop-less Laplacian happens exactly when some connected
    component of the interior has no edge to a boundary node (its potentials are then
    undetermined by the boundary); this is tested on the pattern of Q, not by a numerical
    rank, which refused valid reductions with weights 1e18 apart. A component without such an
    edge but with a shunt (a loopy Laplacian, Dorfler & Bullo 2013, Sec. II) is regular.

    The diagonal is taken from the row sums, which the reduction maps exactly: with s = Q 1 the
    shunts, the reduced matrix has the row sums s_alpha - Q_alpha,beta Q_beta,beta^{-1} s_beta, and
    its diagonal is that less the off-diagonal entries of its row. Computed directly, the diagonal
    is a difference of the large weights at a kept node behind a strong branch and lost them to
    cancellation (0 for 1e-9 on a path of 1e-9 and 1e9). A row sum within its own rounding,
    |s_i| <= n eps sum_j |Q_ij|, is taken as zero: in floating point the row sums of a Laplacian
    are not zero but of the order of eps times the largest weight at the node, and carried
    through the reduction as shunts they exceeded the reduced weights (a path of weights 1e-10
    to 2e8 reduced onto its ends: diagonal 3.6e-9 for the exact 1.3e-10; audit of repairs,
    2026-11-20, A2). It is the test `gridnet.graph.laplacian_pseudoinverse` applies to a shunt.
    """
    Q = np.asarray(Q)
    alpha, beta = _boundary(Q, keep)
    if beta.size == 0:
        return Q[np.ix_(alpha, alpha)].copy()

    Qbb = Q[np.ix_(beta, beta)]
    s = Q.sum(axis=1)                                            # the shunts, zero for a Laplacian
    s = np.where(np.abs(s) <= Q.shape[0] * np.finfo(float).eps * np.abs(Q).sum(axis=1), 0, s)
    _check_interior(Q, alpha, beta)
    X = np.linalg.solve(Qbb, Q[np.ix_(beta, alpha)])
    K = Q[np.ix_(alpha, alpha)] - Q[np.ix_(alpha, beta)] @ X
    rows = s[alpha] - Q[np.ix_(alpha, beta)] @ np.linalg.solve(Qbb, s[beta])
    np.fill_diagonal(K, 0)
    K[np.diag_indices_from(K)] = rows - K.sum(axis=1)
    return K


def accompanying_matrix(Q: NDArray, keep: Sequence[int]) -> NDArray:
    """Where the eliminated injections go: the accompanying matrix of the Kron reduction, for all nodes.

    Q_ac = -Q[alpha, beta] Q[beta, beta]^{-1} "maps internal currents to boundary currents in the reduced
    network", which then carries the injections I_alpha + Q_ac I_beta and reproduces the boundary potentials
    (Dorfler & Bullo 2013, Sec. I). It is nonnegative, and column stochastic if Q is a Laplacian without
    shunts (their Lemma 2.1(3)). These are the distribution factors of Ward 1949, who measured them with a
    unit current at one suppressed bus and the retained terminals short-circuited, and minus the matrix D = Y_AB Y_BB^{-1} of
    Tinney & Powell 1977, eq. (4), whose product with the eliminated injections they subtract. Read as a random walk with transition probabilities w_xy / w_x, the entry
    (a, x) is the probability that the walk started at x reaches the kept node a before any other kept node
    (Doyle & Snell 2000, Sec. 1.3.2, voltage as hitting probability). In the linear flow model the injections are
    active powers and Q the susceptance Laplacian.

    Returns the |keep| x n matrix with the identity in the columns of the kept nodes, rows in the order of
    `keep`, so that the reduced injections are D @ p. Raises ValueError as `kron_reduce` does when an
    interior component has no edge to a kept node and no shunt: its injections would reach no kept node."""
    Q = np.asarray(Q)
    n = Q.shape[0]
    alpha, beta = _boundary(Q, keep)
    D = np.zeros((alpha.size, n), dtype=Q.dtype)
    D[np.arange(alpha.size), alpha] = 1.0
    if beta.size:
        _check_interior(Q, alpha, beta)
        D[:, beta] = -np.linalg.solve(Q[np.ix_(beta, beta)].T, Q[np.ix_(alpha, beta)].T).T
    return D


def laplacian_to_graph(
    Q: NDArray[np.float64], tol: float = 1e-12
) -> tuple[list[tuple[int, int]], NDArray[np.float64], NDArray[np.float64]]:
    """Read a (loopy) Laplacian as a graph: edges (i < j), weights -Q_ij, shunts = row sums.

    The inverse of `gridnet.graph.laplacian` plus a diagonal, used to interpret a
    Kron-reduced matrix as the network it induces (Lemma 2.1). Off-diagonal entries
    with |Q_ij| <= tol are treated as absent edges.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = -Q[iu, ju]
    mask = np.abs(w) > tol
    edges = list(zip(iu[mask].tolist(), ju[mask].tolist()))
    shunts = Q.sum(axis=1)
    shunts[np.abs(shunts) <= tol] = 0.0
    return edges, w[mask], shunts


# --- comparing an exact reduction with an aggregation -----------------------------------


def pick_representatives(
    xy: NDArray, labels: Sequence, centroids: dict, mode: str = "nearest", degrees: NDArray | None = None,
) -> dict:
    """One node per cluster to keep in a Kron reduction that is to be compared with an
    aggregation of the same clusters.

    An aggregation (PyPSA-Eur's `cluster_network`, Hoersch & Brown 2017) replaces each
    cluster by one new node at the load-weighted centroid of its buses; that node is not a
    bus of the original network. A Kron reduction (Dorfler & Bullo 2013, eq. 2.1) keeps
    actual nodes. To compare the two, each cluster needs a representative among its own
    nodes. `mode="nearest"` takes the node closest to the cluster's centroid in the plane
    of `xy`; `mode="degree"` takes the node of largest weighted degree, the most strongly
    coupled bus of the cluster. Returns {label: node index}.
    """
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(labels)
    if mode == "degree" and degrees is None:
        raise ValueError("mode='degree' needs the weighted degrees")
    reps = {}
    for lab, (cx, cy) in centroids.items():
        members = np.flatnonzero(labels == lab)
        if members.size == 0:
            continue
        if mode == "nearest":
            d = np.hypot(xy[members, 0] - cx, xy[members, 1] - cy)
            reps[lab] = int(members[np.argmin(d)])
        elif mode == "degree":
            reps[lab] = int(members[np.argmax(np.asarray(degrees)[members])])
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return reps


def reduced_couplings(L: NDArray, keep: Sequence[int]) -> tuple[NDArray, NDArray]:
    """Couplings and effective resistances among the kept nodes after Kron reduction.

    The reduced matrix Q_red = L / L(beta, beta) is again a Laplacian (Dorfler & Bullo
    2013, Lemma 2.1), so W = -Q_red with zero diagonal is the matrix of equivalent
    couplings between kept nodes, generally dense (Thm. 3.4: two kept nodes are coupled
    whenever a path through eliminated nodes joins them). The effective resistances among
    kept nodes are those of the original network (Thm. 3.9), computed here from the
    reduced matrix. Returns (W, R), both k-by-k, symmetric, zero diagonal.
    L must be loop-less, its rows summing to zero (ValueError otherwise): a shunt would be
    read as part of the couplings and the resistances taken from a matrix that is not a Laplacian.
    """
    L = np.asarray(L, dtype=float)
    if np.any(np.abs(L.sum(axis=1)) > 1e-9 * np.abs(np.diag(L)).max()):
        raise ValueError("reduced_couplings needs a loop-less Laplacian (rows summing to zero)")
    Q = kron_reduce(L, keep)
    W = -np.asarray(Q, dtype=float)
    np.fill_diagonal(W, 0.0)
    return W, effective_resistance(Q)


def laplacian_from_couplings(W: NDArray, mask: NDArray | None = None) -> NDArray:
    """Laplacian of the graph with symmetric nonnegative couplings W, optionally keeping only
    the pairs where `mask` is True.

    With no mask this inverts `reduced_couplings`, returning the Kron-reduced Laplacian.
    With a mask it is the sparse graph that keeps the exact Kron couplings on the retained
    pairs and drops the rest, the simplest aggregation that respects the series rule of the
    reduction: the parallel paths through eliminated nodes are already folded
    into each kept weight, and only the fill-in between non-retained pairs is discarded.
    """
    W = np.asarray(W, dtype=float)
    if mask is not None:
        W = np.where(mask | mask.T, W, 0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return np.diag(W.sum(axis=1)) - W
