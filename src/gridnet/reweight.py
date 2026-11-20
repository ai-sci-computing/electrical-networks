"""The best weights on a prescribed edge set: reweighting a sparse graph against a dense Laplacian.

Setting: L is the exact Kron reduction of a network onto a few buses, a complete graph;
E is a sparse edge set on the same buses, for the European grid the lines of the clustered network.
Three ways to choose weights w >= 0 on E, L~(w) = sum_e w_e b_e b_e^T:

* `reweight_spectral`: minimize the spectral error eps with (1 - eps) L <= L~(w) <= (1 + eps) L, the
  measure of Spielman & Srivastava 2011, eq. (2). Both bounds are linear matrix inequalities in
  (w, eps), so this is a semidefinite program with a global optimum, and its value is a lower bound
  on the error of every graph supported on E.
* `reweight_frobenius`: minimize || L^{+/2} L~(w) L^{+/2} - I ||_F, the same comparison in the mean
  over all directions instead of the worst one; a nonnegative least-squares problem.
* `reweight_ptdf`: the practice of power system equivalents, fit the susceptances so that the
  reduced network's power transfer distribution factors match those of the original network
  summed over the zones (Fortenbacher, Demiray & Schaffner 2018, arXiv:1711.01079, who restate
  the methods of Oh 2010 and Shi & Tylavsky 2015 in one notation; eqs. (5), (15), (16)).

The PTDF H, the matrix of power transfer distribution factors, maps bus injections to branch flows in the linear flow model. Here it is
built with the pseudoinverse, H = diag(b) C^T L^+, which acts on balanced injections and needs no
slack bus; Fortenbacher et al. remove a slack column instead, the same map on balanced injections.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from gridnet.graph import (
    IllConditionedLaplacian,
    incidence_matrix,
    laplacian,
    laplacian_pseudoinverse,
    partition_labels,
    range_whitening,
)


def _whitened_edge_vectors(L: NDArray, edges: Sequence[tuple[int, int]]) -> NDArray[np.float64]:
    """Columns u_e = L^{+/2} b_e restricted to the range of L, so that L^{+/2} L~(w) L^{+/2} = sum_e w_e u_e u_e^T
    and L itself becomes the identity. Raises ValueError unless L is positive semidefinite (`range_whitening`)."""
    half = range_whitening(L)
    B = incidence_matrix(L.shape[0], list(edges))                # n x m, column e = b_e
    return half.T @ B


def _psd_part(Z: NDArray) -> NDArray[np.float64]:
    """The positive semidefinite part of the symmetrized Z (negative eigenvalues set to zero): the nearest
    positive semidefinite matrix in the Frobenius norm, used to make the solver's dual multipliers feasible."""
    lam, V = np.linalg.eigh((Z + Z.T) / 2)
    return (V * np.clip(lam, 0.0, None)) @ V.T


def reweight_spectral(L: NDArray, edges: Sequence[tuple[int, int]], solver: str | None = None) -> tuple[NDArray[np.float64], float, float]:
    """Weights on `edges` with the smallest spectral error against L, that error, and a proven lower bound.

    Primal: minimize eps subject to (1 - eps) I <= M(w) <= (1 + eps) I, w >= 0, with M(w) = sum_e w_e u_e u_e^T
    and u_e = L^{+/2} b_e (Spielman & Srivastava 2011, eq. (2), written on the range of L). Dual, from the
    Lagrangian eps - <Z1, M - (1 - eps) I> - <Z2, (1 + eps) I - M>: maximize tr Z1 - tr Z2 subject to
    Z1, Z2 >= 0, tr Z1 + tr Z2 = 1 and u_e^T (Z1 - Z2) u_e <= 0 for every edge. Any such pair proves that no
    nonnegative weights on these edges reach an error below tr Z1 - tr Z2 (weak duality). The solver's
    multipliers are checked here in numpy, made feasible if they miss by rounding, and the resulting bound
    is returned, so the optimality of eps does not rest on the solver's status.

    Units: the weights are measured in units of the geometric mean of the couplings L carries on these
    edges. This is one common factor, w = c v, and changes neither the problem nor its optimum, but
    without it the coefficients u_e u_e^T are small against the identity while the unknowns are large, and
    the interior-point solver fails to factor its first system; one common factor suffices, a per-edge
    scaling is not needed.

    Returns (w, eps, bound) with bound <= optimal eps <= eps; eps is recomputed from the eigenvalues of M(w)."""
    import cvxpy as cp

    U = _whitened_edge_vectors(L, edges)
    k, m = U.shape
    e = np.asarray(edges, dtype=int)
    on_edges = -np.asarray(L, dtype=float)[e[:, 0], e[:, 1]]
    c = float(np.exp(np.mean(np.log(on_edges[on_edges > 0])))) if np.any(on_edges > 0) else 1.0
    Us = U * np.sqrt(c)
    v = cp.Variable(m, nonneg=True); eps = cp.Variable()
    M = Us @ cp.diag(v) @ Us.T
    M = (M + M.T) / 2
    lower, upper = M >> (1 - eps) * np.eye(k), M << (1 + eps) * np.eye(k)
    prob = cp.Problem(cp.Minimize(eps), [lower, upper])
    prob.solve(solver=solver or cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"semidefinite program not solved: {prob.status}")
    w = c * np.asarray(v.value, dtype=float)
    mu = np.linalg.eigvalsh(U @ np.diag(w) @ U.T)
    achieved = float(max(mu.max() - 1.0, 1.0 - mu.min()))

    Z1, Z2 = _psd_part(lower.dual_value), _psd_part(upper.dual_value)
    # feasibility of the edge conditions by shifting Z2 by a multiple of the identity, then normalizing the trace
    slack = np.einsum("ie,ij,je->e", U, Z1 - Z2, U)
    shift = max(0.0, float(np.max(slack / np.sum(U * U, axis=0))))
    Z2 = Z2 + shift * np.eye(k)
    total = np.trace(Z1) + np.trace(Z2)
    bound = float((np.trace(Z1) - np.trace(Z2)) / total)
    return w, achieved, bound


def reweight_frobenius(L: NDArray, edges: Sequence[tuple[int, int]]) -> NDArray[np.float64]:
    """Weights on `edges` minimizing || sum_e w_e u_e u_e^T - I ||_F over w >= 0 (nonnegative least squares)."""
    from scipy.optimize import nnls

    U = _whitened_edge_vectors(L, edges)
    k = U.shape[0]
    A = np.stack([np.outer(U[:, e], U[:, e]).ravel() for e in range(U.shape[1])], axis=1)
    return nnls(A, np.eye(k).ravel())[0]


def ptdf_from_susceptances(n: int, edges: Sequence[tuple[int, int]], b: NDArray) -> NDArray[np.float64]:
    """H = diag(b) C^T L^+, branch flows per bus injection (Fortenbacher et al. 2018, eqs. (2), (15), with
    the pseudoinverse in place of a removed slack column)."""
    C = incidence_matrix(n, list(edges))
    return (np.asarray(b, dtype=float)[:, None] * C.T) @ laplacian_pseudoinverse(laplacian(n, list(edges), b))


def _zone_weights(weights: NDArray | None, zones: NDArray, n: int) -> NDArray[np.float64]:
    """The weights of `reduced_ptdf` and `zonal_operator`, checked: one finite nonnegative number per bus and a
    positive sum in every zone (a zone of zero weights would be divided by zero). None gives equal weights."""
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be one finite number per bus, {n} in all")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    empty = np.flatnonzero(np.bincount(zones, weights=w, minlength=int(zones.max()) + 1) <= 0)
    if len(empty):
        raise ValueError(f"the weights of zone {', '.join(map(str, empty))} are all zero")
    return w


def reduced_ptdf(n: int, edges: Sequence[tuple[int, int]], b: NDArray, zones: NDArray,
                 weights: NDArray | None = None) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """The zonal PTDF of a network and the zonal interconnections it refers to.

    Fortenbacher et al. 2018, eq. (5), the injection-independent form after Oh 2010:
    H_red = T_f H_f T_bz^T (T_bz T_bz^T)^{-1}, where T_f sums the flows of the branches between two
    zones (signed from the lower to the higher zone number) and T_bz T_bz^T = diag(zone sizes), so an
    injection into a zone is spread evenly over its buses. `zones` holds a zone number 0..z-1 per bus,
    every number used (an empty zone raises ValueError).
    The flows T_f H_f are computed by solving with the Laplacian, not by forming H_f.
    `weights` (one nonnegative number per bus, normalized within each zone) distribute a zone's injection
    over its buses instead, the participation factors of Cheng & Overbye 2005 (p. 1869), as in
    `zonal_operator`; all weight on one bus per zone gives the zonal PTDF of transfers between those buses,
    the one that belongs to the Kron reduction onto them. Equal weights give eq. (5).
    Returns (H_red, interconnections as sorted pairs of zone numbers)."""
    zones = partition_labels(zones, n)                        # zone numbers 0..z-1, none empty
    nz = int(zones.max()) + 1
    b = np.asarray(b, dtype=float)
    pairs = sorted({(min(zones[i], zones[j]), max(zones[i], zones[j])) for i, j in edges if zones[i] != zones[j]})
    index = {p: k for k, p in enumerate(pairs)}
    # rows of T_f diag(b) C^T: for every interconnection the signed, weighted incidence of its branches
    G = np.zeros((len(pairs), n))
    for (i, j), be in zip(edges, b):
        zi, zj = zones[i], zones[j]
        if zi != zj:
            s = 1.0 if zi < zj else -1.0
            G[index[(min(zi, zj), max(zi, zj))], i] += s * be
            G[index[(min(zi, zj), max(zi, zj))], j] -= s * be
    flows = G @ laplacian_pseudoinverse(laplacian(n, list(edges), b))       # T_f H_f, interconnections x buses
    w = _zone_weights(weights, zones, n)
    T = np.zeros((nz, n)); T[zones, np.arange(n)] = w
    H = flows @ T.T / T.sum(1)[None, :]
    # gauge: an unbalanced zone injection would be withdrawn evenly over all buses here but evenly over all
    # zones in the reduced network's pseudoinverse PTDF; projecting onto balanced zone injections removes
    # the difference, and the rows then sum to zero like those of `ptdf_from_susceptances`
    H = H - H.mean(axis=1, keepdims=True)
    return H, [(int(a), int(c)) for a, c in pairs]


def fill_zones(zones: NDArray, edges: Sequence[tuple[int, int]], b: NDArray) -> NDArray[np.int64]:
    """Zone numbers with every unlabeled bus (zone -1) put in the zone of the nearest labeled bus along
    the network, the path length measured in reactance 1/b (Dijkstra). For the buses a cluster map
    lacks, such as those PyPSA-Eur's `simplify_network` drops. An unlabeled bus without a path to a
    labeled one raises ValueError."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    zones = np.asarray(zones, dtype=np.int64).copy()
    lost = np.flatnonzero(zones < 0)
    if lost.size == 0:
        return zones
    i, j = (np.array(c, dtype=np.int64) for c in zip(*edges))
    n = len(zones)
    # parallel branches combine by their susceptances; coo_matrix adds duplicates, so the susceptances
    # are summed first and inverted after (summing the reactances would put them in series)
    B = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    B.sum_duplicates()
    X = B.copy(); X.data = 1.0 / X.data
    dist = dijkstra(X, directed=False, indices=lost)
    dist[:, lost] = np.inf                                    # only labeled buses are targets
    if not np.all(np.isfinite(dist.min(axis=1))):
        raise ValueError("an unlabeled bus has no path to a labeled one")
    zones[lost] = zones[np.argmin(dist, axis=1)]
    return zones


def quotient_susceptances(zones: NDArray, edges: Sequence[tuple[int, int]], b: NDArray,
                          zone_edges: Sequence[tuple[int, int]]) -> NDArray[np.float64]:
    """The susceptance of every interconnection in `zone_edges` (sorted zone pairs) as the sum of the
    susceptances of the branches between its two zones, the quotient graph of the network under the
    zone map (Fortenbacher et al. 2018, eq. (8), their b_phys)."""
    zones = np.asarray(zones)
    where = {tuple(e): k for k, e in enumerate(zone_edges)}
    out = np.zeros(len(zone_edges))
    for (i, j), be in zip(edges, np.asarray(b, dtype=float)):
        a, c = zones[i], zones[j]
        if a != c:
            out[where[(min(a, c), max(a, c))]] += be
    return out


def zonal_ptdf_error(La: NDArray, zone_edges: Sequence[tuple[int, int]], H_ref: NDArray) -> float:
    """Relative Frobenius distance between the zonal PTDF H_ref (rows: `zone_edges`, as from
    `reduced_ptdf`) and the flows a graph with Laplacian La sends over the same zone pairs per balanced
    zone injection, w_ac (e_a - e_c)^T La^+ with w_ac = -La[a, c] (zero where the graph has no such edge;
    Fortenbacher et al. 2018, eq. (16), the objective of the fit)."""
    Lp = laplacian_pseudoinverse(np.asarray(La, dtype=float))
    H = np.array([-La[a, c] * (Lp[a] - Lp[c]) for a, c in zone_edges])
    return float(np.linalg.norm(H - H_ref) / np.linalg.norm(H_ref))


def reweight_ptdf(H_red: NDArray, n_zones: int, zone_edges: Sequence[tuple[int, int]], reference: NDArray,
                  basis: NDArray | None = None) -> NDArray[np.float64]:
    """Susceptances of the zonal network whose PTDF is closest to H_red in the Frobenius norm.

    Fortenbacher et al. 2018, eq. (16): minimize || H_red - diag(b) C (C^T diag(b) C)^{-1} ||_F. What a PTDF
    determines is limited (Cheng & Overbye 2005, Secs. IV and V): the network separates at its cut nodes
    into independent subsystems, its biconnected blocks; within a block the PTDF fixes the susceptances up
    to one common factor, and the susceptance of a radial branch (a bridge) affects no PTDF at all. So
    every block with more than one edge is pinned at its strongest `reference` edge, at the reference
    value (Fortenbacher et al., Sec. II-B2, pin the strongest physical interconnection; `reference` is
    their b_phys, the summed physical susceptances), and bridges keep their reference value. The free
    susceptances start at the reference and stay positive by optimizing their logarithms (nonlinear least
    squares, scipy). The logarithms are unbounded, so a trial step can reach susceptances so far apart
    that the Laplacian has no pseudoinverse to any digit (`gridnet.graph.IllConditionedLaplacian`); such
    a point is no candidate, and its residual is set to 1e100 in every entry, which the trust-region
    method answers by rejecting the step and shrinking its region. `basis` (zones x k) restricts the comparison to the zone injections in its columns,
    H(b) @ basis against H_red; with the columns e_z - e_slack this is the slack-adjusted PTDF of
    Fortenbacher et al., used to reproduce their Table III."""
    import networkx as nx
    from scipy.optimize import least_squares

    m = len(zone_edges)
    reference = np.asarray(reference, dtype=float)
    index = {tuple(sorted(e)): k for k, e in enumerate(zone_edges)}
    if len(index) < m:
        raise ValueError("a zone pair is listed twice in zone_edges (join parallel interfaces into one)")
    fixed = np.zeros(m, dtype=bool)
    for nodes in nx.biconnected_components(nx.Graph(list(zone_edges))):
        block = [index[tuple(sorted(e))] for e in nx.Graph(list(zone_edges)).subgraph(nodes).edges()]
        if len(block) == 1:
            fixed[block[0]] = True                              # a bridge: the PTDF says nothing about it
        else:
            fixed[block[int(np.argmax(reference[block]))]] = True
    free = np.flatnonzero(~fixed)

    def build(u):
        b = reference.copy(); b[free] = np.exp(u)
        return b

    def residual(u):
        try:
            H = ptdf_from_susceptances(n_zones, zone_edges, build(u))
        except IllConditionedLaplacian:
            return np.full(np.asarray(H_red).size, 1e100)       # outside the domain: the step is rejected
        return ((H if basis is None else H @ basis) - H_red).ravel()

    if free.size == 0:
        return reference.copy()
    return build(least_squares(residual, np.log(reference[free]), xtol=1e-12, ftol=1e-12, gtol=1e-12).x)


def reweight_ptdf_equations(Psi: NDArray, n_zones: int, zone_edges: Sequence[tuple[int, int]],
                            slack: int = 0, pin: int = 0) -> NDArray[np.float64]:
    """Susceptances of the zonal network by the linear least squares of Cheng & Overbye 2005, eqs. (20)-(26).

    `Psi` (branches x zones less the slack) holds the zonal ISFs to match: column n is the flow over every
    branch of `zone_edges`, positive from its first zone to its second, for a transfer from zone n (in
    increasing order, the slack left out) to the slack. With A the node-branch incidence of the non-slack
    zones (their eq. 16), the exact condition Psi = diag(b) A^T (A diag(b) A^T)^{-1} reads
    (Psi A - I) diag(b) A^T = 0, linear and homogeneous in b (eq. 15); per zone n it is
    (Psi A - I) W_n b = 0 with W_n = diag(N_n) (eqs. 19, 20), and the stacked F b = 0 (eqs. 21, 22) is solved
    by least squares with b[pin] = 1 (eqs. 23-26). What is minimized is the residual of these equations,
    not the distance of the resulting ISFs from Psi, as in `reweight_ptdf`; the minimizer is not a common
    multiple of one solution for every choice of `pin`, so the pinned branch changes the equivalent where
    Psi cannot be matched exactly (on the six-bus system of their Fig. 5 a reactance by up to 21 % over the five choices of the pin, each
    equivalent scaled to the same geometric mean of its reactances; 22 % scaled to the same sum).

    The least squares of eq. (25) assumes that the matrix of eq. (24) has full rank. A zone whose removal
    disconnects the zonal network (a cut node, every bridge has one) breaks that: each block is then fixed
    only up to a factor of its own, as Cheng & Overbye note in Sec. V, where they split the network at such
    zones. The function raises ValueError there instead of returning the minimum-norm solution, which put
    zero or negative susceptances on the bridges; `reweight_ptdf` handles blocks and bridges. The cut zone
    is found on the zonal graph (networkx), not by the numerical rank of the matrix: a Psi that is not the
    exact PTDF of the zonal graph, as one from a full network, gives that matrix full rank at a cut zone too,
    and the least squares returned negative susceptances there (audit of repairs, 2026-11-20, A3). A rank
    deficiency left after that test is refused as well."""
    import networkx as nx

    edges = [tuple(e) for e in zone_edges]
    others = [z for z in range(n_zones) if z != slack]
    A = np.zeros((len(others), len(edges)))
    for l, (f, t) in enumerate(edges):
        if f != slack:
            A[others.index(f), l] += 1.0
        if t != slack:
            A[others.index(t), l] -= 1.0
    C = np.asarray(Psi, dtype=float) @ A - np.eye(len(edges))                 # eq. (28)
    F = np.vstack([C * A[n] for n in range(len(others))])                     # C W_n, stacked as in eq. (22)
    g, H = F[:, pin], np.delete(F, pin, axis=1)                               # eq. (23)
    graph = nx.Graph(edges)
    graph.add_nodes_from(range(n_zones))
    cut = sorted(nx.articulation_points(graph)) if nx.is_connected(graph) else None
    if cut is None or cut:                                                    # eqs. (24)-(26) assume full rank
        raise ValueError(("the zonal network is not connected" if cut is None else f"the zonal network has the cut zone(s) {cut}")
                         + ", where each block is determined only up to its own factor; split it there "
                         "(Cheng & Overbye 2005, Sec. V) or use reweight_ptdf")
    rank = np.linalg.matrix_rank(H)
    if rank < H.shape[1]:
        raise ValueError(f"the equations have rank {rank} for {H.shape[1]} unknowns")
    rest = -np.linalg.lstsq(H, g, rcond=None)[0]                              # eq. (25)
    return np.insert(rest, pin, 1.0)                                          # eq. (26)


def best_scales(L: NDArray, components: Sequence[NDArray], solver: str | None = None) -> tuple[NDArray[np.float64], float, float]:
    """Nonnegative factors s_k minimizing the spectral error of sum_k s_k L_k against L, that error, and a
    lower bound certified as in `reweight_spectral` (edge condition replaced by <Z1 - Z2, A_k> <= 0 with
    A_k = L^{+/2} L_k L^{+/2}). Used to give a reduction that is determined only up to factors, a PTDF fit
    with one free factor per biconnected block and per bridge (Cheng & Overbye 2005, Sec. V), the most
    favorable choice of them. L may be any symmetric positive semidefinite reference with the constants as
    its null space, not only a graph Laplacian."""
    import cvxpy as cp

    half = range_whitening(L)                                    # raises unless L is positive semidefinite
    A = [half.T @ np.asarray(Lk, dtype=float) @ half for Lk in components]
    A = [(a + a.T) / 2 for a in A]
    zero = [k for k, a in enumerate(A) if np.trace(a) <= 1e-12 * max(1.0, max(np.trace(b) for b in A))]
    if zero:
        raise ValueError(f"components {zero} are zero on the range of L; no factor of theirs changes the error")
    if len(A) == 1:
        # exact: s mu_min >= 1 - eps and s mu_max <= 1 + eps, so eps >= (mu_max - mu_min) / (mu_max + mu_min),
        # attained at s = 2 / (mu_max + mu_min). (As a semidefinite program a graph against itself is
        # degenerate, every eigenvalue active in both inequalities, and the interior-point solver fails.)
        mu = np.linalg.eigvalsh(A[0])
        eps_1 = float((mu.max() - mu.min()) / (mu.max() + mu.min()))
        return np.array([2.0 / (mu.max() + mu.min())]), eps_1, eps_1
    unit = np.array([max(np.trace(a) / a.shape[0], 1e-300) for a in A])          # each factor in units that make A_k of order one
    k = A[0].shape[0]
    v = cp.Variable(len(A), nonneg=True); eps = cp.Variable()
    M = sum(v[i] * (A[i] / unit[i]) for i in range(len(A)))
    lower, upper = M >> (1 - eps) * np.eye(k), M << (1 + eps) * np.eye(k)
    prob = cp.Problem(cp.Minimize(eps), [lower, upper])
    prob.solve(solver=solver or cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"semidefinite program not solved: {prob.status}")
    scales = np.asarray(v.value, dtype=float) / unit
    mu = np.linalg.eigvalsh(sum(sc * a for sc, a in zip(scales, A)))
    achieved = float(max(mu.max() - 1.0, 1.0 - mu.min()))

    Z1, Z2 = _psd_part(lower.dual_value), _psd_part(upper.dual_value)
    shift = max(0.0, max(float(np.sum((Z1 - Z2) * a) / np.trace(a)) for a in A))
    Z2 = Z2 + shift * np.eye(k)
    bound = float((np.trace(Z1) - np.trace(Z2)) / (np.trace(Z1) + np.trace(Z2)))
    return scales, achieved, bound


def zonal_operator(L: NDArray, zones: NDArray, weights: NDArray | None = None) -> NDArray[np.float64]:
    """The exact operator of the zonal view: balanced zone injections, each spread evenly over the buses of
    its zone (the convention of the zonal PTDF, Fortenbacher et al. 2018, eq. (5)), mapped to the zone means
    of the angles, theta_z = P^T L^+ P p with P = T^T diag(1/size); the operator is the pseudoinverse of
    that map on balanced injections. It is symmetric positive semidefinite with the constants as null space
    but in general not a graph Laplacian (positive off-diagonal entries occur). It is to the zonal view
    what the Kron reduction onto one representative bus per zone is to the view from those buses. It is a
    known object: the complete aggregate K_a + K_a^i of area aggregation (Chow & Kokotovic 1985, eqs. (4.3),
    (4.4)), the Schur complement of the network onto the area coordinates, up to the scaling by the area
    sizes; its first term alone, the summed external connections, is their rigid aggregate (eq. (4.7)), and
    the second "accounts for the fact that the internal connections are not infinitely stiff". The two
    aggregates coincide when every zone is a single bus and in the limit of strong internal connections.

    `weights` (one nonnegative number per bus, normalized within each zone) say how a zone's injection is
    distributed over its buses and, by the same weights, how its angle is averaged: the participation factors
    of Cheng & Overbye 2005 (p. 1869: "the PTDFs depend upon the assumed generator participation values"); Chow
    & Kokotovic's area variable is the "center of inertia" of the area, the plain mean in their model with unit
    inertias (eq. (3.1)), an inertia-weighted mean otherwise. Equal weights give the operator above.
    All weight on one bus per zone gives the Kron reduction onto those buses (Dorfler & Bullo 2013, eq. (2.1)):
    then P^T L^+ P is the block of L^+ on those buses, whose pseudoinverse on balanced injections is the Schur
    complement. The Kron reduction and the complete aggregate are thus two members of one family.
    `zones` holds a zone number 0..z-1 per bus, every number used (an empty zone raises ValueError)."""
    zones = partition_labels(zones, L.shape[0])               # zone numbers 0..z-1, none empty
    n, nz = L.shape[0], int(zones.max()) + 1
    w = _zone_weights(weights, zones, n)
    P = np.zeros((n, nz)); P[np.arange(n), zones] = w
    P /= P.sum(0, keepdims=True)
    G = P.T @ laplacian_pseudoinverse(np.asarray(L, dtype=float)) @ P
    Pi = np.eye(nz) - 1.0 / nz
    G = Pi @ ((G + G.T) / 2) @ Pi
    Lz = np.linalg.pinv(G)
    return (Lz + Lz.T) / 2

