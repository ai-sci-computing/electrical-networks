"""Oriented incidence matrix and weighted graph Laplacian.

Mathematical setting
--------------------
A network is a finite graph with n nodes (buses) and m edges (branches: lines and
transformers). We fix an arbitrary orientation of every edge; for a power network the
natural orientation is the "from" bus to the "to" bus of the branch data. Orientation
never affects the Laplacian, only the sign convention of edge quantities such as flows
and angle differences.

Sources
-------
* Oriented incidence matrix and the factorization L = A W A^T:
  von Luxburg, "A Tutorial on Spectral Clustering", Statistics and Computing 17 (2007),
  Sec. 3.1, Prop. 1 (the quadratic form f^T L f = 1/2 sum w_ij (f_i - f_j)^2 is exactly
  (A^T f)^T W (A^T f)).
* Sign convention (+1 at the from-bus, -1 at the to-bus): MATPOWER, makeBdc, where the
  branch-by-bus "connection matrix" Cft = Cf - Ct is the transpose of A used here.
  Zimmerman, Murillo-Sanchez, Thomas, IEEE Trans. Power Systems 26(1) (2011), Sec. II.B
  (Cf, Ct) and Sec. II.G, eqs. (26), (27) (Bf, Bbus), and the MATPOWER 8.0 User's Manual
  (2024), Sec. 3.7 (DC modeling).
* Rank of A equals n minus the number of connected components: since L = A A^T has the
  same rank as A, this is von Luxburg (2007), Prop. 2 (multiplicity of the eigenvalue 0
  of L equals the number of connected components).

Electrical reading
------------------
For the DC power flow (MATPOWER manual Sec. 3.7), with w_l = 1/x_l the inverse series
reactance of branch l (in per-unit), the matrix L = A W A^T is MATPOWER's Bbus and the
linearized branch flows are f = W A^T theta, i.e. P = L theta. The imaginary part of the
bus admittance matrix of a lossless, shunt-free network is Im Y = -L (see
`gridnet.admittance`).

Most branches have x > 0 and positive weights; series capacitors (x < 0, in MATPOWER's
case300 and case9241pegase) give negative weights and an indefinite L. `laplacian`,
`laplacian_pseudoinverse` and `effective_resistance` accept them; the PSD, kernel and
metric statements below hold for nonnegative (positive) weights, and `range_whitening`
refuses an indefinite L.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

Edge = tuple[int, int]


def incidence_matrix(n_nodes: int, edges: Sequence[Edge]) -> NDArray[np.float64]:
    """Oriented node-by-edge incidence matrix A of shape (n_nodes, len(edges)).

    Column l of A is e_i - e_j for the edge l = (i, j): +1 at the from-node i and -1 at
    the to-node j (MATPOWER convention, Zimmerman et al. 2011, Sec. II.B, transposed).
    Consequently every column sums to zero, A^T 1 = 0, and for a vector theta of nodal
    values, (A^T theta)_l = theta_i - theta_j is the difference across the edge.

    Parameters
    ----------
    n_nodes : number of nodes; nodes are indexed 0 .. n_nodes-1.
    edges   : sequence of (from, to) index pairs. Parallel edges are allowed and get
              separate columns; self-loops are rejected because they contribute nothing
              to A (e_i - e_i = 0) and would silently vanish.

    Raises
    ------
    ValueError for a self-loop or an index outside 0 .. n_nodes-1.
    """
    _check_edges(n_nodes, edges)
    m = len(edges)
    A = np.zeros((n_nodes, m), dtype=np.float64)
    for l, (i, j) in enumerate(edges):
        A[i, l] = 1.0
        A[j, l] = -1.0
    return A


def _check_edges(n_nodes: int, edges: Sequence[Edge]) -> None:
    """ValueError for a self-loop or an index outside 0 .. n_nodes-1."""
    for l, (i, j) in enumerate(edges):
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError(f"edge {l} = ({i}, {j}) has an index outside 0..{n_nodes - 1}")
        if i == j:
            raise ValueError(f"edge {l} = ({i}, {j}) is a self-loop")


def laplacian(
    n_nodes: int,
    edges: Sequence[Edge],
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Weighted graph Laplacian L = A diag(w) A^T of shape (n_nodes, n_nodes).

    Entrywise (von Luxburg 2007, Sec. 3.1): L_ii = sum of the weights of edges at node i,
    L_ij = -(sum of weights of edges between i and j) for i != j. L is symmetric, positive
    semidefinite for nonnegative weights, and its kernel contains the constant vector.
    The dimension of the kernel equals the number of connected components when all
    weights are positive (von Luxburg 2007, Prop. 2).

    With w_l = 1/x_l this is the DC power flow matrix Bbus of MATPOWER (manual Sec. 3.7,
    eq. 3.31; Zimmerman et al. 2011, eq. (27)).

    Parameters
    ----------
    n_nodes : number of nodes.
    edges   : sequence of (from, to) index pairs, as for `incidence_matrix`.
    weights : one weight per edge; defaults to all ones (the combinatorial Laplacian).

    Raises
    ------
    ValueError if the number of weights differs from the number of edges.
    """
    A = incidence_matrix(n_nodes, edges)
    m = A.shape[1]
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (m,):
        raise ValueError(f"expected {m} weights, got shape {w.shape}")
    # A * w scales column l by w_l, so (A * w) @ A.T == A @ diag(w) @ A.T without forming
    # the m-by-m diagonal matrix.
    return (A * w) @ A.T


class IllConditionedLaplacian(ValueError):
    """A connected Laplacian with positive weights whose smallest nonzero eigenvalue lies below the
    rounding of its entries (`_range_spectrum`): it has no pseudoinverse to any digit."""


def _range_spectrum(L: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The nonzero eigenvalues w and orthonormal eigenvectors X (n x r) of a symmetric Laplacian,
    L = X diag(w) X^T, with its kernel found as `laplacian_pseudoinverse` describes: per connected
    component of the nonzero pattern, the constants removed exactly through the Householder
    reflector H = I - 2 u u^T / (u^T u) that maps 1/sqrt(m) to e_1 (its last m - 1 columns are an
    orthonormal basis of the zero-sum vectors, and H L H costs two rank-one updates); a component
    whose row sums are not small against that spectrum, and larger than the rounding of the sums
    themselves (m eps times the absolute row sum), is a loopy Laplacian and decomposed directly;
    within it an eigenvalue counts as zero when |lambda| <= m eps max|lambda|. On the zero-sum
    vectors of a component whose weights are all positive no eigenvalue is zero (a connected graph
    has one zero eigenvalue), so one below that cutoff is a lambda_2 the matrix does not determine
    any more, the weak weights lost in the rounding of the diagonal: `IllConditionedLaplacian`, a ValueError. The components are those
    of the matrix: parallel edges whose weights cancel leave a zero entry, L is then block diagonal
    there, and its kernel and pseudoinverse are those of separate components."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    pattern = L != 0
    np.fill_diagonal(pattern, False)
    _, comp = connected_components(csr_matrix(pattern), directed=False)
    ws, cols = [], []
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        m = len(idx)
        A = L[np.ix_(idx, idx)]
        if m == 1:
            if abs(A[0, 0]) > 0:                               # an isolated node with a shunt (loopy)
                ws.append(A[0, 0:1].copy()); x = np.zeros((n, 1)); x[idx, 0] = 1.0; cols.append(x)
            continue
        u = np.full(m, 1 / np.sqrt(m)); u[0] -= 1.0
        cu = u @ u
        Au = A @ u
        HAH = A - 2 * np.outer(u, Au) / cu - 2 * np.outer(Au, u) / cu + 4 * (u @ Au) * np.outer(u, u) / cu**2
        Lc = HAH[1:, 1:]
        w, V = np.linalg.eigh((Lc + Lc.T) / 2)
        eps = np.finfo(float).eps
        keep = np.abs(w) > m * eps * np.max(np.abs(w))
        rows = np.abs(A.sum(axis=1))
        if (keep.any() and np.max(rows) > 1e-6 * np.min(np.abs(w[keep]))
                and np.any(rows > m * eps * np.abs(A).sum(axis=1))):
            # the row sums are not small against the spectrum on the zero-sum vectors: a loopy
            # Laplacian (shunts on the diagonal), whose constants are no kernel; decomposed directly.
            # Rounding in a reduction formed by cancellation leaves row sums many orders below it;
            # a true shunt below 1e-6 of that spectrum is taken for rounding and its constants removed.
            # Rounding in a Laplacian built from widely spread weights leaves row sums of the order of
            # eps times the row's own entries, which can exceed 1e-6 of a small lambda_2: the second
            # condition, a row sum above the worst rounding of its m terms, keeps these out.
            w, V = np.linalg.eigh((A + A.T) / 2)
            keep = np.abs(w) > m * np.finfo(float).eps * np.max(np.abs(w))
            QV = V[:, keep]
        else:
            if not keep.all() and not np.any(np.triu(A, 1) > 0):
                # all weights positive (no positive off-diagonal entry) and the pattern connected: no
                # eigenvalue is zero on the zero-sum vectors, so the ones below the cutoff are real
                # and below what the entries of the matrix carry
                raise IllConditionedLaplacian(
                    f"the Laplacian is too badly conditioned to resolve: a connected component of {m} nodes with positive "
                    f"weights has {int((~keep).sum())} eigenvalue(s) on the zero-sum vectors below m eps = {m * eps:.1e} times "
                    f"its largest ({np.max(np.abs(w)):.3g}), which rounding in its entries has destroyed; a weight that weak "
                    "is a separation at this precision (remove the branch) or the weights need rescaling")
            QV = np.vstack([np.zeros((1, keep.sum())), V[:, keep]])
            QV -= 2 * np.outer(u, u @ QV) / cu                  # H applied to the kept eigenvectors
        x = np.zeros((n, QV.shape[1])); x[idx] = QV
        ws.append(w[keep]); cols.append(x)
    if not ws:
        return np.zeros(0), np.zeros((n, 0))
    return np.concatenate(ws), np.hstack(cols)

def laplacian_pseudoinverse(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Moore-Penrose pseudoinverse L^+ of a symmetric Laplacian.

    L is singular (L 1 = 0), so the inverse is taken on the orthogonal complement of the
    kernel: L L^+ = L^+ L = I - 1 1^T / n for a connected graph, and L^+ 1 = 0. The kernel
    of a Laplacian holds the constants on every connected component (of the nonzero pattern
    of L), and these are removed exactly: on each component L is written in an orthonormal
    basis Q of the vectors with zero sum, Q^T L Q is decomposed, and L^+ = Q (Q^T L Q)^+ Q^T.
    A cutoff on the eigenvalues alone fails on a Laplacian formed by cancellation (a Kron
    reduction or a quotient of a network with large internal weights), whose computed zero
    eigenvalue can exceed n eps max|lambda| and is then inverted. A component whose row sums are
    not small against its spectrum on the zero-sum vectors (above 1e-6 times the smallest nonzero
    eigenvalue there: a loopy Laplacian, with shunts on the diagonal) has no constant kernel and is
    decomposed directly, which gives its inverse where it is regular. The two cases are told apart
    by that ratio alone, so both sides of 1e-6 have a limit: a true shunt below it is taken for
    rounding, and a matrix formed by cancellation whose row sums keep the rounding of its large
    weights (about eps times the largest weight eliminated, so weights more than about 5e9 times
    lambda_2) is taken for a loopy one and its rounding inverted. `gridnet.kron.kron_reduce` sets
    such row sums to zero; a reduction made otherwise should do the same (audit of repairs,
    2026-11-20, A1). Within the complement an
    eigenvalue counts as zero when |lambda| <= tol = n eps max|lambda|, a further kernel that
    only a Laplacian with negative weights can have; with positive weights alone such an eigenvalue
    is a lambda_2 lost to rounding (a condition number above 1/(n eps), 2e12 at 2,000 nodes), and
    the matrix is refused (ValueError) where it used to be returned as if separated. Row sums no
    larger than the rounding of the sums themselves are never taken for shunts. Negative eigenvalues are kept: a negative weight (a series
    capacitor, x < 0, as in MATPOWER's case300 and case9241pegase) makes L indefinite, and
    the Moore-Penrose inverse (Penrose's conditions L L^+ L = L, L^+ L L^+ = L^+) inverts
    every nonzero eigenvalue whatever its sign. Physically L^+ is the "impedance matrix"
    of the resistive network: potentials V = L^+ I for a balanced current injection I
    (Dorfler & Bullo 2013, Remark 1.1).
    """
    w, X = _range_spectrum(L)
    return (X / w) @ X.T

def range_whitening(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """The n x r matrix H = V_r Lambda_r^{-1/2} with H^T L H = I_r, i.e. L^{+/2} restricted to the range of L.

    V_r and Lambda_r are the eigenvectors and the r nonzero eigenvalues of a symmetric positive
    semidefinite L. A second quadratic form L~ is compared with L on the range of L through the
    eigenvalues of H^T L~ H, the relative spectrum of Spielman & Srivastava 2011, eq. (2), and of the
    reweighting and sparsification in `gridnet.reweight` and `gridnet.sparsify`. The kernel is that of
    `laplacian_pseudoinverse`: the constants of every component removed exactly, and within their
    complement the eigenvalues with |lambda| <= n eps max|lambda|.

    Raises ValueError if L has an eigenvalue below -tol: those spectral statements hold for positive
    semidefinite L only, and a Laplacian with a negative weight (a series capacitor) is indefinite.
    """
    w, X = _range_spectrum(L)
    if w.size and w.min() < 0:
        raise ValueError(f"the reference must be positive semidefinite; it has the eigenvalue {w.min():.3g} "
                         "(a negative weight, e.g. a series capacitor with x < 0)")
    return X / np.sqrt(w)


def effective_resistance(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix of effective resistances R_ij = (e_i - e_j)^T L^+ (e_i - e_j).

    Dorfler & Bullo 2013, eq. (1.3): R_ij = L^+_ii + L^+_jj - 2 L^+_ij is the potential
    difference between i and j when a unit current enters at i and leaves at j
    (Remark 1.1). R is symmetric with zero diagonal and, for positive weights, a metric on the
    nodes; between nodes of different components it is infinite. With a negative weight (a series
    capacitor) the series and parallel rules still hold but R may be negative. With
    edge weights b = 1/x, the resistance across a branch times its own weight, b_k R_k,
    is the fraction of a unit transfer between its endpoints that flows through the
    branch itself, the diagonal entry of the branch-to-branch power transfer distribution
    factors of the linear flow model.
    """
    Lp = laplacian_pseudoinverse(L)
    d = np.diag(Lp)
    R = d[:, None] + d[None, :] - 2 * Lp
    # across components no current can flow and the resistance is infinite; the formula above is
    # finite there (0 between two isolated nodes), so the components are found and the blocks set
    offdiag = np.asarray(L, dtype=float).copy()
    np.fill_diagonal(offdiag, 0.0)
    n_comp, label = connected_components(csr_matrix(offdiag != 0), directed=False)
    if n_comp > 1:
        R = np.where(label[:, None] == label[None, :], R, np.inf)
    return R


def partition_labels(labels: ArrayLike, n: int | None = None) -> NDArray[np.int64]:
    """Labels of a partition as integers 0..k-1 with every label used, validated.

    The coarsening and zonal functions of `gridnet.spectral` and `gridnet.reweight` index the
    parts by their labels (the columns of the indicator matrix U of Loukas 2019, Def. 2.5, or of
    the zone matrix T_bz of Fortenbacher et al. 2018, eq. (5)) and divide by the part sizes. An
    unused label is an empty part, a zero column and a division by zero, so it is refused rather
    than relabeled: the labels also name the parts in the results (the zone pairs of
    `reweight.reduced_ptdf`). `n`, if given, is the required number of labels (nodes).

    Raises ValueError for labels that are not the integers 0..k-1, each used at least once.
    """
    raw = np.asarray(labels)
    lab = raw.astype(np.int64)
    if lab.ndim != 1 or (n is not None and lab.size != n) or not np.array_equal(lab, raw):
        raise ValueError(f"labels must be one integer per node (expected {n} labels), got shape {raw.shape}")
    if lab.size == 0 or lab.min() < 0:
        raise ValueError("labels must be the integers 0..k-1; negative labels are not allowed")
    unused = np.flatnonzero(np.bincount(lab) == 0)
    if unused.size:
        raise ValueError(f"labels must be the integers 0..k-1, each used at least once; unused: {unused[:10].tolist()}")
    return lab


def rerouting_distances(n_nodes: int, edges: Sequence[Edge], k: int, weights: ArrayLike | None = None,
                        disjoint: str = "vertex") -> NDArray[np.float64]:
    """Rerouting distance from edge k to every edge (Strake, Kaiser, Basiri, Ronellenfitsch & Witthaut 2019).

    Their Definition 2: the length of the shortest cycle that contains both edges, inf if there is none,
    0 for k itself; the length is the number of edges, or the sum of `weights` (positive). Their Appendix
    D (Lemma 4) computes it as a shortest pair of edge-disjoint paths by Suurballe's algorithm (Suurballe
    & Tarjan 1984). A cycle in Definition 2 visits no vertex twice, which needs vertex-disjoint paths;
    edge-disjoint paths may meet at a vertex, and their union is then a closed walk through it (two
    triangles sharing one vertex: inf against 6). `disjoint` chooses "vertex" (Definition 2, which the proof
    of Lemma 4 also needs) or "edge" (Lemma 4 as worded); their Table I is reproduced by "vertex" only.

    Every edge is subdivided at a midpoint into two halves of half its weight. A cycle through edges k
    and j is then a pair of disjoint paths from the midpoint of k to the midpoint of j, one leaving
    through each end of k and one arriving through each end of j, since a midpoint has only those two
    neighbors. For vertex-disjoint paths every vertex is split into an in-node and an out-node joined by
    an arc of capacity one. Suurballe's algorithm: a shortest-path tree from the source, then for each
    target one more shortest path in the residual graph (the arcs of the first path reversed) under the
    reduced costs c(u, v) + d(u) - d(v) >= 0; the pair costs 2 d(t) + d'(t). Python Dijkstra, one per
    target: fine for grids of a few hundred edges, slow for thousands.
    """
    import heapq

    if disjoint not in ("vertex", "edge"):
        raise ValueError(f"disjoint must be 'vertex' or 'edge', not {disjoint!r}")
    _check_edges(n_nodes, edges)
    m = len(edges)
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (m,) or np.any(w <= 0):
        raise ValueError("weights must be one positive number per edge")
    split = disjoint == "vertex"
    v_in = (lambda v: 2 * v) if split else (lambda v: v)
    v_out = (lambda v: 2 * v + 1) if split else (lambda v: v)
    base = 2 * n_nodes if split else n_nodes
    N = base + m                                                        # the midpoint of edge j is base + j
    out: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    if split:
        for v in range(n_nodes):
            out[v_in(v)].append((v_out(v), 0.0))
    for j, (a, b) in enumerate(edges):
        t, h = base + j, w[j] / 2
        for x in (a, b):
            out[v_out(x)].append((t, h))
            out[t].append((v_in(x), h))

    def dijkstra(source, arcs):
        dist = np.full(N, np.inf); pred = np.full(N, -1); dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            for v, c in arcs(u):
                if du + c < dist[v]:
                    dist[v] = du + c; pred[v] = u; heapq.heappush(heap, (du + c, v))
        return dist, pred

    S = base + k
    d, pred = dijkstra(S, lambda u: out[u])
    result = np.full(m, np.inf); result[k] = 0.0
    for j in range(m):
        t = base + j
        if j == k or not np.isfinite(d[t]):
            continue
        used, rev = set(), {}
        v = t
        while v != S:                                                   # the first path, reversed in the residual
            u = int(pred[v]); used.add((u, v)); rev.setdefault(v, []).append(u); v = u

        def residual(u, used=used, rev=rev):
            for v, c in out[u]:
                if (u, v) not in used and np.isfinite(d[v]):
                    yield v, c + d[u] - d[v]
            for v in rev.get(u, ()):
                yield v, 0.0

        d2, _ = dijkstra(S, residual)
        if np.isfinite(d2[t]):
            result[j] = 2 * d[t] + d2[t]
    return result
