"""Spectral partitioning tools: Fiedler vector, normalized Laplacians, cuts, Cheeger bounds.

References
----------
* von Luxburg, "A Tutorial on Spectral Clustering", Statistics and Computing 17 (2007):
  unnormalized Laplacian L (Sec. 3.1), normalized L_sym = D^{-1/2} L D^{-1/2} and
  L_rw = D^{-1} L (Sec. 3.2, Prop. 3), spectral clustering algorithms (Sec. 4), and the
  graph-cut interpretation: unnormalized spectral clustering relaxes RatioCut,
  normalized spectral clustering relaxes Ncut (Secs. 5.1-5.3).
* Bandeira, Singer & Spielman, SIAM J. Matrix Anal. Appl. 34(4) (2013), Sec. 1.2:
  Cheeger constant h_G = min_S cut(S)/min(vol S, vol S^c), the inequality
  lambda_2(L_sym)/2 <= h_G <= sqrt(2 lambda_2(L_sym)) (Thm. 1.1), and its constructive
  proof: sweeping a threshold over the second eigenvector yields a set attaining the
  upper bound.
* Sanchez-Garcia et al., IEEE Trans. Power Systems 29(5) (2014), Sec. II-B: for power
  grids the natural edge weights are the line admittances 1/x, making L the DC
  susceptance matrix; the normalized Laplacian is used for the hierarchical clustering.

For the mathematician
---------------------
The Fiedler vector is the minimizer of the Rayleigh quotient x^T L x / x^T x on the
complement of the constants, i.e. the smoothest non-constant function on the graph in
the Dirichlet energy sum w_ij (x_i - x_j)^2. Thresholding it is the relaxation of the
NP-hard balanced-cut problems, and Cheeger's inequality is the guarantee that the
relaxation is not arbitrarily bad. Everything below is dense and intended for networks
up to a few thousand nodes, except `coarsened_eigenvalues` with k given, which accepts a
scipy sparse Laplacian and uses a sparse eigensolver.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from gridnet.graph import partition_labels


def _degrees(L: NDArray) -> NDArray:
    """Weighted degrees d_i = sum_j w_ij, read from the diagonal of a loop-less Laplacian. A matrix whose rows do not
    sum to zero (a shunt on the diagonal) raises ValueError, since its diagonal would count the shunt as degree."""
    A = np.asarray(L, dtype=float)
    d = A.diagonal().copy()
    rows = A.sum(axis=1)
    if np.any(np.abs(rows) > 1e-9 * max(np.abs(d).max(), 1e-300)):
        raise ValueError("the spectral functions need a loop-less Laplacian (rows summing to zero)")
    return d


def normalized_laplacian(L: NDArray, kind: str = "sym") -> NDArray[np.float64]:
    """L_sym = D^{-1/2} L D^{-1/2} or L_rw = D^{-1} L (von Luxburg 2007, Sec. 3.2).

    Both have the same eigenvalues, in [0, 2] for L_sym, with eigenvectors related by
    v_rw = D^{-1/2} v_sym (Prop. 3). Isolated nodes (degree 0) would make D singular
    and are rejected.
    """
    d = _degrees(L)
    if np.any(d <= 0):
        raise ValueError("normalized Laplacian needs positive degrees (no isolated nodes)")
    if kind == "sym":
        s = 1 / np.sqrt(d)
        return s[:, None] * L * s[None, :]
    if kind == "rw":
        return L / d[:, None]
    raise ValueError("kind must be 'sym' or 'rw'")


def fiedler(L: NDArray, normalized: bool = False) -> tuple[float, NDArray[np.float64]]:
    """Algebraic connectivity lambda_2 and Fiedler vector of L (von Luxburg 2007, Sec. 3).

    normalized=False: second eigenpair of L. normalized=True: second eigenpair of the
    generalized problem L v = lambda D v, i.e. of L_rw, obtained from the symmetric
    L_sym by v = D^{-1/2} u (Prop. 3(3)); the eigenvalue is that of L_sym. The vector is
    unit-norm with an arbitrary sign. For a disconnected graph lambda_2 = 0 and the
    vector lies in the kernel. If lambda_2 is a multiple eigenvalue (lambda_2 = lambda_3, as
    on a ring or another symmetric graph), the vector is one of a whole eigenspace, chosen by
    LAPACK and not by the graph; the sweep cuts and the bipartitions built on it, and the
    label numbering and `history` sides of `recursive_bipartition`, which follow its sign,
    are then one choice among many.
    """
    # only the two lowest eigenpairs are needed; scipy's subset driver avoids the full
    # decomposition, which matters at a few thousand nodes
    if normalized:
        d = _degrees(L)
        w, U = scipy.linalg.eigh(normalized_laplacian(L, "sym"), subset_by_index=[0, 1])
        v = U[:, 1] / np.sqrt(d)
        return float(w[1]), v / np.linalg.norm(v)
    w, U = scipy.linalg.eigh(np.asarray(L, dtype=float), subset_by_index=[0, 1])
    return float(w[1]), U[:, 1]


def cut_weight(L: NDArray, S: ArrayLike) -> float:
    """cut(S, S^c) = sum of weights across the partition = 1_S^T L 1_S."""
    x = np.asarray(S, dtype=bool).astype(float)
    return float(x @ L @ x)


def conductance(L: NDArray, S: ArrayLike) -> float:
    """h_S = cut(S)/min(vol S, vol S^c) with vol = sum of degrees (Bandeira 2013, Sec. 1.2)."""
    S = np.asarray(S, dtype=bool)
    d = _degrees(L)
    denom = min(d[S].sum(), d[~S].sum())
    return float(cut_weight(L, S) / denom) if denom > 0 else np.inf


def ratio_cut(L: NDArray, S: ArrayLike) -> float:
    """RatioCut(S, S^c) = cut(S) (1/|S| + 1/|S^c|) (von Luxburg 2007, Sec. 5, the definitions before 5.1).

    cut(S) = W(S, S^c) is the weight across the partition. von Luxburg writes RatioCut both as
    1/2 sum_i W(A_i, A_i^c)/|A_i| and as sum_i cut(A_i, A_i^c)/|A_i|, which differ by a factor 2 under her
    own definition of cut; this is the second form, the one of her Sec. 5.1 derivation f^T L f = |V| RatioCut.
    """
    S = np.asarray(S, dtype=bool)
    a, b = S.sum(), (~S).sum()
    return float(cut_weight(L, S) * (1 / a + 1 / b)) if a and b else np.inf


def ncut(L: NDArray, labels: ArrayLike) -> float:
    """Normalized cut of a k-way partition, Ncut = sum_i cut(A_i, V \\ A_i) / vol(A_i)
    (von Luxburg 2007, Sec. 5, the definitions before 5.1, in the second of her two forms as for
    `ratio_cut`); normalized spectral clustering is its relaxation (her Sec. 5.2)."""
    labels = np.asarray(labels)
    d = _degrees(L)
    total = 0.0
    for c in np.unique(labels):
        S = labels == c
        total += cut_weight(L, S) / d[S].sum()
    return float(total)


def sweep_cut(L: NDArray, v: ArrayLike, objective: str = "conductance") -> tuple[NDArray[np.bool_], float]:
    """Best threshold set of a vector: S_t = {i : v_i <= v_(t)} minimizing the objective.

    The rounding step of the Cheeger inequality proof (Bandeira 2013, Thm. 1.1): with v
    the normalized Fiedler vector and objective "conductance" the returned value is at
    most sqrt(2 lambda_2(L_sym)). With objective "ratio" it is the RatioCut counterpart
    for the unnormalized vector. Returns (indicator of S, objective value); S is the
    side containing the smallest entries of v.
    """
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    n = v.size
    d = _degrees(L)
    # incremental cut: adding node i moves its edges to S-nodes from "cut" to "inside"
    W = np.diag(d) - np.asarray(L, dtype=float)              # weighted adjacency
    inS = np.zeros(n, dtype=bool)
    cut = 0.0
    volS = 0.0
    best_val, best_t = np.inf, 0
    for t in range(n - 1):                                    # never the full set
        i = order[t]
        cut += d[i] - 2 * W[i, inS].sum()
        inS[i] = True
        volS += d[i]
        if objective == "conductance":
            val = cut / min(volS, d.sum() - volS)
        elif objective == "ratio":
            val = cut * (1 / (t + 1) + 1 / (n - t - 1))
        else:
            raise ValueError("objective must be 'conductance' or 'ratio'")
        if val < best_val:
            best_val, best_t = val, t
    S = np.zeros(n, dtype=bool)
    S[order[: best_t + 1]] = True
    return S, float(best_val)


def cheeger_bounds(L: NDArray) -> tuple[float, float]:
    """(lambda_2/2, sqrt(2 lambda_2)) for the normalized Laplacian: bounds on h_G (Thm. 1.1)."""
    lam, _ = fiedler(L, normalized=True)
    lam = max(lam, 0.0)
    return lam / 2, float(np.sqrt(2 * lam))


def spectral_embedding(L: NDArray, k: int, normalized: bool = True) -> NDArray[np.float64]:
    """First k eigenvectors as rows per node (von Luxburg 2007, Sec. 4).

    normalized=True gives the embedding of the Shi-Malik algorithm (eigenvectors of
    L_rw, i.e. generalized eigenvectors of (L, D)); False the unnormalized one.
    """
    # only the k lowest eigenpairs are needed; the subset driver is much faster than the full
    # decomposition at a few thousand nodes
    if normalized:
        d = _degrees(L)
        _, U = scipy.linalg.eigh(normalized_laplacian(L, "sym"), subset_by_index=[0, k - 1])
        return U / np.sqrt(d)[:, None]
    _, U = scipy.linalg.eigh(np.asarray(L, dtype=float), subset_by_index=[0, k - 1])
    return U


def spectral_clustering(L: NDArray, k: int, normalized: bool = True, seed: int | None = None) -> NDArray[np.int64]:
    """k-way spectral clustering: k-means on the spectral embedding (von Luxburg 2007, Sec. 4).

    Requires scikit-learn (the "ml" extra). Labels are arbitrary integers 0..k-1.
    """
    from sklearn.cluster import KMeans

    X = spectral_embedding(L, k, normalized)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X).astype(np.int64)


def variation_of_information(a: ArrayLike, b: ArrayLike) -> float:
    """Variation of information VI(a, b) = H(a) + H(b) - 2 I(a; b), Meila 2007, eq. (19).

    A metric on the partitions of a set (Meila 2007, Property 1), bounded by log n
    (Property 3) and by 2 log K for at most K clusters each (Property 4). Logarithms to
    base 2 as in the paper, so the unit is bits. The adjusted
    Rand index (Hubert & Arabie 1985, eqs. (4), (5)) and normalized mutual information are taken
    from scikit-learn; VI is not there and is computed from the contingency table.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    n = a.size
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1)
    p = table / n
    pa, pb = p.sum(axis=1), p.sum(axis=0)

    def H(q):
        q = q[q > 0]
        return -np.sum(q * np.log2(q))

    nz = p > 0
    mi = np.sum(p[nz] * np.log2(p[nz] / np.outer(pa, pb)[nz]))
    vi = H(pa) + H(pb) - 2 * mi
    # H + H - 2I is exactly zero for equal partitions; remove rounding residue there
    return float(0.0 if abs(vi) < 1e-12 else vi)


def recursive_bipartition(L: NDArray, k: int, normalized: bool = True, history: list | None = None) -> NDArray[np.int64]:
    """Partition into k parts by repeated two-way spectral cuts (Shi & Malik 2000, Sec. 3.2;
    von Luxburg 2007, Sec. 5 on the two-way relaxation).

    Start with the whole node set. Repeatedly compute, for every current part, the sweep
    cut of its normalized Fiedler vector on the induced subgraph and its conductance, and
    split the part whose sweep cut has the smallest conductance, until there are k parts.
    A part whose induced subgraph is disconnected has a cut of weight zero: it is split into
    the component of its first node and the rest, with conductance zero, before any connected
    part. This also covers a node left without an edge inside its part, which has degree zero
    and no normalized Laplacian (von Luxburg 2007, Sec. 3.2).
    Splitting the weakest cut first makes the sequence of cuts a hierarchy: the first cut
    is the sweep cut of the whole graph, the next the weakest cut inside either side, and
    so on. Labels are integers 0..k-1 in order of creation. If `history` is a list, one
    dict per split is appended to it with the parent label, the new label, the conductance
    of the split, and the node indices of both sides at the time of the split.
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    if k < 2 or k > n:
        raise ValueError("k must be between 2 and the number of nodes")
    labels = np.zeros(n, dtype=np.int64)
    best: dict[int, tuple[float, NDArray]] = {}

    def evaluate(part: int):
        idx = np.flatnonzero(labels == part)
        if idx.size < 2:
            return
        Ls = L[np.ix_(idx, idx)]
        Ls = Ls - np.diag(Ls.sum(axis=1))          # induced subgraph: drop the severed weight from the diagonal
        off = Ls.copy()
        np.fill_diagonal(off, 0.0)
        n_comp, comp = connected_components(csr_matrix(off != 0), directed=False)
        if n_comp > 1:                             # a cut of weight zero, found without an eigenvector
            best[part] = (0.0, idx[comp != comp[0]])
            return
        _, v = fiedler(Ls, normalized=normalized)
        S, h = sweep_cut(Ls, v, objective="conductance")
        best[part] = (h, idx[S])

    evaluate(0)
    while len(set(labels.tolist())) < k:
        if not best:
            raise ValueError("no part can be split further")
        part = min(best, key=lambda p: best[p][0])
        h, side = best.pop(part)
        new = int(labels.max()) + 1
        if history is not None:
            rest = np.setdiff1d(np.flatnonzero(labels == part), side)
            history.append({"parent": part, "new": new, "conductance": float(h), "new_side": side, "parent_side": rest})
        labels[side] = new
        evaluate(part); evaluate(new)
    return labels


def area_parameters(edges, labels: ArrayLike) -> dict:
    """Node and area parameters of a partition into areas (Chow & Kokotovic 1985, Sec. II-A).

    An internal connection joins two nodes of the same area, an external one nodes of different areas.
    With c^I the smallest number of internal connections of any node and c^E the largest number of external
    ones (their eq. (2.2)), the node parameter is d = c^E / c^I (eq. (2.3)); with gamma^E the largest number of
    external connections of any area and m the smallest number of nodes of an area, the area parameter is
    delta = gamma^E / (m c^I) (eq. (2.7)). Areas are internally dense and sparsely connected, and the network has two
    time scales, when both are small against one. "In a large network, a small number of nodes may violate
    this requirement. Then the average values ... can be used to define the node parameter d": `d_average`
    is the mean number of external over the mean number of internal connections per node, and
    `delta_average` the mean number of external connections of an area over the mean of m_a times the mean
    internal connections per node. Connections are counted, not weighted, as in the paper, which allows
    "strong but sparse external connections". Every entry of `edges` counts as one connection, so parallel
    edges (double-circuit lines entered as two lines) count twice; pass the distinct node pairs to count
    connections between nodes. Labels must be 0..k-1, each used. Returns a dict with d, delta, d_average, delta_average,
    c_internal_min, c_external_max."""
    labels = partition_labels(labels)
    n = len(labels)
    e = np.asarray(list(edges), dtype=int)
    same = labels[e[:, 0]] == labels[e[:, 1]]
    c_int = np.bincount(np.r_[e[same, 0], e[same, 1]], minlength=n)
    c_ext = np.bincount(np.r_[e[~same, 0], e[~same, 1]], minlength=n)
    k = int(labels.max()) + 1
    gamma_ext = np.bincount(np.r_[labels[e[~same, 0]], labels[e[~same, 1]]], minlength=k)
    size = np.bincount(labels, minlength=k)
    c_i, c_e = int(c_int.min()), int(c_ext.max())
    return {"d": c_e / c_i if c_i else float("inf"), "delta": gamma_ext.max() / (size.min() * c_i) if c_i else float("inf"),
                "d_average": float(c_ext.mean() / c_int.mean()), "delta_average": float(gamma_ext.mean() / (size.mean() * c_int.mean())),
                "c_internal_min": c_i, "c_external_max": c_e}


def _averaging_projector(labels: NDArray) -> tuple[NDArray, NDArray]:
    """Indicator matrix U of a partition and Pi = U (U^T U)^{-1} U^T, the averaging over its parts; Pi = P^+ P for the
    coarsening matrix P with rows 1/|set| (Loukas 2019, Def. 2.5 and Prop. 2.1)."""
    labels = partition_labels(labels)
    U = np.zeros((len(labels), int(labels.max()) + 1))
    U[np.arange(len(labels)), labels] = 1.0
    return U, (U / U.sum(0)) @ U.T


def restricted_spectral_error(L: NDArray, labels: ArrayLike, k: int) -> float:
    """The constant eps_k of restricted spectral approximation for the coarsening of L by a partition.

    Loukas 2019, Def. 3.1: L_c and L are (R, eps)-similar if || x - Pi x ||_L <= eps || x ||_L for all x in R,
    where Pi = P^+ P lifts the coarsened vector back, here the averaging over the parts, and || x ||_L^2 =
    x^T L x. With R = R_k, the span of the eigenvectors of the k smallest eigenvalues, and x = V Lambda^{-1/2} y,
    eps_k is the largest singular value of L^{1/2} (I - Pi) V Lambda^{-1/2} taken over the non-constant
    eigenvectors. eps_k = 0 if these eigenvectors are constant on the parts. His Thm. 3.3 bounds the first k
    eigenvalues of the coarsened graph when eps_k < sqrt(lambda_2 / lambda_k) (the restricted similarity
    gives || x - Pi x ||^2 <= eps_k^2 lambda_k || x ||^2 / lambda_2 on R_k), Thm. 3.4 the angles between the
    eigenspaces and Thm. 3.5 the cuts. His public code reports the norm-ratio form | || L^{1/2} Pi V Lambda^{-1/2} || - 1 |
    of Cor. 3.2 as `error_subspace`, which is at most eps_k. Raises ValueError for a disconnected graph,
    where lambda_2 = 0 and the scaling Lambda^{-1/2} does not exist. R_k is well defined only when
    lambda_k < lambda_{k+1}; if they are equal, the k - 1 eigenvectors taken here are one choice from the
    eigenspace and eps_k depends on that choice (Loukas's R_k assumes the gap)."""
    L = np.asarray(L, dtype=float)
    lam, V = np.linalg.eigh(L)
    if lam[1] <= L.shape[0] * np.finfo(float).eps * np.max(np.abs(lam)):
        raise ValueError("restricted_spectral_error needs a connected graph (lambda_2 = 0)")
    _, Pi = _averaging_projector(np.asarray(labels))
    X = V[:, 1:k] / np.sqrt(lam[1:k])                              # unit L-norm, the constant eigenvector left out
    R = X - Pi @ X
    return float(np.sqrt(max(np.linalg.eigvalsh(R.T @ L @ R).max(), 0.0)))


def coarsened_eigenvalues(L, labels: ArrayLike, k: int | None = None, seed: int | None = None) -> NDArray[np.float64]:
    """Eigenvalues of the coarsened Laplacian relative to the sizes of the parts, those of D^{-1/2} U^T L U D^{-1/2}.

    U^T L U is the coarsened Laplacian L_c = P^-+ L P^+ of Loukas 2019, Prop. 2.2, the quotient graph; the
    normalization by the sizes D = U^T U is the one of his public code (rows of C scaled by 1 / sqrt(size)) and of
    his Table 1, and the eigenvalues are those of M_a^{-1} K_a, the rigid aggregate of Chow & Kokotovic 1985,
    eq. (4.7), whose modes are their square roots.

    With k = None all eigenvalues are computed densely. With k given, L may be a scipy sparse matrix and the k
    smallest are computed by shift-invert Lanczos about a small negative shift (scipy's eigsh with sigma, ARPACK
    mode 3), which finds the zero eigenvalue reliably. ARPACK's which='SM' at a loose tolerance does not: it may
    stop before the zero mode has converged and return eigenvalues 2..k+1 (the evaluation function of Loukas's
    public code, tol = 1e-3, does so in about one run in ten on the Minnesota graph). `seed` fixes the starting
    vector. Labels must be 0..k-1, each used (`gridnet.graph.partition_labels`)."""
    from scipy.sparse import diags, issparse
    from scipy.sparse.linalg import eigsh

    labels = partition_labels(labels)
    n, parts = len(labels), int(labels.max()) + 1
    if k is None:
        U, _ = _averaging_projector(labels)
        d = U.sum(0)
        return np.linalg.eigvalsh((U.T @ np.asarray(L.toarray() if issparse(L) else L, dtype=float) @ U) / np.sqrt(np.outer(d, d)))
    U = csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, parts))
    scale = diags(1.0 / np.sqrt(np.asarray(U.sum(0)).ravel()))
    Lc = (scale @ (U.T @ csr_matrix(L) @ U) @ scale).tocsc()
    Lc = (Lc + Lc.T) / 2
    v0 = np.random.default_rng(seed).normal(size=parts)
    values = eigsh(Lc, k=k, sigma=-1e-3 * Lc.diagonal().max(), which="LM", v0=v0, return_eigenvectors=False)
    return np.sort(values)

