"""Spectral sparsification of a weighted graph by effective-resistance sampling.

Spielman & Srivastava, "Graph Sparsification by Effective Resistances", SIAM J. Comput. 40(6)
(2011), arXiv:0803.0929. A graph H on the nodes of G is a spectral sparsifier with error eps if

    (1 - eps) x^T L x  <=  x^T L~ x  <=  (1 + eps) x^T L x      for all x        (their eq. (2)),

which keeps every cut (x an indicator vector) and every eigenvalue within factors 1 -+ eps, and
every effective resistance within factors 1/(1 +- eps), since eq. (2) implies
x^T L^+ x / (1 + eps) <= x^T L~^+ x <= x^T L^+ x / (1 - eps) (the remarks after their Thm. 1).
In the power network L is the susceptance Laplacian of the linear flow model (`gridnet.admittance`),
x^T L x the losses-like quadratic form of an angle pattern, and the statement of interest is about
dense Laplacians: the Kron reduction of a network is a full matrix (`gridnet.kron`). `sparsify`
thins such a matrix with a guarantee; `approx_schur` never builds it, eliminating the interior
buses one at a time and sampling the clique each elimination would create (Kyng & Sachdeva 2016;
Durfee et al. 2017).

The sampling probability of an edge is proportional to w_e R_e, its weight times the effective
resistance between its endpoints. That product is the share of a unit transfer between the
endpoints that the edge itself carries (the diagonal of the branch-to-branch power transfer distribution factors): an edge with
many parallel paths is sampled rarely, a bridge has w_e R_e = 1 and is sampled most.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gridnet.graph import effective_resistance, range_whitening


def _edges_of(L, tol: float = 0.0) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Edges (i < j, sorted by row, then column) and weights w_ij = -L_ij > tol of a dense or scipy sparse Laplacian.

    Raises ValueError for a negative weight (a positive off-diagonal entry above 1e-12 of the largest
    entry, which leaves room for the rounding of a reduction formed by cancellation): the sampling and
    elimination of this module assume nonnegative weights, and dropping the edge would sparsify
    another graph."""
    if hasattr(L, "tocoo"):
        from scipy.sparse import triu
        U = triu(-L, 1).tocsr().tocoo()
        scale = abs(L).max() if L.nnz else 0.0
        if np.any(U.data < -1e-12 * scale):
            raise ValueError("the Laplacian has a negative weight; sparsification needs nonnegative weights")
        sel = U.data > tol
        return np.column_stack([U.row[sel], U.col[sel]]).astype(np.int64), U.data[sel].astype(float)
    A = np.asarray(L, dtype=float)
    W = np.triu(-A, 1)
    if np.any(W < -1e-12 * (np.abs(A).max() if A.size else 0.0)):
        raise ValueError("the Laplacian has a negative weight; sparsification needs nonnegative weights")
    i, j = np.nonzero(W > tol)
    return np.column_stack([i, j]).astype(np.int64), -A[i, j]


def leverage_scores(L: NDArray) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Edges of L and their sampling weights w_e R_e (Spielman & Srivastava 2011, Sec. 1.1).

    The scores lie in (0, 1], equal 1 exactly on bridges, and sum to n - c for c components
    (Foster's theorem: sum_e w_e R_e = tr(L^+ L)). Returns (edges as an array of
    pairs i < j, scores)."""
    edges, w = _edges_of(L)
    R = effective_resistance(np.asarray(L, dtype=float))
    return edges, w * R[edges[:, 0], edges[:, 1]]


def sparsify(L: NDArray, q: int, seed: int | None = None) -> NDArray[np.float64]:
    """H = Sparsify(G, q) of Spielman & Srivastava 2011, Sec. 1.1, as a Laplacian.

    Draw q edges independently with replacement, edge e with probability p_e proportional to
    w_e R_e, and give each draw the weight w_e / (q p_e), summing repeated draws. The result is
    unbiased, E L~ = L (their Sec. 3), has at most q edges, and by their Thm. 1 satisfies the
    spectral guarantee with error eps with probability at least 1/2 once q = 9 C^2 n log n / eps^2,
    for 1/sqrt(n) < eps <= 1 and n sufficiently large (the hypotheses of Thm. 1). The constant C
    (of their Lemma 5) is not explicit, so measure the error with `spectral_error`."""
    edges, scores = leverage_scores(L)
    w = -np.asarray(L, dtype=float)[edges[:, 0], edges[:, 1]]
    p = scores / scores.sum()
    counts = np.random.default_rng(seed).multinomial(q, p)
    new = np.where(counts > 0, w * counts / (q * p), 0.0)
    n = L.shape[0]
    H = np.zeros((n, n))
    H[edges[:, 0], edges[:, 1]] = -new
    H = H + H.T
    H[np.arange(n), np.arange(n)] = -H.sum(1)
    return H


def spectral_error(L: NDArray, L_approx: NDArray) -> float:
    """Smallest eps with (1 - eps) x^T L x <= x^T L~ x <= (1 + eps) x^T L x for all x
    (Spielman & Srivastava 2011, eq. (2)).

    On the range of L the two forms are compared through the eigenvalues mu of
    L^{+/2} L~ L^{+/2}; eps = max |mu - 1|. A component split in L~ gives mu = 0 and eps = 1.
    L~ must vanish on the null space of L for a finite answer: where it does not (it joins
    components that L keeps apart), some x has x^T L x = 0 < x^T L~ x, the inequality fails
    for every eps, and inf is returned. Raises ValueError unless L is positive semidefinite
    (`gridnet.graph.range_whitening`); a negative weight, a series capacitor, makes it indefinite."""
    half = range_whitening(L)                                   # L^{+/2} restricted to the range
    L_approx = np.asarray(L_approx, dtype=float)
    if _nonzero_on_kernel(half, L_approx):
        return np.inf
    mu = np.linalg.eigvalsh(half.T @ L_approx @ half)
    return float(np.max(np.abs(mu - 1.0)))


def _nonzero_on_kernel(half: NDArray, L_approx: NDArray) -> bool:
    """Whether the symmetric L~ is nonzero on the null space of L, given the range whitening of L:
    with U the orthonormal basis of the range (the normalized columns of `half`), L~ vanishes on
    the null space exactly when L~ = L~ U U^T; tested to 1e-8 of the largest entry of L~."""
    U = half / np.linalg.norm(half, axis=0)
    return bool(np.abs(L_approx - (L_approx @ U) @ U.T).max() > 1e-8 * np.abs(L_approx).max())


def spectral_error_best_scale(L: NDArray, L_approx: NDArray) -> tuple[float, float]:
    """Spectral error of L_approx against L at the most favorable common factor, and that factor.

    With mu_min and mu_max the extreme eigenvalues of L^{+/2} L~ L^{+/2} on the range of L, the factor c
    maps the interval to [c mu_min, c mu_max], and max(1 - c mu_min, c mu_max - 1) is least when the two
    are equal, at c = 2 / (mu_min + mu_max), where it is (kappa - 1)/(kappa + 1) with kappa = mu_max/mu_min.
    Derived here; the same closed form is the one-component case of gridnet.reweight.best_scales. A network
    that falls apart has mu_min = 0 and keeps the error one for every factor; the factor returned is then
    the one that puts mu_max at 2. An L_approx that is nonzero on the null space of L (it joins components
    of L) has an infinite error at every factor: (inf, nan), as in `spectral_error`. Raises ValueError
    unless L is positive semidefinite, and when L_approx
    is zero or indefinite on the range of L (no common factor makes it an approximation)."""
    half = range_whitening(L)
    L_approx = np.asarray(L_approx, dtype=float)
    if _nonzero_on_kernel(half, L_approx):
        return np.inf, np.nan
    mu = np.linalg.eigvalsh(half.T @ L_approx @ half)
    if mu[-1] <= 0:
        raise ValueError("L_approx is zero (or negative semidefinite) on the range of L")
    if mu[0] < -1e-10 * mu[-1]:
        raise ValueError(f"L_approx is indefinite on the range of L (smallest relative eigenvalue {mu[0]:.3g})")
    lo, hi = max(float(mu[0]), 0.0), float(mu[-1])
    c = 2.0 / (lo + hi)
    return float((hi - lo) / (hi + lo)), c


def bss_sparsify(L: NDArray, d: float, history: list | None = None) -> NDArray[np.float64]:
    """The deterministic sparsifier of Batson, Spielman & Srivastava, "Twice-Ramanujan Sparsifiers", SIAM J.
    Comput. 41 (2012), arXiv:0808.0163v3, as a Laplacian on the nodes of L.

    Theorem 1.1: for d > 1 every weighted graph on n vertices contains a reweighted subgraph H with at most
    ceil(d (n - 1)) edges and x^T L x <= x^T L_H x <= kappa x^T L x, kappa = (d + 1 + 2 sqrt d)/(d + 1 - 2 sqrt d).
    The proof is the algorithm (Sec. 3.2). With r = rank L, the vectors v_e = sqrt(w_e) L^{+/2} a_e in the range
    of L satisfy sum_e v_e v_e^T = I_r (their eq. after Thm. 3.1). Starting from A = 0 with barriers
    u_0 = r/eps_U and l_0 = -r/eps_L, every step shifts the barriers by delta_U and delta_L and adds t v v^T for
    one edge with

        L_A(v) >= 1/t >= U_A(v),                                                        (Lemma 3.5)
        U_A(v) = v^T (u' - A)^{-2} v / (Phi^u(A) - Phi^{u'}(A)) + v^T (u' - A)^{-1} v,   u' = u + delta_U   (Lemma 3.3)
        L_A(v) = v^T (A - l')^{-2} v / (Phi_{l'}(A) - Phi_l(A)) - v^T (A - l')^{-1} v,   l' = l + delta_L   (Lemma 3.4)

    where Phi^u(A) = tr (u - A)^{-1} and Phi_l(A) = tr (A - l)^{-1} are the upper and lower potentials (Def. 3.2),
    and delta_L = 1, eps_L = 1/sqrt d, delta_U = (sqrt d + 1)/(sqrt d - 1), eps_U = (sqrt d - 1)/(d + sqrt d)
    (proof of Thm. 3.1). Neither potential increases, no eigenvalue crosses a barrier, and after ceil(d r) steps
    lambda_max / lambda_min <= (u_0 + Q delta_U)/(l_0 + Q delta_L) <= kappa.

    The paper leaves two choices free: which admissible edge, and which t in the interval. Here the edge with the
    largest gap L_A - U_A is taken and 1/t is the midpoint of the interval. The result is returned in the two-sided
    scaling, multiplied by 2/(lambda_min + lambda_max), so that `spectral_error` gives (kappa' - 1)/(kappa' + 1)
    for the achieved condition number kappa', at most 2 sqrt(d)/(d + 1).

    L must be positive semidefinite (nonnegative weights); an indefinite L raises ValueError
    (`gridnet.graph.range_whitening`). If `history` is a list, one dict per step is appended with the
    barriers, the two potentials at those barriers and the extreme eigenvalues of A after the step. Cost
    O(Q (m r^2 + r^3)) with m edges, dense; meant for the reduced networks of this package (r about a
    hundred), not for the base network."""
    if d <= 1:
        raise ValueError("d must exceed 1 (Batson, Spielman & Srivastava 2012, Thm. 1.1)")
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    pairs, w = _edges_of(L)
    ei, ej = pairs[:, 0], pairs[:, 1]
    half = range_whitening(L)                                    # n x r, L^{+/2} restricted to the range of L
    Vec = (half[ei] - half[ej]) * np.sqrt(w)[:, None]            # m x r, rows v_e; sum_e v_e v_e^T = I_r
    r = Vec.shape[1]
    rd = np.sqrt(d)
    dL, eL, dU, eU = 1.0, 1.0 / rd, (rd + 1) / (rd - 1), (rd - 1) / (d + rd)
    u, l = r / eU, -r / eL
    A = np.zeros((r, r)); s = np.zeros(len(w))
    for _ in range(int(np.ceil(d * r))):
        a = np.linalg.eigvalsh(A)
        MU = np.linalg.inv((u + dU) * np.eye(r) - A); ML = np.linalg.inv(A - (l + dL) * np.eye(r))
        drop_U = np.sum(1 / (u - a)) - np.sum(1 / (u + dU - a))          # Phi^u(A) - Phi^{u'}(A) > 0
        rise_L = np.sum(1 / (a - l - dL)) - np.sum(1 / (a - l))          # Phi_{l'}(A) - Phi_l(A) > 0
        VU, VL = Vec @ MU, Vec @ ML
        UA = np.einsum("ij,ij->i", VU, VU) / drop_U + np.einsum("ij,ij->i", VU, Vec)
        LA = np.einsum("ij,ij->i", VL, VL) / rise_L - np.einsum("ij,ij->i", VL, Vec)
        k = int(np.argmax(LA - UA))
        if LA[k] < UA[k]:                                                # excluded by Lemma 3.5; would be a bug
            raise RuntimeError("no admissible edge: the invariants of the construction are violated")
        t = 2.0 / (LA[k] + UA[k])
        A += t * np.outer(Vec[k], Vec[k]); s[k] += t
        u += dU; l += dL
        if history is not None:
            a = np.linalg.eigvalsh(A)
            history.append({"upper_barrier": u, "lower_barrier": l, "upper_potential": float(np.sum(1 / (u - a))),
                                "lower_potential": float(np.sum(1 / (a - l))), "lambda_min": float(a[0]), "lambda_max": float(a[-1])})
    a = np.linalg.eigvalsh(A)
    s *= 2.0 / (a[0] + a[-1])                                            # two-sided scaling
    H = np.zeros_like(L)
    np.add.at(H, (ei, ej), -s * w); np.add.at(H, (ej, ei), -s * w)
    H[np.diag_indices(n)] = -H.sum(1)
    return H


def approx_schur(L: NDArray, keep, copies: float, seed: int | None = None, leverage: bool = True,
                 scores: NDArray | None = None) -> NDArray[np.float64]:
    """Sparse approximation of the Kron reduction (Schur complement) of a Laplacian onto `keep`.

    ApproxSchur of Durfee, Kyng, Peebles, Rao & Sachdeva 2017, Algorithm 3, without its final
    sparsification (apply `sparsify` to the result for that). Every edge is split into rho_e
    multi-edges of weight w_e / rho_e, with rho_e = ceil(copies * w_e R_e) in proportion to its
    leverage score (their line 3, where `copies` stands for 12 (eps/2)^-2 ln^2(3n/delta); with
    leverage=False every edge gets ceil(copies) copies as in Kyng & Sachdeva 2016, Algorithm 1).
    The vertices outside `keep` are eliminated in uniformly random order. Eliminating v exactly
    would join all its neighbors into a clique with weights w_i w_j / w(v) (the star-mesh
    transform, `gridnet.kron`); CliqueSample (Kyng & Sachdeva 2016, Algorithm 2) replaces the clique by
    deg(v) sampled multi-edges instead: draw one multi-edge at v with probability w(e)/w(v), a
    second uniformly, and if their far ends u1 != u2 differ add the edge u1-u2 with weight
    w(e1) w(e2) / (w(e1) + w(e2)). The expectation of the samples is the exact clique, and the
    number of multi-edges never grows, which is what keeps the reduction sparse. By Durfee et al.'s
    Thm. 2.2 the result approximates the exact Schur complement with spectral error eps with
    high probability; measure it with `spectral_error` against `gridnet.kron.kron_reduce`.

    `scores` may supply the leverage scores w_e R_e of the edges in the order of `leverage_scores`,
    which otherwise are computed from a dense pseudoinverse; L may be a scipy sparse matrix then.

    Returns the Laplacian on the kept nodes, in the order of `keep`."""
    n = L.shape[0]
    keep = [int(k) for k in keep]
    if len(set(keep)) != len(keep):
        raise ValueError("keep contains a node twice")
    rng = np.random.default_rng(seed)
    edges, w = _edges_of(L)
    if scores is not None:
        rho = np.ceil(copies * np.asarray(scores, dtype=float)).astype(int)
    elif leverage:
        R = effective_resistance(np.asarray(L, dtype=float))
        rho = np.ceil(copies * w * R[edges[:, 0], edges[:, 1]]).astype(int)
    else:
        rho = np.full(len(w), int(np.ceil(copies)))
    rho = np.maximum(rho, 1)
    # multi-edges at every vertex as lists of arrays (far end, weight); entries whose far end has been
    # eliminated are stale and dropped when the vertex is visited
    nbr = [[] for _ in range(n)]; wts = [[] for _ in range(n)]
    for (i, j), we, r in zip(edges, w, rho):
        for a, b in ((i, j), (j, i)):
            nbr[a].append(np.full(r, b)); wts[a].append(np.full(r, we / r))
    gone = np.zeros(n, dtype=bool)
    interior = np.setdiff1d(np.arange(n), keep)
    for v in rng.permutation(interior):
        gone[v] = True
        if not nbr[v]:
            continue
        u, wv = np.concatenate(nbr[v]), np.concatenate(wts[v])
        live = ~gone[u]; u, wv = u[live], wv[live]
        nbr[v], wts[v] = [], []
        d = len(u)
        if d == 0:
            continue
        e1 = rng.choice(d, size=d, p=wv / wv.sum()); e2 = rng.integers(0, d, size=d)
        ok = u[e1] != u[e2]
        a, b = u[e1][ok], u[e2][ok]
        new = wv[e1][ok] * wv[e2][ok] / (wv[e1][ok] + wv[e2][ok])
        for x in np.unique(np.r_[a, b]):
            at_a, at_b = a == x, b == x
            nbr[x].append(np.r_[b[at_a], a[at_b]]); wts[x].append(np.r_[new[at_a], new[at_b]])
    pos = -np.ones(n, dtype=int); pos[keep] = np.arange(len(keep))
    H = np.zeros((len(keep), len(keep)))
    for k in keep:
        if nbr[k]:
            u, wk = np.concatenate(nbr[k]), np.concatenate(wts[k])
            live = ~gone[u]
            np.add.at(H[pos[k]], pos[u[live]], -wk[live])
    H = (H + H.T) / 2                                            # each multi-edge is stored at both ends
    H[np.arange(len(keep)), np.arange(len(keep))] = 0.0
    H[np.arange(len(keep)), np.arange(len(keep))] = -H.sum(1)
    return H
