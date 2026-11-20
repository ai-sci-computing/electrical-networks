"""Sparse reductions of the European grid with guarantees, against the exact Kron reduction.

kron_vs_cluster.py measured PyPSA-Eur's 128-cluster network against the Kron reduction of the base network
onto one bus per cluster (Dorfler & Bullo 2013, eq. 2.1, Thm. 3.9): the Kron matrix is exact in
the effective resistances but dense, the network on the clustered lines alone is sparse but off
by factors. This script asks what lies between: graphs on the same nodes, the 96 clusters of
the 128-cluster run that lie on the continental component, with a prescribed
number of edges, judged by the spectral error

    eps = min { e : (1 - e) x^T L x <= x^T L~ x <= (1 + e) x^T L x for all x }

(Spielman & Srivastava 2011, eq. (2)) and by the effective resistances between all pairs.

Step 1 builds the Kron matrix as in kron_vs_cluster.py (transformer reactance 0.1 p.u., the bus
nearest to each cluster centroid as its representative) and caches it in results/kron128.npz (named after the
128-cluster run; the reduction has 96 nodes).
Step 2, this file's first method: Sparsify(G, q) of Spielman & Srivastava 2011, Sec. 1.1, on the
Kron matrix over a range of sample numbers q, against two baselines with a fixed edge set, the
Kron couplings restricted to the clustered lines (the Kron-adjacent graph of kron_vs_cluster.py) and PyPSA-Eur's
clustered network itself.
Step 3: ApproxSchur of Durfee et al. 2017, Algorithm 3 (CliqueSample of Kyng & Sachdeva 2016),
on the base network of 5,577 buses: the interior buses are eliminated one at a time and every
clique is sampled, so the dense matrix is never formed. The column q holds the number of copies
per unit leverage score.
Step 4: the best weights on a fixed sparse edge set (gridnet.reweight): the spectral optimum on the
clustered lines by a semidefinite program, which bounds every graph on those lines from below, the
Frobenius optimum, and the practice of power system equivalents, susceptances fitted to the zonal
PTDF (Fortenbacher et al. 2018, eqs. (5), (8), (16), after Oh 2010 and Shi & Tylavsky 2015).

Run: .venv/bin/python studies/aggregation/european_grid/sparse_reductions.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from _clustered import ARCHIVE, PYPSA_EUR, kept_buses, on_clusters, read_clustered

from gridnet.graph import effective_resistance, laplacian
from gridnet.kron import laplacian_from_couplings, reduced_couplings
from gridnet.reweight import (
    fill_zones,
    ptdf_from_susceptances,
    quotient_susceptances,
    reduced_ptdf,
    reweight_frobenius,
    reweight_ptdf,
    reweight_spectral,
    zonal_operator,
    zonal_ptdf_error,
)
from gridnet.sparsify import approx_schur, sparsify, spectral_error, spectral_error_best_scale

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
_log = open(RESULTS / "sparse_reductions.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/sparse_reductions.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()


def pieces(H):
    """The number of connected components of the reduced network and of its isolated nodes (a network that
    "falls apart")."""
    from scipy.sparse.csgraph import connected_components
    adj = (np.abs(H) > 0) & ~np.eye(len(H), dtype=bool)
    return connected_components(adj, directed=False)[0], int((adj.sum(1) == 0).sum())
CACHE = RESULTS / "kron128.npz"
TRAFO_X = 0.1
INPUTS = [PYPSA_EUR / "networks" / "clustered.nc", PYPSA_EUR / "busmap_simplify_network.csv",
          PYPSA_EUR / "busmap_cluster_network.csv", Path(__file__).parent / "PYPSA_EUR_COMMIT",
          *sorted(ARCHIVE.glob("*.csv"))]


def fingerprint() -> str:
    """SHA-256 over the inputs of the cache (PyPSA-Eur's clustered network and bus maps, the
    commit file, the archive), its settings, and the source of the code that builds it
    (this script and the gridnet modules it reduces with), stored with the cache; a changed
    input or a changed reduction makes main() rebuild it."""
    import gridnet.graph
    import gridnet.kron
    import gridnet.pypsa_io
    import gridnet.spectral
    h = hashlib.sha256(f"trafo_x={TRAFO_X};representatives=nearest, great-circle km".encode())
    for f in INPUTS + [Path(__file__), Path(__file__).parent / "_clustered.py",
                       *(Path(m.__file__) for m in (gridnet.graph, gridnet.kron, gridnet.pypsa_io, gridnet.spectral))]:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def cache_is_current() -> bool:
    if not CACHE.exists():
        return False
    with np.load(CACHE) as z:
        return "fingerprint" in z and str(z["fingerprint"]) == fingerprint()


def build_cache():
    """The Kron matrix W (couplings, zero diagonal), the clustered network's couplings Bc on the
    same clusters, their names and centroids; built by _clustered.py like kron_vs_cluster.py at x_T = 0.1, 'nearest'."""
    clustered = read_clustered()
    setup = on_clusters(clustered, TRAFO_X)
    clusters, cidx, labels, buses, L, Bc = (setup[k] for k in ("clusters", "cidx", "labels", "buses", "L", "Bc"))
    centroids, simp = clustered["centroids"], clustered["simp"]
    keep = kept_buses(clustered, setup, "nearest")
    W, _ = reduced_couplings(L, keep)
    # leverage scores b_e R_e of the base network's branches from one grounded inverse (Dorfler & Bullo 2013, eq. 1.3)
    G = np.zeros_like(L); G[1:, 1:] = np.linalg.inv(L[1:, 1:])
    i, j = np.nonzero(np.triu(-L, 1) > 0)
    scores = -L[i, j] * (G[i, i] + G[j, j] - 2 * G[i, j])
    zone = np.array([cidx.get(l, -1) if isinstance(l, str) else -1 for l in labels])     # cluster number per base bus, -1 without one
    np.savez_compressed(CACHE, W=W, Bc=Bc, clusters=np.array(clusters), xy=np.array([centroids[c] for c in clusters]),
                        base_i=i, base_j=j, base_b=-L[i, j], base_scores=scores, keep=keep, n_base=len(L), zone=zone,
                        bus_xy=buses[["x", "y"]].values, bus_id=buses.bus_id.values.astype(str),
                        simplified_bus=buses.bus_id.map(simp).fillna("").values.astype(str),
                        fingerprint=np.array(fingerprint()))


def resistance_ratios(L_approx, R_exact):
    """Share of pairs the reduction disconnects (infinite resistance) and the 5 %, 50 %, 95 % quantiles
    of R~ / R over the pairs it keeps connected."""
    iu = np.triu_indices(len(R_exact), 1)
    r = effective_resistance(L_approx)[iu] / R_exact[iu]
    cut = np.isinf(r)
    return (float(cut.mean()), *np.quantile(r[~cut], [0.05, 0.5, 0.95]))


def main():
    if not cache_is_current():
        build_cache()
    z = np.load(CACHE)
    W, Bc = z["W"], z["Bc"]
    n = len(W)
    Lk = laplacian_from_couplings(W)
    Rk = effective_resistance(Lk)
    adjacent = Bc > 0
    n_lines = int(np.triu(adjacent, 1).sum())
    print(f"{n} nodes; Kron matrix {int(np.triu(W > 0, 1).sum())} couplings of {n * (n - 1) // 2} pairs; clustered network {n_lines} lines")

    rows = []
    for name, La in (("clustered network (PyPSA-Eur)", laplacian_from_couplings(Bc)),
                     ("Kron couplings on the clustered lines", laplacian_from_couplings(W, mask=adjacent))):
        cut, q05, q50, q95 = resistance_ratios(La, Rk)
        rows.append({"method": name, "q": np.nan, "edges": n_lines, "eps": spectral_error(Lk, La), "eps_best": spectral_error_best_scale(Lk, La)[0],
                         "pairs_cut": cut, "R_q05": q05, "R_median": q50, "R_q95": q95})
    for q in (300, 600, 1200, 2500, 5000, 10000, 20000, 40000):
        runs = []
        apart = []
        for seed in range(10):
            H = sparsify(Lk, q, seed=seed)
            if pieces(H)[0] > 1:
                apart.append((seed, *pieces(H)))
            runs.append((int(np.count_nonzero(np.triu(H, 1))), spectral_error(Lk, H), spectral_error_best_scale(Lk, H)[0], *resistance_ratios(H, Rk)))
        e, eps, eps_best, cut, q05, q50, q95 = np.median(np.array(runs), axis=0)   # each column its own median: not one run
        print(f"Sparsify(Kron, {q}): runs that fall apart (seed, components, isolated nodes): {apart}")
        rows.append({"method": "Sparsify(Kron, q), median of 10 seeds", "q": q, "edges": int(e), "eps": eps, "eps_best": eps_best, "pairs_cut": cut, "R_q05": q05, "R_median": q50, "R_q95": q95})
    # method 2: approximate elimination of the base network itself, never forming the dense matrix
    from scipy.sparse import coo_matrix, diags
    i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
    A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
    Lb = (diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    print(f"base network {nb} buses, {len(b)} distinct branches, Foster sum {z['base_scores'].sum():.1f} = n - 1 = {nb - 1}")
    # the copies per unit leverage that Durfee et al. 2017, Algorithm 3 line 3 (printed p. 20) ask for,
    # rho_e = tau_e * 12 (eps/2)^-2 ln^2(3n/delta), at eps = delta = 1/2
    print(f"Durfee et al.'s copies per unit leverage at eps = delta = 1/2: {12 * (0.5 / 2) ** -2 * np.log(3 * nb / 0.5) ** 2:.0f}")
    for copies in (1, 3, 10, 30, 100, 300):
        runs = []
        apart = []
        for seed in range(10):
            H = approx_schur(Lb, z["keep"], copies=copies, seed=seed, scores=z["base_scores"])
            apart.append((int(np.count_nonzero(np.triu(H, 1))), *pieces(H)))
            runs.append((int(np.count_nonzero(np.triu(H, 1))), spectral_error(Lk, H), spectral_error_best_scale(Lk, H)[0], *resistance_ratios(H, Rk)))
        e, eps, eps_best, cut, q05, q50, q95 = np.median(np.array(runs), axis=0)
        print(f"ApproxSchur, {copies} copies, ten runs (edges, components, isolated kept buses): {apart}")
        rows.append({"method": "ApproxSchur(base network), median of 10 seeds", "q": copies, "edges": int(e), "eps": eps, "eps_best": eps_best, "pairs_cut": cut, "R_q05": q05, "R_median": q50, "R_q95": q95})
    # Durfee et al. 2017, Algorithm 3 in full: the elimination followed by a sparsification of its result
    for q in (600, 1200, 2500, 5000, 10000):
        runs = []
        for seed in range(10):
            S = approx_schur(Lb, z["keep"], copies=300, seed=seed, scores=z["base_scores"])
            H = sparsify(S, q, seed=seed)
            runs.append((int(np.count_nonzero(np.triu(H, 1))), spectral_error(Lk, H), spectral_error_best_scale(Lk, H)[0], *resistance_ratios(H, Rk)))
        e, eps, eps_best, cut, q05, q50, q95 = np.median(np.array(runs), axis=0)
        rows.append({"method": "ApproxSchur(300 copies) then Sparsify(q), median of 10 seeds", "q": q, "edges": int(e), "eps": eps, "eps_best": eps_best, "pairs_cut": cut, "R_q05": q05, "R_median": q50, "R_q95": q95})
    # step 4: the best weights on a fixed sparse edge set (gridnet.reweight)
    edges_c = [(int(a), int(c)) for a, c in zip(*np.nonzero(np.triu(adjacent, 1)))]
    w_spec, eps_spec, bound_spec = reweight_spectral(Lk, edges_c)
    print(f"semidefinite program on the {len(edges_c)} clustered lines: spectral error {eps_spec:.5f}, certified lower bound {bound_spec:.5f}, {int((w_spec > 1e-6 * w_spec.max()).sum())} lines with weight")
    w_frob = reweight_frobenius(Lk, edges_c)
    # zones for the PTDF fit: every base bus in the cluster of PyPSA-Eur's busmap, the buses the busmap lacks
    # (dropped by simplify_network) in the zone of the nearest labeled bus along the network, by reactance
    edges_base = list(zip(i.tolist(), j.tolist()))
    zone = fill_zones(z["zone"], edges_base, b)
    assert np.array_equal(zone[z["keep"]], np.arange(n)), "the kept bus of cluster c must lie in zone c"
    H_red, zone_edges = reduced_ptdf(nb, edges_base, b, zone)
    b_phys = quotient_susceptances(zone, edges_base, b, zone_edges)      # Fortenbacher et al. 2018, eq. (8)
    b_ptdf = reweight_ptdf(H_red, n, zone_edges, reference=b_phys)       # every block pinned, bridges at their physical sum
    err = lambda bb: np.linalg.norm(ptdf_from_susceptances(n, zone_edges, bb) - H_red) / np.linalg.norm(H_red)
    print(f"PTDF fit on {len(zone_edges)} interconnections: relative Frobenius error of the zonal PTDF {err(b_phys):.3f} (summed susceptances) -> {err(b_ptdf):.3f} (fitted)")
    def ptdf_error(La):
        return zonal_ptdf_error(La, zone_edges, H_red)

    for row, La in zip(rows[:2], (laplacian_from_couplings(Bc), laplacian_from_couplings(W, mask=adjacent))):
        row["ptdf_error"] = ptdf_error(La)
    exact_row = {"method": "exact Kron reduction", "q": np.nan, "edges": int(np.triu(W > 0, 1).sum()), "eps": 0.0, "eps_best": 0.0, "pairs_cut": 0.0,
                     "R_q05": 1.0, "R_median": 1.0, "R_q95": 1.0, "ptdf_error": ptdf_error(Lk)}
    for name, La in (("clustered lines, spectral optimum (SDP)", laplacian(n, edges_c, w_spec)),
                     ("clustered lines, Frobenius optimum (NNLS)", laplacian(n, edges_c, w_frob)),
                     ("zonal interconnections, summed susceptances", laplacian(n, zone_edges, b_phys)),
                     ("zonal interconnections, PTDF fit", laplacian(n, zone_edges, b_ptdf))):
        cut, q05, q50, q95 = resistance_ratios(La, Rk)
        eps = spectral_error(Lk, La)
        # a PTDF fixes the susceptances only up to a common factor: also report the error at the best factor
        rows.append({"method": name, "q": np.nan, "edges": int(np.count_nonzero(np.triu(La, 1))), "eps": eps, "pairs_cut": cut,
                         "R_q05": q05, "R_median": q50, "R_q95": q95, "eps_best": spectral_error_best_scale(Lk, La)[0],
                         "ptdf_error": ptdf_error(La)})
    rows.append(exact_row)
    # step 5: two references. The Kron matrix sees the network from one representative bus per cluster. The zonal
    # view of the PTDF practice spreads a zone's injection evenly over its buses (Fortenbacher et al. 2018, eq. (5));
    # its exact operator maps balanced zone injections p to the zone means of the angles, theta_z = P^T L^+ P p
    # with P = T^T diag(1/size), and L_z is the pseudoinverse of that map on balanced injections.
    import networkx as nx

    from gridnet.reweight import best_scales
    from gridnet.sparsify import range_whitening

    Lz = zonal_operator(Lb.toarray(), zone)
    offdiag = Lz - np.diag(np.diag(Lz))
    print(f"zonal operator L_z: smallest nonzero eigenvalue {np.linalg.eigvalsh(Lz)[1]:.3g}, positive off-diagonal mass {offdiag[offdiag > 0].sum() / np.abs(offdiag).sum():.3f} (0 for a graph Laplacian)")
    G = nx.Graph(zone_edges); where_e = {tuple(sorted(e)): k for k, e in enumerate(zone_edges)}
    parts = []
    for nodes in nx.biconnected_components(G):
        idx = [where_e[tuple(sorted(e))] for e in G.subgraph(nodes).edges()]
        parts.append(laplacian(n, [zone_edges[k] for k in idx], b_ptdf[idx]))
    ref_rows = []
    candidates = [("clustered network (PyPSA-Eur)", [laplacian_from_couplings(Bc)]),
                  ("Kron couplings on the clustered lines", [laplacian_from_couplings(W, mask=adjacent)]),
                  ("clustered lines, spectral optimum against Kron", [laplacian(n, edges_c, w_spec)]),
                  ("zonal interconnections, summed susceptances", [laplacian(n, zone_edges, b_phys)]),
                  ("zonal interconnections, PTDF fit, one factor", [laplacian(n, zone_edges, b_ptdf)]),
                  (f"zonal interconnections, PTDF fit, {len(parts)} free factors", parts),
                  ("exact Kron reduction", [Lk]), ("zonal operator L_z", [Lz])]
    for name, comps in candidates:
        row = {"reduction": name}
        for ref_name, ref in (("Kron", Lk), ("L_z", Lz)):
            _, eps_best, bound = best_scales(ref, comps)
            row[f"eps_vs_{ref_name}"] = eps_best; row[f"bound_vs_{ref_name}"] = bound
        for ref_name, ref in (("Kron", Lk), ("L_z", Lz)):
            # the same comparison in the mean over all directions: min_s || s A - I ||_F / sqrt(k), attained at
            # s = tr A / tr A^2, with A = ref^{+/2} L~ ref^{+/2}; equals sqrt(1 - (tr A)^2 / (k tr A^2))
            hf = range_whitening(ref)                                # ref^{+/2} on its range, the kernel removed exactly
            Am = hf.T @ sum(comps) @ hf
            row[f"mean_dev_vs_{ref_name}"] = float(np.sqrt(max(0.0, 1.0 - np.trace(Am) ** 2 / (Am.shape[0] * np.sum(Am * Am)))))
        row["ptdf_error"] = ptdf_error(sum(comps)) if name not in ("exact Kron reduction", "zonal operator L_z") else np.nan
        ref_rows.append(row)
    refs = pd.DataFrame(ref_rows)
    refs.to_csv(RESULTS / "sparse_reductions_references.csv", index=False)
    print("spectral error at the most favorable scale, against both references:")
    print(refs.round(4).to_string(index=False))
    _, eps_spec_z, bound_z = reweight_spectral(Lz, edges_c)
    print(f"spectral optimum on the clustered lines against L_z: {eps_spec_z:.5f} (certified {bound_z:.5f})")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sparse_reductions.csv", index=False)
    print(out.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    styles = {"Sparsify(Kron": ("tab:blue", "o", "sampling the exact Kron matrix"),
              "ApproxSchur(base": ("tab:orange", "s", "approximate elimination of the base network"),
              "ApproxSchur(300": ("tab:green", "^", "approximate elimination, then sampling")}
    for key, (color, marker, label) in styles.items():
        d = out[out.method.str.startswith(key)]
        axes[0].plot(d.edges, d.eps, marker=marker, color=color, lw=1.5, ms=5, label=label)
        d = d[d.pairs_cut == 0]                                  # resistances only where at least six of ten runs are connected
        axes[1].plot(d.edges, d.R_median, marker=marker, color=color, lw=1.5, ms=5)
        axes[1].fill_between(d.edges, d.R_q05, d.R_q95, color=color, alpha=0.12, lw=0)
    for name, color, label in (("clustered network", "tab:red", "PyPSA-Eur's clustered network"), ("Kron couplings", "0.35", "Kron couplings on the clustered lines")):
        d = out[out.method.str.startswith(name)].iloc[0]
        axes[0].plot(d.edges, d.eps, "*", color=color, ms=13, label=label)
        axes[1].errorbar(d.edges, d.R_median, yerr=[[d.R_median - d.R_q05], [d.R_q95 - d.R_median]], fmt="*", color=color, ms=13, capsize=3)
    axes[0].axhline(1.0, color="0.6", lw=0.8, ls=":"); axes[1].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].set_xlabel("edges"); axes[1].set_xlabel("edges")
    axes[0].set_ylabel("spectral error"); axes[1].set_ylabel("effective resistance / exact")
    axes[0].set_title(f"Error against the exact Kron reduction, {n} nodes"); axes[1].set_title("Electrical distances over all pairs: median, 5 % to 95 %")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    figdir = Path(__file__).parent / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "sparse_reductions.png", dpi=200)
    print("wrote", figdir / "sparse_reductions.png")


if __name__ == "__main__":
    main()
