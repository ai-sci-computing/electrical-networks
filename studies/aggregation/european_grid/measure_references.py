"""The two measures of a sparse reduction taken against one reference at a time.

sparse_reductions.py compares networks on the lines between neighboring clusters by two measures, the
spectral error (Spielman & Srivastava 2011, eq. (2)) and the relative Frobenius error of the zonal PTDF
(Fortenbacher et al. 2018, eqs. (5), (16)). Its table takes the first against the Kron reduction onto one
bus per cluster and the second against the zonal PTDF with every zone's injection spread evenly over its
buses (Oh 2010), which belongs to the other exact reference, the complete aggregate with uniform
distributions (zonal_operator; Chow & Kokotovic 1985, eqs. (4.3), (4.4)). The question
here is whether the conclusion survives when both measures are
taken against the same reference. Both references are members of one family, the participation factors
of Cheng & Overbye 2005 (p. 1869): equal weights per zone, or all weight on the kept bus of each zone.
For each family this script measures every network by the spectral error at the most favorable common
factor against the family's exact operator, and by the PTDF error against the family's zonal PTDF, and it
adds the two networks that the comparison was missing: the spectral optimum against L_z, and the PTDF fit
to the single-bus zonal PTDF.

Run: .venv/bin/python studies/aggregation/european_grid/measure_references.py (needs results/kron128.npz
from sparse_reductions.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gridnet.graph import laplacian
from gridnet.kron import laplacian_from_couplings
from gridnet.reweight import (
    best_scales,
    fill_zones,
    quotient_susceptances,
    reduced_ptdf,
    reweight_ptdf,
    reweight_spectral,
    zonal_operator,
    zonal_ptdf_error,
)

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
_log = open(RESULTS / "measure_references.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/measure_references.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()


def main():
    z = np.load(RESULTS / "kron128.npz")
    W, Bc, keep = z["W"], z["Bc"], z["keep"]
    n = len(W)
    Lk = laplacian_from_couplings(W)
    i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
    edges_base = list(zip(i.tolist(), j.tolist()))
    # zones as in sparse_reductions.py: buses the busmap lacks join the zone of the nearest labeled bus by reactance
    zone = fill_zones(z["zone"], edges_base, b)
    assert np.array_equal(zone[keep], np.arange(n)), "the kept bus of cluster c must lie in zone c"
    point = np.zeros(nb); point[keep] = 1.0

    Ld = laplacian(nb, edges_base, b)
    Lz = zonal_operator(Ld, zone)
    print(f"L_z against the Kron reduction: spectral error at the best factor {best_scales(Lk, [Lz])[1]:.4f}")
    H_eq, zone_edges = reduced_ptdf(nb, edges_base, b, zone)
    H_pt, zone_edges_pt = reduced_ptdf(nb, edges_base, b, zone, weights=point)
    assert zone_edges_pt == zone_edges

    b_phys = quotient_susceptances(zone, edges_base, b, zone_edges)      # Fortenbacher et al. 2018, eq. (8)
    edges_c = [(int(a), int(c)) for a, c in zip(*np.nonzero(np.triu(Bc > 0, 1)))]
    w_spec_k, eps_k, bound_k = reweight_spectral(Lk, edges_c)
    w_spec_z, eps_z, bound_z = reweight_spectral(Lz, edges_c)
    print(f"spectral optimum on {len(edges_c)} lines: {eps_k:.5f} against Kron (certified {bound_k:.5f}), {eps_z:.5f} against L_z (certified {bound_z:.5f})")
    # the optimum and its certificate on the clustered lines that are also zone pairs (190 of the 194)
    common = sorted(set(edges_c) & {tuple(sorted(map(int, e))) for e in zone_edges})
    _, eps_c, bound_c = reweight_spectral(Lk, common)
    print(f"spectral optimum on the {len(common)} lines that are also zone pairs: {eps_c:.5f} against Kron (certified {bound_c:.5f})")
    # the optimum on the zone pairs alone and on the union of zone pairs and clustered lines
    zone_set = sorted({tuple(sorted(map(int, e))) for e in zone_edges})
    union = sorted(set(edges_c) | set(zone_set))
    for label, es in ((f"the {len(zone_set)} zone pairs", zone_set), (f"the union, {len(union)} pairs", union)):
        _, eps_u, bound_u = reweight_spectral(Lk, es)
        print(f"spectral optimum on {label}: {eps_u:.5f} against Kron (certified {bound_u:.5f})")
    b_fit_eq = reweight_ptdf(H_eq, n, zone_edges, reference=b_phys)
    b_fit_pt = reweight_ptdf(H_pt, n, zone_edges, reference=b_phys)

    def ptdf_error(La, H_ref):
        return zonal_ptdf_error(La, zone_edges, H_ref)

    networks = [("aggregated network of PyPSA-Eur", laplacian_from_couplings(Bc)),
                ("Kron couplings between neighbors", laplacian_from_couplings(W, mask=Bc > 0)),
                ("summed susceptances", laplacian(n, zone_edges, b_phys)),
                ("spectral optimum against Kron", laplacian(n, edges_c, w_spec_k)),
                ("spectral optimum against L_z", laplacian(n, edges_c, w_spec_z)),
                ("PTDF fit, equal division", laplacian(n, zone_edges, b_fit_eq)),
                ("PTDF fit, kept buses", laplacian(n, zone_edges, b_fit_pt))]
    rows = []
    for name, La in networks:
        rows.append({"network": name, "edges": int(np.count_nonzero(np.triu(La, 1))),
                         "eps_vs_Kron": best_scales(Lk, [La])[1], "ptdf_vs_kept_buses": ptdf_error(La, H_pt),
                         "eps_vs_L_z": best_scales(Lz, [La])[1], "ptdf_vs_equal_division": ptdf_error(La, H_eq)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "measure_references.csv", index=False)
    print(out.round(3).to_string(index=False))

    # The distribution-factor error is taken on the rows of the 191 zone pairs (zonal_ptdf_error). A network on the
    # 194 lines of PyPSA-Eur's aggregate has lines outside the pairs, whose flows are not scored, and lacks a pair,
    # which is scored at zero flow; both measured here against the norm of the reference.
    from gridnet.graph import laplacian_pseudoinverse
    pairs = set(map(tuple, zone_edges))
    print("lines outside the zone pairs:", sorted(set(map(tuple, edges_c)) - pairs),
          "; zone pairs without a line:", sorted(pairs - set(map(tuple, edges_c))))
    for name, La in networks:
        Lp = laplacian_pseudoinverse(La)
        out_lines = [(int(a), int(c)) for a, c in zip(*np.nonzero(np.triu(La, 1))) if (int(a), int(c)) not in pairs]
        h_out = np.array([-La[a, c] * (Lp[a] - Lp[c]) for a, c in out_lines]) if out_lines else np.zeros((0, n))
        lacking = [k for k, (a, c) in enumerate(zone_edges) if La[a, c] == 0]
        for ref_name, H_ref in (("kept buses", H_pt), ("equal division", H_eq)):
            nref = np.linalg.norm(H_ref)
            print(f"  {name}, {ref_name}: {len(out_lines)} unscored lines carrying {np.linalg.norm(h_out) / nref:.4f} of |H_ref|; "
                  f"{len(lacking)} pairs scored at zero, the reference there {np.linalg.norm(H_ref[lacking]) / nref:.4f}")

    # A transfer distribution factor fixes the susceptances only up to one factor per block of the zonal graph (its
    # biconnected components; a bridge is a block of its own and has no effect at all, Cheng & Overbye 2005). The
    # post's "with the seven numbers chosen in the fits' favor they are lower by 0.02 to 0.04": both fits with one
    # factor per block chosen to minimize the spectral error (gridnet.reweight.best_scales on the block Laplacians).
    import networkx as nx
    Gz = nx.Graph(zone_edges); where_e = {tuple(sorted(e)): k for k, e in enumerate(zone_edges)}
    blocks = [[where_e[tuple(sorted(e))] for e in Gz.subgraph(nodes).edges()] for nodes in nx.biconnected_components(Gz)]
    print(f"blocks of the zonal graph: {len(blocks)}, of sizes {sorted((len(x) for x in blocks), reverse=True)[:3]} ...")
    for name, bb in (("PTDF fit, equal division", b_fit_eq), ("PTDF fit, kept buses", b_fit_pt)):
        comps = [laplacian(n, [zone_edges[k] for k in idx], bb[idx]) for idx in blocks]
        one = (best_scales(Lk, [laplacian(n, zone_edges, bb)])[1], best_scales(Lz, [laplacian(n, zone_edges, bb)])[1])
        free = (best_scales(Lk, comps)[1], best_scales(Lz, comps)[1])
        print(f"{name}: spectral error against Kron {one[0]:.4f} -> {free[0]:.4f} ({free[0] - one[0]:+.4f}), "
              f"against L_z {one[1]:.4f} -> {free[1]:.4f} ({free[1] - one[1]:+.4f}) with one factor per block")


if __name__ == "__main__":
    main()
