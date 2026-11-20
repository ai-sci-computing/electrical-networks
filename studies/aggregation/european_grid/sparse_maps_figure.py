"""Two map figures of sparse reductions, in the style of the opening image of sparse_teaser_figure.py.

sparse_selection_vs_sampling.png: the 96 kept buses of the continental grid with two sparse reductions of the exact
Kron reduction onto them. Left, the deterministic selection of Batson, Spielman & Srivastava 2012
(gridnet.sparsify.bss_sparsify, d = 3); right, a sample of Spielman & Srivastava 2011 (gridnet.sparsify.sparsify,
q = 1200 draws, the run of median spectral error among seeds 0-9, as in bss_reduction.py). Edges longer than
800 km, which no line of the map matches, are orange.

zonal_weights.png: the lines between neighboring zones with two sets of susceptances, both measured against the
Kron reduction onto the kept buses. Left, the fit of the zonal power transfer distribution factors of transfers
between the kept buses (gridnet.reweight.reweight_ptdf with participation weights on the kept buses; Fortenbacher,
Demiray & Schaffner 2018, eqs. (5), (15), (16); Cheng & Overbye 2005, p. 1869), right, the spectral optimum on
PyPSA-Eur's clustered lines against the Kron reduction (gridnet.reweight.reweight_spectral, a semidefinite program
with a certified bound). Both are computed as in measure_references.py; their errors are printed and checked against
results/measure_references.csv (the rows "PTDF fit, kept buses" and "spectral optimum against Kron", read below;
0.163 and 0.985 for the fit, 0.715 and 0.650 for the optimum on 2026-11-20). Neither scale is physical in the comparison (the fit has
one free factor per block, the optimum is taken at the most favorable common factor), so each panel is drawn relative
to the median of its own susceptances.

Run: .venv/bin/python studies/aggregation/european_grid/sparse_maps_figure.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from gridnet.graph import laplacian
from gridnet.kron import laplacian_from_couplings
from gridnet.pypsa_io import great_circle_km
from gridnet.reweight import ptdf_from_susceptances, reduced_ptdf, reweight_ptdf, reweight_spectral
from gridnet.sparsify import bss_sparsify, sparsify

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
ROOT = HERE.resolve().parents[2]
NE = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.shp"
LON, LAT, STRETCH = (-11.0, 41.5), (34.5, 60.5), EUROPE_STRETCH
INK, ORANGE, BLUE = "#2b2b2a", "#eb6834", "#2a78d6"

z = np.load(HERE / "results" / "kron128.npz")
with open(HERE / "results" / "measure_references.csv") as f:   # the values this script reproduces
    import csv
    MEASURED = {row["network"]: row for row in csv.DictReader(f)}
FIT, OPT = MEASURED["PTDF fit, kept buses"], MEASURED["spectral optimum against Kron"]
xy = z["bus_xy"][z["keep"]]
Lk = laplacian_from_couplings(z["W"])
n = len(Lk)
land = gpd.read_file(NE).cx[LON[0]:LON[1], LAT[0]:LAT[1]]


def draw(ax, i, j, w, scale, mark_far=True):
    """Edges (i, j) with susceptances w on the map; width grows with the square root of w / scale, as in the
    opening image; edges longer than 800 km in orange if mark_far."""
    land.plot(ax=ax, facecolor="#efece6", edgecolor="white", linewidth=0.4, zorder=0)
    length = np.array([great_circle_km(xy[a], xy[b]) for a, b in zip(i, j)])
    width = np.clip(0.75 * np.sqrt(w / scale), 0.15, 2.2)
    far = (length > 800) if mark_far else np.zeros(len(i), dtype=bool)
    for sel, color, alpha, zo in ((~far, INK, 0.6, 1), (far, ORANGE, 0.95, 2)):
        ax.add_collection(LineCollection([xy[[a, b]] for a, b in zip(i[sel], j[sel])], colors=color,
                                         linewidths=width[sel] * (1.4 if color == ORANGE else 1.0), alpha=alpha,
                                         zorder=zo, capstyle="round"))
    ax.scatter(*xy.T, s=9, color=BLUE, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xlim(*LON); ax.set_ylim(*LAT); ax.set_aspect(STRETCH); ax.axis("off")
    return length, far


def error_at_best_factor(L, La):
    """Spectral error of La against L at the most favorable common factor, (kappa - 1)/(kappa + 1) with kappa the
    ratio of the extreme eigenvalues of L^{-1/2} La L^{-1/2} on the complement of the constants."""
    w, V = np.linalg.eigh(L)
    keep = w > len(L) * np.finfo(float).eps * w.max()
    half = V[:, keep] / np.sqrt(w[keep])
    mu = np.linalg.eigvalsh(half.T @ La @ half)
    return float((mu[-1] / mu[0] - 1) / (mu[-1] / mu[0] + 1))


def label(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8, va="top", color=INK)


# --- figure 1: selection against sampling ------------------------------------------------------------------------------
Hs = bss_sparsify(Lk, 3.0)
runs = sorted((error_at_best_factor(Lk, sparsify(Lk, 1200, seed=s)), s) for s in range(10))   # at the best common factor
seed = runs[len(runs) // 2][1]
Hq = sparsify(Lk, 1200, seed=seed)
scale = float(np.median(-np.triu(Hs, 1)[np.triu(Hs, 1) < 0]))           # one scale for both panels
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, H, name in zip(axes, (Hs, Hq), ("selection", "sampling")):
    i, j = np.nonzero(np.triu(-H, 1) > 0)
    length, far = draw(ax, i, j, -H[i, j], scale)
    eps = error_at_best_factor(Lk, H)                                     # the selection is already at its best factor
    label(ax, f"{name}, {len(i)} edges\nspectral error {eps:.2f}")
    print(f"{name}: {len(i)} edges, spectral error {eps:.2f}, {int(far.sum())} longer than 800 km, longest {length.max():.0f} km")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
fig.savefig(HERE / "figures" / "sparse_selection_vs_sampling.png", dpi=200, bbox_inches="tight", pad_inches=0.03, facecolor="white")

# --- figure 2: the fit of distribution factors against the spectral optimum -------------------------------------------
adjacent = z["Bc"] > 0
edges_c = [(int(a), int(c)) for a, c in zip(*np.nonzero(np.triu(adjacent, 1)))]
w_spec, eps_spec, bound_spec = reweight_spectral(Lk, edges_c)
print(f"spectral optimum on {len(edges_c)} clustered lines: {eps_spec:.3f} (certified {bound_spec:.3f}; measure_references.csv {float(OPT['eps_vs_Kron']):.3f}); at the best factor {error_at_best_factor(Lk, laplacian(n, edges_c, w_spec)):.3f}")

# zones as in sparse_reductions.py: PyPSA-Eur's busmap, buses it lacks in the zone of the nearest labeled bus
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
zone = z["zone"].copy()
lost = np.flatnonzero(zone < 0)
X = coo_matrix((np.r_[1 / b, 1 / b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).tocsr()
dist = dijkstra(X, directed=False, indices=lost)
dist[:, lost] = np.inf
zone[lost] = zone[np.argmin(dist, axis=1)]
point = np.zeros(nb); point[z["keep"]] = 1.0                   # the injection of a zone at its kept bus: the Kron family
H_phi, zone_edges = reduced_ptdf(nb, list(zip(i.tolist(), j.tolist())), b, zone, weights=point)
b_phys = np.zeros(len(zone_edges)); where = {e: k for k, e in enumerate(zone_edges)}
for a, c, be in zip(zone[i], zone[j], b):
    if a != c:
        b_phys[where[(min(a, c), max(a, c))]] += be
b_ptdf = reweight_ptdf(H_phi, n, zone_edges, reference=b_phys)
ptdf_err = np.linalg.norm(ptdf_from_susceptances(n, zone_edges, b_ptdf) - H_phi) / np.linalg.norm(H_phi)
eps_fit = error_at_best_factor(Lk, laplacian(n, zone_edges, b_ptdf))
print(f"fit on {len(zone_edges)} interconnections: distribution factor error {ptdf_err:.3f} (measure_references.csv {float(FIT['ptdf_vs_kept_buses']):.3f}), spectral error {eps_fit:.3f} ({float(FIT['eps_vs_Kron']):.3f})")


def zonal_flow_error(La):
    """Relative Frobenius distance between H_phi and the flows the graph La sends over the zone pairs, as in
    measure_references.py (zero where the graph has no edge between the pair)."""
    Lp = np.linalg.pinv(La)
    H = np.array([-La[a, c] * (Lp[a] - Lp[c]) for a, c in zone_edges])
    return float(np.linalg.norm(H - H_phi) / np.linalg.norm(H_phi))


ptdf_err_spec = zonal_flow_error(laplacian(n, edges_c, w_spec))
print(f"spectral optimum: distribution factor error {ptdf_err_spec:.3f} (measure_references.csv {float(OPT['ptdf_vs_kept_buses']):.3f})")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, edges, w, text in ((axes[0], zone_edges, b_ptdf, f"fit of the transfer distribution factors\ntransfer distribution factors {ptdf_err:.2f}, spectral error {eps_fit:.2f}"),
                           (axes[1], edges_c, w_spec, f"spectral optimum\ntransfer distribution factors {ptdf_err_spec:.2f}, spectral error {eps_spec:.2f}")):
    ea = np.array(edges)
    keep_w = w > 1e-6 * w.max()
    draw(ax, ea[keep_w, 0], ea[keep_w, 1], w[keep_w], float(np.median(w[keep_w])), mark_far=False)
    label(ax, text)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
fig.savefig(HERE / "figures" / "zonal_weights.png", dpi=200, bbox_inches="tight", pad_inches=0.03, facecolor="white")
