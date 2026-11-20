"""The susceptance Laplacian of the continental synchronous area, its Fiedler vector,
the first sweep cut, and the sensitivity to the transformer reactance.

Network: PyPSA-Eur OSM archive 0.7, largest AC component (continental Europe), lines with
archive impedances, transformers with reactance x_T per unit on their rating (PyPSA-Eur
default 0.1), all converted to a 100 MVA base (gridnet.pypsa_io). Weighted Laplacian
L = A diag(1/x) A^T (gridnet.graph.laplacian); Fiedler vector and sweep cut of the normalized problem
(von Luxburg 2007, Sec. 3.2 and 5.3; Bandeira et al. 2013, Thm. 1.1); partitions for
different x_T compared by adjusted Rand index and variation of information.
Run:  python studies/aggregation/european_grid/continental_laplacian.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from sklearn.metrics import adjusted_rand_score

from gridnet.graph import laplacian
from gridnet.pypsa_io import read_osm_archive
from gridnet.spectral import cheeger_bounds, fiedler, sweep_cut, variation_of_information

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

TRAFO_X = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
REF = 0.1

rows = []
partitions = {}
for xT in TRAFO_X:
    t0 = time.time()
    net, info = read_osm_archive(ARCHIVE, component=0, trafo_x=xT)
    L = laplacian(net.n_bus, net.edges, 1 / net.x)
    lam2, v = fiedler(L, normalized=True)
    S, h = sweep_cut(L, v, objective="conductance")
    lo, hi = cheeger_bounds(L)
    lam2_u, _ = fiedler(L)
    buses = info["buses"]
    small = S if S.sum() <= (~S).sum() else ~S
    side = buses[small]
    cut_edges = [l for l, (i, j) in enumerate(net.edges) if S[i] != S[j]]
    n_cut_trafo = int(info["is_transformer"][cut_edges].sum())
    countries = side.country.value_counts()
    rows.append({
        "trafo_x": xT, "lambda2_norm": lam2, "lambda2_unnorm": lam2_u, "h_sweep": h,
        "cheeger_lo": lo, "cheeger_hi": hi, "small_side": int(small.sum()),
        "cut_edges": len(cut_edges), "cut_transformers": n_cut_trafo,
        "small_side_countries": {k: int(c) for k, c in countries.head(8).items()},
        "seconds": round(time.time() - t0, 1),
    })
    partitions[xT] = small.astype(int)
    if xT == REF:
        ref = (net, info, L, v, S, cut_edges, lam2, h, lo, hi)
    print(f"x_T={xT:<5} lambda2={lam2:.3e} h={h:.4f} [{lo:.3e},{hi:.3e}] small side {small.sum():5d} buses, "
          f"cut {len(cut_edges)} edges ({n_cut_trafo} transformers), {dict(countries.head(4))}, {rows[-1]['seconds']}s")

# agreement of the partitions with the reference x_T
for r in rows:
    r["ari_vs_ref"] = adjusted_rand_score(partitions[REF], partitions[r["trafo_x"]])
    r["vi_vs_ref"] = variation_of_information(partitions[REF], partitions[r["trafo_x"]])
(RESULTS / "continental_laplacian.json").write_text(json.dumps(rows, indent=1))

# --- figure ------------------------------------------------------------------------------------------------------------
# figure style: column width 7.2 in, no titles inside the plot, direct labels, the caption carries the
# description; the conductance is that of gridnet.spectral.conductance
net, info, L, v, S, cut_edges, lam2, h, lo, hi = ref
buses = info["buses"]
xy = buses[["x", "y"]].values
fig = plt.figure(figsize=(7.2, 3.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.55, wspace=0.28)
ax = fig.add_subplot(gs[:, 0])
segs = np.stack([xy[net.from_bus], xy[net.to_bus]], axis=1)
ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.25, rasterized=True))
vmax = np.percentile(np.abs(v), 99)
sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=1.2, lw=0, zorder=3, rasterized=True)
ax.add_collection(LineCollection(segs[cut_edges], colors="black", linewidths=1.6, zorder=4))
ax.set_xlim(-11, 35); ax.set_ylim(35, 60); ax.set_aspect(EUROPE_STRETCH); ax.axis("off")
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.6)
cb.set_label("Fiedler vector of the normalized problem", fontsize=8); cb.ax.tick_params(labelsize=7); cb.set_ticks([])
ax.text(4.0, 41.2, f"the cut,\n{len(cut_edges)} branches", fontsize=8, va="top")

xs = [r["trafo_x"] for r in rows]
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "0.45"
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(xs, [r["lambda2_norm"] for r in rows], "o-", color=BLUE, ms=3, lw=1.2)
ax1.plot(xs, [r["h_sweep"] for r in rows], "s-", color=ORANGE, ms=3, lw=1.2)
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_ylim(1e-5, 2e-4)
ax1.text(1.0, rows[-1]["lambda2_norm"] * 1.25, "$\\lambda_2$", color=BLUE, fontsize=8, ha="right", va="bottom")
ax1.text(1.0, rows[-1]["h_sweep"] * 1.25, "$h_\\Sigma$ of the cut", color=ORANGE, fontsize=8, ha="right", va="bottom")
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(xs, [r["ari_vs_ref"] for r in rows], "o-", color=GREEN, ms=3, lw=1.2)
ax2.plot(xs, np.array([r["small_side"] for r in rows]) / np.max([r["small_side"] for r in rows]), "^--", color=GRAY, ms=3, lw=1.0)
ax2.set_xscale("log"); ax2.set_ylim(0, 1.15)
ax2.text(0.02, 0.62, "adjusted Rand index against the cut at $x_T = 0.1$", color=GREEN, fontsize=7.5, va="center")
ax2.text(0.02, 0.32, "size of the smaller side, relative", color=GRAY, fontsize=7.5, va="center")
ax2.set_xlabel("transformer reactance $x_T$, per unit on the rating", fontsize=8)
for a_ in (ax1, ax2):
    a_.axvline(REF, color="0.6", ls=":", lw=0.8); a_.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.savefig(FIGDIR / "continental_fiedler.png", dpi=200, bbox_inches="tight")
print("wrote", FIGDIR / "continental_fiedler.png")
