"""Opening image: the exact Kron reduction onto 96 buses of the continental grid and one sparse reduction of it.

The 96 kept buses of the continental grid (the bus nearest to the center of each of PyPSA-Eur's clusters) on the map,
twice. Left, the exact Kron reduction onto them, a complete graph of 4,560 couplings (first the situation, then the
reduction). Right, one sparse reduction of it with a measured error:
the deterministic selection of Batson, Spielman & Srivastava 2012 (gridnet.sparsify.bss_sparsify) with d = 3, 189 edges,
spectral error 0.55. Edge width grows with the square root of the susceptance;
edges longer than 800 km, which no line of the map matches, are orange. Writes figures/sparse_teaser.png.

Run: .venv/bin/python studies/aggregation/european_grid/sparse_teaser_figure.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from gridnet.kron import laplacian_from_couplings
from gridnet.pypsa_io import great_circle_km
from gridnet.sparsify import bss_sparsify, spectral_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
ROOT = HERE.resolve().parents[2]
NE = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.shp"
LON, LAT, STRETCH = (-11.0, 41.5), (34.5, 60.5), EUROPE_STRETCH
INK, ORANGE, BLUE = "#2b2b2a", "#eb6834", "#2a78d6"

z = np.load(HERE / "results" / "kron128.npz")
xy = z["bus_xy"][z["keep"]]
L = laplacian_from_couplings(z["W"])
H = bss_sparsify(L, 3.0)                                 # deterministic, no seed
eps = spectral_error(L, H)
print(f"selection by barriers, d = 3: spectral error {eps:.2f}, {int(np.count_nonzero(np.triu(H, 1)))} edges")


Hw = -np.triu(H, 1)
SCALE = float(np.median(Hw[Hw > 0]))


land = gpd.read_file(NE).cx[LON[0]:LON[1], LAT[0]:LAT[1]]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
# left, the actual situation: the exact reduction, a complete graph on the 96 kept buses; right, its sparsification
for ax, W, name in zip(axes, (z["W"], -H), ("exact reduction", "selected reduction")):
    land.plot(ax=ax, facecolor="#efece6", edgecolor="white", linewidth=0.4, zorder=0)
    W = np.triu(W, 1)
    i, j = np.nonzero(W > 0)
    length = np.array([great_circle_km(xy[a], xy[b]) for a, b in zip(i, j)])
    # one scale of widths for both panels, so that equal susceptances look equal
    width = np.clip(0.75 * np.sqrt(W[i, j] / SCALE), 0.04, 2.2)     # no floor: the many tiny couplings of the complete graph stay hairlines
    far = length > 800
    for sel, color, zo in ((~far, INK, 1), (far, ORANGE if name == "selected reduction" else INK, 2)):
        lw = width[sel] * (1.6 if color == ORANGE and name == "selected reduction" else 1.0)
        ax.add_collection(LineCollection([xy[[a, b]] for a, b in zip(i[sel], j[sel])], colors=color, linewidths=lw,
                                         alpha=0.95 if (color == ORANGE and name == "selected reduction") else 0.6, zorder=zo, capstyle="round"))
    ax.scatter(*xy.T, s=9, color=BLUE, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xlim(*LON); ax.set_ylim(*LAT); ax.set_aspect(STRETCH); ax.axis("off")
    print(f"  {name}: {len(i)} edges, {int(far.sum())} longer than 800 km, longest {length.max():.0f} km")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
fig.savefig(HERE / "figures" / "sparse_teaser.png", dpi=200, bbox_inches="tight", pad_inches=0.03, facecolor="white")
