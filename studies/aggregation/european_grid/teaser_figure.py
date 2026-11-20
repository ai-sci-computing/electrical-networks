"""Opening image: the cuts of the European grid.

The alternating-current lines of the continental synchronous area along their mapped geometries, colored by the part
they belong to in the first eight parts of the cut hierarchy of continental_partitions.py (repeated two-way spectral
cuts, the weakest first; gridnet.spectral.recursive_bipartition after Shi & Malik 2000, Sec. 3.2), over Natural Earth
country polygons. Lines whose two ends lie in different parts, the severed ones, are drawn black and thick. The other
synchronous areas are drawn in light gray. Map conventions as in studies/foundations/setup/europe_map_figure.py. Writes
figures/europe_cuts_teaser.png.

Run: .venv/bin/python studies/aggregation/european_grid/teaser_figure.py   (a few minutes, seven dense eigenproblems)
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from shapely import wkt

from gridnet.spectral import recursive_bipartition

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
_log = open(HERE / "results" / "teaser_figure.log", "w")
_print = print
def print(*a, **k):                                       # every printed number also goes to results/teaser_figure.log
    _print(*a, **k); _print(*a, **k, file=_log); _log.flush()
(HERE / "figures").mkdir(exist_ok=True)
ROOT = HERE.resolve().parents[2]
DATA = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
NE = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.shp"
LON, LAT, STRETCH = (-11.5, 41.5), (34.5, 60.5), EUROPE_STRETCH
K = 8
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

z = np.load(HERE / "results" / "kron128.npz")
i, j, b, nb = z["base_i"], z["base_j"], z["base_b"], int(z["n_base"])
A = coo_matrix((np.r_[b, b], (np.r_[i, j], np.r_[j, i])), shape=(nb, nb)).toarray()
L = np.diag(A.sum(1)) - A
history = []
part = recursive_bipartition(L, K, history=history)
for hrow in history:
    print(f"split of part {hrow['parent']} into {hrow['parent']} and {hrow['new']}, conductance {hrow['conductance']:.2e}")
print("sizes", np.bincount(part))
assert np.bincount(part).min() > 0 and 1223 in np.bincount(part)        # the Iberian side of the first sweep cut is one of the parts

bus_part = dict(zip(z["bus_id"].astype(str), part))
lines = pd.read_csv(DATA / "lines.csv", quotechar="'")
p0, p1 = lines.bus0.map(bus_part), lines.bus1.map(bus_part)


def segments(df):
    out = []
    for g in df.geometry:
        geom = wkt.loads(g.strip("'"))
        out += [np.asarray(p.coords) for p in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])]
    return out


fig, ax = plt.subplots(figsize=(7.2, 4.9))
gpd.read_file(NE).cx[LON[0]:LON[1], LAT[0]:LAT[1]].plot(ax=ax, facecolor="#efece6", edgecolor="white", linewidth=0.5, zorder=0)
ax.add_collection(LineCollection(segments(lines[p0.isna() | p1.isna()]), colors="0.75", linewidths=0.4, zorder=1))
for m in range(K):
    ax.add_collection(LineCollection(segments(lines[(p0 == m) & (p1 == m)]), colors=COLORS[m], linewidths=0.65, zorder=2, capstyle="round"))
severed = lines[p0.notna() & p1.notna() & (p0 != p1)]
ax.add_collection(LineCollection(segments(severed), colors="black", linewidths=2.2, zorder=3, capstyle="round"))
# a transformer is a few hundred meters long on the map, so one in a cut is drawn as a black square at its midpoint
trafos = pd.read_csv(DATA / "transformers.csv", quotechar="'")
q0, q1 = trafos.bus0.map(bus_part), trafos.bus1.map(bus_part)
cut_trafos = trafos[q0.notna() & q1.notna() & (q0 != q1)]
n_trafo_cut = len(cut_trafos)
for g in cut_trafos.geometry:
    c = np.asarray(wkt.loads(g.strip("'")).centroid.coords)[0]
    ax.scatter(*c, s=28, marker="s", color="black", zorder=4)
ax.set_xlim(*LON); ax.set_ylim(*LAT); ax.set_aspect(STRETCH); ax.axis("off")
fig.savefig(HERE / "figures" / "europe_cuts_teaser.png", dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print(f"wrote figures/europe_cuts_teaser.png, {len(severed)} severed lines, {n_trafo_cut} severed transformers")
