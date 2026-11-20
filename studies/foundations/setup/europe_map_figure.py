"""Map of the European transmission network.

Draws the alternating-current lines of the PyPSA-Eur OSM network archive (version 0.7, the
OpenStreetMap-based input of the `base_network` rule, https://data.pypsa.org/workflows/eur/osm/0.7/,
fetched into data/raw/pypsa-eur-osm-0.7/ by `scripts/fetch_data.py --only osm`) along their mapped
geometries, colored by nominal voltage, plus the direct-current links dashed, over Natural Earth
1:50m country polygons (public domain, release v5.1.1, fetched into data/raw/naturalearth/ by
`scripts/fetch_data.py --only naturalearth`). Equirectangular plot with the latitude axis stretched
by 1.25, less than the 1/cos(52 deg) that keeps shapes true, so that the map is wider than tall.
Writes figures/europe_map.png.
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
from matplotlib.lines import Line2D
from shapely import wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the studies folder, for the shared map convention
from shared.maps import EUROPE_STRETCH

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
NE_DIR = ROOT / "data" / "raw" / "naturalearth"
OUT = Path(__file__).parent / "figures" / "europe_map.png"
OUT.parent.mkdir(exist_ok=True)

LON, LAT = (-11.5, 41.5), (34.5, 71.0)
STRETCH = EUROPE_STRETCH  # the one value of all maps of the European grid (studies/shared/maps.py)

# voltage classes of the archive, light to dark, thin to thick
CLASSES = [
    ("220 and 225 kV", (200, 240), "#8fb4d8", 0.5),
    ("275 to 330 kV", (250, 340), "#5aa0c8", 0.6),
    ("380 to 420 kV", (370, 430), "#0b4f8a", 0.8),
    ("500 to 750 kV", (490, 760), "#041f3d", 1.3),
]
DC = "#d9541e"


def countries() -> gpd.GeoDataFrame:
    return gpd.read_file(NE_DIR / "ne_50m_admin_0_countries.shp")


def segments(df: pd.DataFrame) -> list[np.ndarray]:
    """Line geometries as arrays of (lon, lat) vertices, multilinestrings split into parts."""
    out = []
    for g in df.geometry:
        geom = wkt.loads(g.strip("'"))
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        out += [np.asarray(p.coords) for p in parts]
    return out


rd = lambda name: pd.read_csv(DATA / f"{name}.csv", quotechar="'")
lines, links = rd("lines"), rd("links")

fig, ax = plt.subplots(figsize=(7.2, 5.4))
land = countries().cx[LON[0]:LON[1], LAT[0]:LAT[1]]
land.plot(ax=ax, facecolor="#efece6", edgecolor="white", linewidth=0.5, zorder=0)

for _, (lo, hi), color, lw in CLASSES:
    sel = lines[(lines.voltage >= lo) & (lines.voltage <= hi)]
    ax.add_collection(LineCollection(segments(sel), colors=color, linewidths=lw, zorder=2, capstyle="round"))
ax.add_collection(LineCollection(segments(links), colors=DC, linewidths=0.9, linestyles=(0, (3, 2)), zorder=3))

handles = [Line2D([], [], color=c, lw=max(lw * 2, 1.2), label=lab) for lab, _, c, lw in CLASSES]
handles.append(Line2D([], [], color=DC, lw=1.4, ls=(0, (3, 2)), label="direct-current links"))
ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, handlelength=2.2)

ax.set_xlim(*LON); ax.set_ylim(*LAT)
ax.set_aspect(STRETCH)
ax.axis("off")
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("wrote", OUT, f"({len(lines)} AC lines, {len(links)} DC links)")
