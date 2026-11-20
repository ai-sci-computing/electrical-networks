"""Conversion from PyPSA networks and from the PyPSA-Eur OSM archive to `Network`.

Per-unit conventions (PyPSA 1.2, pypsa/network/power_flow.py, functions
`calculate_dependent_values` and `SubNetwork.calculate_Y`; the PyPSA paper, Brown, Hoersch &
Schlachtberger, JORS 6 (2018), describes the model but not this convention, so the source is
the code):

* PyPSA works on a 1 MVA base. Lines carry r, x in ohm and b, g in siemens for the whole
  length; per unit x_pu = x / v_nom^2 with v_nom the voltage of bus0 in kV, b_pu = b v_nom^2.
  A line given by a standard `type` gets r, x, b from n.line_types (per km, times the length,
  over the number of parallel circuits) only in `apply_line_types`, likewise transformer
  types in `apply_transformer_types`.
* Transformers carry r, x per unit on their own rating s_nom (MVA); PyPSA converts with
  x_pu = x / s_nom, b_pu = b s_nom. The tap ratio and phase shift are separate attributes.
  PyPSA's default transformer `model` is "t", half the series impedance on either side of
  the shunt; `apply_transformer_t_model` converts it to the pi-model by the wye-delta
  transform, which changes r_pu, x_pu, g_pu and b_pu of every t-model transformer with a
  nonzero shunt.
* PyPSA's bus admittance (`calculate_Y`) is then MATPOWER's branch model of eq. (3.2) in the
  MATPOWER manual, y_se = 1/(r_pu + j x_pu) with the shunt g_pu + j b_pu split to both ends,
  with the tap at bus0 (`tap_side = 0`, the default) or at bus1 (`tap_side = 1`). PyPSA's
  `x_pu_eff = x_pu * tap_ratio` is only used in its linear power flow and equals MATPOWER's
  b = 1/(x tau) there.

`from_pypsa` therefore calls `n.calculate_dependent_values()` and converts the pi-model
per-unit columns *_pu, so types and the t-model are handled as PyPSA handles them.

gridnet uses MATPOWER's 100 MVA base, so x and r are multiplied by 100 and b divided by
100. Consequently a transformer of rating S with reactance x_T on its own base gets
x = x_T * 100 / S on the system base: **its coupling weight 1/x is proportional to its
rating** and inversely proportional to the one per-unit number x_T. In the PyPSA-Eur
base network x_T is a uniform default (0.1) while S is data, so the relative weights of
the transformers are data and only their common scale is a modeling choice.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from gridnet.network import Network

BASE_MVA = 100.0


# Distances on the map, for bus coordinates (longitude, latitude) in degrees. The mean radius of the Earth,
# R_1 = (2a + b)/3 = 6371.0087714 km of the Geodetic Reference System 1980 (Moritz 1992, Bulletin Geodesique
# 66, 187-192, the table of derived geometric constants), rounded to the kilometer as PyPSA's own haversine
# does (pypsa.geo.haversine_pts, 6371.000 km); KM_PER_DEGREE is the length of one degree along the equator of
# the same ellipsoid, 2 pi a / 360 with a = 6378.137 km, 111.3195 km, used for both coordinates in `planar_km`
# (a degree of latitude is 110.57 to 111.69 km, so the error is below one percent; a common factor of both
# coordinates changes no k-means partition and no ratio of scores).
EARTH_RADIUS_KM = 6371.0
KM_PER_DEGREE = 111.32


def great_circle_km(a, b):
    """Great-circle distance in km between points a and b given as (longitude, latitude) in degrees, or
    arrays of such rows, by the haversine formula on the sphere of radius EARTH_RADIUS_KM, the formula and
    radius of PyPSA's `pypsa.geo.haversine_pts`, which gives the line lengths of PyPSA-Eur's networks
    (checked against closed forms in tests/test_pypsa_io.py)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo1, la1, lo2, la2 = (np.radians(v) for v in (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def nearest_great_circle(points, candidates):
    """For each point, the index of the nearest candidate by great-circle distance and that
    distance in km; points and candidates are rows (longitude, latitude) in degrees. On the sphere
    the chord between two points grows with their great-circle distance, so the nearest point by
    the chord, which a k-d tree over the unit vectors finds exactly, is the nearest by great circle,
    whose length is then 2 R arcsin(chord / 2R)."""
    from scipy.spatial import cKDTree

    def unit(xy):
        lon, lat = np.radians(np.asarray(xy, dtype=float)).T
        return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    chord, j = cKDTree(unit(candidates)).query(unit(points))
    return j, 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0.0, 1.0))


def planar_km(xy):
    """Rows (longitude, latitude) in degrees mapped to plane coordinates in km by the equirectangular
    projection at the mean latitude of the rows: x = KM_PER_DEGREE cos(mean latitude) longitude,
    y = KM_PER_DEGREE latitude. For distances within a region the size of Europe, as k-means needs them."""
    xy = np.asarray(xy, dtype=float)
    lat0 = np.deg2rad(xy[:, 1].mean())
    return np.column_stack([xy[:, 0] * KM_PER_DEGREE * np.cos(lat0), xy[:, 1] * KM_PER_DEGREE])


def from_pypsa(n, base_mva: float = BASE_MVA) -> Network:
    """Convert a `pypsa.Network` (AC lines and transformers) to a gridnet `Network`.

    Buses keep PyPSA's order; `bus_id` is 0..n-1 (PyPSA names are strings). Lines come
    first, then transformers, each with `status` 1 unless PyPSA marks them inactive.
    Loads, generators and shunts are not converted; injections are supplied separately.
    The first bus is marked as the angle reference.

    Calls `n.calculate_dependent_values()` first, as PyPSA's own power flows do, which fills
    in the per-unit columns of n (and applies line and transformer types and the t-model).
    The result has the bus admittance of PyPSA's `calculate_Y`, rescaled from 1 MVA to
    `base_mva` (Y / base_mva). Two cases need a translation:

    * A transformer with `tap_side = 1` has its tap at bus1; MATPOWER's branch has it at the
      from end only, so the branch is reversed, from bus1 to bus0, with its phase shift
      negated (compare PyPSA's Y_10 = -y_se/(tau e^{j shift}) with MATPOWER's
      Y_ft = -y_s/(tau' e^{-j shift'}) of eq. (3.2)). Its flows are then oriented bus1 -> bus0.
    * A branch shunt conductance g_pu has no place in MATPOWER's branch model or in `Network`
      and raises ValueError; this covers a line with g != 0 and a t-model transformer with
      resistance and a shunt, whose pi-equivalent has a conductance.

    A branch with r = x = 0 raises ValueError as well. The rating `rate_a` is PyPSA's `s_nom`;
    its `s_max_pu`, the share of `s_nom` an optimization may use, is not applied. The
    conventions are those of PyPSA 1.2 (pinned in pyproject.toml).
    """
    n.calculate_dependent_values()
    buses = n.buses
    idx = {name: i for i, name in enumerate(buses.index)}
    ln, tr = n.lines, n.transformers
    r_pu = np.concatenate([ln.r_pu.values, tr.r_pu.values])        # pi-model, 1 MVA base
    x_pu = np.concatenate([ln.x_pu.values, tr.x_pu.values])
    g_pu = np.concatenate([ln.g_pu.values, tr.g_pu.values])
    b_pu = np.concatenate([ln.b_pu.values, tr.b_pu.values])
    zero = (r_pu == 0) & (x_pu == 0)
    if np.any(zero):                                            # no series impedance: dc_model would divide by x = 0
        names = list(ln.index) + list(tr.index)
        raise ValueError(f"branches without series impedance (r = x = 0) are not representable: {[names[k] for k in np.flatnonzero(zero)][:5]}")
    # the wye-delta transform of a t-model with r = 0 and a purely susceptive shunt leaves a conductance of
    # rounding size; anything above 1e-12 of the series admittance is a conductance
    y_se = np.abs(1 / (r_pu + 1j * x_pu))
    if np.any(np.abs(g_pu) > 1e-12 * y_se):
        names = list(ln.index) + list(tr.index)
        bad = [names[k] for k in np.flatnonzero(np.abs(g_pu) > 1e-12 * y_se)]
        raise ValueError(f"branch shunt conductance g != 0 is not representable (MATPOWER's branch has none): {bad[:5]}")
    f = np.array([idx[b] for b in ln.bus0] + [idx[b] for b in tr.bus0], dtype=np.int64)
    t = np.array([idx[b] for b in ln.bus1] + [idx[b] for b in tr.bus1], dtype=np.int64)
    r, x, b = r_pu * base_mva, x_pu * base_mva, b_pu / base_mva
    tap = np.concatenate([np.ones(len(ln)), tr.get("tap_ratio", pd.Series(1.0, tr.index)).fillna(1.0).values])
    shift = np.concatenate([np.zeros(len(ln)), tr.get("phase_shift", pd.Series(0.0, tr.index)).fillna(0.0).values])
    # tap on bus1: reverse the branch so that the tap is at its from end, and negate the shift
    side1 = np.concatenate([np.zeros(len(ln), dtype=bool), tr.get("tap_side", pd.Series(0, tr.index)).values == 1])
    f[side1], t[side1] = t[side1], f[side1]
    shift[side1] = -shift[side1]
    active = np.concatenate([ln.get("active", pd.Series(True, ln.index)).values, tr.get("active", pd.Series(True, tr.index)).values])
    rate = np.concatenate([ln.s_nom.values, tr.s_nom.values]) / base_mva
    nb = len(buses)
    bus_type = np.ones(nb, dtype=np.int64)
    bus_type[0] = 3
    return Network(
        base_mva=base_mva, bus_id=np.arange(nb), bus_type=bus_type,
        pd=np.zeros(nb), qd=np.zeros(nb), gs=np.zeros(nb), bs=np.zeros(nb),
        from_bus=f, to_bus=t, r=r, x=x, b=b, tap=tap, shift=shift,
        status=active.astype(np.int64), rate_a=rate,
    )


def read_osm_archive(directory: str | Path, component: int = 0, trafo_x: float = 0.1,
                     base_mva: float = BASE_MVA) -> tuple[Network, dict]:
    """Load one synchronous component of the PyPSA-Eur OSM network archive as a `Network`.

    Reads buses.csv, lines.csv, transformers.csv (geometry strings are single-quoted),
    keeps AC buses, builds the graph of lines and transformers, and selects the
    `component`-th largest connected component (0 = continental Europe in archive 0.7).
    Lines: r, x [ohm] and b [S] from the archive (already divided by / multiplied with
    the number of circuits), converted as PyPSA does. Transformers: reactance `trafo_x`
    per unit on the transformer's own rating (PyPSA-Eur's config default is 0.1), r = 0,
    no tap or shift, rating from the archive. Line ratings from the archive.

    Returns the network and an info dict with the bus table of the component (id,
    voltage, country, x, y), the line and transformer counts, a boolean mask
    `is_transformer` over branches, and `trafo_s_nom`.
    """
    d = Path(directory)
    rd = lambda name: pd.read_csv(d / f"{name}.csv", quotechar="'")
    buses, lines, trafos = rd("buses"), rd("lines"), rd("transformers")
    buses = buses[buses.dc.astype(str).isin(["f", "False"])].reset_index(drop=True)

    G = nx.Graph()
    G.add_nodes_from(buses.bus_id)
    G.add_edges_from(zip(lines.bus0, lines.bus1))
    G.add_edges_from(zip(trafos.bus0, trafos.bus1))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    keep = comps[component]
    buses = buses[buses.bus_id.isin(keep)].reset_index(drop=True)
    lines = lines[lines.bus0.isin(keep) & lines.bus1.isin(keep)].reset_index(drop=True)
    trafos = trafos[trafos.bus0.isin(keep) & trafos.bus1.isin(keep)].reset_index(drop=True)
    idx = {b: i for i, b in enumerate(buses.bus_id)}

    z_base = lines.voltage.astype(float).values**2 / base_mva
    nl, nt = len(lines), len(trafos)
    f = np.array([idx[b] for b in lines.bus0] + [idx[b] for b in trafos.bus0], dtype=np.int64)
    t = np.array([idx[b] for b in lines.bus1] + [idx[b] for b in trafos.bus1], dtype=np.int64)
    x = np.concatenate([lines.x.values / z_base, trafo_x * base_mva / trafos.s_nom.values])
    r = np.concatenate([lines.r.values / z_base, np.zeros(nt)])
    b = np.concatenate([lines.b.values * z_base, np.zeros(nt)])
    rate = np.concatenate([lines.s_nom.values, trafos.s_nom.values]) / base_mva
    nb = len(buses)
    bus_type = np.ones(nb, dtype=np.int64)
    bus_type[0] = 3
    net = Network(
        base_mva=base_mva, bus_id=np.arange(nb), bus_type=bus_type,
        pd=np.zeros(nb), qd=np.zeros(nb), gs=np.zeros(nb), bs=np.zeros(nb),
        from_bus=f, to_bus=t, r=r, x=x, b=b, tap=np.ones(nl + nt), shift=np.zeros(nl + nt),
        status=np.ones(nl + nt, dtype=np.int64), rate_a=rate,
    )
    info = {
        "buses": buses[["bus_id", "voltage", "country", "x", "y"]],
        "n_lines": nl, "n_transformers": nt,
        "is_transformer": np.arange(nl + nt) >= nl,
        "trafo_s_nom": trafos.s_nom.values,
        "line_voltage": lines.voltage.values,
        "line_circuits": lines.circuits.values.astype(int),
        "component_sizes": [len(c) for c in comps],
    }
    return net, info
