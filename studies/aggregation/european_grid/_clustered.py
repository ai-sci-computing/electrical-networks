"""PyPSA-Eur's 128-cluster network and the continental base network on the same clusters.

Shared by kron_vs_cluster.py and sparse_reductions.py (whose cache results/kron128.npz is built from
it): the clustered network's lines on the 100 MVA base of gridnet.pypsa_io, the cluster of every base
bus (busmap of `simplify_network` followed by that of `cluster_network`; Hoersch et al. 2018), the
largest connected component of the clustered lines among the clusters present in the base network's
main component, the two Laplacians, and the representative bus of every cluster
(gridnet.kron.pick_representatives).
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from gridnet.graph import laplacian
from gridnet.kron import pick_representatives
from gridnet.pypsa_io import great_circle_km, read_osm_archive

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data" / "raw" / "pypsa-eur-osm-0.7"
PYPSA_EUR = ROOT / "data" / "raw" / "pypsa-eur-gridnet"
BASE_MVA = 100.0


def read_clustered(pypsa_eur: Path = PYPSA_EUR) -> dict:
    """The clustered network and the bus maps. Keys: `nc` (the pypsa.Network), `lines` (bus0, bus1,
    susceptance b on BASE_MVA, s_nom, length), `centroids` (cluster name -> (x, y)), `clus` (simplified
    bus -> cluster), `base_to_cluster` (base bus -> cluster)."""
    import pypsa

    nc = pypsa.Network(str(pypsa_eur / "networks" / "clustered.nc"))
    nc.calculate_dependent_values()                       # x from line type, length, num_parallel
    x = nc.lines.x_pu.values * BASE_MVA                   # PyPSA's per unit is on 1 MVA
    lines = pd.DataFrame({"bus0": nc.lines.bus0.values, "bus1": nc.lines.bus1.values, "b": 1 / x,
                          "s_nom": nc.lines.s_nom.values, "length": nc.lines.length.values})
    assert np.all(np.isfinite(lines.b)) and np.all(lines.b > 0)
    simp = pd.read_csv(pypsa_eur / "busmap_simplify_network.csv", index_col=0)["busmap"]
    clus = pd.read_csv(pypsa_eur / "busmap_cluster_network.csv", index_col=0)["busmap"]
    return {"nc": nc, "lines": lines, "centroids": {k: (float(r.x), float(r.y)) for k, r in nc.buses.iterrows()},
                "simp": simp, "clus": clus, "base_to_cluster": simp.map(clus)}


def on_clusters(clustered: dict, trafo_x: float, archive: Path = ARCHIVE) -> dict:
    """The base network (main component of the archive, transformers at reactance `trafo_x`) and the
    clustered lines among the clusters of the largest connected component. Keys: `net`, `buses`,
    `labels` (cluster name per base bus, nan without one), `clusters` (sorted names), `cidx` (name ->
    index), `lines` (the clustered lines inside the component), `Lc` (their Laplacian, parallel lines
    added), `Bc` (its couplings, zero diagonal), `edges` and `L` (the base network's branches and
    Laplacian, susceptance 1/x)."""
    net, info = read_osm_archive(archive, component=0, trafo_x=trafo_x)
    buses = info["buses"].reset_index(drop=True)
    labels = buses.bus_id.map(clustered["base_to_cluster"]).values
    present = sorted({str(l) for l in labels if isinstance(l, str)})
    lines = clustered["lines"]
    sub = lines[lines.bus0.isin(present) & lines.bus1.isin(present)]
    G = nx.Graph(); G.add_nodes_from(present); G.add_edges_from(zip(sub.bus0, sub.bus1))
    clusters = sorted(max(nx.connected_components(G), key=len))
    cidx = {c: i for i, c in enumerate(clusters)}
    sub = sub[sub.bus0.isin(clusters) & sub.bus1.isin(clusters)]
    Lc = laplacian(len(clusters), [(cidx[a], cidx[b]) for a, b in zip(sub.bus0, sub.bus1)], sub.b.values)
    Bc = -Lc.copy(); np.fill_diagonal(Bc, 0.0)
    edges = list(zip(net.from_bus.tolist(), net.to_bus.tolist()))
    L = laplacian(len(buses), edges, 1 / net.x)
    return {"net": net, "buses": buses, "labels": labels, "clusters": clusters, "cidx": cidx, "lines": sub, "Lc": Lc, "Bc": Bc,
                "edges": edges, "L": L}


def kept_buses(clustered: dict, setup: dict, mode: str) -> np.ndarray:
    """The representative base bus of every cluster, in the order of `setup['clusters']`: the bus nearest
    to the cluster's centroid ('nearest') or the one of largest weighted degree ('degree'). Nearest is
    measured as the great-circle distance in km (gridnet.pypsa_io.great_circle_km, the formula of PyPSA's
    line lengths); until 2026-11-07 it was the Euclidean distance in degrees of longitude and latitude,
    which counts an east-west separation about 1.5 times too long at Europe's latitudes."""
    centroids = clustered["centroids"]
    xy, labels = setup["buses"][["x", "y"]].values, np.asarray(setup["labels"], dtype=object)
    if mode == "nearest":
        reps = {}
        for c in setup["clusters"]:
            members = np.flatnonzero(labels == c)
            d = great_circle_km(xy[members], np.asarray(centroids[c], dtype=float)[None, :])
            reps[c] = int(members[np.argmin(d)])
    else:
        reps = pick_representatives(xy, labels, {c: centroids[c] for c in setup["clusters"]}, mode=mode, degrees=np.diag(setup["L"]))
    return np.array([reps[c] for c in setup["clusters"]])
