"""Tests for gridnet.pypsa_io: PyPSA network -> gridnet Network, and the OSM archive loader.

Per-unit convention of PyPSA (Brown, Hoersch & Schlachtberger 2018; implemented in
pypsa.network.power_flow.calculate_dependent_values): base power 1 MVA;
lines x_pu = x[Ohm] / v_nom[kV]^2, b_pu = b[S] v_nom^2; transformers x_pu = x / s_nom
(x given per unit on the transformer's own rating), effective x_pu_eff = x_pu tap_ratio.
gridnet uses a 100 MVA base (MATPOWER convention), so x is multiplied by 100 and b
divided by 100. Closed forms below; then PyPSA's own x_pu and its linear power flow as
the reference.
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.admittance import dc_model
from gridnet.graph import laplacian

pypsa = pytest.importorskip("pypsa")
from gridnet.pypsa_io import from_pypsa, read_osm_archive

ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "raw" / "pypsa-eur-osm-0.7"


def small_pypsa_network():
    """Triangle of two 220 kV buses and one 380 kV bus, two lines and one transformer."""
    n = pypsa.Network()
    n.add("Bus", "a", v_nom=220.0)
    n.add("Bus", "b", v_nom=220.0)
    n.add("Bus", "c", v_nom=380.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=48.4, r=4.84, b=2e-4, s_nom=500.0)     # Ohm, S
    n.add("Line", "ab2", bus0="a", bus1="b", x=96.8, r=0.0, b=0.0, s_nom=500.0)
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.01, s_nom=1000.0, tap_ratio=1.05, phase_shift=3.0)
    return n


def test_line_and_transformer_per_unit_closed_forms():
    net = from_pypsa(small_pypsa_network())
    assert net.base_mva == 100.0
    # line ab: Z_base = 220^2 / 100 = 484 Ohm -> x = 48.4/484 = 0.1, r = 0.01; b = 2e-4 * 484 = 0.0968
    assert np.isclose(net.x[0], 0.1) and np.isclose(net.r[0], 0.01) and np.isclose(net.b[0], 2e-4 * 484)
    # transformer: x = 0.1 on 1000 MVA -> 0.1 * 100/1000 = 0.01 on 100 MVA; tap and shift carried over
    assert np.isclose(net.x[2], 0.01) and np.isclose(net.r[2], 0.001)
    assert np.isclose(net.tap[2], 1.05) and np.isclose(net.shift[2], 3.0)
    assert np.all(net.tap[:2] == 1.0) and np.all(net.shift[:2] == 0.0)
    np.testing.assert_array_equal(net.status, 1)
    assert list(net.edges) == [(0, 1), (0, 1), (1, 2)]


def test_matches_pypsa_own_per_unit_values():
    """The per-unit values against PyPSA's own (`x_pu`, `b_pu` on 1 MVA, which `from_pypsa` reads and
    scales by the base of 100 MVA), and against the raw attributes without PyPSA's computation: a line's
    x in ohm over the base impedance v_nom^2 / 100 and its b in siemens times it, a transformer's x in
    per unit of its own s_nom times 100 / s_nom."""
    n = small_pypsa_network()
    raw = n.lines[["x", "r", "b"]].values.copy(), n.buses.v_nom[n.lines.bus0].values.copy(), n.transformers[["x", "r", "s_nom"]].values.copy()
    n.calculate_dependent_values()
    net = from_pypsa(n)
    (xrb, v_nom, tr) = raw
    z_base = v_nom ** 2 / 100.0
    np.testing.assert_allclose(net.x[:2], xrb[:, 0] / z_base, rtol=1e-12)
    np.testing.assert_allclose(net.r[:2], xrb[:, 1] / z_base, rtol=1e-12)
    np.testing.assert_allclose(net.b[:2], xrb[:, 2] * z_base, rtol=1e-12)
    np.testing.assert_allclose([net.x[2], net.r[2]], tr[0, :2] * 100.0 / tr[0, 2], rtol=1e-12)
    np.testing.assert_allclose(net.x[:2], n.lines.x_pu.values * 100, rtol=1e-12)
    np.testing.assert_allclose(net.b[:2], n.lines.b_pu.values / 100, rtol=1e-12)
    np.testing.assert_allclose(net.x[2], n.transformers.x_pu.values[0] * 100, rtol=1e-12)


@pytest.mark.data("osm")
def test_read_osm_archive_continental_component():
    net, info = read_osm_archive(ARCHIVE, component=0, trafo_x=0.1)
    assert net.n_bus == 5577
    assert info["is_transformer"].sum() == info["n_transformers"]
    assert info["n_lines"] + info["n_transformers"] == net.n_branch
    assert info["component_sizes"][0] == 5577 and np.all(net.x > 0)
    # the component is connected, and it is the largest of the archive
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(net.n_branch), (net.from_bus, net.to_bus)), shape=(net.n_bus, net.n_bus))
    assert connected_components(A, directed=False)[0] == 1
    # the first record of lines.csv, worked by hand: 225 kV, r = 0.35581 and x = 1.784981 Ohm,
    # b = 2.329e-05 S, s_nom = 502.728 MVA; per unit on 100 MVA with z_base = 225^2 / 100 = 506.25 Ohm
    buses = info["buses"].bus_id.values
    assert (buses[net.from_bus[0]], buses[net.to_bus[0]]) == ("way/150329778-225", "way/239609000-225")
    np.testing.assert_allclose([net.r[0], net.x[0], net.b[0], net.rate_a[0]],
                               [0.35581 / 506.25, 1.784981 / 506.25, 2.329e-05 * 506.25, 5.02728], rtol=1e-12)
    L = laplacian(net.n_bus, net.edges, 1 / net.x)
    Bbus, _, _, _ = dc_model(net)
    np.testing.assert_allclose(Bbus, L, atol=1e-10)


@pytest.mark.data("osm")
def test_read_osm_archive_transformer_weight_scales_with_rating_over_x():
    net1, info1 = read_osm_archive(ARCHIVE, component=0, trafo_x=0.1)
    net2, _ = read_osm_archive(ARCHIVE, component=0, trafo_x=0.2)
    t = info1["is_transformer"]
    np.testing.assert_allclose(net2.x[t], 2 * net1.x[t])          # transformer x doubles
    np.testing.assert_allclose(net2.x[~t], net1.x[~t])            # lines untouched
    # x_pu(100 MVA) = trafo_x * 100 / s_nom
    np.testing.assert_allclose(net1.x[t], 0.1 * 100 / info1["trafo_s_nom"])


# --- the bus admittance against PyPSA's own calculate_Y -------------------------------------------------
# PyPSA 1.2.4, pypsa/network/power_flow.py, SubNetwork.calculate_Y: y_se = 1/(r_pu + j x_pu), y_sh = g_pu +
# j b_pu, the tap on bus0 (tap_side 0) or on bus1 (tap_side 1), on a 1 MVA base. On gridnet's 100 MVA base
# the same matrix is Y / 100. PyPSA orders the buses of a sub-network slack first (buses_o), so the
# comparison permutes gridnet's rows and columns into that order.


def _pypsa_admittance(n):
    """PyPSA's bus admittance of the only sub-network of n, rescaled to 100 MVA, and gridnet's in the same order."""
    from gridnet.admittance import bus_admittance

    Y = bus_admittance(from_pypsa(n))
    n.determine_network_topology()
    sub = n.sub_networks.obj.iloc[0]
    sub.calculate_Y()
    order = [list(n.buses.index).index(b) for b in sub.buses_o]
    return sub.Y.toarray() / 100.0, Y[np.ix_(order, order)]


def _two_voltage_levels():
    n = pypsa.Network()
    n.add("Bus", "a", v_nom=220.0)
    n.add("Bus", "b", v_nom=220.0)
    n.add("Bus", "c", v_nom=380.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=48.4, r=4.84, b=2e-4, s_nom=500.0)
    return n


def test_admittance_matches_pypsa_for_lines_and_a_pi_model_transformer():
    n = _two_voltage_levels()
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.01, b=0.002, s_nom=1000.0, tap_ratio=1.05,
          phase_shift=3.0, model="pi")
    ref, Y = _pypsa_admittance(n)
    np.testing.assert_allclose(Y, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_admittance_matches_pypsa_for_the_tap_on_bus1():
    """tap_side = 1 puts the off-nominal ratio at bus1: Y_11 = (y_se + y_sh/2)/tau^2, Y_00 = y_se + y_sh/2,
    Y_01 = -y_se/(tau e^{-j shift}), Y_10 = -y_se/(tau e^{j shift}). MATPOWER's branch has its tap at the
    from end only (manual eq. (3.2)), so the branch is reversed, bus1 -> bus0, with the shift negated."""
    n = _two_voltage_levels()
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.01, s_nom=1000.0, tap_ratio=1.05,
          phase_shift=3.0, tap_side=1)
    ref, Y = _pypsa_admittance(n)
    np.testing.assert_allclose(Y, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    net = from_pypsa(n)
    assert net.edges[1] == (2, 1) and net.shift[1] == -3.0 and net.tap[1] == 1.05


def test_admittance_matches_pypsa_for_a_t_model_transformer():
    """PyPSA's default transformer model is 't': half the series impedance on either side of the shunt. It
    converts it to the pi-model by the wye-delta transform (apply_transformer_t_model); with r = 0 and a
    purely susceptive shunt the result has no conductance and maps onto MATPOWER's branch."""
    n = _two_voltage_levels()
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.0, b=0.002, s_nom=1000.0, tap_ratio=1.05)
    assert n.transformers.model.iloc[0] == "t"
    ref, Y = _pypsa_admittance(n)
    np.testing.assert_allclose(Y, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_admittance_matches_pypsa_for_a_line_given_by_its_type():
    """A line with a standard type carries r, x, b per km in n.line_types; PyPSA fills in r, x, b only in
    calculate_dependent_values, so before that the line has x = 0."""
    n = pypsa.Network()
    n.add("Bus", "a", v_nom=380.0)
    n.add("Bus", "b", v_nom=380.0)
    n.add("Line", "ab", bus0="a", bus1="b", type="490-AL1/64-ST1A 380.0", length=100.0, s_nom=1000.0)
    ref, Y = _pypsa_admittance(n)
    np.testing.assert_allclose(Y, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    assert np.all(np.isfinite(Y)) and from_pypsa(n).x[0] > 0


def test_branch_conductance_is_refused():
    """MATPOWER's branch model, and `Network`, have no shunt conductance on a branch (manual eq. (3.2)), so
    a line with g != 0, or a t-model transformer with resistance and a shunt (whose pi-equivalent has a
    conductance), cannot be converted exactly."""
    n = _two_voltage_levels()
    n.lines.loc["ab", "g"] = 1e-5
    with pytest.raises(ValueError, match="conductance"):
        from_pypsa(n)
    n = _two_voltage_levels()
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.01, b=0.002, s_nom=1000.0)
    with pytest.raises(ValueError, match="conductance"):
        from_pypsa(n)


def test_nearest_great_circle_is_the_nearest_by_great_circle():
    """The nearest of a set of points by great-circle distance, and that distance. Against the
    brute-force minimum of great_circle_km over random points in and around Europe, and on a case
    in which the projection x = R lon cos(lat), each point at its own latitude, picks the wrong
    point: the plant at (40, 60), point A at (40, 60.9), 100 km north, point B at (42, 60), 111 km
    east. In that projection A lies sqrt((40 cos 60.9 - 40 cos 60)^2 + 0.9^2) = 1.05 degrees away
    and B 2 cos 60 = 1.00 degrees, so B would be taken."""
    from gridnet.pypsa_io import great_circle_km, nearest_great_circle
    j, d = nearest_great_circle(np.array([[40.0, 60.0]]), np.array([[42.0, 60.0], [40.0, 60.9]]))
    assert j[0] == 1 and d[0] == pytest.approx(great_circle_km((40.0, 60.0), (40.0, 60.9)), rel=1e-9)
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-25, 45, 400), rng.uniform(34, 72, 400)])
    cand = np.column_stack([rng.uniform(-25, 45, 900), rng.uniform(34, 72, 900)])
    j, d = nearest_great_circle(pts, cand)
    full = great_circle_km(pts[:, None, :], cand[None, :, :])
    np.testing.assert_array_equal(j, full.argmin(axis=1))
    np.testing.assert_allclose(d, full.min(axis=1), rtol=1e-9)


def test_great_circle_km_closed_forms():
    """On a sphere of radius R: a quarter meridian is pi R / 2, a quarter of the equator the same, the
    antipode pi R, a point to itself 0; rows of (longitude, latitude) arrays are handled one by one."""
    from gridnet.pypsa_io import EARTH_RADIUS_KM, great_circle_km

    R = EARTH_RADIUS_KM
    assert great_circle_km((0.0, 0.0), (0.0, 90.0)) == pytest.approx(np.pi * R / 2, rel=1e-12)
    assert great_circle_km((0.0, 0.0), (90.0, 0.0)) == pytest.approx(np.pi * R / 2, rel=1e-12)
    assert great_circle_km((10.0, 20.0), (-170.0, -20.0)) == pytest.approx(np.pi * R, rel=1e-12)
    a = np.array([[0.0, 0.0], [5.0, 45.0]]); b = np.array([[0.0, 90.0], [5.0, 45.0]])
    np.testing.assert_allclose(great_circle_km(a, b), [np.pi * R / 2, 0.0], atol=1e-9)


def test_planar_km_scales_longitude_by_the_cosine_of_the_mean_latitude():
    from gridnet.pypsa_io import KM_PER_DEGREE, planar_km

    xy = np.array([[1.0, 59.0], [3.0, 61.0]])                     # mean latitude 60 degrees
    np.testing.assert_allclose(planar_km(xy), [[0.5 * KM_PER_DEGREE, 59 * KM_PER_DEGREE], [1.5 * KM_PER_DEGREE, 61 * KM_PER_DEGREE]], rtol=1e-12)


def test_aggregatelines_stretches_every_line_to_the_centroid_distance_before_the_parallel_sum():
    """PyPSA's length convention as used in kron_vs_cluster.py's closing check (pypsa/clustering/spatial.py,
    `aggregatelines`, strategy 'reciprocal_voltage_weighted_average' for x): an aggregated line of length L
    (the great-circle distance of the two cluster buses) replacing lines of lengths l_k and reactances x_k
    has x = L / sum_k (l_k / x_k), so x / x_parallel = L sum_k (1/x_k) / sum_k (l_k / x_k)."""
    from pypsa.clustering.spatial import aggregatelines

    from gridnet.pypsa_io import great_circle_km

    n = pypsa.Network()
    for name, x, y in (("a1", 0.0, 50.0), ("a2", 0.2, 50.0), ("b1", 3.0, 50.0), ("b2", 3.2, 50.0)):
        n.add("Bus", name, x=x, y=y, v_nom=380.0)
    n.add("Line", "l1", bus0="a1", bus1="b1", x=10.0, r=0.0, length=150.0, s_nom=1000.0)
    n.add("Line", "l2", bus0="a2", bus1="b2", x=40.0, r=0.0, length=400.0, s_nom=1000.0)
    import pandas as pd

    busmap = pd.Series({"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
    static, _, _ = aggregatelines(n, busmap, with_time=False)
    L = great_circle_km((0.1, 50.0), (3.1, 50.0))                 # the cluster buses sit at the mean coordinates
    l, x = np.array([150.0, 400.0]), np.array([10.0, 40.0])
    assert float(static.length.iloc[0]) == pytest.approx(L, rel=1e-3)
    assert float(static.x.iloc[0]) == pytest.approx(L / np.sum(l / x), rel=1e-3)
    x_parallel = 1 / np.sum(1 / x)
    assert float(static.x.iloc[0]) / x_parallel == pytest.approx(L * np.sum(1 / x) / np.sum(l / x), rel=1e-3)


@pytest.mark.data("osm")
def test_archive_counts_of_aggregation_two_and_the_other_components():
    """The archive as described: 6,792 AC buses in 36 countries, 9,162 lines, 878
    transformers, six synchronous components of 5,577, 620, 503, 65, 18 and 9 buses; the continental one has
    8,340 branches and 31 countries. Every component reads as a connected network of its stated size."""
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rd = lambda name: pd.read_csv(ARCHIVE / f"{name}.csv", quotechar="'")
    buses = rd("buses")
    ac = buses[buses.dc.astype(str).isin(["f", "False"])]
    assert len(ac) == 6792 and ac.country.nunique() == 36
    assert len(rd("lines")) == 9162 and len(rd("transformers")) == 878
    net, info = read_osm_archive(ARCHIVE, component=0)
    assert info["component_sizes"] == [5577, 620, 503, 65, 18, 9]
    assert net.n_branch == 8340 and info["buses"].country.nunique() == 31
    for c, size in enumerate(info["component_sizes"]):
        n_c, _ = read_osm_archive(ARCHIVE, component=c)
        A = coo_matrix((np.ones(n_c.n_branch), (n_c.from_bus, n_c.to_bus)), shape=(n_c.n_bus, n_c.n_bus))
        assert n_c.n_bus == size and connected_components(A, directed=False)[0] == 1


def test_from_pypsa_marks_an_inactive_line_out_of_service():
    """`from_pypsa` keeps a line PyPSA marks active=False as a branch with status 0 (the active=False path)."""
    n = small_pypsa_network()
    name = n.lines.index[0]
    n.lines.loc[name, "active"] = False
    net = from_pypsa(n)
    assert net.status[0] == 0 and net.status[1:].all()


def test_a_branch_without_impedance_is_refused():
    """A line with r = x = 0 has no place in the linear model (dc_model divides by x); `from_pypsa` refuses
    it by name instead of returning a branch with x = 0."""
    n = pypsa.Network()
    n.add("Bus", "a", v_nom=220.0)
    n.add("Bus", "b", v_nom=220.0)
    n.add("Line", "short", bus0="a", bus1="b", x=0.0, r=0.0, s_nom=500.0)
    with pytest.raises(ValueError, match="short"):
        from_pypsa(n)
