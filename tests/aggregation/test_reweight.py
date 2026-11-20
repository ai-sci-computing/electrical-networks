"""Tests for gridnet.reweight: the best weights on a prescribed edge set.

Measure: the spectral error of Spielman & Srivastava 2011, eq. (2). PTDF matching: Fortenbacher,
Demiray & Schaffner 2018 (arXiv:1711.01079), eq. (5) for the reduced PTDF (after Oh 2010), eq. (15)
for the PTDF of a network from its susceptances, eq. (16) for the fit, and Sec. II-B2 for the scale.

Closed form, derived here (not from a paper): the unit triangle approximated on the path 0-1-2 with
weights w1, w2. On the complement of the constants the triangle is L = 3 I, so L^{+/2} L~ L^{+/2} =
L~ / 3 there, and the nonzero eigenvalues of the weighted path are s +- sqrt(s^2 - 3 w1 w2) with
s = w1 + w2 (characteristic polynomial lam^2 - 2 s lam + 3 w1 w2). For fixed s the spread is smallest
at w1 = w2 = s/2, eigenvalues 3s/2 and s/2, and any other split is strictly worse for both
objectives, so both optima are symmetric and unique.
  spectral:  s/2 - 1 = -(s/6 - 1)  ->  s = 3, w = 3/2, eps = 1/2
  Frobenius: minimize 2 (s/3 - 1)^2 + s^2/18  ->  s = 12/5, w = 6/5
"""

import numpy as np
import pytest

from gridnet.graph import laplacian
from gridnet.reweight import (
    best_scales,
    ptdf_from_susceptances,
    reduced_ptdf,
    reweight_frobenius,
    reweight_ptdf,
    reweight_spectral,
    zonal_operator,
)
from gridnet.sparsify import spectral_error

pytest.importorskip("cvxpy")

TRIANGLE = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, 1.0])
PATH = [(0, 1), (1, 2)]


def test_spectral_reweighting_of_the_triangle_on_a_path():
    w, eps, bound = reweight_spectral(TRIANGLE, PATH)
    assert w == pytest.approx([1.5, 1.5], abs=1e-4) and eps == pytest.approx(0.5, abs=1e-4)
    assert bound <= eps + 1e-12 and bound == pytest.approx(0.5, abs=1e-4)      # the dual certificate closes the gap
    assert spectral_error(TRIANGLE, laplacian(3, PATH, w)) == pytest.approx(eps, abs=1e-4)


def test_frobenius_reweighting_of_the_triangle_on_a_path():
    assert reweight_frobenius(TRIANGLE, PATH) == pytest.approx([1.2, 1.2], abs=1e-8)


def test_reweighting_on_the_full_edge_set_returns_the_graph():
    rng = np.random.default_rng(0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    b = rng.uniform(0.5, 2.0, len(edges))
    L = laplacian(4, edges, b)
    w, eps, bound = reweight_spectral(L, edges)
    assert eps == pytest.approx(0.0, abs=1e-5) and w == pytest.approx(b, abs=1e-4) and bound <= eps + 1e-12
    assert reweight_frobenius(L, edges) == pytest.approx(b, abs=1e-8)


def test_reduced_ptdf_with_one_bus_per_zone_is_the_ptdf():
    """With every bus its own zone and every branch its own interconnection, eq. (5) returns the
    PTDF of the network, eq. (15) reproduces it from the susceptances, and the fit of eq. (16)
    returns the susceptances, the scale pinned at the strongest branch."""
    rng = np.random.default_rng(1)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    b = rng.uniform(0.5, 2.0, len(edges))
    H = ptdf_from_susceptances(4, edges, b)
    flow = H @ np.array([1.0, 0.0, -1.0, 0.0])                   # a unit transfer from bus 0 to bus 2
    assert flow[0] + flow[1] + flow[2] == pytest.approx(1.0)     # leaves bus 0 over its three branches
    Hr, zone_edges = reduced_ptdf(4, edges, b, zones=np.arange(4))
    assert zone_edges == edges and np.allclose(Hr, H)
    fit = reweight_ptdf(Hr, 4, zone_edges, reference=b * np.where(b == b.max(), 1.0, 0.5))   # right only at the pin
    assert fit == pytest.approx(b, rel=1e-5)


def test_reduced_ptdf_of_two_zones_carries_the_whole_transfer():
    """Path 0-1-2-3 with zones {0,1} and {2,3}: one interconnection, and a transfer from zone 0 to
    zone 1 flows over it entirely, whatever the susceptances."""
    edges = [(0, 1), (1, 2), (2, 3)]
    Hr, zone_edges = reduced_ptdf(4, edges, np.array([1.0, 2.0, 3.0]), zones=np.array([0, 0, 1, 1]))
    assert zone_edges == [(0, 1)]
    assert Hr @ np.array([1.0, -1.0]) == pytest.approx([1.0])


def test_reduced_ptdf_is_gauge_free_and_matches_a_direct_computation():
    """The zonal PTDF must act on balanced zone injections only: its rows sum to zero, and a transfer
    from zone a to zone c, spread evenly over the buses of each (eq. (5): T_bz^T (T_bz T_bz^T)^{-1}),
    gives the same interzonal flows as solving the full network directly. A zonal PTDF with rows not
    summing to zero is compared with the reduced network's in different gauges (found 2026-09-19)."""
    rng = np.random.default_rng(2)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4), (2, 5)]
    b = rng.uniform(0.5, 2.0, len(edges))
    zones = np.array([0, 0, 1, 1, 2, 2])
    Hr, ze = reduced_ptdf(6, edges, b, zones)
    assert np.allclose(Hr.sum(1), 0.0)
    p = np.array([0.5, 0.5, 0.0, 0.0, -0.5, -0.5])               # one unit from zone 0 to zone 2
    theta = np.linalg.pinv(laplacian(6, edges, b)) @ p
    direct = np.zeros(len(ze))
    for (i, j), be in zip(edges, b):
        if zones[i] != zones[j]:
            k = ze.index((min(zones[i], zones[j]), max(zones[i], zones[j])))
            direct[k] += (1.0 if zones[i] < zones[j] else -1.0) * be * (theta[i] - theta[j])
    assert Hr @ np.array([1.0, 0.0, -1.0]) == pytest.approx(direct)


def test_reduced_ptdf_with_participation_weights():
    """With `weights` a zone's injection is distributed over its buses by the weights normalized within
    the zone, the participation factors of Cheng & Overbye 2005 (p. 1869), as in `zonal_operator`. Equal
    weights give the default of eq. (5); all weight on one bus per zone gives the flows of transfers
    between those buses, the zonal PTDF that belongs to the Kron reduction onto them. Both are checked
    against a direct solution of the full network."""
    rng = np.random.default_rng(3)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4), (2, 5)]
    b = rng.uniform(0.5, 2.0, len(edges))
    zones = np.array([0, 0, 1, 1, 2, 2])
    H_default, ze = reduced_ptdf(6, edges, b, zones)
    H_equal, ze_equal = reduced_ptdf(6, edges, b, zones, weights=np.full(6, 7.0))
    assert ze_equal == ze and np.allclose(H_equal, H_default)
    point = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])            # buses 1, 3, 4 carry their zones
    H_point, _ = reduced_ptdf(6, edges, b, zones, weights=point)
    assert np.allclose(H_point.sum(1), 0.0)
    theta = np.linalg.pinv(laplacian(6, edges, b)) @ np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    direct = np.zeros(len(ze))
    for (i, j), be in zip(edges, b):
        if zones[i] != zones[j]:
            k = ze.index((min(zones[i], zones[j]), max(zones[i], zones[j])))
            direct[k] += (1.0 if zones[i] < zones[j] else -1.0) * be * (theta[i] - theta[j])
    assert H_point @ np.array([1.0, 0.0, -1.0]) == pytest.approx(direct)
    assert not np.allclose(H_point, H_default)


@pytest.mark.data("matpower")
def test_reduced_ptdf_reproduces_fortenbacher_table_ii_on_ieee14():
    """Fortenbacher et al. 2018, Sec. III-C: IEEE 14 in four zones (Table I), interconnections 1-4, 1-3,
    4-3, 1-2, 3-2, and the injection-independent zonal PTDF of Table II, columns for zones 2, 3, 4 with
    bus 1 as slack. First their eq. (5) verbatim against the table; then `reduced_ptdf`, which is
    slack-free, against it on the transfers between zones 2, 3, 4, where the slack cancels."""
    from pathlib import Path

    from gridnet.network import read_matpower_case

    case = Path(__file__).resolve().parents[2] / "data" / "matpower" / "case14.m"
    net = read_matpower_case(case)
    edges = list(zip(net.from_bus.tolist(), net.to_bus.tolist()))
    b = 1.0 / net.x
    zones = np.array([1, 1, 4, 3, 1, 2, 3, 3, 3, 2, 2, 2, 2, 2]) - 1
    table = np.array([[-0.126, -0.143, -0.532], [-0.343, -0.676, -0.450], [-0.126, -0.143, 0.468],
                      [-0.530, -0.179, -0.017], [-0.469, 0.179, 0.017]])
    rows = [(0, 3), (0, 2), (3, 2), (0, 1), (2, 1)]              # their flow directions, zone numbers from 0
    # eq. (5) verbatim: slack-adjusted H_f (bus 1 removed), T_f, and T_bz for zones 2, 3, 4
    from gridnet.graph import incidence_matrix, laplacian
    C = incidence_matrix(14, edges)
    L = laplacian(14, edges, b)
    Hf = np.zeros((len(edges), 14)); Hf[:, 1:] = (b[:, None] * C.T)[:, 1:] @ np.linalg.inv(L[1:, 1:])
    Tf = np.zeros((5, len(edges)))
    for k, (i, j) in enumerate(edges):
        for r, (za, zc) in enumerate(rows):
            if (zones[i], zones[j]) == (za, zc): Tf[r, k] = 1.0
            if (zones[i], zones[j]) == (zc, za): Tf[r, k] = -1.0
    Tbz = np.zeros((3, 14))
    for col, zn in enumerate((1, 2, 3)):
        Tbz[col, zones == zn] = 1.0
    H_ind = Tf @ Hf @ Tbz.T @ np.linalg.inv(Tbz @ Tbz.T)
    assert np.allclose(H_ind, table, atol=2e-3)
    # the slack-free function on transfers among zones 2, 3, 4
    Hr, ze = reduced_ptdf(14, edges, b, zones)
    mine = np.array([(1.0 if (za, zc) in ze else -1.0) * Hr[ze.index((min(za, zc), max(za, zc)))] for za, zc in rows])
    for a, c in ((1, 2), (1, 3), (2, 3)):
        d = np.zeros(4); d[a], d[c] = 1.0, -1.0
        assert np.allclose(mine @ d, H_ind[:, a - 1] - H_ind[:, c - 1], atol=1e-10)


@pytest.mark.data("matpower")
def test_ptdf_fit_reproduces_fortenbacher_table_iii_on_ieee14():
    """Fortenbacher et al. 2018, Table III: the summed susceptances b_phys = (5.05, 29.41, 5.84, 3.96, 15.53)
    of the interconnections 1-4, 1-3, 4-3, 1-2, 3-2 and the fit of eq. (16) to the injection-independent
    zonal PTDF of Table II, b_opt = (12.47, 29.41, 16.97, 11.04, 12.98), pinned at the strongest, b_13.
    Their PTDF has zone 1 as slack, so the fit is done on the transfers from zones 2, 3, 4 to zone 1."""
    from pathlib import Path

    from gridnet.network import read_matpower_case

    case = Path(__file__).resolve().parents[2] / "data" / "matpower" / "case14.m"
    net = read_matpower_case(case)
    edges = list(zip(net.from_bus.tolist(), net.to_bus.tolist()))
    b = 1.0 / net.x
    zones = np.array([1, 1, 4, 3, 1, 2, 3, 3, 3, 2, 2, 2, 2, 2]) - 1
    zone_edges = [(0, 3), (0, 2), (3, 2), (0, 1), (2, 1)]
    b_phys = np.zeros(5)
    for (i, j), be in zip(edges, b):
        for k, (za, zc) in enumerate(zone_edges):
            if {zones[i], zones[j]} == {za, zc}:
                b_phys[k] += be
    assert b_phys == pytest.approx([5.05, 29.41, 5.84, 3.96, 15.53], abs=1e-2)       # their table truncates (29.419, 5.847, 3.968)
    # the target is eq. (5) computed exactly (their Table II has three decimals, and fitting that rounded
    # table moves the result by up to 0.16 %); their tables truncate, so compare the truncated values
    from gridnet.graph import incidence_matrix
    C = incidence_matrix(14, edges)
    L = laplacian(14, edges, b)
    Hf = np.zeros((len(edges), 14)); Hf[:, 1:] = (b[:, None] * C.T)[:, 1:] @ np.linalg.inv(L[1:, 1:])
    Tf = np.zeros((5, len(edges)))
    for k, (i, j) in enumerate(edges):
        for r, (za, zc) in enumerate(zone_edges):
            if (zones[i], zones[j]) == (za, zc): Tf[r, k] = 1.0
            if (zones[i], zones[j]) == (zc, za): Tf[r, k] = -1.0
    Tbz = np.zeros((3, 14))
    for col, zn in enumerate((1, 2, 3)):
        Tbz[col, zones == zn] = 1.0
    H_ind = Tf @ Hf @ Tbz.T @ np.linalg.inv(Tbz @ Tbz.T)
    to_slack = np.zeros((4, 3)); to_slack[0, :] = -1.0; to_slack[[1, 2, 3], [0, 1, 2]] = 1.0
    fit = reweight_ptdf(H_ind, 4, zone_edges, reference=b_phys, basis=to_slack)
    assert np.array_equal(np.floor(fit * 100) / 100, [12.47, 29.41, 16.97, 11.04, 12.98])
    published = np.array([12.47, 29.41, 16.97, 11.04, 12.98])
    objective = lambda bb: np.linalg.norm(ptdf_from_susceptances(4, zone_edges, bb) @ to_slack - H_ind)
    assert objective(fit) <= objective(published) + 1e-9         # at least as good a minimizer of their eq. (16)


def test_ptdf_is_blind_to_bridges_and_to_the_scale_of_each_block():
    """Cheng & Overbye 2005, Sec. V: a network separates at its cut nodes into independent subsystems; the
    PTDF fixes the susceptances of each only up to its own common factor, and "the susceptance value of [a
    radial branch] does not affect any ISF or PTDF". Two triangles sharing node 2, and a bridge 4-5."""
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)]
    b = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 0.5, 4.0])
    H = ptdf_from_susceptances(6, edges, b)
    scaled = b.copy(); scaled[:3] *= 7.0; scaled[3:6] *= 0.2; scaled[6] *= 1e3
    assert np.allclose(ptdf_from_susceptances(6, edges, scaled), H, atol=1e-10)


def test_ptdf_fit_pins_every_block_and_leaves_bridges_at_their_reference():
    """The fit recovers the susceptances from their PTDF when every block is pinned at the reference value
    of its strongest edge, and returns the reference on bridges, about which the PTDF says nothing. (With
    one pin for the whole network the bridges of the European zonal graph ran away, found 2026-09-19.)"""
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)]
    b = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 0.5, 4.0])
    H = ptdf_from_susceptances(6, edges, b)
    reference = b * np.array([0.5, 0.5, 1.0, 0.5, 1.0, 0.3, 0.25])       # true only at the strongest edge of each block
    assert np.argmax(reference[:3]) == 2 and np.argmax(reference[3:6]) == 1
    fit = reweight_ptdf(H, 6, edges, reference=reference)
    assert fit[:6] == pytest.approx(b[:6], rel=1e-5)
    assert fit[6] == reference[6]


def test_spectral_reweighting_does_not_depend_on_the_units_of_the_weights():
    """The same graph in other units, L -> 1000 L and L -> L / 1000: the weights scale along, the error
    and its certified bound do not change. (Unscaled, the solver failed on the European Kron matrix.)"""
    rng = np.random.default_rng(5)
    edges_all = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    L = laplacian(7, edges_all, rng.uniform(0.5, 2.0, len(edges_all)))
    ring = [(i, (i + 1) % 7) for i in range(7)]
    w1, e1, b1 = reweight_spectral(L, ring)
    assert b1 <= e1 + 1e-12 and e1 - b1 < 1e-5
    for factor in (1e3, 1e-3):
        w, e, bnd = reweight_spectral(factor * L, ring)
        assert e == pytest.approx(e1, abs=1e-5) and bnd == pytest.approx(b1, abs=1e-5)
        assert w == pytest.approx(factor * w1, rel=1e-3)


def test_best_scales_closed_forms():
    """One component, the path in the unit triangle: the best common factor is the spectral optimum 3/2
    with eps = 1/2 (closed form above). Two components that are the graph's own parts, scaled wrongly:
    the factors undo the scaling and the error vanishes."""
    s, eps, bound = best_scales(TRIANGLE, [laplacian(3, PATH, [1.0, 1.0])])
    assert s == pytest.approx([1.5], abs=1e-4) and eps == pytest.approx(0.5, abs=1e-4) and bound == pytest.approx(0.5, abs=1e-4)
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    b = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 0.5])
    L = laplacian(5, edges, b)
    parts = [laplacian(5, edges[:3], 4.0 * b[:3]), laplacian(5, edges[3:], 0.1 * b[3:])]
    s, eps, bound = best_scales(L, parts)
    assert s == pytest.approx([0.25, 10.0], rel=1e-4) and eps == pytest.approx(0.0, abs=1e-5)


def test_ptdf_reproduces_cheng_overbye_six_bus_figures():
    """Cheng & Overbye 2005, Figs. 1 and 2: six buses, seven lines of 0.1j per unit, a transfer from area A
    (buses 1, 2) to the slack bus 0. With bus 1 supplying all of it the lines carry 73, 27, 27, 20, 7, 7, 7
    per cent; with bus 1 supplying 25 % and bus 2 75 % they carry 53, 28, 47, 35, 12, 12, 12. Their point:
    "the PTDFs depend upon the assumed generator participation values", where the injection is placed."""
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 5), (3, 5)]
    H = ptdf_from_susceptances(6, edges, np.full(7, 10.0))
    for shares, published in (({1: 1.0}, [73, 27, 27, 20, 7, 7, 7]), ({1: 0.25, 2: 0.75}, [53, 28, 47, 35, 12, 12, 12])):
        p = np.zeros(6); p[0] = -1.0
        for bus, share in shares.items():
            p[bus] = share
        assert np.round(100 * np.abs(H @ p)).astype(int).tolist() == published


def test_best_scale_of_one_component_is_the_closed_form():
    """One component: s mu_min >= 1 - eps and s mu_max <= 1 + eps give eps = (mu_max - mu_min) / (mu_max + mu_min)
    at s = 2 / (mu_max + mu_min), mu the eigenvalues of L^{+/2} L_1 L^{+/2}. A graph against itself has eps = 0
    exactly (as a semidefinite program this instance is degenerate, every eigenvalue active in both
    inequalities, and the interior-point solver failed on it, found 2026-09-19)."""
    rng = np.random.default_rng(7)
    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    L = laplacian(6, edges, rng.uniform(0.5, 2.0, len(edges)))
    s, eps, bound = best_scales(L, [L])
    assert s == pytest.approx([1.0]) and eps == pytest.approx(0.0, abs=1e-12) and bound == pytest.approx(0.0, abs=1e-12)
    ring = laplacian(6, [(i, (i + 1) % 6) for i in range(6)], np.ones(6))
    lam, V = np.linalg.eigh(L); half = V[:, 1:] / np.sqrt(lam[1:])
    mu = np.linalg.eigvalsh(half.T @ ring @ half)
    s, eps, bound = best_scales(L, [ring])
    assert eps == pytest.approx((mu.max() - mu.min()) / (mu.max() + mu.min())) and bound == eps
    assert s[0] == pytest.approx(2 / (mu.max() + mu.min()))


def _zoned_network(internal, seed=0):
    """Five zones of four buses; every zone a complete graph with susceptance `internal`; a ring of zones plus
    one chord, each interconnection made of two parallel lines between different bus pairs."""
    rng = np.random.default_rng(seed)
    edges, b = [], []
    for zn in range(5):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append((4 * zn + i, 4 * zn + j)); b.append(internal)
    ties = {}
    for za, zc in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]:
        total = 0.0
        for (i, j) in ((0, 1), (2, 3)):
            w = rng.uniform(0.5, 2.0); edges.append((4 * za + i, 4 * zc + j)); b.append(w); total += w
        ties[(za, zc)] = total
    return edges, np.array(b), np.repeat(np.arange(5), 4), ties


def test_all_measures_agree_when_every_bus_is_its_own_zone():
    """Truth known: with one bus per zone the zonal operator is the Laplacian itself, the zonal PTDF is the
    PTDF, and the fit returns the susceptances, so every error vanishes."""
    rng = np.random.default_rng(3)
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (2, 4)]
    b = rng.uniform(0.5, 2.0, len(edges))
    L = laplacian(5, edges, b)
    assert np.allclose(zonal_operator(L, np.arange(5)), L, atol=1e-10)
    Hr, ze = reduced_ptdf(5, edges, b, np.arange(5))
    fit = reweight_ptdf(Hr, 5, ze, reference=np.where(b == b.max(), b, 1.0))
    assert best_scales(L, [laplacian(5, ze, fit)])[1] == pytest.approx(0.0, abs=1e-6)


def test_all_measures_agree_in_the_limit_of_weakly_connected_areas():
    """Truth known in the limit: if the lines inside a zone are far stronger than the interconnections, a zone
    behaves as one node. Then the zonal operator (injections spread over the zone, angles averaged), the Kron
    reduction onto one bus per zone, and the network of summed interconnection susceptances coincide, and the
    PTDF fit returns the summed susceptances. This is the setting of area aggregation, densely connected areas
    with sparse or weak connections between them (Chow & Kokotovic 1985, Sec. II), where the rigid and the
    complete aggregate coincide. The agreement must tighten as the internal connections strengthen; without that separation
    the three differ, which is the situation of the European grid (studies/aggregation/european_grid)."""
    from gridnet.kron import kron_reduce

    errors = {}
    for internal in (1.0, 1e2, 1e4):
        edges, b, zones, ties = _zoned_network(internal)
        L = laplacian(20, edges, b)
        Lz = zonal_operator(L, zones)
        kron = kron_reduce(L, keep=[0, 4, 8, 12, 16])
        ze = sorted(ties); summed = np.array([ties[e] for e in ze])
        quotient = laplacian(5, ze, summed)
        Hr, ze2 = reduced_ptdf(20, edges, b, zones)
        assert ze2 == ze
        fit = reweight_ptdf(Hr, 5, ze, reference=summed)
        errors[internal] = (best_scales(Lz, [kron])[1], best_scales(Lz, [quotient])[1], best_scales(Lz, [laplacian(5, ze, fit)])[1],
                            float(np.max(np.abs(fit / summed - 1))))
    assert all(errors[1e4][k] < errors[1e2][k] < errors[1.0][k] for k in range(4))
    assert max(errors[1e4]) < 1e-3
    assert errors[1.0][0] > 0.05                                 # no such separation: representative bus and zone mean differ


def test_zonal_operator_is_the_complete_aggregate_of_chow_and_kokotovic():
    """Chow & Kokotovic 1985, eqs. (4.3), (4.4): in coordinates y (area means) and z (differences inside the
    areas) the slow subsystem is M_a y' = (K_a + K_a^i) y, where K_a sums the external connections (the rigid
    aggregate, areas as rigid bodies) and K_a^i = -K_yz K_zz^{-1} K_zy "accounts for the fact that the internal
    connections are not infinitely stiff"; together they are the Schur complement of the network onto the area
    coordinates. With U the area indicator matrix and D = U^T U the area sizes, that Schur complement S, taken
    in the orthonormal coordinates U D^{-1/2}, satisfies zonal_operator = Pi D^{1/2} S D^{1/2} Pi on balanced
    zone injections, and the rigid aggregate is the quotient graph U^T L U."""
    from scipy.linalg import null_space

    edges, b, zones, ties = _zoned_network(3.0, seed=4)
    L = laplacian(20, edges, b)
    U = np.zeros((20, 5)); U[np.arange(20), zones] = 1.0
    D = U.T @ U
    Y = U @ np.diag(np.diag(D) ** -0.5)                          # orthonormal area coordinates
    Z = null_space(U.T)                                          # differences inside the areas
    S = Y.T @ L @ Y - Y.T @ L @ Z @ np.linalg.inv(Z.T @ L @ Z) @ Z.T @ L @ Y
    complete = np.sqrt(D) @ S @ np.sqrt(D)
    Pi = np.eye(5) - 1.0 / 5
    Lz = zonal_operator(L, zones)
    # both are pseudoinverses of the same compliance on balanced injections: compare through it
    P = U @ np.linalg.inv(D)
    assert np.allclose(np.linalg.pinv(Pi @ (P.T @ np.linalg.pinv(L) @ P) @ Pi), Lz, atol=1e-9)
    assert np.allclose(Pi @ np.linalg.pinv(Pi @ np.linalg.pinv(complete) @ Pi) @ Pi, Lz, atol=1e-8)
    rigid = U.T @ L @ U
    quotient = laplacian(5, sorted(ties), np.array([ties[e] for e in sorted(ties)]))
    assert np.allclose(rigid, quotient, atol=1e-12)


def test_zonal_operator_with_participation_weights_contains_the_kron_reduction():
    """The complete aggregate depends on how a zone's injection is distributed over its buses, the participation
    factors of Cheng & Overbye 2005 (p. 1869), and on the same weights for the zone's angle (Chow & Kokotovic 1985,
    eq. (3.1), take the area's center of inertia). With all weight on one bus per zone the operator is
    the Kron reduction onto those buses (Dorfler & Bullo 2013, eq. (2.1)); with equal weights it is the operator
    without weights; the weights are normalized within each zone."""
    from gridnet.kron import kron_reduce

    edges, b, zones, _ = _zoned_network(1.0, seed=2)
    L = laplacian(20, edges, b)
    reps = [1, 6, 8, 15, 17]                                     # one bus in each of the five zones
    onehot = np.zeros(20); onehot[reps] = 1.0
    assert np.allclose(zonal_operator(L, zones, weights=onehot), kron_reduce(L, reps), atol=1e-9)
    assert np.allclose(zonal_operator(L, zones, weights=np.full(20, 3.0)), zonal_operator(L, zones), atol=1e-10)
    nearly = onehot + 1e-7                                       # continuity: nearly all weight on the representatives
    assert np.allclose(zonal_operator(L, zones, weights=nearly), kron_reduce(L, reps), atol=1e-4)


def test_distance_between_zones_is_the_squared_distance_of_barycenters():
    """The distance between two zones in the complete aggregate with weights p, q is half the energy distance of
    p and q with respect to the resistance metric. Sejdinovic, Sriperumbudur, Gretton & Fukumizu 2013, Def. 5,
    eq. (2.5): for a semimetric rho of negative type D_E(P, Q) = 2 E rho(Z, W) - E rho(Z, Z') - E rho(W, W'),
    in integral form -int rho d([P - Q] x [P - Q]) (eq. (2.6)); Thm. 22: D_E = 2 gamma_k^2 for any kernel k that
    generates rho by rho(z, w) = k(z, z) + k(w, w) - 2 k(z, w), gamma_k the distance of the mean embeddings. The
    effective resistance is generated by k = L^+ and is of negative type, being a squared Euclidean distance
    (their Prop. 3; von Luxburg 2007, Sec. 6.2; Devriendt 2022, eq. (5)), so

        (p - q)^T L^+ (p - q) = sum p_i q_j R_ij - 1/2 sum p_i p_i' R_ii' - 1/2 sum q_j q_j' R_jj',

    the squared distance of the two barycenters in the embedding z_i = rows of U (Lambda^+)^{1/2}."""
    from gridnet.graph import effective_resistance

    edges, b, zones, _ = _zoned_network(1.0, seed=5)
    L = laplacian(20, edges, b)
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 1.0, 20)
    R = effective_resistance(L)
    G = np.linalg.pinv(zonal_operator(L, zones, weights=w)); d = np.diag(G)
    Rz = d[:, None] + d[None, :] - 2 * G
    lam, V = np.linalg.eigh(L); Z = V[:, 1:] / np.sqrt(lam[1:])                   # commute-time embedding, rows z_i
    for a, c in ((0, 1), (0, 3), (2, 4)):
        p = np.where(zones == a, w, 0.0); p /= p.sum()
        q = np.where(zones == c, w, 0.0); q /= q.sum()
        identity = p @ R @ q - 0.5 * p @ R @ p - 0.5 * q @ R @ q
        barycenters = np.sum((p @ Z - q @ Z) ** 2)
        assert Rz[a, c] == pytest.approx(identity, rel=1e-10) and Rz[a, c] == pytest.approx(barycenters, rel=1e-10)


def test_the_distances_between_two_sets_range_from_the_shorted_resistance_to_the_largest_bus_pair():
    """The distance (p - q)^T L^+ (p - q) between two sets A, C is defined per pair of probability weights. Its
    minimum over all weights is the effective resistance between A and C with each set short-circuited to one node:
    a unit flow from A to C has divergence p - q for some weights, its least energy for given weights is the
    quadratic form (Thomson's principle, Doyle & Snell 2000, Sec. 1.3.5), and the current with A and C shorted is the flow
    of least energy among all of them. This minimum is the literature's definition of the effective resistance between
    two sets: Lyons & Peres 2016, Ch. 2, eq. (2.4) with the sets identified to single vertices, and Exercise 2.13,
    R(A <-> Z) = min { sum_e r(e) theta(e)^2 : theta a unit flow from A to Z }. The minimizing weights are the currents entering the network at the buses of
    A, nonnegative since A is at the highest potential. The maximum is attained at point masses, the form being
    convex in the weights, and is the largest resistance between a bus of A and a bus of C. Shorting further sets
    can only lower the resistance (Shorting Law, Doyle & Snell 2000, Sec. 2.2), so the quotient graph, which shorts every
    cluster at once, lies below the pairwise minimum."""
    from gridnet.graph import effective_resistance

    rng = np.random.default_rng(0)
    n = 14
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 7), (2, 9), (4, 11), (1, 5), (8, 12)]
    L = laplacian(n, edges, rng.uniform(0.5, 2.0, len(edges)))
    A, C = [0, 1, 2, 3], [7, 8, 9]
    Lp, R = np.linalg.pinv(L), effective_resistance(L)
    idx = A + C
    K = Lp[np.ix_(idx, idx)]
    E = np.zeros((2, len(idx))); E[0, :len(A)] = 1.0; E[1, len(A):] = -1.0          # x = (p, -q), sum p = sum q = 1
    x = np.linalg.solve(np.block([[2 * K, E.T], [E, np.zeros((2, 2))]]), np.r_[np.zeros(len(idx)), 1.0, 1.0])[:len(idx)]
    p, q = x[:len(A)], -x[len(A):]
    assert np.all(p > 0) and np.all(q > 0)
    lab = np.arange(n); lab[A] = A[0]; lab[C] = C[0]
    lab = np.unique(lab, return_inverse=True)[1]
    U = np.zeros((n, lab.max() + 1)); U[np.arange(n), lab] = 1.0
    shorted = effective_resistance(U.T @ L @ U)[lab[A[0]], lab[C[0]]]
    assert x @ K @ x == pytest.approx(shorted, rel=1e-10)
    for _ in range(200):                                          # every pair of probability weights lies in the range
        pp = np.zeros(n); pp[A] = rng.dirichlet(np.ones(len(A)))
        qq = np.zeros(n); qq[C] = rng.dirichlet(np.ones(len(C)))
        assert shorted - 1e-12 <= (pp - qq) @ Lp @ (pp - qq) <= R[np.ix_(A, C)].max() + 1e-12
    third = [11, 12]                                              # shorting a further set lowers the resistance
    lab2 = np.arange(n); lab2[A] = A[0]; lab2[C] = C[0]; lab2[third] = third[0]
    lab2 = np.unique(lab2, return_inverse=True)[1]
    U2 = np.zeros((n, lab2.max() + 1)); U2[np.arange(n), lab2] = 1.0
    assert effective_resistance(U2.T @ L @ U2)[lab2[A[0]], lab2[C[0]]] <= shorted + 1e-12


def test_rigid_aggregate_dominates_the_complete_aggregate_and_its_two_cluster_resistance_is_one_over_the_cut():
    """The two aggregates are the two Galerkin restrictions of L theta = mu. Restricting the potentials to be constant
    on the clusters gives U^T L U, the rigid aggregate (Dirichlet's principle: an energy minimized over fewer
    potentials is larger). Restricting the sources to the span of one distribution per cluster, the columns of P,
    gives (P^T L^+ P)^+, the complete aggregate (Thomson's principle; Doyle & Snell 2000, Sec. 1.3.5; Lyons & Peres 2016,
    Exercise 2.13). With P^T U = I, for balanced y and z,  y^T z = (U y)^T L^{1/2} L^{+/2} (P z) <= |U y|_L |P z|_{L^+},
    and the supremum over z of (y^T z)^2 / (z^T P^T L^+ P z) is y^T (P^T L^+ P)^+ y, so U^T L U dominates the complete
    aggregate as a quadratic form, for any shares. This is the Gauss-Markov theorem: for a linear model with covariance
    Sigma and design X every linear unbiased estimator C, C X = I, has covariance C Sigma C^T >= (X^T Sigma^-1 X)^-1, the
    covariance of the best one (stated in Barooah & Hespanha 2009, Sec. I, with the covariance of that estimator equal to
    an effective resistance, their Sec. IV-A); here Sigma = L, C = U^T, X = P. The name of U^T L U in numerical analysis
    is the Galerkin coarse grid matrix of aggregation with piecewise constant prolongation (Napov & Notay 2011, Sec. 2). For two clusters the rigid aggregate is a single
    branch whose susceptance is the cut, the sum of the susceptances between the parts, so its resistance is 1 / cut."""
    from gridnet.graph import effective_resistance

    for internal in (0.3, 1.0, 10.0):
        edges, b, zones, _ = _zoned_network(internal, seed=1)
        L = laplacian(20, edges, b)
        U = np.zeros((20, 5)); U[np.arange(20), zones] = 1.0
        for weights in (None, np.random.default_rng(3).uniform(0.2, 1.0, 20)):
            gap = np.linalg.eigvalsh(U.T @ L @ U - zonal_operator(L, zones, weights=weights))
            assert abs(gap[0]) < 1e-9 and gap[1] > 0                    # the constants, then strictly positive
    rng = np.random.default_rng(1)
    n = 16
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 8), (3, 11), (5, 13)]
    b = rng.uniform(0.5, 2.0, len(edges))
    L = laplacian(n, edges, b)
    part = (np.arange(n) >= 8).astype(int)
    U = np.zeros((n, 2)); U[np.arange(n), part] = 1.0
    cut = sum(be for (i, j), be in zip(edges, b) if part[i] != part[j])
    assert effective_resistance(U.T @ L @ U)[0, 1] == pytest.approx(1.0 / cut)


def test_two_triangles_joined_by_a_bridge_have_the_three_distances_in_closed_form():
    """Two triangles with branch susceptance w,
    joined by a bridge of susceptance 1 between buses 2 and 3, each triangle a cluster. Derived here by
    series and parallel rules: inside a triangle R = 2/(3w) between two buses; the rigid aggregate is the
    bridge alone, distance 1 = 1/cut; with uniform distributions the distance is the mean resistance between
    the clusters, 1 + 8/(9w), less half the mean inside each, 4/(9w) each (the diagonal included), hence
    1 + 4/(9w); the Kron reduction onto two buses away from the bridge gives 2/(3w) + 1 + 2/(3w)."""
    from gridnet.kron import kron_reduce
    zones = np.array([0, 0, 0, 1, 1, 1])
    U = np.eye(2)[zones]
    for w in (1.0, 10.0, 100.0):
        L = laplacian(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)], [w] * 6 + [1.0])
        assert 1 / (U.T @ L @ U)[0, 0] == pytest.approx(1.0)
        assert 1 / zonal_operator(L, zones)[0, 0] == pytest.approx(1 + 4 / (9 * w))
        assert 1 / kron_reduce(L, [0, 5])[0, 0] == pytest.approx(1 + 4 / (3 * w))
        assert 1 / kron_reduce(L, [2, 3])[0, 0] == pytest.approx(1.0)


def test_reweighting_rejects_an_indefinite_reference():
    """The whitening L^{+/2} of the reweighting problems needs L positive semidefinite; the triangle with
    weights 1, 1, -0.8 (a series capacitor, eigenvalues -0.6, 0, 3) is not, and the functions refuse
    instead of silently dropping the negative eigenvalue."""
    L = laplacian(3, [(0, 1), (1, 2), (0, 2)], [1.0, 1.0, -0.8])
    for f in (lambda: reweight_frobenius(L, PATH), lambda: reweight_spectral(L, PATH),
              lambda: best_scales(L, [L, TRIANGLE])):
        with pytest.raises(ValueError, match="positive semidefinite"):
            f()


def test_zone_numbers_must_be_consecutive_from_zero():
    """reduced_ptdf and zonal_operator index the zones by their numbers 0..z-1; an unused number is an empty
    zone, a division by its size zero (NaN in the PTDF, a failing SVD in the pseudoinverse), so it is refused."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    b = np.ones(4)
    gap = np.array([0, 0, 2, 2])
    with pytest.raises(ValueError, match="labels"):
        reduced_ptdf(4, edges, b, gap)
    with pytest.raises(ValueError, match="labels"):
        zonal_operator(laplacian(4, edges, b), gap)


def test_fill_zones_sends_each_unlabeled_bus_to_the_nearest_zone_by_reactance():
    """Path 0-1-2-3 with reactances 1, 4, 1 and zones 0 and 1 at the ends: bus 1 is at reactance 1
    from zone 0 and 5 from zone 1, bus 2 the other way round. A bus with no path to a labeled bus
    raises."""
    from gridnet.reweight import fill_zones

    edges, b = [(0, 1), (1, 2), (2, 3)], np.array([1.0, 0.25, 1.0])
    assert fill_zones(np.array([0, -1, -1, 1]), edges, b).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        fill_zones(np.array([0, -1, -1, 1, -1]), edges, b)            # bus 4 has no branch


def test_quotient_susceptances_add_the_branches_between_two_zones():
    """Fortenbacher et al. 2018, eq. (8): the susceptance of an interconnection is the sum of the
    susceptances of the branches between its two zones; branches inside a zone do not count."""
    from gridnet.reweight import quotient_susceptances

    zones = np.array([0, 0, 1, 1])
    edges, b = [(0, 2), (1, 3), (0, 1), (2, 3)], np.array([1.0, 2.0, 5.0, 7.0])
    assert quotient_susceptances(zones, edges, b, [(0, 1)]) == pytest.approx([3.0])


def test_zonal_ptdf_error_vanishes_on_the_graph_that_defines_the_ptdf():
    """A graph on the zone pairs themselves sends the flows of its own PTDF, at any common scale of
    its susceptances (the PTDF is homogeneous of degree zero), and another split of the same total
    does not."""
    from gridnet.reweight import zonal_ptdf_error

    edges, b = [(0, 1), (1, 2), (0, 2), (2, 3)], np.array([1.0, 2.0, 3.0, 0.5])
    H = ptdf_from_susceptances(4, edges, b)
    assert zonal_ptdf_error(laplacian(4, edges, b), edges, H) == pytest.approx(0.0, abs=1e-12)
    assert zonal_ptdf_error(laplacian(4, edges, 2 * b), edges, H) == pytest.approx(0.0, abs=1e-12)
    assert zonal_ptdf_error(laplacian(4, edges, b[[1, 0, 2, 3]]), edges, H) > 0.05


def test_fill_zones_combines_parallel_branches_by_their_susceptances():
    """Bus 1 is unlabeled, joined to bus 0 (zone 0) by a double circuit of reactance 1 each and to bus
    2 (zone 1) by one branch of reactance 1.5. In parallel the double circuit is 0.5, so bus 1 is
    nearer zone 0. Summing the duplicate reactances (series, 2.0) put it in zone 1 (found and
    reproduced in 2026-10)."""
    from gridnet.reweight import fill_zones

    zones = fill_zones(np.array([0, -1, 1]), [(0, 1), (0, 1), (1, 2)], np.array([1.0, 1.0, 1 / 1.5]))
    assert zones.tolist() == [0, 0, 1]


def test_reduced_ptdf_reproduces_oh_six_bus_example():
    """Oh 2010, Sec. V-A (printed p. 682): six buses, lines 1-2, 1-5, 2-3, 3-4, 3-6, 4-6, 5-6 of reactance
    0.1, groups I = {1}, II = {2, 3}, III = {4}, IV = {5, 6}. Eq. (29) is the PTDF with bus 1 as slack,
    eq. (32) the reduced PTDF of the groups with group I as slack, eq. (33) the flows for the injection
    (-5, 1, 1, 1, 1, 1). `reduced_ptdf` is slack-free; less its column of group I it is eq. (32)."""
    from gridnet.graph import incidence_matrix, laplacian
    from gridnet.reweight import reduced_ptdf

    edges = [(0, 1), (0, 4), (1, 2), (2, 3), (2, 5), (3, 5), (4, 5)]
    b = np.full(7, 10.0)
    C, L = incidence_matrix(6, edges), laplacian(6, edges, b)
    H = np.zeros((7, 6)); H[:, 1:] = (b[:, None] * C.T)[:, 1:] @ np.linalg.inv(L[1:, 1:])
    eq29 = np.array([[0, -0.786, -0.571, -0.500, -0.214, -0.429], [0, -0.214, -0.429, -0.500, -0.786, -0.571],
                     [0, 0.214, -0.571, -0.500, -0.214, -0.429], [0, 0.071, 0.143, -0.500, -0.071, -0.143],
                     [0, 0.143, 0.286, 0, -0.143, -0.286], [0, 0.071, 0.143, 0.500, -0.071, -0.143],
                     [0, -0.214, -0.429, -0.500, 0.214, -0.571]])
    np.testing.assert_allclose(H, eq29, atol=5e-4 + 1e-9)
    Hr, pairs = reduced_ptdf(6, edges, b, np.array([0, 1, 1, 2, 3, 3]))
    assert pairs == [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]   # I-II, I-IV, II-III, II-IV, III-IV, as in eq. (32)
    eq32 = np.array([[0, -0.679, -0.500, -0.321], [0, -0.321, -0.500, -0.679], [0, 0.107, -0.500, -0.107],
                     [0, 0.214, 0, -0.214], [0, 0.107, 0.500, -0.107]])
    np.testing.assert_allclose(Hr - Hr[:, [0]], eq32, atol=5e-4 + 1e-9)
    np.testing.assert_allclose(H @ np.array([-5.0, 1, 1, 1, 1, 1]), [-2.5, -2.5, -1.5, -0.5, 0, 0.5, -1.5], atol=1e-12)


@pytest.mark.data("matpower")
def test_nrmse_of_fortenbacher_figure_3a_and_4a():
    """Fortenbacher et al. 2018, eq. (17): NRMSE = RMSE(T_f H_f p - H T_bz p) / mean |T_f H_f p| over the
    five interconnections of IEEE 14 in four zones at the injections of Table I (p. 4). Printed in Fig. 3a:
    H_red^dep 0.038, H_red^ind 0.093; in Fig. 4a (p. 5): b^phys 0.33. Here H^dep from their Table II gives
    0.0378; H^ind gives 0.0939 from their Table II and 0.0937 from `reduced_ptdf`, and the summed
    susceptances 0.336, one in the last printed digit off in both (a recorded deviation of 2026-10-24)."""
    from pathlib import Path

    from gridnet.graph import incidence_matrix
    from gridnet.network import read_matpower_case
    from gridnet.reweight import quotient_susceptances, reduced_ptdf

    net = read_matpower_case(Path(__file__).resolve().parents[2] / "data" / "matpower" / "case14.m")
    edges = list(zip(net.from_bus.tolist(), net.to_bus.tolist())); b = 1.0 / net.x
    zones = np.array([1, 1, 4, 3, 1, 2, 3, 3, 3, 2, 2, 2, 2, 2]) - 1
    p = np.array([41, 46, 37, -57, 34, 13, -94, -20, -22, 61, -27, -21, 13, -4], dtype=float)
    C, L = incidence_matrix(14, edges), laplacian(14, edges, b)
    rows = [(0, 3), (0, 2), (3, 2), (0, 1), (2, 1)]                   # their interconnections 1-4, 1-3, 4-3, 1-2, 3-2
    Tf = np.array([[1.0 if (zones[i], zones[j]) == r else -1.0 if (zones[j], zones[i]) == r else 0.0 for i, j in edges] for r in rows])
    ref = Tf @ ((b[:, None] * C.T) @ np.linalg.pinv(L) @ p)
    pz = np.bincount(zones, weights=p)
    nrmse = lambda f: np.sqrt(np.mean((ref - f) ** 2)) / np.mean(np.abs(ref))
    dep = np.array([[-0.138, -0.144, -0.532], [-0.582, -0.695, -0.450], [-0.138, -0.144, 0.468], [-0.278, -0.159, -0.017],
                    [-0.721, 0.159, 0.017]])
    ind = np.array([[-0.126, -0.143, -0.532], [-0.343, -0.676, -0.450], [-0.126, -0.143, 0.468], [-0.530, -0.179, -0.017],
                    [-0.469, 0.179, 0.017]])
    assert round(nrmse(dep @ pz[1:]), 3) == 0.038
    assert round(nrmse(ind @ pz[1:]), 4) == 0.0939                     # printed 0.093
    Hr, pairs = reduced_ptdf(14, edges, b, zones)
    k = [pairs.index((min(r), max(r))) for r in rows]; s = np.array([1.0 if r[0] < r[1] else -1.0 for r in rows])
    assert round(nrmse(s * (Hr[k] @ pz)), 4) == 0.0937
    bz = quotient_susceptances(zones, edges, b, pairs)
    Lz = laplacian(4, pairs, bz); th = np.linalg.pinv(Lz) @ pz
    fz = np.array([bz[m] * (th[i] - th[j]) for m, (i, j) in enumerate(pairs)])
    assert round(nrmse(s * fz[k]), 3) == 0.336                          # printed 0.33


def test_cheng_overbye_zonal_isfs_and_equivalent_reactances():
    """Cheng & Overbye 2005, Sec. VI (printed p. 1873). The six-bus system of Figs. 1 and 2 in four areas,
    Slack {0}, A {1, 2}, B {3}, C {4, 5}, interfaces A-B, A-Slack, B-C, B-Slack, C-Slack. Eqs. (35) and (36):
    the zonal ISFs for transfers from A, B and C to the slack, with bus 1 supplying all of A's transfer
    (Psi_1) or 25 % with bus 2 the rest (Psi_2), and C's split equally between buses 4 and 5 (not stated in
    the text, inferred from the numbers). Table I: the equivalent reactances of the four-node model, scaled
    so that B-Slack is 0.1, from `reweight_ptdf` against the zonal PTDF of each case."""
    from gridnet.reweight import quotient_susceptances, reduced_ptdf, reweight_ptdf

    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 5), (3, 5)]
    b = np.full(7, 10.0)
    H = ptdf_from_susceptances(6, edges, b)
    # interface flows from the line flows: A-B = 2->3, A-Slack = 1->0, B-C = 3->5, B-Slack = 3->0, C-Slack = 4->0
    T = np.zeros((5, 7)); T[0, 2] = 1; T[1, 0] = -1; T[2, 6] = 1; T[3, 3] = -1; T[4, 4] = -1
    zones = np.array([0, 1, 1, 2, 3, 3])
    printed = {0: (np.array([[.2667, -.2, -.1], [.7333, .2, .1], [.0667, .2, -.4], [.2, .6, .3], [.0667, .2, .6]]),
                   [0.200, 0.100, 0.150, 0.100, 0.150]),
               1: (np.array([[.4667, -.2, -.1], [.5333, .2, .1], [.1167, .2, -.4], [.35, .6, .3], [.1167, .2, .6]]),
                   [0.125, 0.175, 0.150, 0.100, 0.150])}
    for case, share_1 in ((0, 1.0), (1, 0.25)):
        w = np.array([1.0, share_1, 1.0 - share_1, 1.0, 0.5, 0.5])        # participation within each area
        Psi = np.zeros((5, 3))
        for col, area in enumerate((1, 2, 3)):
            p = np.where(zones == area, w, 0.0); p[0] = -1.0
            Psi[:, col] = T @ H @ p
        np.testing.assert_allclose(Psi, printed[case][0], atol=5e-5 + 1e-9)
        Hr, pairs = reduced_ptdf(6, edges, b, zones, weights=w)
        assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
        basis = np.eye(4)[:, 1:] - np.eye(4)[:, [0]]
        bz = reweight_ptdf(Hr @ basis, 4, pairs, quotient_susceptances(zones, edges, b, pairs), basis=basis)
        x = 1.0 / bz; x = 0.1 * x / x[1]                                  # B-Slack at 0.1
        order = [3, 0, 4, 1, 2]                                         # A-B, A-Slack, B-C, B-Slack, C-Slack
        np.testing.assert_allclose(x[order], printed[case][1], atol=5e-4 + 1e-9)


def test_cheng_overbye_tables_2_and_3_ward_and_ptdf_equivalents():
    """Cheng & Overbye 2005, Sec. VII (printed p. 1874), Fig. 5: the six-bus system of Figs. 1 and 2 with the
    branch 0-3 replaced by 2-5, all reactances 0.1, areas Slack {0}, A {1, 2}, B {3}, C {4, 5}, each area's
    transfer supplied by one generator (buses 1, 3, 5). Table II: the Ward equivalent (buses 2 and 4
    eliminated, `kron_reduce`), the PTDF-based equivalent of their eqs. (20)-(26) with A-S pinned, and the
    ISFs of a transfer from B to the slack; Table III those from A and from C. The tables truncate (16.67 is
    printed 16.6, the reactance 0.3598 is printed 0.359), so every value is checked to one unit of its last
    digit. The printed Ward ISFs of A to B and A to C for a transfer from B have the wrong sign (+33.3, +16.6):
    A has no injection in that transfer and sends 50 % to the slack, which conservation balances only with
    -33.3 and -16.7, and the errors of the text (4.8 % mean, 16.7 % largest) are those with the minus sign."""
    from gridnet.kron import kron_reduce
    from gridnet.reweight import quotient_susceptances, reweight_ptdf_equations

    edges = [(0, 1), (1, 2), (2, 3), (3, 5), (4, 5), (0, 4), (2, 5)]
    b = np.full(7, 10.0)
    zones = np.array([0, 1, 1, 2, 3, 3])
    w = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    # the interfaces in the order and direction of the tables: A to B, A to C, A to S, B to C, C to S
    branches = [(1, 2), (1, 3), (1, 0), (2, 3), (3, 0)]

    def isfs(n, edges_, b_, zone_of_bus, weights=None):
        """Percent flow over each table interface per transfer from A, B, C to the slack."""
        Hr, pairs = reduced_ptdf(n, edges_, b_, zone_of_bus, weights=weights)
        rows = [(1 if f < t else -1) * Hr[pairs.index((min(f, t), max(f, t)))] for f, t in branches]
        return 100 * np.array([[r[z] - r[0] for z in (1, 2, 3)] for r in rows])

    full = isfs(6, edges, b, zones, w)
    K = kron_reduce(laplacian(6, edges, b), [0, 1, 3, 5])                 # Ward: buses 2 and 4 eliminated
    ward_edges = [(i, j) for i in range(4) for j in range(i + 1, 4) if K[i, j] != 0]
    ward_b = np.array([-K[i, j] for i, j in ward_edges])
    ward = isfs(4, ward_edges, ward_b, np.arange(4))
    x_ward = {tuple(sorted(e)): 1 / v for e, v in zip(ward_edges, ward_b)}
    np.testing.assert_allclose([x_ward[tuple(sorted(e))] for e in branches], [0.300, 0.300, 0.100, 0.075, 0.200], atol=1e-12)

    b_new = reweight_ptdf_equations(full / 100, 4, branches, pin=2)          # eqs. (20)-(26), b of A to S = 1
    x_new = 0.1 / b_new                                                      # scaled so that A to S is 0.1
    new = isfs(4, [tuple(sorted(e)) for e in branches], b_new, np.arange(4))

    one_unit = lambda ours, printed, unit: np.testing.assert_array_less(np.abs(np.asarray(ours) - printed), unit)
    one_unit(x_new, [0.359, 0.308, 0.100, 0.315, 0.309], 1e-3)
    printed_full = [[7.1, -50.0, -14.3], [14.3, 0.0, -28.6], [78.6, 50.0, 42.9], [7.1, 50.0, -14.3], [21.4, 50.0, 57.1]]
    printed_ward = [[9.5, -33.3, -19.1], [11.9, -16.6, -23.8], [78.6, 50.0, 42.9], [9.5, 66.7, -19.1], [21.4, 50.0, 57.1]]
    printed_new = [[5.0, -52.6, -15.6], [11.1, -13.1, -34.2], [83.9, 65.7, 49.8], [5.0, 47.4, -15.6], [16.1, 34.3, 50.2]]
    one_unit(full, printed_full, 0.1)
    one_unit(ward, printed_ward, 0.1)                                         # B column: the sign corrected in rows 1, 2
    one_unit(new, printed_new, 0.1)
    # the text: Ward 4.8 % mean absolute error and 16.7 % largest, the new method 6.0 %
    assert round(np.abs(ward - full).mean(), 1) == 4.8 and round(np.abs(ward - full).max(), 1) == 16.7
    assert round(np.abs(new - full).mean(), 1) == 6.0
    # the minimizer of the PTDF distance itself (Fortenbacher et al. 2018, eq. 16) halves that error
    Hr, pairs = reduced_ptdf(6, edges, b, zones, weights=w)
    basis = np.eye(4)[:, 1:] - np.eye(4)[:, [0]]
    b_fro = reweight_ptdf(Hr @ basis, 4, pairs, quotient_susceptances(zones, edges, b, pairs), basis=basis)
    assert np.abs(isfs(4, pairs, b_fro, np.arange(4)) - full).mean() < 0.5 * np.abs(new - full).mean()


@pytest.mark.parametrize("pin", [0, 2])
def test_ptdf_equations_refuse_a_network_with_a_cut_node(pin):
    """Cheng & Overbye 2005 solve eq. (24) under the assumption that the stacked matrix has full rank, which
    fails at a cut node, where each block is determined only up to its own factor; they split the network
    there (Sec. V). On the path 0-1-2 joined at 2 to the triangle 2-3-4 (b = 1 to 5, the exact ISFs of the
    network itself, slack 0) the least squares returned zero or negative susceptances on the two bridges
    (pin 2) or zeros everywhere but the pinned bridge (pin 0) without a warning; the function now refuses."""
    from gridnet.reweight import reweight_ptdf_equations

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4)]
    b = np.arange(1.0, 6.0)
    Hr, pairs = reduced_ptdf(5, edges, b, np.arange(5))
    Psi = np.array([[Hr[pairs.index(e)][z] - Hr[pairs.index(e)][0] for z in range(1, 5)] for e in edges])
    with pytest.raises(ValueError, match="cut zone"):
        reweight_ptdf_equations(Psi, 5, edges, pin=pin)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ptdf_equations_refuse_a_cut_node_also_when_psi_is_not_an_exact_ptdf(seed):
    """The refusal above tested the numerical rank of the stacked matrix. A Psi that is not the exact PTDF of
    the zonal graph (as from a full network, here the exact one with 5 % noise) gives that matrix full rank,
    and the least squares returned negative susceptances on the bridges unrefused (audit of 2026-11-20, A3;
    all seven noisy cut-zone cases there). Whether a zone is a cut node is a property of the zonal graph."""
    from gridnet.reweight import reweight_ptdf_equations

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4)]
    b = np.arange(1.0, 6.0)
    Hr, pairs = reduced_ptdf(5, edges, b, np.arange(5))
    Psi = np.array([[Hr[pairs.index(e)][z] - Hr[pairs.index(e)][0] for z in range(1, 5)] for e in edges])
    Psi = Psi * (1 + 0.05 * np.random.default_rng(seed).standard_normal(Psi.shape))
    with pytest.raises(ValueError, match="cut zone"):
        reweight_ptdf_equations(Psi, 5, edges)


def test_the_pin_of_cheng_and_overbye_changes_the_equivalent():
    """`reweight_ptdf_equations` minimizes the residual of eqs. (20)-(26), not the distance of the ISFs, so the
    pinned branch changes the equivalent where Psi cannot be matched: on their Fig. 5 system the reactances of
    the five choices of the pin, each scaled to the same geometric mean, differ by up to 21.3 % (21.6 % scaled
    to the same sum), the largest on C-Slack."""
    from gridnet.reweight import reweight_ptdf_equations

    edges = [(0, 1), (1, 2), (2, 3), (3, 5), (4, 5), (0, 4), (2, 5)]
    zones = np.array([0, 1, 1, 2, 3, 3]); w = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    branches = [(1, 2), (1, 3), (1, 0), (2, 3), (3, 0)]
    Hr, pairs = reduced_ptdf(6, edges, np.full(7, 10.0), zones, weights=w)
    rows = [(1 if f < t else -1) * Hr[pairs.index((min(f, t), max(f, t)))] for f, t in branches]
    psi = np.array([[r[z] - r[0] for z in (1, 2, 3)] for r in rows])
    X = np.array([1 / reweight_ptdf_equations(psi, 4, branches, pin=pin) for pin in range(5)])
    geo = X / np.exp(np.log(X).mean(axis=1, keepdims=True))
    spread = (geo.max(0) - geo.min(0)) / geo.min(0)
    assert round(100 * spread.max(), 1) == 21.3 and np.argmax(spread) == 4
    tot = X / X.sum(axis=1, keepdims=True)
    assert round(100 * ((tot.max(0) - tot.min(0)) / tot.min(0)).max(), 1) == 21.6


@pytest.mark.parametrize("fn", ["reduced_ptdf", "zonal_operator"])
def test_zone_weights_must_be_nonnegative_and_not_all_zero_in_a_zone(fn):
    """The docstrings say nonnegative; a zone of zero weights divided by zero and gave NaN (code/01 first #14)."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    b = np.ones(4)
    zones = np.array([0, 0, 1, 1])

    def call(w):
        if fn == "reduced_ptdf":
            return reduced_ptdf(4, edges, b, zones, weights=w)
        return zonal_operator(laplacian(4, edges, b), zones, weights=w)

    call(np.array([1.0, 0.0, 0.5, 0.5]))                                # one bus per zone may carry it all
    with pytest.raises(ValueError, match="nonnegative"):
        call(np.array([1.5, -0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="zone 1"):
        call(np.array([0.5, 0.5, 0.0, 0.0]))
    with pytest.raises(ValueError, match="one .* per bus"):
        call(np.ones(3))


def test_degenerate_inputs_of_the_spectral_fits_are_refused():
    """`spectral_error_best_scale` refuses an approximation that is zero on the range of L (the factor
    2 / (mu_min + mu_max) would be 0/0) or indefinite there (a negative mu_min, which used to be clipped to 0
    and read as error one); `best_scales` refuses a component that is zero on that range; `reweight_ptdf`
    refuses a zone pair listed twice (its index would keep the last copy only)."""
    from gridnet.graph import laplacian
    from gridnet.reweight import best_scales, reweight_ptdf
    from gridnet.sparsify import spectral_error_best_scale

    edges = [(0, 1), (1, 2), (2, 0)]
    L = laplacian(3, edges)
    with pytest.raises(ValueError, match="zero"):
        spectral_error_best_scale(L, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="indefinite"):
        spectral_error_best_scale(L, laplacian(3, edges, np.array([1.0, 1.0, -2.0])))
    with pytest.raises(ValueError, match="zero"):
        best_scales(L, [laplacian(3, [(0, 1)]), np.zeros((3, 3))])
    with pytest.raises(ValueError, match="twice"):
        reweight_ptdf(np.zeros((4, 3)), 3, [(0, 1), (1, 2), (2, 0), (1, 0)], np.ones(4))


def test_ptdf_fit_survives_a_trial_point_whose_laplacian_cannot_be_resolved(monkeypatch):
    """The fit optimizes the logarithms of the susceptances without bounds, and a trial step of
    the trust-region method can land on susceptances thirty orders of magnitude apart (seen on
    the 96 zones of the European grid: 2.2e30). The Laplacian there has no pseudoinverse to
    any digit (`gridnet.graph.IllConditionedLaplacian`), so the point is no candidate: its
    residual is a finite penalty far above any real one, which makes the method reject the step
    and shrink its region. Here the optimizer is made to try such a point first: the residual is
    finite and at least 1e50, and the fit that follows is the one without the detour."""
    import scipy.optimize

    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]
    b_true = np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    H = ptdf_from_susceptances(4, edges, b_true)
    reference = np.array([1.0, 1.0, 1.0, 1.0, 3.0])            # pinned at the strongest edge, at its true value
    plain = reweight_ptdf(H, 4, edges, reference)
    np.testing.assert_allclose(plain, b_true, rtol=1e-6)
    real, seen = scipy.optimize.least_squares, {}

    def detour(residual, x0, **kw):
        far = np.asarray(x0, dtype=float).copy(); far[0] = np.log(1e30)
        seen["r"] = residual(far)
        return real(residual, x0, **kw)

    monkeypatch.setattr(scipy.optimize, "least_squares", detour)
    np.testing.assert_allclose(reweight_ptdf(H, 4, edges, reference), plain, rtol=1e-12)
    assert np.isfinite(seen["r"]).all() and np.abs(seen["r"]).min() >= 1e50
