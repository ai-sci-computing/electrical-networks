"""Tests for gridnet.powerflow: Newton-Raphson AC power flow in polar coordinates.

Reference: MATPOWER User's Manual 8.0 (Zimmerman & Murillo-Sanchez 2024), Sec. 4.1
(Newton's method for the AC power flow, eqs. 4.1-4.10 for the mismatch and Jacobian),
Zimmerman et al. 2011, Sec. III.C. Bus types (manual Sec. 3.1): PQ buses have P and Q
specified, PV buses P and |V|, the reference bus |V| and angle.
Closed form: a two-bus system (reference + PQ load through a lossless line) has the
quadratic solution |V|^2 known in closed form. Machine precision against MATPOWER's own
Newton-Raphson (via its Python port PYPOWER) on MATPOWER cases 9 to 300.
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.admittance import bus_admittance
from gridnet.network import Network, read_matpower_case, read_matpower_dict
from gridnet.powerflow import newton_raphson

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


def two_bus(p_load, q_load, x=0.1):
    return Network(
        base_mva=100.0, bus_id=np.array([1, 2]), bus_type=np.array([3, 1]),
        pd=np.array([0.0, p_load]), qd=np.array([0.0, q_load]), gs=np.zeros(2), bs=np.zeros(2),
        from_bus=np.array([0]), to_bus=np.array([1]), r=np.array([0.0]), x=np.array([x]), b=np.array([0.0]),
        tap=np.ones(1), shift=np.zeros(1), status=np.ones(1, dtype=int), rate_a=np.zeros(1),
    )


def test_two_bus_closed_form():
    """Reference bus at 1 p.u., load P + jQ through reactance x: with u = |V_2|^2,
    u^2 - (1 - 2 Q x) u + x^2 (P^2 + Q^2) = 0 (from |S| = |V_2| |I| and the voltage drop);
    the high-voltage root is the operating solution."""
    p, q, x = 0.4, 0.2, 0.1
    net = two_bus(p, q, x)
    sol = newton_raphson(net, vm0=np.ones(2), va0=np.zeros(2))
    assert sol.converged
    a = 1 - 2 * q * x
    u_hi = (a + np.sqrt(a**2 - 4 * x**2 * (p**2 + q**2))) / 2
    assert np.isclose(sol.vm[1] ** 2, u_hi, atol=1e-12)
    # angle from P = |V1||V2| sin(theta_12)/x
    assert np.isclose(np.sin(sol.va[0] - sol.va[1]), p * x / sol.vm[1], atol=1e-12)


def test_two_bus_beyond_the_nose_does_not_converge():
    """No real solution when the discriminant is negative: the load exceeds the
    maximum transfer (the 'nose' of the P-V curve)."""
    p, q, x = 3.0, 1.0, 0.5
    a = 1 - 2 * q * x
    assert a**2 - 4 * x**2 * (p**2 + q**2) < 0
    sol = newton_raphson(two_bus(p, q, x), vm0=np.ones(2), va0=np.zeros(2), max_iter=30)
    assert not sol.converged


@pytest.mark.data("matpower")
@pytest.mark.parametrize("case", ["case9", "case14", "case30", "case57", "case118", "case300"])
def test_matches_matpower_algorithm_via_pypower(case):
    """PYPOWER's runpf is MATPOWER's Newton power flow ported to Python, operating on the
    raw case dict without any model conversion: the reference for our implementation."""
    pypower = pytest.importorskip("pypower.api")
    from gridnet.network import from_ppc, read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(CASES / f"{case}.m").items() if v is not None}
    res, ok = pypower.runpf(ppc, pypower.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-12, PF_MAX_IT=30))
    assert ok
    net = from_ppc(ppc)
    ext2int = {int(b): i for i, b in enumerate(ppc["bus"][:, 0])}
    vset = np.ones(net.n_bus)
    for row in ppc["gen"]:
        if row[7] > 0:
            vset[ext2int[int(row[0])]] = row[5]
    ref = int(np.flatnonzero(net.bus_type == 3)[0])
    sol = newton_raphson(net, vset=vset, tol=1e-12, theta_ref=np.deg2rad(ppc["bus"][ref, 8]))
    assert sol.converged
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-10)


@pytest.mark.data("matpower")
def test_power_balance_at_the_solution():
    net = read_matpower_case(CASES / "case14.m")
    vset = np.ones(net.n_bus); vset[[0, 1, 2, 5, 7]] = [1.06, 1.045, 1.01, 1.07, 1.09]
    sol = newton_raphson(net, vset=vset, tol=1e-12)
    V = sol.vm * np.exp(1j * sol.va)
    S = V * np.conj(bus_admittance(net) @ V)                # eq. (3.14)
    pq = net.bus_type == 1
    # shunts are part of Y_bus, so the injection equals generation minus demand exactly
    np.testing.assert_allclose(S.real[pq], -net.pd[pq], atol=1e-10)
    np.testing.assert_allclose(S.imag[pq], -net.qd[pq], atol=1e-10)
    pv = net.bus_type == 2
    np.testing.assert_allclose(S.real[pv], net.pg[pv] - net.pd[pv], atol=1e-10)


def _ppc_and_vset(case):
    """A raw case dict for PYPOWER and the voltage setpoints per bus for our solver."""
    from gridnet.network import from_ppc, read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(CASES / f"{case}.m").items() if v is not None}
    net = from_ppc(ppc)
    ext2int = {int(b): i for i, b in enumerate(ppc["bus"][:, 0])}
    vset = np.ones(net.n_bus)
    for row in ppc["gen"]:
        if row[7] > 0:
            vset[ext2int[int(row[0])]] = row[5]
    return ppc, net, vset


@pytest.mark.data("matpower")
def test_sparse_jacobian_matches_dense():
    """The sparse path (scipy.sparse Jacobian and solve) is the same Newton iteration as the
    dense one: same iterates to rounding, on the 118-bus system."""
    _, net, vset = _ppc_and_vset("case118")
    dense = newton_raphson(net, vset=vset, tol=1e-12, sparse=False)
    sparse = newton_raphson(net, vset=vset, tol=1e-12, sparse=True)
    assert dense.converged and sparse.converged
    assert dense.iterations == sparse.iterations
    np.testing.assert_allclose(sparse.vm, dense.vm, atol=1e-12)
    np.testing.assert_allclose(sparse.va, dense.va, atol=1e-12)
    np.testing.assert_allclose(sparse.qg, dense.qg, atol=1e-10)


@pytest.mark.data("matpower")
@pytest.mark.parametrize("case", ["case2383wp", "case9241pegase"])
def test_large_cases_match_pypower(case):
    """The Polish grid (2,383 buses) and PEGASE (9,241 buses, Josz et al. 2016) against
    PYPOWER; the dense Jacobian of PEGASE would need 1.4 GB, so these run sparse by default."""
    pypower = pytest.importorskip("pypower.api")
    ppc, net, vset = _ppc_and_vset(case)
    # 1e-12 is below rounding on these cases (the mismatch stalls near 5e-12 in both solvers)
    res, ok = pypower.runpf(ppc, pypower.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=30))
    assert ok
    ref = int(np.flatnonzero(net.bus_type == 3)[0])
    sol = newton_raphson(net, vset=vset, tol=1e-10, theta_ref=np.deg2rad(ppc["bus"][ref, 8]))
    assert sol.converged
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-8)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-6)


def _pypower_against_ours(ppc, tol):
    """Our power flow at the case's own setpoints against PYPOWER's runpf on the same dict."""
    pypower = pytest.importorskip("pypower.api")
    from gridnet.network import from_ppc

    res, ok = pypower.runpf({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()},
                            pypower.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=tol, PF_MAX_IT=30))
    assert ok
    net = from_ppc(ppc)
    ref = int(np.flatnonzero(net.bus_type == 3)[0])
    sol = newton_raphson(net, tol=tol, theta_ref=np.deg2rad(ppc["bus"][ref, 8]))
    assert sol.converged
    return net, sol, res


@pytest.mark.data("matpower")
def test_a_generator_bus_without_a_generator_in_service_is_a_load_bus():
    """A PV bus whose generators are all out of service holds no voltage: MATPOWER's bustypes
    (Zimmerman et al. 2011; PYPOWER's bustypes.py is the port) counts a bus as PV or reference
    only if a generator is in service there, and treats the rest as PQ buses. Case14 with the
    synchronous condenser at bus 6 switched off: bus 6 must float, as in PYPOWER."""
    from gridnet.network import read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(CASES / "case14.m").items() if v is not None}
    ppc["gen"][ppc["gen"][:, 0] == 6, 7] = 0
    net, sol, res = _pypower_against_ours(ppc, 1e-12)
    assert net.bus_type[5] == 1
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-10)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-8)


@pytest.mark.data("matpower")
def test_a_reference_bus_without_a_generator_hands_the_reference_to_the_first_pv_bus():
    """MATPOWER's bustypes: with the reference bus's generators out, the first PV bus (in bus
    order) becomes the reference. Case14 with the generator at bus 1 switched off: bus 2
    takes over, and the solution matches PYPOWER's."""
    from gridnet.network import read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(CASES / "case14.m").items() if v is not None}
    ppc["gen"][ppc["gen"][:, 0] == 1, 7] = 0
    ppc["bus"][1, 8] = 0.0      # PYPOWER keeps the new reference's angle from the bus table
    net, sol, res = _pypower_against_ours(ppc, 1e-12)
    assert net.bus_type[0] == 1 and net.bus_type[1] == 3
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-10)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-8)


@pytest.mark.data("matpower")
def test_a_generator_at_a_load_bus_injects_its_active_power_and_a_reactive_one_is_refused():
    """MATPOWER injects a generator in service at a PQ bus as a fixed Pg + j Qg (makeSbus, with the
    bus types of bustypes). The network carries the active part per bus; a nonzero QG there would be
    dropped, so the reader refuses it. Case14 with bus 6 typed PQ: with the condenser's 12.2 MVAr the
    case is refused (the voltages would differ from PYPOWER's by 0.024 per unit), with QG = 0 and
    20 MW of PG the solution is PYPOWER's. A bus typed isolated (4) with its branches out leaves the
    rest as in PYPOWER, which drops the bus."""
    from gridnet.network import from_ppc, read_matpower_dict

    def case14():
        return {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
                for k, v in read_matpower_dict(CASES / "case14.m").items() if v is not None}
    ppc = case14()
    ppc["bus"][5, 1] = 1
    with pytest.raises(ValueError, match="PQ bus"):
        from_ppc(ppc)
    at6 = ppc["gen"][:, 0] == 6
    ppc["gen"][at6, 2] = 0.0; ppc["gen"][at6, 1] = 20.0
    net, sol, res = _pypower_against_ours(ppc, 1e-12)
    assert net.bus_type[5] == 1 and net.pg[5] == pytest.approx(0.2)
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-10)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-8)

    ppc = case14()
    ppc["bus"][13, 1] = 4
    ppc["branch"][(ppc["branch"][:, 0] == 14) | (ppc["branch"][:, 1] == 14), 10] = 0
    net, sol, res = _pypower_against_ours(ppc, 1e-12)
    keep = np.arange(13)
    np.testing.assert_allclose(sol.vm[keep], res["bus"][keep, 7], atol=1e-10)
    np.testing.assert_allclose(np.rad2deg(sol.va[keep]), res["bus"][keep, 8], atol=1e-8)


@pytest.mark.data("matpower")
def test_an_isolated_bus_takes_its_branches_and_generators_out_of_service():
    """MATPOWER removes a bus typed isolated (4) from the case together with every branch and
    every generator at it (ext2int; PYPOWER's ext2int.py: gen status and branch status are
    and-ed with the status of their buses), whatever their own status says. Case14 with bus 8
    typed 4 while branch 7-8 and the condenser at bus 8 stay in service in the tables: the
    reader switches both off and drops the bus's load and shunt, and the solution on the other
    buses is PYPOWER's. Left in service, the branch would tie bus 7 to a bus held at 1.0 per
    unit and angle zero, a second reference: 1.0236 per unit at bus 7 where PYPOWER has 1.0365."""
    from gridnet.network import read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(CASES / "case14.m").items() if v is not None}
    ppc["bus"][7, 1] = 4
    ppc["bus"][7, 2:6] = [5.0, 2.0, 1.0, 3.0]                   # a load and a shunt at the isolated bus
    link = np.flatnonzero((ppc["branch"][:, 0] == 7) & (ppc["branch"][:, 1] == 8))
    assert link.size == 1 and ppc["branch"][link[0], 10] == 1 and ppc["gen"][ppc["gen"][:, 0] == 8, 7].all()
    net, sol, res = _pypower_against_ours(ppc, 1e-12)
    assert net.bus_type[7] == 4 and net.status[link[0]] == 0 and net.status.sum() == len(net.status) - 1
    assert net.pg[7] == 0 and np.isnan(net.vg[7])
    assert net.pd[7] == 0 and net.qd[7] == 0 and net.gs[7] == 0 and net.bs[7] == 0
    keep = np.arange(14) != 7
    np.testing.assert_allclose(sol.vm[keep], res["bus"][keep, 7], atol=1e-10)
    np.testing.assert_allclose(np.rad2deg(sol.va[keep]), res["bus"][keep, 8], atol=1e-8)


_TEXAS = Path(__file__).resolve().parents[2] / "data" / "ACTIVSg2000" / "case_ACTIVSg2000.m"


@pytest.mark.data("ACTIVSg2000")
def test_texas_matches_pypower():
    """The synthetic Texas grid ACTIVSg2000 (Birchfield et al. 2017) has 93 PV buses whose
    generators are all out of service; with them treated as PQ buses the solution is PYPOWER's."""
    from gridnet.network import read_matpower_dict

    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v)
           for k, v in read_matpower_dict(_TEXAS).items() if v is not None}
    net, sol, res = _pypower_against_ours(ppc, 1e-10)
    assert int(np.sum(net.bus_type == 2)) == int(np.sum(ppc["bus"][:, 1] == 2)) - 93
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-8)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-6)


CASES_ON_DISK = ["case4gs", "case5", "case6ww", "case9", "case14", "case_ieee30", "case30", "case39", "case57", "case118",
                 "case300", "case2383wp", "case9241pegase"]


@pytest.mark.data("matpower")
@pytest.mark.parametrize("case", CASES_ON_DISK)
def test_default_power_flow_matches_pypower_in_every_quantity(case):
    """The power flow as the projects call it, `newton_raphson(net)` with the setpoints the reader derives
    (the path where PV buses without a generator in service once kept their setpoint), against PYPOWER's
    runpf on every readable MATPOWER case on disk: voltage magnitudes and angles, the reactive output of
    every generator bus, the active output at the reference bus, and the complex flows at both ends of
    every branch. The earlier comparisons built the setpoints themselves and compared magnitudes only.
    (PYPOWER's own reactive split is nan where a generator's limits are equal or infinite; those buses
    are skipped.)"""
    pypower = pytest.importorskip("pypower.api")
    from pypower.idx_brch import PF, PT, QF, QT
    from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG

    from gridnet.admittance import bus_admittance

    ppc = read_matpower_dict(CASES / f"{case}.m")
    net = read_matpower_case(CASES / f"{case}.m")
    res, ok = pypower.runpf({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()},
                            pypower.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=30))
    assert ok
    ref = int(np.flatnonzero(net.bus_type == 3)[0])
    sol = newton_raphson(net, tol=1e-10, theta_ref=np.deg2rad(ppc["bus"][ref, 8]))
    assert sol.converged
    base = net.base_mva
    np.testing.assert_allclose(sol.vm, res["bus"][:, 7], atol=1e-8)
    np.testing.assert_allclose(np.rad2deg(sol.va), res["bus"][:, 8], atol=1e-6)
    number = {int(b): i for i, b in enumerate(net.bus_id)}
    g = res["gen"][res["gen"][:, GEN_STATUS] > 0]
    qg = np.zeros(net.n_bus); pg = np.zeros(net.n_bus)
    for row in g:
        qg[number[int(row[GEN_BUS])]] += row[QG]; pg[number[int(row[GEN_BUS])]] += row[PG]
    gen_bus = net.bus_type >= 2
    # PYPOWER's pfsoln shares a bus's reactive output in proportion to Q_max - Q_min and returns nan for a
    # generator whose two limits are equal or infinite (six with +-Inf on the Polish grid); only those
    # buses are left out
    from pypower.idx_gen import QMAX, QMIN
    equal = {number[int(r[GEN_BUS])] for r in g if r[QMAX] == r[QMIN] or not np.isfinite(r[QMAX] - r[QMIN])}
    assert all(i in equal for i in np.flatnonzero(np.isnan(qg)))
    keep = gen_bus & ~np.isnan(qg)
    np.testing.assert_allclose(sol.qg[keep] * base, qg[keep], atol=1e-5)
    assert sol.pg_ref * base == pytest.approx(pg[ref], abs=1e-5)
    _Y, Yf, Yt = bus_admittance(net, return_branch_matrices=True, sparse=net.n_bus > 1000)
    V = sol.vm * np.exp(1j * sol.va)
    Sf = V[net.from_bus] * np.conj(Yf @ V) * base
    St = V[net.to_bus] * np.conj(Yt @ V) * base
    on = net.status > 0
    np.testing.assert_allclose(Sf[on], (res["branch"][:, PF] + 1j * res["branch"][:, QF])[on], atol=1e-6)
    np.testing.assert_allclose(St[on], (res["branch"][:, PT] + 1j * res["branch"][:, QT])[on], atol=1e-6)
