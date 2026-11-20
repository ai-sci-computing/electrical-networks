"""Tests for gridnet.admittance: bus admittance matrix, DC model, lossless identities.

References (MATPOWER User's Manual 8.0, Zimmerman & Murillo-Sanchez 2024):
  branch admittance matrix with tap and shift   eq. (3.2)
  connection matrices Cf, Ct                    Sec. 3.2
  Yf, Yt, Ybus assembly                         eqs. (3.11)-(3.13)
  DC model: b = 1/(x tau), shift injections     eqs. (3.24), (3.27)-(3.31)
Machine-precision comparison against pandapower's internal Ybus (Thurner et al. 2018),
which implements the same MATPOWER-derived formulas.
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.admittance import bus_admittance, dc_model, magnetic_laplacian
from gridnet.graph import laplacian
from gridnet.network import Network, from_ppc, read_matpower_case, read_matpower_dict

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


def two_bus(r=0.0, x=0.1, b=0.0, tap=1.0, shift=0.0, gs=(0.0, 0.0), bs=(0.0, 0.0)):
    """A single branch 0 -> 1 with the given parameters (per-unit, shift in degrees)."""
    return Network(
        base_mva=100.0,
        bus_id=np.array([1, 2]), bus_type=np.array([3, 1]),
        pd=np.zeros(2), qd=np.zeros(2), gs=np.array(gs), bs=np.array(bs),
        from_bus=np.array([0]), to_bus=np.array([1]),
        r=np.array([r]), x=np.array([x]), b=np.array([b]),
        tap=np.array([tap]), shift=np.array([shift]),
        status=np.array([1]), rate_a=np.array([0.0]),
    )


# --- closed-form: the 2x2 branch matrix of eq. (3.2) ------------------------

def test_two_bus_ybus_matches_manual_eq_3_2():
    r, x, b, tap, shift_deg = 0.01, 0.1, 0.02, 1.05, 10.0
    net = two_bus(r, x, b, tap, shift_deg)
    ys = 1 / (r + 1j * x)
    t = tap * np.exp(1j * np.deg2rad(shift_deg))
    expected = np.array([
        [(ys + 1j * b / 2) / tap**2, -ys / np.conj(t)],
        [-ys / t, ys + 1j * b / 2],
    ])
    np.testing.assert_allclose(bus_admittance(net), expected, rtol=0, atol=1e-15)


def test_shunts_add_to_the_diagonal_only():
    net = two_bus(gs=(0.05, 0.0), bs=(0.0, -0.2))
    base = bus_admittance(two_bus())
    Y = bus_admittance(net)
    np.testing.assert_allclose(Y - base, np.diag([0.05, -0.2j]), atol=1e-15)


def test_out_of_service_branch_contributes_nothing():
    net = two_bus()
    net.status[:] = 0
    np.testing.assert_allclose(bus_admittance(net), np.zeros((2, 2)), atol=1e-15)


# --- machine precision against pandapower --------------------------------------

@pytest.mark.parametrize("case", ["case14", "case30", "case39", "case57", "case118", "case300"])
def test_ybus_matches_pandapower(case):
    """Same Ybus as pandapower's makeYbus (a port of MATPOWER's) on the same tables.

    pandapower extends the MATPOWER branch model by a per-branch shunt conductance
    column BR_G (transformer iron losses / magnetizing current); it is nonzero on a few
    branches of case118 and case300 and has no counterpart in eq. (3.2). The reference
    is therefore built with that column zeroed, which is the MATPOWER model exactly.
    """
    pp = pytest.importorskip("pandapower")
    pn = pytest.importorskip("pandapower.networks")
    from pandapower.pypower.idx_brch import BR_G
    from pandapower.pypower.makeYbus import makeYbus

    net_pp = getattr(pn, case)()
    pp.runpp(net_pp, numba=False)
    ppc = net_pp._ppc
    branch = ppc["branch"].copy()
    branch[:, BR_G] = 0.0
    Y_ref, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], branch)
    net = from_ppc({"baseMVA": ppc["baseMVA"], "bus": ppc["bus"], "branch": branch})
    np.testing.assert_allclose(bus_admittance(net), Y_ref.toarray(), rtol=1e-12, atol=1e-12)


@pytest.mark.data("matpower")
def test_reader_and_pandapower_agree_on_case14():
    """The .m reader and pandapower's own case14 give the same Ybus (bus order 1..14)."""
    pp = pytest.importorskip("pandapower")
    pn = pytest.importorskip("pandapower.networks")
    net_pp = pn.case14()
    pp.runpp(net_pp, numba=False)
    Y_ref = net_pp._ppc["internal"]["Ybus"].toarray()
    net = read_matpower_case(CASES / "case14.m")
    np.testing.assert_allclose(bus_admittance(net), Y_ref, rtol=1e-12, atol=1e-12)


# --- DC model, eqs. (3.24), (3.27)-(3.31) ---------------------------------------

def test_two_bus_dc_model_with_tap_and_shift():
    x, tap, shift_deg = 0.2, 1.1, 5.0
    net = two_bus(x=x, tap=tap, shift=shift_deg)
    Bbus, Bf, Pbus_shift, Pf_shift = dc_model(net)
    b = 1 / (x * tap)
    np.testing.assert_allclose(Bbus, b * np.array([[1, -1], [-1, 1]]))
    np.testing.assert_allclose(Bf, b * np.array([[1, -1]]))
    np.testing.assert_allclose(Pf_shift, [-np.deg2rad(shift_deg) * b])
    np.testing.assert_allclose(Pbus_shift, [-np.deg2rad(shift_deg) * b, np.deg2rad(shift_deg) * b])


@pytest.mark.data("matpower")
def test_dc_bbus_is_the_graph_laplacian_with_weights_one_over_x_tau():
    net = read_matpower_case(CASES / "case57.m")     # has 15 transformers
    Bbus, _, _, _ = dc_model(net)
    L = laplacian(net.n_bus, net.edges, net.status / (net.x * net.tap))
    np.testing.assert_allclose(Bbus, L, atol=1e-13)


def test_dc_bbus_matches_pandapower_dc_power_flow():
    pp = pytest.importorskip("pandapower")
    pn = pytest.importorskip("pandapower.networks")
    net_pp = pn.case118()
    pp.rundcpp(net_pp, numba=False)
    B_ref = net_pp._ppc["internal"]["Bbus"].toarray()
    net = from_ppc(net_pp._ppc)
    Bbus, _, _, _ = dc_model(net)
    np.testing.assert_allclose(Bbus, B_ref, rtol=1e-12, atol=1e-12)


# --- lossless identities and the magnetic structure -----------------------------

@pytest.mark.data("matpower")
def test_lossless_identity_im_ybus_is_minus_laplacian():
    """For r = 0, b = 0, no shunts, tau = 1, shift = 0: Im(Ybus) = -L(1/x) = -Bbus."""
    net = read_matpower_case(CASES / "case14.m").lossless()
    Y = bus_admittance(net)
    L = laplacian(net.n_bus, net.edges, 1 / net.x)
    np.testing.assert_allclose(Y.real, 0, atol=1e-15)
    np.testing.assert_allclose(Y.imag, -L, atol=1e-13)
    np.testing.assert_allclose(dc_model(net)[0], L, atol=1e-13)


@pytest.mark.data("matpower")
def test_magnetic_laplacian_is_hermitian_iff_lossless():
    net = read_matpower_case(CASES / "case14.m").lossless()
    net.shift[:3] = [10.0, -20.0, 35.0]      # add phase shifters: a U(1) connection
    net.tap[:3] = [1.05, 0.95, 1.1]
    M = magnetic_laplacian(net)
    np.testing.assert_allclose(M, M.conj().T, atol=1e-13)
    # with no shifters and unit taps it reduces to the real Laplacian L(1/x)
    net.shift[:] = 0.0
    net.tap[:] = 1.0
    np.testing.assert_allclose(magnetic_laplacian(net), laplacian(net.n_bus, net.edges, 1 / net.x), atol=1e-13)
    # with resistance it is not Hermitian and the function refuses
    lossy = read_matpower_case(CASES / "case14.m")
    with pytest.raises(ValueError):
        magnetic_laplacian(lossy)


def test_magnetic_laplacian_refuses_charging_and_shunt_susceptance():
    """With r = gs = 0 the matrix j Y_bus is Hermitian, but charging and shunt susceptance enter its
    diagonal as -b/(2 tau^2), -b/2 and -bs (eqs. (3.2), (3.13)), and then it is not positive
    semidefinite. Closed form on one branch with x = 0.1, b = 3, unit tap: M = [[10 - 1.5, -10],
    [-10, 10 - 1.5]], eigenvalues -1.5 and 18.5."""
    M = 1j * bus_admittance(two_bus(x=0.1, b=3.0))
    np.testing.assert_allclose(np.linalg.eigvalsh(M), [-1.5, 18.5], atol=1e-12)
    for net in (two_bus(x=0.1, b=3.0), two_bus(x=0.1, bs=(0.0, 0.2))):
        with pytest.raises(ValueError, match="b = 0, bs = 0"):
            magnetic_laplacian(net)


@pytest.mark.data("matpower")
def test_magnetic_laplacian_is_positive_semidefinite_with_taps_and_shifts():
    """Without charging and shunts M = j Y_bus = sum over branches of a rank-one Hermitian term
    (1/x) c c^*, c = e_f / tau - e^{-j shift} e_t, hence positive semidefinite (it is a Laplacian with
    U(1) connection and tap weights); with the charging and the bus shunt of case14 kept it is not (min eig -0.029; -0.0153 with the
    charging alone)."""
    net = read_matpower_case(CASES / "case14.m").lossless()
    net.shift[:3] = [10.0, -20.0, 35.0]
    net.tap[:3] = [1.05, 0.95, 1.1]
    assert np.linalg.eigvalsh(magnetic_laplacian(net)).min() > -1e-12
    charged = read_matpower_case(CASES / "case14.m")
    charged.r[:] = 0.0; charged.gs[:] = 0.0
    assert np.linalg.eigvalsh(1j * bus_admittance(charged)).min() < -0.02
    with pytest.raises(ValueError):
        magnetic_laplacian(charged)


@pytest.mark.data("matpower")
def test_sparse_assembly_equals_dense():
    """The sparse paths of bus_admittance and dc_model assemble the same entries from the branch list as the
    dense ones (makeYbus and makeBdc), here on case14 with phase shifters and an out-of-service branch."""
    net = read_matpower_case(CASES / "case14.m")
    net.shift[:3] = [10.0, -20.0, 35.0]
    net.status[5] = 0
    for dense, sparse in zip(bus_admittance(net, return_branch_matrices=True),
                             bus_admittance(net, return_branch_matrices=True, sparse=True)):
        np.testing.assert_allclose(sparse.toarray(), dense, atol=1e-14)
    for dense, sparse in zip(dc_model(net), dc_model(net, sparse=True)):
        np.testing.assert_allclose(sparse.toarray() if hasattr(sparse, "toarray") else sparse, dense, atol=1e-14)


@pytest.mark.data("matpower")
@pytest.mark.parametrize("case", ["case4gs", "case5", "case6ww", "case9", "case14", "case_ieee30", "case30", "case39",
                                  "case57", "case118", "case300", "case1354pegase", "case2383wp", "case9241pegase"])
def test_ybus_and_dc_model_against_pypower_on_the_case_files(case):
    """MATPOWER's makeYbus and makeBdc (PYPOWER's ports) on the tables our reader produced, after
    PYPOWER's ext2int, against `bus_admittance` and `dc_model` of the same file: Ybus, Yf, Yt, Bbus, Bf
    and the phase-shift injections Pbus_shift and Pf_shift (case2383wp has six shifters, case300 a series
    capacitor, the PEGASE cases negative resistances and reactances). With the first branch taken out of
    service (ext2int then drops it) the bus matrices agree as well. The earlier test above uses
    pandapower's bundled cases, which are not MATPOWER's files (its case118 differs near buses 68, 81, 86)."""
    ext2int = pytest.importorskip("pypower.ext2int").ext2int
    makeBdc = pytest.importorskip("pypower.makeBdc").makeBdc
    makeYbus = pytest.importorskip("pypower.makeYbus").makeYbus

    d = read_matpower_dict(CASES / f"{case}.m")
    for out in (False, True):
        ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v) for k, v in d.items() if v is not None}
        if out:
            ppc["branch"][0, 10] = 0                                       # BR_STATUS
        net = from_ppc({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()})
        i2e = ext2int({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()})
        Y, Yf, Yt = makeYbus(i2e["baseMVA"], i2e["bus"], i2e["branch"])
        B, Bf, Pbusinj, Pfinj = makeBdc(i2e["baseMVA"], i2e["bus"], i2e["branch"])
        Y_ours, Yf_ours, Yt_ours = bus_admittance(net, return_branch_matrices=True, sparse=True)
        Bbus, Bf_ours, Pbus_shift, Pf_shift = dc_model(net, sparse=True)
        keep = np.flatnonzero(net.status != 0)

        def gap(ours, theirs):                                            # largest entry of the difference, relative
            diff = (ours - theirs).tocoo()
            return (np.abs(diff.data).max() if diff.nnz else 0.0) / abs(theirs).max()

        assert gap(Y_ours, Y) < 1e-12 and gap(Bbus, B) < 1e-12
        assert gap(Yf_ours[keep], Yf) < 1e-12 and gap(Yt_ours[keep], Yt) < 1e-12 and gap(Bf_ours[keep], Bf) < 1e-12
        np.testing.assert_allclose(Pbus_shift, Pbusinj, atol=1e-10)                  # both per unit
        np.testing.assert_allclose(np.asarray(Pf_shift)[keep], Pfinj, atol=1e-10)


@pytest.mark.data("matpower")
def test_ybus_and_dc_model_of_a_network_in_two_islands_against_pypower():
    """case9 with the lines 5-6 and 8-9 out of service falls into the islands {1, 4, 5, 9} and
    {2, 3, 6, 7, 8}; the bus matrices do not depend on connectivity, and `bus_admittance` and
    `dc_model` equal MATPOWER's makeYbus and makeBdc (PYPOWER's ports) after ext2int, block diagonal
    in the two islands."""
    ext2int = pytest.importorskip("pypower.ext2int").ext2int
    makeBdc = pytest.importorskip("pypower.makeBdc").makeBdc
    makeYbus = pytest.importorskip("pypower.makeYbus").makeYbus

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    d = read_matpower_dict(CASES / "case9.m")
    ppc = {k: (np.array(v, dtype=float) if k != "baseMVA" else v) for k, v in d.items() if v is not None}
    ppc["branch"][[2, 7], 10] = 0                                     # 5-6 and 8-9 out
    net = from_ppc({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()})
    on = net.status != 0
    n, labels = connected_components(coo_matrix((np.ones(int(on.sum())), (net.from_bus[on], net.to_bus[on])), shape=(net.n_bus, net.n_bus)), directed=False)
    assert n == 2 and sorted(net.bus_id[labels == labels[0]].tolist()) == [1, 4, 5, 9]
    i2e = ext2int({k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()})
    Y, _, _ = makeYbus(i2e["baseMVA"], i2e["bus"], i2e["branch"])
    B, _, Pbusinj, _ = makeBdc(i2e["baseMVA"], i2e["bus"], i2e["branch"])
    Y_ours = bus_admittance(net)
    Bbus, _, Pbus_shift, _ = dc_model(net)
    np.testing.assert_allclose(Y_ours, Y.toarray(), atol=1e-12)
    np.testing.assert_allclose(Bbus, B.toarray(), atol=1e-12)
    np.testing.assert_allclose(Pbus_shift, Pbusinj, atol=1e-12)
    assert np.all(Y_ours[np.ix_(labels == 0, labels == 1)] == 0)


def test_an_out_of_service_branch_without_impedance_drops_out():
    """MATPOWER drops off-line branches before it divides by r + jx; here status 0 multiplied a 0/0 and every entry
    of Y_bus and Bbus became NaN (code/02 first #8). Two parallel branches, the second out of service with r = x = 0:
    the matrices are those of the first branch alone."""
    from dataclasses import replace

    one = two_bus(r=0.01, x=0.1, b=0.02)
    both = replace(one, from_bus=np.array([0, 0]), to_bus=np.array([1, 1]), r=np.array([0.01, 0.0]),
                   x=np.array([0.1, 0.0]), b=np.array([0.02, 0.0]), tap=np.ones(2), shift=np.zeros(2),
                   status=np.array([1, 0]), rate_a=np.zeros(2))
    np.testing.assert_array_equal(bus_admittance(both), bus_admittance(one))
    np.testing.assert_array_equal(dc_model(both)[0], dc_model(one)[0])
    np.testing.assert_array_equal(bus_admittance(both, sparse=True).toarray(), bus_admittance(one))


def test_magnetic_laplacian_ignores_the_resistance_of_an_out_of_service_branch():
    """Like the charging, the resistance of a branch out of service is not in Y_bus, so it is not refused
    (code/02 first #31)."""
    from dataclasses import replace

    one = two_bus(r=0.0, x=0.1)
    both = replace(one, from_bus=np.array([0, 0]), to_bus=np.array([1, 1]), r=np.array([0.0, 0.05]),
                   x=np.array([0.1, 0.2]), b=np.zeros(2), tap=np.ones(2), shift=np.zeros(2),
                   status=np.array([1, 0]), rate_a=np.zeros(2))
    np.testing.assert_array_equal(magnetic_laplacian(both), magnetic_laplacian(one))
