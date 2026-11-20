"""Tests for gridnet.rei: the REI equivalent (radial, equivalent, independent) of Dimo, after Tinney & Powell,
"The REI Approach to Power Network Equivalents", PICA 1977, Sec. IV, eqs. (8)-(13), and their appendix.

The appendix is a complete numerical example on a dc network: p1 = 12 W at v1 = 6 V, p2 = 15 W at v2 = 3 V give
i1 = 2 A, i2 = 5 A, iR = 7 A, pR = 27 W, vR = 27/7 V, and with v_G = 0 the conductances g1 = -1/3 S, g2 = -5/3 S,
gR = 49/27 S; "the 27 watts of power consumed by the positive conductance of branch RG is generated in the
negative conductances of the other two branches".
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.rei import rei_equivalent, rei_star

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


def test_rei_star_reproduces_the_appendix_of_tinney_and_powell():
    star = rei_star(S=np.array([12.0, 15.0]), V=np.array([6.0, 3.0]))
    assert star.I == pytest.approx([2.0, 5.0]) and star.I_R == pytest.approx(7.0)
    assert star.S_R == pytest.approx(27.0) and star.V_R == pytest.approx(27 / 7)
    assert star.Y == pytest.approx([-1 / 3, -5 / 3]) and star.Y_R == pytest.approx(49 / 27)
    # power consumed in each branch, |current|^2 / conductance: 27 W in R-G, -12 W and -15 W in the other two
    consumed = np.r_[star.I_R ** 2 / star.Y_R, star.I ** 2 / star.Y]
    assert consumed == pytest.approx([27.0, -12.0, -15.0]) and consumed.sum() == pytest.approx(0.0)


def test_rei_star_with_a_nonzero_center_voltage():
    """Eqs. (13a), (13b): Y_i = I_i / (V_G - V_i), Y_R = I_R / (V_R - V_G); the terminal quantities (8)-(11) do
    not depend on V_G, and the star stays lossless."""
    a, c = rei_star(np.array([12.0, 15.0]), np.array([6.0, 3.0])), rei_star(np.array([12.0, 15.0]), np.array([6.0, 3.0]), v_g=1.0)
    assert c.V_R == pytest.approx(a.V_R) and c.I_R == pytest.approx(a.I_R)
    assert c.Y == pytest.approx([2.0 / (1.0 - 6.0), 5.0 / (1.0 - 3.0)]) and c.Y_R == pytest.approx(7.0 / (27 / 7 - 1.0))
    loss = c.Y_R * (c.V_R - 1.0) ** 2 + np.sum(c.Y * (1.0 - np.array([6.0, 3.0])) ** 2)
    assert loss == pytest.approx(0.0)


def test_rei_star_refuses_a_group_without_net_current_and_a_center_at_a_bus_voltage():
    """V_R = S_R / conj(I_R) (eq. (11)) needs I_R != 0: a generator and a load of equal size at the same
    voltage cancel, I_R = 0, and the node R has no voltage. The admittances Y_i = I_i / (V_G - V_i) and
    Y_R = I_R / (V_R - V_G) (eqs. (13a), (13b)) need V_G different from every V_i and from V_R. With the
    appendix's data V_R = 27/7."""
    with pytest.raises(ValueError, match="I_R"):
        rei_star(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
    S, V = np.array([12.0, 15.0]), np.array([6.0, 3.0])
    for v_g in (6.0, 3.0, 27 / 7):
        with pytest.raises(ValueError, match="v_g"):
            rei_star(S, V, v_g=v_g)


@pytest.mark.data("matpower")
def test_rei_equivalent_preserves_the_operating_point_of_ieee14():
    """Sec. IV-a: after the REI network is attached the N nodes are passive and can be eliminated "without
    affecting the conditions at remaining nodes"; the equivalent with the single injection S_R reproduces the
    voltages of the solved case at every remaining bus. IEEE 14, solved by Newton-Raphson (`gridnet.powerflow`); the group
    is the five load buses 10 to 14."""
    from gridnet.admittance import bus_admittance
    from gridnet.network import read_matpower_case
    from gridnet.powerflow import newton_raphson

    case = CASES / "case14.m"
    net = read_matpower_case(case)
    sol = newton_raphson(net)
    assert sol.converged
    V = sol.vm * np.exp(1j * sol.va)
    Y = bus_admittance(net)
    S = V * np.conj(Y @ V)                                       # complex power injected at every bus
    group = [9, 10, 11, 12, 13]
    eq = rei_equivalent(Y, group, S[group], V[group])
    rest = [k for k in range(14) if k not in group]
    assert eq.nodes == rest and eq.Y.shape == (10, 10)           # nine remaining buses and the REI node R, last
    I = np.r_[np.conj(S[rest] / V[rest]), eq.star.I_R]           # current injections of the equivalent
    V_eq = np.linalg.solve(eq.Y, I)
    assert np.allclose(V_eq[:-1], V[rest], atol=1e-10) and V_eq[-1] == pytest.approx(eq.star.V_R)
    assert eq.star.S_R == pytest.approx(S[group].sum())          # zero power balance of the REI network
    assert V_eq[-1] * np.conj(eq.star.I_R) == pytest.approx(eq.star.S_R)
    other = rei_equivalent(Y, group, S[group], V[group], v_g=0.5 + 0.1j)      # another center voltage, another matrix, the same state
    assert not np.allclose(other.Y, eq.Y)
    assert np.allclose(np.linalg.solve(other.Y, I)[:-1], V[rest], atol=1e-10)
    # loads carry negative injections, so their star branches have positive conductance and the branch of R negative
    assert np.all(eq.star.Y.real > 0) and eq.star.Y_R.real < 0


def test_rei_equivalent_refuses_groups_that_do_not_match_their_data():
    """`group`, `S` and `V` must have one entry per bus of the group and no bus twice; zip would truncate."""
    Y = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]], dtype=complex) * -10j
    S, V = np.array([1.0 + 0.2j, 0.5 + 0.1j]), np.array([1.0, 0.98 - 0.02j])
    with pytest.raises(ValueError, match="one entry per bus"):
        rei_equivalent(Y, [0, 1], S[:1], V[:1])
    with pytest.raises(ValueError, match="twice"):
        rei_equivalent(Y, [1, 1], S, V)
