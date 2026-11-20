"""The REI equivalent: aggregating the injections of a group of buses into one fictitious bus.

Source: Tinney & Powell, "The REI Approach to Power Network Equivalents", Proc. PICA 1977, pp. 314-320, who
present the method of P. Dimo (Nodal Analysis of Power Systems, 1975). Equation numbers refer to their paper.
REI stands for radial, equivalent, independent.

The idea (their Sec. IV-a). A network with a solved operating point has a group of N buses with complex power
injections S_i at voltages V_i. A fictitious lossless network, the REI network, is attached to these buses; it
has one new active node R whose injection is the sum S_R of the S_i, and it delivers to every bus of the group
exactly the power that was injected there. The N buses are now passive, carry no injection, and can be
eliminated by Gaussian elimination, the Kron reduction (`gridnet.kron`). What remains is the rest of the network plus
the node R, and at the operating point its state is that of the original network.

The star form (their Fig. 2b): a passive center G with a voltage V_G that may be chosen, a branch G-i to every
bus of the group and a branch R-G. With the currents I_i = conj(S_i / V_i) (eq. (8)), I_R = sum I_i (eq. (9)),
S_R = sum S_i (eq. (10)) and V_R = S_R / conj(I_R) (eq. (11)) the admittances are

    Y_i = I_i / (V_G - V_i),      Y_R = I_R / (V_R - V_G)                       (eqs. (13a), (13b)),

and for V_G = 0, Dimo's choice, Y_i = -conj(S_i) / |V_i|^2 and Y_R = conj(S_R) / |V_R|^2 (eqs. (12a), (12b)).

What sets it apart from the reductions of `gridnet.kron` and `gridnet.reweight`. The star depends on the operating point, where the Kron
reduction and its accompanying matrix depend on the network alone. Its admittances are not those of physical
lines: the star as a whole consumes no power, so what the branch R-G consumes the other branches generate, and
some conductances are negative (their appendix). The reduced matrix is therefore not a Laplacian with
nonnegative weights, and spectral comparisons of Laplacians do not apply to it. What it offers instead is the
aggregation of injections that a clustering also performs, generators and loads of a group on one bus, exact at
one operating point.

For a mathematician: bus = node, admittance = complex edge weight, the bus admittance matrix Y maps the node
voltages to the injected currents, I = Y V (`gridnet.admittance`), and complex power is S = V conj(I).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridnet.kron import kron_reduce


@dataclass
class ReiStar:
    I: NDArray                       # currents into the buses of the group, eq. (8)
    I_R: complex                     # current entering the node R, eq. (9)
    S_R: complex                     # power injected at R, eq. (10)
    V_R: complex                     # voltage of R, eq. (11)
    Y: NDArray                       # admittances of the branches G-i, eq. (12a) or (13a)
    Y_R: complex                     # admittance of the branch R-G, eq. (12b) or (13b)
    v_g: complex                     # voltage of the center G


@dataclass
class ReiEquivalent:
    Y: NDArray                       # admittance matrix on the remaining buses and R (last)
    nodes: list                      # the remaining buses, in the order of the rows of Y
    star: ReiStar


def rei_star(S: ArrayLike, V: ArrayLike, v_g: complex = 0.0) -> ReiStar:
    """The REI star of a group of buses with injections S at voltages V (Tinney & Powell 1977, eqs. (8)-(13)).

    Real input gives the dc network of their appendix (powers in W, voltages in V, conductances in S), complex
    input the ac case. With v_g = 0, Dimo's choice, the admittances are eqs. (12a), (12b).

    Raises ValueError if the currents of the group cancel, I_R = 0 (then eq. (11) gives R no voltage, for
    example a generator and a load of equal size at one voltage), or if v_g coincides with a bus voltage V_i or
    with V_R (then eq. (13a) or (13b) divides by zero). Both are judged relative to 1e-12 of the magnitudes."""
    S, V = np.asarray(S), np.asarray(V)
    I = np.conj(S / V)                                           # eq. (8): I_i = S_i^* / V_i^*
    I_R = I.sum()                                                # eq. (9)
    if abs(I_R) <= 1e-12 * np.abs(I).sum():
        raise ValueError("the currents of the group cancel (I_R = 0): the REI node R has no voltage, eq. (11)")
    S_R = S.sum()                                                # eq. (10)
    V_R = S_R / np.conj(I_R)                                     # eq. (11)
    scale = max(np.abs(V).max(), abs(V_R))
    if np.any(np.abs(v_g - V) <= 1e-12 * scale) or abs(V_R - v_g) <= 1e-12 * scale:
        raise ValueError("v_g coincides with a bus voltage or with V_R: a star admittance, eq. (13a) or (13b), "
                         "would be infinite")
    return ReiStar(I=I, I_R=I_R, S_R=S_R, V_R=V_R, Y=I / (v_g - V), Y_R=I_R / (V_R - v_g), v_g=v_g)


def rei_equivalent(Y: NDArray, group: Sequence[int], S: ArrayLike, V: ArrayLike, v_g: complex = 0.0) -> ReiEquivalent:
    """Replace the buses `group` of a network with admittance matrix Y by one REI node.

    The star of `rei_star` is attached (nodes G and R are added, each star branch entering the admittance matrix
    like a line), then the buses of the group and the center G, all passive now, are eliminated by the Kron
    reduction (Tinney & Powell 1977, Sec. IV-a, Fig. 1; Sec. IV-b on eliminating G). The result is the
    admittance matrix on the remaining buses, in their original order, with R as the last node. At the
    operating point, the current injections of the remaining buses together with I_R at R reproduce their
    voltages and V_R, for any center voltage v_g: the currents at G balance, I_R = sum I_i, so G is passive."""
    Y = np.asarray(Y, dtype=complex)
    n = Y.shape[0]
    group = [int(g) for g in group]
    if not (len(group) == np.size(S) == np.size(V)):
        raise ValueError(f"group, S and V need one entry per bus of the group ({len(group)}, {np.size(S)}, {np.size(V)})")
    if len(set(group)) < len(group):
        raise ValueError("a bus appears twice in the group")
    star = rei_star(S, V, v_g)
    g_node, r_node = n, n + 1
    Ya = np.zeros((n + 2, n + 2), dtype=complex)
    Ya[:n, :n] = Y
    for bus, y in zip(group, star.Y):                            # branch G-bus
        Ya[bus, bus] += y
        Ya[g_node, g_node] += y
        Ya[bus, g_node] -= y
        Ya[g_node, bus] -= y
    Ya[r_node, r_node] += star.Y_R                               # branch R-G
    Ya[g_node, g_node] += star.Y_R
    Ya[r_node, g_node] -= star.Y_R
    Ya[g_node, r_node] -= star.Y_R
    in_group = set(group)
    rest = [k for k in range(n) if k not in in_group]
    return ReiEquivalent(Y=kron_reduce(Ya, rest + [r_node]), nodes=rest, star=star)
