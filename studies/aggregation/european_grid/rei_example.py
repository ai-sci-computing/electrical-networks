"""The REI equivalent as a worked example: Tinney & Powell's appendix and IEEE 14.

REI aggregates the injections of a group of buses into one fictitious bus R through a lossless star computed at a
solved operating point, then eliminates the group (Tinney & Powell 1977, Sec. IV, eqs. (8)-(13), after Dimo 1975).
It is the classical counterpart of what a clustering does with generators and loads, treated here as an
example only: it needs an operating point, which the European base network does not carry, and its admittances
are not those of a graph with nonnegative weights.

1. The appendix of the paper, a dc network, reproduced.
2. IEEE 14 (MATPOWER case14, solved by gridnet.powerflow.newton_raphson with the case file's voltage setpoints,
   its default): the five load buses 10 to 14 become one REI node; the state is reproduced at the operating point; the star's admittances.
3. Dependence on the operating point: the star recomputed with all loads at 80 % and 120 %, and the equivalent of
   the base case fed with the currents of those states, against Ward's reduction (Kron reduction with the currents
   carried by the accompanying matrix), which is linear and exact for any currents.

Prediction written before the first run (2026-09-19): under a uniform load change of +-20 % the base-case REI
equivalent stays accurate to about a percent in the voltages; Ward's reduction is exact to rounding.

Run: .venv/bin/python studies/aggregation/european_grid/rei_example.py
"""

import copy
from pathlib import Path

import numpy as np

from gridnet.admittance import bus_admittance
from gridnet.kron import accompanying_matrix, kron_reduce
from gridnet.network import read_matpower_case
from gridnet.powerflow import newton_raphson
from gridnet.rei import rei_equivalent, rei_star

ROOT = Path(__file__).resolve().parents[3]

print("1. Tinney & Powell 1977, appendix: p = (12, 15) W at v = (6, 3) V")
star = rei_star(np.array([12.0, 15.0]), np.array([6.0, 3.0]))
print(f"   i = {star.I}, i_R = {star.I_R}, p_R = {star.S_R}, v_R = {star.V_R:.6f} (27/7 = {27/7:.6f})")
print(f"   g = {star.Y} (-1/3, -5/3), g_R = {star.Y_R:.6f} (49/27 = {49/27:.6f}); power consumed per branch: R-G {star.I_R**2 / star.Y_R:.1f} W, G-1 {star.I[0]**2 / star.Y[0]:.1f} W, G-2 {star.I[1]**2 / star.Y[1]:.1f} W")

net = read_matpower_case(ROOT / "data" / "matpower" / "case14.m")
Y = bus_admittance(net)
group = [9, 10, 11, 12, 13]
rest = [k for k in range(14) if k not in group]


def solved(scale):
    """State with the loads multiplied by `scale`, a number or one factor per bus."""
    n2 = copy.deepcopy(net); n2.pd = net.pd * scale; n2.qd = net.qd * scale
    sol = newton_raphson(n2); assert sol.converged
    V = sol.vm * np.exp(1j * sol.va)
    return V, V * np.conj(Y @ V)


V0, S0 = solved(1.0)
eq = rei_equivalent(Y, group, S0[group], V0[group])
print("\n2. IEEE 14, buses 10 to 14 as one REI node")
print(f"   S_R = {eq.star.S_R * 100:.2f} MVA (sum of the five injections), V_R = {abs(eq.star.V_R):.4f} p.u. at {np.degrees(np.angle(eq.star.V_R)):.2f} deg; the group's voltages: {np.abs(V0[group]).round(4)}")
print(f"   star admittances G-i (p.u.): {eq.star.Y.round(4)};  R-G: {eq.star.Y_R:.4f}")
I0 = np.r_[np.conj(S0[rest] / V0[rest]), eq.star.I_R]
print(f"   largest voltage error of the equivalent at the operating point: {np.abs(np.linalg.solve(eq.Y, I0)[:-1] - V0[rest]).max():.1e}")
offdiag = eq.Y - np.diag(np.diag(eq.Y))
print(f"   reduced matrix: {eq.Y.shape[0]} nodes; couplings with negative conductance (Re(-Y_ij) < 0): {int(np.sum(np.triu(-offdiag.real < -1e-12, 1)))} of {int(np.sum(np.triu(np.abs(offdiag) > 1e-12, 1)))}")

print("\n3. other operating points: all loads scaled")
Yw = kron_reduce(Y, rest); D = accompanying_matrix(Y, rest)
for scale in (0.8, 1.2):
    V, S = solved(scale)
    I = np.conj(S / V)
    s2 = rei_star(S[group], V[group])
    change = np.abs(s2.Y / eq.star.Y - 1)
    V_rei = np.linalg.solve(eq.Y, np.r_[I[rest], I[group].sum()])[:-1]          # base-case equivalent, the new currents, the group's current summed at R
    V_ward = np.linalg.solve(Yw, D @ I)                                        # Ward: linear in the currents
    print(f"   loads x {scale}: star admittances change by {100 * change.min():.1f} to {100 * change.max():.1f} %, Y_R by {100 * abs(s2.Y_R / eq.star.Y_R - 1):.1f} %; "
          f"largest voltage error at the remaining buses: REI of the base case {np.abs(V_rei - V[rest]).max():.2e} p.u., Ward {np.abs(V_ward - V[rest]).max():.1e} p.u.")

# a change inside the group that does not keep the proportions of its currents: only the load of bus 14 doubles
factors = np.ones(14); factors[13] = 2.0
V, S = solved(factors)
I = np.conj(S / V)
V_rei = np.linalg.solve(eq.Y, np.r_[I[rest], I[group].sum()])[:-1]
V_ward = np.linalg.solve(Yw, D @ I)
print(f"   load of bus 14 doubled: largest voltage error at the remaining buses: REI of the base case {np.abs(V_rei - V[rest]).max():.2e} p.u. "
      f"(the true change of those voltages is {np.abs(V[rest] - V0[rest]).max():.2e}), Ward {np.abs(V_ward - V[rest]).max():.1e} p.u.")

