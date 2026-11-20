"""AC power flow by Newton's method in polar coordinates.

References: MATPOWER User's Manual 8.0 (Zimmerman & Murillo-Sanchez 2024), Sec. 4.1:
the mismatch equations (4.2)-(4.3), the unknown vector of non-reference angles and PQ
magnitudes (4.5), Newton iteration on (4.4); the derivatives of the complex bus power
injections S_bus = [V] conj(Y_bus V) with respect to angles and magnitudes are those of
Zimmerman, "AC Power Flows, Generalized OPF Costs and their Derivatives using Complex
Matrix Notation", MATPOWER Technical Note 2 (2010, revision 7 of 2019), Sec. 4.2.1,
eqs. (29) and (32) (implemented in MATPOWER's dSbus_dV). Bus types per manual Sec. 4.1. Verified to machine precision against
MATPOWER's own algorithm through its Python port PYPOWER (runpf).

For the mathematician
---------------------
The unknowns are the phases theta (all buses but the reference) and the magnitudes |V|
(PQ buses only; PV buses hold |V| at a setpoint by adjusting their reactive output). The
equations are the real and reactive parts of the power balance g(x) = S_bus(V) + S_d -
S_g = 0 at those buses, a quadratic system in the complex voltages. Newton's method
solves g(x) = 0 with the exact Jacobian; from a flat start (|V| = 1, theta = 0) it
converges quadratically on normally loaded networks and typically fails beyond the
maximum loadability, where real solutions cease to exist (the "nose" of the P-V curve:
the two-bus closed form in the tests shows the discriminant turning negative).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

from gridnet.admittance import bus_admittance
from gridnet.network import Network


@dataclass
class PowerFlowSolution:
    vm: NDArray[np.float64]        # voltage magnitudes, per unit
    va: NDArray[np.float64]        # voltage angles, radians
    converged: bool
    iterations: int
    mismatch: float                # final max |g|
    qg: NDArray[np.float64]        # reactive generation at PV/reference buses (per unit)
    pg_ref: float                  # active generation at the reference bus (per unit)


def reference_bus(net: Network) -> int:
    """The one bus of type 3, which holds the angle. A network with none or with several is
    refused: the models of this package fix one angle, and a second bus of type 3 used to be
    held at angle zero and magnitude one without a word."""
    ref = np.flatnonzero(net.bus_type == 3)
    if ref.size != 1:
        raise ValueError(f"the network must have exactly one reference bus (type 3), it has {ref.size}")
    return int(ref[0])


def generation(net: Network, sol: PowerFlowSolution) -> NDArray[np.complex128]:
    """The complex generation per bus at the solved point, P_g + j Q_g: the case's P_g with
    the reference bus's entry replaced by the solved `pg_ref` (the case file's value there
    is only the start of the balance), and the solved Q_g."""
    pg = net.pg.astype(float).copy()
    pg[reference_bus(net)] = sol.pg_ref
    return pg + 1j * sol.qg


def check_machine_and_static_buses(gen_buses: ArrayLike, static_gen_buses: ArrayLike | None) -> None:
    """Every machine bus may be named once, and the buses of the machines and those of the
    generation without a machine must be disjoint. A machine gets the whole generation of its
    bus, so a bus named twice would have its generation twice; the machines of one bus are
    joined into one beforehand (inertias and dampings added, reactances in parallel). A bus that is both a
    machine's and a static source's would count its generator twice, once behind x'_d and
    once as a negative load."""
    buses, count = np.unique(np.asarray(gen_buses, dtype=int), return_counts=True)
    if np.any(count > 1):
        raise ValueError(f"buses {buses[count > 1].tolist()} carry more than one machine; join the machines of a bus into one first")
    if static_gen_buses is None:
        return
    both = np.intersect1d(buses, np.asarray(static_gen_buses, dtype=int))
    if both.size:
        raise ValueError(f"buses {both.tolist()} are both machine buses and buses of generation without a machine")


def setpoints(net: Network) -> NDArray[np.float64]:
    """The voltage magnitudes the voltage-holding buses keep: the setpoints VG of the case's
    generators (`Network.vg`, the last in-service generator's at a bus, as in MATPOWER), 1.0
    elsewhere and for a network that carries none (a network built by hand). The default of `newton_raphson`
    and of every solver built on it. A reference or PV bus without a setpoint
    (no generator in service there) is held at 1.0, with a warning."""
    if net.vg is None or np.all(np.isnan(net.vg)):
        return np.ones(net.n_bus)
    vg = np.asarray(net.vg, dtype=float)
    held = (net.bus_type == 2) | (net.bus_type == 3)
    if np.any(held & np.isnan(vg)):
        warnings.warn(f"no voltage setpoint at the voltage-holding buses {np.flatnonzero(held & np.isnan(vg)).tolist()}"
                      " (internal index); held at 1.0")
    return np.where(np.isnan(vg), 1.0, vg)


def _dSbus_dV(Y, V: NDArray) -> tuple:
    """dS/dtheta and dS/d|V| (TN2, Sec. 4.2.1, eqs. (29), (32)):
    dS/dtheta = j [V] conj([I] - Y [V]),  dS/d|V| = [V] conj([I] + Y [V]) [|V|]^{-1}.
    Dense for a dense Y, sparse (CSR) for a sparse Y; the formulas are the same."""
    I = Y @ V
    if sps.issparse(Y):
        dV = sps.diags(V)
        dI = sps.diags(I)
        dS_dva = (1j * dV @ (dI - Y @ dV).conj()).tocsr()
        dS_dvm = (dV @ (dI + Y @ dV).conj() @ sps.diags(1 / np.abs(V))).tocsr()
        return dS_dva, dS_dvm
    dV = np.diag(V)
    dI = np.diag(I)
    dS_dva = 1j * dV @ np.conj(dI - Y @ dV)
    dS_dvm = dV @ np.conj(dI + Y @ dV) @ np.diag(1 / np.abs(V))
    return dS_dva, dS_dvm


def jacobian(Y, V: NDArray, pvpq: NDArray, pq: NDArray):
    """The power flow Jacobian (manual eq. 4.4), the four blocks dP/dtheta, dP/d|V|,
    dQ/dtheta, dQ/d|V| restricted to the unknowns: angles at PV and PQ buses, magnitudes
    at PQ buses. Dense for a dense Y, CSC for a sparse Y."""
    dS_dva, dS_dvm = _dSbus_dV(Y, V)
    if sps.issparse(Y):
        return sps.bmat([
            [dS_dva[pvpq][:, pvpq].real, dS_dvm[pvpq][:, pq].real],
            [dS_dva[pq][:, pvpq].imag, dS_dvm[pq][:, pq].imag],
        ], format="csc")
    return np.block([
        [dS_dva[np.ix_(pvpq, pvpq)].real, dS_dvm[np.ix_(pvpq, pq)].real],
        [dS_dva[np.ix_(pq, pvpq)].imag, dS_dvm[np.ix_(pq, pq)].imag],
    ])


def hessian_of_injections(Y, V: NDArray, w_p: NDArray, w_q: NDArray, pvpq: NDArray, pq: NDArray) -> NDArray:
    """Hessian of w_p^T P(V) + w_q^T Q(V) with respect to the unknowns (angles at PV and
    PQ buses, magnitudes at PQ buses), from the second derivatives of the complex
    injections in Zimmerman's Technical Note 2, Sec. 4.2.2, eqs. (35) to (42), the mixed
    blocks (43) to (51) and the magnitude block (52) to (56): with [.] the diagonal matrix of a vector,
    A = [lambda][V], B = Y[V], C = A conj(B), D = Y^H [V], E = [V*](D[lambda] - [D lambda]),
    F = C - A [I*], G = [1/|V|], the blocks are G_thth = E + F, G_Vth = jG(E - F),
    G_thV = G_Vth^T, G_VV = G(C + C^T)G, for a complex multiplier lambda; the real
    Hessian of the real and reactive parts is the real part with lambda = w_p plus the
    imaginary part with lambda = w_q (MATPOWER's opf_hessfcn). Dense; the full matrix
    is formed and then restricted, so this is for networks of a few thousand buses."""
    Y = Y.toarray() if sps.issparse(Y) else Y
    n = len(V)
    I = Y @ V
    absV = np.abs(V)
    blocks = []
    for lam in (w_p, w_q):
        dlam = np.diag(lam)
        dV = np.diag(V)
        A = np.diag(lam * V)
        B = Y @ dV
        C = A @ np.conj(B)
        D = Y.conj().T @ dV
        E = np.conj(dV) @ (D @ dlam - np.diag(D @ lam))
        F = C - A @ np.diag(np.conj(I))
        G = np.diag(1 / absV)
        Gaa = E + F
        Gva = 1j * G @ (E - F)
        Gav = Gva.T
        Gvv = G @ (C + C.T) @ G
        blocks.append(np.block([[Gaa, Gav], [Gva, Gvv]]))
    H = blocks[0].real + blocks[1].imag
    idx = np.concatenate([pvpq, n + pq])
    return H[np.ix_(idx, idx)]


def newton_raphson(
    net: Network,
    vset: ArrayLike | None = None,
    vm0: ArrayLike | None = None,
    va0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 20,
    theta_ref: float = 0.0,
    sparse: bool | None = None,
) -> PowerFlowSolution:
    """Solve the AC power flow of `net` (manual Sec. 4.1, eqs. 4.2-4.5).

    Specified quantities: net.pg - net.pd at every non-reference bus, -net.qd at PQ
    buses (type 1), |V| = vset at PV (type 2) and reference (type 3) buses (by default the
    case's setpoints, `setpoints`), angle
    `theta_ref` (radians) at the reference bus. Shunts gs, bs enter through Y_bus. Starts from (vm0, va0), by
    default the flat start with vset at voltage-controlled buses. Convergence: max
    mismatch below `tol` in per unit. `sparse` selects a scipy.sparse Jacobian and a
    sparse LU solve (SuperLU), the same iteration as the dense one; by default it is
    used above 1,000 buses.
    """
    n = net.n_bus
    if sparse is None:
        sparse = n > 1000
    Y = bus_admittance(net, sparse=sparse)
    ref = reference_bus(net)
    pv = np.flatnonzero(net.bus_type == 2)
    pq = np.flatnonzero(net.bus_type == 1)
    npv, npq = len(pv), len(pq)
    vset = setpoints(net) if vset is None else np.asarray(vset, dtype=float)
    vm = np.ones(n) if vm0 is None else np.asarray(vm0, dtype=float).copy()
    va = np.zeros(n) if va0 is None else np.asarray(va0, dtype=float).copy()
    vm[pv] = vset[pv]
    vm[ref] = vset[ref]
    va[ref] = float(theta_ref)
    s_spec = (net.pg - net.pd) + 1j * (-net.qd)         # specified net injection (Q only used at PQ)
    pvpq = np.concatenate([pv, pq])

    def mismatch(vm, va):
        V = vm * np.exp(1j * va)
        S = V * np.conj(Y @ V)                           # manual eq. (3.14)
        g = S - s_spec
        return np.concatenate([g.real[pvpq], g.imag[pq]]), V, S

    converged = False
    it = 0
    g, V, S = mismatch(vm, va)
    norm = np.max(np.abs(g)) if g.size else 0.0
    while it < max_iter:
        if norm < tol:
            converged = True
            break
        J = jacobian(Y, V, pvpq, pq)
        if sparse:
            try:
                dx = spla.spsolve(J, -g)
            except RuntimeError:
                break
        else:
            try:
                dx = np.linalg.solve(J, -g)
            except np.linalg.LinAlgError:
                break
        if not np.all(np.isfinite(dx)):
            break
        va[pvpq] += dx[: npv + npq]
        vm[pq] += dx[npv + npq :]
        it += 1
        g, V, S = mismatch(vm, va)
        norm = np.max(np.abs(g)) if g.size else 0.0
        if not np.isfinite(norm) or norm > 1e6:
            break
    if norm < tol:
        converged = True
    qg = np.zeros(n)
    gen_buses = np.concatenate([[ref], pv])
    qg[gen_buses] = S.imag[gen_buses] + net.qd[gen_buses]
    pg_ref = float(S.real[ref] + net.pd[ref])
    return PowerFlowSolution(vm=vm, va=va, converged=converged, iterations=it, mismatch=float(norm), qg=qg, pg_ref=pg_ref)
