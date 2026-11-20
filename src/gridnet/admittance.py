"""Bus admittance matrix, DC power flow matrices, and the lossless/magnetic structure.

All formulas are those of the MATPOWER User's Manual 8.0 (Zimmerman & Murillo-Sanchez
2024), cited by equation number; MATPOWER's makeYbus and makeBdc implement the same.

Electrical background for the mathematician
-------------------------------------------
Kirchhoff's current law at every bus, with the branch currents expressed through
complex bus voltages V (magnitude and phase as one complex number, "phasor"), gives the
linear relation I_bus = Y_bus V (eq. 3.8). Y_bus is the "bus admittance matrix", the
complex, weighted Laplacian-like operator of the network; admittance is the inverse of
impedance, y = 1/z. Power, not current, is what is specified at buses, so the power flow
equations S = diag(V) conj(Y_bus V) (eq. 3.14) are quadratic in V; everything nonlinear
in power systems enters through this one product.

Branch model (eq. 3.2, Figure 3-1)
----------------------------------
Each branch is a pi-section (series admittance y_s = 1/(r + jx), charging susceptance
b split as jb/2 on each end) in series with an ideal transformer of complex ratio
tau e^{j shift} at the from end. Its 2x2 admittance matrix is

    Y_br = [[ (y_s + jb/2)/tau^2 ,  -y_s/(tau e^{-j shift}) ],
            [ -y_s/(tau e^{ j shift}),  y_s + jb/2            ]].

The phase shift makes Y_br non-symmetric: Y_ft != Y_tf. That is the discrete analog of
a U(1) connection on the edge; without shifts Y_bus is symmetric (not Hermitian:
complex symmetric).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gridnet.network import Network


def branch_admittances(net: Network) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The four entries (Yff, Yft, Ytf, Ytt) of every branch's 2x2 matrix, eq. (3.2).

    Out-of-service branches (status 0) get zeros, as in MATPOWER's makeYbus where the
    series admittance is multiplied by the status flag.
    """
    stat = net.status.astype(float)
    # series admittance, manual Sec. 3.2; an out-of-service branch gets 0 without dividing, as MATPOWER drops it
    # first, so that one with r = x = 0 does not put 0/0 into every entry
    on = stat > 0
    ys = np.zeros(len(stat), dtype=complex)
    ys[on] = stat[on] / (net.r[on] + 1j * net.x[on])
    bc = stat * net.b                                 # total charging susceptance
    tau = net.tap * np.exp(1j * np.deg2rad(net.shift))  # complex tap ratio tau e^{j shift}
    yff = (ys + 1j * bc / 2) / (net.tap**2)           # |tau|^2 = tap^2
    ytt = ys + 1j * bc / 2
    yft = -ys / np.conj(tau)
    ytf = -ys / tau
    return yff, yft, ytf, ytt


def connection_matrices(net: Network) -> tuple[NDArray, NDArray]:
    """Branch-by-bus connection matrices Cf, Ct (manual Sec. 3.2): Cf[l, f_l] = 1, Ct[l, t_l] = 1.

    Cf - Ct is the transpose of the oriented incidence matrix of `gridnet.graph`.
    """
    m, n = net.n_branch, net.n_bus
    Cf = np.zeros((m, n))
    Ct = np.zeros((m, n))
    Cf[np.arange(m), net.from_bus] = 1.0
    Ct[np.arange(m), net.to_bus] = 1.0
    return Cf, Ct


def bus_admittance(net: Network, return_branch_matrices: bool = False, sparse: bool = False):
    """Bus admittance matrix Y_bus (eq. 3.13), optionally with Yf, Yt (eqs. 3.11, 3.12).

    Y_bus = Cf^T Yf + Ct^T Yt + diag(Y_sh), with Yf = diag(Yff) Cf + diag(Yft) Ct and
    Yt = diag(Ytf) Cf + diag(Ytt) Ct, and the bus shunts Y_sh = gs + j bs (eq. 3.7).
    Returns dense complex arrays, or scipy.sparse CSR matrices with `sparse=True`, the
    same entries assembled from the branch list as MATPOWER's makeYbus does (the dense
    Y_bus of the 9,241-bus PEGASE case would need 1.4 GB).
    """
    yff, yft, ytf, ytt = branch_admittances(net)
    if sparse:
        import scipy.sparse as sps

        m, n = net.n_branch, net.n_bus
        rows = np.arange(m)
        Yf = sps.csr_matrix((np.concatenate([yff, yft]), (np.concatenate([rows, rows]),
                             np.concatenate([net.from_bus, net.to_bus]))), shape=(m, n))
        Yt = sps.csr_matrix((np.concatenate([ytf, ytt]), (np.concatenate([rows, rows]),
                             np.concatenate([net.from_bus, net.to_bus]))), shape=(m, n))
        Cf = sps.csr_matrix((np.ones(m), (rows, net.from_bus)), shape=(m, n))
        Ct = sps.csr_matrix((np.ones(m), (rows, net.to_bus)), shape=(m, n))
        Ybus = (Cf.T @ Yf + Ct.T @ Yt + sps.diags(net.gs + 1j * net.bs)).tocsr()
        return (Ybus, Yf, Yt) if return_branch_matrices else Ybus
    Cf, Ct = connection_matrices(net)
    Yf = yff[:, None] * Cf + yft[:, None] * Ct
    Yt = ytf[:, None] * Cf + ytt[:, None] * Ct
    Ybus = Cf.T @ Yf + Ct.T @ Yt + np.diag(net.gs + 1j * net.bs)
    return (Ybus, Yf, Yt) if return_branch_matrices else Ybus


def dc_model(net: Network, sparse: bool = False) -> tuple:
    """DC power flow matrices (Bbus, Bf, Pbus_shift, Pf_shift), eqs. (3.24), (3.27)-(3.31);
    dense arrays, or CSR matrices for Bbus and Bf with `sparse=True`.

    Under the DC assumptions (eqs. 3.18-3.20: lossless branches, flat voltage
    magnitudes, small angle differences) the from-end active flow of branch l is
    p_f = b_l (theta_f - theta_t - shift_l) with b_l = 1/(x_l tau_l), so

        P_bus = Bbus theta + Pbus_shift,   P_f = Bf theta + Pf_shift,
        Bf = diag(b) (Cf - Ct),  Bbus = (Cf - Ct)^T Bf,
        Pf_shift = -shift b,     Pbus_shift = (Cf - Ct)^T Pf_shift.

    Bbus is exactly the weighted graph Laplacian A diag(b) A^T of `gridnet.graph` with
    A = (Cf - Ct)^T. Phase shifts do not enter the operator, only the source term: in
    the linear model a phase shifter is a fixed injection pair (-shift b at the from bus,
    +shift b at the to bus). Shifts are converted from degrees to radians here.
    """
    stat = net.status.astype(float)
    b = np.zeros(len(stat))
    b[stat > 0] = stat[stat > 0] / (net.x * net.tap)[stat > 0]   # out-of-service branches 0 without dividing (see bus_admittance)
    Pf_shift = -np.deg2rad(net.shift) * b
    if sparse:
        import scipy.sparse as sps

        m, n = net.n_branch, net.n_bus
        rows = np.arange(m)
        Cft = sps.csr_matrix((np.concatenate([np.ones(m), -np.ones(m)]),
                              (np.concatenate([rows, rows]), np.concatenate([net.from_bus, net.to_bus]))), shape=(m, n))
        Bf = (sps.diags(b) @ Cft).tocsr()
        Bbus = (Cft.T @ Bf).tocsr()
        return Bbus, Bf, Cft.T @ Pf_shift, Pf_shift
    Cf, Ct = connection_matrices(net)
    Cft = Cf - Ct
    Bf = b[:, None] * Cft
    Bbus = Cft.T @ Bf
    Pbus_shift = Cft.T @ Pf_shift
    return Bbus, Bf, Pbus_shift, Pf_shift


def magnetic_laplacian(net: Network) -> NDArray[np.complex128]:
    """The Hermitian positive semidefinite operator M = j Y_bus of a lossless network without charging.

    For r = 0 and gs = 0 the series admittance is y_s = -j/x, so Y_bus = -j L(1/x) for a
    plain lossless network and M = j Y_bus = L(1/x). With taps and phase shifts every
    entry of Y_bus is still j times a real number up to the phase factors e^{+-j shift},
    and M is Hermitian with M_ft = -(1/(x tau)) e^{j shift} = conj(M_tf). Its diagonal
    collects 1/(x tau^2) from every branch at its from end and 1/x at its to end (eq. 3.2
    with y_s = -j/x; the tap sits at the from end). Each branch contributes the rank-one
    term (1/x) c c^* with c = e_f / tau - e^{-j shift} e_t, so M is positive semidefinite:
    a magnetic Laplacian with tap weights, which without shifts and with unit taps reduces
    to the real Laplacian L(1/x).

    Charging and shunt susceptance would add -b/(2 tau^2), -b/2 and -bs to the diagonal
    (eqs. 3.2, 3.13): M stays Hermitian but is no longer positive semidefinite (case14 with
    r = gs = 0 and its charging and bus shunt kept has the smallest eigenvalue -0.029; the
    charging alone gives -0.0153). Those networks
    are therefore refused; `Network.lossless()` sets b = bs = 0.

    Sign: texts that write the bus admittance as Y = G + jB with B = -L(1/x) (MATPOWER's
    convention, used here) get the Laplacian as +j Y; texts with the opposite sign of B
    write -j Y. The operator is the same.

    With unit taps this is the discrete magnetic Laplacian (graph connection Laplacian with
    U(1) connection e^{j shift} on each edge) in the sense of Bandeira, Singer & Spielman,
    SIAM J. Matrix Anal. Appl. 34(4) (2013), Sec. 1; the physical origin of the
    connection is the winding arrangement of phase-shifting transformers, not a magnetic
    flux. Off-nominal taps weight the two ends of an edge differently.

    Raises ValueError unless r = 0, gs = 0 (otherwise j Y_bus is not Hermitian) and
    b = 0, bs = 0 (otherwise it is not positive semidefinite), since the spectral
    statements need both.
    """
    if np.any((net.status != 0) & (net.r != 0)) or np.any(net.gs != 0):     # as below, out-of-service branches are not in Y_bus
        raise ValueError("magnetic_laplacian requires a lossless network (r = 0, gs = 0)")
    if np.any((net.status != 0) & (net.b != 0)) or np.any(net.bs != 0):     # out-of-service charging is not in Y_bus
        raise ValueError("magnetic_laplacian requires b = 0, bs = 0: charging and shunt susceptance "
                         "make j Y_bus indefinite (use Network.lossless())")
    return 1j * bus_admittance(net)
