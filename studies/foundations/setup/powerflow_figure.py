"""Explanatory figure for the AC power flow: the two-bus P-V "nose" curve and Newton's
quadratic convergence.

(a) Two-bus system (reference bus, PQ load through reactance x): the closed-form |V_2|
    as a function of load P at fixed power factor (the two roots of the quadratic in
    |V_2|^2), with Newton solutions from a flat start marked; beyond the nose no real
    solution exists and Newton diverges (MATPOWER manual 8.0, Sec. 4.1, for the method;
    the P-V curve is the textbook maximum-loadability picture).
(b) Mismatch per iteration on IEEE 118 and 300: the slope doubling on the log scale is
    quadratic convergence.
Run:  python studies/foundations/setup/powerflow_figure.py   (needs data/matpower/)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridnet.network import Network, read_matpower_case, read_matpower_dict
from gridnet.powerflow import newton_raphson

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figures" / "powerflow.png"
OUT.parent.mkdir(exist_ok=True)


def two_bus(p, q, x=0.2):
    return Network(
        base_mva=100.0, bus_id=np.array([1, 2]), bus_type=np.array([3, 1]),
        pd=np.array([0.0, p]), qd=np.array([0.0, q]), gs=np.zeros(2), bs=np.zeros(2),
        from_bus=np.array([0]), to_bus=np.array([1]), r=np.array([0.0]), x=np.array([x]), b=np.array([0.0]),
        tap=np.ones(1), shift=np.zeros(1), status=np.ones(1, dtype=int), rate_a=np.zeros(1),
    )


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
x, tanphi = 0.2, 0.5                                   # load with Q = 0.5 P
P = np.linspace(0, 3, 400)
Q = tanphi * P
a = 1 - 2 * Q * x
disc = a**2 - 4 * x**2 * (P**2 + Q**2)
ok = disc >= 0
hi = np.sqrt((a[ok] + np.sqrt(disc[ok])) / 2)
lo = np.sqrt((a[ok] - np.sqrt(disc[ok])) / 2)
ax1.plot(P[ok], hi, color="tab:blue", label="closed form, high-voltage root")
ax1.plot(P[ok], lo, color="tab:blue", ls="--", label="closed form, low-voltage root")
Pn = np.linspace(0.1, 2.6, 14)
newton = []
for p in Pn:
    sol = newton_raphson(two_bus(p, tanphi * p, x), vm0=np.ones(2), va0=np.zeros(2), max_iter=25)
    newton.append(sol.vm[1] if sol.converged else np.nan)
ax1.plot(Pn, newton, "o", color="tab:red", label="Newton from flat start")
nose = (np.sqrt(1 + tanphi**2) - tanphi) / (2 * x)                   # the closed form: the discriminant vanishes, 1 - 4 t P x - 4 x^2 P^2 = 0
assert P[ok][-1] <= nose < P[ok][-1] + (P[1] - P[0])               # the last point of the grid with two roots lies just below it
ax1.axvline(nose, color="0.6", ls=":", lw=1)
ax1.text(nose, 0.1, f"nose: $P_{{max}}$={nose:.3f}", ha="right", fontsize=8)
ax1.set_xlabel("load $P$ [p.u.] at $Q = 0.5P$, $x$ = 0.2"); ax1.set_ylabel("$|V_2|$ [p.u.]")
ax1.set_title("(a) two-bus P-V curve: solutions and the loadability limit")
ax1.legend(frameon=False, fontsize=8, loc="lower left")

for case, col in [("case118", "tab:green"), ("case300", "tab:purple")]:
    net = read_matpower_case(ROOT / "data" / "matpower" / f"{case}.m")
    ppc = read_matpower_dict(ROOT / "data" / "matpower" / f"{case}.m")
    ext2int = {int(b): i for i, b in enumerate(ppc["bus"][:, 0])}
    vset = np.ones(net.n_bus)
    for row in ppc["gen"]:
        if row[7] > 0:
            vset[ext2int[int(row[0])]] = row[5]
    hist = []
    for it in range(9):
        sol = newton_raphson(net, vset=vset, tol=0.0, max_iter=it)
        hist.append(max(sol.mismatch, 1e-16))
    ax2.semilogy(range(len(hist)), hist, "o-", color=col, label=f"{case} ({net.n_bus} buses)")
ax2.set_xlabel("Newton iteration"); ax2.set_ylabel("max power mismatch [p.u.]")
ax2.set_title("(b) quadratic convergence from a flat start")
ax2.legend(frameon=False, fontsize=8); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
