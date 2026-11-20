"""Two illustrations of the problems of the field.

figures/time_scales.png  The dynamics of a power system by time frame, after Machowski 2020, Sec. 1.3
                         and Figure 1.3 (wave, electromagnetic, electromechanical, thermodynamic), and
                         underneath the window each model covers: the phasor description
                         from about a tenth of a second on (Kundur 1994, Sec. 3.1, the neglect of stator
                         transients), the classical model for the electromechanical transient (Machowski
                         2020, Sec. 5.4), the linearization on the same window, the steady state once
                         the swings have died out.
figures/landscape.png    The potential U of the lossless model (Bergen & Hill 1981, Sec. IV; Mehta et al.
                         2015, eq. 2). Left: one machine against a large grid, U = -b cos(theta) - P theta,
                         the tilted washboard with its minima, maxima and the barrier between them. Right:
                         a triangle with a reference bus, U on the two remaining angles, with its critical
                         points found by Newton's method and classified by the Hessian.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DPI = 200
BLUE, ORANGE, GREEN, RED = "tab:blue", "tab:orange", "tab:green", "tab:red"

# =====================================================================================
# 1. Time scales
# =====================================================================================
fig, ax = plt.subplots(figsize=(7.2, 3.8))
phenomena = [  # label, start [s], end [s]   (Machowski 2020, Fig. 1.3)
    ("lightning and switching surges", 1e-6, 1e-3),
    ("electromagnetic transients of windings and lines", 1e-3, 1e0),
    ("rotor swings", 1e-1, 1e1),
    ("excitation and turbine control", 1e-1, 1e2),
    ("boilers, tap changers, load recovery", 1e1, 1e4),
]
models = [  # label, start, end (None = open to the right)
    ("phasor description, the reference system", 1e-1, None),
    ("classical model", 1e-1, 1e0),  # valid while the internal voltages stay constant, about a second
    ("small motions", 1e-1, 1e1),
    ("steady state", 1e1, None),
]
XMAX = 3e4
rows = []
y = 0
for label, a, b in phenomena:
    ax.barh(y, np.log10(b) - np.log10(a), left=np.log10(a), height=0.62, color="0.82", edgecolor="0.55", lw=0.8)
    ax.text(np.log10(a) - 0.12 if a > 1e-5 else np.log10(b) + 0.12, y, label, fontsize=8.5, va="center",
            ha="right" if a > 1e-5 else "left", color="0.25")
    y -= 1
y -= 0.4
ax.axhline(y + 0.5, color="0.7", lw=0.8, ls=":")
ax.text(-6.2, y - 0.05, "models", fontsize=9, color=BLUE, ha="left", va="center", style="italic")
y -= 0.9
for label, a, b in models:
    right = np.log10(b) if b else np.log10(XMAX)
    ax.barh(y, right - np.log10(a), left=np.log10(a), height=0.62, color=BLUE, alpha=0.85 if b else 0.5, edgecolor="none")
    if b is None:
        ax.annotate("", xy=(np.log10(XMAX) + 0.25, y), xytext=(np.log10(XMAX) - 0.05, y),
                    arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.8, "mutation_scale": 14, "alpha": 0.5})
    ax.text(np.log10(a) - 0.12, y, label, fontsize=8.5, va="center", ha="right", color=BLUE)
    y -= 1
ax.text(-6.2, 0.9, "phenomena", fontsize=9, color="0.25", ha="left", va="bottom", style="italic")
ticks = np.arange(-6, 5)
ax.set_xticks(ticks)
ax.set_xticklabels([{-6: "1 µs", -3: "1 ms", 0: "1 s", 3: "1000 s"}.get(t, "") for t in ticks], fontsize=9)
ax.set_xlim(-6.3, np.log10(XMAX) + 0.4); ax.set_ylim(y + 0.3, 0.9)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("time after a disturbance", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "time_scales.png", dpi=DPI, bbox_inches="tight")
print("wrote", OUT / "time_scales.png")

# =====================================================================================
# 2. The potential landscape
# =====================================================================================
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})

# --- one machine: tilted washboard --------------------------------------------------
b, P = 1.0, 0.4
th = np.linspace(-1.2 * np.pi, 3.2 * np.pi, 800)
U1 = -b * np.cos(th) - P * th
ax.plot(th, U1, color="0.3", lw=1.8)
tmin = np.arcsin(P / b)          # minima at arcsin(P/b) + 2 pi k
tmax = np.pi - np.arcsin(P / b)  # maxima at pi - arcsin(P/b) + 2 pi k
for k in (-1, 0, 1):
    for x, col, mk, sz in ((tmin + 2 * np.pi * k, BLUE, "o", 60), (tmax + 2 * np.pi * k, RED, "D", 45)):
        if th[0] <= x <= th[-1]:
            ax.scatter(x, -b * np.cos(x) - P * x, s=sz, color=col, zorder=3, marker=mk)
m, M = tmin, tmax
Um, UM = -b * np.cos(m) - P * m, -b * np.cos(M) - P * M
ax.annotate("", xy=(M + 0.35, UM), xytext=(M + 0.35, Um), arrowprops={"arrowstyle": "<->", "color": GREEN, "lw": 1.5})
ax.plot([m, M + 0.35], [Um, Um], color=GREEN, lw=0.8, ls=":")
ax.text(M + 0.5, (Um + UM) / 2, "barrier", color=GREEN, fontsize=9, va="center")
ax.text(m, Um - 0.32, "stable\nminimum", color=BLUE, fontsize=8.5, ha="center", va="top", linespacing=1.2)
ax.text(M, UM + 0.22, "unstable", color=RED, fontsize=8.5, ha="center", va="bottom")
ax.text(2.6 * np.pi, -b * np.cos(2.6 * np.pi) - P * 2.6 * np.pi + 0.5, "$U = -b\\cos\\theta - P\\theta$", fontsize=10, ha="center")
ax.set_xticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$", "$2\\pi$", "$3\\pi$"])
ax.set_yticks([]); ax.set_xlabel("$\\theta$")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title("one machine against a large grid", fontsize=10)

# --- triangle: two free angles ------------------------------------------------------
P1, P2 = 0.3, -0.3   # injections at buses 1 and 2; bus 0 is the reference and takes the rest

def U2(t1, t2):
    return -np.cos(t1) - np.cos(t2) - np.cos(t1 - t2) - P1 * t1 - P2 * t2

def grad(t):
    t1, t2 = t
    return np.array([np.sin(t1) + np.sin(t1 - t2) - P1, np.sin(t2) - np.sin(t1 - t2) - P2])

def hess(t):
    t1, t2 = t
    c, c12 = np.cos(t1), np.cos(t1 - t2)
    return np.array([[c + c12, -c12], [-c12, np.cos(t2) + c12]])

crit = []
for s1 in np.linspace(-np.pi, np.pi, 25):
    for s2 in np.linspace(-np.pi, np.pi, 25):
        t = np.array([s1, s2])
        for _ in range(40):
            g = grad(t)
            try:
                t = t - np.linalg.solve(hess(t), g)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(g) < 1e-12:
                break
        if np.linalg.norm(grad(t)) < 1e-9:
            t = (t + np.pi) % (2 * np.pi) - np.pi  # the gradient is 2 pi periodic, so wrap into the square
            if not any(np.linalg.norm(t - c) < 1e-6 for c in crit):
                crit.append(t)
T1, T2 = np.meshgrid(np.linspace(-np.pi, np.pi, 300), np.linspace(-np.pi, np.pi, 300))
bx.contour(T1, T2, U2(T1, T2), levels=24, colors="0.6", linewidths=0.6)
bx.contourf(T1, T2, U2(T1, T2), levels=24, cmap="Greys", alpha=0.35)
for t in crit:
    ev = np.linalg.eigvalsh(hess(t))
    idx = int(np.sum(ev < 0))
    style = {0: {"color": BLUE, "marker": "o", "s": 60}, 1: {"color": RED, "marker": "D", "s": 45}, 2: {"color": "0.2", "marker": "^", "s": 55}}[idx]
    bx.scatter(*t, zorder=3, **style)
bx.scatter([], [], color=BLUE, marker="o", s=40, label="minimum")
bx.scatter([], [], color=RED, marker="D", s=30, label="saddle")
bx.scatter([], [], color="0.2", marker="^", s=40, label="maximum")
bx.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9, handletextpad=0.3)
bx.set_xticks([-np.pi, 0, np.pi]); bx.set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
bx.set_yticks([-np.pi, 0, np.pi]); bx.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
bx.set_xlabel("$\\theta_1$"); bx.set_ylabel("$\\theta_2$", rotation=0, labelpad=8)
bx.set_aspect("equal")
bx.set_title("a triangle, reference bus fixed", fontsize=10)
print("critical points of the triangle:", len(crit),
      "indices", sorted(int(np.sum(np.linalg.eigvalsh(hess(t)) < 0)) for t in crit))
fig.tight_layout()
fig.savefig(OUT / "landscape.png", dpi=DPI, bbox_inches="tight")
print("wrote", OUT / "landscape.png")
