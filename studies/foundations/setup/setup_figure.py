"""Three schematics of the setup (phasors, bus types, the generator), each sized for a narrow column.

figures/phasor.png     A voltage and a current at the grid frequency and their phasors, the complex
                       numbers |V| exp(j theta) that encode amplitude and phase; |V| is the RMS value,
                       the amplitude divided by sqrt 2, as in steady-state analysis (Kundur 1994,
                       Sec. 3.6.2, p. 95); the current lags the voltage by the phase of the impedance.
figures/bus_types.png  A four-bus network with the quantities that live on it. Voltage phasors on the
                       buses, admittances and currents on the oriented branches, power injected by
                       generators and drawn by loads, and at each bus the given and the unknown
                       real quantities of its type, generator (PV) and load (PQ), with one
                       generator bus the angle reference and the generators sharing the balancing
                       power (MATPOWER manual 2024, Sec. 3.1-3.2 and 4.1; Kundur 1994, Sec. 6.4.1;
                       Van Cutsem & Vournas 1998, Sec. 6.5.4, and Avalos et al. 2009, eq. (13), for
                       the sharing in place of a single slack bus).
figures/rotor.png      Cross section of a synchronous generator. The rotor is a magnet spinning inside
                       the stator winding, its angle theta is measured from an axis rotating at the
                       grid frequency, the turbine torque accelerates it and the electrical torque of
                       the network brakes it, and Newton's law for the angle is the swing equation
                       (Machowski 2020, Sec. 5.1; Kundur 1994, Sec. 3.9 and 13.1).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Circle, Rectangle, Wedge

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DPI = 200

BLUE, ORANGE, GREEN, RED, GRAY = "tab:blue", "tab:orange", "tab:green", "tab:red", "0.45"


def arrow(ax, start, end, color, lw=1.8, scale=14):
    """Straight arrow from start to end."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "mutation_scale": scale, "shrinkA": 0, "shrinkB": 0})


def arc_arrow(ax, r, a0, a1, color, center=(0.0, 0.0), lw=2.0, zorder=4):
    """Circular arc of radius r from angle a0 to a1 (degrees) with an arrowhead at a1."""
    a = np.radians(np.linspace(a0, a1, 60))
    x, y = center[0] + r * np.cos(a), center[1] + r * np.sin(a)
    ax.plot(x[:-2], y[:-2], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-3], y[-3]), zorder=zorder,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "mutation_scale": 16, "shrinkA": 0, "shrinkB": 0})


# =====================================================================================
# 1. Phasors: a sinusoid at fixed frequency is a complex number
# =====================================================================================
fig, (ax, px) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})
theta, phi = np.radians(35), np.radians(50)  # voltage phase, impedance angle (current lags by phi)
t = np.linspace(0, 2, 500)
r2 = np.sqrt(2)  # the phasor magnitude is the RMS value, the sinusoid's amplitude is sqrt 2 times it
ax.plot(t, r2 * np.cos(2 * np.pi * t + theta), color=BLUE, lw=1.8,
        label="voltage $\\sqrt{2}\\,|V|\\cos(\\omega_s t + \\theta)$")
ax.plot(t, r2 * 0.7 * np.cos(2 * np.pi * t + theta - phi), color=ORANGE, lw=1.8,
        label="current $\\sqrt{2}\\,|I|\\cos(\\omega_s t + \\theta - \\varphi)$")
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xlim(0, 2); ax.set_ylim(-1.9, 1.9)
ax.set_xlabel("time in cycles of 50 Hz")
ax.set_yticks([-r2, 0, r2], ["$-\\sqrt{2}$", "0", "$\\sqrt{2}$"])
ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9, borderpad=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

px.add_patch(Circle((0, 0), 1, fill=False, color="0.75", lw=0.8))
px.plot([-1.25, 1.25], [0, 0], color="0.85", lw=0.8)
px.plot([0, 0], [-1.25, 1.25], color="0.85", lw=0.8)
arrow(px, (0, 0), (np.cos(theta), np.sin(theta)), BLUE, lw=2.2, scale=16)
arrow(px, (0, 0), (0.7 * np.cos(theta - phi), 0.7 * np.sin(theta - phi)), ORANGE, lw=2.2, scale=16)
a = np.radians(np.linspace(0, 35, 30))
px.plot(0.45 * np.cos(a), 0.45 * np.sin(a), color=BLUE, lw=0.9)
px.text(0.5, 0.11, "$\\theta$", color=BLUE, fontsize=10)
a = np.radians(np.linspace(-15, 35, 30))
px.plot(0.3 * np.cos(a), 0.3 * np.sin(a), color=ORANGE, lw=0.9)
px.text(0.33, -0.12, "$\\varphi$", color=ORANGE, fontsize=10, va="top")
px.text(0.72, 0.72, "$V = |V|e^{\\mathrm{j}\\theta}$", color=BLUE, fontsize=10)
px.text(0.72, -0.35, "$I = V/z$", color=ORANGE, fontsize=10)
px.text(1.2, -0.1, "Re", color="0.5", fontsize=8, ha="right", va="top")
px.text(0.05, 1.15, "Im", color="0.5", fontsize=8, va="center")
px.set_xlim(-1.3, 1.6); px.set_ylim(-1.3, 1.3); px.set_aspect("equal"); px.axis("off")
fig.tight_layout()
fig.savefig(OUT / "phasor.png", dpi=DPI, bbox_inches="tight")
print("wrote", OUT / "phasor.png")

# =====================================================================================
# 2. Bus types: what lives on the graph and what is given at each bus
# =====================================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.2))
pos = {1: (0.0, 2.4), 2: (3.4, 2.4), 3: (3.4, 0.0), 4: (0.0, 0.0)}
edges = [(1, 2), (2, 3), (4, 3), (1, 4), (1, 3)]  # oriented tail -> head
R = 0.27  # bus radius in data units

for (a_, b_) in edges:
    (x1, y1), (x2, y2) = pos[a_], pos[b_]
    ax.plot([x1, x2], [y1, y2], color="0.6", lw=2.2, zorder=1)
    ux, uy = (x2 - x1), (y2 - y1)
    L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    arrow(ax, (mx - 0.22 * ux, my - 0.22 * uy), (mx + 0.22 * ux, my + 0.22 * uy), ORANGE, lw=2.0, scale=15)
    nx_, ny_ = -uy, ux  # left normal
    if (a_, b_) == (1, 2):
        off = (0, 0.28)
    elif (a_, b_) == (4, 3):
        off = (0, -0.28)
    elif (a_, b_) == (2, 3):
        off = (0.66, 0)
    elif (a_, b_) == (1, 4):
        off = (-0.66, 0)
    else:
        off = (0.55 * nx_, 0.55 * ny_)
    ax.text(mx + off[0], my + off[1], f"$y_{{{a_}{b_}}}$,  $I_{{{a_}{b_}}}$", fontsize=10, color="0.25",
            ha="center", va="center", zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})

for k, (x, y) in pos.items():
    ax.add_patch(Circle((x, y), R, color=BLUE, zorder=3))
    ax.text(x, y, f"$V_{k}$", color="white", fontsize=10, ha="center", va="center", zorder=4)

info = {
    1: ("generator bus, reference", "given its schedule, $|V_1|$, $\\theta_1 = 0$", "unknown $Q_1$"),
    2: ("generator bus", "given its schedule and $|V_2|$", "unknown $Q_2$, $\\theta_2$"),
    3: ("load bus", "given $P_3$ and $Q_3$", "unknown $|V_3|$, $\\theta_3$"),
    4: ("load bus", "given $P_4$ and $Q_4$", "unknown $|V_4|$, $\\theta_4$"),
}
side = {1: -1, 2: 1, 3: 1, 4: -1}
for k, (x, y) in pos.items():
    name, given, unknown = info[k]
    ax.text(x + side[k] * 0.5, y + 0.02, f"{name}\n{given}\n{unknown}", fontsize=9.5,
            ha="right" if side[k] < 0 else "left", va="center", linespacing=1.4)

# generators feeding the two upper buses, loads drawing from the two lower buses
for k in (1, 2):
    x, y = pos[k]
    ax.add_patch(Circle((x, y + 1.15), 0.3, fill=False, color=GREEN, lw=1.8))
    s = np.linspace(-0.2, 0.2, 40)
    ax.plot(x + s, y + 1.15 + 0.1 * np.sin(s / 0.2 * np.pi), color=GREEN, lw=1.4)
    arrow(ax, (x, y + 0.85), (x, y + R), GREEN, lw=2.0, scale=15)
    ax.text(x + 0.12, y + 0.6, f"$S_{k} = P_{k} + \\mathrm{{j}}Q_{k}$", color=GREEN, fontsize=9.5, va="center")
ax.text(1.7, 3.55, "$P_i$ = schedule + $c_i P^{\\mathrm{bal}}$", color=GREEN, fontsize=9.5, ha="center", va="center")
for k in (3, 4):
    x, y = pos[k]
    arrow(ax, (x, y - R), (x, y - 0.95), RED, lw=2.0, scale=15)
    ax.text(x + 0.12, y - 0.6, f"$P^{{\\rm d}}_{k} + \\mathrm{{j}}Q^{{\\rm d}}_{k}$", color=RED, fontsize=9.5, va="center")
ax.text(0.0, 3.85, "generator", color=GREEN, fontsize=9, ha="center", va="bottom")
ax.text(0.0, -1.1, "load", color=RED, fontsize=9, ha="center", va="top")

ax.set_xlim(-2.9, 6.3); ax.set_ylim(-1.55, 4.2); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "bus_types.png", dpi=DPI, bbox_inches="tight")
print("wrote", OUT / "bus_types.png")

# =====================================================================================
# 3. Rotor: the phase of a generator is the angle of a spinning mass
# =====================================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
r_in, r_out = 1.35, 1.62
ang = 32.0  # rotor angle theta in degrees

# stator: an annulus with the coil sides of the winding as small circles on its inner rim
ax.add_patch(Wedge((0, 0), r_out, 0, 360, width=r_out - r_in, fc="0.88", ec="0.55", lw=1.2, zorder=1))
for a in np.radians(np.arange(15, 360, 30)):
    ax.add_patch(Circle((1.45 * np.cos(a), 1.45 * np.sin(a)), 0.075, fc="white", ec="0.45", lw=1.0, zorder=2))
ax.text(-1.15, 1.45, "stator\nwinding", fontsize=9, color="0.35", ha="right", va="center", linespacing=1.3)

# rotor: a bar magnet on the shaft, north half red, south half blue, turned by theta
rot = transforms.Affine2D().rotate_deg(ang) + ax.transData
Lb, wb = 1.15, 0.46
ax.add_patch(Rectangle((0, -wb / 2), Lb, wb, fc="#e8a0a0", ec="0.3", lw=1.0, transform=rot, zorder=3))
ax.add_patch(Rectangle((-Lb, -wb / 2), Lb, wb, fc="#a0b8e8", ec="0.3", lw=1.0, transform=rot, zorder=3))
ax.text(0.75 * np.cos(np.radians(ang)), 0.75 * np.sin(np.radians(ang)), "N", fontsize=11, ha="center",
        va="center", zorder=5, rotation=ang)
ax.text(-0.75 * np.cos(np.radians(ang)), -0.75 * np.sin(np.radians(ang)), "S", fontsize=11, ha="center",
        va="center", zorder=5, rotation=ang)
ax.add_patch(Circle((0, 0), 0.1, fc="0.3", ec="0.3", zorder=6))

# reference axis rotating at grid frequency, and the angle theta from it to the rotor
ax.plot([0, 2.35], [0, 0], color="0.45", lw=1.0, ls="--", zorder=2)
ax.text(2.42, 0.0, "reference axis,\nrotating at 50 Hz", fontsize=9, color="0.35", va="center", linespacing=1.3)
a = np.radians(np.linspace(0, ang, 30))
ax.plot(1.05 * np.cos(a), 1.05 * np.sin(a), color="0.2", lw=1.0, zorder=5)
ax.text(1.14 * np.cos(np.radians(ang / 2)), 1.14 * np.sin(np.radians(ang / 2)), "$\\theta$", fontsize=12,
        ha="left", va="center", zorder=5)

# torques: the turbine drives the shaft forward, the network pulls the rotor back
arc_arrow(ax, 0.24, 250, 520, GREEN, lw=2.4, zorder=7)
# the labels sit in the two quadrants the magnet leaves free, clear of the arcs and the bar
ax.text(0.62, -0.55, "turbine\ntorque $P^{\\rm m}$", color=GREEN, fontsize=9.5, ha="center", va="center",
        linespacing=1.3, zorder=5)
arc_arrow(ax, 1.2, 150, 75, RED)
ax.text(-0.45, 0.55, "electrical\ntorque $P(\\theta, |V|)$", color=RED, fontsize=9.5, ha="center", va="center",
        linespacing=1.3, zorder=5)

# terminals: the induced voltage leaves the stator toward the network
for dy in (-0.08, 0.08):
    ax.plot([r_out, 2.2], [-1.0 + dy, -1.0 + dy], color="0.3", lw=1.2, zorder=2)
ax.text(2.3, -1.0, "induced voltage $|V|e^{\\mathrm{j}\\theta}$\nto the network", fontsize=9, color="0.2", va="center",
        linespacing=1.3)

ax.text(0.0, -1.95, "$M\\ddot\\theta + D\\dot\\theta = P^{\\rm m} - P(\\theta, |V|)$", fontsize=12,
        ha="center", va="top")
ax.set_xlim(-2.5, 4.6); ax.set_ylim(-2.45, 1.85); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "rotor.png", dpi=DPI, bbox_inches="tight")
print("wrote", OUT / "rotor.png")
