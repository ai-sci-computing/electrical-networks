"""The example that separates cut approximation from spectral approximation.

Spielman & Teng 2011 (arXiv:0808.4134), Sec. 5.3, Example 3: n groups of k vertices on a ring, consecutive groups
joined by complete bipartite graphs, plus one chord from a vertex of group 0 to a vertex of group n/2. Here n = 16,
k = 2, all weights one (their own drawing has n = 8, k = 4, where both factors below are 1.125 and the point does
not show). In the closed forms n counts the groups; the right panel of the figure counts buses on its axis and
in its label, n_buses = k n, so that the energy factor reads 1 + n_buses/(4 k^3) there. Closed forms, tested in tests/test_sparsify.py:

  cuts     the minimum cut is 2k = 4, so dropping the chord changes no cut by more than 1 + 1/(2k) = 1.25,
           attained by the cut that isolates one end of the chord (5 edges with the chord, 4 without);
  energy   for x(u, i) = min(u, n - u), the distance around the ring from group 0, every ring edge has difference 1
           and there are n k^2 = 64 of them; the chord has difference n/2 = 8 and energy 64; the form is 128 with
           the chord and 64 without, a factor 1 + n/(4 k^2) = 2, which grows with n while the cut factor does not.

Writes figures/cut_vs_spectral.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridnet.graph import laplacian

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
N, K = 16, 2
INK, RED, BLUE = "#2b2b2a", "#e34948", "#2a78d6"

idx = lambda u, i: (u % N) * K + i
ring = [(idx(u, i), idx(u + 1, j)) for u in range(N) for i in range(K) for j in range(K)]
chord = (idx(0, 0), idx(N // 2, 0))
L_without = laplacian(N * K, ring, [1.0] * len(ring))
L_with = laplacian(N * K, ring + [chord], [1.0] * (len(ring) + 1))
x = np.repeat([min(u, N - u) for u in range(N)], K).astype(float)
e_without, e_with = float(x @ L_without @ x), float(x @ L_with @ x)
assert (e_without, e_with) == (N * K**2, N * K**2 + (N // 2) ** 2)

# positions: group u at angle, its two vertices on an inner and an outer circle; group 0 at the left
ang = np.pi - 2 * np.pi * np.arange(N) / N
radius = (0.82, 1.0)
XY = np.array([[radius[i] * np.cos(ang[u]), radius[i] * np.sin(ang[u])] for u in range(N) for i in range(K)])

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})

# --- the vector x: every ring edge has difference one, the chord has difference n/2 -------------------------------
ax = axes[0]
for (a, b) in ring:
    ax.plot(*XY[[a, b]].T, color="0.55", lw=0.7, zorder=1)
ax.plot(*XY[list(chord)].T, color=RED, lw=2.2, zorder=2)
sc = ax.scatter(*XY.T, c=x, cmap="Blues", vmin=-2, vmax=N // 2, s=34, edgecolor=INK, lw=0.6, zorder=3)
for u in range(N):
    ax.text(1.2 * np.cos(ang[u]), 1.2 * np.sin(ang[u]), f"{min(u, N - u)}", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(0, 0.1, f"difference {N // 2}", ha="center", va="bottom", fontsize=7.5, color=RED)
ax.text(0, -1.5, f"{len(ring)} ring edges, difference 1, energy 1 each\nthe chord, difference {N // 2}, energy {(N // 2) ** 2}\n"
        f"energy of $\\theta$: {e_with:.0f} with the chord, {e_without:.0f} without", ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.35)

# --- the cut that the chord changes most: one end of the chord against the rest --------------------------------------
ax = axes[1]
end = chord[0]
for (a, b) in ring:
    severed = end in (a, b)
    ax.plot(*XY[[a, b]].T, color=INK if severed else "0.75", lw=1.6 if severed else 0.6, ls=(0, (2.5, 1.5)) if severed else "-", zorder=1)
ax.plot(*XY[list(chord)].T, color=RED, lw=2.2, ls=(0, (2.5, 1.5)), zorder=2)
ax.scatter(*XY.T, s=34, color="white", edgecolor=INK, lw=0.6, zorder=3)
ax.scatter(*XY[end], s=60, color=BLUE, edgecolor=INK, lw=0.8, zorder=4)
ax.text(0, -1.5, "the cut that the chord changes most,\none bus against the rest\ncut: 5 with the chord, 4 without",
        ha="center", va="top", fontsize=7.5, color=INK, linespacing=1.35)
for a_ in axes[:2]:
    a_.set_xlim(-1.4, 1.4); a_.set_ylim(-2.35, 1.4); a_.set_aspect("equal"); a_.axis("off")

# --- the two factors against the length of the ring -------------------------------------------------------------------
bx = axes[2]
g = np.arange(4, 65, 2)                                    # groups on the ring; the axis counts buses, n = K g
n = K * g
bx.plot(n, 1 + g / (4 * K**2), color=RED, lw=1.8)
bx.plot(n, np.full(n.shape, 1 + 1 / (2 * K)), color=BLUE, lw=1.8)
bx.scatter([N * K, N * K], [1 + N / (4 * K**2), 1 + 1 / (2 * K)], s=26, color=[RED, BLUE], zorder=3)
bx.text(72, 1 + 44 / (4 * K**2), f"energy of $\\theta$\n$1 + n/{4 * K**3}$", color=RED, fontsize=8, ha="right", va="bottom")
bx.text(128, 1.25 - 0.1, f"every cut, at most {1 + 1 / (2 * K):.2f}", color=BLUE, fontsize=8, ha="right", va="top")
bx.set_xlabel(f"buses on the ring, $n$\n(groups of {K})", fontsize=8)
bx.set_ylabel("with the chord over without", fontsize=8)
bx.set_ylim(0.7, 5.2); bx.set_xticks([8, 32, 64, 96, 128]); bx.tick_params(labelsize=7.5)
for side in ("top", "right"):
    bx.spines[side].set_visible(False)
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "cut_vs_spectral.png", dpi=200)
print(f"energy of x: {e_with:.0f} with the chord, {e_without:.0f} without, factor {e_with / e_without:.3f}; cut factor at most {1 + 1 / (2 * K):.3f}")
