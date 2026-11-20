"""Figure for the graph basis: Laplacian spectra of path, cycle and complete graph.

Plots the numerically computed eigenvalues of the unit-weight Laplacian (gridnet.graph)
against the closed forms (Brouwer & Haemers, Spectra of Graphs, 2012, Secs. 1.4.1, 1.4.3, 1.4.4):
    path P_n:      2 - 2 cos(pi k / n)
    cycle C_n:     2 - 2 cos(2 pi k / n)
    complete K_n:  0, then n with multiplicity n - 1
Run:  python studies/foundations/setup/spectra_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; the script only writes a file
import matplotlib.pyplot as plt
import numpy as np

from gridnet.graph import laplacian

N = 12
OUT = Path(__file__).parent / "figures" / "laplacian_spectra.png"


def path_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n):
    return path_edges(n) + [(n - 1, 0)]


def complete_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


cases = [
    (f"path $P_{{{N}}}$", path_edges(N), 2 - 2 * np.cos(np.pi * np.arange(N) / N)),
    (f"cycle $C_{{{N}}}$", cycle_edges(N), 2 - 2 * np.cos(2 * np.pi * np.arange(N) / N)),
    (f"complete $K_{{{N}}}$", complete_edges(N), np.array([0.0] + [N] * (N - 1))),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
for ax, (name, edges, closed) in zip(axes, cases):
    numeric = np.linalg.eigvalsh(laplacian(N, edges))
    k = np.arange(N)
    ax.plot(k, np.sort(closed), "o", ms=8, mfc="none", color="tab:blue", label="closed form")
    ax.plot(k, numeric, ".", ms=6, color="tab:red", label="numerical")
    ax.set_title(name)
    ax.set_xlabel("index $k$")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r"eigenvalue $\lambda_k$")
axes[0].legend(loc="upper left", frameon=False)
fig.suptitle("Unit-weight Laplacian spectra: numerical vs. closed form", y=1.02)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
