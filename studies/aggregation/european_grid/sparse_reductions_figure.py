"""The figure of the sparse reductions, drawn from the stored results of sparse_reductions.py.

Left: spectral error (Spielman & Srivastava 2011, eq. (2)) at the most favorable common factor (gridnet.sparsify.spectral_error_best_scale)
against the exact Kron reduction onto 96 buses, by number of
edges, for sampling the exact Kron matrix (their Sec. 1.1), approximate elimination of the base network (Durfee et al.
2017, Algorithm 3 without its last line), the same thinned by sampling, and the three networks on the 194 clustered
lines: PyPSA-Eur's, the Kron couplings on those lines, and the spectral optimum (gridnet.reweight, certified 0.690), and the deterministic selection of
Batson, Spielman & Srivastava 2012 (gridnet.sparsify.bss_sparsify, results/bss_reduction.csv).
Right: effective resistances relative to the exact ones over all pairs, median and the band from 5 to 95 percent.
Every column is its own median over ten seeds (the row is not one run); resistance quantiles are drawn only where
the median share of disconnected pairs is zero, i.e. where at least six of the ten runs are connected.

Run: .venv/bin/python studies/aggregation/european_grid/sparse_reductions_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
d = pd.read_csv(HERE / "results" / "sparse_reductions.csv")
bss = pd.read_csv(HERE / "results" / "bss_reduction.csv").sort_values("edges")     # bss_reduction.py, deterministic
ref = pd.read_csv(HERE / "results" / "sparse_reductions_references.csv").set_index("reduction")
BLUE, ORANGE, GREEN, RED, INK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#2b2b2a", "#4a3aa7"
series = [("Sparsify(Kron", BLUE, "o", "sampling the exact reduction"),
          ("ApproxSchur(base network)", ORANGE, "s", "approximate elimination"),
          ("ApproxSchur(300 copies) then Sparsify", GREEN, "^", "elimination, then sampling")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.5))
for key, color, marker, label in series:
    s = d[d.method.str.startswith(key)].sort_values("edges")
    ax.plot(s.edges, s.eps_best, color=color, marker=marker, ms=4, lw=1.6)     # at the most favorable common factor
    ok = s[s.pairs_cut == 0]
    bx.fill_between(ok.edges, ok.R_q05, ok.R_q95, color=color, alpha=0.16, lw=0)
    bx.plot(ok.edges, ok.R_median, color=color, marker=marker, ms=4, lw=1.6)
ax.plot(bss.edges, bss.eps, color=VIOLET, marker="D", ms=3.5, lw=1.6)
bx.fill_between(bss.edges, bss.R_q05, bss.R_q95, color=VIOLET, alpha=0.16, lw=0)
bx.plot(bss.edges, bss.R_median, color=VIOLET, marker="D", ms=3.5, lw=1.6)

# the four curves are named in the free lower left corner, each with its marker, in the order of the curves from the left
for row, (color, marker, label) in enumerate(((VIOLET, "D", "selection by barriers"), (BLUE, "o", "sampling the exact reduction"),
                                              (GREEN, "^", "elimination, then sampling"), (ORANGE, "s", "approximate elimination"))):
    yy = 0.34 - 0.075 * row
    ax.plot([0.04], [yy], marker=marker, ms=4, color=color, transform=ax.transAxes, clip_on=False)
    ax.text(0.08, yy, label, color=color, fontsize=8, va="center", transform=ax.transAxes)

fixed = [("clustered network (PyPSA-Eur)", RED, "PyPSA-Eur's network"), ("Kron couplings on the clustered lines", INK, "Kron couplings\non the 194 lines")]
for name, color, label in fixed:
    r = d[d.method == name].iloc[0]
    ax.scatter(r.edges, r.eps_best, marker="*", s=90, color=color, zorder=3)
    bx.errorbar(r.edges, r.R_median, yerr=[[r.R_median - r.R_q05], [r.R_q95 - r.R_median]], fmt="*", ms=9, color=color, capsize=3, lw=1.2)
ax.text(100, 0.90, "PyPSA-Eur's network", color=RED, fontsize=7.5, va="center", ha="right")
ax.text(100, 0.78, "Kron couplings on the lines", color=INK, fontsize=7.5, va="center", ha="right")
opt = ref.loc["clustered lines, spectral optimum against Kron", "eps_vs_Kron"]
ax.scatter(194, opt, marker="D", s=30, color=INK, zorder=3)
ax.text(100, opt, "best weights on the lines", color=INK, fontsize=7.5, va="center", ha="right")
bx.text(215, 0.52, "PyPSA-Eur's network", color=RED, fontsize=8, va="center")
bx.text(215, 3.3, "Kron couplings\non the 194 lines", color=INK, fontsize=8, va="center")

ax.axhline(1, color="0.6", lw=0.8, ls=":"); bx.axhline(1, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log"); bx.set_xscale("log"); bx.set_yscale("log")
ax.set_xlim(9, 4500); bx.set_xlim(150, 2600)
for a_ in (ax, bx):
    a_.set_xticks([100, 200, 500, 1000, 2000]); a_.set_xticklabels(["100", "200", "500", "1000", "2000"])
ax.set_ylabel("spectral error", fontsize=8.5); bx.set_ylabel("effective resistance over the exact one", fontsize=8.5)
bx.set_yticks([0.3, 0.5, 1, 2, 4]); bx.set_yticklabels(["0.3", "0.5", "1", "2", "4"]); bx.minorticks_off()
ax.set_ylim(0.08, 1.1); ax.set_yticks([0.1, 0.2, 0.5, 1]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
for a in (ax, bx):
    a.set_xlabel("edges of the reduced network", fontsize=8.5); a.tick_params(labelsize=8)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "figures" / "sparse_reductions_edges.png", dpi=200)
print("wrote figures/sparse_reductions_edges.png")
