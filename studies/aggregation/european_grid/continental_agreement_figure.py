"""The two ways to many parts, the cut hierarchy and spectral clustering, against each other on the
continental component: their agreement (adjusted Rand index, Hubert & Arabie 1985) and the ratio of
their normalized cuts (Shi & Malik 2000) against the number of parts, from the scores that
continental_partitions.py writes.
Run: python studies/aggregation/european_grid/continental_agreement_figure.py   (after continental_partitions.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
scores = pd.read_csv(HERE / "results" / "continental_partitions_scores.csv").sort_values("k")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax.plot(scores.k, scores.ari_recursive_kmeans, "o-", color="tab:blue", ms=3, lw=1.2)
ax.set_xscale("log"); ax.set_ylim(0, 1)
ax.set_xlabel("number of parts"); ax.set_ylabel("adjusted Rand index\nhierarchy against spectral clustering")
bx.plot(scores.k, scores.ncut_kmeans / scores.ncut_recursive, "o-", color="tab:orange", ms=3, lw=1.2)
bx.axhline(1, color="0.6", lw=0.8, ls=":")
bx.set_xscale("log"); bx.set_ylim(0, 5.5)
bx.set_xlabel("number of parts"); bx.set_ylabel("normalized cut of spectral clustering\nover that of the hierarchy")
for a in (ax, bx):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.set_xticks([2, 5, 10, 20, 50, 100]); a.set_xticklabels(["2", "5", "10", "20", "50", "100"])
fig.tight_layout()
(HERE / "figures").mkdir(exist_ok=True)
fig.savefig(HERE / "figures" / "continental_agreement.png", dpi=200, bbox_inches="tight")
print("wrote", HERE / "figures" / "continental_agreement.png")
