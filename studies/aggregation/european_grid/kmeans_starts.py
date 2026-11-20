"""Whether more random starts of k-means change the k-way spectral partition of the continental grid.

The claim: "Hundreds of random starts change nothing, and the hierarchy's partition scores worse under
the k-means objective than the one k-means returns." The check (200 starts against 10 at
k = 5, 6, 8) had no results file; this script is that check. Same graph, embedding and seed as
continental_partitions.py (von Luxburg 2007, Sec. 4, the normalized embedding; k-means of scikit-learn); the
recursive two-way cuts are Shi & Malik 2000, Sec. 3.2. The k-means objective of a partition is the sum of the
squared distances of the embedded buses to the means of their parts.

Run:  python studies/aggregation/european_grid/kmeans_starts.py
Writes results/kmeans_starts.log.
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from gridnet.graph import laplacian
from gridnet.pypsa_io import read_osm_archive
from gridnet.spectral import ncut, recursive_bipartition, spectral_embedding

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
log = open(RESULTS / "kmeans_starts.log", "w")
say = lambda *a: (print(*a), print(*a, file=log), log.flush())
SEED = 0
KS = (5, 6, 8)

net, _ = read_osm_archive(ROOT / "data" / "raw" / "pypsa-eur-osm-0.7", component=0, trafo_x=0.1)
L = laplacian(net.n_bus, net.edges, 1 / net.x)
X = spectral_embedding(L, max(KS), normalized=True)
hist = []
recursive_bipartition(L, max(KS), normalized=True, history=hist)


def objective(Y, labels):
    return float(sum(((Y[labels == c] - Y[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels)))


say(f"continental component, {net.n_bus} buses; k-means on the first k eigenvectors of the normalized Laplacian, seed {SEED}")
for k in KS:
    Y = X[:, :k]
    km10 = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Y)
    km200 = KMeans(n_clusters=k, n_init=200, random_state=SEED).fit(Y)
    rec = np.zeros(net.n_bus, dtype=np.int64)
    for h in hist[: k - 1]:
        rec[h["new_side"]] = h["new"]
    say(f"k={k}: 10 against 200 starts ARI {adjusted_rand_score(km10.labels_, km200.labels_):.4f}, "
        f"Ncut {ncut(L, km10.labels_):.5f} and {ncut(L, km200.labels_):.5f}, k-means objective {km10.inertia_:.6f} and {km200.inertia_:.6f}; "
        f"the hierarchy's partition: Ncut {ncut(L, rec):.5f}, k-means objective {objective(Y, rec):.6f}")
