# The European grid: clusters, cuts and sparse reductions

Spectral partition of the unsimplified European transmission network (PyPSA-Eur `base_network`) compared with the partitions PyPSA-Eur produces after `simplify_network`. PyPSA-Eur's outputs come from its run at the commit recorded in `PYPSA_EUR_COMMIT` and are kept in `data/raw/pypsa-eur-gridnet/`.

## Data

The OSM base network is the versioned archive PyPSA-Eur's `retrieve_osm_archive` rule downloads (version in `PYPSA_EUR_COMMIT`). Fetch the five files into `data/raw/pypsa-eur-osm-0.7/` and the Natural Earth borders for the maps into `data/raw/naturalearth/`:

```
python scripts/fetch_data.py --only osm naturalearth
python studies/aggregation/european_grid/kill_point.py
```

## PyPSA-Eur's own clustering (the comparison target)

PyPSA-Eur is fetched as the source archive of the commit in `PYPSA_EUR_COMMIT` into `external/pypsa-eur/` of this repository (ignored by git), with its environment created by micromamba from its `envs/default_osx-arm64.pin.txt` under `external/mamba/`, and `config.gridnet.yaml` of this directory copied to its `config/`. With the root prefix and the caches inside `external/` (`MAMBA_ROOT_PREFIX`, `HOME` and the `XDG_*` directories set there), the run writes nothing outside the repository:

```
cd external/pypsa-eur
snakemake -call resources/gridnet/networks/clustered.nc resources/gridnet/electricity_demand_simplified.nc \
    --configfile config/config.gridnet.yaml --cores 4
```

with the environment's `bin/` first on the path; 28 jobs, about half an hour, the downloads from data.pypsa.org and zenodo.org. The five outputs used here are copied to `data/raw/pypsa-eur-gridnet/` (their SHA-256 in its `SOURCE.md`): `busmap_simplify_network.csv` and `busmap_cluster_network.csv`, which map base-network buses to simplified buses and simplified buses to clusters, `networks/clustered.nc`, `networks/simplified.nc` and `electricity_demand_simplified.nc`. The checkout and the environment (about 5 GB) can be deleted afterwards.

`kron_vs_cluster.py` compares PyPSA-Eur's 128-cluster network with the Kron reduction of the base network onto one bus per cluster: aggregated against exact couplings on the clustered lines, effective resistances for all cluster pairs, and the share of exact coupling the sparse aggregation has no line for. Needs the archive and `data/raw/pypsa-eur-gridnet/`.

`continental_partitions.py` partitions the whole continental component into $k$ parts with borders ignored, compares with the countries and with PyPSA-Eur's clusters, and lists the recursive cuts after the Pyrenees.

## Sparse reductions and the shared cache

`sparse_reductions.py` runs first: it builds `results/kron128.npz` (named after the 128-cluster run; the Kron reduction of the base network onto one bus per PyPSA-Eur cluster of the continental component, 96 of them, the clustered network's couplings on the same clusters, the base network's leverage scores) and rebuilds it whenever one of its inputs changes (the file carries a SHA-256 fingerprint of the PyPSA-Eur outputs, `PYPSA_EUR_COMMIT`, the archive and the transformer reactance). The scripts that read the cache come after it: `bss_reduction.py` (Batson, Spielman & Srivastava's deterministic sparsifier), `measure_references.py` (both measures against one reference at a time), `matched_reference.py` and `matched_reference_figure.py` (the reference that matches a clustering), `coarsening.py` (Loukas' coarsening against the partitions; needs the `coarsening` extra), `cluster_structure.py` (three measurements of how well-clustered the grid is), `injection_map.py` (Ward's distribution of the eliminated injections), `pyrenees_distance.py` (the Iberian cut as a distance), and the figures `sparse_reductions_figure.py`, `sparse_maps_figure.py`, `sparse_teaser_figure.py`, `teaser_figure.py`.

Also here: `continental_laplacian.py` (the susceptance Laplacian of the continental synchronous area and its first sweep cut, the Pyrenees), `country_partitions.py` (partitions within countries, electrical against geographic), `pypsa_eur_comparison.py` (PyPSA-Eur's clustering against the electrical partition per country), and `rei_example.py` (the REI equivalent of Tinney & Powell on IEEE 14).

Further scripts: `_clustered.py` (the clustered network and the base network on the same clusters, shared by `kron_vs_cluster.py` and `sparse_reductions.py`), `continental_agreement_figure.py` (the cut hierarchy against spectral clustering on the continental component, their adjusted Rand index and the ratio of their normalized cuts, from the scores of `continental_partitions.py`), `kmeans_starts.py` (whether more random starts of $k$-means change the spectral partition of the continental grid) and `loukas_paper_check.py` (Loukas's local variation as his code does it against the algorithm of his paper, with the switchable changes of `third_party/coarsening_paper.py`).

`sparsification_example_figure.py`: Spielman and Teng's ring of groups with one chord, where dropping the chord changes every cut by at most 1.25 and the energy by a factor that grows with the ring (`cut_vs_spectral.png`).
