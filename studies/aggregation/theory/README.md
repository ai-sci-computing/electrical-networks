# Aggregation, the theory

Small examples for the reductions of a network: by eliminating buses, by merging clusters, and the cut a spectrum finds. Every script writes into `figures/`.

- `aggregation_figure.py`: two triangles reduced three ways (`three_reductions.png`), where the injections go under Ward's factors and under the sum over a cluster (`injection_rules.png`), the sweep cut against Cheeger's bounds (`sweep_cut.png`) and the three reductions converging on five clusters (`reductions_converge.png`); numbers to `results/aggregation_figure.log`.
- `cycles_figure.py`: a spanning tree with its fundamental cycles and the winding number of a ring (`cycle_basis.png`); needs `data/matpower/`.
- `kron_spectral_figure.py`: a Kron reduction next to the sweep cut of the Fiedler vector (`kron_spectral.png`); needs `data/matpower/`.
