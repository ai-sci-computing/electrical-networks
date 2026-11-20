# Setup

The schematics, the map and the previews that introduce the models, and three figures for the core package. Every script writes into `figures/`; none depends on the output of another.

- `setup_figure.py`: three schematics, a phasor pair (`phasor.png`), a four-bus network with its bus types (`bus_types.png`) and a generator rotor in cross section (`rotor.png`).
- `europe_map_figure.py`: a map of the European network (`europe_map.png`); needs the datasets `osm` and `naturalearth` of `scripts/fetch_data.py`.
- `problems_figure.py`: the time frames of power system dynamics with the windows of the models (`time_scales.png`) and the potential landscape of one machine and of a triangle (`landscape.png`).
- `teaser_figures.py`: the opening images, the potential of a triangle of buses (`problems_teaser.png`) and a network of clusters as nodes (`aggregation_teaser.png`).
- `spectra_figure.py`: Laplacian spectra against closed forms (`laplacian_spectra.png`).
- `ieee14_figure.py`: the IEEE 14-bus network with its bus admittance matrix (`ieee14_admittance.png`); needs `data/matpower/`.
- `powerflow_figure.py`: the two-bus P-V curve with Newton's convergence (`powerflow.png`); needs `data/matpower/`.
