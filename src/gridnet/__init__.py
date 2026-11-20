"""gridnet: electrical networks as weighted graphs.

Common basis for all studies: incidence matrix, weighted Laplacians (the
susceptance Laplacian B and the operating-point Laplacian), conventions for
transformers, shunts and phase shifters, cycle basis from a spanning tree, Kron
reduction, PTDF/LODF.

Every function's docstring names the publication (author, year, section or equation)
whose method it implements; full citations are in references.bib. Every module is
developed red-green and verified against analytical cases before real networks.
"""

__version__ = "0.0.1"
