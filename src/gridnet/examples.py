"""Small constructed networks whose flows and cascades have closed forms.

Shared by the tests and by the scripts that draw the same cases, so that both use one
definition.
"""

from __future__ import annotations

import numpy as np

from .network import Network

__all__ = ["lossless_network", "parallel_lines", "ring"]


def lossless_network(n, edges, b, ref=0) -> Network:
    """Lossless network with unit taps and reactances x = 1/b (so that b = 1/x).

    `edges` are pairs of bus indices from 0 to n - 1, `b` their susceptances,
    `ref` the index of the reference bus. Loads, shunts and ratings are zero.
    """
    m = len(edges)
    f = np.array([e[0] for e in edges]); t = np.array([e[1] for e in edges])
    bus_type = np.ones(n, dtype=int); bus_type[ref] = 3
    return Network(
        base_mva=100.0, bus_id=np.arange(1, n + 1), bus_type=bus_type,
        pd=np.zeros(n), qd=np.zeros(n), gs=np.zeros(n), bs=np.zeros(n),
        from_bus=f, to_bus=t, r=np.zeros(m), x=1 / np.asarray(b, float), b=np.zeros(m),
        tap=np.ones(m), shift=np.zeros(m), status=np.ones(m, dtype=int), rate_a=np.zeros(m),
    )


def parallel_lines(n, load, rating):
    """Two buses, a generator at bus 0 that supplies `load` at bus 1 over n identical lines."""
    net = lossless_network(2, [(0, 1)] * n, [1.0] * n)
    net.rate_a[:] = rating
    pg = np.array([load, 0.0]); pd = np.array([0.0, load])
    return net, pg, pd


def ring(n, d, rating):
    """A ring of n buses with unit susceptances, bus 0 generating (n - 1) d and every other bus
    drawing d."""
    net = lossless_network(n, [(k, (k + 1) % n) for k in range(n)], [1.0] * n)
    net.rate_a[:] = rating
    pg = np.zeros(n); pg[0] = (n - 1) * d
    pd = np.full(n, d); pd[0] = 0.0
    return net, pg, pd
