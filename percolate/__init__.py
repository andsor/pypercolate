#!/usr/bin/env python
# encoding: utf-8

"""
Implements the Newman-Ziff algorithm for Monte Carlo simulation of percolation

This module implements the Newman-Ziff algorithm for Monte Carlo simulation of
Bernoulli percolation on arbitrary graphs.

The :mod:`percolate` module provides these high-level functions from the
:mod:`percolate.percolate` module:

.. autosummary::

   percolate.sample_states
   percolate.single_run_arrays
   percolate.microcanonical_averages
   percolate.microcanonical_averages_arrays
   percolate.canonical_averages
   percolate.spanning_1d_chain
   percolate.spanning_2d_grid
   percolate.statistics

See Also
--------

percolate.percolate : low-level functions

Notes
-----

Currently, the module only implements bond percolation.
Spanning cluster detection is implemented, but wrapping detection is not.

The elementary unit of computation is the *sample state*:
This is one particular realization with a given number of edges---one member of
the *microcanonical ensemble*.
As Newman & Ziff suggest [1]_, the module evolves a sample state by
successively adding edges, in a random but predetermined order.
This is implemented as a generator function :func:`sample_states` to iterate
over.
Each step of the iteration adds one edge.

A collection of sample states (realizations) evolved in parallel form a
*microcanonical ensemble* at each iteration step.
A microcanonical ensemble is hence a collection of different sample states
(realizations) but with the same number of edges (*occupation number*).
The :func:`microcanonical_averages` generator function evolves a microcanonical
ensemble.
At each step, it calculates the cluster statistics over all realizations in the
ensemble.
The :func:`microcanonical_averages_arrays` helper function collects these
statistics over all iteration steps into single numpy arrays.

Finally, the :func:`canonical_averages` function calculates the statistics of
the *canonical ensemble* from the microcanonical ensembles.

References
----------

.. [1] Newman, M. E. J. & Ziff, R. M. Fast monte carlo algorithm for site
   or bond percolation. Physical Review E 64, 016706+ (2001),
   `10.1103/physreve.64.016706`__

__ http://dx.doi.org/10.1103/physreve.64.016706


.. todo::

   `Implement site percolation`__

__ https://github.com/andsor/pypercolate/issues/5

.. todo::

   `Implement wrapping detection`__

__ https://github.com/andsor/pypercolate/issues/6

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from percolate.percolate import (
    sample_states,
    single_run_arrays,
    microcanonical_averages,
    microcanonical_averages_arrays,
    canonical_averages,
    spanning_1d_chain,
    spanning_2d_grid,
    statistics,
)

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
