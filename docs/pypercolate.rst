A Python Implementation of the Newman--Ziff algorithm
=====================================================

The *pypercolate* Python package implements the Newman--Ziff algorithm for bond
percolation on graphs.

The elementary unit of computation is the sample state:
This is one particular realization with a given number of edges—one member of
the microcanonical ensemble.
As Newman & Ziff suggest :cite:`Newman2001Fast`, the package evolves a sample
state by successively adding edges, in a random but predetermined order.
This is implemented as a generator function
:func:`percolate.percolate.sample_states` to iterate over.
Each step of the iteration adds one edge.

A collection of sample states (realizations) evolved in parallel form a
microcanonical ensemble at each iteration step.
A microcanonical ensemble is hence a collection of different sample states
(realizations) but with the same number of edges (occupation number).
The :func:`percolate.percolate.microcanonical_averages` generator function
evolves a microcanonical ensemble.
At each step, it calculates the cluster statistics over all realizations in the
ensemble.
The :func:`percolate.percolate.microcanonical_averages_arrays` helper function
collects these statistics over all iteration steps into single numpy arrays.

Finally, the :func:`percolate.percolate.canonical_averages` function calculates
the statistics of the canonical ensemble from the microcanonical ensembles.

The :func:`percolate.percolate.sample_states` generator handles cluster joining
by the `networkx.utils.union_find.UnionFind`__ data structure.

__ http://networkx.github.io/documentation/latest/reference/generated/networkx.utils.union_find.UnionFind.union.html#networkx.utils.union_find.UnionFind.union


