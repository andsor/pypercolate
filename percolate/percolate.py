#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import numpy as np
import networkx as nx


'''

References
----------
.. [1] Newman, M. E. J. & Ziff, R. M. Fast monte carlo algorithm for site
    or bond percolation. Physical Review E 64, 016706+ (2001),
    `doi:10.1103/physreve.64.016706 <http://dx.doi.org/10.1103/physreve.64.016706>`_.

.. [2] Stauffer, D. & Aharony, A. Introduction to Percolation Theory (Taylor &
   Francis, London, 1994), second edn.

.. [3] Binder, K. & Heermann, D. W. Monte Carlo Simulation in Statistical
   Physics (Springer, Berlin, Heidelberg, 2010),
   `doi:10.1007/978-3-642-03163-2 <http://dx.doi.org/10.1007/978-3-642-03163-2>`_.
'''

def sample_states(graph, spanning_cluster=True, model='bond'):
    '''
    Generate successive sample states of the percolation model

    This is a :ref:`generator function <python:tut-generators>` to successively
    add one edge at a time from the graph to the percolation model.
    At each iteration, it calculates and returns the cluster statistics.

    Parameters
    ----------
    graph : networkx.Graph
        The substrate graph on which percolation is to take place

    spanning_cluster : bool, optional
        Defaults to ``True``.

    model : str, optional
        The percolation model (either ``'bond'`` or ``'site'``).
        Defaults to ``'bond'``.

        .. note:: Other models than ``'bond'`` are not supported yet.

    Yields
    ------
    ret : dict
        Cluster statistics

    ret['n'] : int
        Number of occupied bonds

    ret['has_spanning_cluster'] : bool
        ``True`` if there is a spanning cluster, ``False`` otherwise.

    ret['max_cluster_size'] : float
        Size of the largest cluster, as a fraction of the total number of
        sites.

    ret['moments'] : 1-D :py:class:`numpy.ndarray`
        Array of size ``5``.
        The ``k``-th entry is the ``k``-th raw moment of the cluster size
        distribution, with ``k`` ranging from ``0`` to ``4``.

    Raises
    ------
    ValueError
        If `model` does not equal ``'bond'``.

    ValueError
        If `spanning_cluster` is ``True``, but `graph` does not contain any
        auxiliary nodes to detect spanning clusters.

    Notes
    -----
    Iterating through this generator is a single run of the Newman-Ziff
    algorithm. [1]_
    The first iteration yields the trivial state with :math:`n = 0` occupied
    bonds.

    Spanning cluster

        In order to detect a spanning cluster, `graph` needs to contain
        auxiliary nodes, cf. Reference [1]_, Figure 6.
        The auxiliary nodes have the ``'span'`` `attribute
        <http://networkx.github.io/documentation/latest/tutorial/tutorial.html#node-attributes>`_.
        The value is either ``0`` or ``1``, distinguishing the two sides of the
        graph to span.

    Raw moments of the cluster size distribution

        The :math:`k`-th raw moment of the cluster size distribution is
        :math:`\sum_s' s^k n_s`, where :math:`s` is the cluster size and
        :math:`n_s` is the number of clusters of size :math:`s` per site. [2]_
        The primed sum :math:`\sum'` signifies that the largest cluster is
        excluded from the sum. [3]_

    '''

    if model != 'bond':
        raise ValueError('Only bond percolation supported.')

    if spanning_cluster:
        auxiliary_node_attributes = nx.get_node_attributes(graph, 'span')
        auxiliary_nodes = auxiliary_node_attributes.keys()
        if not list(auxiliary_nodes):
            raise ValueError(
                'Spanning cluster is to be detected, but no auxiliary nodes '
                'given.'
            )

        spanning_sides = list(set(auxiliary_node_attributes.values()))
        if len(spanning_sides) != 2:
            raise ValueError(
                'Spanning cluster is to be detected, but auxiliary nodes '
                'of less or more than 2 types (sides) given.'
            )

    # get subgraph on which percolation is to take place (strip off the
    # auxiliary nodes)
    if spanning_cluster:
        perc_graph = graph.subgraph(
            [ node for node in graph.nodes_iter() if 'span' not in graph.node[node] ]
        )
    else:
        perc_graph = graph

    # get a list of edges for easy access in later iterations
    perc_edges = perc_graph.edges()

    # number of nodes N
    num_nodes = nx.number_of_nodes(perc_graph)

    # number of edges M
    num_edges = nx.number_of_edges(perc_graph)

    # initialize union/find (disjoint set) data structure
    perc_ds = nx.utils.UnionFind()

    # initial iteration: no edges added yet (n == 0)
    ret = dict()

    ret['n'] = 0
    ret['max_cluster_size'] = 1 / num_nodes
    ret['moments'] = np.ones(5) * (num_nodes - 1) / num_nodes

    if spanning_cluster:
        ret['has_spanning_cluster'] = False

    yield ret

    # loop over all edges (n == 1..M)

