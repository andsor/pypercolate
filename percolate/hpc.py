# encoding: utf-8

"""
Low-level routines to implement the Newman-Ziff algorithm for HPC

See also
--------

percolate : The high-level module
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import dict, next, range

import numpy as np
import networkx as nx
import scipy.stats
import simoa


def _ndarray_dtype(fields):
    """
    Return the NumPy structured array data type

    Helper function
    """
    return [
        (np.str_(key), values)
        for key, values in fields
    ]


def microcanonical_statistics_dtype(spanning_cluster=True):
    """
    Return the numpy structured array data type for sample states

    Helper function

    Parameters
    ----------
    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    Returns
    -------
    ret : list of pairs of str
        A list of tuples of field names and data types to be used as ``dtype``
        argument in numpy ndarray constructors

    See Also
    --------
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    canonical_statistics_dtype
    """
    fields = list()
    fields.extend([
        ('n', 'uint32'),
        ('edge', 'uint32'),
    ])
    if spanning_cluster:
        fields.extend([
            ('has_spanning_cluster', 'bool'),
        ])
    fields.extend([
        ('max_cluster_size', 'uint32'),
        ('moments', '(5,)uint64'),
    ])
    return _ndarray_dtype(fields)


def bond_sample_states(
    perc_graph, num_nodes, num_edges, seed, spanning_cluster=True,
    auxiliary_node_attributes=None, auxiliary_edge_attributes=None,
    spanning_sides=None,
    **kwargs
):
    '''
    Generate successive sample states of the bond percolation model

    This is a :ref:`generator function <python:tut-generators>` to successively
    add one edge at a time from the graph to the percolation model.
    At each iteration, it calculates and returns the cluster statistics.
    CAUTION: it returns a reference to the internal array, not a copy.

    Parameters
    ----------
    perc_graph : networkx.Graph
        The substrate graph on which percolation is to take place

    num_nodes : int
        Number ``N`` of sites in the graph

    num_edges : int
        Number ``M`` of bonds in the graph

    seed : {None, int, array_like}
        Random seed initializing the pseudo-random number generator.
        Piped through to `numpy.random.RandomState` constructor.

    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    auxiliary_node_attributes : optional
        Return value of ``networkx.get_node_attributes(graph, 'span')``

    auxiliary_edge_attributes : optional
        Return value of ``networkx.get_edge_attributes(graph, 'span')``

    spanning_sides : list, optional
        List of keys (attribute values) of the two sides of the auxiliary
        nodes.
        Return value of ``list(set(auxiliary_node_attributes.values()))``

    Yields
    ------
    ret : ndarray
        Structured array with dtype ``dtype=[('has_spanning_cluster', 'bool'),
        ('max_cluster_size', 'uint32'), ('moments', 'int64', 5)]``

    ret['n'] : ndarray of int
        The number of bonds added at the particular iteration

    ret['edge'] : ndarray of int
        The index of the edge added at the particular iteration
        Note that in the first step, when ``ret['n'] == 0``, this value is
        undefined!

    ret['has_spanning_cluster'] : ndarray of bool
        ``True`` if there is a spanning cluster, ``False`` otherwise.
        Only exists if `spanning_cluster` argument is set to ``True``.

    ret['max_cluster_size'] : int
        Size of the largest cluster (absolute number of sites)

    ret['moments'] : 1-D :py:class:`numpy.ndarray` of int
        Array of size ``5``.
        The ``k``-th entry is the ``k``-th raw moment of the (absolute) cluster
        size distribution, with ``k`` ranging from ``0`` to ``4``.

    Raises
    ------
    ValueError
        If `spanning_cluster` is ``True``, but `graph` does not contain any
        auxiliary nodes to detect spanning clusters.

    See also
    --------

    numpy.random.RandomState

    microcanonical_statistics_dtype

    Notes
    -----
    Iterating through this generator is a single run of the Newman-Ziff
    algorithm. [2]_
    The first iteration yields the trivial state with :math:`n = 0` occupied
    bonds.

    Spanning cluster

        In order to detect a spanning cluster, `graph` needs to contain
        auxiliary nodes and edges, cf. Reference [2]_, Figure 6.
        The auxiliary nodes and edges have the ``'span'`` `attribute
        <http://networkx.github.io/documentation/latest/tutorial/tutorial.html#node-attributes>`_.
        The value is either ``0`` or ``1``, distinguishing the two sides of the
        graph to span.

    Raw moments of the cluster size distribution

        The :math:`k`-th raw moment of the (absolute) cluster size distribution
        is :math:`\sum_s' s^k N_s`, where :math:`s` is the cluster size and
        :math:`N_s` is the number of clusters of size :math:`s`. [3]_
        The primed sum :math:`\sum'` signifies that the largest cluster is
        excluded from the sum. [4]_

    References
    ----------
    .. [2] Newman, M. E. J. & Ziff, R. M. Fast monte carlo algorithm for site
        or bond percolation. Physical Review E 64, 016706+ (2001),
        `doi:10.1103/physreve.64.016706 <http://dx.doi.org/10.1103/physreve.64.016706>`_.

    .. [3] Stauffer, D. & Aharony, A. Introduction to Percolation Theory (Taylor &
       Francis, London, 1994), second edn.

    .. [4] Binder, K. & Heermann, D. W. Monte Carlo Simulation in Statistical
       Physics (Springer, Berlin, Heidelberg, 2010),
       `doi:10.1007/978-3-642-03163-2 <http://dx.doi.org/10.1007/978-3-642-03163-2>`_.
    '''

    # construct random number generator
    rng = np.random.RandomState(seed=seed)

    if spanning_cluster:
        if len(spanning_sides) != 2:
            raise ValueError(
                'Spanning cluster is to be detected, but auxiliary nodes '
                'of less or more than 2 types (sides) given.'
            )

    # get a list of edges for easy access in later iterations
    perc_edges = perc_graph.edges()
    perm_edges = rng.permutation(num_edges)

    # initial iteration: no edges added yet (n == 0)
    ret = np.empty(
        1, dtype=microcanonical_statistics_dtype(spanning_cluster)
    )

    ret['n'] = 0
    ret['max_cluster_size'] = 1
    ret['moments'] = np.ones(5) * (num_nodes - 1)

    if spanning_cluster:
        ret['has_spanning_cluster'] = False

    # yield cluster statistics for n == 0
    yield ret

    # set up disjoint set (union-find) data structure
    ds = nx.utils.union_find.UnionFind()
    if spanning_cluster:
        ds_spanning = nx.utils.union_find.UnionFind()

        # merge all auxiliary nodes for each side
        side_roots = dict()
        for side in spanning_sides:
            nodes = [
                node for (node, node_side) in auxiliary_node_attributes.items()
                if node_side is side
            ]
            ds_spanning.union(*nodes)
            side_roots[side] = ds_spanning[nodes[0]]

        for (edge, edge_side) in auxiliary_edge_attributes.items():
            ds_spanning.union(side_roots[edge_side], *edge)

        side_roots = [
            ds_spanning[side_root] for side_root in side_roots.values()
        ]

    # get first node
    max_cluster_root = next(perc_graph.nodes_iter())

    # loop over all edges (n == 1..M)
    for n in range(num_edges):

        ret['n'] += 1

        # draw new edge from permutation
        edge_index = perm_edges[n]
        edge = perc_edges[edge_index]
        ret['edge'] = edge_index

        # find roots and weights
        roots = [
            ds[node] for node in edge
        ]
        weights = [
            ds.weights[root] for root in roots
        ]

        if roots[0] is not roots[1]:
            # not same cluster: union!
            ds.union(*roots)
            if spanning_cluster:
                ds_spanning.union(*roots)

                ret['has_spanning_cluster'] = (
                    ds_spanning[side_roots[0]] == ds_spanning[side_roots[1]]
                )

            # find new root and weight
            root = ds[edge[0]]
            weight = ds.weights[root]

            # moments and maximum cluster size

            # deduct the previous sub-maximum clusters from moments
            for i in [0, 1]:
                if roots[i] is max_cluster_root:
                    continue
                ret['moments'] -= weights[i] ** np.arange(5)

            if max_cluster_root in roots:
                # merged with maximum cluster
                max_cluster_root = root
                ret['max_cluster_size'] = weight
            else:
                # merged previously sub-maximum clusters
                if ret['max_cluster_size'] >= weight:
                    # previously largest cluster remains largest cluster
                    # add merged cluster to moments
                    ret['moments'] += weight ** np.arange(5)
                else:
                    # merged cluster overtook previously largest cluster
                    # add previously largest cluster to moments
                    max_cluster_root = root
                    ret['moments'] += ret['max_cluster_size'] ** np.arange(5)
                    ret['max_cluster_size'] = weight

        yield ret


def bond_microcanonical_statistics(
    perc_graph, num_nodes, num_edges, seed,
    spanning_cluster=True,
    auxiliary_node_attributes=None, auxiliary_edge_attributes=None,
    spanning_sides=None,
    **kwargs
):
    """
    Evolve a single run over all microstates (bond occupation numbers)

    Return the cluster statistics for each microstate

    Parameters
    ----------
    perc_graph : networkx.Graph
        The substrate graph on which percolation is to take place

    num_nodes : int
        Number ``N`` of sites in the graph

    num_edges : int
        Number ``M`` of bonds in the graph

    seed : {None, int, array_like}
        Random seed initializing the pseudo-random number generator.
        Piped through to `numpy.random.RandomState` constructor.

    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    auxiliary_node_attributes : optional
        Value of ``networkx.get_node_attributes(graph, 'span')``

    auxiliary_edge_attributes : optional
        Value of ``networkx.get_edge_attributes(graph, 'span')``

    spanning_sides : list, optional
        List of keys (attribute values) of the two sides of the auxiliary
        nodes.
        Return value of ``list(set(auxiliary_node_attributes.values()))``

    Returns
    -------
    ret : ndarray of size ``num_edges + 1``
        Structured array with dtype ``dtype=[('has_spanning_cluster', 'bool'),
        ('max_cluster_size', 'uint32'), ('moments', 'uint64', 5)]``

    ret['n'] : ndarray of int
        The number of bonds added at the particular iteration

    ret['edge'] : ndarray of int
        The index of the edge added at the particular iteration.
        Note that ``ret['edge'][0]`` is undefined!

    ret['has_spanning_cluster'] : ndarray of bool
        ``True`` if there is a spanning cluster, ``False`` otherwise.
        Only exists if `spanning_cluster` argument is set to ``True``.

    ret['max_cluster_size'] : int
        Size of the largest cluster (absolute number of sites)

    ret['moments'] : 2-D :py:class:`numpy.ndarray` of int
        Array of shape ``(num_edges + 1, 5)``.
        The ``k``-th entry is the ``k``-th raw moment of the (absolute) cluster
        size distribution, with ``k`` ranging from ``0`` to ``4``.

    See also
    --------

    bond_sample_states
    microcanonical_statistics_dtype

    numpy.random.RandomState

    """

    # initialize generator
    sample_states = bond_sample_states(
        perc_graph=perc_graph,
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        spanning_cluster=spanning_cluster,
        auxiliary_node_attributes=auxiliary_node_attributes,
        auxiliary_edge_attributes=auxiliary_edge_attributes,
        spanning_sides=spanning_sides,
    )

    # get cluster statistics over all microstates
    return np.fromiter(
        sample_states,
        dtype=microcanonical_statistics_dtype(spanning_cluster),
        count=num_edges + 1
    )


def canonical_statistics_dtype(spanning_cluster=True):
    """
    The NumPy Structured Array type for canonical statistics

    Helper function

    Parameters
    ----------
    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    Returns
    -------
    ret : list of pairs of str
        A list of tuples of field names and data types to be used as ``dtype``
        argument in numpy ndarray constructors

    See Also
    --------
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    microcanoncial_statistics_dtype
    canonical_averages_dtype
    """
    fields = list()
    if spanning_cluster:
        fields.extend([
            ('percolation_probability', 'float64'),
        ])
    fields.extend([
        ('max_cluster_size', 'float64'),
        ('moments', '(5,)float64'),
    ])
    return _ndarray_dtype(fields)


def bond_canonical_statistics(
    microcanonical_statistics,
    convolution_factors,
    **kwargs
):
    """
    canonical cluster statistics for a single run and a single probability

    Parameters
    ----------

    microcanonical_statistics : ndarray
        Return value of `bond_microcanonical_statistics`

    convolution_factors : 1-D array_like
        The coefficients of the convolution for the given probabilty ``p``
        and for each occupation number ``n``.

    Returns
    -------
    ret : ndarray of size ``1``
        Structured array with dtype as returned by
        `canonical_statistics_dtype`

    ret['percolation_probability'] : ndarray of float
        The "percolation probability" of this run at the value of ``p``.
        Only exists if `microcanonical_statistics` argument has the
        ``has_spanning_cluster`` field.

    ret['max_cluster_size'] : ndarray of int
        Weighted size of the largest cluster (absolute number of sites)

    ret['moments'] : 1-D :py:class:`numpy.ndarray` of float
        Array of size ``5``.
        The ``k``-th entry is the weighted ``k``-th raw moment of the
        (absolute) cluster size distribution, with ``k`` ranging from ``0`` to
        ``4``.

    See Also
    --------

    bond_microcanonical_statistics
    canonical_statistics_dtype

    """
    # initialize return array
    spanning_cluster = (
        'has_spanning_cluster' in microcanonical_statistics.dtype.names
    )
    ret = np.empty(1, dtype=canonical_statistics_dtype(spanning_cluster))

    # compute percolation probability
    if spanning_cluster:
        ret['percolation_probability'] = np.sum(
            convolution_factors *
            microcanonical_statistics['has_spanning_cluster']
        )

    # convolve maximum cluster size
    ret['max_cluster_size'] = np.sum(
        convolution_factors *
        microcanonical_statistics['max_cluster_size']
    )

    # convolve moments
    ret['moments'] = np.sum(
        convolution_factors[:, np.newaxis] *
        microcanonical_statistics['moments'],
        axis=0,
    )

    # return convolved cluster statistics
    return ret


def canonical_averages_dtype(spanning_cluster=True):
    """
    The NumPy Structured Array type for canonical averages over several
    runs

    Helper function

    Parameters
    ----------
    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    Returns
    -------
    ret : list of pairs of str
        A list of tuples of field names and data types to be used as ``dtype``
        argument in numpy ndarray constructors

    See Also
    --------
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    canonical_statistics_dtype
    finalized_canonical_averages_dtype
    """
    fields = list()
    fields.extend([
        ('number_of_runs', 'uint32'),
    ])
    if spanning_cluster:
        fields.extend([
            ('percolation_probability_mean', 'float64'),
            ('percolation_probability_m2', 'float64'),
        ])
    fields.extend([
        ('max_cluster_size_mean', 'float64'),
        ('max_cluster_size_m2', 'float64'),
        ('moments_mean', '(5,)float64'),
        ('moments_m2', '(5,)float64'),
    ])
    return _ndarray_dtype(fields)


def bond_initialize_canonical_averages(
    canonical_statistics, **kwargs
):
    """
    Initialize the canonical averages from a single-run cluster statistics

    Parameters
    ----------
    canonical_statistics : 1-D structured ndarray
        Typically contains the canonical statistics for a range of values
        of the occupation probability ``p``.
        The dtype is the result of `canonical_statistics_dtype`.

    Returns
    -------
    ret : structured ndarray
        The dype is the result of `canonical_averages_dtype`.

    ret['number_of_runs'] : 1-D ndarray of int
        Equals ``1`` (initial run).

    ret['percolation_probability_mean'] : 1-D array of float
        Equals ``canonical_statistics['percolation_probability']``
        (if ``percolation_probability`` is present)

    ret['percolation_probability_m2'] : 1-D array of float
        Each entry is ``0.0``

    ret['max_cluster_size_mean'] : 1-D array of float
        Equals ``canonical_statistics['max_cluster_size']``

    ret['max_cluster_size_m2'] : 1-D array of float
        Each entry is ``0.0``

    ret['moments_mean'] : 2-D array of float
        Equals ``canonical_statistics['moments']``

    ret['moments_m2'] : 2-D array of float
        Each entry is ``0.0``

    See Also
    --------
    canonical_averages_dtype
    bond_canonical_statistics

    """
    # initialize return array
    spanning_cluster = (
        'percolation_probability' in canonical_statistics.dtype.names
    )
    # array should have the same size as the input array
    ret = np.empty_like(
        canonical_statistics,
        dtype=canonical_averages_dtype(spanning_cluster=spanning_cluster),
    )
    ret['number_of_runs'] = 1

    # initialize percolation probability mean and sum of squared differences
    if spanning_cluster:
        ret['percolation_probability_mean'] = (
            canonical_statistics['percolation_probability']
        )
        ret['percolation_probability_m2'] = 0.0

    # initialize maximum cluster size mean and sum of squared differences
    ret['max_cluster_size_mean'] = (
        canonical_statistics['max_cluster_size']
    )
    ret['max_cluster_size_m2'] = 0.0

    # initialize moments means and sums of squared differences
    ret['moments_mean'] = canonical_statistics['moments']
    ret['moments_m2'] = 0.0

    return ret


def bond_reduce(row_a, row_b):
    """
    Reduce the canonical averages over several runs

    This is a "true" reducer.
    It is associative and commutative.

    This is a wrapper around `simoa.stats.online_variance`.

    Parameters
    ----------
    row_a, row_b : structured ndarrays
        Output of this function, or initial input from
        `bond_initialize_canonical_averages`

    Returns
    -------
    ret : structured ndarray
        Array is of dtype as returned by `canonical_averages_dtype`

    See Also
    --------
    bond_initialize_canonical_averages
    canonical_averages_dtype
    simoa.stats.online_variance
    """
    spanning_cluster = (
        'percolation_probability_mean' in row_a.dtype.names and
        'percolation_probability_mean' in row_b.dtype.names and
        'percolation_probability_m2' in row_a.dtype.names and
        'percolation_probability_m2' in row_b.dtype.names
    )

    # initialize return array
    ret = np.empty_like(row_a)

    def _reducer(key, transpose=False):
        mean_key = '{}_mean'.format(key)
        m2_key = '{}_m2'.format(key)
        res = simoa.stats.online_variance(*[
            (
                row['number_of_runs'],
                row[mean_key].T if transpose else row[mean_key],
                row[m2_key].T if transpose else row[m2_key],
            )
            for row in [row_a, row_b]
        ])

        (
            ret[mean_key],
            ret[m2_key],
        ) = (
            res[1].T,
            res[2].T,
        ) if transpose else res[1:]

    if spanning_cluster:
        _reducer('percolation_probability')

    _reducer('max_cluster_size')
    _reducer('moments', transpose=True)

    ret['number_of_runs'] = row_a['number_of_runs'] + row_b['number_of_runs']

    return ret


def finalized_canonical_averages_dtype(spanning_cluster=True):
    """
    The NumPy Structured Array type for finalized canonical averages over
    several runs

    Helper function

    Parameters
    ----------
    spanning_cluster : bool, optional
        Whether to detect a spanning cluster or not.
        Defaults to ``True``.

    Returns
    -------
    ret : list of pairs of str
        A list of tuples of field names and data types to be used as ``dtype``
        argument in numpy ndarray constructors

    See Also
    --------
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    canonical_averages_dtype
    """
    fields = list()
    fields.extend([
        ('number_of_runs', 'uint32'),
        ('p', 'float64'),
        ('alpha', 'float64'),
    ])
    if spanning_cluster:
        fields.extend([
            ('percolation_probability_mean', 'float64'),
            ('percolation_probability_std', 'float64'),
            ('percolation_probability_ci', '(2,)float64'),
        ])
    fields.extend([
        ('percolation_strength_mean', 'float64'),
        ('percolation_strength_std', 'float64'),
        ('percolation_strength_ci', '(2,)float64'),
        ('moments_mean', '(5,)float64'),
        ('moments_std', '(5,)float64'),
        ('moments_ci', '(5,2)float64'),
    ])
    return _ndarray_dtype(fields)


def finalize_canonical_averages(
    number_of_nodes, ps, canonical_averages, alpha,
):
    """
    Finalize canonical averages
    """

    spanning_cluster = (
        (
            'percolation_probability_mean' in
            canonical_averages.dtype.names
        ) and
        'percolation_probability_m2' in canonical_averages.dtype.names
    )

    # append values of p as an additional field
    ret = np.empty_like(
        canonical_averages,
        dtype=finalized_canonical_averages_dtype(
            spanning_cluster=spanning_cluster
        ),
    )

    n = canonical_averages['number_of_runs']
    sqrt_n = np.sqrt(canonical_averages['number_of_runs'])

    ret['number_of_runs'] = n
    ret['p'] = ps
    ret['alpha'] = alpha

    def _transform(
        original_key, final_key=None, normalize=False, transpose=False,
    ):
        if final_key is None:
            final_key = original_key
        keys_mean = [
            '{}_mean'.format(key)
            for key in [original_key, final_key]
        ]
        keys_std = [
            '{}_m2'.format(original_key),
            '{}_std'.format(final_key),
        ]
        key_ci = '{}_ci'.format(final_key)

        # calculate sample mean
        ret[keys_mean[1]] = canonical_averages[keys_mean[0]]
        if normalize:
            ret[keys_mean[1]] /= number_of_nodes

        # calculate sample standard deviation
        array = canonical_averages[keys_std[0]]
        result = np.sqrt(
            (array.T if transpose else array) / (n - 1)
        )
        ret[keys_std[1]] = (
            result.T if transpose else result
        )
        if normalize:
            ret[keys_std[1]] /= number_of_nodes

        # calculate standard normal confidence interval
        array = ret[keys_std[1]]
        scale = (array.T if transpose else array) / sqrt_n
        array = ret[keys_mean[1]]
        mean = (array.T if transpose else array)
        result = scipy.stats.t.interval(
            1 - alpha,
            df=n - 1,
            loc=mean,
            scale=scale,
        )
        (
            ret[key_ci][..., 0], ret[key_ci][..., 1]
        ) = ([array.T for array in result] if transpose else result)

    if spanning_cluster:
        _transform('percolation_probability')

    _transform('max_cluster_size', 'percolation_strength', normalize=True)
    _transform('moments', normalize=True, transpose=True)

    return ret
