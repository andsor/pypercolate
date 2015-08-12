# encoding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import functools
import string

import hypothesis
import hypothesis.strategies as st

import networkx as nx
import numpy as np
import percolate.percolate
import percolate.hpc
import pytest
import scipy.stats

MAX_SEED = 4294967296 - 1


def test_numpy_dtype_string():
    """
    Test Python2/3 string compatibility with Python data type
    """
    np.empty(1, dtype=[(np.str_('a'), '?')])


@hypothesis.given(
    st.lists(
        elements=st.tuples(
            st.text(alphabet=string.printable, min_size=1,),
            st.builds(
                '{}{}'.format,
                st.one_of(
                    st.just(''),
                    st.builds(
                        tuple,
                        st.lists(
                            elements=st.integers(min_value=1, max_value=100),
                            min_size=1,
                            max_size=2,
                        )
                    )
                ),
                st.sampled_from(
                    np.typecodes['AllFloat'] + np.typecodes['AllInteger']
                ),
            ),
        ),
        min_size=1,
        unique_by=lambda x: x[0],
    ),
    settings=hypothesis.Settings(max_examples=10),
)
def test_ndarray_type(fields):
    """
    Test that _ndarray_dtype returns a valid NumPy array dtype
    """
    dtype = percolate.hpc._ndarray_dtype(fields=fields)
    np.empty(1, dtype=dtype)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_microcanonical_statistics_dtype(spanning_cluster):
    dtype = percolate.hpc.microcanonical_statistics_dtype(
        spanning_cluster=spanning_cluster
    )
    np.empty(1, dtype=dtype)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_microcanonical_statistics_dtype_spanning_cluster(spanning_cluster):
    dtype = percolate.hpc.microcanonical_statistics_dtype(
        spanning_cluster=spanning_cluster
    )
    array = np.empty(1, dtype=dtype)
    assert spanning_cluster == ('has_spanning_cluster' in array.dtype.names)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_macrocanonical_statistics_dtype(spanning_cluster):
    dtype = percolate.hpc.macrocanonical_statistics_dtype(
        spanning_cluster=spanning_cluster
    )
    np.empty(1, dtype=dtype)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_macrocanonical_statistics_dtype_spanning_cluster(spanning_cluster):
    dtype = percolate.hpc.macrocanonical_statistics_dtype(
        spanning_cluster=spanning_cluster
    )
    array = np.empty(1, dtype=dtype)
    assert spanning_cluster == ('percolation_probability' in array.dtype.names)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_macrocanonical_averages_dtype(spanning_cluster):
    dtype = percolate.hpc.macrocanonical_averages_dtype(
        spanning_cluster=spanning_cluster
    )
    np.empty(1, dtype=dtype)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_macrocanonical_averages_dtype_spanning_cluster(spanning_cluster):
    dtype = percolate.hpc.macrocanonical_averages_dtype(
        spanning_cluster=spanning_cluster
    )
    array = np.empty(1, dtype=dtype)
    assert spanning_cluster == (
        'percolation_probability_mean' in array.dtype.names and
        'percolation_probability_m2' in array.dtype.names
    )


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_finalized_macrocanonical_averages_dtype(spanning_cluster):
    dtype = percolate.hpc.finalized_macrocanonical_averages_dtype(
        spanning_cluster=spanning_cluster
    )
    np.empty(1, dtype=dtype)


@pytest.mark.parametrize("spanning_cluster", [True, False])
def test_finalized_macrocanonical_averages_dtype_spanning_cluster(
    spanning_cluster
):
    dtype = percolate.hpc.finalized_macrocanonical_averages_dtype(
        spanning_cluster=spanning_cluster
    )
    array = np.empty(1, dtype=dtype)
    assert spanning_cluster == (
        'percolation_probability_mean' in array.dtype.names and
        'percolation_probability_std' in array.dtype.names and
        'percolation_probability_ci' in array.dtype.names
    )


@pytest.fixture(params=[True, False])
def percolation_graph(request):

    spanning_cluster = request.param
    if spanning_cluster:
        graph = percolate.percolate.spanning_2d_grid(length=3)
    else:
        graph = nx.grid_2d_graph(3, 3)
    return percolate.percolate.percolation_graph(
        graph=graph, spanning_cluster=spanning_cluster,
    )


def grid_2d_has_spanning_cluster(
    graph, selected_edges, **kwds
):

    # auxiliary nodes
    aux_nodes_attrs = nx.get_node_attributes(graph, 'span')
    sides = list(set(aux_nodes_attrs.values()))
    aux_nodes = dict()
    for side in sides:
        aux_nodes[side] = [
            node for (node, node_side) in aux_nodes_attrs.items()
            if node_side == side
        ]

    # create subgraph
    # with all nodes
    # and all auxiliary edges and all selected edges
    auxiliary_edges = list(nx.get_edge_attributes(graph, 'span'))
    subgraph_edges = auxiliary_edges + selected_edges

    # the subgraph contains all auxiliary nodes of the original graph
    subgraph = nx.Graph(data=subgraph_edges)

    # calculate all shortest pairs
    shortest_paths = nx.all_pairs_shortest_path_length(subgraph)

    # check if there is a shortest path between opposite auxiliary nodes
    ret = False
    for source_node in aux_nodes[sides[0]]:
        for target_node in aux_nodes[sides[1]]:
            if target_node in shortest_paths[source_node]:
                ret = True
                break
        if ret:
            break

    return ret


def test_internal_spanning_detection():
    graph = percolate.percolate.spanning_2d_grid(3)
    assert not grid_2d_has_spanning_cluster(graph, [])
    assert not grid_2d_has_spanning_cluster(graph, [((1, 0), (2, 0))])
    assert grid_2d_has_spanning_cluster(
        graph, [((1, 0), (2, 0)), ((3, 0), (2, 0))]
    )


def get_cluster_sizes(perc_graph, selected_edges, **kwds):
    # create subgraph with all nodes and all selected edges
    subgraph = nx.Graph(data=selected_edges)
    subgraph.add_nodes_from(perc_graph)

    # get clusters
    clusters = list(nx.connected_components(subgraph))
    sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    return np.array(sizes)


def test_internal_cluster_sizes(percolation_graph):
    np.testing.assert_array_equal(
        get_cluster_sizes(selected_edges=[], **percolation_graph),
        9 * [1]
    )
    np.testing.assert_array_equal(
        get_cluster_sizes(
            selected_edges=[((1, 0), (1, 1))],
            **percolation_graph
        ),
        [2] + 7 * [1]
    )
    np.testing.assert_array_equal(
        get_cluster_sizes(
            selected_edges=[
                ((1, 0), (1, 1)),
                ((2, 0), (1, 0)),
            ],
            **percolation_graph
        ),
        [3] + 6 * [1]
    )
    np.testing.assert_array_equal(
        get_cluster_sizes(
            selected_edges=[
                ((1, 0), (1, 1)),
                ((2, 0), (2, 1)),
            ],
            **percolation_graph
        ),
        2 * [2] + 5 * [1]
    )


@hypothesis.given(seed=st.integers(min_value=0, max_value=MAX_SEED))
def test_bond_sample_states(percolation_graph, seed):
    spanning_cluster = percolation_graph['spanning_cluster']
    graph = percolation_graph['graph']
    all_edges = percolation_graph['perc_graph'].edges()
    selected_edges = list()
    previous_cluster_sizes = None
    previous_max_cluster_size = None
    for n, sample_state in enumerate(percolate.hpc.bond_sample_states(
        seed=seed, **percolation_graph
    )):
        assert n == int(sample_state['n'])
        if sample_state['n'] > 0:
            # add edge
            edge = all_edges[sample_state['edge']]
            assert edge not in selected_edges
            selected_edges.append(edge)

        if spanning_cluster:
            assert (
                grid_2d_has_spanning_cluster(graph, selected_edges) ==
                bool(sample_state['has_spanning_cluster'])
            )

        cluster_sizes = get_cluster_sizes(
            selected_edges=selected_edges, **percolation_graph
        )
        assert cluster_sizes.sum() == percolation_graph['num_nodes']

        max_cluster_size = sample_state['max_cluster_size'][0]
        assert max_cluster_size == cluster_sizes[0]
        if n > 0:
            assert cluster_sizes.size <= previous_cluster_sizes.size
            assert max_cluster_size >= previous_max_cluster_size

        for k in range(5):
            assert (
                np.sum(cluster_sizes[1:] ** k) ==
                sample_state['moments'][0][k]
            )

        previous_max_cluster_size = max_cluster_size
        previous_cluster_sizes = cluster_sizes


def test_bond_sample_state_raises_more_than_two_spanning_sides(
    percolation_graph
):
    if percolation_graph['spanning_cluster']:
        percolation_graph['spanning_sides'] = [0, 1, 2]
        with pytest.raises(ValueError):
            next(percolate.hpc.bond_sample_states(
                seed=0, **percolation_graph
            ))


@hypothesis.given(seed=st.integers(min_value=0, max_value=MAX_SEED))
def test_bond_microcanonical_statistics(percolation_graph, seed):
    result = percolate.hpc.bond_microcanonical_statistics(
        seed=seed, **percolation_graph
    )
    sample_states = percolate.hpc.bond_sample_states(
        seed=seed, **percolation_graph
    )
    for n in range(percolation_graph['num_edges'] + 1):
        result_row = result[n]
        sample_state = next(sample_states)

        def _assert_equal(key):
            assert result_row[key] == sample_state[key][0]

        assert result_row['n'] == n
        assert sample_state['n'][0] == n

        if n > 0:
            # first edge is undefined!
            _assert_equal('edge')

        if percolation_graph['spanning_cluster']:
            _assert_equal('has_spanning_cluster')
        _assert_equal('max_cluster_size')
        np.testing.assert_array_equal(
            result_row['moments'],
            sample_state['moments'][0],
        )


@hypothesis.given(
    seed=st.integers(min_value=0, max_value=MAX_SEED),
    p=st.floats(min_value=0.0, max_value=1.0),
)
def test_bond_macrocanonical_statistics(percolation_graph, seed, p):
    microcanonical_statistics = percolate.hpc.bond_microcanonical_statistics(
        seed=seed, **percolation_graph
    )
    convolution_factors = percolate.percolate._binomial_pmf(
        n=percolation_graph['num_edges'], p=p,
    )
    result = percolate.hpc.bond_macrocanonical_statistics(
        microcanonical_statistics=microcanonical_statistics,
        convolution_factors=convolution_factors,
    )
    assert (
        ('percolation_probability' in result.dtype.names) ==
        percolation_graph['spanning_cluster']
    )
    if percolation_graph['spanning_cluster']:
        np.testing.assert_almost_equal(
            np.sum(
                microcanonical_statistics['has_spanning_cluster'] *
                convolution_factors
            ),
            result['percolation_probability'],
        )
    np.testing.assert_almost_equal(
        np.sum(
            microcanonical_statistics['max_cluster_size'] *
            convolution_factors
        ),
        result['max_cluster_size'],
    )
    for k in range(5):
        np.testing.assert_almost_equal(
            np.sum(
                microcanonical_statistics['moments'][:, k] *
                convolution_factors
            ),
            result['moments'][0, k],
        )


@hypothesis.given(
    seed=st.integers(min_value=0, max_value=MAX_SEED),
    ps=st.lists(
        elements=st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=10,
    ),
)
def test_bond_initialize_macrocanonical_averages(percolation_graph, seed, ps):
    spanning_cluster = percolation_graph['spanning_cluster']
    microcanonical_statistics = percolate.hpc.bond_microcanonical_statistics(
        seed=seed, **percolation_graph
    )
    convolution_factors = [
        percolate.percolate._binomial_pmf(
            n=percolation_graph['num_edges'], p=p,
        )
        for p in ps
    ]
    macrocanonical_statistics = np.fromiter(
        (
            percolate.hpc.bond_macrocanonical_statistics(
                microcanonical_statistics=microcanonical_statistics,
                convolution_factors=my_convolution_factors,
            )
            for my_convolution_factors in convolution_factors
        ),
        dtype=percolate.hpc.macrocanonical_statistics_dtype(
            spanning_cluster=spanning_cluster
        ),
    )
    assert macrocanonical_statistics.size == len(ps)
    result = percolate.hpc.bond_initialize_macrocanonical_averages(
        macrocanonical_statistics=macrocanonical_statistics
    )
    assert (
        ('percolation_probability_mean' in result.dtype.names) ==
        percolation_graph['spanning_cluster']
    )
    assert (
        ('percolation_probability_m2' in result.dtype.names) ==
        percolation_graph['spanning_cluster']
    )
    np.testing.assert_array_equal(
        result['number_of_runs'], np.ones_like(result['number_of_runs'])
    )
    if spanning_cluster:
        np.testing.assert_array_equal(
            result['percolation_probability_mean'],
            macrocanonical_statistics['percolation_probability'],
        )
        np.testing.assert_array_equal(
            result['percolation_probability_m2'],
            np.zeros_like(result['percolation_probability_m2'])
        )
    np.testing.assert_array_equal(
        result['max_cluster_size_mean'],
        macrocanonical_statistics['max_cluster_size'],
    )
    np.testing.assert_array_equal(
        result['max_cluster_size_m2'],
        np.zeros_like(result['max_cluster_size_m2']),
    )
    np.testing.assert_array_equal(
        result['moments_mean'],
        macrocanonical_statistics['moments'],
    )
    np.testing.assert_array_equal(
        result['moments_m2'],
        np.zeros_like(result['moments_m2']),
    )


@hypothesis.given(
    seeds=st.lists(
        elements=st.integers(min_value=0, max_value=MAX_SEED),
        min_size=2, max_size=20,
    ),
    ps=st.lists(
        elements=st.floats(min_value=0.1, max_value=1.0),
        min_size=1,
        max_size=10,
    ),
)
def test_bond_reduce_incremental(percolation_graph, seeds, ps):
    spanning_cluster = percolation_graph['spanning_cluster']
    convolution_factors = [
        percolate.percolate._binomial_pmf(
            n=percolation_graph['num_edges'], p=p,
        )
        for p in ps
    ]

    initial_arrays = list()
    for seed in seeds:
        microcanonical_statistics = (
            percolate.hpc.bond_microcanonical_statistics(
                seed=seed, **percolation_graph
            )
        )
        macrocanonical_statistics = np.fromiter(
            (
                percolate.hpc.bond_macrocanonical_statistics(
                    microcanonical_statistics=microcanonical_statistics,
                    convolution_factors=my_convolution_factors,
                )
                for my_convolution_factors in convolution_factors
            ),
            dtype=percolate.hpc.macrocanonical_statistics_dtype(
                spanning_cluster=spanning_cluster
            ),
        )
        initial_arrays.append(
            percolate.hpc.bond_initialize_macrocanonical_averages(
                macrocanonical_statistics=macrocanonical_statistics
            )
        )

    result = functools.reduce(percolate.hpc.bond_reduce, initial_arrays)

    assert np.all(result['number_of_runs'] == len(seeds))

    def _assert_mean_m2(key):
        mean_key = '{}_mean'.format(key)
        mean = np.mean(
            [initial_array[mean_key] for initial_array in initial_arrays],
            axis=0,
        )
        np.testing.assert_allclose(result[mean_key], mean)
        m2_key = '{}_m2'.format(key)
        m2 = np.sum(
            np.square(np.subtract(
                [initial_array[mean_key] for initial_array in initial_arrays],
                mean
            )),
            axis=0,
        )
        np.testing.assert_allclose(result[m2_key], m2, atol=1e-6)

    if spanning_cluster:
        _assert_mean_m2('percolation_probability')

    _assert_mean_m2('max_cluster_size')
    _assert_mean_m2('moments')


@hypothesis.given(
    seeds=st.lists(
        elements=st.integers(min_value=0, max_value=MAX_SEED),
        min_size=10,
        max_size=20,
        unique_by=lambda x: x,
    ),
    ps=st.lists(
        elements=st.floats(min_value=0.1, max_value=1.0),
        min_size=1,
        max_size=10,
    ),
    alpha=st.floats(min_value=0.01, max_value=0.31),
)
def test_finalize_macrocanonical_averages(percolation_graph, seeds, ps, alpha):
    spanning_cluster = percolation_graph['spanning_cluster']
    num_nodes = percolation_graph['num_nodes']
    convolution_factors = [
        percolate.percolate._binomial_pmf(
            n=percolation_graph['num_edges'], p=p,
        )
        for p in ps
    ]

    initial_arrays = list()
    for seed in seeds:
        microcanonical_statistics = (
            percolate.hpc.bond_microcanonical_statistics(
                seed=seed, **percolation_graph
            )
        )
        macrocanonical_statistics = np.fromiter(
            (
                percolate.hpc.bond_macrocanonical_statistics(
                    microcanonical_statistics=microcanonical_statistics,
                    convolution_factors=my_convolution_factors,
                )
                for my_convolution_factors in convolution_factors
            ),
            dtype=percolate.hpc.macrocanonical_statistics_dtype(
                spanning_cluster=spanning_cluster
            ),
        )
        initial_arrays.append(
            percolate.hpc.bond_initialize_macrocanonical_averages(
                macrocanonical_statistics=macrocanonical_statistics
            )
        )

    macrocanonical_averages = functools.reduce(
        percolate.hpc.bond_reduce, initial_arrays,
    )

    result = percolate.hpc.finalize_macrocanonical_averages(
        number_of_nodes=percolation_graph['num_nodes'],
        ps=ps,
        macrocanonical_averages=macrocanonical_averages,
        alpha=alpha,
    )

    assert np.all(result['number_of_runs'] == len(seeds))
    np.testing.assert_array_equal(ps, result['p'])
    assert np.all(result['alpha'] == alpha)
    for key in ['mean', 'std', 'ci']:
        assert spanning_cluster == (
            'percolation_probability_{}'.format(key) in result.dtype.names
        )
    if spanning_cluster:
        mean_key = 'percolation_probability_mean'
        np.testing.assert_array_equal(
            result[mean_key], macrocanonical_averages[mean_key],
        )
        np.testing.assert_allclose(
            result['percolation_probability_std'],
            np.sqrt(
                macrocanonical_averages['percolation_probability_m2'] /
                (len(seeds) - 1)
            ),
        )
        for bound, my_alpha in zip([0, 1], [alpha / 2, 1 - alpha / 2]):
            for p_index in range(len(ps)):
                if (
                    result['percolation_probability_std'][p_index] /
                    result['percolation_probability_mean'][p_index] <
                    1e-10
                ):
                    continue
                alphas = scipy.stats.t.cdf(
                    result['percolation_probability_ci'][p_index, bound],
                    loc=result['percolation_probability_mean'][p_index],
                    scale=(
                        result['percolation_probability_std'][p_index] /
                        np.sqrt(len(seeds))
                    ),
                    df=len(seeds) - 1,
                )
                if np.isnan(alphas):
                    continue
                np.testing.assert_allclose(
                    alphas, my_alpha, rtol=1e-4,
                )

    np.testing.assert_allclose(
        result['percolation_strength_mean'],
        macrocanonical_averages['max_cluster_size_mean'] / num_nodes,
    )
    np.testing.assert_allclose(
        result['percolation_strength_std'],
        np.sqrt(
            macrocanonical_averages['max_cluster_size_m2'] / (len(seeds) - 1)
        ) / num_nodes,
    )
    for bound, my_alpha in zip([0, 1], [alpha / 2, 1 - alpha / 2]):
        for p_index in range(len(ps)):
            if (
                result['percolation_strength_std'][p_index] /
                result['percolation_strength_mean'][p_index] <
                1e-10
            ):
                continue
            alphas = scipy.stats.t.cdf(
                result['percolation_strength_ci'][p_index, bound],
                loc=result['percolation_strength_mean'][p_index],
                scale=(
                    result['percolation_strength_std'][p_index] /
                    np.sqrt(len(seeds))
                ),
                df=len(seeds) - 1,
            )
            if np.isnan(alphas):
                continue
            np.testing.assert_allclose(
                alphas, my_alpha, rtol=1e-4,
            )

    np.testing.assert_allclose(
        result['moments_mean'],
        macrocanonical_averages['moments_mean'] / num_nodes,
    )
    np.testing.assert_allclose(
        result['moments_std'],
        np.sqrt(
            macrocanonical_averages['moments_m2'] / (len(seeds) - 1)
        ) / num_nodes,
    )
    for k in range(5):
        for bound, my_alpha in zip([0, 1], [alpha / 2, 1 - alpha / 2]):
            for p_index in range(len(ps)):
                if (
                    result['moments_std'][p_index, k] /
                    result['moments_mean'][p_index, k] <
                    1e-10
                ):
                    continue
                alphas = scipy.stats.t.cdf(
                    result['moments_ci'][p_index, k, bound],
                    loc=result['moments_mean'][p_index, k],
                    scale=(
                        result['moments_std'][p_index, k] /
                        np.sqrt(len(seeds))
                    ),
                    df=len(seeds) - 1,
                )
                if np.isnan(alphas):
                    continue
                np.testing.assert_allclose(
                    alphas, my_alpha, rtol=1e-4,
                )
