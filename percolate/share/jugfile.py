# encoding: utf-8

"""
A sample jugfile to use with Jug

jug execute jugfile
"""

# Python 2/3 compatibility
from __future__ import unicode_literals
# http://python-future.org/compatible_idioms.html#zip-izip
from builtins import map, zip
from future.utils import iteritems

from functools import reduce
import itertools

import numpy as np
import scipy.stats

from jug import Task, TaskGenerator, barrier
from jug.utils import CustomHash
import jug.mapreduce

import percolate
import percolate.hpc


# script parameters
SYSTEM_DIMENSIONS = [8, 16, 32]
RUNS_PER_TASK = 100
NUMBER_OF_TASKS = 100
NUMBER_OF_RUNS = NUMBER_OF_TASKS * RUNS_PER_TASK
PS = np.linspace(0.4, 0.6, num=40)
SPANNING_CLUSTER = True
UINT32_MAX = 4294967296
ALPHA = 2 * scipy.stats.norm.cdf(-1.0)  # 1 sigma


# in general:
# avoid premature optimization: design simulation for large system sizes
# (small system sizes take much less time anyway)
for dimension in SYSTEM_DIMENSIONS:
    # building the graph takes ca. 10s per 10^6 nodes
    '''
    >>> import timeit
    >>> timeit.timeit(
    ...     stmt='percolate.spanning_2d_grid(1000)',
    ...     setup='import percolate',
    ...     number=1
    ... )
    >>>
    '''


@TaskGenerator
def prepare_percolation_graph(dimension):
    graph = percolate.spanning_2d_grid(length=dimension)
    (
        perc_graph,
        num_nodes,
        num_edges,
        auxiliary_node_attributes,
        auxiliary_edge_attributes,
    ) = percolate.hpc.percolation_graph(
        graph=graph, spanning_cluster=SPANNING_CLUSTER
    )
    return (
        perc_graph,
        num_nodes,
        num_edges,
        auxiliary_node_attributes,
        auxiliary_edge_attributes,
    )


@TaskGenerator
def compute_convolution_factors_for_single_p(perc_graph_result, p):
    return percolate.percolate._binomial_pmf(
        n=perc_graph_result[2],
        p=p,
    )


def bond_run(perc_graph_result, seed, ps, convolution_factors_tasks):
    """
    Perform a single run (realization) over all microstates and return the
    macrocanonical cluster statistics
    """
    (
        perc_graph,
        num_nodes,
        num_edges,
        auxiliary_node_attributes,
        auxiliary_edge_attributes,
    ) = perc_graph_result

    microcanonical_statistics = percolate.hpc.bond_microcanonical_statistics(
        perc_graph=perc_graph,
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        spanning_cluster=SPANNING_CLUSTER,
        auxiliary_node_attributes=auxiliary_node_attributes,
        auxiliary_edge_attributes=auxiliary_edge_attributes,
    )

    # initialize statistics array
    macrocanonical_statistics = np.empty(
        ps.size,
        dtype=percolate.hpc.macrocanonical_statistics_dtype(
            spanning_cluster=SPANNING_CLUSTER,
        )
    )

    # loop over all p's and convolve macrocanonical statistics
    # http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html#modifying-array-values
    for row, convolution_factors_task in zip(
        np.nditer(macrocanonical_statistics, op_flags=['writeonly']),
        convolution_factors_tasks,
    ):
        # load task result
        # http://jug.readthedocs.org/en/latest/api.html#jug.Task.load
        assert not convolution_factors_task.is_loaded()
        convolution_factors_task.load()
        # fetch task result
        my_convolution_factors = convolution_factors_task.result

        # convolve to macrocanonical statistics
        row[...] = percolate.hpc.bond_macrocanonical_statistics(
            microcanonical_statistics=microcanonical_statistics,
            convolution_factors=my_convolution_factors,
            spanning_cluster=SPANNING_CLUSTER,
        )
        # explicitly unload task to save memory
        # http://jug.readthedocs.org/en/latest/api.html#jug.Task.unload
        convolution_factors_task.unload()

    # initialize macrocanonical averages for reduce
    ret = percolate.hpc.bond_initialize_macrocanonical_averages(
        macrocanonical_statistics=macrocanonical_statistics,
        spanning_cluster=SPANNING_CLUSTER,
    )

    return ret


@TaskGenerator
def bond_task(
    perc_graph_result, seeds, ps, convolution_factors_tasks_iterator
):
    """
    Perform a number of runs

    The number of runs is the number of seeds

    convolution_factors_tasks_iterator needs to be an iterator

    We shield the convolution factors tasks from jug value/result mechanism
    by supplying an iterator to the list of tasks for lazy evaluation
    http://github.com/luispedro/jug/blob/43f0d80a78f418fd3aa2b8705eaf7c4a5175fff7/jug/task.py#L100
    http://github.com/luispedro/jug/blob/43f0d80a78f418fd3aa2b8705eaf7c4a5175fff7/jug/task.py#L455
    """

    # restore the list of convolution factors tasks
    convolution_factors_tasks = list(convolution_factors_tasks_iterator)

    return reduce(
        percolate.hpc.bond_reduce,
        map(
            bond_run,
            itertools.repeat(perc_graph_result),
            seeds,
            itertools.repeat(ps),
            itertools.repeat(convolution_factors_tasks),
        )
    )


@TaskGenerator
def write_to_disk(dimension, macrocanonical_averages):
    import h5py

    # Read/write if exsits, create otherwise
    f = h5py.File('percolate_hpc.hdf5', mode='a')

    key = '{}'.format(dimension)

    # dataset should not exist yet!
    if key in f:
        raise RuntimeError

    f.create_dataset(
        name=key,
        data=macrocanonical_averages,
    )

    f.close()


def dummy_hash(x):
    """
    Supplies a constant dummy hash
    """
    return 'dummy_hash'.encode('utf-8')


convolution_factors = {}
perc_graph_results = {}
reduced_macrocanonical_averages = {}
final_macrocanonical_averages = {}

# initialize random number generator
rng = np.random.RandomState(seed=201508061904 % UINT32_MAX)


# in general:
# avoid premature optimization: design simulation for large system sizes
# (small system sizes take much less time anyway)
for dimension in SYSTEM_DIMENSIONS:
    # building the graph takes ca. 10s per 10^6 nodes
    '''
    >>> import timeit
    >>> timeit.timeit(
    ...     stmt='percolate.spanning_2d_grid(1000)',
    ...     setup='import percolate',
    ...     number=1
    ... )
    >>>
    '''
    # So that is why we store every graph on disk (as a task)
    perc_graph_results[dimension] = prepare_percolation_graph(dimension)

    # computing the binomial PMF takes ca. 1s per 10^6 number of states *
    # number of p's
    '''
    >>> import timeit
    >>> s = """\
    ... [
    ...     percolate.percolate._binomial_pmf(n=1e4, p=p)
    ...     for p in 100 * [0.5]
    ... ]
    >>> timeit.timeit(stmt=s, setup='import percolate', number=1)
    '''
    # so store all binomial PMF (for every dimension, p) on disk
    convolution_factors[dimension] = [
        compute_convolution_factors_for_single_p(
            perc_graph_result=perc_graph_results[dimension], p=p,
        )
        for p in PS
    ]

    # finish all tasks up to here first before proceeding
    # http://jug.readthedocs.org/en/latest/barrier.html#barriers
    barrier()

    seeds = rng.randint(UINT32_MAX, size=NUMBER_OF_RUNS)

    # now process the actual tasks
    bond_tasks = list()
    for my_seeds in np.split(seeds, NUMBER_OF_TASKS):
        # pass an iterator to list of convolution factors tasks to circumvent
        # jug automatic fetching results of each task (iterator is not resolved
        # and hence we achieve lazy evaluation of tasks here)
        convolution_iterator = iter(
            convolution_factors[dimension]
        )
        # iter cannot be pickled / hashed, so supply a dummy hash
        convolution_iterator_hashed = CustomHash(
            convolution_iterator, dummy_hash,
        )
        bond_tasks.append(
            bond_task(
                perc_graph_result=perc_graph_results[dimension],
                seeds=my_seeds,
                ps=PS,
                convolution_factors_tasks_iterator=convolution_iterator_hashed,
            )
        )

    # reduce
    reduced_macrocanonical_averages[dimension] = jug.mapreduce.reduce(
        reducer=percolate.hpc.bond_reduce,
        inputs=bond_tasks,
        reduce_step=100,
    )

    # finalize macrocanonical averages
    final_macrocanonical_averages[dimension] = Task(
        percolate.hpc.finalize_macrocanonical_averages,
        number_of_nodes=perc_graph_results[dimension][1],
        ps=PS,
        macrocanonical_averages=reduced_macrocanonical_averages[dimension],
        alpha=ALPHA,
    )

    # write to disk
    write_to_disk(
        dimension=dimension,
        macrocanonical_averages=final_macrocanonical_averages[dimension],
    )
