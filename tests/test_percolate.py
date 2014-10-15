#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

import pytest
import inspect
import percolate
import numpy as np


def _test_existence(module, function):
    return hasattr(module, function)


def _test_signature(function, params):
    try:  # python 3
        args = inspect.signature(function).parameters
    except:  # python 2
        args = inspect.getargspec(function).args

    for param in params:
        assert param in args


def test_sample_state_existence():
    assert _test_existence(percolate, 'sample_states')


def test_sample_state_signature():
    _test_signature(
        percolate.sample_states,
        ['graph', 'spanning_cluster', 'model']
    )


@pytest.fixture
def empty_graph():
    import networkx

    return networkx.Graph()


@pytest.fixture(params=[True, False])
def grid_3x3_graph(request):
    import networkx

    if request.param:
        ret = networkx.grid_2d_graph(5, 3)
        for i in [0, 4]:
            for j in range(3):
                ret.node[(i, j)]['span'] = i
    else:
        ret = networkx.grid_2d_graph(3, 3)

    ret.graph['span'] = request.param
    return ret

def test_sample_state_not_implemented_model(empty_graph):
    with pytest.raises(ValueError):
        next(percolate.sample_states(empty_graph, model='site'))


def test_sample_state_no_auxiliary_nodes(empty_graph):
    with pytest.raises(ValueError):
        next(percolate.sample_states(empty_graph, spanning_cluster=True))


def test_sample_state_one_sided_auxiliary_nodes(empty_graph):
    empty_graph.add_node(1, span=0)
    with pytest.raises(ValueError):
        next(percolate.sample_states(empty_graph, spanning_cluster=True))


def test_initial_iteration(grid_3x3_graph):

    spanning_cluster = grid_3x3_graph.graph['span']

    ret = next(percolate.sample_states(
        grid_3x3_graph, spanning_cluster=spanning_cluster
    ))

    assert ret['n'] == 0
    np.testing.assert_allclose(ret['max_cluster_size'], 1 / 9)
    assert np.array_equal(
        ret['moments'], np.ones(5) * 8 / 9
    )

    assert ('has_spanning_cluster' in ret) == spanning_cluster

    if spanning_cluster:
        assert not ret['has_spanning_cluster']
