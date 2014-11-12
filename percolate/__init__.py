#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from percolate.percolate import (
    sample_states, microcanonical_averages, statistics
)
