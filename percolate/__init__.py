#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

import percolate
