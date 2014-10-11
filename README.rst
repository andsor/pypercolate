===========
pypercolate
===========

Scientific Python package for Monte-Carlo simulation of percolation on graphs

.. image:: https://readthedocs.org/projects/pypercolate/badge/?version=latest
   :target: https://readthedocs.org/projects/pypercolate/?badge=latest
   :alt: Documentation Status

.. image:: http://img.shields.io/pypi/l/pypercolate.svg
   :target: http://pypercolate.readthedocs.org/en/latest/license.html
   :alt: License

* **Documentation**: `pypercolate.readthedocs.org <http://pypercolate.readthedocs.org>`_
* **Repository**: http://github.com/andsor/pypercolate


License
=======

See `LICENSE.txt <LICENSE.txt>`_.

Developing
==========


Packaging
---------

This package uses `setuptools <http://pythonhosted.org/setuptools/>`_.

Run ``python setup.py sdist``, ``python setup.py bdist`` or
``python setup.py bdist_wheel`` to build a source, binary or wheel
distribution.


Complete Git Integration
------------------------

Your project is already an initialised Git repository and ``setup.py`` uses the
information of tags to infer the version of your project with the help of
`versioneer <https://github.com/warner/python-versioneer>`_.

To use this feature you need to tag with the format
``vMAJOR.MINOR[.REVISION]``, e.g. ``v0.0.1`` or ``v0.1``.
The prefix ``v`` is needed!
Run ``python setup.py version`` to retrieve the current `PEP440
<http://www.python.org/dev/peps/pep-0440/>`_-compliant version.
This version will be used when building a package and is also accessible
through ``percolate.__version__``.
The version will be ``unknown`` until you have added a first tag.


Sphinx Documentation
--------------------

Build the documentation with ``python setup.py docs`` and run doctests with
``python setup.py doctest``.

Start editing the file `docs/index.rst <docs/index.rst>`_ to extend the
documentation.

The documentation also works with `Read the Docs <https://readthedocs.org/>`_.

Add `requirements
<http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files>`_ for
building the documentation to the
`doc-requirements.txt <doc-requirements.txt>`_ file.

Continuous documentation building
---------------------------------

For continuously building the documentation, run ``./autodocs.sh``.

Unittest & Coverage
-------------------

Run ``python setup.py test`` to run all unittests defined in the subfolder
``tests`` with the help of `tox <http://tox.testrun.org>`_ and
`py.test <http://pytest.org/>`_.

The py.test plugin `pytest-cov <https://github.com/schlamar/pytest-cov>`_ is
used to automatically generate a coverage report. 

Continuous testing
------------------

For continuous testing in a **Python 2.7** environment, run ``python setup.py
test --tox-args='-c toxdev.ini -e py27'``.

For continuous testing in a **Python 3.4** environment, run ``python setup.py
test --tox-args='-c toxdev.ini -e py34'``.


Requirements Management
-----------------------

Add `requirements
<http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files>`_ to
to the `requirements.txt <requirements.txt>`_ file which will be automatically
used by ``setup.py``.

