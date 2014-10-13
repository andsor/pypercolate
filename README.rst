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
* **Read the Docs**: https://readthedocs.org/projects/pypercolate


Requirements and Python 2/3 compatibility
=========================================

This package runs under **Python 2** and **Python 3**, and has been tested with
**Python 2.7.6** and **Python 3.4.0**.

License
=======

See `LICENSE.txt <LICENSE.txt>`_.

Developing
==========

Development environment
-----------------------

Use `tox`_ to `prepare virtual environments for development`_.

.. _prepare virtual environments for development: http://testrun.org/tox/latest/example/devenv.html>

.. _tox: http://tox.testrun.org

To set up a **Python 2.7** environment in ``.devenv27``, run::

    $ tox -e devenv27

To set up a **Python 3.4** environment in ``.devenv34``, run::

    $ tox -e devenv34

Packaging
---------

This package uses `setuptools`_.

.. _setuptools: http://pythonhosted.org/setuptools

Run ::

    $ python setup.py sdist
   
or ::

    $ python setup.py bdist
   
or ::

    $ python setup.py bdist_wheel
    
to build a source, binary or wheel distribution.


Complete Git Integration
------------------------

Your project is already an initialised Git repository and ``setup.py`` uses the
information of tags to infer the version of your project with the help of
`versioneer <https://github.com/warner/python-versioneer>`_.

To use this feature you need to tag with the format
``vMAJOR.MINOR[.REVISION]``, e.g. ``v0.0.1`` or ``v0.1``.
The prefix ``v`` is needed!

Run ::
        
    $ python setup.py version
    
to retrieve the current `PEP440`_-compliant version.
This version will be used when building a package and is also accessible
through ``percolate.__version__``.
The version will be ``unknown`` until you have added a first tag.

.. _PEP440: http://www.python.org/dev/peps/pep-0440

Sphinx Documentation
--------------------

Build the documentation with ::
        
    $ python setup.py docs
    
and run doctests with ::

    $ python setup.py doctest

Start editing the file `docs/index.rst <docs/index.rst>`_ to extend the
documentation.

`Read the Docs`_ hosts the project at
https://readthedocs.org/projects/pypercolate. 

.. _Read the Docs:  http://readthedocs.org/

Add `requirements`_ for building the documentation to the
`doc-requirements.txt <doc-requirements.txt>`_ file.

.. _requirements: http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files

Continuous documentation building
---------------------------------

For continuously building the documentation, run::
        
    $ ./autodocs.sh

Unittest & Coverage
-------------------

Run ::

    $ python setup.py test
    
to run all unittests defined in the subfolder ``tests`` with the help of `tox`_
and `py.test`_.

.. _py.test: http://pytest.org

The py.test plugin `pytest-cov`_ is used to automatically generate a coverage
report. 

.. _pytest-cov: http://github.com/schlamar/pytest-cov

Continuous testing
------------------

For continuous testing in a **Python 2.7** environment, run::
       
    $ python setup.py test --tox-args='-c toxdev.ini -e py27'

For continuous testing in a **Python 3.4** environment, run::
       
    $ python setup.py test --tox-args='-c toxdev.ini -e py34'


Requirements Management
-----------------------

Add `requirements`_ to the `requirements.txt <requirements.txt>`_ file which
will be automatically used by ``setup.py``.

