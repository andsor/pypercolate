Developer Guide
===============

* **Repository**: `github.com/andsor/pypercolate <http://github.com/andsor/pypercolate>`_
* **Read the Docs**: `readthedocs.org/projects/pypercolate <https://readthedocs.org/projects/pypercolate>`_
* **Bibliography**: `www.citeulike.org/group/19226 <http://www.citeulike.org/group/19226>`_

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

This project follows the `NumPy documentation style
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Build the documentation with ::
        
    $ python setup.py docs
    
and run doctests with ::

    $ python setup.py doctest

Alternatively, let `tox`_
`configure the virtual environment and run sphinx <http://tox.readthedocs.org/en/latest/example/general.html#integrating-sphinx-documentation-checks>`_::

    $ tox -e docs

Add further options separated from tox options by a double dash ``--``::

    $ tox -e docs -- --help

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

For continuously building the documentation during development, run::
        
    $ python setup.py autodocs

Unittest & Coverage
-------------------

Run ::

    $ python setup.py test
    
to run all unittests defined in the subfolder ``test`` with the help of `tox`_
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

Bibliography
------------

A `CiteULike group`_ manages the bibliography.

.. _CiteULike group: http://www.citeulike.org/group/19226
