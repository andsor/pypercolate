Quickstart High-Performance Computing with pypercolate
======================================================

The :any:`percolate.hpc` module is an :doc:`implementation of the Newman--Ziff algorithm for High-Performance Computing (HPC) <pypercolate-hpc>`.
Here we demonstrate how to compute the raw data for a finite-size scaling study of bond percolation on a square grid.
For high-performance computing, we utilize the task-based parallelization framework `Jug <http://luispedro.org/software/jug/>`_.
Our simulation design is as follows::

    For each system size L:
        Prepare the graph of the current system size L
        Prepare the graph for pypercolate and precompute auxiliary information
        For each value p of the occupation probability:
            Precompute the convolution coefficients f
        Map-Reduce T tasks:
            Map-Reduce R runs:
                Seed the Random Number Generator
                For each occupation number n:
                    Compute the microcanonical cluster statistics
                For each occupation probability p:
                    Convolve the microcanonical statistics into the canonical statistics
        Finalize tasks results
        Write results to disk

A jugfile is a Python script to which Jug transparently adds parallelization magic.

Imports
-------

First, we import the packages needed.

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 9-26

Define input
------------

Here we assign the input parameters of the simulation study.

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 29-38

The ``SPANNING_CLUSTER`` variable defines whether the script should detect spanning clusters.
The ``UINT32_MAX`` integer is the upper bound of the seeds.
The ``SEED`` seeds the central random number generator which in turn seeds the task random number generators.


Jug task to prepare the percolation graph and precomput the convolution factors
-------------------------------------------------------------------------------

The following jug task creates the square grid graph of :math:`L \times L` nodes, where :math:`L` is the system size.

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 41-46 

Next, the jugfile defines a task to precompute the convolution factors for a given occupation probability :math:`p`:

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 49-54

An elementary run (sub-task)
----------------------------

Every task samples the bond percolation setting with a number of runs :math:`N_{\text{runs}}`.
The following function implements a single run as described in the user guide.
Note here that we carefully circumvent Jug's automatic loading of task results:
Instead of piping through the convolution factor tasks directly, we add a layer of indirectness and supply an iterator to these tasks.
This way, Jug does not automatically load the task results, which would unnecessarily increase memory consumption and limit the system size.
The inner loop convolves the microcanonical statistics for one occupation number after the other, carefully loading and unloading the respective convolution factor task results.

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 57-103
   :linenos:
   :emphasize-lines: 27,39

A simple task
-------------

Here is the definition of the actual task, each instance of which conducts a number of runs and reduces their results:

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 106-135 

Write to disk
-------------

Another task is to write the finalized results for a system size to disk.
We provide an example implementation here that writes to a HDF5 file:

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 138-156


Hashing the iterator
--------------------

For supplying a task iterator, for technical reasons we need to supply a dummy hash function to Jug:

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 159-163

The control flow
----------------

Now, we are in the position to implement the control flow:

.. literalinclude:: /../percolate/share/jugfile.py
   :lines: 166-259
   :linenos:
   :emphasize-lines: 48

Note the use of Jug's ``barrier`` command.
It ensures that Jug finishes the computation of the percolation graph and all convolution factors, before proceeding with the actual bond tasks.
Normally, Jug auto-detects such dependencies and transparently makes sure that all previous tasks a particular task depends on have been run.
However, here we need to explicitly call ``barrier``, as we circumvent this very mechanism.
As explained earlier, this is to avoid the convolution factors of all occupation probabilities to be loaded into memory at the same time.

Executing the tasks
-------------------

Finally, running ::

   $ jug-execute jugfile.py

executes the tasks.
This command can be run multiple times, in parallel: for example, on a cluster.
Jug will automagically handle processing the tasks and storing intermediary results across worker nodes via the shared file system (NFS).


The complete jugfile
--------------------

.. literalinclude:: /../percolate/share/jugfile.py
   :linenos:
