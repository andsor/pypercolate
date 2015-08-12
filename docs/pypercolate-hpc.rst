A Python Implementation of the Newman--Ziff Algorithm for High-Performance Computing (HPC)
==========================================================================================

The *percolate.hpc* module implements the Newman--Ziff algorithm for bond percolation on graphs, tailored towards high-performance computing (HPC).
This guide introduces the design of the *hpc* module to the simulationist.
It assumes that the reader is familiar with the :doc:`pypercolate <pypercolate>` package and high-performance computing in general.

Performance considerations
--------------------------

The *hpc* module is designed to allow for percolation studies on medium-sized and large graphs with approximately :math:`10^3` to :math:`10^7` nodes), and averaged over an arbitrarily large number of runs.
Currently, it is limited by memory consumption in particular of the networkx graph.
A networkx graph consumes approximately :math:`1` GiB per :math:`10^6` nodes.
An efficient implementation would store the list of edges in memory.
Let us assume that the number of edges is of the same order as the number of nodes :math:`N`.
For up to approximately :math:`10^9` nodes, each node label requires an unsigned integer of :math:`32` bits or :math:`4` bytes.
Hence, each edge consumes :math:`2 \times 4` bytes of memory.
At :math:`N` edges, the list of edges requires of the order of :math:`10 N` bytes of memory.
A more efficient implementation hence would require about 2 orders of magnitude less memory.
The `graph-tool <http://graph-tool.skewed.de/>`_ Python package provides a more efficient implementation through the native-C++ Boost Graph Library :cite:`Peixoto2014Graphtool`.
Future improvements of the percolate.hpc module could implement graph-tool graphs.
However, graph-tool requires compilation and hence, careful setup, while networkx is trivial to install into every Python environment.

Template for a HPC bond percolation finite-size scaling study
-------------------------------------------------------------

percolate.hpc supports the following control flow for a bond percolation finite-size scaling study.
The input parameters of such a study are:

* a list of finite system sizes :math:`L_i`
* the number of runs per task :math:`n_{\text{runs}}`
* the number of tasks :math:`N_{\text{tasks}}`
* the total number of runs :math:`N_{\text{runs}} = N_{\text{tasks}} \cdot n_{\text{runs}}`
* the realizations :math:`\omega_j` (one per run)
* the occupation probabilities :math:`p_k` at which to compute the macrocanonical averages for each system size :math:`L_i`
* the confidence level :math:`1 - \alpha` for the confidence intervals

Each run is determined by a particular realization :math:`\omega_j` of a permutation of bonds to add one-by-one.
Typically, this realization :math:`\omega_j` is the seed of a pseudo-random number generator (RNG) which reproducibly produces the permutation.
The usual considerations of parallel stochastic experiments apply here.
For example, the seeds and streams of random numbers of distinct runs need to be independent.
For this study, we assume that the astronomically large size of the Mersenne--Twister RNG sufficiently avoids any overlap between runs.

First, each finite system size provides for its own independent sub-study.
Each sub-study is structured as follows.

#. Prepare the NetworkX graph of the current system size :math:`L_i` (user-defined).
#. Prepare the graph and auxiliary information for the pypercolate routines with :meth:`percolate.percolate.percolation_graph`.
#. For every value :math:`p_k` of the occupation probability at the current system size, precompute the convolution coefficients :math:`f_n` with :meth:`percolate.percolate._binomial_pmf` for all occupation numbers :math:`n = 0, \ldots, M_i` (where :math:`M_i` is the number of edges at system size :math:`L_i`).
#. Execute :math:`N_{\text{tasks}}` tasks.
#. Reduce the tasks results with :meth:`percolate.hpc.bond_reduce`.
#. Finalize the tasks results with :meth:`percolate.hpc.finalize_macrocanonical_averages`.
#. Write finalized results to disk.

Each task employs the Map--Reduce paradigm to perform :math:`n_{\text{runs}}` independent runs.
Basically, a tasks *maps* a *run* function onto the seeds :math:`\omega_j`, *reduces* the outputs with :meth:`percolate.hpc.bond_reduce`, and returns the result.
The output of a task is a NumPy structured array with one row per occupation probability :math:`p_k` and the following fields:

* The number of runs :math:`N_{\text{runs}}`
* The mean and sum of squared differences to the mean for the

  - percolation probability
  - maximum cluster size (absolute number of nodes)
  - raw moments (absolute number of nodes)
  
The :meth:`~percolate.hpc.bond_reduce` method wraps the :meth:`simoa.stats.online_variance` method, which in turn implements the single-pass pairwise algorithm to calculate the variance by Chan et al.
This algorithm stores and updates the sum of squared differences to the mean.
Dividing this sum by the sample size minus :math:`1` yields the variance.

The :meth:`~percolate.hpc.finalize_macrocanonical_averages` takes the reduced output of all tasks of a system size.
It throughputs the means, and calculates the sample standard deviation from the sum of squared differences.
It further computes the standard confidence intervals at the given :math:`1 - \alpha` level of confidence.
The output is a NumPy structured array with one row for each occupation probability :math:`p_k` and the following fields:

* the total number of runs :math:`N_{\text{runs}}`,
* the occupation probability :math:`p_k`,
* the confidence level :math:`1 - \alpha`,
* mean, sample standard deviation and :math:`1 - \alpha` confidence interval of
  
  - the percolation probability,
  - the percolation strength (the maximum cluster size normalized by the total number of nodes)
  - the raw moments (normalized by the total number of nodes)

Each *run* consists of the following steps:

#. Accumulate the microcanonical cluster statistics :math:`Q_n` across all occupation numbers :math:`n` with :meth:`percolate.hpc.bond_microcanonical_statistics`.
#. For each occupation probability :math:`p_k`, convolve the microcanonical cluster statistics :math:`Q_n` into the macrocanonical statistics :math:`Q_p` with the pre-computed convolution coefficients :math:`f_n` and :meth:`percolate.hpc.bond_macrocanonical_statistics`
#. Prepare the canonical statistics of this single run for reducing with other runs with :meth:`percolate.hpc.bond_initialize_macrocanonical_averages`.

Amendment to the original Newman--Ziff algorithm
------------------------------------------------

Newman & Ziff point out that the linear operations of averaging and convolving commute.
As the convolution is more intricate, they advise to average over the runs first (microcanonical averages), and apply the convolution to the microcanonical averages to arrive at the canonical averages.
The design choice here is to perform the convolution first.
The reason is that the number of occupation probabilities typically is of the order of :math:`10^1` to :math:`10^2`, irrespective of the system size.
In contrast, the order of occupation numbers is of the order of the system size, thus reaching :math:`10^6` and beyond.
In order to restrain memory consumption, the convolution immediately reduces the result of each run to a negligible amount of memory.
Intermediary run results thus need not be kept in memory.
On the other hand, we could also reduce the microcanonical run results on the fly, which would also spare us from keeping intermediary results.
However, we note here that the calculation of the variance and confidence intervals is a non-linear operation, and hence in general the order of performing this operation and averaging matters.
As the variances and confidence intervals are due to ensemble averaging, they should be performed only on the canonical cluster statistics for each single run.
This is also how they would be performed if one were to sample the canonical ensemble directly, by creating different realizations of the bond percolation setting at every occupation probability.

Implicit NumPy Multithreading
-----------------------------

We also note another design choice here that relates to high-performance computing on a shared cluster.
NumPy is built against sophisticated linear algebra libraries (BLAS and derivates), and technically, these libraries might use multithreading for certain operations.
For example, the *numpy.dot* function might transparently run on several cores/threads for 2-dimensional matrices.
While this is a design feature running NumPy on a single workstation, it is generally undesired in a shared cluster computing environment.
As a queueing system handles the allocation of tasks to the worker nodes, it expects each task to consume only one core.
Using multiple cores impedes other tasks running concurrently on the same worker node.
Hence, we deliberately avoid the *dot* product and stick to elementary NumPy operations.
For example, the following two statements are equivalent::

    np.dot(a, b) # might use multithreading
    np.sum(a * b) # does not use multithreading
