The Newman--Ziff Algorithm
==========================

Introduction
------------

The scope of the Newman--Ziff algorithm is the simulation and statistical
analysis of percolation on graphs :cite:`Newman2001Fast`.
On regular lattices with :math:`N` sites, the algorithm takes time
:math:`\mathcal{O}(N)`, which is an enormous improvement compared to the
:math:`\mathcal{O}(N^2)` complexity of conventional percolation simulation
algorithms.

In *site percolation* problems, each node of the graph is either "occupied" or
"empty".
The occupied nodes induce a subgraph.
When the largest component of this subgraph spans the whole original graph, the
system is said to percolate.
The notion of such a spanning cluster is straightforward in regular lattices:
it either extends from one boundary to the opposite boundary, or wraps around
the whole system if there are no boundaries (e.g. periodic boundary
conditions).

In *bond percolation* problems, each edge of the graph is either "occupied" or
"empty".
The occupied edges induce a subgraph.
(A variant is to consider an edge either to exist or not, and consider the
resulting graph.)
As in site percolation, the system is said to percolate if the largest
component (cluster) of this subgraph extends across the whole original graph.

In *Bernoulli percolation* problems, let it be either site or bond percolation,
each site or bond is independently occupied with a probability :math:`p`.
This is a paradigm of percolation theory.
As the number of configurations grow exponentially with system size, one
resorts to Monte Carlo simulation: that is, sampling the space of all
configurations and computing the statistics from these samples.
In order to numerically trace the percolation transition with finite-size
scaling analysis, one needs to simulate several realizations (samples, runs)
over a range of increasing system sizes and sufficiently many values of the
order paramater :math:`p` in the critical region.
This entails simulating independent runs for almost arbitrarily close values of
:math:`p` in high-resolution studies.
While the computation time grows linearly with resolution, accuracy does not.
The intrinsic variance of the statistics at two adjacent values of :math:`p`,
due to the independence of runs, dominates the difference caused by the
differing values of :math:`p`.

The identification of clusters in any given configuration conventionally takes
:math:`\mathcal{O}(N)` time, which amounts to the overall
:math:`\mathcal{O}(N^2)` time requirement.


From microcanonical to canonical statistics
-------------------------------------------

In their 2001 paper, Neman & Ziff point out the following
:cite:`Newman2001Fast`.
The order parameter :math:`p` is originally defined microscopically as the
occupation probability.
However, and consequently, it is also the macroscopic average occupation ratio.
This is a weighted average over all configurations (microstates).
Their weight is of course determined by the microscopic occupation probability
:math:`p`.
Each such microstate has a fixed occupation number :math:`n`.
All configurations at a fixed :math:`n` constitute the *microcanonical
ensemble* at that number.
Weighting and averaging over a statistic of each microcanonical ensemble yields
the *canonical average* of that statistic.
Newman & Ziff refer to this thermodynamic analogy, where :math:`p` plays the
role of temperature, and the occupation number :math:`n` the role of the energy
of a microstate.

First, we sample all microcanonical ensembles (for each occupation number
:math:`n`), and compute the cluster statistic of interest.
Then, we convolve the resulting microcanonical averages with their weights in
the canonical ensemble of a given value of :math:`p`.
This yields the canonical average of the statistic at that value of :math:`p`.
This procedure enables us to compute the cluster statistics at an arbitrary
resolution of the order parameter :math:`p` from the precomputed microcanonical
averages.

For bond percolation, the probability that :math:`n` out of a total of
:math:`M` bonds are occupied at occupation probability :math:`p` is the
binomial probability mass function (:cite:`Newman2001Fast`, Equation 1):

.. math::

   B(M,n,p) = \binom{M}{n} p^n (1 - p)^{M-n}.

A microcanonical statistic :math:`\{Q_m\}` measured at each occupation number
:math:`m` transforms into the canonical average :math:`Q(p)` for occupation
probability :math:`p` according to the convolution (:cite:`Newman2001Fast`,
Equation 2):

.. math::

   Q(p) = \sum_{m=0}^M B(M,m,p)Q_m


Common Random Numbers and incremental cluster detection
-------------------------------------------------------

At the core of the Newman-Ziff algorithm is the incremental evolution of a set
of sample states (realizations) by successively adding bonds to an initially
empty lattice.
Specifically, a sample state with :math:`n + 1` occupied bonds derives from a
sample state with :math:`n` occupied bonds and one additional bond.
Instead of identifying the clusters for :math:`n + 1` from scratch, it takes
only little effort to derive the clusters from the previous configuration with
:math:`n` occupied bonds.
As a result, the overall time complexity of the algorithm reduces to
:math:`\mathcal{O}(N)`.

A convenient side effect is that this method effectively employs the variance
reduction technique of *common random numbers* for runs across all occupation
numbers :math:`n`.
Different realizations, or run, still remain independent, which is all we need.
At the same time, fluctuations solely due to using different realizations when
changing the occupation number :math:`n`, or, ultimately, the occupation
probability :math:`p`, now remain absent from the statistics.

Starting each run with a lattice with all :math:`M` bonds empty, we
successively add a bond to each configuration.
Addings bonds is random, but in a predetermined random permutation.
Each new bond might either join two sites from different clusters, resulting in
the merger of these two clusters, or the new bond joins two sites of the same
cluster.
Keeping track of this merging of clusters is taken care of by a weighted
union/find algorithm with path compression (see :cite:`Newman2001Fast` and
references therein).

In order to detect a percolating, spanning cluster, we introduce auxiliary
nodes to the graph, see :cite:`Newman2001Fast`, Section II.D and Figure 6.


