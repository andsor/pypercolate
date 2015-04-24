The Newman--Ziff Algorithm
==========================

Introduction
------------

Newman & Ziff observed that usually, to record cluster statistics depending on
the occupation probability :math:`p` in Bernoulli percolation, one
independently simulates a number of realizations at each parameter value
:math:`p`.
This means the runtime is proportional to the number of sampled parameter
values :math:`p`.

Common Random Numbers
---------------------

A first (and common) enhancement is to reduce the variance by using *common
random numbers*:
instead of simulating entirely different realizations at each parameter value
:math:`p`, one uses the same random numbers for the "same" run at each
:math:`p`.
This way, the runs remain independent, which is all we need.
At the same time, fluctuations solely due to using different realizations when
changing the paramter value now remain absent from the statistics.
Another way of picturing this is to say that by Common Random Numbers, we
simulate the whole trajectory across all values of :math:`p` first, and then
average over all trajectories.
Without Common Random Numbers, we would instantiate different trajectories and
average different trajectories at each :math:`p` independently.

From microcanonical to canonical statistics
-------------------------------------------

The main idea of Newman & Ziff is the following:
Instead of recalculating the lattice configuration and all statistics at each
parameter value :math:`p`, it suffices to calculate the statistics at each
number of edges (occupation number) an then average over all configurations
according to ther weight at each parameter value of :math:`p`.
As this resembles the thermodynamic averaging of microstates of fixed energy to
states of fixed temperature (average energy), with weights given by the
Boltzmann factors, Newman & Ziff call this procedure determining the
*canonical* average (average occupation probability :math:`p`) from the
*microcanonical ensembles* at fixed occupation numbers.

For bond percolation, the probability that :math:`m` out of a total of
:math:`M` bonds are occupied at occupation probability :math:`p` is the
binomial probability mass function (:cite:`Newman2001Fast`, Equation (1)):

.. math::

   B(M,m,p) = \binom{M}{m} p^n (1 - p)^{M-m}.

A microcanonical statistic :math:`\{Q_m\}` measured at each occupation number
:math:`m` transforms into the canonical average :math:`Q(p)` for occupation
probability :math:`p` according to the convolution (:cite:`Newman2001Fast`,
Equation (2)):

.. math::

   Q(p) = \sum_{m=0}^M B(M,m,p)Q_m

Evolving a single realization
-----------------------------

Starting with a lattice with all :math:`M` bonds unoccupied, we successively
add a bond.
Addings bonds is random, but in a predetermined random permutation.
Each new bond might either join two sites from different clusters, resulting in
the merger of these two clusters, or the new bond joins two sites of the same
cluster.
Keeping track of this merging of clusters is taken care of by a weighted
union/find algorithm with path compression (see :cite:`Newman2001Fast` and
references therein).

In order to detect a percolating, spanning cluster, we introduce auxiliary
nodes to the graph, see :cite:`Newman2001Fast`, Section II.D and Figure 6.
