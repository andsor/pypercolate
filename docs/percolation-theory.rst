Percolation Theory
==================

Introduction
------------

Percolation theory characterizes how global connectivity emerges in a system of
a large number of objects.
These objects connect according to some local rule constrained by an underlying
topology.
Thus, given the topology and the local rule, percolation theory yields the
global, emergent behavior :cite:`Hunt2014Percolation`.
Early occurrences of percolation theory in the literature include the classic
works by Flory and Stockmayer on polymerization and the sol-gel transition
:cite:`Flory1941Molecular,Stockmayer1943Theory`.
However, it is only later that a theory of percolation starts to condense
:cite:`Broadbent1957Percolation`.

**Definition** (:cite:`Hunt2014Percolation`, p. 2):
We say a system *is at percolation*, or *percolates*, if sufficiently many
objects are connected locally such that a global connection emerges.
This global connection is a continuous "chain" or "cluster" of locally
connected objects, which is unbounded in size (in infinite systems), or of the
order of the system size (in finite systems).

Typically, percolation also refers to a stochastic process of increasing connectivity and eventual emergence of the giant cluster.
In an infinite system, this emergence in an ensemble of system configurations
constitutes a phase transition.
In fact, percolation is a phase transition paradigm
:cite:`Stauffer1994Introduction`.

The central quantity in percolation settings is the cluster size distribution
:math:`n_s`, which we will introduce shortly.
The setting of percolation is a graph.
A typical setting is a regular lattice of sites connected to their nearest
neighbors.
In *site percolation*, all sites are subsequently *occupied*.
In *bond percolation*, it is the bonds that are subsequently added to form a giant cluster of connected sites.

In the following, we introduce the concepts and notation mainly according to
Stauffer's and Aharony's classical textbook :cite:`Stauffer1994Introduction`.

The cluster size distribution
-----------------------------

**Definition**:
In the regular lattice setting, a *cluster* is a maximum set of occupied sites
which are pairwise joined by paths on the lattice only traversing occupied
sites.
In general, a *cluster* is component of the graph.

**Definition**:
The *size* :math:`s` of a cluster is the number of nodes in the component.

Note that infinite graphs allow for infinite cluster sizes.

The occupation of sites, or the cluster sizes, typically depend on a (global)
system parameter.
For example, the paradigmatic percolation model is that of each site
independently being occupied with some probability :math:`p` (Bernoulli percolation).
All the following statistics only require the general percolation setting of a
graph.
Let :math:`\varrho` denote the system parameter.

**Definition**:
For any given cluster size :math:`s`, let the *cluster number*
:math:`n_s(\varrho, L)` be the *density* of clusters of size :math:`s` with
respect to the system size :math:`L`.

In other words, in a system of :math:`L` sites, the cluster number
:math:`n_s(\varrho, L)` is the number :math:`N_s(\varrho, L)` of clusters of
size :math:`s` divided by the total number :math:`L` of sites,

.. math::

   n_s(\varrho, L) = \frac{N_s(\varrho,L)}{L}.

This definition also applies to systems of infinite size as

.. math::

   n_s(\varrho) = \lim_{L \to \infty} \frac{N_s(\varrho,L)}{L}.

*The cluster size distribution :math:`n_s` is the fundamental quantity in
percolation theory*.

Percolation threshold and characteristic cluster size
-----------------------------------------------------

Typically, in an infinite system the largest cluster grows with increasing
parameter :math:`\varrho`, and at some critical value :math:`\varrho_c`, an
infinite cluster appears.
This number :math:`\varrho_c` is the *percolation threshold*.
At and above :math:`\varrho_c`, there is an infinite cluster, and the system is
said to *percolate*.

The probability that a system of size :math:`L` percolates at parameter
:math:`\varrho`, i.e. has a cluster of order of the system size, is
:math:`\Pi(\varrho,L)`.
In the infinite system, we have

.. math::

   \Pi(\varrho) = \lim_{L \to \infty} \Pi(\varrho, L) = \begin{cases}
   0 & \varrho < \varrho_c, \\
   1 & \varrho \geq \varrho_c.
   \end{cases}

The *percolation strength* :math:`P(\varrho, L)` is the fraction of sites
belonging to the infinite cluster.
In the infinite system, the limit strength :math:`P(\varrho) = \lim_{L \to
\infty} P(\varrho, L)` is the typical *order parameter* of the percolation
transition.

The cluster number distribution typically is of the form

.. math::

   n_s(\varrho) \sim s^{-\tau} e^{- s/s_\xi}, \qquad (s \to \infty)

for large :math:`s` and with some characteristic cluster size :math:`s_\xi`.
At the percolation transition, the characteristic cluster size :math:`s_\xi`
diverges as a power law,

.. math::

   s_\xi \sim |\varrho_c - \varrho|^{-1/\sigma}, \qquad (\varrho \to \varrho_c)

with the *critical exponent* :math:`\sigma`.

In general, clusters of size :math:`s < s_\xi \sim |\varrho - \varrho_c|^{-1 /
\sigma}` dominate the moments of the cluster size distribution.
These clusters effectively follow a power-law distribution :math:`n_s(\varrho)
\sim s^{-\tau}`, as all clusters at the critical point :math:`n_s(\varrho_c)
\sim s^{-\tau}`.
For :math:`s \gg s_\xi`, the distribution is cut off exponentially.
Thus, clusters in this regime do not exhibit "critical" behavior.

Average cluster size
--------------------

For any given site, the probability that it is part of a cluster of size
:math:`s` is :math:`s n_s`. The *occupation probability*
:math:`p(\varrho, L)` is the probability that any given site is part of
a finite cluster, in a system of size :math:`L` (may be infinite) at
parameter :math:`\varrho`,

.. math::

   p(\varrho, L) = \sum_{s=1}^\infty s n_s(\varrho, L) = M_1(\varrho, L),

which is the first moment of the cluster size distribution.

Hence, for any given site of any given finite cluster, the probability
:math:`w_s(\varrho, L)` that the cluster is of size :math:`s`, is

.. math::

   w_s(\varrho, L) = \frac{1}{p(\varrho,L)} s n_s(\varrho, L),

with :math:`\sum_{s=1}^\infty w_s(\varrho, L) = 1`.

For any given site of any given finite cluster, the average size
:math:`S(\varrho, L)` of the cluster is

.. math::

   S(\varrho, L) = \sum_{s=1}^\infty s w_s(\varrho, L) = \frac{1}{p(\varrho, L}
   \sum_{s=1}^\infty s^2 n_s(\varrho, L) = \frac{M_2(\varrho, L)}{M_1(\varrho,
   L)},

which is the second moment divided by the first moment of the cluster size
distribution.
Note that this average is different from the average of the (finite) cluster
sizes in the system.
The *average cluster size* :math:`S(\varrho, L)` is defined with respect to a
site, and thus, it is an intensive quantity :cite:`Stauffer1994Introduction`.

Note that for infinite systems (:math:`L\to\infty`), these statistics exclude
the infinite cluster.
At the critical point, the average cluster size :math:`S(\varrho_c)`
nevertheless diverges as

.. math::

   S(\varrho) \sim |\varrho - \varrho_c|^{- \gamma}, \qquad (\varrho \to
   \varrho_c)

with the *critical exponent* :math:`\gamma`.
As :math:`S` is the second moment of the cluster size distribution (up to a
factor), it is a measure of fluctuations in the system.
*Thus, divergence of :math:`S` actually defines the percolation phase
transition.*

Correlation length
------------------

**Definition**:
The *correlation function* :math:`g(\mathbf{r})` is the probability that a site
at position :math:`\mathbf{r}` from an occupied site in a *finite* cluster
belongs to the same cluster.

Typically, for large :math:`r \equiv |\mathbf{r}|`, there is an exponential
cutoff, i.e. :math:`g(\mathbf{r}) \sim e^{-r/\xi}`, at the *correlation length*
:math:`\xi`.
Another critical exponent :math:`\nu` determines the divergence of :math:`\xi`
at the critical point as

.. math::

   \xi \sim |\varrho - \varrho_c|^{-\nu} \qquad (\varrho \to \varrho_c).

**Definition**:
The *correlation length* :math:`\xi` is defined as

.. math::

   \xi^2 = \frac{\sum_{\mathbf{r}} r^2 g(\mathbf{r})}{\sum_{\mathbf{r}}
   g(\mathbf{r})}.

For a cluster of size :math:`s`, its *radius of gyration* :math:`R_s` is
defined as the average square distance to the cluster center of mass :cite:`Stauffer1994Introduction`.
It turns out that :math:`2 R_s^2` is the average square distance between two
sites of the same (finite) cluster.
Averaging over :math:`2R_s^2` yields the squared correlation length
:cite:`Stauffer1994Introduction`,

.. math::

   \xi^2 = \frac{\sum_s 2 R_s^2 s^2 n_s}{\sum_s s^2 n_s},

since :math:`s^2 n_s` is the weight of clusters of size :math:`s`.
Hence, the correlation length is the radius of the clusters that dominate the
second moment of the cluster size distribution, or, the fluctuations.

The divergence of quantities at the critical point involves sums over all
cluster sizes :math:`s`.
The cutoff of the cluster number :math:`n_s` at :math:`s_\xi \sim |\varrho -
\varrho_c|^{-1/\sigma}` marks the *cluster sizes* :math:`s \approx s_\xi` that
contribute the most to the sums and hence, to the divergence.
This also holds for the correlation length :math:`\xi`, which is the radius of
those clusters of sizes :math:`s \approx s_\xi`.
As such, this is the one and only length scale which characterizes the behavior
of an infinite system in the critical region :cite:`Stauffer1994Introduction`.

The correlation length :math:`\xi` defines the relevant length scale.
As :math:`\xi` diverges at :math:`\varrho \to \varrho_c`, the length scale
vanishes at the percolation transition :math:`\varrho = \varrho_c`.
This lack of a relevant length scale is a typical example of *scale
invariance*.
This implies that the system appears to be self-similar on length scales
smaller than :math:`\xi`.
As :math:`\xi` becomes infinite at :math:`\varrho_c`, the whole system becomes
self-similar.
The lack of a relevant length scale also implies that functions of powers
(*power laws*) describe the relevant quantities in the critical region.
In particular, the correlation length itself diverges as a power law,

.. math::

   \xi \sim (\varrho - \varrho_c)^{-\nu}. \qquad (\varrho \to \varrho_c)

The form of this divergence is the same in all systems, which is called
*universal* behavior.
The *critical exponent* :math:`\nu` depends only on general features of the
topology and the local rule, giving rise to *universality classes* of systems
with the same critical exponents.

Scaling relations
-----------------

The scaling theory of percolation clusters relates the critical exponents of
the percolation transition to the cluster size distribution
:cite:`Stauffer1979Scaling`.
As the critical point lacks any length scale, the cluster sizes also need to
follow a power law,

.. math::

   n_s(\varrho_c) \sim s^{-\tau}, \qquad (\varrho \to \varrho_c, s \gg 1)

with the *Fisher exponent* :math:`\tau` :cite:`Fisher1967Theory`.
The scaling assumption is that the ratio :math:`n_s(\varrho) / n_s(\varrho_c)`
is a function of the ratio :math:`s / s_\xi(\varrho)` only
:cite:`Stauffer1979Scaling`,

.. math::

   \frac{n_s(\varrho)}{n_s(\varrho_c)} = f\left( \frac{s}{s_\xi(\varrho)}
   \right), \qquad (\varrho \to \varrho_c, s \gg 1).

As in the critical region, the characteristic cluster size diverges as
:math:`s_\xi \sim |\varrho - \varrho_c|^{-1/\sigma}`, we have :math:`s /
s_\xi(\varrho) \sim |(\varrho - \varrho_c) s^\sigma |^{1/\sigma}`, and hence

.. math::

   n_s(\varrho) \sim s^{-\tau} f((\varrho - \varrho_c) s^\sigma), \qquad
   (\varrho \to \varrho_c, s \gg 1).

The following scaling law relates the system dimensionality :math:`d` and the
fractal dimensionality :math:`D = \frac{1}{\sigma \nu}` of the infinite cluster
to the exponents of the cluster size distribution :cite:`Hunt2014Percolation`.

.. math::

   \frac{\tau - 1}{\sigma \nu} = d, \qquad \tau = 1 + \frac{d}{D}



Consider the :math:`k`-th raw moment of the cluster size distribution

.. math::

   M_k(\varrho) = \sum_s s^k n_s(\varrho)

which scales as

.. math::

   M_k(\varrho) \sim \sum_s s^{k-\tau} e^{-s/s_\xi(\varrho)} \sim |\varrho -
   \varrho_c|^{(\tau -1 - k)/\sigma} \qquad (\varrho \to \varrho_c)

in the critical region.

In this region, above the percolation threshold (:math:`\varrho > \varrho_c`),
the percolation strength behaves as :cite:`Stauffer1994Introduction`

.. math::

   P(\varrho) \sim \sum_s s (n_s(\varrho_c) - n_s(\varrho)) \sim \sum_s
   s^{1-\tau}  \left(1 - e^{-s/s_\xi(\varrho)} \right) \sim (\varrho -
   \varrho_c)^{(\tau -2)\sigma} \equiv (\varrho - \varrho_c)^\beta

defining the *critical exponent* :math:`\beta` as

.. math::

   \beta = \frac{\tau - 2}{\sigma}.

As the second raw moment :math:`M_2(\varrho) \sim |\varrho - \varrho_c|^{(\tau
- 3)/\sigma}`, we have

.. math::

   \gamma = \frac{3 - \tau}{\sigma},

or

.. math::

   \sigma = \frac{1}{\beta + \gamma}, \tau = 2 + \frac{\beta}{\beta + \gamma}.

These are the *scaling relations* between the critical exponents, which all
derive from the exponents :math:`\tau` and :math:`\sigma` of the cluster size
distribution.

Cluster numbers typically scale as

.. math::

   n_s(\varrho) \sim s^{-\tau} f((\varrho - \varrho_c) s^\sigma), \qquad
   (\varrho \to \varrho_c, s \to \infty)

with some scaling function :math:`f` which rapidly decays to zero, :math:`f(x)
\to 0` for :math:`|x| > 1` (:math:`s > s_\xi`)
:cite:`Stauffer1994Introduction`.

It remains to determine the scaling relationship of cluster radius :math:`R_s`
and cluster size :math:`s` in the critical region.
For :math:`s \sim R_s^D` for some possibly fractal cluster dimension :math:`D`,
we have :cite:`Stauffer1994Introduction`

.. math::

   \frac{1}{D} = \sigma \nu.

The cutoff cluster size :math:`s_\xi` was the *crossover size* separating
critical behavior (:math:`n_s \sim s^{-\tau}`) from non-critical behavior
(:math:`n_s \to 0` exponentially fast).
Now, the correlation length :math:`\xi \sim s_\xi^{1/D} = s_\xi^{\sigma \nu}`
is the *crossover length* separating the critical and non-critical regimes.
