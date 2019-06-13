.. _user_basis-label:


What does the `user_basis` allow and how does it work?
--------
List of features:

* ...
* ...
* ...

Inner workings and functionality: basic structure of the `user_basis` class
--------
The user basis brings to surface the low-level functionality of the `basis_general` classes and allows the user to customize the relevant function methods. 

Integer representation of states 
++++++++
The computational basis in QuSpin is the local particle/spin-z basis. For computational efficiency, basis states are stored as bit representations of integers. In order to fully grasp the underlying ideas behind the `user_basis` functionality, it is required to explain in detail how this works.


Spin-1/2, Fermions, Hardcore Bosons
````````````
Spin-:math:`1/2`, hardcore-boson, and fermion states all have in common that they allow at most one particle pre site. Thus, in the particle/z-spin basis, the basis states are given by strings of ones and zeros. The number of ones sets the particle number/magnetization of a given state. 

Coincidentally, integers are stored in memory using the so-called binary representation: every integer is assigned a unique bit string. Therefore, a particularly efficient way of storing basis states is via this integer representation. 


**CONVENTION**: in QuSpin, the integer corresponding to a given spin configuration is given by the integer representation of the **reversed** bitstring. For instance in a chain of four sites, we have:

* :math:`|0000\rangle \leftrightarrow 0`
* :math:`|1000\rangle \leftrightarrow 1`
* :math:`|0100\rangle \leftrightarrow 2`
* :math:`|0001\rangle \leftrightarrow 8`

*Note*: the `user_basis` allows the user to adopt their own convention **DOES IT REALLY**?. For consistency, this tutorial follows the above QuSpin convention. 

Definition:
.........

Cosider an $N$-site lattice. Let :math:`c_i` denote the occupation of site :math:`i \in [0,N-1]` (e.g., for :math:`|0100\rangle` we have :math:`c_1=1` and :math:`c_{i\neq 1}=0`). Then, a generic expression for the integer :math:`s`, corresponding to the state :math:`|\{c_i\}\rangle`, is given by

.. math::
	s = \sum_{j=0}^{N-1} c_j 2^j


Reading off particle occupation on a given site:
.........

To read off the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites), do

>>> j = N - j - 1     # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> occ = (s>>j)&1    # occupation: either 0 or 1


Flipping particle occupation on a given site:
.........

To flip the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites), use the XOR operator `^`:

>>> j = N - j - 1     # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = 1; b <<= j    # compute a "mask" integer b which is 1 on site j and zero elsewhere
>>> s ^= b            # flip occupation on site j


Bosons, Higher Spins
````````````


When dealing with bosons or higher spins, the binary representation is no longer sufficient, since the local on-site occupation can be larger than one. 


Definition:
.........

Denoting by :math:`sps` (States Per Site) the local Hilbert space dimension, the integer compression of basis states generalizes to:

.. math::
	s = \sum_{j=0}^{N-1} c_j sps^j

For instance in a chain of four sites with at most two particles per site (:math:`sps=3`), we have:

* :math:`|0000\rangle \leftrightarrow 0`
* :math:`|1000\rangle \leftrightarrow 1`
* :math:`|0100\rangle \leftrightarrow 3`
* :math:`|0200\rangle \leftrightarrow 6`
* :math:`|0120\rangle \leftrightarrow 21`
* :math:`|0001\rangle \leftrightarrow 27`


Reading off particle occupation on a given site:
.........
To read off the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> occ = (s//(sps**j))%sps  # occupation: can be 0, 1, ..., sps-1


Increasing the particle occupation on a given site:
.........
To increase the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = sps**j               # obtain mask integer b
>>> occ = (s//b))%sps        # compute occupation on site j
>>> if (occ+1<sps): r += b   # increase occupation on site j by one



Decreasing the particle occupation on a given site:
.........
To decrease the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = sps**j               # obtain mask integer b
>>> occ = (s//b)%sps         # compute occupation on site j
>>> if (occ>0): r -= b       # decrease occupation on site j by one


*Note*: even though in the case :math:`sps=2`, the above expressions reproduce the corresponding spin-1/2 expressions, they are not as efficient computationally.


`user_basis` methods as customizable function methods of `basis_general`
-------

The core parent class for all `basis_general` classes contains a number of function methods to facilitate the construction of the basis and the basis methods. The `user_basis` exposes those methods which can be re-defined/overridden by the user. This enhances the functionslity of QuSpin, allowing the user maximum flexibility in constructing basis objects. 

Below, we give a brief overview of the methods required to define `user_basis` objects.


`op` (user-defined action of operators on the integer states)
++++++


`next_state` (user-defined particle conservation rule)
++++++


`pre_check_state` (user-defined extra projection of states out of the basis)
++++++


`count_particles`
++++++


Symmetry transformations from bit operations
-------

`maps` dictionary
++++++

System-size independent symmetries
++++++


Symmetries for fixed system sizes using precomputed masks
++++++++



Examples
--------

Scripts to construct spin, fermion, and boson bases 
++++++++
Demonstrate that the `user_basis` recovers the functionality of the `basis_general` classes:

* spin-1/2 Heisenberg model in 1d
* spinless fermions with nearest-neighbor interactions in 1d
* Bose-Hubbard model in 1d


Scripts to demonstrate the additional functionality introduced by the `user_basis`
++++++++
*
*
*


