.. _user_basis-label:


What does the `user_basis` allow and how does it work?
--------
List of features:

* ...
* ...
* ...

For examples, see [link to below]

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

Cosider an :math:`N`-site lattice. Let :math:`c_i` denote the occupation of site :math:`i \in [0,1,\dots,N-1]` (e.g., for :math:`|0100\rangle` we have :math:`c_1=1` and :math:`c_{i\neq 1}=0`). Then, a generic expression for the integer :math:`s`, corresponding to the state :math:`|\{c_i\}\rangle`, is given by

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

Denoting by :math:`sps` (states per site) the local Hilbert space dimension, the integer compression of basis states generalizes to:

.. math::
	s = \sum_{j=0}^{N-1} c_j sps^j

For instance in a chain of four sites with at most two particles per site (i.e., three states: :math:`sps=3`), we have:

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


*Notes*:
```````````` 

* even though in the case :math:`sps=2`, the above expressions reproduce the corresponding spin-1/2 expressions, they are not as efficient computationally.
* convenient quspin functions to transform between integer and quspin bit representations are `basis.int_to_state()` and `basis.state_to_int()`. 
* the attribute `basis.states` holds all states of the basis in their integer representation.
* printing a basis object `print(basis)` displays the states in their quantum mechanical notation. 


`user_basis` function methods
-------

The core parent class for all `basis_general` classes contains a number of function methods to facilitate the construction of the basis and the basis methods. The `user_basis` exposes those methods which can be re-defined/overridden by the user. This enhances the functionality of QuSpin, allowing the user maximum flexibility in constructing basis objects. 

Below, we give a brief overview of the methods required to define `user_basis` objects.


`op()`
++++++
This method contains user-defined action of operators on the integer states.


`next_state()` 
++++++
This method provides a user-defined particle conservation rule.


`pre_check_state()`
++++++
This *optional* method provides user-defined extra projection of states out of the basis.

`count_particles()`
++++++
This *optional* method counts the total number of particles/magnetization in a given state.



Symmetry transformations from bit operations
-------
Any discrete symmetry is uniquely defined by its action on the basis states. Since the basis is stored in the integer representation, the symmetry operations have to be defined to transform integers. In the `basis_1d` and `basis_general` classes this is done under the hood; the `user_basis` brings this functionality to the surface, and allows the user to modify it accordingly.

  

System-size independent symmetries
++++++
System-size independent symmetries contain as a parameter the system size :math:`N`. As a result, they apply to all system sizes. Examples of such  


Symmetries for fixed system sizes using precomputed masks
++++++++
The convenience to define symmetry maps which apply to all system sizes comes at a certain efficiency cost. This can be circumvented by defining system-size specific maps, using integer masks to perform the bit operations. These masks also depend on the data type of the integer storing the state. 

Luckily, there is a great tool to compute the symmetry maps, available at http://programming.sirrida.de/calcperm.php. All one needs to do is find the permutation of the lattice sites under the symmetry, and pass it to the tool to obain the symmetry map that acts on integers. Let us demonstrate how this works using two examples.

parity 
````````
Consider a ladder of :math:`2\times 10` sites, labelled 0 through 19. The action of parity/reflection along the long ladder axis is easily defined on the lattice sites to be

.. math::
	[0,\ 1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9;\ 10,\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19] \mapsto [9,\ 8,\ 7,\ 6,\ 5,\ 4,\ 3,\ 2,\ 1,\ 0;\ 19,\ 18,\ 17,\ 16,\ 15,\ 14,\ 13,\ 12,\ 11,\ 10]

Passing the transformed integer sequence (right-hand side) to the online generator http://programming.sirrida.de/calcperm.php, it returns the symmetry map

.. code-block:: python
   
   def parity(x,N,sign_ptr):
       """ works for N=10 sites and 32 bit-integers spin-1/2 states only """
       return 	 (  ((x & 0x00004010) << 1)
                  | ((x & 0x00002008) << 3)
                  | ((x & 0x00001004) << 5)
                  | ((x & 0x00000802) << 7)
                  | ((x & 0x00000401) << 9)
                  | ((x & 0x00080200) >> 9)
                  | ((x & 0x00040100) >> 7)
                  | ((x & 0x00020080) >> 5)
                  | ((x & 0x00010040) >> 3)
                  | ((x & 0x00008020) >> 1)) 

This map works only for this system size, and for 32-bit integers. 

translation
````````
Consider again a ladder of :math:`2\times 10` sites, labelled 0 through 19. The action of translation along the long ladder axis is easily defined on the lattice sites to be

.. math::
	[0,\ 1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9;\ 10,\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19] \mapsto [1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 0;\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19,\ 10]

corresponds to the bit operation (again, fixed system size and data type):

.. code-block:: python
   
   def translation(x,N,sign_ptr):
       """ works for N=10 sites and 32 bit-integers spin-1/2 states only. """
       return ((x & 0x0007fdff) << 1) | ((x & 0x00080200) >> 9)

`maps` dictionary
++++++
In the `user_basis`, the functions encoding the symmetry action are referred to as maps. Every map has as its first argument the integer (state) to be tansformed, followed by the number of sites. For fermionic systems, the symmetry action can also modify the fermion sign of a given state. Therefore, the last argument is a `sign_ptr`. 


Symmtries are passed to the `user_basis` constructor via a python dictionary, called `maps`. The keys are arbitrary strings which define a unique name for each map; the corresponding values are tuples of three entries: `(map function, symmetry periodicity, quantum number)`. The symmetry periodicity (or cyclicity) is the smallest integer :math:`l`, such that :math:`T^l = T`. 

>>> maps = dict(T=(translation,10,0), P=(parity,2,0), )

**Note**: the map functions need to be cast as decorated numba cfuncs (see below).


Using `numba` to pass python functions to the `C++` `user_basis` constructor
-------
The function methods of `user_basis` discussed above, are passed to the `user_basis` constructor. Since the latter is written in `C++` for speed, we use  the `numba` package to decorate python functions which are automatically compiled to `C++` and then parsed to the `user_basis`. 


Data types
++++++++

Function decorators
++++++++



Examples
--------
Below, we provide examples which demonstrate how to use the `user_basis` class. 


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


