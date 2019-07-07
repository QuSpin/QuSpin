.. _user_basis-label:


A tutorial on QuSpin's `user_basis`
--------
List of functionality features which `user_basis` allows the user to access, and which are **not** accessible in the `*_basis_1d` and `*_basis_general`:

* ...
* ...
* ...

The user basis brings to surface the low-level functionality of the `basis_general` classes and allows the user to customize the relevant function methods. 

**TUTORIAL CONTENTS:**

* `Integer representation of states`_
	* `Spin-1/2, Fermions, Hardcore Bosons`_
	* `Bosons, Higher Spins`_
* `user_basis function methods`_
	* `op(op_struct_ptr, op_str, site_ind, N, args)`_
	* `next_state(s, counter, N, args)`_
	* `pre_check_state(s, N, args)`_
	* `count_particles()`_
* `Symmetry transformations from bit operations`_
	* `System-size independent symmetries`_
	* `Symmetries for fixed system sizes using precomputed masks`_
	* `Symmetry maps dictionary`_
* `Using numba to pass python functions to the c++ user_basis constructor`_
	* `Data types`_
	* `Function decorators`_
* `Example Scripts`_
	* `Scripts to construct spin, fermion, and boson bases`_
	* `Scripts to demonstrate the additional functionality introduced by the user_basis`_






Integer representation of states 
--------
The computational basis in QuSpin is the local particle/spin-z basis. For computational efficiency, basis states are stored as bit representations of integers. In order to fully grasp the underlying ideas behind the `user_basis` functionality, it is required to explain in detail how this works.


Spin-1/2, Fermions, Hardcore Bosons
++++++++
Spin-:math:`1/2`, hardcore-boson, and fermion states all have in common that they allow at most one particle pre site. Thus, in the particle/z-spin basis, the basis states are given by strings of ones and zeros. The number of ones sets the particle number/magnetization of a given state. 

Coincidentally, integers are stored in memory using the so-called binary representation: every integer is assigned a unique bit string. Therefore, a particularly efficient way of storing basis states is via this integer representation. 


bitstring is the same; site label is reversed relative to bitstring; 

**CONVENTION**: in QuSpin, the integer corresponding to a given spin configuration is given by the integer representation of the Fock state bitstring. The lattice indices are defined in **reversed** order. For instance in a chain of four sites, we have:

* :math:`|0000\rangle \leftrightarrow 0`:    empty lattice,
* :math:`|0001\rangle \leftrightarrow 1`:    one particle at site 3,
* :math:`|0010\rangle \leftrightarrow 2`:    one particle at site 2,
* :math:`|1000\rangle \leftrightarrow 8`:    one particle at site 0,

*Note*: the `user_basis` allows the user to adopt their own convention. For consistency, this tutorial follows the above QuSpin convention. 

Definition:
````````````

Cosider an :math:`N`-site lattice. Let :math:`c_i` denote the occupation of site :math:`i \in [0,1,\dots,N-1]` (e.g., for :math:`|0100\rangle` we have :math:`c_1=1` and :math:`c_{i\neq 1}=0`). Then, a generic expression for the integer :math:`s`, corresponding to the state :math:`|\{c_i\}\rangle`, is given by

.. math::
	s = \sum_{j=0}^{N-1} c_j 2^{N-1-j}


Reading off particle occupation on a given site:
````````````

To read off the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites), do

>>> j = N - j - 1     # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> occ = (s>>j)&1    # occupation: either 0 or 1


Flipping particle occupation on a given site:
````````````

To flip the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites), use the XOR operator `^`:

>>> j = N - j - 1     # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = 1; b <<= j    # compute a "mask" integer b which is 1 on site j and zero elsewhere
>>> s ^= b            # flip occupation on site j


Bosons, Higher Spins
+++++++


When dealing with bosons or higher spins, the binary representation is no longer sufficient, since the local on-site occupation can be larger than one. 


Definition:
````````````

Denoting by :math:`sps` (states per site) the local Hilbert space dimension, the integer compression of basis states generalizes to:

.. math::
	s = \sum_{j=0}^{N-1} c_j sps^{N-1-j}

For instance in a chain of four sites with at most two particles per site (i.e., three states: :math:`sps=3`), we have:

* :math:`|0000\rangle \leftrightarrow 0`:    empty lattice,
* :math:`|0001\rangle \leftrightarrow 1`:    one particle at site 3,
* :math:`|0010\rangle \leftrightarrow 3`:    one particle at site 2,
* :math:`|0020\rangle \leftrightarrow 6`:    two particles at site 2,
* :math:`|0210\rangle \leftrightarrow 21`:    one particle at site 2 and two particles at site 1,
* :math:`|1000\rangle \leftrightarrow 27`:    one particle at site 0,


Reading off particle occupation on a given site:
````````````
To read off the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> occ = (s//(sps**j))%sps  # occupation: can be 0, 1, ..., sps-1


Increasing the particle occupation on a given site:
````````````

To increase the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = sps**j               # obtain mask integer b
>>> occ = (s//b))%sps        # compute occupation on site j
>>> if (occ+1<sps): r += b   # increase occupation on site j by one



Decreasing the particle occupation on a given site:
````````````

To decrease the particle occupation on site :math:`j` of the state :math:`s` (with a total of :math:`N` sites and :math:`sps` states per site), do

>>> j = N - j - 1            # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
>>> b = sps**j               # obtain mask integer b
>>> occ = (s//b)%sps         # compute occupation on site j
>>> if (occ>0): r -= b       # decrease occupation on site j by one


*Notes*:
+++++++

* even though in the case :math:`sps=2`, the above expressions reproduce the corresponding spin-1/2 expressions, they are not as efficient computationally.
* convenient quspin functions to transform between integer and quspin bit representations are `basis.int_to_state()` and `basis.state_to_int()`. 
* the attribute `basis.states` holds all states of the basis in their integer representation.
* printing a basis object `print(basis)` displays the states in their quantum mechanical notation. 


`user_basis` function methods
-------

The core parent class for all `basis_general` classes contains a number of function methods to facilitate the construction of the basis and the basis methods. The `user_basis` exposes those methods which can be re-defined/overridden by the user. This enhances the functionality of QuSpin, allowing the user maximum flexibility in constructing basis objects. 

Below, we give a brief overview of the methods required to define `user_basis` objects.


`op(op_struct_ptr, op_str, site_ind, N, args)`
++++++
This function method contains user-defined action of operators :math:`O` on the integer states :math:`|s\rangle` which produces the matrix elements :math:`\mathrm{me}` via :math:`O|s\rangle = \mathrm{me}|s'\rangle`.

* `op_struct_ptr`: an cpp-pointer to an object which, after being cast into an array using `op_struct=carray(op_struct_ptr,1)[0]`, contains the attributes `op_struct.state` (which contains the quantum state in integer representation), and `op_struct.matrix_ele` (the value of the corresponding matrix element which defines
the action of the operator :math:`O`.).  

* `op_str`: holds the operator string (e.g. `+`, `-`, `z`, `n`, or any custom user-defined letter). Note that the underlying cpp-code uses integers to store the `op_str`, e.g. `+` corresponds to the integer `43`. It is these integers that are used in the body of `op()` to distinguish the different `op_str`'s. The integer, corresponding to any string `str` can be found in python using `ord(str)`.

* `N`: the total number of lattice sites.

* `args`: optional arguments passed into the CFunc `op`; must be a `np.ndarray` of dtype `basis_dtype`.  

The CFunc `op` returns an integer `err` which is used by QuSpin to throw different error messages **CHECK!!!**:

* `err=0`: the calculation was completed successfully.

* `err=-1`: no matching operator string was found.

**Notes** 

* this functionality will not support branching, i.e. no linear combination of multiple states in the basis, e.g. :math:`O|s\rangle = \mathrm{me}_1|s'_1\rangle + \mathrm{me}_2|s'_2\rangle + \dots`, is allowed.



`next_state(s, counter, N, args)` 
++++++
This functions method provides a user-defined particle conservation rule, which constructs the basis in lexicographical order **(DEFINE!)**. Given the initial state `s0`, `next_state()` generates all other states recursively. Hence, if `next_state()` is set to conserve particle number then the particle number sector is defined by the initial state `s0`. 

* `s`: quantum state in integer representation.

* `counter`: an integer which counts internally how many times the function has been called. The incrementation of `counter` will occur in the underlying cpp code, i.e. the user should not attempt to do this in the function body of `next_state()`. Can be used, e.g., to index an array passed in `args`.

* `args`: a `np.ndarray` of the same data type as the `user_basis`. Can be used to pass optional arguments, e.g. to pass a precomputed basis into QuSpin in order to reduce it to a given symmetry sector: ** see Example ??? below**.


**Two extra python functions required**: they are **not** called inside `next_state()`, but are required by QuSpin to run `next_state()`.

* get_s0_pcon(N,Np): given the total number of sites `N` and (the tuple of) particle sector `Np` this function computes the initial state, to be used by `next_state()` to construct the entire basis.

* get_Ns_pcon(N,Np): given the total number of sites `N` and (the tuple of) particle sector `Np` this function computes the Hilbert space dimension (i.e. the size of the basis) **with particle umber conservation only** (In other words, `get_Ns_pcon()` should be equal to the number of iterations in `next_state()` required to exhaust the states search. `get_Ns_pcon()` returns an integer required to allocate memory for the particle-conserving basis. Note that `get_Ns_pcon()` ignores any possible reduction due to lattice symmetries (see the maps below), i.e. `get_Ns_pcon()` may not correspond to the final integer `basis.Ns`.  


**Notes**

* there is no need to define `next_state()` if no particle number conservation use is intended. ** See example ??? below**.
* one can use this function, e.g., to implement sublattice particle number conservation, and similar features. 
* `next_state()`, together with the entire set of related functions and variables is passed to the `user_basis` constructor via the `pcon_dict` dictionary.
* `next_state()` is a numba.CFunc object, but `get_s0_pcon()` and `get_Ns_pcon()` are regular python functions.


`pre_check_state(s, N, args)`
++++++
This *optional* function method provides user-defined extra filtering of basis states. The function body contains a boolean operation which, when applied to the basis states one at a time, determines whether to keep a state in the basis or not. 

A simple example of what `pre_check_state()` can be useful for is this: suppose you want a `spinful_fermion_basis()` without doubly occupied sites. This can be achieved by ajusting the body of `pre_check_state()` to eliminate such states. QuSpin will then first generate the basis with doble occupancies using `next_state()`, and subsequntly get rid of the doubly-occupied states using `pre_check_state()`. Another example is shown in ** Example ??? below **.

* `s`: quantum state in integer representation.

* `N`: the total number of lattice sites.

* `args`: a `np.ndarray` of the same data type as the `user_basis`. Can be used to pass optional arguments.


`count_particles()`
++++++
This *optional* function method counts the total number of particles/magnetization in a given state.

**NEED TO SHOW EXAMPLES AND UPDATE THE CLASS DOC**




Symmetry transformations from bit operations
-------
Any discrete symmetry is uniquely defined by its action on the basis states. Since the basis is stored in the integer representation, the symmetry operations have to be defined to transform integers. In the `basis_1d` and `basis_general` classes this is done under the hood; the `user_basis` brings this functionality to the surface, and allows the user to modify it accordingly.

  

System-size independent symmetries
++++++
System-size independent symmetries contain as a parameter the system size :math:`N`. As a result, they apply to all system sizes. Examples of such symmetries are

parity in 1d 
````````

Parity is the reflection of a state w.r.t. the middle of the chain.

.. code-block:: python

	def parity(x,N,sign_ptr,args):
		""" works for all system sizes N, spin-1/2 only. """
		out = 0 
		s = N-1
		#
		out ^= (x&1)
		x >>= 1
		while(x):
			out <<= 1
			out ^= (x&1)
			x >>= 1
			s -= 1
		#
		out <<= s
		return out


translation in 1d 
````````

We consider translation by `shift=1` sites, but the code can easily be generalized to a larger-shift translation. 

.. code-block:: python

	def translation(x,N,sign_ptr,args):
		""" works for all system sizes N, spin-1/2 only. """
		shift = 1 # translate state by shift sites
		period = N # periodicity/cyclicity of translation
		xmax = (1<<N)-1 # largest integer allowed to appear in the basis
		#
		l = (shift+period)%period
		x1 = (x >> (period - l))
		x2 = ((x << l) & xmax)
		#
		return (x2 | x1)


Symmetries for fixed system sizes using precomputed masks
++++++++
The convenience to define symmetry maps which apply to all system sizes comes at a certain efficiency cost. This can be circumvented by defining system-size specific maps, using integer masks to perform the bit operations. These masks also depend on the data type of the integer storing the state. 

Luckily, there is a great tool to compute the symmetry maps, available at http://programming.sirrida.de/calcperm.php. All one needs to do is find the permutation of the lattice sites under the symmetry, and pass it to the tool to obain the symmetry map that acts on integers. Let us demonstrate how this works using two examples.

parity in 1d
````````
Consider a ladder of :math:`2\times 10` sites, labelled 0 through 19. The action of parity/reflection along the long ladder axis is easily defined on the lattice sites to be

.. math::
	[0,\ 1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9;\ 10,\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19] \mapsto [9,\ 8,\ 7,\ 6,\ 5,\ 4,\ 3,\ 2,\ 1,\ 0;\ 19,\ 18,\ 17,\ 16,\ 15,\ 14,\ 13,\ 12,\ 11,\ 10]

Passing the transformed integer sequence (right-hand side) to the online generator http://programming.sirrida.de/calcperm.php, it returns the symmetry map

.. code-block:: python
   
   def parity(x,N,sign_ptr,args):
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

translation in 1d
````````
Consider again a ladder of :math:`2\times 10` sites, labelled 0 through 19. The action of translation along the long ladder axis is easily defined on the lattice sites to be

.. math::
	[0,\ 1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9;\ 10,\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19] \mapsto [1,\ 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 0;\ 11,\ 12,\ 13,\ 14,\ 15,\ 16,\ 17,\ 18,\ 19,\ 10]

corresponds to the bit operation (again, fixed system size and data type):

.. code-block:: python
   
   def translation(x,N,sign_ptr,args):
       """ works for N=10 sites and 32 bit-integers spin-1/2 states only. """
       return ((x & 0x0007fdff) << 1) | ((x & 0x00080200) >> 9)

Symmetry `maps` dictionary
++++++
In the `user_basis`, the functions encoding the symmetry action are referred to as maps. Every map has as its first argument the integer (state) to be tansformed, followed by the number of sites. For fermionic systems, the symmetry action can also modify the fermion sign of a given state. Therefore, the last argument is a `sign_ptr`. 


Symmtries are passed to the `user_basis` constructor via a python dictionary, called `maps`. The keys are arbitrary strings which define a unique name for each map; the corresponding values are tuples of three entries: `(map function, periodicity, quantum number, args)`. The symmetry periodicity (or cyclicity, or multiplicity) is the smallest integer :math:`m_Q`, such that :math:`Q^{m_Q} = 1`. 

>>> maps = dict(T_block=(translation,10,0,T_args), P_block=(parity,2,0,P_args), )

**Notes**: 

* all map functions need to be cast as decorated numba cfuncs **(SEE below)**.
* even though some arguments of the map functions are not used in the function bodies, the user is required to define them (and no mores). This allows to keep the code general. The names of these arguments are arbitrary, but their data typs are **not**. 




Using `numba` to pass python functions to the `C++` `user_basis` constructor
-------
The function methods of `user_basis` discussed above, are passed to the `user_basis` constructor. Since the latter is written in `C++` for speed, we use  the [numba](https://numba.pydata.org/) package to decorate python functions which are automatically compiled to `C++` and then parsed to the `user_basis`. 


Data types
++++++++
Unlike python, cpp code requires the user to specify the data types of all variables (so called strong typing). For this purpose, numba supports various data types, e.g. `uint32`, or `int32`. They are typically imported from numba in the beginning of the python script.

Function decorators
++++++++
To indicate that the function we wrote in python should be compiled as a cpp code by numba, we use the `@cfunc(signature,locals=dict())` decorator. The arguments of the decorator are the function variable signature (which contains the data times of all function variables), and `locals` which is a dictionary containing the data types of all other variables defined and used privately inside the function body. 

In QuSpin, we provide the precompiled signatures `next_state_sig_32`, `op_sig_32`, `map_sig_32`, `next_state_sig_64`, `op_sig_64`, `map_sig_64`. The name of the signature refers to the function type it is designed for, and the integer in the end specifies the data type the `user_basis` will be constructed with. These signaturescan be imported from the `user_basis`. 

As an example, consider the `translation()` python function defined above. To make this a `numba.CFunc` object, it suffices to place the decorator:

.. code-block:: python

	from quspin.basis.user import map_sig_32 # user basis data types
	from numba import cfunc
	from numba import uint32,int32 # numba data types
	#
	@cfunc(map_sig_32,
		locals=dict(shift=uint32,xmax=uint32,x1=uint32,x2=uint32,period=int32,l=int32,) )
	def translation(x,N,sign_ptr,args):
		""" works for all system sizes N. """
		shift = 1 # translate state by shift sites
		period = N # periodicity/cyclicity of translation
		xmax = (1<<N-1)
		#
		l = (shift+period)%period
		x1 = (x >> (period - l))
		x2 = ((x << l) & xmax)
		#
		return (x2 | x1)

We use the signature `map_sig_32` because it is designed to decorate symmetry map functions. Moreover, the local (private) variable data types are defined via `locals=dict(shift=uint32,xmax=uint32,x1=uint32,x2=uint32,period=int32,l=int32,)`. These variables appear in the function body.

**Notes**

* because QuSpin provides predefined CFunc signatures, every CFunc (see function methods above) has a predefined, **fixed** number of arguments. Moreover, the data types of the arguments is also fixed. Even if some arguments are not used in the CFunc body, they have to appear in the function definition.

* if you mess up the data types, most likely you will receive a numba error. In such cases, we suggest that you remove the CFunc decorator and debug your function in python as yous would normally do. Once you ares confident that the function does it job, put back the decorator and pass it to the `user_basis` constructor. 



Example Scripts
--------
Below, we provide examples which demonstrate how to use the `user_basis` class. 


Scripts to construct spin, fermion, and boson bases 
++++++++
The following three examples demonstrate how the `user_basis` recovers the functionality of the `basis_general` classes:

* :ref:`user-basis_example0-label`, :download:`download script <../doc_examples/user_basis_trivial-spin.py>` 
* :ref:`user-basis_example1-label`, :download:`download script <../doc_examples/user_basis_trivial-spinless_fermion.py>`
* :ref:`user-basis_example2-label`, :download:`download script <../doc_examples/user_basis_trivial-boson.py>`


Scripts to demonstrate the additional functionality introduced by the `user_basis`
++++++++
*
*
*


