:math:`\mathrm{\color{red} {Highlights}}`: OpenMP support now here!
======

Check out :ref:`parallelization-label` and the example script :ref:`example12-label`!

Complete list of the most recent features 
=============================

Added in v. 0.3.1
-----------------

Improved Functionality
+++++

* support for python 3.7.
* :math:`\mathrm{\color{red} {discontinued\ support}}` for python 3.5 on all platforms and python 2.7 on windows. These versions will remain available up to version 0.3.0. 
* matplotlib is no longer a required package to install quspin. It is still required to run the examples, though.
* parallelization: New parallel features added or improved + OpenMP support for osx. Requires a different build of QuSpin (see also :ref:`parallelization-label`).
* new OpenMP features in operators module (see :ref:`parallelization-label` and example script :ref:`example12-label`).
* improved OpenMP features in the `*_general_basis` classes.
* new example scripts: (i) use of some new `*_basis_general` methods, (ii) use of OpenMP and QuSpin's parallel features.
* faster implementation of spin-1/2 and hard-core bosons in the general basis classes. 
* more memory efficient versions of matrix-vector/matrix products implemented for both `hamiltonian` and `quantum_operator` classes. Allows using OpenMP in the `hamiltonian.evolve()` function method.
* refactored code for `*_general_basis` classes.
* large integer support for `*_general_basis` classes allows to build lattices with more than 64 sites. 

New Attributes, Functions, Methods and Classes
+++++

* new argument `make_basis` for `*_basis_general` classes allows to use some of the basis functionality without constructing the basis. 
* new `*_basis_general` class methods: `Op_bra_ket()`, `representative()`, `normalization()`, `inplace_Op()`.
* support for Quantum Computing definition of `"+"`, `"-"` Pauli matrices: see `pauli` argument of the `spin_basis_*` classes.  
* adding argument `p_con` to `*_basis_general.get_vec()` and `*_basis_general.get_proj()` functions. 
* adding functions `basis.int_to_state()` and `basis.state_to_int()` to convert between spin and integer representation of the states.
* new `basis.states` attribute to show the list of basis states in their integer representation.
* new methods of the `*_basis_general` classes for bitwise operations on basis states stored in integer representation. 
* both `hamiltonian` and `quantum_operator` classes support a new `out` argument for `dot` and `rdot` which allows the user to specify an output array for the result.
* both `hamiltonian` and `quantum_operator` classes support a new `overwrite_out` argument which allows the user to toggle between overwriting the data within `out` or adding the result to `out` inplace without allocating extra data.

