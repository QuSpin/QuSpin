

:red:`QuSpin 0.3.4` here (updated on 2020.??.??)
================================================


:green:`Highlights:` Lanczos module now here; Constrained Hilbert spaces support here; OpenMP support here!
===========================================================================================================

Check out :ref:`parallelization-label` and the example script :ref:`example12-label`.

For a tutorial in QuSpin's `user_basis` which allows the user to define custom bases with constraints, check out: :ref:`user_basis-label`.


Complete list of the most recent features 
=========================================


Added in v. 0.3.4 (2020.??.??)
------------------------------

Improved Functionality
++++++++++++++++++++++

* revised `user_basis` tutorial for spinless fermions: introduced function `_count_particles_32()`.
* added optional arguments `svd_solver`, `svd_kwargs` to `basis.ent_entropy()`.
* fixed bugs.


New Attributes, Functions, Methods and Classes
++++++++++++++++++++++++++++++++++++++++++++++
* new `tools.Lanczos` module for Lanczos type calculations. 
* new function method `Op_shift_sector` of the `*basis_general_` classes allows to apply operators, which do not preserve the symmetry sector, to quantum states in the reduced basis. Useful for computing correlation functions. See Example 19.
* new required package for QuSpin: `numexpr` (ADD TO MANUAL INSTALL AND README).


Added in v. 0.3.3 (2019.10.15)
------------------------------

Improved Functionality
++++++++++++++++++++++

* introducing improvements to Example 11 to perform Monte Carlo sampling in the symmetry-reduced Hilbert space.
* new examples:
	* Example 13 to showcase `double_occupancy` option of the `spinful_fermion_basis_*`.
	* Examples 14-16 demonstrate the usage of `user_basis`.
	* Example 17 shows how to use QuSpin for Lindblad dynamics and demonstrates the use of the omp-parallelzied `matvec` function for speedup.
	* Example 18 shows how to construct Hamiltinians on a hexagonal lattice. 
* improved functionality of the `tools.evolution.evolve()` function.
* fixed import issue with scipy's `comb` function.
* fixed a number of small bugs. 

New Attributes, Functions, Methods and Classes
++++++++++++++++++++++++++++++++++++++++++++++

* adding `*_basis_general.get_amp()` function method which effectively provides a partial `get_vec()` function but does not require the basis to be constructed ahead of time.
* adding optional argument `double_occupancy` to the `spinful_fermion_basis_*` classes to control whether doubly occupied sites should be part of the basis or not. 
* adding the `user_basis` class which enables the user to build in Hilbert-space constraints, and exposes the inner workings of QuSpin's core function to give the user almost complete control (see :ref:`user_basis-label`).
* adding `tools.misc.matvec()` and `tools.misc.get_matvec()` functions with omp-parallelized implementation which outperforms scipy and numpy in computing matrix-vector peroducts.
* adding optional arguments to the `dot()` and `rdot()` functions of the operators module.



Added in v. 0.3.2 (2019.03.11)
------------------------------

Improved Functionality
++++++++++++++++++++++

* improved performance for matrix vector product in _oputils and expm_multiply_parallel. Leads to significant speedup in the `hamiltonian` and `quantum_operator` classes (e.g. in the `hamiltonian.evolve()` function) and the `tools.evolution.expm_multiply_parallel()` function.



Added in v. 0.3.1 (2019.03.08)
------------------------------


Improved Functionality
++++++++++++++++++++++

* support for python 3.7.
* :red:`discontinued support` for python 3.5 on all platforms and python 2.7 on windows. QuSpin for these versions will remain available to download up to and including QuSpin 0.3.0, but they are no longer being maintained. 
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
++++++++++++++++++++++++++++++++++++++++++++++

* new argument `make_basis` for `*_basis_general` classes allows to use some of the basis functionality without constructing the basis. 
* new `*_basis_general` class methods: `Op_bra_ket()`, `representative()`, `normalization()`, `inplace_Op()`.
* support for Quantum Computing definition of `"+"`, `"-"` Pauli matrices: see `pauli` argument of the `spin_basis_*` classes.  
* adding argument `p_con` to `*_basis_general.get_vec()` and `*_basis_general.get_proj()` functions. 
* adding functions `basis.int_to_state()` and `basis.state_to_int()` to convert between spin and integer representation of the states.
* new `basis.states` attribute to show the list of basis states in their integer representation.
* new methods of the `*_basis_general` classes for bitwise operations on basis states stored in integer representation. 
* both `hamiltonian` and `quantum_operator` classes support a new `out` argument for `dot` and `rdot` which allows the user to specify an output array for the result.
* both `hamiltonian` and `quantum_operator` classes support a new `overwrite_out` argument which allows the user to toggle between overwriting the data within `out` or adding the result to `out` inplace without allocating extra data.

