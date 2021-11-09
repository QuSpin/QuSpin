

:orange:`QuSpin 0.3.7` (released on 2021.12.01)
================================================


:magenta:`Highlights:` Lanczos module now here; Constrained Hilbert spaces support available; OpenMP support available!
========================================================================================================================

Check out :ref:`parallelization-label` and the example script :ref:`example12-label`.

For a tutorial in QuSpin's `user_basis` which allows the user to define custom bases with constraints, check out: :ref:`user_basis-label`.





Complete list of the most recent features 
=========================================

Added in v. 0.3.7 (2021.12.01)
------------------------------

Improved Functionality
++++++++++++++++++++++
* fixed bug with automatic symmetry checks for `*_basis_1d`.
* fixed bug with data types in `quantum_operator` for systems of `N>31` sites.
* fixed bug with using `Nup` argument of `*_spin_basis_*`.
* fixed bug with `return_rdm` optional argument of `tensor_basis`.
* added warning to notify users that hermiticity/symmetry checks and not exhaustive.



Added in v. 0.3.6 (2021.05.01)
------------------------------

Improved Functionality
++++++++++++++++++++++
* adding python 3.9 support.
* fixed a bug with *non-contiguous* subsystems in `basis.partial_trace()` and `basis.ent_entropy()` for the fermionic basis clases `*_fermion_basis_*`.
* fixed a bug with defining mixed particle sectors in `*_fermion_basis_general`.
* fixed some typos in the documentation.
* new `issues templates on Github <https://github.com/weinbe58/QuSpin/issues/new/choose>`_.


New Attributes, Functions, Methods, Classes, and Examples
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
* added optional argument `noncommuting_bits` to `user_basis` to specify the bits that represent a fermion degree of freedom.


Added in v. 0.3.5 (2021.02.15)
------------------------------

Improved Functionality
++++++++++++++++++++++
* adding python 3.8 support
* adding quspin application to test quspin on hpc clusters. 
* added 2d array/batch support to `tools.evolution.expm_multiply_parallel`.
* fixed errors caused by new releases of some of the dependencies. 
* fixed bugs with: 
	* particle-hole symmetry in `*_fermion_basis_general`;
	* memory leakage in `tools.lanczos`;
	* the `reduce_output` argument of `basis_general.Op_bra_ket()`. 



New Attributes, Functions, Methods, Classes, and Examples
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
* :red:`deprecated` function `tools.misc.csr_matvec`.
* added Majorana fermion operator strings to the `*_fermion_basis_general`.
* added `int_to_state` and `state_to_int` functions to `spinful_fermion_basis_*` classes.
* added properties `shape` and `ndim` to classes in the `operator` module. 
* new examples: 
	* Majorana fermion operators, cf. :ref:`example23-label`;
	* Gell-Mann operators for spin-1 systems, cf. :ref:`example24-label`;
	* Majorana SYK model, cf. :ref:`example25-label`.
	* Calculation of spectral functions using symmetries, cf. :ref:`example26-label`.
	* Tutorial on using QuSpin `basis` objects, cf. :ref:`example_00-label`.



Added in v. 0.3.4 (2020.04.17)
------------------------------

Improved Functionality
++++++++++++++++++++++

* :red:`discontinued` support for python 2.7. Installing QuSpin for py27 will by defult result in version 0.3.3.
* :red:`deprecated` function `basis.get_vec()`: use `basis.project_from()` instead.
* revised `user_basis` tutorial for spinless fermions and introduced function `_count_particles_32()`.
* added optional arguments `svd_solver`, `svd_kwargs` to `basis.ent_entropy()`; allows to use some scipy svd solvers, which are typically more stable. 
* `expm_multiply_parallel` now supports the option to give the operator an explicit dtype, see example :ref:`example22-label`.
* fixed bugs:
	* computing the entanglement entropy when using the `spinful_fermion_basis_general`.
	* constructing operators for higher-spin operators (S>1/2). 




New Attributes, Functions, Methods and Classes
++++++++++++++++++++++++++++++++++++++++++++++
* new `*_basis_general` functions -- `basis.project_from()` and its inverse `basis.project_to()` -- to transform states between a symmetry-reduced basis and the full basis.
* new `tools.Lanczos` module for Lanczos type calculations, see examples :ref:`example20-label`, :ref:`example21-label`.
* new function method `Op_shift_sector` of the `*basis_general_` classes allows to apply operators, which do not preserve the symmetry sector, to quantum states in the reduced basis. Useful for computing correlation functions. See example :ref:`example19-label`.
* new required support package for QuSpin: `numexpr`.



Added in v. 0.3.3 (2019.10.15)
------------------------------

Improved Functionality
++++++++++++++++++++++

* introducing improvements to Example :ref:`example11-label` to perform Monte Carlo sampling in the symmetry-reduced Hilbert space.
* new examples:
	* Example :ref:`example13-label` to showcase `double_occupancy` option of the `spinful_fermion_basis_*`.
	* Examples :ref:`example14-label`, :ref:`example15-label`, :ref:`example16-label` demonstrate the usage of `user_basis`.
	* Example :ref:`example17-label` shows how to use QuSpin for Lindblad dynamics and demonstrates the use of the omp-parallelzied `matvec` function for speedup.
	* Example :ref:`example18-label` shows how to construct Hamiltinians on a hexagonal lattice. 
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

