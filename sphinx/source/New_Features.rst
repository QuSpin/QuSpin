List of new features 
============

Added in v. 0.3.1
-----------------

* support for python 3.7.
* matplotlib is no longer a required package to install quspin.
* parallelization: OpenMP support for osx and windows. Requires a different build (see Installation).
* new OpenMP features in `hamiltonian.evolve()` and `tools.evolution.evolve`().
* new example script: use of OpenMP and some new `*_basis_general` methods.
|
* new argument `make_basis` for `*_basis_general` classes allows to use some of the basis functionality without constructing the basis. 
* new `*_basis_general` class methods: `Op_bra_ket()`, `representative()`, `normalization()`.
* support for Quantum Computing defition of `"+"`, `"-"` Pauli matrices: see `pauli` argument of the `spin_basis_*` classes.  
* adding argument `p_con` to `*_basis_general.get_vec()` and `*_basis_general.get_proj()` functions. 
* adding functions `basis.int_to_state()` and `basis.state_to_int()` to convert between spin and integer representation of the states.
* new `basis.states` attribute to show the list of basis states in their integer representation.

