Complete list of new features 
=============================

Added in v. 0.3.1
-----------------

* support for python 3.7.
* matplotlib is no longer a required package to install quspin. It is still required to run the examples, though.
* parallelization: New parallel features added or improved + OpenMP support for osx. Requires a different build of QuSpin (see Installation).
* new OpenMP features in operators module (see section on parallel computing support).
* new example script: use of some new `*_basis_general` methods.
* faster implementation of spin-1/2 and hard-core bosons in the general basis classes. 
* both `hamiltonian` and `quantum_operator` classes support a new `out` argument for `dot` and `rdot` which provide the user to specify an output array for result.
* both `hamiltonian` and `quantum_operator` classes support a new `overwrite_out` argument which allows the user to toggle between overwriting the data within `out` or adding the result to `out` inplace without allocating extra data.
* more memory efficient versions of matrix-vector/matrix products implemented for both `hamiltonian` and `quantum_operator` classes.
|
* new argument `make_basis` for `*_basis_general` classes allows to use some of the basis functionality without constructing the basis. 
* new `*_basis_general` class methods: `Op_bra_ket()`, `representative()`, `normalization()`.
* support for Quantum Computing defition of `"+"`, `"-"` Pauli matrices: see `pauli` argument of the `spin_basis_*` classes.  
* adding argument `p_con` to `*_basis_general.get_vec()` and `*_basis_general.get_proj()` functions. 
* adding functions `basis.int_to_state()` and `basis.state_to_int()` to convert between spin and integer representation of the states.
* new `basis.states` attribute to show the list of basis states in their integer representation.

