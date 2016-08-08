# **Basis objects**

The basis objects provide a way of constructing all the necessary information needed to construct a sparse matrix from a list of operators. There are two different kinds of basis classes at this point in the development of the code, All basis objects are derived from the same base object class and have mostly the same functionality but there are two subtypes of this class. The first are basis objects which provide the bulk operations required to create a sparse matrix out of an operator string. Right now we have included only two of these types of classes. One type of class preforms calculations of hamiltonian matrix elements while the other basis types wrap the first type together in tensor style basis types. 


 spin_basis_1d: 
--------------------
This basis class provides everything necessary to create a hamiltonian of a spin system in 1d. The available operators one can use are the the standard spin operators: ```x,y,z,+,-``` which either represent the pauli operators or spin 1/2 operators. The ```+,-``` operators are always constructed as ```x +/- i y```.


It also allows the user to create the hamiltonian in block reduced by symmetries like:

* Magnetization symmetries: 
 *  ```Nup=0,1,...,L # pick single magnetization sector```
 * ```Nup = [0,1,...] # pick list of magnetization sectors```
* Parity symmetry: ```pblock = +/- 1```
* Spin Inversion symmetry: ```zblock = +/- 1```
* (Spin Inversion)*(Parity) symmetry: ```pzblock = +/- 1 ```
* Spin inversion on sublattice A (even sites): ```zAblock = +/- 1```
* Spin inversion on sublattice B (odd sites): ```zAblock = +/- 1```
* Translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

Harmonic Oscillator basis:
--------------------------
This basis implements a single harmonic oscillator mode. The available operators are ```+,-,n```.

Tensor basis:
-------------

In version 0.1.0 we have created new classes which allow basis to be tensored together. 

* tensor_basis class: two basis objects b1 and b2 the tensor_basis class will combine them together to create a new basis objects which can be used to create the tensored hamiltonian of both basis:

	```python
	basis1 = spin_basis_1d(L,Nup=L/2)
	basis2 = spin_basis_1d(L,Nup=L/2)
	t_basis = tensor_basis(basis1,basis2)
	```

	The syntax for the operator strings are as follows. The operator strings are separated by a '|' while the index array has no splitting character.

	```python
	# tensoring two z spin operators at sites 1 for basis1 and 5 for basis2
	opstr = "z|z" 
	indx = [1,5] 
	```

	if there are no operator strings on one side of the '|' then an identity operator is assumed.

* photon_basis class: This class allows the user to define a basis which couples to a single photon mode. There are two types of basis objects that one can create, a particle (magnetization + photon or particle + photon) conserving basis or a non-conserving basis.  In the former case one can specify the total number of quanta using the the Ntot keyword arguement:

	```python
	p_basis = photon_basis(basis_class,*basis_args,Ntot=...,**symmetry_blocks)
	```

	while for for non-conserving basis you must specify the number of photon states with Nph:

	```python
	p_basis = photon_basis(basis_class,*basis_args,Nph=...,**symmetry_blocks)
	```

	For this basis class you can't pass not a basis object, but the constructor for you basis object. The operators for the photon sector are '+','-','n', and 'I'.

# Checks on operator strings
new in version 0.2.0 we have included a new functionality classes which check various properties of a given static and dynamic operator lists. They include the following:

* Checks if complete list of opertors obey the given symmetry of that basis. The check can be turned off with the flag ```check_symm=False ``` in the [hamiltonian](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/hamiltonian) class. 
* Checks of the given set of operators are hermitian. The check can be turned out with the flag ```check_herm=False``` in the [hamiltonian](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/hamiltonian) class. 
* Checks of the given set of operators obey particle conservation (for spin systems this means magnetization sectors do not mix). The check can be turned out with the flag ```check_pcon=False``` in the [hamiltonian](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/hamiltonian) class. 

# **methods of basis class:**

* Op(indx,opstr,J,dtype)
* get_vec(v0)
* get_proj(dtype)


