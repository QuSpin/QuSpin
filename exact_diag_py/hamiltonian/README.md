# Tensoring basis.

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
