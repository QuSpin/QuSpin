# exact_diag_py
public repository for a simple python library used for ED calculations of quantum many particle systems.

# Basic usage:

many-spin operators are represented by string of letters representing the type of operator:

|      opstr       |      indx      |        operator string      |
|:----------------:|:--------------:|:---------------------------:|
|'a<sub>1</sub>...a<sub>n</sub>'|[J,i<sub>1</sub>,...,i<sub>n</sub>]|J S<sub>i<sub>1</sub></sub><sup>a<sub>1</sub></sup>...S<sub>i<sub>n</sub></sub><sup>a<sub>n</sub></sup>|

where a<sub>i</sub> can be x, y, z, +, or -. this gives the full range of possible spin operators that can be constructed. The hamiltonian is split into two parts, static and dynamic. static parts are all added up into a single sparse matrix, while the dynamic parts are separated from each other and grouped with the time dependent function.

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],...]
```

the code calculates each operator string as a scipy.sparse csr_matrix acting on the manybody basis.

example, transverse field ising model with time dependent field for L=10 chain:

```python
# python script
from exact_diag_py.spins import hamiltonian

L=10
v=0.01

def drive(t,v):
  return v*t
  
drive_args=[v]

ising_indx=[[-1.0,i,(i+1)%L] for i in xrange(L)]
field_indx=[[-1.0,i] for i in xrange(L)]
static_list=[['zz',ising_indx],['x',field_indx]]
dynamic_list=[['x',field_indx,drive,drive_args]]

H=hamiltonian(static_list,dynamic_list,L)
```

# Using symmetries:
Adding symmetries is easy, either you can just add extra keyword arguements (by default all are set to None):

magnetization sector:
Nup=0,...,L 

parity sector:
pblock=+/-1

spin inversion sector:
zblock=+/-1

(parity)*(spin inversion) sector:
pzblock=+/-1

momentum sector:
kblock=0,..,L-1

```python
H=hamiltonian(static_list,dynamic_list,L,Nup=Nup,pblock=pblock,...)
```

One can also use the basis1d class to construct the basis object with a particular set of symmetries, then pass that into the the constructor of the hamiltonian:

```python
from exact_diag_py.basis import basis1d

basis=basis1d(L,Nup=Nup,pblock=pblock,...)
H=hamiltonian(static_list,dynamic_list,L,basis=basis)
```
NOTE: Using symmetry reduction on hamiltonians which do not have said symmetry will cause the code to behave incorrectly. later we will impliment checks to see which symmetries are allowed based on the user input.

# Numpy dtype:
The user can specify the numpy data type to store the matrix elements with. It supports float32, float64, complex64, and complex128. The default type is complex128.

# Using hamiltonian:
The hamiltonian class has built in methods which are useful for doing ED calculations:

* Full diagonalization:
```python
eigenvalues,eigenvectors=H.eigh(time=time)
```
* sparse diagonalization, which uses ARPACK (see scipy.sparse.linalg docs):
```python
eigenvalues,eigenvectors=H.eigsh(time=time)
```
For more information, see the toolbox section for more functionalities. 
