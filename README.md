# exact_diag_py
exact_diag_py is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state of the art exact diagonalization calculations on 1 dimensional spin 1/2 chains with lengths up to 31 sites. The interface allows the user to define any spin 1/2 Hamiltonian which can be constructed from spin operators; while also allowing the user flexibility of accessing all possible symmetries in 1d. There is also a way of specifying the time dependence of operators in the Hamiltonian as well, which can be used to solve the time dependent Schrodinger equation numerically for these systems. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access the powerful linear algebra tools. 

This latest version of this package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux and OS-X and windows 64-bit systems. In order to install this you need to get Anaconda package manager for python. Then all one has to do to install is run:

```
$ conda install -c weinbe58 exact_diag_py
```

This will install the latest version on to your computer. If this fails for whatever reason this means that either your OS is not supported or your compiler is too old. In this case one can also manually install the package:


1) Manual install: to install manually download source code either from the latest [release](https://github.com/weinbe58/exact_diag_py/releases) section or cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

unix:
```
python setup.py install
```

or windows command line:
```
setup.py install
```
NOTE: you must write permission to the standard places python is installed as well as have all the prerequisite python packages (numpy >= 1.10.0, scipy >= 0.14.0) installed first to do this type of install since the actual build itself relies on numpy. We recommend [Anaconda](https://www.continuum.io/downloads) to manage your python packages. 

# Basic usage:

many-spin operators are represented by string of letters representing the type of operator:

|      opstr       |      indx      |        operator string      |
|:----------------:|:--------------:|:---------------------------:|
|'a<sub>1</sub>...a<sub>n</sub>'|[J,i<sub>1</sub>,...,i<sub>n</sub>]|J S<sub>i<sub>1</sub></sub><sup>a<sub>1</sub></sup>...S<sub>i<sub>n</sub></sub><sup>a<sub>n</sub></sup>|

where a<sub>i</sub> can be x, y, z, +, or -. The object indx specifies the coupling as well as the sites for which the operator acts. This gives the full range of possible spin operators that can be constructed. The hamiltonian is split into two parts, static and dynamic. static parts are all added up into a single sparse matrix, while the dynamic parts are separated from each other and grouped with the time dependent function. When needed the time dependence is evaluated on the fly with the time always passed into a method by an arguement time. All this data is input in the following format:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],...]
```

The wrapper accesses custom cython libraries which calculates the action of each operator string on the states of the manybody S<sup>z</sup> basis. This data is then stored as a Scipy sparse csr_matrix. 

example, transverse field ising model with time dependent field for L=10 chain:

```python
# python script
from exact_diag_py.hamiltonian import hamiltonian

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

Here is an example of a 3 spin operator as well:

```python
op_indx=[[1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)]
op_indx_cc=[[-1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)]

static_list=[['-z+',op_indx],['+z-',op_indx_cc]]
```
Notice that I need to include both '-z+' and '+z-' operators to make sure our Hamiltonian is hermitian.

(versions >= 0.0.3b) One can also specify which types of operators to use with the option pauli: 
```python 
H=hamiltonian(...,pauli=True,...) 
``` 
If pauli is set to True then the hamiltonian will be created assuming you have Pauli matrices while for pauli set to False you use spin 1/2 matrices. By default pauli is set to True.

# Using symmetries:
Adding symmetries is easy, either you can just add extra keyword arguements:

magnetization sector:
Nup=0,...,L 

parity sector:
pblock=+/-1

spin inversion sector:
zblock=+/-1

(parity)*(spin inversion) sector:
pzblock=+/-1

momentum sector:
kblock=any integer

```python
H=hamiltonian(static_list,dynamic_list,L,Nup=Nup,pblock=pblock,...)
```

There are also basis type objects which can be constructed for a given set of symmetry blocks. These basis objects can be passed to hamiltonian which will then use that basis to construct the matrix elements for symmetry sector of the basis object. One basis object is called spin_basis_1d which is used to create the basis object for a 1d spin chain:

```python
from exact_diag_py.basis import spin_basis_1d

basis=spin_basis_1d(L,Nup=Nup,pblock=pblock,...)
H=hamiltonian(static_list,dynamic_list,L,basis=basis)
```
NOTE: for beta versions spin_basis_1d is named as basis1d.

More information about the basis objects can be found here(link).





# Numpy dtype:
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements with. It supports float32, float64, complex64, and complex128. The default type is complex128. To specify the dtype use the dtype keyword arguement:

```python
H=hamiltonian(...,dtype=np.float32,...)
```

* New in verions 0.1.0
We now let the user use quadruple precision dtypes float128 and complex256, but note that not all scipy and numpy functions support this dtypes.

* New in version 0.0.5b 
We now let the user initialize with scipy sparse matrices or dense numpy arrays. To do this just replace the opstr and indx tuple with your matrix. you can even mix them together in the same list:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```
The input matrices must have the same dimensions as the matrices created from the operator strings, if not an error is thrown to the user. 

NOTE: if no operator strings are present one must specify the shape of the matrices being used as well as the dtype using the keyword arguement shape: ```H=hmailtonian(static,dynamic,shape=shape,dtype=dtype)```. By default the dtype is set to numpy.complex128, double precision complex numbers.



