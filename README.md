# exact_diag_py
exact_diag_py is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state of the art exact diagonalization calculations on 1 dimensional spin 1/2 chains with lengths up to 31 sites. The interface allows the user to define any spin 1/2 Hamiltonian which can be constructed from spin operators; while also allowing the user flexibility of accessing all possible symmetries in 1d. There is also a way of spcifying the time dependence of operators in the hamiltonian as well, which can be easily interfaced with Scipy [ode solvers](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.integrate.ode.html) to solve the time dependent Schrodinger equation numerically for these systems. All the hamiltonian data is stored using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library which allows the user to access the powerful sparse linear algebra tools. 

The package requires scipy v0.14.0 or later, a compatible version of numpy, and the proper fortran compilers.

to install download source code either from the latest [release](https://github.com/weinbe58/exact_diag_py/releases) section or cloning the git repository. In the top directory of the source code you can execute the following commands:

unix:
```
python setup.py install
```

windows:
```
setup.py install
```

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

The wrapper accesses custom fortran libraries which calculates the action of each operator string on the states of the manybody S<sup>z</sup> basis. This data is then stored as a Scipy sparse csr_matrix.

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

One can also specify which types of operators to use with the option pauli: 
```python 
H=hamiltonian(...,pauli=True,...) 
``` 
If pauli is set to True then the hamiltonian will be created assuming you have Pauli matrices while for pauli set to False you use spin 1/2 matrices. By default pauli is set to True.

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
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements with. It supports float32, float64, complex64, and complex128. The default type is complex128. To specify the dtype use the dtype keyword arguement:

```python
H=hamiltonian(...,dtype=np.float32,...)
```

# Using hamiltonian:
We've included some basic functionality into the hamiltonian class.
* addition/subtraction:

  ```python
  +,-,+=,-=
  ```
* equality/inequality:

  ```python 
  == , !=
  ```
We've also included some functions useful for quantum calculations:

* sparse matrix vector product / dense matrix:

  usage:
    ```python
    v = H.dot(u,time=time)
    ```
  where time is the time to evaluate the Hamiltonian at for the product, by default time=0.
  
* matrix elements:

  usage:
    ```python
    Huv = H.me(u,v,time=time)
    ```
  which evaluates < u|H(time)|v > if u and v are vectors but can also handle u and v as dense matrices.
  
There are also some methods which are useful if you need other functionality from other packages:

* return copy of hamiltonian as csr matrix: 
  ```python
  H_csr = H.tocsr(time=time)
  ```
  
* return copy of hamiltonian as dense matrix: 
  ```python
  H_dense = H.todense(time=time)
  ```

The hamiltonian class also has built in methods which are useful for doing ED calculations:

* Full diagonalization:

  usage:
    ```python
    eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)
    eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)
    ```
  where **eigh_args are optional arguements which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigh](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh) and [eigvalsh](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.eigvalsh.html#scipy.linalg.eigvalsh). 
  
  NOTE: overwrite_a=True always for memory conservation

* Sparse diagonalization, which uses ARPACK:

  usage:
    ```python
    eigenvalues,eigenvectors=H.eigsh(time=time,**eigsh_args)
    ```
  where **eigsh_args are optional arguements which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigsh](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html)


* Schrodinger dynamics:

  The hamiltonian class has 2 private functions which can be passed into Scipy's ode solvers in order to numerically solve the Schrodinger equation in both real and imaginary time:
    1. __SO(t,v) which proforms: -iH(t)|v>
    2. __ISO(t,v) which proforms: -H(t)|v> 
  
  The interface with complex_ode is as easy as:
  
    ```python
    solver = complex_ode(H._hamiltonian__SO)
    ```
  
  From here all one has to do is use the solver object as specified in the scipy [documentation](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.integrate.complex_ode.html#scipy.integrate.complex_ode). Note that if the hamiltonian is not complex and you are using __ISO, the equations are real valued and so it is more effeicent to use ode instead of complex_ode.

