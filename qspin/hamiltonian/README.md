# ** Creating hamiltonians:** 
Many-body operators are represented by string of letters representing the type of operators. For example, in a spin system we can represent multi-spin operators as:

|      opstr       |      indx      |        operator string      |
|:----------------:|:--------------:|:---------------------------:|
|'o<sub>1</sub>...o<sub>n</sub>'|[J,i<sub>1</sub>,...,i<sub>n</sub>]|J S<sub>i<sub>1</sub></sub><sup>o<sub>1</sub></sup>...S<sub>i<sub>n</sub></sub><sup>o<sub>n</sub></sup>|

where o<sub>i</sub> can be x, y, z, +, or -. The object indx specifies the coupling as well as the sites for which the operator acts. This gives the full range of possible spin operators that can be constructed. For different systems there are different types of operators. To see the availible operators for a given type of system checkout out the [basis](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/basis) classes. 

The hamiltonian is split into two parts, static and dynamic. Static parts are all added up into a single matrix, while the dynamic parts are separated from each other and grouped with the time dependent function. When needed the time dependence is evaluated on the fly with the time always passed into a method by an arguement time. All this data is input in the following format:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],...]
```
**New in version 0.0.5b:** We now let the user initialize with Scipy sparse matrices or dense Numpy arrays. To do this just replace the opstr and indx tuple with your matrix. you can even mix them together in the same list:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```
NOTE: if no operator strings or matrices are present one must specify the shape of the matrices being used as well as the dtype using the keyword arguement shape: ```H=hmailtonian([],[],...,shape=shape,...)```.

**example:** transverse field ising model with time dependent field for L=10 chain:

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

## Using symmetries:
Adding symmetries is easy, either you can just add extra keyword arguements to the initialization of your hamiltonian. By default the hamiltonian will pick the spin-1/2 operators as well as 1d-symmetries.
The symmetries for a spin chain in 1d are:

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

used like:
```python
H=hamiltonian(static_list,dynamic_list,L,Nup=Nup,pblock=pblock,...)
```
By doing this option the hamiltonian creates a [spin_basis_1d](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/basis) for the given symmetries and then uses that object to construct the matrix elements. If you don't want to use spin chains there are also other basis type objects which can be constructed for a given set of symmetry blocks. These basis objects can be passed to hamiltonian which will then use that basis to construct the matrix elements for symmetry sector of the basis object. One basis object is called spin_basis_1d which is used to create the basis object for a 1d spin chain:

```python
from exact_diag_py.basis import spin_basis_1d

basis=spin_basis_1d(L,Nup=Nup,pblock=pblock,...)
H1=hamiltonian(static_list_1,dynamic_list_1,basis=basis)
H2=hamiltonian(static_list_2,dynamic_list_2,basis=basis)
```
This is typically more efficient because you can use a basis object for multiple hamiltonians without constructing the basis every time a hamiltonian is created. More information about the basis objects can be found [here](https://github.com/weinbe58/exact_diag_py/tree/master/exact_diag_py/basis).

**NOTE:** for beta versions spin_basis_1d is named as basis1d.

## Numpy dtype:
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements with. It supports float32, float64, complex64, and complex128. The default type is complex128. To specify the dtype use the dtype keyword arguement:

```python
H=hamiltonian(...,dtype=np.float32,...)
```

**New in verions 0.1.0:** We now let the user use quadruple precision dtypes float128 and complex256, but note that not all Scipy and Numpy functions support all dtypes.

# ** Using hamiltonian:**
The hamiltonian objects currectly support certain arithmetic operations with other hamiltonians as well as scipy sparse matrices and numpy dense arrays and scalars:

* between other hamiltonians we have: ```+,-,+=,-=``` 
* between numpy and sparse arrays we have: ```*,+,-,*=,+=.-=``` (versions >= v0.0.5b)
* between scalars: ```*,*=``` (versions >= v0.0.5b)
* negative operator '-H'


We've included some basic functionality into the hamiltonian class useful for quantum calculations:

* sparse matrix vector product / dense matrix:

  usage:
    ```python
    v = H.dot(u,time=0)
    ```
  where time is the time to evaluate the Hamiltonian at for the product, by default time=0. rdot is another function similar to dot just from the right. 
  
* matrix elements:

  usage:
    ```python
    Huv = H.matrix_ele(u,v,time=0)
    ```
  which evaluates < u|H(time)|v > if u and v are vectors but (versions >= v0.0.2b) can also handle u and v as dense matrices. NOTE: the inputs should not be hermitian tranposed, the function will do that for you.

* commutator:
usage:
```python
U = H.comm(O,time=0,check=True,a=1)
```
evaluates a[H(time), O].

* anti-commutator:
usage:
```python
U = H.anti_comm(O,time=0,check=True,a=1)
```
evaluates a{H(time), O}.

* project to new basis:
usage:
```python
H_new = H.project_to(V)
```
returns a new hamiltonian object which is: V<sup>+</sup> H V.
  
* matrix exponential multiplication (versions >= v0.1.0):
  usage:
    ```python
    v = H.expm_multiply(u,a=-1j,time=0,**linspace_args)
    ```
  which evaluates |v > = exp(a H(time))|u > using only the dot product of H on |u >. The extra arguments will allow one to evaluate it at different time points see scipy docs for [expm_multiply](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply) for more information.

* matrix exponential (versions >= v0.2.0):
  usage:
    ```python
    v = H.expm(a=-1j,time=0)
    ```
  which evaluates M = exp(a H(time)) using the pade approximation, see scipy docs of [expm](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.linalg.expm.html) for more information.

* Schrodinger dynamics:

  The hamiltonian class has 2 private functions which can be passed into Scipy's ode solvers in order to numerically solve the Schrodinger equation in both real and imaginary time:
    1. __SO(t,v) which proforms: -iH(t)|v>
    2. __ISO(t,v) which proforms: -H(t)|v> 
  
  The interface with complex_ode is as easy as:
  
    ```python
    solver = complex_ode(H._hamiltonian__SO)
    ```
  
  From here all one has to do is use the solver object as specified in the scipy [documentation](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.integrate.complex_ode.html#scipy.integrate.complex_ode). Note that if the hamiltonian is not complex and you are using __ISO, the equations are real valued and so it is more effeicent to use ode instead of complex_ode.

This functionality is wrapped in a method called evolve (version >= 0.2.0):

```python
vf = H.evolve(v0,t0,times,solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args)
```
* v0: initial state vector.
* t0: initial time.
* times: a time or generator of times to evolve to.
* solver_name: used to pick which Scipy ode solver to use.
* verbose: prints out when the solver has evolved to each time in times
* iterate: returns vf which is an iterator over the evolution vectors without storing the solution for every time in times.
* imag_time: toggles whether to evolve with __SO or __ISO.
* solver_args: the optional arguements which are passed into the solver. 
  
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

There are also some methods which are useful if you need other functionality from other packages:

* return copy of hamiltonian as csr matrix: 
  ```python
  H_csr = H.tocsr(time=time)
  ```
  
* return copy of hamiltonian as dense matrix: 
  ```python
  H_dense = H.todense(time=time,order=None,out=None)
  ```

* return copy of hamiltonian: 
  ```python
  H_new = H.copy()
  ```

* cast hamiltonian to different dtype: 
  ```python
  H_new = H.astype(dtype,copy=True)
  ```





