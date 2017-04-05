# **QuSpin**

This documentation is also available as a [jupyter notebook](./documentation.ipynb) which displays the LaTeX below code properly. 

QuSpin is a python library which wraps Scipy, Numpy, and custom Cython libraries together to do state-of-the art exact diagonalization calculations on one-dimensional interacting spin, fermion and boson chains/ladders. The interface allows the user to define any many-body (and single particle) Hamiltonian which can be constructed from single-particle operators. It also gives the user the flexibility of accessing many symmetries in 1d. Moreover, there is a convenient built-in way to specifying the time dependence of operators in the Hamiltonian, which is interfaced with user-friendly routines to solve the time dependent Schrödinger equation numerically. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse Hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access any powerful Python scientific computing tools.

For an ***indepth introduction*** to the package, check out our paper: https://scipost.org/10.21468/SciPostPhys.2.1.003.

***Examples*** with python script which show how to run QuSpin can be downloaded from the [arXiv ancillary repository](https://arxiv.org/src/1610.03042v4/anc) and from the [examples directory on github](./examples/).


# **Contents**
--------
* [Installation](#installation)
  * [automatic install](#automatic-install)
  * [manual install](#manual-install)
  * [updating the package](#updating-the-package)
* [Basic package usage](#basic-package-usage)
  * [constructing hamiltonians](#constructing-hamiltonians)
  * [using basis objects](#using-basis-objects)
  * [specifying symmetries](#using-symmetries)
* [List of package functions](#list-of-package-functions) 
  * [operator objects](#operator-objects)
    * [hamiltonian class](#hamiltonian-class)
    * [useful hamiltonian functions](#useful-hamiltonian-functions)
    * [matrix exponential](#exp_op-class)
    * [HamiltonianOperator class](#hamiltonianoperator-class)
    * [ops_dict class](#ops_dict-class)
  * [basis objects](#basis-objects)
    * [spins basis in 1d](#spins-basis-in-1d)
    * [boson basis in 1d](#boson-basis-in-1d)
    * [fermion basis in 1d](#fermion-basis-in-1d)
    * [harmonic oscillator basis](#harmonic-oscillator-basis)
    * [tensor basis](#tensor-basis)
    * [photon basis](#photon-basis)
    * [symmetry and hermiticity checks](#symmetry-and-hermiticity-checks)
    * [methods for basis objects](#methods-for-basis-objects)
  * [tools](#tools)
    * [measusrements](#measurements)
    * [periodically driven systems](#periodically-driven-systems)
    * [block diagonalisation](#block-diagonalisation)
     
# **Installation**

### **automatic install**

The latest version of the package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux, OS X and Windows 64-bit systems. The automatic installation of QuSpin requires the Anaconda package manager for Python. Once Anaconda has been installed, all one has to do to install QuSpin is run:
```
$ conda install -c weinbe58 quspin
```
This will install the latest version on your computer. Right now the package is in its beta stages and so it may not be available for installation on all platforms using this method. In such a case one can also manually install the package.

### **manual install**

To install QuSpin manually, download the source code either from the [master](https://github.com/weinbe58/QuSpin/archive/master.zip) branch, or by cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

Unix:
```
python setup.py install 
```
or Windows command line:
```
setup.py install
```
For the manual installation you must have all the prerequisite python packages: [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), and [joblib](https://pythonhosted.org/joblib/) installed. We recommend [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) to manage your python packages.

When installing the package manually, if you add the flag ```--record install.txt```, the location of all the installed files will be output to install.txt which stores information about all installed files. This can prove useful when updating the code. 

### **updating the package**

To update the package with Anaconda, all one has to do is run the installation command again.

To safely update a manually installed version of QuSpin, one must first manually delete the entire package from the python 'site-packages/' directory. In Unix, provided the flag ```--record install.txt``` has been used in the manual installation, the following command is sufficient to completely remove the installed files: ```cat install.txt | xargs rm -rf```. In Windows, it is easiest to just go to the folder and delete it from Windows Explorer. 


# **Basic package usage**

All the calculations done with QuSpin happen through [hamiltonians](#hamiltonian-objects). The hamiltonian is a type which uses Numpy and Scipy matrices to store the quantum Hamiltonian operator. Time-independent operators are summed together into a single static matrix, while each time-dependent operator is stored separatly along with the time-dependent coupling in front of it (operators sharing the same time-dependence are a also summen together into a single matrix). Whenever the user wants to perform an operation invonving a time-dependent operator, the time dependence is evaluated on the fly by specifying the argument ```time```. The user can initialize the hamiltonian types with Numpy arrays or Scipy matrices. Apart from this, we provide a user-friendly representation for constructing the Hamiltonian matrices for many-body operators. 

Many-body operators in QuSpin are defined by a string of letters representing the operator types, together with a list which holds the indices for the sites that each operator acts at on the lattice. For example, in a spin system we can represent any multi-spin operator as:

|      operator string        |      site-coupling list          |               spin operator                 |
|:-----------------:|:------------------:|:--------------------------------------:|
|"$\mu_1,\dots,\mu_n$"  |$[J,i_1,\dots,i_n]$ |$J\sigma_{i_1}^{\mu_1}\cdots\sigma_{i_n}^{\mu_n}$|

where $\mu_i$ can be $x$, $y$, $z$, $+$, or $-$. Then $\sigma_{i_1}^{\mu_1}$ is the Pauli spin operator acting on lattice site $i_n$. This gives the full range of possible spin-1/2 operators that can be constructed. By default, ```hamiltonian``` will use the sigma matrices ${\vec\sigma}$. For different systems, there are different types of operators. To see the available operators for a given type of system check out the [basis](basis-objects) classes. 

### **constructing hamiltonians**
The Hamiltonian is constructed as:
```python
H = hamiltonian(static_list,dynamic_list,**kwargs)
```
where ```static_list``` and ```dynamic_list``` are lists which have the following format for many-body operators:
```python
static_list =  [[opstr_1,[indx_11,...,indx_1m]                   ],...]
dynamic_list = [[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],...]
```
To use Numpy arrays or Scipy matrices the syntax is:
```python
static_list  = [[opstr_1,[indx_11,...,indx_1m]                   ], matrix_2                    ,...]
dynamic_list = [[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```
In the definition of the dynamic list, ```func``` is the python function which goes in front of the matrix or operator given in the same list. ```func_args``` is a list of the all additional function arguments such that the function itself is evaluated as: 
```python
f_val = func(t,*func_args)
```

#### keyword arguments (kwargs):
The ```**kwargs``` give extra information about the Hamiltonian. There is a variety of different features one can input here, and which one to choose depends on what Hamiltonian one would like to create. These arguments are used to specify symmetry blocks, give a shape, provide the floating point type to store the matrix elements with, disable automatic checks on the magnetisation, symemtries and the hermiticity of the hamiltonian, etc.

**providing a shape:**
To construct many-body operators, one must either specify the number of lattice sites with ```N=...``` or pass in a basis object as ```basis=...```, more about basis objects can be found [here](#basis-objects). One can also specify the shape using the ```shape=...``` keyword argument. For input lists which contain matrices only, the shape does not have to be specified. If empty lists are given, then either one of the previous options must be provided to the ```hamiltonian``` constructor.  

**Numpy dtype:**
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements in. It supports ```float32```, ```float64```, ```float128```, ```complex64```, ```complex128```, and ```complex256```. The default type is ```complex128```. To specify the ```dtype``` use the dtype keyword argument:
```python
H=hamiltonian(...,dtype=numpy.float32,...)
```
Note that not all platforms and all of Scipy and Numpy functions support ```dtype``` ```float128``` and ```complex256```.


**Example:** 
constructing the hamiltonian object of the transverse field Ising model with time-dependent field for a $10$-site chain:
```python
# python script
from quspin.operators import hamiltonian
import numpy as np

# set total number of lattice sites
L=10 
# define drive function of time t (called 'func' above)
def drive(t,v):
  return v*t
v=0.01 # set ramp speed
# define the function arguments (called 'func_args' above)  
drive_args=[v]
# define operator strings
Jnn_indx=[[-1.0,i,(i+1)%L] for i in range(L)] # nearest-neighbour interaction with periodic BC
field_indx=[[-1.0,i] for i in range(L)] # on-site external field
# define static and dynamic lists
static_list=[['zz',Jnn_indx],['x',field_indx]] # $H_{stat} = \sum_{j=1}^{L} -\sigma^z_{j}\sigma^z_{j+1} - \sigma^x_j $
dynamic_list=[['x',field_indx,drive,drive_args]] # $H_{dyn}(t) = vt\sum_{j=1}^{L}\sigma^x_j $
# create Hamiltonian object
H=hamiltonian(static_list,dynamic_list,N=L,dtype=np.float64)
```
Here is an example of a 3 spin operator as well:

```python
op_indx=[[1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)] # periodic BC
op_indx_cc=[[-1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)] # periodic BC

static_list=[['-z+',op_indx],['+z-',op_indx_cc]]
```
Notice that one needs to include both ```'-z+'``` and ```'+z-'``` operators to make sure the Hamiltonian is hermitian. If one forgets about this, there is still the automatic hermiticity check which raises an error, see the optional flag ```check_herm``` in the hamiltonian object class organisation [here](#hamiltonian-objects) which is set to ```True``` by default.


### **using basis objects**

Basis objects are another very useful type included in QuSpin: they provide all of the functionality which calculates the matrix elements from the operator string representation of many-body operators. On top of this, they have been programmed to calculate the matrix elements in different symmetry blocks of the many-body Hamiltonian with the help of optional flags. To use a basis object to construct the Hamiltonian, one simply uses the basis keyword argument:
```python
H = hamiltonian(static_list,dynamic_list,...,basis=basis,...)
```
More information about basis objects can be found in the [basis objects](#basis-objects) section.

We recommend using basis objects when multiple separate Hamiltonians, which share the same symmetries, need to be constructed. This provides the advantage of saving time when creating the corresponding symmetry objects.


### **using symmetries**
Adding symmetries is easy and versatile: one can either just add extra keyword arguments to the initialization of ```hamiltonian``` or, when one initializes a ```basis``` object one can build in the desired symmetries. By default, ```hamiltonian``` uses spin-1/2 Pauli operators and 1d-symmetries. Currently, the spin-chain symmetries implemented are for spins-1/2 operators in 1 dimension.

The available symmetries for a spin chain in 1d are:

* magnetization symmetries: 
 *  ```Nup=0,1,...,L # pick single magnetization sector```
 * ```Nup = [0,1,...] # pick list of magnetization sectors (e.g. all even ones)```
* parity symmetry: ```pblock = +/- 1```
* spin inversion symmetry: ```zblock = +/- 1```
* (spin inversion)*(parity) symmetry: ```pzblock = +/- 1 ```
* spin inversion on sublattice A (even sites, first site of chain is even): ```zAblock = +/- 1```
* spin inversion on sublattice B (odd sites): ```zAblock = +/- 1```
* translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

The symmetries can be used like:
```python
H = hamiltonian(static_list,dynamic_list,L,Nup=Nup,pblock=pblock,...)
```
If the user passes the symmetries into the ```hamiltonian``` constructor, the constructor first creates a [spin_basis_1d](spin\_basis\_1d) object for the given symmetries, and then uses that object to construct the matrix elements. Because of this, if one needs to construct multiple hamiltonian objects in the same symmetry block, it is more efficient to first construct the ```basis``` object and then use it to construct all different Hamiltonians:
```python

basis = spin_basis_1d(L,Nup=Nup,pblock=pblock,...)
H1 = hamiltonian(static1_list,dynamic1_list,basis=basis)
H2 = hamiltonian(static2_list,dynamic2_list,basis=basis)
...
```

**NOTE:** for beta versions spin_basis_1d is named as basis1d.

# **List of package functions**

## **operator objects**

### **hamiltonian class**
```python
H = hamiltonian(static_list,dynamic_list,N=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs)
```

The ```hamiltonian``` class wraps most of the functionalty of the package. This object allows the user to construct a many-body Hamiltonian, solve the Schrödinger equation, do full/Lanczos diagonalization and many other things. Below we show the initialization arguments:

--- arguments ---

* static_list: list or tuple (required), list of objects to calculate the static part of hamiltonian operator. The format goes like:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
``` 

* dynamic_list: list or tuple (required), list of objects to calculate the dynamic part of the hamiltonian operator. The format goes like:

```python
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```

 For the dynamic list ```func``` is the function which goes in front of the matrix or operator given in the same list. ```func_args``` is a tuple containing the extra arguments which go into the function to evaluate it like: 
```python
f_val = func(t,*func_args)
```


* N: integer (optional), number of sites to create the Hamiltonian with.

* shape: tuple (optional), shape to create the Hamiltonian with.

* copy: bool (optional,) whether or not to copy the values from the input arrays. 

* check_symm: bool (optional), flag whether or not to check the operator strings if they obey the requested symmetries.

* check_herm: bool (optional), flag whether or not to check if the operator strings create a hermitian matrix. 

* check_pcon: bool (optional), flag whether or not to check if the operator strings conserve magnetization/particle number. 

* dtype: dtype (optional), data type to create the matrices with. 

* kw_args: extra options to pass to the basis class.

--- hamiltonian attributes ---: '_. ' below stands for 'object. '

* _.ndim: number of array dimensions, always $2$.
    
* _.Ns: number of states in the Hilbert space.

* _.get_shape: returns tuple which has the shape of the Hamiltonian (Ns,Ns)

* _.is_dense: returns ```True``` if the Hamiltonian contains a dense matrix as a componnent. 

* _.dtype: returns the data type of the Hamiltonian

* _.static: returns the static part of the Hamiltonian 

* _.dynamic: returns the dynamic parts of the Hamiltonian 

* _.T: returns the transpose of the Hamiltonian.

* _.H: returns the hermitian conjugate of the Hamiltonian  


#### **methods of hamiltonian class**

##### **arithmetic operations**
The ```hamiltonian``` objects currectly support certain arithmetic operations with other hamiltonians, scipy sparse matrices or numpy dense arrays and scalars, as follows:

* between other hamiltonians we have: ```+,-,*,+=,-=``` . Note that ```*``` only works between a static and static hamiltonians or a static and dynamic hamiltonian, but NOT between two dynamic hamiltonians.
* between numpy and sparse square arrays we have: ```*,+,-,*=,+=,-=``` (versions >= v0.0.5b)
* between scalars: ```*,*=``` (versions >= v0.0.5b)
* negative operator '-H'
* indexing and slicing: ```H[times,row,col]``` 

##### **quantum (algebraic) operations**
We have included some basic functionality into the ```hamiltonian``` class, useful for quantum calculations:

* matrix operations:
 * transpose (return copy of transposed hamiltonian set ```copy=True``` otherwise this is done inplace): 
  ```
  H_tran = H.transpose(copy=False)
  ```
 * hermitian conjugate:
  ```
  H_herm = H.getH(copy=False)
  ```
 * conjugate
  ```
  H_conj = H.conj() # always inplace
  ```
 * trace
  ```
  H_tr = H.trace() 
  ```
* matrix vector product / dense matrix:

  usage:
    ```python
    B = H.dot(A,time=0,check=True) # $B = HA$
    B = H.rdot(A,time=0,check=True) # $B = AH$
    ```
where ```time``` is the time to evaluate the Hamiltonian at, for the product, and by default ```time=0```. ```_.rdot``` is another function similar to ```_.dot```, but it performs the matrix multiplication from the right. The ```check``` option lets the user control whether or not to do checks for shape compatibility. If checks are turned off, there will be checks later which will throw a shape error. if `time` is a list of values there are two possible different outcomes:
 1. if ```A.shape[1] == len(time)``` then the hamiltonian is evaluated at the i-th time and dotted into the i-th column of ```A``` to get the ith column of ```B``` 
 2. if ```A.shape[1] == 1,0``` then the hamiltonian dot is evaluated on that one vector for each time. The results are then stacked such that the columns contain all the vectors. 
If either of these cases do not match then an error is thrown.
  
* matrix elements:

  usage:
    ```python
    Huv = H.matrix_ele(u,v,time=0,diagonal=False,check=True)
    ```
which evaluates $\langle u|H(t=0)|v\rangle$ if ```u``` and ```v``` are vectors but (versions >= v0.0.2b) can also handle ```u``` and ```v``` as dense matrices. ```diagonal=True``` then the function will only return the diagonal part of the resulting matrix multiplication. The check option is the same as for 'dot' and 'rdot'. The vectorization with time is the same as for the 'dot' and 'rdot' functions. 
  **NOTE: the inputs should NOT be hermitian conjugated, the function will do that automatically.

* project/rotate a Hamiltonian to a new basis:
  ```python
  H_new = H.project_to(V)
    H_new = H.rotate_by(V,generator=False)
  H_new = H.rotate_by(K,generator=True,**exp_op_args)
  ```
`project_to` returns a new hamiltonian object which is: $V^\dagger H V$. Note that ```V``` need not be a square matrix. `rotate_by`, when ```generator=False```, is the same as `project_to`: $V^\dagger H V$, but with ```generator=True``` this function uses ```K``` as the generator of a transformation: $e^{a^\ast K^\dagger}He^{a K}$. This function uses the [exp_op](#exp_op-class) class and the extra arguments ```**exp_op_args``` are optional arguments for the `exp_op` constructor. 
  
* Schrödinger dynamics:

  The ```hamiltonian``` class has two private functions which can be passed into scipy's ODE solvers in order to numerically solve the Schrödinger equation in both real and imaginary time:
    1. __SO(t,v) which performs: -iH(t)|v>
    2. __ISO(t,v) which performs: -H(t)|v> 
    3. __LO(t,rho) which performs: i[H,rho] (as super operator on flat state) (added v0.2.0)
  
  The interface with ```complex_ode``` is as easy as:
  
    ```python
    solver = complex_ode(H._hamiltonian__SO)
    ```
  
From here all one has to do is use the solver object as specified in the scipy [documentation](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.integrate.complex_ode.html#scipy.integrate.complex_ode). Note that if the hamiltonian is not complex and one is using ```__ISO```, the equations are real-valued and so it is more efficient to use ```ode``` instead of ```complex_ode```.

 This functionality is wrapped in a method called evolve (version >= 0.1.0):

  ```python
  vf = H.evolve(v0,t0,times,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args)
  ```
 * v0:  array (required) initial state array.
 * t0: real scalar (required) initial time.
 * times:  real array like (required) a time or generator of times to evolve up to.
 * eom: string (optional) used to pick between Schrodinger evolution ("SE") or dynamics ("LvNE") (added v0.2.0)
 * solver_name: string (optional) used to pick which scipy ode solver to use.
 * verbose: bool (optional) prints out when the solver has evolved to each time in times
 * iterate: bool (optional) returns 'vf' as an iterator over the evolution vectors without storing the solution for every time in 'times', otherwise the solution is stored with the time index being the last index of the output array. 
 * imag_time: bool (optional) toggles whether to evolve with __SO or __ISO.
 * solver_args: (optional) the optional arguments which are passed into the solver. The default setup is: ```nstep = 2**31 - 1```, ```atol = 1E-9```, ```rtol = 1E-9```.
 
 note that for Liouvillian dynamics the output is indeed a square complex array.
  
The ```hamiltonian``` class also has built-in methods which are useful for doing exact diagonalisation (ED) calculations:

* Full diagonalization:

  usage:
    ```python
    eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)
    eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)
    ```
where ```**eigh_args``` are optional arguments which are passed into the eigenvalue solvers. For more information check out the scipy docs for [eigh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh) and [eigvalsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigvalsh.html#scipy.linalg.eigvalsh). 
  
  **NOTE: ```overwrite_a=True``` by default for memory conservation.

* Sparse diagonalization, which uses ARPACK:

  usage:
    ```python
    eigenvalues,eigenvectors=H.eigsh(time=time,**eigsh_args)
    ```
where ```**eigsh_args``` are optional arguments which are passed into the eigenvalue solvers. For more information check out the scipy docs for [eigsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.sparse.linalg.eigsh.html).

##### **other operations**
There are also some methods which are useful when interfacing QuSpin with functionality from other packages:

* return copy of hamiltonian as csr matrix: 
  ```python
  H_csr = H.tocsr(time=time)
  ```

* return a copy of hamiltonian as csc matrix: 
  ```python
  H_csr = H.tocsc(time=time)
  ```
  
* return a copy of hamiltonian as dense numpy matrix: 
  ```python
  H_dense = H.todense(time=time,order=None,out=None)
  ```
* return a copy of hamiltonian as dense numpy array: 
  ```python
  H_dense = H.toarray(time=time,order=None,out=None)
  ```

* return a copy of hamiltonian: 
  ```python
  H_new = H.copy()
  ```

* cast hamiltonian to different data type: 
  ```python
  H_new = H.astype(dtype,copy=True)
  ```
* changing the sparse format underlying matrices are stored as:
 * change to sparse formats:
 ```
 H_new = H.as_sparse_format(fmt,copy=False)
 ```
   available formats for fmt string: 
    * "csr" for compressed row storage
    * "csc" for compressed column storage
    * "dia" for diagonal storage
    * "bsr" for block compressed row storage
 
 * change to dense format:
 ```python
 H_new = H.as_dense_format(copy=False)
 ```

### **useful hamiltonian functions**

##### **commutator**
```python
commutator(H1,H2)
```
This function returns the commutator of two Hamiltonians H1 and H2.

##### **anti commutator**
```python
anti_commutator(H1,H2)
```
This function returns the anti-commutator of two Hamiltonians H1 and H2.

### **exp_op class**
```python
expO = exp_op(O, a=1.0, start=None, stop=None, num=None, endpoint=None, iterate=False)
```
This class constructs the matrix exponential of the operator (`hamiltonian` object) ```O```. It does not calculate the exact matrix exponential but instead computes the action of the matrix exponential through the Taylor series. This is slower but for sparse arrays this is more memory efficient. All of the functions make use of the [expm_multiply](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply) function in Scipy's sparse library. 

This class also allows the option to specify a grid of points `grid` on a line in the complex plane via the optional arguments `start`, `stop`, `num`, and `endpoint`, so that the exponential is evaluated at each point `a*grid[i]*O`. When this option is specified, the array `grid` is created via the numpy function [linspace](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html), as `grid=linspace(start,stop,num)`.

--- arguments ---

* H: matrix/hamiltonian (compulsory), operator to be exponentiated.

* a: scalar (optional), (complex-valued) prefactor to go in front of the operator in the exponential: exp(a*O)

* start:  scalar (optional) specify the starting point for a grid of points to evaluate the matrix exponential at

* stop: (optional) specify the end point of the grid. 

* num: (optional) specify the number of grid points between start and stop. Default if 50.

* endpoint: (optional) if `True` this will make sure `stop` is included in the set of grid points (Note this changes the grid step size).

* iterate: (optional) if `True` a generator is returned which can iterate over the grid points at a later time, as opposed to producing a list of all evaluated points. This is more memory efficient but at the cost of speed.

--- `exp_op` attributes ---: '_. ' below stands for 'object. '

* `_.a`: returns the prefactor `a`

* `_.ndim`: returns the number of dimensions, always $2$

* `_.H`: returns the hermitian conjugate of this operator

* `_.T`: returns the transpose of this operator

* `_.O`: returns the operator which is being exponentiated

* `_.get_shape`: returns the tuple which contains the shape of the operator

* `_.iterate`: returns a bool telling whether or not this function will iterate over the grid of values or return a list

* `_.grid`: returns the array containing the grid points the exponential will be evaluated at

* `_.step`: returns the step size between the grid points

#### **methods of exp_op class**

* Transpose:
 ```python
   expO_trans = expO.transpose(copy=False)
 ```
 
* Hermitiain conjugate:
 ```python
   expO_H = expO.getH(copy=False)
 ```
 
* complex conjugate:
 ```python
   expO_conj = expO.conj() # always inplace
 ```
 
* setting a new grid:
   ```python
   expO_new = expO.set_grid(start, stop, num=None, endpoint=None)
  ```
 
* unset grid:
 ```python
   expO_new = expO.unset_grid()
 ```
 
* toggle iterate:
 ```python
   expO= expO.set_iterate(value)
 ```
 
* return matrix at given grid point. If `dense=True` the output format is dense.
 ```python
   expO_mat = expO.get_mat(index=None, time=0, dense=False)
 ```
 
#### **mathematical functions**
* dot product:
 ```python
 B = expO.dot(A,time=0)
 B = expO.rdot(A,time=0)
 ```
 
* rotate operator `A` by ```expO```, $B=\left[e^{O}\right]^\dagger A e^{O}$:
 ```python
   B = expO.sandwich(A,copy=False)
 ```
here ```time``` is always a scalar which is the time point at which operator ```O``` is evaluated at for dynamic hamiltonians; for matrices or static hamiltonians this does not have any effect.  If ```iterate=True``` all these functions return generators to return values of the results over the grid points. For example:

```python
expO.set_iterate(True)
B_generator = expO.dot(A)

for B in B_generator:
  # code here
```
 

### **HamiltonianOperator class**
```python
H_operator = HamiltonianOperator(operator_list,system_arg,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args)
```

This class uses the ```basis.Op``` function to calculate the matrix vector product on the fly, reducing the amount of memory needed for a calculation at the cost of speed. This object is useful for doing large scale Lanczos calculations using `eigsh`. 

--- arguments ---

* static_list: (compulsory) list of operator strings to be used for the HamiltonianOperator. The format goes like:

```python
operator_list=[[opstr_1,[indx_11,...,indx_1m]],...]
```
    
* system_arg: int/basis_object (compulsory) number of sites to create basis object/basis object.

* check_symm: bool (optional) flag whether or not to check the operator strings if they obey the given symmetries.

* check_herm: bool (optional) flag whether or not to check if the operator strings create hermitian matrix. 

* check_pcon: bool (optional) flag whether or not to check if the operator string whether or not they conserve magnetization/particle number. 

* dtype: dtype (optional) data type to case the matrices with. 

* kw_args: extra options to pass to the basis class.

--- hamiltonian attributes ---: '_. ' below stands for 'object. '

* _.ndim: number of dimensions, always $2$.
    
* _.Ns: number of states in the Hilbert space.

* _.shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

* _.dtype: returns the data type of the Hamiltonian

* _.operator_list: returns the list of operators given to this  

* _.T: returns the transpose of this operator

* _.H: returns the hermitian conjugate of this operator

* _.basis: returns the basis used by this operator

* _.LinearOperator: returnss a linear operator of this object

#### **Method of HamiltonianOperator class**
* dot products from left and right:

 ```python
 B = H.dot(A) # $B = HA$
 B = H.matvec(A) # $B=HA$
 B = H.rdot(A) # $B = AH$
 ```
* Lanczos Diagonalization:

 ```python
 E,V = H.eigsh(**eigsh_args)
 # or you can pass this object directly into the function itself:
 E,V = scipy.sparse.linalg.eigsh(H,**eigsh_args)
 E,V = scipy.sparse.linalg.eigsh(H.LinearOperator,**eigsh_args)
 E,V = scipy.sparse.linalg.eigsh(H.get_LinearOperator(),**eigsh_args)
 ```
 
### **ops_dict class**
```python
O_dict = ops_dict(self,input_dict,N=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs)
```
this object maps operatators/matricies to keys, which when calling various operations allows to specify the scalar multiples in front of these operators by a key.

--- arguments ---

* input_dict: dictionary (compulsory) this is a dictionary which should contain values which are operator lists like the static_list input to the hamiltonian class while the key's correspond to the key vales which you use to specify the coupling during other method calls, for example:

```python
 input_dict = { "Jzz": [["zz",Jzz_bonds]],"hx" : [["x" ,hx_site  ]] }
``` 
* N: (optional) number of sites to create the hamiltonian with.

* shape: (optional) shape to create the hamiltonian with.

* copy: (optional) weather or not to copy the values from the input arrays. 

* check_symm: (optional) flag whether or not to check the operator strings if they obey the given symmetries.

* check_herm: (optional) flag whether or not to check if the operator strings create hermitian matrix. 

* check_pcon: (optional) flag whether or not to check if the oeprator string whether or not they conserve magnetization/particles. 

* dtype: (optional) data type to case the matrices with. 

* kw_args: extra options to pass to the basis class.    

--- ops_dict attributes ---: '_. ' below stands for 'object. '    

* _.basis: the basis associated with this hamiltonian, is None if hamiltonian has no basis. 

* _.ndim: number of dimensions, always 2.

* _.Ns: number of states in the hilbert space.

* _.get_shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

* _.is_dense: return 'True' if the hamiltonian contains a dense matrix as a componnent. 

* _.dtype: returns the data type of the hamiltonian

#### **Method of ops_dict class**

##### **arithmetic operations**
mathematical operators: This class only supports a limited number of operations so far. Addition and subtraction (```+,-,+=,-=```) are implemented between two ops_dict objects, ops_dict also supports multiplication by scalars (```*,*=,/,/=```).

##### **quantum (algebraic) operations**
 This class shares many of the same features of the hamiltonian class but with some minor differences. One such difference is instead of specifying the ```time=...``` you specify ```pars={...}``` which contains the dictionary whos key values are the scalar multiple for the operators with the same key.  By default if no keys are specified they are assumed to be 1.

* copy:
 ```python
 O_dict_copy = O_dict.copy()
 ```

* tranpose:
 ```python
 O_dict = O_dict.transpose(copy = False)
 ```

* conjugate:
 ```python
 O_dict = O_dict.conj()
 O_dict = O_dict.conjugate()
 ```

* Hermitian conjugate:
 ```python
 O_dict = O_dict.getH(copy=False)
 O_dict = O_dict.H # same as getH(copy=False)
 ```

* recast datatype:
 ```python
 O_dict_new = O_dict.astype(dtype)
 ```
* trace:
 ``` python
 O_trace = O_dict.trace(pars=pars)
 ```


* return sparse matrix evaluated at given parameters:
 ```python
 O_csr = O_dict.tocsr(pars=pars)
 O_csc = O_dict.tocsr(pars=pars)
 O_matrix = O_dict.todense(pars=pars)
 O_array = O_dict.toarray(pars=pars)
 ```
* calling the object does one of the above functions depending on the sparsity of the matrix. If there are no dense components then the call will return a csr matrix otherwise it will return a numpy matrix.
 ``` python 
 O_mat = O_dcit(**pars)
 ```
 
* Convert to a hamiltonian. If a parameters value is a tuple containing  a function and function arguments (f,f_args), that operator is converted to a dynamic operator and all scalar values get converted into static operators and then from these a hamiltonian is constructed:
 ```python
 O_ham = O_dict.tohamiltonian(pars=ham_pars)
 ```
* convert this to a scipy [LinearOperator](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.LinearOperator) which can be used in many of scipy's sparse linear algebra functions. Note that you can also convert this object from scipy's aslinearoperator function, but if this is the case then all parameters are set to 1.
 ```python
 O_LO = O_dict.aslinearoperator(pars=pars)
 O_LO = scipy.sparse.linalg.aslinearoperator(O_dict)
 ```

* general linear algebra operations similar to hamiltonian functions:
 ```python
 O_dict.dot(self,V,pars={},check=True)
 O_dict.matrix_ele(Vl,Vr,pars={},diagonal=False,check=True)
 O_dict.eigsh(pars={},**eigsh_args)
 O_dict.eigh(pars={},**eigh_args)
O_dict. eigvalsh(pars={},**eigvalsh_args)
 ```


## **basis objects**

The `basis` objects provide a way of constructing all the necessary information needed to construct a sparse Hamiltonian from a list of operators. All `basis` objects are derived from the same `base` object class and have mostly the same functionality. There are two subtypes of the `base` object class: the first type consists of `basis` objects which provide the bulk operations required to create a sparse matrix out of an operator string. For example, `basis` type one creates a single spin-chain basis. The second basis type wraps multiple objects of the first type together in a tensor-style `basis` type. For instance, `basis` type two can take two spin-chain bases and create the corresponding tensor product basis out of them.   

### **spins basis in 1d**

The `spin_basis_1d` class provides everything necessary to create the `hamiltonian` for any spin-$S$ system in 1d with $S=1/2,1,3/2,2,5/2,3,\dots$. The available operators one can use are the standard spin operators: ```I,+,-,z``` which represent the Pauli operators (matrices): identity, raising, lowering and $z$-projection, repsectively. Additionally, for spin-$1/2$ one can also define the operators ```x,y``` which are always constructed as ```+/- = x``` $\pm i$ ```y```.

It is also possible to create the `hamiltonian` in a given symmetry-reduced block as follows:

* magnetization symmetries: 
 * pick single magnetization sector by number of up spins: ```Nup=0,1,...,L``` 
 * pick a selection magnetization sectors: ```Nup = [0,1,...]```
 * pick magnetization sector by magnetization density ( $-S \leq M \leq S $, rounding to nearest integer): ```M=0.0,0.3,etc...``` 
 * if ```Nup``` and ```M``` are ```None``` this picks all magnetization sectors. 
* parity (reflection about middle of chain) symmetry: ```pblock = +/- 1```
* spin inversion symmetry: ```zblock = +/- 1```
* (spin inversion)*(parity) symmetry: ```pzblock = +/- 1 ```
* spin inversion on sublattice A (even sites, lattice site `i=0` is even): ```zAblock = +/- 1```
* spin inversion on sublattice B (odd sites): ```zBblock = +/- 1```
* translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

Other optional arguments include:

* ```pauli```: for $S=1/2$, toggle whether or not to use spin-$1/2$ or Pauli matrices for the matrix elements. Default is ```pauli = True```.
* ```a```: the unit cell size for the translational symmetry. Default is ```a = 1```. For example, in the presence of a staggered $z$-magnetisation and translation symmetry use `a=2`. 

Usage of `spin_basis_1d`:

```python
basis = spin_basis_1d(L,S="1/2",**symmetry_blocks)
```
---arguments---

* L: int (required) length of chain to construct `basis` for

* S: str (optional) string specifying the total spin (e.g. S="5/2" is spin-5/2, S="2" is spin-2). 

* symmetry_blocks: (optional) specify which block of a particular symmetry to construct `basis` for. 

--- `spin_basis_1d` attributes ---: '_. ' below stands for 'basis_object. '

* _.L: returns length of the chain as integer

* _.N: returns number of sites in chain as integer

* _.Ns: returns number of states in the Hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 

* _.sps: returns integer with number of states per site (sps=2 for spin-1/2, sps=3 for spin-1, etc.)



### **boson basis in 1d**

The `boson_basis_1d` class provides everything necessary to create the `hamiltonian` for a system of bosons in 1d. The available operators one can use are the standard particle operators: ```I,+,-,n``` which define the identity, raising and lowering operators, and the number operator, respectively. Also available is the higher spin ```z``` operator ```z = n - (sps-1)/2``` with ```sps``` denoting the total number of states per site (note that for hard-core bosons ```sps=2```).

It is also possible to create the `hamiltonian` in a given symmetry-reduced block as follows:

* particle number symmetries: 
 *  pick single particle number sector: ```Nb=0,1,...,L```
 * pick list of particle number sectors: ```Nb = [0,1,...]``` 
 * pick density of particles per site (rounds to nearest integer number of particles): ```nb = 0.1,0.5,etc...```
 * Both ```Nb``` and ```nb```  being ```None``` gives all filling sectors up to ```sps```.
* parity (reflection about middle of chain) symmetry: ```pblock = +/- 1```
* parcticle-hole (charge) symmetry: ```cblock = +/- 1```
* (particle-hole)*(parity) symmetry: ```pcblock = +/- 1 ```
* particle-hole on sublattice A (even sites, lattice site `i=0` is even): ```cAblock = +/- 1```
* particle-hole on sublattice B (odd sites): ```cBblock = +/- 1```
* translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

Other optional arguments include:

* ```a```: the unit cell size for the translational symmetry. Default is ```a = 1```. For example, in the presence of a staggered potential and translation symmetry use `a=2`. 

Usage of `boson_basis_1d`:

```python
basis = boson_basis_1d(L,sps=None,**symmetry_blocks)
```
---arguments---

* L: int (required) length of chain to construct `basis` for

* sps: int (semi-optional) number of states per site. Either ```Nb``` or ```nb``` (see particle symmetries above) or ```sps``` must be specified. If ```Nb``` is specified the default is ```sps=Nb+1```.

* symmetry_blocks: (optional) specify which block of a particular symmetry to construct `basis` for. 

--- `boson_basis_1d` attributes ---: '_. ' below stands for 'basis_object. '

* _.L: returns length of the chain as integer

* _.N: returns number of sites in chain as integer

* _.Ns: returns number of states in the Hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 

* _.sps: returns integer with number of states per site


### **fermion basis in 1d**

The `fermion_basis_1d` class provides everything necessary to create the `hamiltonian` for a system of spinless fermions in 1d. The available operators one can use are the standard particle operators: ```I,+,-,n``` which define the identity, raising and lowering operators, and the number operator, respectively. Also available is the spin-1/2 ```z``` operator ```z=n+1/2```. 

It is also possible to create the `hamiltonian` in a given symmetry-reduced block as follows:

* particle number symmetries: 
 *  pick single particle number sector: ```Nf=0,1,...,L```
 * pick list of particle number sectors: ```Nf = [0,1,...]``` 
 * pick density of particles per site (rounds to nearest integer number of particles): ```nf = 0.1,0.5,etc...```
 *  Both ```Nf``` and ```nf```  being ```None``` gives all filling sectors.
* parity (reflection about middle of chain) symmetry: ```pblock = +/- 1```
* parcticle-hole (charge) symmetry: ```cblock = +/- 1```
* (particle-hole)*(parity) symmetry: ```pcblock = +/- 1 ```
* particle-hole on sublattice A (even sites, lattice site `i=0` is even): ```cAblock = +/- 1```
* particle-hole on sublattice B (odd sites): ```cBblock = +/- 1```
* translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

Other optional arguments include:

* ```a```: the unit cell size for the translational symmetry. Default is ```a = 1```. For example, in the presence of a staggered potential and translation symmetry use `a=2`. 

Usage of `fermion_basis_1d`:

```python
basis = fermion_basis_1d(L,**symmetry_blocks)
```
---arguments---

* L: int (required) length of chain to construct `basis` for

* symmetry_blocks: (optional) specify which block of a particular symmetry to construct `basis` for. 

--- `fermion_basis_1d` attributes ---: '_. ' below stands for 'basis_object. '

* _.L: returns length of the chain as integer

* _.N: returns number of sites in chain as integer

* _.Ns: returns number of states in the Hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 

* _.sps: returns integer with number of states per site (sps=2 for spinless fermions)


### **harmonic oscillator basis**
This basis implements a single harmonic oscillator mode. The available operators are ```I,+,-,n```, corresponding to the identity, raising operator, lowering operator, and the number operator, respectively.  

Usage of `ho_basis`:

```python
basis = ho_basis(Np)
```

---arguments---

* Np: int (compulsory) highest number state to allow in the Hilbert space.

--- `ho_basis` attributes ---: '_. ' below stands for 'object. '

* _.Np: returns the highest number state of this `ho_basis` 

* _.Ns: returns number of states in the Hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 

### **tensor basis** 

The `tensor_basis` class combines two `basis` objects `basis1` and `basis2` together into a new `basis` object which can be then used, e.g., to create the Hamiltonian over the tensoer product Hilbert space:

```python
basis1 = spin_basis_1d(L,Nup=L/2)
basis2 = spin_basis_1d(L,Nup=L/2)
t_basis = tensor_basis(basis1,basis2)
```

The syntax for the operator strings is as follows. The operator strings are separated by a '|' while the index array has no splitting character.

```python
# tensoring two z spin operators at site 1 for basis1, and site 5 for basis2
opstr = "z|z" 
indx = [1,5] 
```
If there are no operator strings on either side of the '|' then an identity operator is assumed. Thus, '`+|`' stands for the operator $\sigma^+_1 = \sigma^+_1\otimes I$, while '`|+`' stands for the operator $\sigma^+_2 = I\otimes\sigma^+_2$.

### **photon basis** 

This class allows the user to define a basis which couples to a single photon mode. The operators for the photon sector are the same as the harmonic oscillator basis: '+', '-', 'n', and 'I'. 

There are two types of `basis` objects that one can create: a particle (magnetization + photon) conserving basis or a non-conserving basis. 

In the conserving case one can specify the total number of quanta using the the `Ntot` keyword argument:

```python
p_basis = photon_basis(basis_class,*basis_args,Ntot=...,**symmetry_blocks)
```

For the non-conserving basis, one must specify the total number of photon (a.k.a harmonic oscillator) states with `Nph`:

```python
p_basis = photon_basis(basis_class,*basis_args,Nph=...,**symmetry_blocks)
```
In both cases, because of the nature of the interaction between the photon mode and the other basis, one must pass the constructor of the `basis` class as opposed to an already constructed basis. This is because the basis has to be constructed for each magnetization/particle sector separately. 

### **symmetry and hermiticity checks**
New in version 0.1.0 we have included functionality classes which check various properties of a given static and dynamic operator lists. They include the following:

* check if the final complete list of opertors obeys the requested symmetry of the given `basis` object passed to the `hamiltonian` constructor. The check can be turned off with the flag ```check_symm=False ``` in the [hamiltonian](#hamiltonian-objects) class. 
* check if the final complete list of operators are hermitian. The check can be turned off with the flag ```check_herm=False``` in the [hamiltonian](#hamiltonian-objects) class. 
* check if the final complete list of opertors obeys magnetization/particle number conservation (for spin systems this means that magnetization sectors do not mix). The check can be turned off with the flag ```check_pcon=False``` in the [hamiltonian](#hamiltonian-class) class. 

### **methods for basis objects**

The following functions are defined for every `basis` class:

```python
basis.Op(opstr,indx,J,dtype)
```
This function takes the string of operators `opstr`, the sites on which they act `indx`, the coupling `J` and the corresponding data dype `dtype`, and returns the matrix elements, their row index and column index in the Hamiltonian matrix for the symmetry sector the basis was initialized with.

---arguments--- (*all required*)

* opstr: string which contains the operator string.
* indx: 1-dim array which contains the site indices where the operators act.
* J: scalar value which is the coefficient in front of the operator (i.e. the coupling constant).
* dtype: the data type the matrix elements should be cast to. 

RETURNS:

* ME: 1-dim array which contains the matrix elements.
* row: 1-dim array containing the row indices of the matrix elements.
* col: 1-dim array containing the column indices of the matrix elements. 

```python
basis.get_vec(v)
```
This function converts a state defined in the symmetry-reduced basis to the full, symmetry-free basis.

---arguments---

* v: two options
  1. 1-dim array which contains the state
  2. 2-dim array which contains multiple states in the columns
  
RETURNS:
state or states in the full basis as columns of the returned array.

```python
basis.get_proj(dtype,pcon=False)
```
This function returns the transformation from the symmetry-reduced basis to the full basis. If ```pcon=True``` this function returns the projector to the particle conserved basis (useful in bosonic/single particle systems)

---arguments---

* dtype: data type to cast projector matrix in. 

RETURNS:
projector to the full basis as a sparse matrix.

```python
partial_trace(state,sub_sys_A=None,return_rdm="A",sparse=False,state_type="pure")
```
This function calculates the reduced density matrix (DM), performing a partial trace of a quantum state.

--- arguments ---

* `state`: (required) the state of the quantum system. Can be a
  1. pure state (default) [numpy array of shape (Ns,)].
  2. density matrix [numpy array of shape (Ns,Ns)].
  3. collection of states [dictionary {'V_states':V_states}] containing the states in the columns of `V_states` [shape (Ns,Nvecs)]
* `sub_sys_A`: (optional) tuple or list to define the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0]. Default is tuple(range(L//2)).
* `return_rdm`: (optional) flag to return the reduced density matrix. Default is `None`.
  These arguments differ when used with `photon` or `tensor` basis.
  1. 'A': str, returns reduced DM of subsystem A 
  2. 'B': str, returns reduced DM of subsystem B
  3. 'both': str, returns reduced DM of both subsystems A and B
* `state_type`: (optional) flag to determine if 'state' is a collection of pure states or a density matrix
  1. 'pure': (default) (a collection of) pure state(s)
  2. 'mixed': mixed state (i.e. a density matrix)
* `sparse`: (optional) flag to enable usage of sparse linear algebra algorithms.

RETURNS: reduced DM

```python
ent_entropy(self,state,sub_sys_A=None,return_rdm=None,state_type="pure",sparse=False,alpha=1.0)
```
This function calculates the entanglement entropy of subsystem A and the corresponding reduced 
density matrix.

--- arguments ---

* `state`: (required) the state of the quantum system. Can be a:
  1. pure state (default) [numpy array of shape (Ns,)].
  2.density matrix [numpy array of shape (Ns,Ns)].
  3.collection of states [dictionary {'V_states':V_states}] containing the states in the columns of `V_states` [shape (Ns,Nvecs)]
* `sub_sys_A`: (optional) tuple or list to define the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0]. Default is tuple(range(L//2)).
* `return_rdm`: (optional) flag to return the reduced density matrix. Default is `None`.
  1. 'A': str, returns reduced DM of subsystem A
  2. 'B': str, returns reduced DM of subsystem B
  3. 'both': str, returns reduced DM of both subsystems A and B
* `state_type`: (optional) flag to determine if 'state' is a collection of pure states or a density matrix
  1. 'pure': (default) (a collection of) pure state(s)
  2. 'mixed': mixed state (i.e. a density matrix)
* `sparse`: (optional) flag to enable usage of sparse linear algebra algorithms.
* `alpha`: (optional) Renyi alpha parameter. Default is 'alpha=1.0'.

RETURNS: dictionary with keys:

'Sent': entanglement entropy.
'rdm_A': (optional) reduced density matrix of subsystem A
'rdm_B': (optional) reduced density matrix of subsystem B

## **tools**

The `tools` package is a collection of useful functionalities to facilitate specific calculations in the studies of quantum many-body systems.

### **measurements** 

#### **entanglement entropy**

```python
ent_entropy(system_state,basis,chain_subsys=None,densities=True,subsys_ordering=True,alpha=1.0,DM=False,svd_return_vec=[False,False,False])
```
This function calculates the entanglement entropy in a lattice quantum subsystem based on the Singular
Value Decomposition (svd).

Consider a quantum chain of $N$ sites, and define a subsystem $A$ of $N_A$ sites and its complement $A^c$: $N=N_A + N_{A^c}$. Given the reduced density matrices 

$$ \rho_A = \mathrm{tr}_B \rho, \qquad \rho_{A^c} = \mathrm{tr}_{A^c} \rho $$

the entanglement entropy density (normalised w.r.t. subsystem $A$) between the two subsystems is given by

$$ S_\mathrm{ent} = -\frac{1}{N_A}\mathrm{tr}_A \rho_A\log\rho_A = -\frac{1}{N_A}\mathrm{tr}_{A^c} \rho_{A^c}\log\rho_{A^c} $$


RETURNS:  dictionary with keys:

* `Sent`: entanglement entropy.

* `DM_chain_subsys`: (optional) reduced density matrix of the chain subsystem retained after the partial trace. The basis in which the reduced DM is returned is the full $z$-basis of the subsystem. For instance, if the subsystem contains $N_A$ sites the reduced DM will be a $(2^{N_A}, 2^{N_A})$ array. This is required because some symmetries of the system might not be inherited by the subsystem. The only exception to this appears when `basis` is an instance of `photon_basis` AND the subbsystem is the entire chain (i.e. one traces out the photon dregree of freedom only and entirely): then the reduced DM is returned in the basis specified by the `..._basis_1d` argument passed into the definition of `photon_basis`, and thus inherits all symmetries of `..._basis_1d` by construction.

* `DM_other_subsys`: (optional) reduced density matrix of the complement subsystem, i.e. the subsystem which is being traced out. The basis the redcuded DM is returned in, is the same as `DM_chain_subsys` above.

* `U`: (optional) svd U matrix

* `V`: (optional) svd V matrix

* `lmbda`: (optional) svd singular values

 --- arguments ---

* `system_state`: (required) the state of the quantum system. Can be a:

  1. pure state [numpy array of shape (Ns,)].

  2. density matrix (DM) [numpy array of shape (Ns,Ns)].

  3. diagonal DM [dictionary ```{'V_rho': V_rho, 'rho_d': rho_d}``` containing the diagonal DM
    `rho_d` [numpy array of shape (Ns,)] and its eigenbasis in the columns of V_rho
    [numpy array of shape (Ns,Ns)]. The dictionary keys CANNOT be chosen arbitrarily.].

  4. collection of states [dictionary ```{'V_states':V_states}```] containing the states
    in the columns of V_states [shape (Ns,Nvecs)].

* `basis`: (required) the basis used to build `system_state`. Must be an instance of `photon_basis`,
  `spin_basis_1d`, `fermion_basis_1d`, `boson_basis_1d`. 

* `chain_subsys`: (optional) a list of lattice sites to specify the chain subsystem. Default is

  * ```[0,1,...,N/2-1,N/2]``` for `spin_basis_1d`, `fermion_basis_1d`, `boson_basis_1d`.

  * ```[0,1,...,N-1,N]``` for `photon_basis`.

* `DM`: (optional) string to enable the calculation of the reduced density matrix. Available options are

  * `chain_subsys`: calculates the reduced DM of the subsystem `chain_subsys` and
    returns it under the key `DM_chain_subsys`.

  * `other_subsys`: calculates the reduced DM of the complement of `chain_subsys` and
    returns it under the key `DM_other_subsys`.

  * `both`: calculates and returns both density matrices as defined above.

  Default is `False`.   

* `alpha`: (optional) Renyi alpha parameter. Default is `alpha=1.0`. When alpha is different from unity,
     the entropy keys have attached `_Renyi` to their standard label.

* `densities`: (optional) if set to `True`, the entanglement entropy is normalised by the size of the
     subsystem [i.e., by the length of `chain_subsys`]. Default is `True`.

* `subsys_ordering`: (optional) if set to `True`, the sites in `chain_subsys` are automatically ordered. Default is      `True`.

* `svd_return_vec`: (optional) list of three booleans to return the Singular Value Decomposition (svd) 
  parameters:

  * `[True, . , . ]` returns the svd matrix `U`.

  * `[ . ,True, . ]` returns the singular values `lmbda`.

  * `[ . , . ,True]` returns the svd matrix `V`.

  Any combination of the above is possible. Default is ```[False,False,False]```.




#### **diagonal ensemble observables**
```python
diag_ensemble(N,system_state,E2,V2,densities=True,alpha=1.0,rho_d=False,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,Srdm_args=())
```

This function calculates the expectation values of physical quantities in the Diagonal ensemble 
set by the initial state (see eg. arXiv:1509.06411). Equivalently, these are also the infinite-time 
expectation values after a sudden quench from a Hamiltonian $H_1$ to a Hamiltonian $H_2$. Let us label the two eigenbases as $V_1=\{|n_1\rangle: H_1|n_1\rangle=E_1|n_1\rangle\}$ and $V_2=\{|n_2\rangle: H_2|n_2\rangle=E_2|n_2\rangle\}$.


RETURNS:  dictionary with keys depending on the passed optional arguments:

replace "..." below by `pure`, `thermal` or `mixed` depending on input params.

* `Obs_...`: infinite time expectation of observable `Obs`.

* `delta_t_Obs_...`: infinite time temporal fluctuations of `Obs`.

* `delta_q_Obs_...`: infinite time quantum fluctuations of `Obs`.

* `Sd_...` (`Sd_Renyi_...` for `alpha!=1.0`): Renyi entropy of density matrix of Diagonal Ensemble with parameter `alpha`.

* `Srdm_...` (`Srdm_Renyi_...` for `alpha!=1.0`): Renyi entropy of reduced density matrix of Diagonal Ensemble with parameter `alpha`.

* `rho_d`: density matrix of diagonal ensemble


--- arguments ---


* `N`: (required) system size $N$.

* `system_state`: (required) the state of the quantum system. Can be a:

  1. pure state [numpy array of shape (Ns,) or (,Ns)].

  2. density matrix (DM) [numpy array of shape (Ns,Ns)].

  3. mixed DM [dictionary] ```{'V1':V1,'E1':E1,'f':f,'f_args':f_args,'V1_state':int,'f_norm':False}``` to 
    define a diagonal DM in the basis `V1` of the Hamiltonian $H_1$. The keys are

    * `V1`: (required) array with the eigenbasis of $H_1$ in the columns.

    * `E1`: (required) eigenenergies of $H_1$.

    * `f`: (optional) the distribution used to define the mixed DM. Default is
      `f = lambda E,beta: numpy.exp(-beta*(E - E[0]) )`. 

    * `f_args`: (required) list of arguments of function `f`. If `f` is not defined, by 
    efault we have `f=numpy.exp(-beta*(E - E[0]))`, and `f_args` specifies the inverse temeprature list [beta].

    * `V1_state` (optional) : list of integers to specify the states of `V1` wholse pure 
      expectations are also returned.

    * `f_norm`: (optional) if set to `False` the mixed DM built from `f` is NOT normalised
      and the norm is returned under the key `f_norm`.
     
    The keys are CANNOT be chosen arbitrarily.
    
    If this option is specified, then all Diagonal ensemble quantities are averaged over the energy distribution $f(E_1,f\_args)$:
    
    $$ \overline{\mathcal{M}_d} = \frac{1}{Z_f}\sum_{n_1} \mathcal{M}^{n_1}_d \times f(E_{n_1},f\_args), \qquad \mathcal{M}^{\psi}_d = \langle\mathcal{O}\rangle_d^\psi,\ \delta_q\mathcal{O}^\psi_d,\ \delta_t\mathcal{O}^\psi_d,\ S_d^\psi,\ S_\mathrm{rdm}^\psi $$

    

* `V2`: (required) numpy array containing the basis of the Hamiltonian $H_2$ in the columns.

* `E2`: (required) numpy array containing the eigenenergies corresponding to the eigenstates in `V2`.
  This variable is only used to check for degeneracies.

* `rho_d`: (optional) When set to `True`, returns the Diagonal ensemble DM under the key `rho_d`. For example, if `system_state` is the pure state $|\psi\rangle$:

 $$\rho_d^\psi = \sum_{n_2} \left|\langle\psi|n_2\rangle\right|^2\left|n_2\rangle\langle n_2\right| = \sum_{n_2} \left(\rho_d^\psi\right)_{n_2n_2}\left|n_2\rangle\langle n_2\right| $$

* `Obs`: (optional) hermitian matrix of the same size as `V2`, to calculate the Diagonal ensemble 
  expectation value of. Appears under the key `Obs`. For example, if `system_state` is the pure state $|\psi\rangle$:
  
  $$ \langle\mathcal{O}\rangle_d^\psi = \lim_{T\to\infty}\frac{1}{T}\int_0^T\mathrm{d}t \frac{1}{N}\langle\psi\left|\mathcal{O}(t)\right|\psi\rangle = \frac{1}{N}\sum_{n_2}\left(\rho_d^\psi\right)_{n_2n_2} \langle n_2\left|\mathcal{O}\right|n_2\rangle$$

* `delta_q_Obs`: (optional) QUANTUM fluctuations of the expectation of `Obs` at infinite-times. 
  Requires `Obs`. Appears under the key `delta_q_Obs`. Returns temporal fluctuations `delta_t_Obs` for free.
  For example, if `system_state` is the pure state $|\psi\rangle$:
  
  $$ \delta_q\mathcal{O}^\psi_d = \frac{1}{N}\sqrt{\lim_{T\to\infty}\frac{1}{T}\int_0^T\mathrm{d}t \langle\psi\left|\mathcal{O}(t)\right|\psi\rangle^2 - \langle\mathcal{O}\rangle_d^2}= \frac{1}{N}\sqrt{ \sum_{n_2\neq m_2} \left(\rho_d^\psi\right)_{n_2n_2} [\mathcal{O}]^2_{n_2m_2} \left(\rho_d^\psi\right)_{m_2m_2} } $$

* `delta_t_Obs`: (optional) TIME fluctuations around infinite-time expectation of `Obs`. Requires `Obs`. 
  Appears under the key `delta_t_Obs`. For example, if `system_state` is the pure state $|\psi\rangle$:
  
  $$ \delta_t\mathcal{O}^\psi_d = \frac{1}{N}\sqrt{ \lim_{T\to\infty}\frac{1}{T}\int_0^T\mathrm{d}t \langle\psi\left|[\mathcal{O}(t)]^2\right|\psi\rangle - \langle\psi\left|\mathcal{O}(t)\right|\psi\rangle^2} =  
 \frac{1}{N}\sqrt{\langle\mathcal{O}^2\rangle_d - \langle\mathcal{O}\rangle_d^2 - \delta_q\mathcal{O}^2 }$$

* `Sd_Renyi`: (optional) diagonal Renyi entropy in the basis of $H_2$. The default Renyi parameter is 
  `alpha=1.0` (see below). Appears under the key `Sd_Renyi`. For example, if `system_state` is the pure state $|\psi\rangle$:
  
  $$ S_d^\psi = \frac{1}{1-\alpha}\log\mathrm{tr}\left(\rho_d^\psi\right)^\alpha $$

* `Srdm_Renyi`: (optional) Renyi entropy of infinite-time reduced density matrix for a subsystem of a choice. The default Renyi parameter is `alpha=1.0` (see below). Appears under the key `Srdm_Renyi`. Requires 
  `Srdm_args`. To specify the subsystem, see documentation of `ent_entropy`. For example, if `system_state` is the pure state $|\psi\rangle$ (see also notation in documentation of `ent_entropy`):
  
  $$ S_\mathrm{rdm}^\psi = \frac{1}{1-\alpha}\log \mathrm{tr}_{A} \left( \mathrm{tr}_{A^c} \rho_d^\psi \right)^\alpha $$

* `Srdm_args`: (optional) dictionary of `ent_entropy` arguments, required when `Srdm_Renyi = True`. Some important 
keys are:

  1. `basis`: (required) the basis used to build `system_state`. Must be an instance of `photon_basis`,
    `spin_basis_1d`, `fermion_basis_1d`, or `boson_basis_1d`. 

  2. `chain_subsys`: (optional) a list of lattice sites to specify the chain subsystem. Default is

   * `[0,1,...,N/2-1,N/2]` for `spin_basis_1d`, `fermion_basis_1d`, `boson_basis_1d`.

   * `[0,1,...,N-1,N]` for `photon_basis`. 

   * `subsys_orderin`: (optional) if set to `True`, the sites in`chain_subsys` are automatically ordered. Default is `True`.

* `densities`: (optional) if set to `True`, all observables are normalised by the system size $N$, except
  for the entanglement entropy which is normalised by the subsystem size [i.e., by the length of `chain_subsys`].   Detault is `True`.

* `alpha`: (optional) Renyi alpha parameter. Default is `alpha=1.0`.




#### **time evolution**
```python
ED_state_vs_time(psi,E,V,times,iterate=False)
```
This routine calculates the time evolved initial state as a function of time. The initial state is `psi` and the time evolution is carried out under the Hamiltonian `H` with the eigensystem (`E`,`V`).

RETURNS:  either a matrix with the time evolved states in the columns, or an iterator which generates the states one by one.

--- arguments --- 

* `psi`: (required) initial state.

* `V`: (required) unitary matrix containing in its columns all eigenstates of the Hamiltonian $H$. 

* `E`: (required) array containing the eigenvalues of the Hamiltonian $H$. The order of the eigenvalues must correspond to the order of the columns of `V`. 

* `times`: (required) an array of times to evaluate the time evolved state at. 

* `iterate`: (optional) if set to `True` this function returns the generator of the time evolved state. 




```python
obs_vs_time(psi_t,times,Obs_dict,return_state=False,Sent_args={})
```


This routine calculates the expectation value as a function of time of a dictionary of observables `Obs_dict`, given an array `psi_t` with each column of which corresponds to a state at for the time vector `times`. 

RETURNS:  dictionary with keys:

* `custom_name` (same as the keys of `Obs_dict`). For each key of `Obs_dict`, the time-dependent expectation of the observable `Obs_dict[key]` is calculated and returned.

* `psi_t`: (optional) returns a 2D array the columns of which give the state at the associated times.

* `Sent_time`: (optional) returns the entanglement entropy of the state for the time points `times`.


--- arguments ---

* `psi_t`: (required) Source of time dependent states, three different types of inputs:


 1. `psi_t`: tuple `(psi, E, V)`  
  * `psi` [1-dim array]: initial state 
  * `V` [2-dim array]: unitary matrix containing in its columns all eigenstates of the Hamiltonian $H$. 
  * `E` [1-dim array]: real vector containing the eigenvalues of the Hamiltonian. The order of the eigenvalues must correspond to the order of the columns of `V`.
 2. `psi_t`: 2-dim array which contains the time dependent states as columns of the array.
 3. `psi_t`:  Iterator generates the states sequentially ( For most evolution functions you can get this my setting `iterate=True`. This is more memory efficient as the states are generated on the fly as opposed to being stored in memory).

* `times`: (required) a real array of times to evaluate the expectation value at. If this is specified, the `hamiltonian` objects will be dynamically evaluated at the times requested. 

* `Obs_dict`: (required) dictionary of objects to take the expecation values with. This accepts NumPy, and SciPy matrices as well as `hamiltonian` objects.

* `return_state`: (optional) when set to `True`, returns a matrix whose columns give the state vector at the times specified by `times`. The return dictonary key is `psi_time`.

* `Sent_args`: (optional) when non-empty, this dictionary containing `ent_entropy` arguments enables the calculation of `Sent_time`. Some important keys are:

  1. `basis`: (required) the basis used to build `system_state`. Must be an instance of `photon_basis`,
    `spin_basis_1d`, `fermion_basis_1d`, `boson_basis_1d`. 

  2. `chain_subsys`: (optional) a list of lattice sites to specify the chain subsystem. Default is

   * ```[0,1,...,N/2-1,N/2]``` for `spin_basis_1d`, `fermion_basis_1d`, `boson_basis_1d`.

   * ```[0,1,...,N-1,N]``` for `photon_basis`. 

   * `subsys_ordering`: (optional) if set to `True`, the sites in`chain_subsys` are automatically ordered. Default is `True`.



``` python
evolve(v0,t0,times,f,solver_name="dop853",real=False,stack_state=False,verbose=False,imag_time=False,iterate=False,f_params=(),**solver_args)
```


This function implements (imaginary) time evolution for a user-defined first-order f function.

RETURNS:  array containing evolved state in time

--- arguments ---

* `v0`: (required) initial state

* `t0`: (required) initial time

* `times`: (required) vector of times to evaluate the time-evolved state at

* `f`: (required) user-defined function to determine the ordinary differential equation (all derivatives must be first order)

* `solver_name`: (optional) scipy solver integrator. Default is `dop853`.

* `real`: (optional) flag to determine if `f` is real or complex-valued. Default is `False`.

* `stack_state`: (optional) if `f` is written to take care of real and imaginary parts separately, this flag will take this into account. Default is `False`.

* `verbose`: (optional) prints normalisation of state at teach time in `times`

* `imag_time`: (optional) must be set to `True` when `f` defines imaginary-time evolution, in order to normalise the state at each time in `times`. Default is `False`.

* `iterate`: (optional) creates a generator object to time-evolve the state. Default is `False`.

* `f_params`: (optional) a tuple to pass all parameters of the function `f` to solver. Default is `f_params=()`.

* `solver_args`: (optional) dictionary with additional [scipy integrator (solver)](https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html) arguments.   




#### **mean level spacing**
```python
mean_level_spacing(E)
```

This routine calculates the mean-level spacing `r_ave` of the energy distribution E, see arXiv:1212.5611.

RETURNS: float with mean-level spacing `r_ave`.

--- arguments ---

* `E`: (required) ordered list of ascending, nondegenerate eigenenergies.




#### **project operator**
```python
project_op(Obs,proj,dtype=_np.complex128):
```
This function takes an observable `Obs` to a different basis. It can be used to switch between symmetry sectors.

RETURNS:  dictionary with keys 

* `Proj_Obs`: projected observable `Obs`.
  
--- arguments ---

* `Obs`: (required) operator to be projected.

* `proj`: (required) `basis` object of the final space after the projection or a matrix which contains the projector.

* `dtype`: (optional) data type. Default is `np.complex128`.




#### **Kullback-Leibler divergence**
```python
KL_div(p1,p2)
```
This routine returns the Kullback-Leibler divergence of the discrete probability distributions `p1` and `p2`. 






### **periodically driven systems**
This package contains tools which can be helpful in simulating periodically-driven (Floquet) systems.


#### **Floquet class**

```python
floquet = Floquet(evo_dict,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1)
```
Calculates the Floquet spectrum for a given protocol, and optionally the Floquet hamiltonian matrix, and Floquet eigenvectors.

--- arguments ---

* `evo_dict`: (required) dictionary which passes the different types of protocols to calculate evolution operator:


 1. Continuous protocol. Each basis state is evolved separately for one driving period. 

  * `H`: (required) hamiltonian object to generate the time evolution. 

  * `T`: (required) period of the protocol. 

  * `rtol`: (optional) relative tolerance for the ode solver. (default = `1E-9`)

  * `atol`: (optional) absolute tolerance for the ode solver. (default = `1E-9`)

 2. Step protocol from a `hamiltonian` object. Uses Matrix exponential to calculate the evolution operator. 

  * `H`: (required) hamiltonian object to generate the hamiltonians at each step.
        
  * `t_list`: (required) list of times to evaluate the hamiltonian at when doing each step.

  * `dt_list`: (required) list of time steps for each step of the evolution. 

 3. Step protocol from a list of hamiltonians. Uses Matrix exponential to calculate the evolution operator. 

  * `H_list`: (required) list of matrices which to evolve with.

  * `dt_list`: (required) list of time steps to evolve with. 

 * `HF`: (optional) if set to `True` returns Floquet hamiltonian. 


* `UF`: (optional) if set to `True` returns evolution operator. 

* `ThetaF`: (optional) if set to `True` returns the eigenvalues of the evolution operator. 

    * VF: (optional) if set to 'True' save the eigenvectors of the evolution operator. 

* `n_jobs`: (optional) set the number of processors to be spawned for the calculation. 

--- `Floquet` attributes ---: '_. ' below stands for 'object. '

Always given:

* `_.EF`: Floquet qausi-energies

Calculate via flags:

* `_.HF`: Floquet Hamiltonian dense array

* `_.UF`: Evolution operator

* `_.VF`: Floquet eigenstates

* `_.thetaF`: eigenvalues of evolution operator




#### **Floquet_t_vec**
```python
tvec = Floquet_t_vec(Omega,N_const,len_T=100,N_up=0,N_down=0)
```
Returns a time vector (np.array) which hits the stroboscopic times, and has as attributes their indices. The time vector can be divided into three stages: up, constant and down (think of a ramp-up stage, a constant amplitude state, and a ramp-down stage).

--- arguments ---

* `Omega`: (required) drive frequency

* `N_const`: (required) number of time periods in the constant period

* `N_up`: (optional) number of time periods in the ramp-up period

* `N_down`: (optional) number of time periods in the ramp-down period

* `len_T`: (optional) number of time points within a period. N.B. the last period interval is assumed open on the right, i.e. $[0,T)$ and the point $T$ is not accounted for in the definition of `len_T`. 


--- `Floquet_t_vec` attributes ---: '_. ' below stands for 'object. '


* `_.vals`: time vector values

* `_.i`: initial time value

* `_.f`: final time value

* `_.tot`: total length of time: `t.i - t.f`

* `_.T`: period of drive

* `_.dt`: time vector spacing

* `_.len`: length of total time vector

* `_.len_T`: number of points in a single period interval, assumed half-open: $[0,T)$

* `_.N`: total number of periods

--- strobo attribues ---

* `_.strobo.vals`: strobosopic time values

* `_.strobo.inds`: strobosopic time indices

--- time vector stage attributes --- (available only if `N_up` or `N_down` are parsed)


* `_.up` : refers to time vector of the up-stage; inherits the above attributes (e.g. `_up.strobo.inds`), except for `_.T`, `_.dt`, and `._lenT`

* `_.const` : refers to time vector of const-stage; inherits the above attributes, except for `_.T`, `_.dt`, and `._lenT`

* `_.down` : refers to time vector of down-stage; inherits the above attributes except `_.T`, `_.dt`, and `._lenT`

This object also acts like an array, you can iterate over it as well as index the values.





### **block diagonalisation**


#### **block_diag_hamiltonian**

```python
P,H = block_diag_hamiltonian(blocks,static,dynamic,basis_con,basis_args,dtype,
                             check_symm=True,check_herm=True,check_pcon=True)
```
This function constructs a hamiltonian object which is block diagonal with the blocks being created by the list 'blocks'.
    
RETURNS:

* `P`: sparse matrix which is the projector to the block space.

* `H`: hamiltonian object in block diagonal form.     
    
--- arguments ---

* `blocks`: (required) list/tuple/iterator which contains a dictionary with the blocks the user would like to put into the hamiltonian constructor.

* `static`: (required) the static operator list which is used to construct the block hamiltonians. Follows `hamiltonian` format.

* `dynamic`: (required) the dynamic operator list which is used to construct the block hamiltonians. Follows `hamiltonian` format.

* `basis_con`: (required) basis constructor to build the `basis` objects used to create the block diagonal Hamiltonian.

* `basis_args`: (required) tuple passed as the first argument for `basis_con`, contains the required arguments. 

* `dtype`: (required) data type of `hamiltonin` to be constructed.

* `check_symm`: (optional) flag which tells the function to check the symmetry of the operators for the first hamiltonian constructed.

* `check_herm`: (optional) same as `check_symm` but for hermiticity.

* `check_pcon`: (optional) same as `check_symm` but for particle conservation.


#### **`block_ops` class**

``` python
block_H=block_ops(blocks,static,dynamic,basis_con,basis_args,dtype,save_previous_data=True,
                    compute_all_blocks=False,check_symm=True,check_herm=True,check_pcon=True)
```
This class is used to split up the dynamics of a state over various symmetry sectors if the initial state does not obey the symmetry but the Hamiltonian does. Moreover we provide a multiprocessing option which allows the user to distribute the caculation of the dynamics over multiple cores.

---arguments---

* `blocks`: (required) list/tuple/iterator which contains the blocks the user would like to put into the Hamiltonian as dictionaries.

* `static`: (required) the static operator list which is used to construct the block Hamiltonians. Follows `hamiltonian` format.

* `dynamic`: (required) the dynamic operator list which is used to construct the block hamiltonians. Follows `hamiltonian` format.

* `basis_con`: (required) basis constructor to build the `basis` objects which will create the block diagonal Hamiltonian.

* `basis_args`: (required) tuple passed as the first argument for `basis_con`, contains required arguments. 

* `dtype`: (required) data type of `hamiltonin` to be constructed.

* `get_proj_args`: (optional) dictionary which contains arguments for basis.get_proj(dtype,...). As of v(0.2.0) the only arguement would be for pcon=True of one would like to project to the particle conserving basis (which is useful to same memory for some systems)

* `save_previous_data`: (optional) when doing the evolution this class constructs the Hamiltonians for the corresponding symmetry blocks. This takes some time and thus by setting this flag to `True`, the class will keep previously calculated Hamiltonians. This might be advantageous if at a later time one needs to do evolution in these blocks again so the corresponding Hamiltonians do not have to be calculated again.

* `compute_all_blocks`: (optional) flag which tells the class to just compute all Hamiltonian blocks upon initialization. This option also sets `save_previous_data=True` by default. 

* `check_symm`: (optional) flag which tells the function to check the symmetry of the operators for the first Hamiltonian constructed.

* `check_herm`: (optional) same for `check_symm` but for hermiticity.

* `check_pcon`: (optional) same for `check_symm` but for particle conservation. 

--- `block_ops` attributes ---: '_. ' below stands for 'object. '

`_.dtype`: the numpy data type the block Hamiltonians are stored with.

`_.save_previous_data`: flag which tells the user if data is being saved. 

`_.H_dict`: dictionary which contains the block hamiltonians under the key `str(block)` where block is the `block` dictionary.

`_.P_dict`: dictionary which contains the block projectors under the same key `H_dict`.

`_.basis_dict`: dictionary which contains the `basis` objects under the same key as `H_dict`. 

`_.static`: list of the static operators used to construct the block Hamiltonians.

`_.dynamic`: list of dynamic operators used to construct the block Hamiltonians.


##### **methods of `block_ops` class**

The following functions are available as attributes of the `block_ops` class:

###### **`evolve`**

```python
block_H.evolve(psi_0,t0,times,iterate=False,n_jobs=1,H_real=False,solver_name="dop853",**solver_args)
```
This function is the creates blocks and then uses them to run H.evole in parallel.

RETURNS:

i) `iterate = True`
    * returns generator which generates the time-dependent state in the full Hilbert space basis.

ii) `iterate = False`
    * returns numpy ndarray which has the time-dependent states in the full Hilbert space basis as rows.

--- arguments ---

* `psi_0`: (required) ndarray/list/tuple of state which defined on the full Hilbert space of the problem. Does not need to obey any symmetry.

* `t0`: (required) time to initiate the dynamics at.

* `times`: (required) list/numpy array (or other iterable) containing the times the solution is avaluated at.

* `iterate`: (optional) if set to `True`, function returns the states as a generator, if set to `False` the states are returned as array.

* `n_jobs`: (optional) number of processes to do dynamics with. NOTE: one of those processes is used to gather results. For optimal performance, all blocks should be approximately the same size and `n_jobs-1` must be a common devisor of the number of blocks, such that there is roughly the same workload for each process. Otherwise the computation will be as slow as the slowest process.

* ...: the rest are just arguments which are used by `H.evolve`, see `hamiltonian` class for more details.


###### **`expm`**
```python
block_H.expm(psi_0,H_time_eval=0.0,iterate=False,n_jobs=1,
             a=-1j,start=None,stop=None,endpoint=None,num=None,shift=None):
```
This function creates blocks and then uses them to evolve the quantum state using scipy.sparse routine `expm_multiply` in parallel.
    
RETURNS:

i) `iterate = True`
    * returns generator which generates the time-dependent state in the full Hilbert space basis.

ii) `iterate = False`
    * returns numpy ndarray which has the time-dependent states in the full Hilbert space basis as rows.

--- arguments ---

* `psi_0`: (required) ndarray/list/tuple of state which defined on the full Hilbert space of the problem. Does not need to obey any symmetry.

* `H_time_eval`: (optional) times to evaluate the Hamiltonian at when doing the matrix exponentiation. 

* `iterate`: (optional) if set to `True`, function returns the states as a generator, if set to `False` the states are returned as array.

* `n_jobs`: (optional) number of processes to do dynamics with. NOTE: one of those processes is used to gather results. For optimal performance, all blocks should be approximately the same size and `n_jobs-1` must be a common devisor of the number of blocks, such that there is roughly the same workload for each process. Otherwise the computation will be as slow as the slowest process.

* ...: the rest are just arguments which are used by the `exp_op` class, cf documentiation for more details.




