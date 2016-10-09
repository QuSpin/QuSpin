#**qspin**
qspin is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state of the art exact diagonalization calculations on one-dimensional spin-1/2 chains with length up to 32 sites (including). The interface allows the user to define any Hamiltonian which can be constructed from spin-1/2 operators. It also gives the user the flexibility of accessing many symmetries in 1d. Moreover, there is a convenient built-in way to specifying the time dependence of operators in the Hamiltonian, which is interfaced with user-friendly routines to solve the time dependent Schrodinger equation numerically. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access any powerful Python scientific computing tools. 




Contents
--------
* [Installation](#installation)
 * [Automatic Install](#automatic-install)
 * [Manual Install](#manual-install)
 * [Updating the Package](#updating-the-package)
* [Basic Package Usage](#using-the-package)
 * [Constructing hamiltonians](#constructing-hamiltonians)
 * [Using basis Objects](#using-basis-objects)
 * [Specifying Symmetries](#specifying-symmetries)
* [List of Package Functions](#list-of-package-functions) 
	* [operator objects](#operator-objects)
	 * [hamiltonian class](#hamiltonian-class)
	 * [functions for hamiltonians](#functions-for-hamiltonians)
	 * [exo\_op class](#exp\_op-class)
	 * [HamiltonianOperator class](#hamiltonianoperator-class)
	* [basis objects](#basis-objects)
	 * [1d\_spin\_basis](#1d\_spin\_basis)
	 * [harmonic oscillator basis](#harmonic-oscillator-basis)
	 * [tensor\_basis objects](#tensor\_basis-objects)
	 * [methods of basis classes](#methods-of-basis-classes)
	* [tools](#tools)
	 * [measusrements](#measurements)
	 * [floquet](#floquet)



#**Installation**

###**automatic install**

The latest version of this package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux, OS X and Windows 64-bit systems. The automatic installation of qspin requires the Anaconda package manager for Python. Once it has been installen, all one has to do to install is run:

```
$ conda install -c weinbe58 qspin
```

This will install the latest version on your computer. Right now the package is in its beta stages and so it may not be availible for installation on all platforms using this method. In this case one can also manually install the package:


###**Manual Install**

To install qspin manually, download the source code either from the [master](https://github.com/weinbe58/qspin/archive/master.zip) branch, or by cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

unix:
```
python setup.py install 
```

or windows command line:
```
setup.py install
```
For the manual installation you must have all the prerequisite python packages: [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), and [joblib](https://pythonhosted.org/joblib/) installed. We recommend [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) to manage your python packages.

When installing the package manually, if you add the flag ```--record install.txt```, the location of all the installed files will be output to install.txt which stores information about all installed files. This can prove useful when updating the code. 

###**Updating the Package**

To update the package with Anaconda, all one has to do is run the installation command again.

To safely update a manually installed version of qspin, one must first manually delete the entire package from the python 'site-packages/' directory. In Unix, provided the flag ```--record install.txt``` has been used in the manual installation, the following command is sufficient to completely remove the installed files: ```cat install.txt | xargs rm -rf```. In Windows, it is easiest to just go to the folder and delete it from Windows Explorer. 


#**Using the Package**
All the calculations done with qspin happen through [hamiltonians](#hamiltonian-objects). The hamiltonian is a type which uses Numpy and Scipy matrices to store the quantum Hamiltonian operator. Time-independent operators are summed together into a single static matrix, while each time-dependent operator is stored separatly along with the time-dependent coupling in front of it. Whenever the user wants to perform an operation invonving a time-dependent operator, the time dependence is evaluated on the fly by specifying the time argument. The user can initialize the hamiltonian types with Numpy arrays or Scipy matrices. Apart from this we provide a user-friendly representation for constructings the Hamiltonian matrices for many-body operators. 

Many-body operators in qspin are defined by a string of letters representing the operator types, together with a list which holds the indices for the sites that each operator acts at on the lattice. For example, in a spin system we can represent any multi-spin operator as:

|      opstr       |      indx      |        operator      |
|:----------------:|:--------------:|:---------------------------:|
|"o<sub>1</sub>...o<sub>n</sub>"|[J,i<sub>1</sub>,...,i<sub>n</sub>]|J S<sub>i<sub>1</sub></sub><sup>o<sub>1</sub></sup>...S<sub>i<sub>n</sub></sub><sup>o<sub>n</sub></sup>|

where o<sub>i</sub> can be x, y, z, +, or -. Then S<sub>i<sub>n</sub></sub><sup>o<sub>n</sub></sup> is the spin-1/2 operator acting on lattice site i<sub>n</sub>. This gives the full range of possible spin-1/2 operators that can be constructed. By default the hamiltonian will use the spin-1/2 operators (a.k.a. sigma matrices). For different systems, there are different types of operators. To see the available operators for a given type of system check out the [basis](basis-objects) classes. 

###**Constructing Hamiltonians**
The hamiltonian is constructed as:
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
static_list  = [[opstr_1,[indx_11,...,indx_1m]                   ], matrix_2,...]
dynamic_list = [[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```
In the definition of the dynamic list, ```func``` is the python function which goes in front of the matrix or operator given in the same list. ```func_args``` is a tuple of the extra function arguments to evaluate it like: 
```python
f_val = func(t,*func_args)
```

####keyword arguments (kwargs):
the ```**kwargs``` give extra information about the hamiltonian. There is a variety of different things one can input here, and which one to choose depends on what Hamiltonian one would like to create. These arguments are used to specify symmetry blocks, give a shape and provide the floating point type to store the matrix elements with, disable automatic checks on the symemtries and the hermiticity of the hamiltonian, etc.

**Providing a Shape:**
to construct many-body operators one must either specify the number of lattice sites with ```N=...``` or pass in a basis object as ```basis=...```, more about basis objects can be found [here](#basis-objects). You can also specify the shape using the ```shape=...``` keyword argument. For input lists which contain matrices only, the shape does not have to be specified. If empty lists are given, then either one of the previous options must be provided to the hamiltonian constructor.  

**Numpy dtype:**
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements in. It supports ```float32```, ```float64```, ```float128```, ```complex64```, ```complex128```, and ```complex256```. The default type is ```complex128```. To specify the dtype use the dtype keyword argument:

```python
H=hamiltonian(...,dtype=numpy.float32,...)
```
Note that not all platforms and all of Scipy and Numpy functions support dtype float128 and complex256.


**Example:** 
constructing the hamiltonian object of the transverse field Ising model with time-dependent field for a 10-site chain:

```python
# python script
from qspin.operators import hamiltonian
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
Notice that one needs to include both '-z+' and '+z-' operators to make sure the Hamiltonian is hermitian. If one forgets about this, there is still the automatic hermiticity check which raises an error, see the optional flag ```check_herm``` in the hamiltonian object class organisation [here](#hamiltonian-objects) which is set to ```True``` by default.

###**Using basis Objects**

Basis objects are another very useful type included in qspin: they provide all of the functionality which calculates the matrix elements from the operator string representation of many-body operators. On top of this, they have been programmed to calculate the matrix elements in different symmetry blocks of the many-body hamiltonian with the help of optional flags. To use a basis object to construct the hamiltonian, one simply uses the basis keyword argument:
```python
H = hamiltonian(static_list,dynamic_list,...,basis=basis,...)
```
More information about basis objects can be found in the [basis objects](#basis-objects) section.

We recommend using basis objects when multiple separate Hamiltonians which share the same symmetries need to be constructed. This provides the advantage of saving time when creating the corresponding symmetry objects.


###**Specifying Symmetries**
Adding symmetries is easy and versatile: one can either just add extra keyword arguments to the initialization of your hamiltonian or, when one initializes a basis object one can build in the desired symmetries. By default, the hamiltonian uses spin-1/2 operators and 1d-symmetries. At this point the spin-chain symmetries implemented are for spins-1/2 operators in 1 dimension. 
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
If the user passes the symmetries into the hamiltonian constructor, the constructor first creates a [spin_basis_1d](spin\_basis\_1d) object for the given symmetries, and then uses that object to construct the matrix elements. Because of this, if one needs to construct multiple hamiltonian objects in the same symmetry block, it is more efficient to first construct the basis object and then use it to construct all different hamiltonians:

```python

basis = spin_basis_1d(L,Nup=Nup,pblock=pblock,...)
H1 = hamiltonian(static1_list,dynamic1_list,basis=basis)
H2 = hamiltonian(static2_list,dynamic2_list,basis=basis)
...
```

**NOTE:** for beta versions spin_basis_1d is named as basis1d.

#**List of Package Functions**

##**Operator Objects**

###**hamiltonian Class**:
```python
H = hamiltonian(static_list,dynamic_list,N=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs)
```

The hamiltonian class wraps most of the functionalty of the package. This object allows the user to construct a many-body hamiltonian, solve the schrodinger equation, do full/lanczos diagonalization and many other things. Below shows the initialization arguments:

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


* N: integer (optional), number of sites to create the hamiltonian with.

* shape: tuple (optional), shape to create the hamiltonian with.

* copy: bool (optional,) whether or not to copy the values from the input arrays. 

* check_symm: bool (optional), flag whether or not to check the operator strings if they obey the requested symmetries.

* check_herm: bool (optional), flag whether or not to check if the operator strings create a hermitian matrix. 

* check_pcon: bool (optional), flag whether or not to check if the operator strings conserve magnetization/particle number. 

* dtype: dtype (optional), data type to create the matrices with. 

* kw_args: extra options to pass to the basis class.

--- hamiltonian attributes ---: '_. ' below stands for 'object. '

* _.ndim: number of array dimensions, always 2.
		
* _.Ns: number of states in the Hilbert space.

* _.get_shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

* _.is_dense: return ```True``` if the hamiltonian contains a dense matrix as a componnent. 

* _.dtype: returns the data type of the hamiltonian

* _.static: return the static part of the hamiltonian 

* _.dynamic: returns the dynamic parts of the hamiltonian 

* _.T: returns the transpose of the hamiltonian.

* _.H: returns the hermitian conjugate of the hamiltonian  


####**Methods of hamiltonian Class**

#####**Arithmetic Operations**
The hamiltonian objects currectly support certain arithmetic operations with other hamiltonians, scipy sparse matrices or numpy dense arrays and scalars, as follows:

* between other hamiltonians we have: ```+,-,*,+=,-=``` . Note that ```*``` only works between a static and static hamiltonians or a static and dynamic hamiltonians, but NOT between two dynamic hamiltonians.
* between numpy and sparse square arrays we have: ```*,+,-,*=,+=,-=``` (versions >= v0.0.5b)
* between scalars: ```*,*=``` (versions >= v0.0.5b)
* negative operator '-H'
* indexing and slicing: ```H[times,row,col]``` 

#####**Quantum (algebraic) Operations**
We have included some basic functionality into the hamiltonian class, useful for quantum calculations:

* matrix transformations:
 * transpose (return copy of transposed hamiltonian set copy=True otherwise this is done inplace): 
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
* matrix vector product / dense matrix:

  usage:
    ```python
    B = H.dot(A,time=0,check=True) # $B = HA$
    B = H.rdot(A,time=0,check=True) # $B = AH$
    ```
  where time is the time to evaluate the Hamiltonian at for the product, by default time=0. ```_.rdot``` is another function similar to ```_.dot```, but it performs the matrix multiplication from the right. The ```check``` option lets the user control whether or not to do checks for shape compatibility. If checks are turned off, there will be checks later which will throw a shape error. if `time` is a list of values there are two different results that can happen:
 1. if ```A.shape[1] == len(time)``` then the hamiltonian is evaluated at the ith time and dotted into the ith column of ```A``` to get the ith column of ```B``` 
 2. if ```A.shape[1] == 1,0``` then the hamiltonian dot is evaluated on that one vector for each time. the results are then stacked such that the columns contain all the vectors. 
If either of these cases do not match then an error is thrown.
  
* matrix elements:

  usage:
    ```python
    Huv = H.matrix_ele(u,v,time=0,diagonal=False,check=True)
    ```
  which evaluates < u|H(time)|v > if u and v are vectors but (versions >= v0.0.2b) can also handle u and v as dense matrices. ```diagonal=True``` then the function will only return the diagonal part of the resulting matrix multiplication. The check option is the same as for 'dot' and 'rdot'. the vectorization with time is the same as for the 'dot' and 'rdot' functions. 
  **NOTE: the inputs should not be hermitian conjugated, the function will do that automatically.

* project a Hamiltonian to a new basis:
	```python
	H_new = H.project_to(V)
	H_new = H.rotate_by(O,generator=False,**exp_op_args)
	```
The First function returns a new hamiltonian object which is: V<sup>+</sup> H V. Note that V need not be a square matrix. The second function when ```generator=False``` is the same as the first function but with ```generator=True``` the function uses O as the generator of a transformation. This function uses the [exp_op](#exp_op-class) class and the extra arguments ```**exp_op_args``` are optional arguments for the exp_op constructor. 
  
* Schroedinger dynamics:

  The hamiltonian class has 2 private functions which can be passed into Scipy's ode solvers in order to numerically solve the Schroedinger equation in both real and imaginary time:
    1. __SO(t,v) which proforms: -iH(t)|v>
    2. __ISO(t,v) which proforms: -H(t)|v> 
  
  The interface with complex_ode is as easy as:
  
    ```python
    solver = complex_ode(H._hamiltonian__SO)
    ```
  
  From here all one has to do is use the solver object as specified in the scipy [documentation](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.integrate.complex_ode.html#scipy.integrate.complex_ode). Note that if the hamiltonian is not complex and you are using __ISO, the equations are real valued and so it is more efficient to use ```ode``` instead of ```complex_ode```.

 This functionality is wrapped in a method called evolve (version >= 0.2.0):

	```python
	vf = H.evolve(v0,t0,times,solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args)
	```
 * v0:  array (required) initial state array.
 * t0: real scalar (required) initial time.
 * times:  real array like (required) a time or generator of times to evolve up to.
 * solver_name: string (optional) used to pick which Scipy ode solver to use.
 * verbose: bool (optional) prints out when the solver has evolved to each time in times
 * iterate: bool (optional) returns 'vf' as an iterator over the evolution vectors without storing the solution for every time in 'times'. 
 * imag_time: bool (optional) toggles whether to evolve with __SO or __ISO.
 * solver_args: (optional) the optional arguments which are passed into the solver. The default setup is: ```nstep = 2**31 - 1```, ```atol = 1E-9```, ```rtol = 1E-9```.
  
The hamiltonian class also has built in methods which are useful for doing exact diagonalisation (ED) calculations:

* Full diagonalization:

  usage:
    ```python
    eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)
    eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)
    ```
  where ```**eigh_args``` are optional arguments which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh) and [eigvalsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigvalsh.html#scipy.linalg.eigvalsh). 
  
  **NOTE: ```overwrite_a=True``` always for memory conservation.

* Sparse diagonalization, which uses ARPACK:

  usage:
    ```python
    eigenvalues,eigenvectors=H.eigsh(time=time,**eigsh_args)
    ```
  where ```**eigsh_args``` are optional arguments which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.sparse.linalg.eigsh.html).

#####**Other Operations**
There are also some methods which are useful when interfacing qspin with functionality from other packages:

* return copy of hamiltonian as csr matrix: 
  ```python
  H_csr = H.tocsr(time=time)
  ```

* return copy of hamiltonian as csc matrix: 
  ```python
  H_csr = H.tocsc(time=time)
  ```
  
* return copy of hamiltonian as dense numpy matrix: 
  ```python
  H_dense = H.todense(time=time,order=None,out=None)
  ```
* return copy of hamiltonian as dense numpy array: 
  ```python
  H_dense = H.toarray(time=time,order=None,out=None)
  ```

* return copy of hamiltonian: 
  ```python
  H_new = H.copy()
  ```

* cast hamiltonian to different dtype: 
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
 ```
 H_new = H.as_dense_format(copy=False)
 ```

### **Functions for hamiltonians**

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

###**exp_op Class**
```
exp_O = exp_op(O, a=1.0, start=None, stop=None, num=None, endpoint=None, iterate=False)
```
This class constructs an object which acts on various objects with the matrix exponential of the matrix/hamiltonian ```H```. It does not calculate the actual matrix exponential but instead computes the action of the matrix exponential through the taylor series. This is slower but for sparse arrays this is more memory efficient. All of the functions make use of the [expm_multiply](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply) function in Scipy's sparse library. This class also allows the option to specify a grid of points on a line in the complex plane via the optional arguements. if this is specified then an array `grid` is created via the numpy function [linspace](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html), then every time a math function is called the exponential is evaluated with for `a*grid[i]*O`.

--- arguments ---

* H: matrix/hamiltonian (compulsory), The operator which to be exponentiated.

* a: scalar (optional), prefactor to go in front of the operator in the exponential: exp(a*O)

* start:  scalar (optional) specify the starting point for a grid of points to evaluate the matrix exponential at. see below.

* stop: (optional) specify the end point of the grid of points. 

* num: (optional) specify the number of grid points between start and stop (Default if 50)

* endpoint: (optional) if True this will make sure stop is included in the set of grid points (Note this changes the grid step size).

* iterate: (optional) if True when mathematical methods are called they will return iterators which will iterate over the grid points as opposed to producing a list of all the evaluated points. This is more memory efficient but at the sacrifice of speed.

--- exp_op attributes ---: '_. ' below stands for 'object. '

* _.a: returns the prefactor a

* _.ndim: returns the number of dimensions, always 2

* _.H: returns the hermitian conjugate of this operator

* _.T: returns the transpose of this operator

* _.O: returns the operator which is being exponentiated

* _.get_shape: returns the tuple which contains the shape of the operator

* _.iterate: returns a bool telling whether or not this function will iterate over the grid of values or return a list

* _.grid: returns the array containing the grid points the exponential will be evaluated at

* _.step: returns the step size between the grid points

####**Methods of exp_op Class**

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
 
* return matrix at given grid point:
 ```python
   expO_mat = expO.get_mat(index=None, time=0)
 ```
 
 ####**mathematical functions**
* dot product:
 ```python
   B = expO.dot(A,time=0)
   B = expO.rdot(A,time=0)
 ```
 
* rotate operator by ```expO```:
 ```python
   B = expO.sandwich(copy=False)
 ```

here time is always a scalar which is the time point at which operator ```O``` is evaluated at for dynamic hamiltonians, other wise for matrices or static hamiltonians this does not do anything.  If ```iterate=True``` all these functions return generators which return values of the results over the grid points. for example:

```python
expO.set_iterate(True)
B_generator = expO.dot(A)

for B in B_generator:
	# code here
```
 
###**HamiltonianOperator Class**


```python
H_operator = HamiltonianOperator(operator_list,system_arg,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args)
```

This class uses the basis.Op function to calculate the matrix vector product on the fly greating reducing the amount of memory needed for a calculation at the cost of speed. This object is useful for doing large scale Lanczos calculations using eigsh. 

--- arguments ---

* static_list: (compulsory) list of operator strings to be used for the HamiltonianOperator. The format goes like:

```python
operator_list=[[opstr_1,[indx_11,...,indx_1m]],...]
```
		
* system_arg: int/basis_object (compulsory) number of sites to create basis object/basis object.

* check_symm: bool (optional) flag whether or not to check the operator strings if they obey the given symmetries.

* check_herm: bool (optional) flag whether or not to check if the operator strings create hermitian matrix. 

* check_pcon: bool (optional) flag whether or not to check if the operator string whether or not they conserve magnetization/particles. 

* dtype: dtype (optional) data type to case the matrices with. 

* kw_args: extra options to pass to the basis class.

--- hamiltonian attributes ---: '_. ' below stands for 'object. '

* _.ndim: number of dimensions, always 2.
		
* _.Ns: number of states in the hilbert space.

* _.shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

* _.dtype: returns the data type of the hamiltonian

* _.operator_list: return the list of operators given to this  

* _.T: return the transpose of this operator

* _.H: return the hermitian conjugate of this operator

* _.basis: return the basis used by this operator

* _.LinearOperator: returns a linear operator of this object

####**Method of HamiltonianOperator Class**
* dot product from left and right:

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
 


## **basis Objects**

The basis objects provide a way of constructing all the necessary information needed to construct a sparse matrix from a list of operators. All basis objects are derived from the same base object class and have mostly the same functionality. There are two subtypes of the base object class. The first type consists of basis objects which provide the bulk operations required to create a sparse matrix out of an operator string. For example, basis type one creates a single spin-chain basis. The second basis type wraps multiple objects of the first type together in a tensor-style basis type. For instance, basis type two can take two spin-chain bases and create the corresponding tensor product basis our of them.   

###**1d_spin_basis**
The spin_basis_1d class provides everything necessary to create a hamiltonian of a spin-1/2 system in 1d. The available operators one can use are the the standard spin operators: ```x,y,z,+,-``` which either represent the Pauli operators (matrices) or spin-1/2 operators. The ```+/-``` operators are always constructed as ```x +/- i y```.


It is also possible to create the hamiltonian in a given symmetry-reduced block as follows:

* magnetization symmetries: 
 *  ```Nup=0,1,...,L # pick single magnetization sector```
 * ```Nup = [0,1,...] # pick list of magnetization sectors```
* parity symmetry: ```pblock = +/- 1```
* spin inversion symmetry: ```zblock = +/- 1```
* (spin inversion)*(parity) symmetry: ```pzblock = +/- 1 ```
* spin inversion on sublattice A (even sites, first lattice site is even): ```zAblock = +/- 1```
* spin inversion on sublattice B (odd sites): ```zAblock = +/- 1```
* translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```

Other optional arguments include:

* pauli: toggle whether or not to use spin-1/2 or Pauli matrices for matrix elements. Default is ```pauli = True```.
* a: the lattice spacing for the translational symmetry. Default is ```a = 1```.

Usage of spin_basis_1d:

```python
basis = spin_basis_1d(L,**symmetry_blocks)
```
---arguements---

* L: int (compulsory) length of chain to construct basis for

* symmetry_blocks: (optional) specify which block of a particular symmetry to construct the basis for. 

--- spin_basis_1d attributes ---: '_. ' below stands for 'object. '

* _.L: returns length of the chain as integer

* _.N: return number of sites in chain as integer

* _.Ns: returns number of states in the hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 






###**harmonic oscillator basis**
This basis implements a single harmonic oscillator mode. The available operators are ```+,-,n,I```, corresponding to the raising, lowering, the number operator, and the identity, respectively.  

usage of ho_basis:

```python
basis = ho_basis(Np)
```

---arguements---

* Np: int (compulsory) highest number state to allow in the Hilbert space.

--- spin_basis_1d attributes ---: '_. ' below stands for 'object. '

* _.Np: returns the highest number state of this ho_basis 

* _.Ns: returns number of states in the Hilbert space

* _.operators: returns string which lists information about the operators of this basis class. 

###**tensor_basis Objects** 

  The tensor_basis class will combine two basis objects b1 and b2 together into a new basis object which can be then used, e.g., to create the tensored hamiltonian of both basis:

```python
basis1 = spin_basis_1d(L,Nup=L/2)
basis2 = spin_basis_1d(L,Nup=L/2)
t_basis = tensor_basis(basis1,basis2)
```

 The syntax for the operator strings is as follows. The operator strings are separated by a '|' while the index array has no splitting character.

```python
# tensoring two z spin operators at site 1 for basis1 and site 5 for basis2
opstr = "z|z" 
indx = [1,5] 
```

if there are no operator strings on either side of the '|' then an identity operator is assumed.

###**photon_basis Class** 

This class allows the user to define a basis which couples to a single photon mode. The operators for the photon sector are the same as the harmonic oscilaltor basis: '+', '-', 'n', and 'I'. 

There are two types of basis objects that one can create: a particle (magnetization + photon) conserving basis or a non-conserving basis. 

In the conserving case one can specify the total number of quanta using the the Ntot keyword argument:

```python
p_basis = photon_basis(basis_class,*basis_args,Ntot=...,**symmetry_blocks)
```

For the non-conserving basis, one must specify the total number of photon (a.k.a harmonic oscillator) states with Nph:

```python
p_basis = photon_basis(basis_class,*basis_args,Nph=...,**symmetry_blocks)
```
Here because the the nature of the interaction between the photon mode and the other basis, you must pass the constructor of the basis class into here as opposed to an already constructed basis. This is because the basis has to be constructed for each magnetization/particle sector of the basis. 

###**Checks on Operator Strings**
New in version 0.1.0 we have included new functionality classes which check various properties of a given static and dynamic operator lists. They include the following:

* check if the final complete list of opertors obeys the requested symmetry of that basis. The check can be turned off with the flag ```check_symm=False ``` in the [hamiltonian](#hamiltonian-objects) class. 
* check if the final complete list of operators are hermitian. The check can be turned out with the flag ```check_herm=False``` in the [hamiltonian](#hamiltonian-objects) class. 
* check if the final complete list of opertors obeys particle number conservation (for spin systems this means that magnetization sectors do not mix). The check can be turned out with the flag ```check_pcon=False``` in the [hamiltonian](#hamiltonian-class) class. 

###**Methods of basis Classes**

These functions are defined for every basis class:

```python
basis.Op(opstr,indx,J,dtype)
```
This function takes the string of operators and the sites on which they act, and returns the matrix elements, their row index and column index in the Hamiltonian matrix for the symmetry sector the basis was initialized with.
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
This function converts a state defined in the symmetry-reduced basis to the full basis.
---arguments---

* v: two options
  1. 1-dim array which contains the state
  2. 2-dim array which contains multiple states in the columns
  
RETURNS:
state or states in the full basis as columns of the returned array.

```python
basis.get_proj(dtype)
```
This function returns the transformation from the symmetry-reduced basis to the full basis
---arguments---

* dtype: data type to cast projector matrix in. 

RETURNS:
projector to the full basis as a sparse matrix. 

## **Tools**
###**Measurements** 
#### **entanglement entropy**

```python
ent_entropy(system_state,basis,chain_subsys=None,densities=True,subsys_ordering=True,alpha=1.0,DM=False,svd_return_vec=[False,False,False])
```

This function calculates the entanglement entropy of a lattice quantum subsystem based on the Singular
Value Decomposition (svd).

RETURNS:  dictionary with keys:

* 'Sent': entanglement entropy.

* 'DM_chain_subsys': (optional) reduced density matrix of chain subsystem. The basis in which
  the reduced DM is returned is the full z-basis of the subsystem. For instance, if the subsystem contains N_A sites
  the reduced DM will be a (2^{N_A}, 2^{N_A}) array. This is required because some symmrtries of the system
  might not be inherited by the subsystem. The only exception to this appears when 'basis' is an instance of 'photon_basis'
  AND the subbsystem is the entire chain (i.e. one traces out the photon dregree of freedom only): then the reduced DM is returned in the basis specified by the '..._basis_1d' argument passed into the definition of 'photon_basis', and thus inherits all symmetries of '..._basis_1d' by construction.

* 'DM_other_subsys': (optional) reduced density matrix of the complement subsystem. The basis the redcuded DM is returned in, is the     same as 'DM_chain_subsys' above.

* 'U': (optional) svd U matrix

* 'V': (optional) svd V matrix

* 'lmbda': (optional) svd singular values

 --- arguments ---

* system_state: (required) the state of the quantum system. Can be a:

  1. pure state [numpy array of shape (Ns,)].

  2. density matrix (DM) [numpy array of shape (Ns,Ns)].

  3. diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
    rho_d [numpy array of shape (Ns,)] and its eigenbasis in the columns of V_rho
    [numpy arary of shape (Ns,Ns)]. The keys CANNOT be chosen arbitrarily.].

  4. a collection of states [dictionary {'V_states':V_states}] containing the states
    in the columns of V_states [shape (Ns,Nvecs)]

* basis: (required) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
  'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

* chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

  * [0,1,...,N/2-1,N/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

  * [0,1,...,N-1,N] for 'photon_basis'.

* DM: (optional) String to enable the calculation of the reduced density matrix. Available options are

  * 'chain_subsys': calculates the reduced DM of the subsystem 'chain_subsys' and
    returns it under the key 'DM_chain_subsys'.

  * 'other_subsys': calculates the reduced DM of the complement of 'chain_subsys' and
    returns it under the key 'DM_other_subsys'.

  * 'both': calculates and returns both density matrices as defined above.

  Default is 'False'.   

* alpha: (optional) Renyi alpha parameter. Default is '1.0'. When alpha is different from unity,
        the entropy keys have attached '_Renyi' to their label.

* densities: (optional) if set to 'True', the entanglement entropy is normalised by the size of the
        subsystem [i.e., by the length of 'chain_subsys']. Detault is 'True'.

* subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.

* svd_return_vec: (optional) list of three booleans to return Singular Value Decomposition (svd) 
  parameters:

  * [True, . , . ] returns the svd matrix 'U'.

  * [ . ,True, . ] returns the singular values 'lmbda'.

  * [ . , . ,True] returns the svd matrix 'V'.

  Any combination of the above is possible. Default is [False,False,False].




#### **diagonal ensemble observables**
```python
diag_ensemble(N,system_state,E2,V2,densities=True,alpha=1.0,rho_d=False,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,Srdm_args=())
```

This function calculates the expectation values of physical quantities in the Diagonal ensemble 
set by the initial state (see eg. arXiv:1509.06411). Equivalently, these are the infinite-time 
expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2.


RETURNS:  dictionary with keys depending on the passed optional arguments:

replace "..." below by 'pure', 'thermal' or 'mixed' depending on input params.

* 'Obs_...': infinite time expectation of observable 'Obs'.

* 'delta_t_Obs_...': infinite time temporal fluctuations of 'Obs'.

* 'delta_q_Obs_...': infinite time quantum fluctuations of 'Obs'.

* 'Sd_...' ('Sd_Renyi_...' for alpha!=1.0): Renyi entropy of density matrix of Diagonal Ensemble with parameter 'alpha'.

* 'Srdm_...' ('Srdm_Renyi_...' for alpha!=1.0): Renyi entropy of reduced density matrix of Diagonal Ensemble with parameter 'alpha'.

* 'rho_d': density matrix of diagonal ensemble


--- arguments ---


* N: (required) system size N.

* system_state: (required) the state of the quantum system. Can be a:

  1. pure state [numpy array of shape (Ns,) or (,Ns)].

  2. density matrix (DM) [numpy array of shape (Ns,Ns)].

  3. mixed DM [dictionary] {'V1':V1,'E1':E1,'f':f,'f_args':f_args,'V1_state':int,'f_norm':False} to 
    define a diagonal DM in the basis 'V1' of the Hamiltonian H1. The keys are

    * 'V1': (required) array with the eigenbasis of H1 in the columns.

    * 'E1': (required) eigenenergies of H1.

    * 'f': (optional) the distribution used to define the mixed DM. Default is
      'f = lambda E,beta: numpy.exp(-beta*(E - E[0]) )'. 

    * 'f_args': (required) list of arguments of function 'f'. If 'f' is not defined, by 
	  efault we have 'f=numpy.exp(-beta*(E - E[0]))', and 'f_args' specifies the inverse temeprature list [beta].

    * 'V1_state' (optional) : list of integers to specify the states of 'V1' wholse pure 
      expectations are also returned.

    * 'f_norm': (optional) if set to 'False' the mixed DM built from 'f' is NOT normalised
      and the norm is returned under the key 'f_norm'. 

    The keys are CANNOT be chosen arbitrarily.

* V2: (required) numpy array containing the basis of the Hamiltonian H2 in the columns.

* E2: (required) numpy array containing the eigenenergies corresponding to the eigenstates in 'V2'.
  This variable is only used to check for degeneracies.

* rho_d: (optional) When set to 'True', returns the Diagonal ensemble DM under the key 'rho_d'. 

* Obs: (optional) hermitian matrix of the same size as V2, to calculate the Diagonal ensemble 
  expectation value of. Appears under the key 'Obs'.

* delta_t_Obs: (optional) TIME fluctuations around infinite-time expectation of 'Obs'. Requires 'Obs'. 
  Appears under the key 'delta_t_Obs'.

* delta_q_Obs: (optional) QUANTUM fluctuations of the expectation of 'Obs' at infinite-times. 
  Requires 'Obs'. Appears under the key 'delta_q_Obs'. Returns temporal fluctuations 
  'delta_t_Obs' for free.

* Sd_Renyi: (optional) diagonal Renyi entropy in the basis of H2. The default Renyi parameter is 
  'alpha=1.0' (see below). Appears under the key Sd_Renyi'.

* Srdm_Renyi: (optional) entanglement Renyi entropy of a subsystem of a choice. The default Renyi 
  parameter is 'alpha=1.0' (see below). Appears under the key Srdm_Renyi'. Requires 
  'Srdm_args'. To specify the subsystem, see documentation of '_reshape_as_subsys'.

* Srdm_args: (optional) dictionary of ent_entropy arguments, required when 'Srdm_Renyi = True'. The 
following keys are allowed:

  1. basis: (required) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
    'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

  2. chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

   * [0,1,...,N/2-1,N/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

   * [0,1,...,N-1,N] for 'photon_basis'. 

   * subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.

* densities: (optional) if set to 'True', all observables are normalised by the system size N, except
  for the entanglement entropy which is normalised by the subsystem size 
  [i.e., by the length of 'chain_subsys']. Detault is 'True'.

* alpha: (optional) Renyi alpha parameter. Default is '1.0'.




####**project operator**
```python
project_op(Obs,proj,dtype=_np.complex128):
```
This function takes an observable 'Obs' and a reduced basis 'reduced_basis' and projects 'Obs' onto that reduced basis.

RETURNS:  dictionary with keys 

* 'Proj_Obs': projected observable 'Obs'.
	
--- arguments ---

* Obs: (required) operator to be projected.

* proj: (required) basis of the final space after the projection or a matrix which contains the projector.

* dtype: (optional) data type. Default is np.complex128.






#### **Kullback-Leibler divergence**
```python
KL_div(p1,p2)
```
This routine returns the Kullback-Leibler divergence of the discrete probabilities p1 and p2. 







####**time evolution**
```python
ED_state_vs_time(psi,E,V,times,iterate=False):
```
This routine calculates the time evolved initial state as a function of time. The initial state is 'psi' and the time evolution is carried out under the Hamiltonian H.

RETURNS:  either a matrix with the time evolved states as rows, or an iterator which generates the states one by one.

--- arguments --- 

* psi: (required) initial state.

* V: (required) unitary matrix containing in its columns all eigenstates of the Hamiltonian H. 

* E: (required) array containing the eigenvalues of the Hamiltonian H. 
			The order of the eigenvalues must correspond to the order of the columns of V. 

* times: (required) an array of times to evaluate the time evolved state at. 

* iterate: (optional) if True this function returns the generator of the time evolved state. 


```python
obs_vs_time(psi_t,times,Obs_dict,return_state=False,Sent_args={})
```
This routine calculate the expectation value as a function of time of an observable Obs. The initial state is psi and the time evolution is carried out under the Hamiltonian H2. Returns a dictionary in which the time-dependent expectation value has the key 'Expt_time'.

RETURNS:  dictionary with keys:

* 'custom_name' (same as the keys of 'Obs_dict'). For each key of 'Obs_dict', the time-dependent expectation of the observable 'Obs_dict[key]' is calculated and returned.

* 'psi_t': (optional) returns a 2D array the columns of which give the state at the associated time.

* 'Sent_time': (optional) returns the entanglement entropy of the state at time 'times'.


--- arguments ---

* psi_t: (required) Source of time dependent states, three different types of inputs:


 1. psi_t: tuple(psi, E, V)  
	* psi [1-dim array]: initial state 
	* V [2-dim array]: unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. 
	* E [1-dim array]: real vector containing the eigenvalues of the Hamiltonian. The order of the eigenvalues must correspond to the order of the columns of V2.
 2. psi_t: 2-dim array which contains the time dependent states as columns of the array.
 3. psi_t:  Iterator generates the states sequentially ( For most evolution functions you can get this my setting ```iterate=True```. This is more memory efficient as the states are generated on the fly as opposed to being stored in memory )

* times: (required) a real array of times to evaluate the expectation value at. always fifth argument. If this is specified, the hamiltonian objects will be dynamically evaluated at the times specified. The function will also 

* Obs_dict: (required) Dictionary of objects to take the expecation values with. This accepts NumPy, and SciPy matrices as well as hamiltonian objects.

* return_state: (optional) when set to 'True' or 'Sent_args' is nonempty, returns a matrix whose columns give the state vector at the times specified by the row index. The return dictonary key is 'psi_time'.

* Sent_args: (optional) when non-empty, this dictionary of ent_entropy arguments enables the calculation of 'Sent_time'. The 
following keys are allowed:

  1. basis: (required) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
    'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

  2. chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

   * [0,1,...,N/2-1,N/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

   * [0,1,...,N-1,N] for 'photon_basis'. 

   * subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.



### **mean level spacing**
```python
mean_level_spacing(E)
```

This routine calculates the mean-level spacing 'r_ave' of the energy distribution E, see arXiv:1212.5611.

RETURNS: float with mean-level spacing 'r_ave'.

--- arguments ---

* E: (required) ordered list of ascending, nondegenerate eigenenergies.





###**Floquet**
This package contains tools which contains tools which can be helpful in simulating Floquet systems. 

####**Floquet class**

```python
floquet = Floquet(evo_dict,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1)
```
Calculates the Floquet spectrum for a given protocol, and optionally the Floquet hamiltonian matrix, and Floquet eigen-vectors.

--- arguments ---

* evo_dict: (required) dictionary which passes the different types of protocols to calculate evolution operator:


 1. Continuous protocol.

  * 'H': (required) hamiltonian object to generate the time evolution. 

  * 'T': (required) period of the protocol. 

  * 'rtol': (optional) relative tolerance for the ode solver. (default = 1E-9)

  * 'atol': (optional) absolute tolerance for the ode solver. (default = 1E-9)

 2. Step protocol from a hamiltonian object. 

  * 'H': (required) hamiltonian object to generate the hamiltonians at each step.
				
  * 't_list': (required) list of times to evaluate the hamiltonian at when doing each step.

  * 'dt_list': (required) list of time steps for each step of the evolution. 

 3. Step protocol from a list of hamiltonians. 

  * 'H_list': (required) list of matrices which to evolve with.

  * 'dt_list': (required) list of time steps to evolve with. 

 * HF: (optional) if set to 'True' calculate Floquet hamiltonian. 


* UF: (optional) if set to 'True' save evolution operator. 

* ThetaF: (optional) if set to 'True' save the eigenvalues of the evolution operator. 

		* VF: (optional) if set to 'True' save the eigenvectors of the evolution operator. 

* n_jobs: (optional) set the number of processors which are used when looping over the basis states. 

--- Floquet attributes ---: '_. ' below stands for 'object. '

Always given:

* _.EF: Floquet qausi-energies

Calculate via flags:

* _.HF: Floquet Hamiltonian dense array

* _.UF: Evolution operator

* _.VF: Floquet eigenstates

* _.thetaF: eigenvalues of evolution operator

####**Floquet_t_vec**
```python
tvec = Floquet_t_vec(Omega,N_const,len_T=100,N_up=0,N_down=0)
```
Returns a time vector (np.array) which hits the stroboscopic times, and has as attributes their indices. The time vector can be divided in three regimes: ramp-up, constant and ramp-down.

--- arguments ---

* Omega: (required) drive frequency

* N_const: (required) # of time periods in the constant period

* N_up: (optional) # of time periods in the ramp-up period

* N_down: (optional) # of time periods in the ramp-down period

* len_T: (optional) # of time points within a period. N.B. the last period interval is assumed  open on the right, i.e. [0,T) and the point T does not go into the definition of 'len_T'. 


--- Floquet_t_vec attributes ---: '_. ' below stands for 'object. '


* _.vals: time vector values

* _.i: initial time value

* _.f: final time value

* _.tot: total length of time: t.i - t.f 

* _.T: period of drive

* _.dt: time vector spacing

* _.len: length of total time vector

* _.len_T: # of points in a single period interval, assumed half-open: [0,T)

* _.N: total # of periods

--- strobo attribues ---

* _.strobo.vals: strobosopic time values

* _.strobo.inds: strobosopic time indices

--- regime attributes --- (available if N_up or N_down are parsed)


* _.up : referes to time vector of up-regime; inherits the above attributes (e.g. _up.strobo.inds) except _.T, _.dt, and ._lenT

* _.const : referes to time vector of const-regime; inherits the above attributes except _.T, _.dt, and ._lenT

* _.down : referes to time vector of down-regime; inherits the above attributes except _.T, _.dt, and ._lenT

This object also acts like an array, you can iterate over it as well as index the values.

