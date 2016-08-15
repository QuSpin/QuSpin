#**qspin**
qspin is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state of the art exact diagonalization calculations on 1 dimensional spin 1/2 chains with lengths up to 32 sites. The interface allows the user to define any spin 1/2 Hamiltonian which can be constructed from spin operators; while also allowing the user flexibility of accessing many symmetries in 1d. There is also a way of specifying the time dependence of operators in the Hamiltonian as well, which can be used to solve the time dependent Schrodinger equation numerically for these systems. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access the powerful linear algebra tools. 




Contents
--------
* [Installation](#installation)
* [using the package](#using-the-package)
 * [constructing hamiltonians](#constructing-hamiltonians)
 * [using basis objects](#using-basis-objects)
 * [using symmetries](#using-symmetries)
* [hamiltonian objects](#hamiltonian-objects)
 * [normal operations](#normal-operations)
 * [quantum operations](#quantum-operations)
 * [other operations](#other-operations)
* [basis objects](#basis-objects)
 * [1d Spin Basis](#1d-spin-basis)
 * [Harmonic Oscillator basis](#harmonic-oscillator-basis)
 * [Tensor basis objects](#tensor-basis-objects)
 * [Methods of Basis Classes](#methods-of-basis-classes)
* [Tools](#tools)
 * [Observables](#observables)
 * [Floquet](#floquet)


#**Installation**
This latest version of this package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux and OS-X and windows 64-bit systems. In order to install this you need to get Anaconda package manager for python. Then all one has to do to install is run:

```
$ conda install -c weinbe58 qspin
```

This will install the latest version on to your computer. Right now the package is in its beta stages and so it may not be availible for installation on all platforms using this method. In this case one can also manually install the package:


1) Manual install: to install manually download source code either by downloading the [master](https://github.com/weinbe58/qspin/archive/master.zip) branch or cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

unix:
```
python setup.py install 
```

or windows command line:
```
setup.py install
```
For the manual installation you must have all the prerequisite python packages: [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), and [joblib](https://pythonhosted.org/joblib/) installed. We recommend [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) to manage your python packages. 

to update the package with conda all one has to do is run the installation command again, but for a manual install you must delete the package manually from the python 'site-packages/' directory. When installing the package if you add the flag ```--record install.txt```, the location of all the installed files will be output to install.txt which will tell you which files to delete when you want to upgrade the code. In unix the following command is sufficient to completely remove the installed files: ```cat install.txt | xargs rm -rf```, while for windows it is easiest to just go to the folder and delete it from windows explorer. 


#**using the package**
All of the calculations done with our package happen through the [hamiltonians](#hamiltonian-objects). The hamiltonian is a type which uses Numpy and Scipy matrices to store the Quantum Hamiltonian. Time independent operators are summed together into a single  static matrix while all time dependent operators are stored separatly along with the time dependent coupling in front of it. When needed, the time dependence is evaluated on the fly for doing calculations that would involve time dependent operators. The user can initialize the hamiltonian types with Numpy arrays or Scipy matrices. Beyond this we have also created an representation which allows the user to construct the matrices for many-body operators. 

Many-body operators are represented by string of letters representing the type of operators and a tuple which holds the indices for the sites that each operator acts at on the lattice. For example, in a spin system we can represent multi-spin operators as:

|      opstr       |      indx      |        operator      |
|:----------------:|:--------------:|:---------------------------:|
|"o<sub>1</sub>...o<sub>n</sub>"|[J,i<sub>1</sub>,...,i<sub>n</sub>]|J S<sub>i<sub>1</sub></sub><sup>o<sub>1</sub></sup>...S<sub>i<sub>n</sub></sub><sup>o<sub>n</sub></sup>|

where o<sub>i</sub> can be x, y, z, +, or -. This gives the full range of possible spin operators that can be constructed. For different systems there are different types of operators. To see the available operators for a given type of system checkout out the [basis](basis-objects) classes. 

###**Constructing hamiltonians**
The hamiltonian is constructed as:
```python
H = hamiltonian(static_list,dynamic_list,**kwargs)
```
where the static_list and dynamic_list are lists have the following format for many-body operators:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],...]
```
To use Numpy arrays or Scipy matrices the syntax is:

```python
static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
```
For the dynamic list the ```func``` is the function which goes in front of the matrix or operator given in the same list. ```func_args``` is a tuple of the extra arguements which go into the function to evaluate it like: 
```python
f_val = func(t,*func_args)
```

####keyword arguements (kwargs):
the ```**kwargs``` give extra information about the hamiltonian. There are different things one can input for this and which one depends on what object you would like to create. They are used to specify symmetry blocks, give a shape and provide the floating point type to store the matrix elements with.

**providing a shape:**
If there are many-body operators one must either specify the number of sites with ```N=...``` or pass in a basis object as ```basis=...```, more about basis objects later [section](#basis-objects). You can also specify the shape using the ```shape=...``` keyword argument. For input lists which contain matrices only, the shape does not have to be specified. If empty lists are given, then either one of the previous options must be provided to the hamiltonian constructor.  

**Numpy dtype:**
The user can specify the numpy data type ([dtype](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.html)) to store the matrix elements with. It supports float32, float64, float128, complex64, and complex128, complex256. The default type is complex128. To specify the dtype use the dtype keyword arguement:

```python
H=hamiltonian(...,dtype=np.float32,...)
```
Note that not all platforms as well as all of Scipy and Numpy functions support dtype float128 and complex256.


**example:** 
constructing a hamiltonian object of the transverse field ising model with time dependent field for 10 site chain:

```python
# python script
from qspin.hamiltonian import hamiltonian

L=10
v=0.01

def drive(t,v):
  return v*t
  
drive_args=[v]

ising_indx=[[-1.0,i,(i+1)%L] for i in xrange(L)]
field_indx=[[-1.0,i] for i in xrange(L)]
static_list=[['zz',ising_indx],['x',field_indx]]
dynamic_list=[['x',field_indx,drive,drive_args]]

H=hamiltonian(static_list,dynamic_list,N=L,dtype=np.float64)
```
Here is an example of a 3 spin operator as well:

```python
op_indx=[[1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)]
op_indx_cc=[[-1.0j,i,(i+1)%L,(i+2)%L] for i in xrange(L)]

static_list=[['-z+',op_indx],['+z-',op_indx_cc]]
```
Notice that I need to include both '-z+' and '+z-' operators to make sure our Hamiltonian is hermitian. 

###**Using basis objects**

Basis objects are another type included in this package which provide all of the functionality which calculates the matrix elements from the operator string representation of the many-body operators. On top of this, some of them have been programmed to calculate the matrix elements in different symmetry blocks of the many-body hamiltonian. to use a basis object to construct the hamiltonian just use the basis keyword argument:
```python
H = hamiltonian(static_list,dynamic_list,...,basis=basis,...)
```
More information about basis objects can be found in the [basis objects](#basis-objects) section.  

###**Using symmetries**
Adding symmetries is easy, either you can just add extra keyword arguments to the initialization of your hamiltonian or when you initialize a basis object. By default the hamiltonian will use the spin-1/2 operators as well as 1d-symmetries. At this point the only symmetries implemented are for spins-1/2 operators in 1 dimension. 
The symmetries for a spin chain in 1d are:

* Magnetization symmetries: 
 *  ```Nup=0,1,...,L # pick single magnetization sector```
 * ```Nup = [0,1,...] # pick list of magnetization sectors```
* Parity symmetry: ```pblock = +/- 1```
* Spin Inversion symmetry: ```zblock = +/- 1```
* (Spin Inversion)*(Parity) symmetry: ```pzblock = +/- 1 ```
* Spin inversion on sublattice A (even sites): ```zAblock = +/- 1```
* Spin inversion on sublattice B (odd sites): ```zAblock = +/- 1```
* Translational symmetry: ```kblock = ...,-1,0,1,.... # all integers available```
used like:
```python
H = hamiltonian(static_list,dynamic_list,L,Nup=Nup,pblock=pblock,...)
```
If the user passes the symmetries into the hamiltonian constructor, the constructor creates a [spin_basis_1d](spin-basis-1d) object for the given symmetries and then uses that object to construct the matrix elements. because of this, If one is constructing multiply hamiltonian objects within the same symmetry block it is more efficient to first construct the basis object and then use the basis object to construct the different hamiltonians:

```python

basis = spin_basis_1d(L,Nup=Nup,pblock=pblock,...)
H1 = hamiltonian(static1_list,dynamic1_list,basis=basis)
H2 = hamiltonian(static2_list,dynamic2_list,basis=basis)
...
```

**NOTE:** for beta versions spin_basis_1d is named as basis1d.

#**hamiltonian objects**

##**Class organization**:
```python
H = hamiltonian(static_list,dynamic_list,N=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs)
```

The hamiltonian class wraps most of the functionalty of the package. Below shows the initialization arguements:

--- arguments ---

* static_list: (compulsory) list of objects to calculate the static part of hamiltonian operator. The format goes like:

 ```python
 static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
 ```
	

* dynamic_list: (compulsory) list of objects to calculate the dynamic part of the hamiltonian operator.The format goes like:

 ```python
 dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args] [matrix_2,func_2,func_2_args],...]
 ```

 For the dynamic list the ```func``` is the function which goes in front of the matrix or operator given in the same list. ```func_args``` is a tuple of the extra arguements which go into the function to evaluate it like: 
 ```python
 f_val = func(t,*func_args)
 ```


* N: (optional) number of sites to create the hamiltonian with.

* shape: (optional) shape to create the hamiltonian with.

* copy: (optional) weather or not to copy the values from the input arrays. 

* check_symm: (optional) flag whether or not to check the operator strings if they obey the given symmetries.

* check_herm: (optional) flag whether or not to check if the operator strings create hermitian matrix. 

* check_pcon: (optional) flag whether or not to check if the oeprator string whether or not they conserve magnetization/particles. 

* dtype: (optional) data type to case the matrices with. 

* kw_args: extra options to pass to the basis class.

--- hamiltonian attributes ---: '_. ' below stands for 'object. '

* _.ndim: number of dimensions, always 2.
		
* _.Ns: number of states in the hilbert space.

* _.get_shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

* _.is_dense: return 'True' if the hamiltonian contains a dense matrix as a componnent. 

* _.dtype: returns the data type of the hamiltonian

* _.static: return the static part of the hamiltonian 

* _.dynamic: returns the dynamic parts of the hamiltonian 




###**normal operations**
The hamiltonian objects currectly support certain arithmetic operations with other hamiltonians as well as scipy sparse matrices and numpy dense arrays and scalars:

* between other hamiltonians we have: ```+,-,*,+=,-=``` . Note that ```*``` only works between a static and static hamiltonians or a static and dynamic hamiltonians.
* between numpy and sparse square arrays we have: ```*,+,-,*=,+=.-=``` (versions >= v0.0.5b)
* between scalars: ```*,*=``` (versions >= v0.0.5b)
* negative operator '-H'
* indexing and slicing: ```H[times,row,col]``` 

###**quantum operations**
We've included some basic functionality into the hamiltonian class useful for quantum calculations:

* matrix vector product / dense matrix:

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

* project to new basis:
usage:
	```python
	H_new = H.project_to(V)
	```
returns a new hamiltonian object which is: V<sup>+</sup> H V. Note that V need not be a square matrix.
  
* matrix exponential multiplication (versions >= v0.1.0):
  usage:
	```
	v = H.expm_multiply(u,a=-1j,time=0,times=(),iterate=False,**linspace_args)
	```
 * u: (compulsory) vector to act on
 * a: (optional) factor to multiply H by in the exponential
 * time: (optional) time to evaluate H at
 * times: (optional) lines of times to exponentiate to
 * iterate: (optional) bool to return generate or list of vectors
 * linspace_args: (optional) arguements to pass into expm_multiply
	
  which evaluates |v > = exp(a H(time))|u > using only the dot product of H on |u >. The extra arguments will allow one to evaluate it at different time points see scipy docs for [expm_multiply](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply) for more information. If iterate is True the function returns a generate which yields a vector evaluated at the time points specified either by time or by the linspace arguments:
	```python
	v_iter = H.expm_multiply(u,a=-1j,time=0,times=(1,2,3,4,5,6),iterate=True)
	for v in v_iter:
		#do stuff with v
		# iterate to next vector
	```
If iterate is false then times is ignored. 

* matrix exponential (versions >= v0.2.0):
  usage:
    ```python
    U = H.expm(a=-1j,time=0)
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
 * v0:  (compulsory) initial state array.
 * t0: (compulsory) initial time.
 * times: (compulsory) a time or generator of times to evolve to.
 * solver_name: (optional) used to pick which Scipy ode solver to use.
 * verbose: (optional) prints out when the solver has evolved to each time in times
 * iterate: (optional) returns vf which is an iterator over the evolution vectors without storing the solution for every time in times. 
 * imag_time: (optional) toggles whether to evolve with __SO or __ISO.
 * solver_args: (optional) the optional arguements which are passed into the solver. The default setup is: nstep = 2**31 - 1 ,atol = 1E-9, rtol = 1E-9
  
The hamiltonian class also has built in methods which are useful for doing ED calculations:

* Full diagonalization:

  usage:
    ```python
    eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)
    eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)
    ```
  where **eigh_args are optional arguements which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh) and [eigvalsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.linalg.eigvalsh.html#scipy.linalg.eigvalsh). 
  
  NOTE: overwrite_a=True always for memory conservation

* Sparse diagonalization, which uses ARPACK:

  usage:
    ```python
    eigenvalues,eigenvectors=H.eigsh(time=time,**eigsh_args)
    ```
  where **eigsh_args are optional arguements which are passed into the eigenvalue solvers. For more information checkout the scipy docs for [eigsh](http://docs.scipy.org/doc/scipy-0.18.0/reference/generated/scipy.sparse.linalg.eigsh.html)

###**other operations**
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



# **Basis objects**

The basis objects provide a way of constructing all the necessary information needed to construct a sparse matrix from a list of operators. There are two different kinds of basis classes at this point in the development of the code, All basis objects are derived from the same base object class and have mostly the same functionality but there are two subtypes of this class. The first are basis objects which provide the bulk operations required to create a sparse matrix out of an operator string. Right now we have included only two of these types of classes. One type of class preforms calculations of hamiltonian matrix elements while the other basis types wrap the first type together in tensor style basis types. 


###**1d Spin Basis**
```python
basis = spin_basis_1d(L,**symmtry_blocks)
```
the spin_basis_1d class provides everything necessary to create a hamiltonian of a spin system in 1d. The available operators one can use are the the standard spin operators: ```x,y,z,+,-``` which either represent the pauli operators or spin 1/2 operators. The ```+,-``` operators are always constructed as ```x +/- i y```.


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

Other arguements which are optional are:

* pauli: toggle whether or not to use spin-1/2 or pauli matrices for matrix elements (default pauli = True).
* a: the lattice spacing for the translational symmetry (default a = 1).



###**Harmonic Oscillator Basis**
This basis implements a single harmonic oscillator mode. The available operators are ```+,-,n```.

###**Tensor Basis Objects**
In version 0.1.0b we have created new classes which allow basis to be tensored together. 

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

	For this basis class you can't pass not a basis object, but the constructor for you basis object. The operators for the photon sector are '+', '-', 'n', and 'I'.

###**Checks on operator strings**
new in version 0.2.0 we have included a new functionality classes which check various properties of a given static and dynamic operator lists. They include the following:

* Checks if complete list of opertors obey the given symmetry of that basis. The check can be turned off with the flag ```check_symm=False ``` in the [hamiltonian](#hamiltonian-objects) class. 
* Checks of the given set of operators are hermitian. The check can be turned out with the flag ```check_herm=False``` in the [hamiltonian](#hamiltonian-objects) class. 
* Checks of the given set of operators obey particle conservation (for spin systems this means magnetization sectors do not mix). The check can be turned out with the flag ```check_pcon=False``` in the [hamiltonian](#hamiltonian-objects) class. 

###**Methods of Basis Classes**
```python
basis.Op(opstr,indx,J,dtype)
```
This function takes the string of operators and index which they act returns the matrix elements, row index and column index of those matrix elements in the Hamiltonian matrix for the symmetry sector the basis was initialized with.
---arguments--- (*all compulsory*)

* opstr: string which contains the operator string.
* indx: 1-dim array which contains the index where the operators act.
* J: scalar value which is the coefficient in front of the operator.
* dtype: the data type the matrix elements should be cast to. 

RETURNS:

* ME: 1-dim array which contains the matrix elements.
* row: 1-dim array containing the row indices of the matrix elements.
* col: 1-dim array containing the column indices of the matrix elements. 

```python
basis.get_vec(v)
```
This function converts a state in the symmetry reduced basis to the full basis.
---arguments---

* v:
  1. 1-dim array which contains the state
  2. 2-dim array which contains multiply states in the columns
  
RETURNS:
state or states in the full Hilbert space.

```python
basis.get_proj(dtype)
```
This function returns the transformation from the symmetry reduced basis to the full basis
---arguments---

* dtype: data type to cast projector matrix in. 

RETURNS:
projector to the full basis as a sparse matrix. 

# **Tools**

##**Observables** 

#### **Entanglement entropy**

```python
Entanglement_Entropy(system_state,basis,chain_subsys=None,densities=True,subsys_ordering=True,alpha=1.0,DM=False,svd_return_vec=[False,False,False])
```
	
This function calculates the entanglement entropy of a lattice quantum subsystem based on the Singular Value Decomposition (svd).


--- arguments ---

* system_state: (compulsory) the state of the quantum system. Can be a:

 1. pure state [1-dim array].

 2. density matrix (DM) [2-dim array].

 3. diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
  * rho_d [numpy array of shape (Ns,)] and its eigenbasis in the columns of V_rho [numpy arary of shape (Ns,Ns)]. The keys are CANNOT be chosen arbitrarily.].

 4. a collection of states [dictionary {'V_states':V_states}] containing the states in the columns of V_states [shape (Ns,Nvecs)]

* basis: (compulsory) the basis used to build 'system_state'. Must be an instance of 'photon_basis', 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

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

* alpha: (optional) Renyi alpha parameter. Default is '1.0'.

* densities: (optional) if set to 'True', the entanglement entropy is normalised by the size of the
				subsystem [i.e., by the length of 'chain_subsys']. Detault is 'True'.

* subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.

* svd_return_vec: (optional) list of three booleans to return Singular Value Decomposition (svd) parameters:

  * [True, . , . ] returns the svd matrix 'U'.

  * [ . ,True, . ] returns the singular values 'lmbda'.

  * [ . , . ,True] returns the svd matrix 'V'.

 Any combination of the above is possible. Default is [False,False,False].

RETURNS:	

dictionary in which the entanglement entropy has the key 'Sent'.






#### **Diagonal Ensemble Observables**
```python
Diag_Ens_Observables(N,system_state,V2,densities=True,alpha=1.0,rho_d=False,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Sent_Renyi=False,Sent_args=())
```

This function calculates the expectation values of physical quantities in the Diagonal ensemble set by the initial state (see eg. arXiv:1509.06411). Equivalently, these are the infinite-time expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2.
--- arguments ---

* N: (compulsory) system size N.

* system_state: (compulsory) the initial state of the quantum system. Can be a:

 1. pure state [1-dim array].

 2. density matrix (DM) [2-dim array].

  3. mixed DM [dictionary]{'V1':V1, 'E1':E1, 'f':f, 'f_args':f_args, 'V1_state':int, 'f_norm':False} to define a diagonal DM in the basis 'V1' of the Hamiltonian H1. The keys are

   * 'V1': (compulsory) array with the eigenbasis of H1 in the columns.

   * 'E1': (compulsory) eigenenergies of H1.

   * 'f': (optional) the distribution used to define the mixed DM. Default is
						'f = lambda E,beta: numpy.exp(-beta*(E - E[0]) )'. 

   * 'f_args': (compulsory) list of arguments of function 'f'. If 'f' is not defined, 
						it specifies the inverse temeprature list [beta].

   * 'V1_state' (optional) : list of integers to specify the states of 'V1' wholse pure 
						expectations are also returned.

   * 'f_norm': (optional) if set to 'False' the mixed DM built from 'f' is NOT normalised
						and the norm is returned under the key 'f_norm'. 

 The keys are CANNOT be chosen arbitrarily.

* V2: (compulsory) numpy array containing the basis of the Hamiltonian H2 in the columns.

* rho_d: (optional) When set to 'True', returns the Diagonal ensemble DM under the key 'rho_d'. 

* Obs: (optional) hermitian matrix of the same size as V2, to calculate the Diagonal ensemble 
			expectation value of. Appears under the key 'Obs'.

* delta_t_Obs: (optional) TIME fluctuations around infinite-time expectation of 'Obs'. Requires 'Obs'. 
			Appears under the key 'delta_t_Obs'.

* delta_q_Obs: (optional) QUANTUM fluctuations of the expectation of 'Obs' at infinite-times. 
			Requires 'Obs'. Appears under the key 'delta_q_Obs'.

* Sd_Renyi: (optional) diagonal Renyi entropy in the basis of H2. The default Renyi parameter is 
			'alpha=1.0' (see below). Appears under the key Sd_Renyi'.

* Sent_Renyi: (optional) entanglement Renyi entropy of a subsystem of a choice. The default Renyi 
			parameter is 'alpha=1.0' (see below). Appears under the key Sent_Renyi'. Requires 
			'Sent_args'. To specify the subsystem, see documentation of 'reshape_as_subsys'.

* Sent_args: (optional) tuple of Entanglement_Entropy arguments, required when 'Sent_Renyi = True'.
			At least 'Sent_args=(basis)' is required. If not passed, assumes the default 'chain_subsys', 
			see documentation of 'reshape_as_subsys'.

* densities: (optional) if set to 'True', all observables are normalised by the system size N, except
				for the entanglement entropy which is normalised by the subsystem size 
				[i.e., by the length of 'chain_subsys']. Detault is 'True'.

* alpha: (optional) Renyi alpha parameter. Default is '1.0'.

RETURNS: 	

dictionary

####**Project Operator**
```python
Project_Operator(Obs,proj,dtype=_np.complex128):
```
This function takes an observable 'Obs' and a reduced basis 'reduced_basis' and projects 'Obs' onto the reduced basis.

	
--- arguments ---

* Obs: (compulsory) operator to be projected.

* proj: (compulsory) basis of the final space after the projection or a matrix which contains the projector.

* dtype: (optional) data type. Default is np.complex128.

RETURNS: 	

* dictionary with keys 'Proj_Obs' and value the projected observable.






#### **Kullback-Leibler divergence**
```python
Kullback_Leibler_div(p1,p2)
```
This routine returns the Kullback-Leibler divergence of the discrete probabilities p1 and p2. 







####**Time Evolution**
```python
ED_state_vs_time(psi,V,E,times,iterate=False):
```
This routine calculates the time evolved initial state as a function of time. The initial state is 'psi' and the time evolution is carried out under the Hamiltonian H. 
--- arguments --- 

* psi: (compulsory) initial state.

* V: (compulsory) unitary matrix containing in its columns all eigenstates of the Hamiltonian H. 

* E: (compulsory) array containing the eigenvalues of the Hamiltonian H. 
			The order of the eigenvalues must correspond to the order of the columns of V. 

* times: (compulsory) an array of times to evaluate the time evolved state at. 

* iterate: (optional) if True this function returns the generator of the time evolved state. 

RETURNS:

* Returns a matrix with the time evolve states in the columns, or an iterator which generates the states in sequence.



```python
Observable_vs_time(psi_t,Obs_list,times=None,return_state=False)
```
This routine calculate the expectation value as a function of time of an observable Obs. The initial state is psi and the time evolution is carried out under the Hamiltonian H2. Returns a dictionary in which the time-dependent expectation value has the key 'Expt_time'.
--- arguements ---

* psi_t: (compulsory) Source of time dependent states, three different types of inputs:


 1. psi_t: tuple(psi, E, V, times)  
	* psi [1-dim array]: initial state 
	* V [2-dim array]: unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. 
	* E [1-dim array]: real vector containing the eigenvalues of the Hamiltonian. The order of the eigenvalues must correspond to the order of the columns of V2.
	* times: list or array of times to evolve to.
 2. psi_t: 2-dim array which contains the time dependent states as columns of the array.
 3. psi_t:  Iterator generates the states sequentially ( For most evolution functions you can get this my setting ```iterate=True```. This is more memory efficient as the states are generated on the fly as opposed to being stored in memory )


* Obs_list: (compulsory) List of objects to take the expecation values with. This accepts NumPy, and SciPy matrices as well as hamiltonian objects.

* times: (optional) a real array of times to evaluate the expectation value at. always fifth argument. If this is specified, the hamiltonian objects will be dynamically evaluated at the times specified. The function will also 

* return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector at the times specified by the row index. The return dictonary key is 'psi_time'.

RETURNS:

* output: dictionary
 * 'Expt_time' array 2-dimensions: contains the time dependent expectation values of the operators. The row index is the time index and the column index in the index of the observable in Obs_list. 
 * 'psi_time' array 2-dimensional: contains the states evaluated at each of the time points, where the column index refers to the time index. 

### **Mean Level spacing**
```python
Mean_Level_Spacing(E)
```
This routine returns the mean-level spacing r_ave of the energy distribution E, see [arXiv:1212.5611](http://arxiv.org/abs/1212.5611). 

E: (compulsory) ordered list of ascending, nondegenerate eigenenergies. 

###**Floquet**
This package contains tools which contains tools which can be helpful in simulating Floquet systems. 

####**Floquet class**

```python
floquet = Floquet(evo_dict,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1)
```
Calculates the Floquet spectrum for a given protocol, and optionally the Floquet hamiltonian matrix, and Floquet eigen-vectors.

--- arguments ---

* evo_dict: (compulsory) dictionary which passes the different types of protocols to calculate evolution operator:

 1. Continuous protocol.

  * 'H': (compulsory) hamiltonian object to generate the time evolution. 

  * 'T': (compulsory) period of the protocol. 

  * 'rtol': (optional) relative tolerance for the ode solver. (default = 1E-9)

  * 'atol': (optional) absolute tolerance for the ode solver. (defauly = 1E-9)

 2. Step protocol from a hamiltonian object. 

  * 'H': (compulsory) hamiltonian object to generate the hamiltonians at each step.
				
  * 't_list': (compulsory) list of times to evaluate the hamiltonian at when doing each step.

  * 'dt_list': (compulsory) list of time steps for each step of the evolution. 

 3. Step protocol from a list of hamiltonians. 

  * 'H_list': (compulsory) list of matrices which to evolve with.

  * 'dt_list': (compulsory) list of time steps to evolve with. 

 * HF: (optional) if set to 'True' calculate Floquet hamiltonian. 


* UF: (optional) if set to 'True' save eovlution operator. 

* ThetaF: (optional) if set to 'True' save the eigen-values of the evolution operator. 

		* VF: (optional) if set to 'True' save the eigen-vectors of the evolution operator. 

* n_jobs: (optional) set the number of processors which are used when looping over the basis states. 

--- Floquet attributes ---: '_. ' below stands for 'object. '

Always given:

* _.EF: Floquet qausi-energies

Calculate via flags:

* _.HF: Floquet Hamiltonian dense array

* _.UF: Evolution operator

* _.VF: Floquet eigen-states

* _.thetaF: eigen-values of evolution operator

####**Floquet_t_vec**
```python
tvec = Floquet_t_vec(Omega,N_const,len_T=100,N_up=0,N_down=0)
```
Returns a time vector (np.array) which hits the stroboscopic times, and has as attributes their indices. The time vector can be divided in three regimes: ramp-up, constant and ramp-down.

--- arguments ---

* Omega: (compulsory) drive frequency

* N_const: (compulsory) # of time periods in the constant period

* N_up: (optional) # of time periods in the ramp-up period

* N_down: (optional) # of time periods in the ramp-down period

* len_T: (optional) # of time points within a period. N.B. the last period interval is assumed  open on the right, i.e. [0,T) and the poin T does not go into the definition of 'len_T'. 


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


* _.up : referes to time ector of up-regime; inherits the above attributes (e.g. _up.strobo.inds)

* _.const : referes to time ector of const-regime; inherits the above attributes

* _.down : referes to time ector of down-regime; inherits the above attributes

This object also acts like an array, you can iterate over it as well as index the values.











