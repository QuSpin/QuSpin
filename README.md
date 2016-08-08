#**qspin**
qspin is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state of the art exact diagonalization calculations on 1 dimensional spin 1/2 chains with lengths up to 31 sites. The interface allows the user to define any spin 1/2 Hamiltonian which can be constructed from spin operators; while also allowing the user flexibility of accessing all possible symmetries in 1d. There is also a way of specifying the time dependence of operators in the Hamiltonian as well, which can be used to solve the time dependent Schrodinger equation numerically for these systems. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access the powerful linear algebra tools. 




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


#Installation
This latest version of this package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux and OS-X and windows 64-bit systems. In order to install this you need to get Anaconda package manager for python. Then all one has to do to install is run:

```
$ conda install -c weinbe58 qspin
```

This will install the latest version on to your computer. If this fails for whatever reason this means that either your OS is not supported or your compiler is too old. In this case one can also manually install the package:


1) Manual install: to install manually download source code either from the latest [release](https://github.com/weinbe58/qspin/releases) section or cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

unix:
```
python setup.py install
```

or windows command line:
```
setup.py install
```
NOTE:** you must write permission to the standard places python is installed as well as have all the prerequisite python packages (numpy >= 1.10.0, scipy >= 0.14.0) installed first to do this type of install since the actual build itself relies on numpy. We recommend [Anaconda](https://www.continuum.io/downloads) to manage your python packages. 


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

magnetization sector:
Nup = integers: 0,...,L; or list: [1,...] 

parity sector:
pblock = +/-1

spin inversion sector:
zblock = +/-1

(parity)*(spin inversion) sector:
pzblock = +/-1

momentum sector:
kblock = any integer

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

###**normal operations**
The hamiltonian objects currectly support certain arithmetic operations with other hamiltonians as well as scipy sparse matrices and numpy dense arrays and scalars:

* between other hamiltonians we have: ```+,-,+=,-=``` 
* between numpy and sparse arrays we have: ```*,+,-,*=,+=.-=``` (versions >= v0.0.5b)
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
	```
	v = H.expm_multiply(u,a=-1j,time=0,times=(),iterate=False,**linspace_args)
	```
	* u: vector to act on
	* a: factor to multiply H by in the exponential
	* time: time to evaluate H at
	* times: lines of times to exponentiate to
	* iterate: bool to return generate or list of vectors
	* linspace_args: arguements to pass into expm_multiply
	
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

###**Harmonic Oscillator Basis**
This basis implements a single harmonic oscillator mode. The available operators are ```+,-,n```.

###**Tensor Basis Objects**
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

###**Checks on operator strings**
new in version 0.2.0 we have included a new functionality classes which check various properties of a given static and dynamic operator lists. They include the following:

* Checks if complete list of opertors obey the given symmetry of that basis. The check can be turned off with the flag ```check_symm=False ``` in the [hamiltonian](https://github.com/weinbe58/qspin/tree/master/qspin/hamiltonian) class. 
* Checks of the given set of operators are hermitian. The check can be turned out with the flag ```check_herm=False``` in the [hamiltonian](https://github.com/weinbe58/qspin/tree/master/qspin/hamiltonian) class. 
* Checks of the given set of operators obey particle conservation (for spin systems this means magnetization sectors do not mix). The check can be turned out with the flag ```check_pcon=False``` in the [hamiltonian](https://github.com/weinbe58/qspin/tree/master/qspin/hamiltonian) class. 

###**Methods of Basis Classes**
* Op(indx,opstr,J,dtype)
* get_vec(v0)
* get_proj(dtype)


# **Tools**

##**Observables** 

#### **Entanglement entropy**

```python
Entanglement_entropy(L,psi,subsys=[i for i in xrange( int(L/2) )],basis=None,alpha=1.0, DM=False, DM_basis=False) 
```

This is routine calculates the entanglement (Renyi) entropy if a pure squantum state in a subsystem of choice. It returns a dictionary in which the entanglement (Renyi) entropy has the key 'Sent'. The arguments are:

L: (compulsory) chain length. Always the first argument.

psi: (compulsory) a pure quantum state, to calculate the entanglement entropy of. Always the second argument.

basis: (semi-compulsory) basis of psi. If the state psi is written in a symmetry-reduced basis, then one must also parse the basis in which psi is given. However, if no symmetry is invoked and the basis of psi contains all 2^L states, one can ommit the basis argument.

subsys: (optional) a list of site numbers which define uniquely the subsystem of which the entanglement entropy (reduced and density matrix) are calculated. Notice that the site labelling of the chain goes as [0,1,....,L-1]. If not specified, the default subsystem chosen is [0,...,floor(L/2)].

alpha: (optional) Renyi parameter alpha. The default is 'alpha=1.0'.

DM: (optional) when set to True, the returned dictionary contains the reduced density matrix under the key 'DM'. Note that the
reduced DM is written in the full basis containing 2^L states. 









#### **Diagonal Ensemble Observables**
```python
Diag_Ens_Observables(L,V1,V2,E1,betavec=[],alpha=1.0, Obs=False, Ed=False, S_double_quench=False, Sd_Renyi=False, deltaE=False)
```

This is routine calculates the expectation values of physical quantities in the Diagonal ensemble (see eg. arXiv:1509.06411). It returns a dictionary. Equivalently, these are the infinite-time expectation values after a sudden quench at time t=0 from the Hamiltonian H1 to the Hamiltonian H2. 

L: (compulsory) chain length. Always the first argument.

V1: (compulsory) unitary square matrix. Contains the eigenvectors of H1 in the columns. The initial state is the first column of V1. Always the second argument.

V2: (compulsory) unitary square matrix. Contains the eigenvectors of H2 in the columns. Must have the same size as V1. Always the third argument.

E1: (compulsory) vector of real numbers. Contains the eigenenergies of H1. The order of the eigenvalues must correspond to the order of the columns of V1. Always the fourth argument.

Obs: (optional) hermitian matrix of the same size as V1. Infinite-time expectation value of the observable Obs in the state V1[:,0]. Has the key 'Obs' in the returned dictionary.

Ed: (optional) infinite-time expectation value of the Hamiltonian H1 in the state V1[:,0]. Hast the key 'Ed' in the returned dictionary.

deltaE: (optional) infinite-time fluctuations around the energy expectation Ed. Has the key 'deltaE' in the returned dictionary.

Sd_Renyi: (optional) diagonal Renyi entropy after a quench H1->H2. The default Renyi parameter is 'alpha=1.0'. Has the key 'Sd_Renyi' in the returned dictionary.

alpha: (optional) diagonal Renyi entropy parameter. Default value is 'alpha=1.0'.

S_double_quench: (optional) diagonal entropy after a double quench H1->H2->H1. Has the key 'S_double_quench' in the returned dictionary.

betavec: (optional) a list of INVERSE temperatures to specify the distribution of an initial thermal state. When passed the routine returns the corresponding finite-temperature expectation of each specified quantity defined above. The corresponding keys in the returned dictionary are 'Obs_T', 'Ed_T', 'deltaE_T', 'Sd_Renyi_T', 'S_double_quench_T'. 







#### **Kullback-Leibler divergence**
```python
Kullback_Leibler_div(p1,p2)
```
This routine returns the Kullback-Leibler divergence of the discrete probabilities p1 and p2. 







####**Time Evolution**
```python
Observable_vs_time(psi,V2,E2,Obs,times,return_state=False)
```
This routine calculate the expectation value as a function of time of an observable Obs. The initial state is psi and the time evolution is carried out under the Hamiltonian H2. Returns a dictionary in which the time-dependent expectation value has the key 'Expt_time'.

psi: (compulsory) initial state. Always first argument.

V2: (compulsory) unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. Always second argument.

E2: (compulsory) real vector containing the eigenvalues of the Hamiltonian H2. The order of the eigenvalues must correspond to the order of the columns of V2. Always third argument.

Obs: (compulsory) hermitian matrix to calculate its time-dependent expectation value. Always fourth argument.

times: (compulsory) a vector of times to evaluate the expectation value at. always fifth argument.

return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector at the times specified by the row index. The return dictonary key is 'psi_time'.



### **Mean Level spacing**
```python
Mean_Level_Spacing(E)
```
This routine returns the mean-level spacing r_ave of the energy distribution E, see [arXiv:1212.5611](http://arxiv.org/abs/1212.5611). 

E: (compulsory) ordered list of ascending, nondegenerate eigenenergies. 

###**Floquet**










