# Using hamiltonian:
The hamiltonian objects currectly support certain arithmetic operations with other hamiltonians as well as scipy sparse matrices and numpy dense arrays and scalars:

* between other hamiltonians we have: ```+,-,+=,-=``` 
* between numpy and sparse arrays we have: ```*,+,-,*=,+=.-=``` (versions >= v0.0.5b)
* between scalars: ```*,*=``` (versions >= v0.0.5b)


We've included some basic functionality into the hamiltonian class useful for quantum calculations:

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
  which evaluates < u|H(time)|v > if u and v are vectors but (versions >= v0.0.2b) can also handle u and v as dense matrices. NOTE: the inputs should not be hermitian tranposed, the function will do that for you.
  
* matrix exponential (versions >= v0.0.5b):
  usage:
    ```python
    v = H.expm_multiply(u,a=a,time=time,**linspace_args)
    ```
  which evaluates |v > = exp(a H(time))|u > using only the dot product of H on |u >. The extra arguements will allow one to evaluate it at different time points see scipy docs for [expm_multiply](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply) for more information.
  
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


