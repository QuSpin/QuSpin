.. _parallelization-label:

Parallel Computing Support
==========================

In the recent years we have witnessed the move for processors to have more and more individual cores on a single chip, both in the consumer markets as well as in the server markets for CPUs. This means that as these chips come out we will see less improvements in terms of the actual speed of individual processors and more improvements in terms of memory access speeds and large cache sizes which will boost the performance of parallel computing. 

Having software which can easily be supported on these architectures is increasingly important to the users of QuSpin in order to keep this software capable of performing state-of-the-art calculations in the future. In the next few years we will start to see CPUs in high performance computing (hpc) clusters that will have at or above 30 cores on a single chip. A goal for QuSpin's developers is to allow the user to harness all that computational power without having to make significant changes to already developed QuSpin code. This can accomplished using `OpenMP <https://www.openmp.org/>`_ which works on the shared memory model of parallel computing. this model is ideal for new architectures with large numbers of cores. It also fits within the computing model we have with QuSpin, in which we focus on very general kinds of exact diagonalization calculations.

Below, we introduce some of the new features in QuSpin 0.3.1 which are included in the OpenMP enabled version of QuSpin.

Check out also our example script :ref:`example12-label`, which demonstrates how to use multi-threading with QuSpin in practice. 

1. Multi-threading via OpenMP in QuSpin:
-----------------------------------

1.1. Install QuSpin with OpenMP support:
````````````````

In order to make use of OpenMP features in QuSpin, one just needs to install the `omp` metapackage which will track the OpenMP compiled version of QuSpin for your platform. Starting from QuSpin 0.3.1, we have OpenMP support across the different operating systems. To install the OpenMP version of QuSpin simply run:
::
	$ conda install -c weinbe58 omp quspin

If you would like to go back to the single-threaded (i.e. no-OpenMP) version of QuSpin run:
::
	$ conda remove --features omp -c weinbe58

upon which you will be asked by anaconda if you want to downgrade you QuSpin version to a version which no longer tracks the `omp` feature. 



1.2. Multi-threaded support for QuSpin functions:
````````````````

All the support for QuSpin's OpenMP multi-threading can be accessed using the OpenMP environment variable: `OMP_NUM_THREADS`. Simply put, if you want to use multiple cores when running your script, set that variable equal to the number of cores you request during the calculation. Then the segments of code which use OpenMP will automatically begin to use those extra cores. 

There are two ways to set up the OpenMP environment variable:

1) in the terminal/Anaconda prompt, set
::
	$ export OMP_NUM_THREADS = 4
	$ echo $OMP_NUM_THREADS

Make sure you run your script from that terminal window. If you run your code from a different terminal window, you have to set this variable again.

2) in the beginning of your python script, **before you import QuSpin**,  set
::
	import os
	os.environ['OMP_NUM_THREADS'] = '4' # set number of OpenMP threads to run in parallel

This allows to change the OpenMP variable dynamically from your python script.

While this is very convenient, it does not make it clear which segments of the code will run faster. Thus, let us now go over all features which take advantage of this multi-threading. The function `tools.misc.csr_matvec()` wraps an efficient version of csr matrix-vector product based on a scheme which provides equal work load to all the threads, regardless of the sparsity structure of the matrix (see `this paper <https://ieeexplore.ieee.org/document/7877136>`_ for more details). This speedup will be inherited by the function `tools.evolution.expm_multiply_parallel()`, which creates a more efficient multi-threaded version of SciPy's `SciPy.sparse.linalg.expm_multiply` function. 
Notice that these function would have to be explicitly used by the user in order for a calculation to gain speedup via OMP. 



In QuSpin 0.3.1 we have worked on trying to make the user experience seamless so that the user does not need to write special code in order to take advantage of these parallelized functions, much like how NumPy uses MKL for doing linear algebra operations. In the next two sections we discuss where the new OpenMP support happens so that one can more easily write new code which takes advantage of it. 


1.2.1 Parallel support in the operator module: `hamiltonian`, `quantum_operator` and `quantum_LinearOperator`
+++++++++

One of the most ubiquitous operations in exact diagonalization codes is the matrix-vector product: the matrix represents a quantum operator and the vector -- the quantum state being acted on by the operator. This is used pretty much everywhere except for full diagonalization of the matrix: from evolution to Lanczos methods. 

In QuSpin, operators are represented by large sparse matrices and the quantum states are typically represented by dense vectors. In the computer science/mathematics literature, it is well known that this matrix-vector product is one of the most important operations done during a computation, so there has been a lot of work on trying to efficiently implement this operation in parallel. Most of the literature discusses only sparse-matrix -- vector product as opposed to sparse-matrix -- dense matrix products, and this is reflected in QuSpin's implementation. Currently QuSpin supports multi-threading only when the multiplication is on a vector (even though multiplication by two-dimensional arrays is allowed as well, but the code switches to a single-threaded version). 

We have specifically designed QuSpin to work very efficiently with the structure of the `hamiltonian` and `quantum_operator` classes. This low level code replaces the use of SciPy's default functionality (which adds unnecessary overhead and slows down the code when pushing to large system sizes). This required to limit the number of supported matrix formats used by QuSpin's operator classes. Currently, we support: `csr`, `csc`, `dia` and also dense matrices when constructing a `hamiltonian` or `quantum_operator` objects to allow for a broad range of applicability. For example, one can get a performance boost when transposing your `hamiltonian` or `quantum_operator` as `csr` <--> `csc` and `dia` <--> `dia` without having to copy any data. The dense matrices we fall back on NumPy's library to do the calculation as it is specifically optimized for the kinds of calculations we need in QuSpin. 

For the supported sparse-matrix formats `csr`, `csc`, and `dia`, we have implemented multi-threaded matrix-vector products (see `tools.misc.matvec()` function), all of which show very nearly linear scaling with increasing the number of cores on modern processors. Even though the performance gains are more modest on older CPU architectures, they can still be useful when simulating large system sizes as one typically needs to allocate a lot of memory space when submitting a job (which usually just means requesting more cores). 

To sum up, whenever one can prefer matrix-vector products in the code, using QuSpin's interface this will lead to the automatic use of multi-threading, when the OpenMP version is used. For instance, one commonly used function, which automatically benefits from multi-threading via the parallel matrix-vector product, is `hamiltonian.evolve()`. 

At the same time, in some places automatic multithreading is not so obvious: for instance if one is trying to find the ground state of a particular `hamiltonian` object `H` one might do the following:
::
	E,V = H.eigsh(time=t0,k=1,which="SA")
The code just above will actually not use any multi-threading: this is because this code is actually equivilent to doing:
::
	E,V = eigsh(H.tocsr(time=t0),k=1,which="SA")
However, one can still beneft from the multi-threaded matrix-vector product by using the `H.aslinearoperator(time=t0)` method:
::
	E,V = eigsh(H.aslinearoperator(time=t0),k=1,which="SA")
Casting `H` as a `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_ object enables the use of the methods `H.dot()` and `H.transpose().conj().dot()`. These methods will be used to do the eigenvalue calculation, which will then benefit from multi-threading (note that one cannot use `LinearOperator` by default when calling `H.eigsh()` since it limits the code functionality).

.. Now one might ask: why not use the LinearOperator wrapper of the Hamiltonian class by default when calling `H.eigsh`? This works in many cases however there can be problems that will not work for LinearOperators. One example of this is solving for eigenvalues in the middle of the spectrum `eigsh`. We are not sure if this will ever be fixed in future versions on SciPy as it does not appear to be related to ARPACK (used by `eigsh`), but the convergence of some other algorithm which is called during the process for inverting the LinearOperator. This is evident by to the traceback:

..	Traceback (most recent call last):
	  File "test_LinearOperator_eigsh.py", line ##, in <module>
	    E_gs,gs = sla.eigsh(H.aslinearoperator(),k=2,sigma=0)
	  File ".../anaconda2/lib/python2.7/site-packages/scipy/sparse/linalg/eigen/arpack/arpack.py", line 1651, in eigsh
	    params.iterate()
	  File ".../anaconda2/lib/python2.7/site-packages/scipy/sparse/linalg/eigen/arpack/arpack.py", line 559, in iterate
	    self.workd[yslice] = self.OPa(self.workd[Bxslice])
	  File ".../anaconda2/lib/python2.7/site-packages/scipy/sparse/linalg/interface.py", line 219, in matvec
	    y = self._matvec(x)
	  File ".../anaconda2/lib/python2.7/site-packages/scipy/sparse/linalg/eigen/arpack/arpack.py", line 975, in _matvec
	    % (self.ifunc.__name__, info))
	ValueError: Error in inverting M: function gmres_loose did not converge (info = 2570).

1.2.2 Parallel support in the general basis classes `*_basis_general`
+++++++++

Starting from QuSpin 0.3.1, we have efficient implementation of parallel support for the methods in the `*_general_basis` classes.
Additionally, we have also added an implementation of `inplace_Op` which is used to do 'on the fly' calculation of an operator acting on a state using multi-threading OpenMP speed-up (which can be accessed simply by using any general basis in the `quantum_LinearOperator` class).

Note that the `*_basis_1d` classes do **not** support OpenMP. 

2. Multi-threading via MKL and NumPy/SciPy in QuSpin:
--------------------------------------------

Depending on the version of NumPy you have installed, you may also be able to access some additional multi-threading to speed up diagonalization, e.g. using `eigh()`, `eigvalsh()`, or `svd()` operations during calculations of eigenvalues/vectors or entanglement entropy. 
To do this, the default version of NumPy installed with Anaconda must be linked against Intel's Math Kernel Library (MKL) which implemented very efficient multi-threaded variations of LAPACK functions. If you use Anaconda 2.5 or later, MKL is the default numpy version. To turn on the multi-threading, simply use the MKL environment variables. For more info visit this `MKL website <https://software.intel.com/en-us/mkl-linux-developer-guide-intel-mkl-specific-environment-variables-for-openmp-threading-control>`_.

There are two ways to set up the MKL environment variable:

1) in the terminal/Anaconda prompt, set
::
	$ export MKL_NUM_THREADS = 4
	$ echo $MKL_NUM_THREADS

Make sure you run your script from that terminal window. If you run your code from a different terminal window, you have to set this variable again.

2) in the beginning of your python script, **before you import NumPy or SciPy** set
::
	import os
	os.environ['MKL_NUM_THREADS'] = '4' # set number of MKL threads to run in parallel

This allows to change the MKL variable dynamically from your python script.

Another useful python package for changing the number of cores MKL is using at runtime is `mkl-service <https://docs.anaconda.com/mkl-service/>`_. For more information about MKL-accelerated versions of NumPy, check out this `website <https://docs.anaconda.com/mkl-optimizations/>`_.




