Parallel Computing Support
==========================

In recent years we've seen the move for processors to have more and more indivicual cores on a single chip, both in the consumer markets as well as in the server markets for CPUs. This means that as thsese chips come out we will see less improvements in terms of the actual speed of individual processors and more improvements in terms of memory access speeds and large cache sizes which will improve the performance of parallel computing. Having software which can easily be supported on these archetectures is incredibly important to the developers of QuSpin in order to keep this software relelvant in the future. In the next few years we will start to see CPUs in high proformence computing clusters that will have at or above 30 cores on a single chip. The goal for the developers of QuSpin will be to allow the user to harness all of that computational power without having to make significant changes to a current set of code the user might have already developed. This is accomplished using OpenMP which works on the shared memory model of parallel computing. this model is ideal for these new archetectures with large numbers of cores. It also fits within the computing model we have with QuSpin in which we focus on very general kinds of exact diagonalization calculations.

Here we will introduce some of the new features in QuSpin 0.3.1 which are included in the OpenMP enabled version of QuSpin.

Install QuSpin with OpenMP support:
-----------------------------------

First, in order to install the OpenMP version of QuSpin one simply just needs to install the `omp` metapackage which will track the OpenMP compiled version of QuSpin for your platform. Note in this new version we have support across the different operating systems! So to install the OpenMP version of QuSpin simply run:
::
	$ conda install -c weinbe58 omp quspin

If you would like to go back to the single threaded version of QuSpin simply run:
::
	$ conda remove --features omp -c weinbe58

where you should then be asked if you want to change you quspin version to a version which no longer tracks the `omp` feature. 



QuSpins's Multi-threading via MKL and NumPy:
--------------------------------------------

Depending on your Version of NumPy you have installed, you may also be able to access some additional multi-threading during diagonalization using `eigh` or `eigvalsh` or `svd` calculations used during calculations of eigenvalues/vectors or entanglement entropy respectively. The default version of NumPy installed with Anaconda will be linked against Intel's Math Kernel Library (MKL) which implemented very efficient multi-threaded variations of LAPACK functions. To turn on the multi-threading simply use the MKL environment variables. For more info visit this `MKL website <https://software.intel.com/en-us/mkl-linux-developer-guide-intel-mkl-specific-environment-variables-for-openmp-threading-control>`_.

For more information about MKL accelerated versions of NumPy check out this `website <https://docs.anaconda.com/mkl-optimizations/>`_

QuSpins's new Multi-threaded support:
-------------------------------------

All the support for QuSpin's OpenMP multi-threading can be accessed using the OpenMP environment variable: `OMP_NUM_THREADS`. Simply put if you want to use multiple cores when running your script, set that variable to equal the number of cores you want to use during the calculation and the segments of code which use OpenMP will automatically begin to use those extra cores. 

This is very convenient, however, this does not make it clear which segments of the code this will speed up so we shall go through all the features which take advantage of this multi-threading. Recall in version 0.3.0 we added a function to the `tools.misc` called `csr_matvec`. This function wraps an efficient version of csr matrix-vector product based on a scheme to provide equal work load to all the threads regardless of the sparsity structure of the matrix (see `this paper <https://ieeexplore.ieee.org/document/7877136>`_ for more details). This was also used in the function `expm_multiply_parallel` in `tools.evolution` in order to create a more efficient multi-threaded version of scipy's `expm_multiply` function. These are nice functions, however, they would have to be explicitly used by the user in order to actually be relevant for a calculation and I imagine that some of you reading this may not have even known about these functions. 

In QuSpin 0.3.1 we have worked on trying to make the user experience seamless so that a user does not need to write special code in order to take advantage of these parallelized functions much like how NumPy uses MKL for doing linear algebra operations. In the next two sections we will discuss where the new OpenMP support happens so that you can more easily write new code which takes advantage of this. 


Parallel Support in Sparse Matrix-Vector Product
------------------------------------------------

One of the most ubiquitous operations in exact diagonalization codes is the matrix-vector product; the matrix being your quantum operator and the vector being the quantum state being acted on by the operator. This is used in everything except full diagonalization of the matrix, from evolution to Lanczos methods. In the case of QuSpin, the operators are represented by large sparse matrices and the quantum states are typically represented by dense array. In the computer science/mathematics literature it is well known that this matrix-vector product is one of the most important operations done during a computation so there has been a lot of work on trying to efficiently implement this operation in parallel. Most of the literature discusses only sparse matrix-vector product as opposed to sparse matrix-dense matrix products and this is reflected in QuSpin's implementation. At this stage we support only multi-threading when the operation is on vector, however some might recall that QuSpin's functionality actually works on 2-D arrays as well. In this situation the code then switches to a single threaded version. Now this is not to say that it is all bad as we have specifically designed this low level code to work very efficiently with the structure of the `hamiltonian` and `quantum_operator` classes. This low level code replaces the use of SciPy's default functionality which adds extra overhead slows the code down when pushing to large system sizes. However this comes as a cost as we have to limit the number of supported matrix formats used by QuSpin. We support: csr, csc, dia and also dense matrices when constructing a `hamiltonian` or `quantum_operator` objects which allows for a broad range of applicability, for example you will still get a performance boost when transposing your `hamiltonian` or `quantum_operator` as csr <--> csc and dia -> dia without having to copy any data. The dense matrices we fall back on NumPy's library to do the calculation as it is specifically optimized for the kinds of calculations we do here. For these three sparse matrix formats we implement a multi-threaded matrix-vector products all of which shows very nearly linear scaling when adding cores on more modern processors. The performance gains are more modest on older CPU architectures however still useful when doing large system sizes as one typically needs to allocate a lot of memory space when submitting a job which usually means requesting more cores which would before lay idle. 

So in summary whenever you can prefer matrix-vector products in your calculation using QuSpin's interface this will lead to the automatic use of multi-threading. For example, one place which can benefit from the parallel matrix-vector product is in `H.evolve` which will automatically start to use multi-threading if you evolve a vector. However in some places it is not so obvious for example: if we are trying to find the ground state of a particular Hamiltonian one might do the following, given `H`:
::
	E,V = H.eigsh(time=t0,k=1,which="SA")
however this actually will not use any multi-threading, this is because the code is actually equivilent to doing:
::
	E,V = eigsh(H.tocsr(time=t0),k=1,which="SA")
If you want to use the multi-threaded matrix vector product you would have to use the `H.aslinearoperator(time=t0)` method:
::
	E,V = eigsh(H.aslinearoperator(time=t0),k=1,which="SA")

Because the method will return a `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_ object which wraps `H.dot` and `H.tranpose().conj().dot` and uses those to do the calculation, which will then use the multi-threaded code.

Now one might ask: why not use the LinearOperator wrapper of the Hamiltonian class by default when calling `H.eigsh`? This works in many cases however there can be problems that will not work for LinearOperators. One example of this is solving for eigenvalues in the middle of the spectrum `eigsh`. We are not sure if this will ever be fixed in future versions on SciPy as it does not appear to be related to ARPACK (used by `eigsh`), but the convergence of some other algorithm which is called during the process for inverting the LinearOperator. This is evident by to the traceback:
::
	Traceback (most recent call last):
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

Parallel Support in the General Basis Blasses
---------------------------------------------

In QuSpin 0.3.0 we introduced the general basis classes which had some OpenMP support built in, we're happy to announce that in QuSpin 0.3.1 we have updated the parallel support to make the functionality more efficient. On top of this we have also added an implementation of `inplace_Op` which is used to do 'on the fly' calculation of an operator acting on a state which also uses OpenMP to speed up the calculation. This can be accessed simply by using any general basis in the `quantum_LinearOperator` class.





