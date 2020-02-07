.. _example20-label:

Lanczos module: time-evolution and ground state search
------------------------------------------------------

This example demonstrates how to use the `Lanczos` submodule of the `tools` module to do time evolvution and ground state search in the Heisenberg model:


.. math::
	H = J\sum_{j=0}^{N-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j,

where :math:`S_j` is the spin-1/2 operator on lattice site :math:`j`; we use periodic boundary conditions.

The Lanczos decomposition for the :math:`n\times n` Hamiltonian matrix is defined as

.. math::
	H \approx Q T Q^\dagger

for a real-valued, symmetric tridiagonal :math:`m\times m` matrix :math:`T=Q^\dagger HQ`, and (in general) a complex-valued :math:`n\times m` matrix :math:`Q` containing the orthonormal Lanczos vectors in the rows. Here :math:`m` is the number of states kept in the Krylov subspace which controls the quality of the "Lanczos compression" of :math:`H`. We further apply the eigenvalue decomposition :math:`T=V \mathrm{diag}(E) V^T` and compute the eigenvectors :math:`V` of :math:`T` (note that this is computationally cheap for :math:`m\ll n`).   

Time evolution
++++++++++++++

With this information, we can compute an approximation to the matrix exponential, applied to a state :math:`|\psi\rangle` as follows:

.. math::
	\exp(-i a H)|\psi\rangle \approx Q \exp(-i a T) Q^\dagger |\psi\rangle = Q V \mathrm{diag}(e^{-i a E}) V^T Q^\dagger |\psi\rangle.

If we use :math:`|\psi\rangle` as the (nondegenerate) initial state for the Lanczos algorithm, then :math:`\sum_{j,k}V^T_{ij}Q^\dagger_{jk}\psi_k = \sum_{j}V_{ji}\delta_{0,j} = V_{i,0}` [by construction :math:`\psi_k` is the zero-th row of :math:`Q` and all the rows are orthonormal], and the expression simplifies further. Notice that these lines of thought apply to any matrix function, not just the matrix exponential. 

The convergence of this method depends heavily on the function that is being approximated as well as the structure of the matrix. For the matrix exponential there is some literature in math/computer science that discuss error a-priori and a-posteriori bounds for krylow methods. The a-priori bounds are typically far from saturation when performing numerical experiments and the a-posteriori bounds are often impracticle to implement. One can easily check convergence by calculating the lanczos basis of size `m` but performing the calculation with `m` and `m-1` basis vectors and comparing the results, or by comparing the results of the lanczos calculation to some other method that is implemented, e.g. `expm_multiply` or `expm_multiply_parallel`. 

In the case for `expm_multiply_parallel` the convergence is always gaurenteed to be machine precision. The tolerance can be slightly controlled by switching between single and double precision floating point types which can often speed up the calculation by a factor of 1.5. That being said, it is possible to get faster code, however this requires more memory to store the lanczos vectors in memory during the calculation and often one has to experiment quite a bit to find the optimal time-step and number of lanczos vectors required ot beat `expm_multiply_parallel`. 

Ground State Search
+++++++++++++++++++

When using the Lanczos algorithm to compute (part of) the eigensystem of :math:`H`, it should be noted that the :math:`m` eigenvalues of :math:`T` approximate the :math:`m` largest eigenvalues of :math:`H`. Thus, when looking for ground-state properties, one typically applies the Lanczos algorithm to :math:`-H`. 

One may be tempted to use imaginary time evolution to find the ground state. Some advantages of this is that it is possible to systematically control when to stop by observing the evolution of the expectation value of the energy :math:`\langle\psi(\tau)|H|\psi(\tau)\rangle/\langle\psi(\tau)|\psi(\tau)\rangle`, as :math:`\tau\rightarrow\infty`. This can be done using multiple time-steps or in one large time step using the lanczos matrix exponential. Interestly enough if we assume that the number of lanczos vectors is large enough for the matrix exponential to converge that implies that, for the large time-step, the ground state as calculated from the krylow-subspace has to be the ground state. In that sense the matrix exponential for large :math:`\tau` is no different then calculating the ground state directly from the lanczos basis. 


Thermal Expectation Values 
++++++++++++++++++++++++++

**@Phil, add explanation + example to the code**



Script
------

:download:`download script <../../../examples/scripts/example20.py>`

.. literalinclude:: ../../../examples/scripts/example20.py
	:linenos:
	:language: python
	:lines: 1-

