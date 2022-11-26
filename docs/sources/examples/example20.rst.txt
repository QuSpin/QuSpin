:orphan:

.. _example20-label:


Lanczos module: time-evolution and ground state search
------------------------------------------------------

This example demonstrates how to use the `Lanczos` submodule of the `tools` module to do time evolvution and ground state search in the Heisenberg model:


.. math::
	H = J\sum_{j=0}^{L-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j,

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

If we use :math:`|\psi\rangle` as the (nondegenerate) initial state for the Lanczos algorithm, then :math:`\sum_{j,k}V^T_{ij}Q^\dagger_{jk}\psi_k = \sum_{j}V_{ji}\delta_{0,j} = V_{i,0}` [by construction, :math:`\psi_k` is the zero-th row of :math:`Q` and all the rows are orthonormal], and the expression simplifies further. Notice that these lines of thought apply to any matrix function, not just the matrix exponential. 

The convergence of this method depends heavily on the function that is being approximated as well as the structure of the matrix. For the matrix exponential there is some literature in math/computer science that discuss error a-priori and a-posteriori bounds for krylow methods. The a-priori bounds are typically far from saturation when performing numerical experiments and the a-posteriori bounds are often impracticle to implement. One can easily check convergence by calculating the lanczos basis of size `m` but performing the calculation with `m` and `m-1` basis vectors and comparing the results, or by comparing the results of the lanczos calculation to some other method that is implemented, e.g. `expm_multiply` or `expm_multiply_parallel`. 

In the case for `expm_multiply_parallel` the convergence is always guaranteed to be machine precision. The tolerance can be slightly controlled by switching between single and double precision floating point types which can often speed up the calculation by a factor of about 1.5. That being said, it is possible to get faster code; however, this requires more memory to store the lanczos vectors in memory during the calculation and often one has to experiment quite a bit to find the optimal time-step and number of lanczos vectors required to beat `expm_multiply_parallel`. 

Ground State Search
+++++++++++++++++++

One of the major uses of the Lanczos method is to find the ground state of a given matrix. It is important to remember that the Lanczos iteration projects out the eigenstates with the largest magnitude eigenvalues of the operator. As such, depending on which eigenvalues one is targeting one might have to transform the operator to make sure that the Lanczos algorithm targets that particular eigenvalue. In the case of the ground state, one either shift the operator by a constant as to make the magitude of the ground state the largest, however, in a lot of cases the ground state already has one of the largest magnitude eigenvalues. 

After creating the lanczos basis, QuSpin will return the eigenvalues and vectors of the Krylov sub-space matrix :math:`T`. If the operator has been transformed to create the Lanczos basis, one should perform the inverse transform of the eigenvalues to get the eigenvalues of the original operator. In the example below the ground state energy is the largest magnitude eigenvalue, hence we do not need to transform the Hamiltonian and likewise, the eigenvalues. The eigenvectors of the Hamiltonian can be constructed by taking linear combinations of the Lanczos basis with coefficients given by the eigenvectors of :math:`T`.

Script
------

:download:`download script <../../../examples/scripts/example20.py>`

.. literalinclude:: ../../../examples/scripts/example20.py
	:linenos:
	:language: python
	:lines: 1-

