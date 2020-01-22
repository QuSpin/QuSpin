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

**@Phil, PUT HERE NOTES ON CONVERGENCE AND COMPARISON WITH expm_multiply_parallel**

Ground State Search
+++++++++++++++++++

When using the Lanczos algorithm to compute (part of) the eigensystem of :math:`H`, it should be noted that the :math:`m` eigenvalues of :math:`T` approximate the :math:`m` largest eigenvalues of :math:`H`. Thus, when looking for ground-state properties, one typically applies the Lanczos algorithm to :math:`-H`. 

**@Phil, explain advantages/disadvantages**

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

