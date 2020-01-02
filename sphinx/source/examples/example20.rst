.. _example20-label:

Lanczos module: time-evolution and ground state search
------------------------------------------------------

This example demonstrates how to use the `Lanczos` submodule of the `tools` module to do time evolvution and ground state search in the Heisenberg model:


.. math::
	H = J\sum_{j=0}^{N-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j,

where :math:`S_j` is the spin-1/2 operator on lattice site :math:`j`; we use periodic boundary conditions.

The Lanczos decomposition for the :math:`n\times n` Hamiltonian matrix is defined as **check positions of transpose/dagger, also in the docstr for tools.lanczos**

.. math::
	H \approx Q^\dagger T Q 

for a real-valued, symmetric tridiagonal :math:`m\times m` matrix :math:`T=QHQ^\dagger`, and (in general) a complex-valued :math:`m\times n` matrix :math:`Q` with orthonormal columns. Here :math:`m` is the number of states kept in the Krylov subspace which controls the quality of the "Lanczos compression" of :math:`H`. We further apply the eigenvalue decomposition :math:`T=V^T \mathrm{diag}(E) V` and compute the eigenvectors :math:`V` of :math:`T` (note that this is actually cheap for :math:`m\ll n`.   

With this information, we can compute an approximation to the matrix exponential, applied to a state :math:`|\psi\rangle` as follows:

.. math::
	\exp(-i a H)|\psi\rangle \approx Q^\dagger \exp(-i a T) Q |\psi\rangle = Q^\dagger V^T \mathrm{diag}(e^{-i a E}) V Q |\psi\rangle

If we use :math:`|\psi\rangle` as the (nondegenerate) initial state for the Lanczos algorithm, then :math:`\sum_{j,k}V_{ij}Q_{jk}\psi_k = \sum_{j}V_{ij}\delta_{j,0} = V_{i,0}` [by construction :math:`\psi_k` is the zero-th column of :math:`Q` and all the columns are orthonormal], and the expression simplifies further. Notice that these lines of thought apply to any matrix function, not just the matrix exponential. 

When using the Lanczos algorithm to compute (part of) the eigensystem of :math:`H`, it should be noted that the :math:`m` eigenvalues of :math:`T` approximate the :math:`m` largest eigenvalues of :math:`H`. Thus, when looking for ground-state properties, one typically applies the Lanczos algorithm to :math:`-H`. 

**@Phil, PUT HERE NOTES ON CONVERGENCE AND COMPARISON WITH expm_multiply_parallel**


Script
------

:download:`download script <../../../examples/scripts/example20.py>`

.. literalinclude:: ../../../examples/scripts/example20.py
	:linenos:
	:language: python
	:lines: 1-

