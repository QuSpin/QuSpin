:orphan:

.. _example27-label:


Liouville-von Neumann Equation using the MKL-enhanced Sparse Matrix Product [courtesy of J. Verlage]
----------------------------------------------------------------------------------------------------

This example shows how one can combine QuSpin with the external library `sparse_dot <https://github.com/flatironinstitute/sparse_dot.git>`_ which supports an MKL-parallelized sparse matrix product.

To this end, we consider the numerical solution of the Liouville-von Neumann (LvN) equation for the density matrix: 

.. math::
	i\partial_t \rho(t) = [H,\rho(t)].

The system is the Fermi-Hubbard modelon a square lattice: 

.. math::
	H = -J\sum_{j,\sigma} \left( c^\dagger_{j+1\sigma}c_{j\sigma} + \mathrm{h.c.} \right) + U\sum_j n_{j\uparrow}n_{j\downarrow},

where :math:`j=(x,y)` denotes the lattice site. We choose a mean-field initial state, 

.. math::
	\rho(0)=\bigotimes_j \rho_j, \qquad \mathrm{where} \qquad  \rho_j = \frac{1}{2}\left( |\uparrow_j\rangle\langle \uparrow_j|+ |\downarrow_j\rangle\langle \downarrow_j| \right),

that cannot be written as a pure state [hence the necessity to solve the LvN equation rather than Schroedinger's equation].

Note that the initial state :math:`\rho(0)` is diagonal in the particle number basis; therefore, since the Hamiltonian :math:`H` is also sparse, we expect that the time-evolved density operator will remain sparse at least for small times [compared to :math:`U^{-1}, J^{-1}`]. 
Since we are limited to small system sizes by the exponentially growing Hilbert space dimension, we need a memory-efficient way to store the quantum state, e.g., using a sparse matrix. In turn, this requires:

	* an efficient, ideally parallelized, sparse-spase matrix product;
	* a solver for differential equations that allows us to keep the variable [here :math:`\rho`] in sparse format at all times. 

To this end, we can use the open-source python library `sparse_dot <https://github.com/flatironinstitute/sparse_dot.git>`_, which provides the MKL-paralellized function `dot_product_mkl`. We use it to write our own fourth-order Runge-Kutta (RK) solver for the LvN equation. Note that, unlike the RK solver provided in Scipy where the step size is chosen adaptively, our RK implementation has a fixed step size; however, scipy's solver does not allow us to keep the state as a sparse matrix at all times.



Script
------

:download:`download script <../../../examples/scripts/example27.py>`

.. literalinclude:: ../../../examples/scripts/example27.py
	:linenos:
	:language: python
	:lines: 1-

