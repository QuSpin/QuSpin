.. _example17-label:

Optical Bloch equations: Lindblad dynamics using the fast (omp-parallelized) `matvec` function
----------------------------------------------------------------------------------------------

This example uses the omp-parallelized `tools.misc.matvec()` function to define a Lindblad equation and solve the ODE using the `tools.evolution.evolve()` function. 

Consider the the two-level system: 

.. math::
	H &= \delta\sigma^z + \Omega_0\sigma^x, \\
	L &= \sigma^+.

where :math:`H` is the Hamiltonian of the two-level system, and :math:`L` is the Lindblad (or jump) operator. The Lindblad equation is a non-unitary extension of the Liouville-von Neumann equation for the density matrix :math:`\rho(t)`:

.. math::
	\partial_t\rho(t) = -i[H,\rho(t)] + 2\gamma\left(L\rho(t)L^\dagger - \frac{1}{2}\{L^\dagger L,\rho(t) \} \right),

where :math:`[\cdot,\cdot]` is the commutator, and :math:`\{\cdot,\cdot\}` is the anti-commutator. The system of equations for this specific problem is also known as the optical Bloch equations.

Below, we provide three ways to define the function for the Lindblad ODE. The first version is very intuitive, but rather slow. The second version uses the `hamiltonian.dot()` and `hamiltonian.rdot()` functions and is a bit more sophisticated and a bit faster. The third version uses the `matvec()` function and is faster than the previous two (but may not be memory efficient for large systems).

Note that this way of simulating the Lindblad equation has severe limitations for many-body systems. An alternative, parallelizable way to effectively simulate a subset of Lindblad dynamics using unitary evolution is described in 
`arXiv:1608.01317 <https://arxiv.org/pdf/1608.01317.pdf>`_.

Script
------

:download:`download script <../../../examples/scripts/example17.py>`

.. literalinclude:: ../../../examples/scripts/example17.py
	:linenos:
	:language: python
	:lines: 1-

