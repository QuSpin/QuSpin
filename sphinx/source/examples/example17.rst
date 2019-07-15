.. _example17-label:

Optical Bloch equations: Lindblad dynamics using the fast `matvec` function.
--------

This example uses the omp-parallelized `tools.misc.matvec()` function to define and Lindblad equation for a two-level system. 
We then solve the ODE using the `tools.evolution.evolve()` function.

.. math::
	H &= \delta\sigma^z + \Omega_0\sigma^x, \\
	L &= \sigma^+.

where :math:`H` is the Hamiltonian of the two-level system, and :math:`L` is the Lindblad (or jump) operator.

The Lindblad equation is a non-unitary extension of the Liouville-von Neumann equation for te density matrix :math:`\rho(t)`:

.. math::
	\partial_t\rho(t) = -i[H,\rho] + 2\gamma\left(L\rho(t)L^\dagger - \frac{1}{2}\{L^\dagger L,\rho(t) \} \right).

The Lindblad equation for this example are also known as the optical Bloch equations.

Below, we provide two ways to define the function for the Lindblad ODE. The first version is very intuitive, but slow. The second version
uses the `matvec()` function and is fast (but not necessarily memory efficient for large systems).


Script
------

:download:`download script <../../../examples/scripts/example17.py>`

.. literalinclude:: ../../../examples/scripts/example17.py
	:linenos:
	:language: python
	:lines: 1-

