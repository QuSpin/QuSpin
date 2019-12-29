.. _example20-label:

Lanczos module: time-evolution and ground state search
------------------------------------------------------

This example demonstrates how to use the `Lanczos` submodule of the `tools` module to do time evolvution and ground state search in the Heisenberg model:


.. math::
	H = J\sum_{j=0}^{N-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j.

where :math:`S_j` is the spin-1/2 operator on lattice site :math:`j`; we use periodic boundary conditions.

The Lanczos decomposition is defined as ...



Script
------

:download:`download script <../../../examples/scripts/example20.py>`

.. literalinclude:: ../../../examples/scripts/example20.py
	:linenos:
	:language: python
	:lines: 1-

