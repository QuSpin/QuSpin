.. _example0-label:

Exact Diagonalisation of Spin Hamiltonians
------------------------------------------

This example shows how to code up the Heisenberg Hamiltonian:

.. math::
	H = \sum_{j=0}^{L-2}\frac{J_{xy}}{2}\left(S^+_{j+1}S^-_{j} + \mathrm{h.c.}\right) + J_{zz}S^z_{j+1}S^z_{j} + h_z\sum_{j=0}^{L-1}S^z_{j}.

Details about the code below can be found in `this tutorial paper <https://scipost.org/SciPostPhys.2.1.003>`_.


Script
------

:download:`download script <../../../examples/scripts/example0.py>`


.. literalinclude:: ../../../examples/scripts/example0.py
	:linenos:
	:language: python
	:lines: 1-
