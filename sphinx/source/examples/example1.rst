:orphan:

.. _example1-label:


Adiabatic Control of Parameters in Many-Body Localised Phases
-------------------------------------------------------------

This example shows how to code up the time-dependent disordered spin Hamiltonian:

.. math::
	H(t) &=& \sum_{j=0}^{L-2}\frac{J_{xy}}{2}\left(S^+_{j+1}S^-_{j} + \mathrm{h.c.}\right) + J_{zz}(t)S^z_{j+1}S^z_{j} + \sum_{j=0}^{L-1}h_jS^z_{j},\nonumber\\
	J_{zz}(t) &=&(1/2 + vt)J_{zz}(0).

Details about the code below can be found in `SciPost Phys. 2, 003 (2017) <https://scipost.org/10.21468/SciPostPhys.2.1.003>`_.


Script
------

:download:`download script <../../../examples/scripts/example1.py>`


.. literalinclude:: ../../../examples/scripts/example1.py
	:linenos:
	:language: python
	:lines: 1-
