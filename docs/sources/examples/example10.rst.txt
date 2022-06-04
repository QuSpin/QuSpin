:orphan:

.. _example10-label:


Out-of-Equilibrium Bose-Fermi Mixtures
--------------------------------------

This example shows how to code up the Bose-Fermi mixture Hamiltonian:

.. math::
	H(t) &=& H_\mathrm{b} + H_\mathrm{f}(t) + H_\mathrm{bf},\nonumber\\
	H_\mathrm{b} &=& -J_\mathrm{b}\sum_{j}\left(b^\dagger_{j+1}b_j + \mathrm{h.c.}\right) - \frac{U_\mathrm{bb}}{2}\sum_j n^\mathrm{b}_j + \frac{U_\mathrm{bb}}{2}\sum_j n^\mathrm{b}_jn^\mathrm{b}_j,\nonumber\\
	H_\mathrm{f}(t) &=& -J_\mathrm{f}\sum_{j}\left(c^\dagger_{j+1}c_j - c_{j+1}c^\dagger_j\right) + A\cos\Omega t\sum_j (-1)^j n^\mathrm{f}_j +  U_\mathrm{ff}\sum_j n^\mathrm{f}_jn^\mathrm{f}_{j+1},\nonumber\\
	H_\mathrm{bf} &=& U_\mathrm{bf}\sum_j n^\mathrm{b}_jn^\mathrm{f}_j

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example10.py>`

.. literalinclude:: ../../../examples/scripts/example10.py
	:linenos:
	:language: python
	:lines: 1-
