.. _example8-label:

The Gross-Pitaevskii Equation and Nonlinear Time Evolution
----------------------------------------------------------


This example shows how to code up the Gross-Pitaevskii equation for a system in a one-dimensional lattice subject to a harmonic trapping potential:

.. math::
	i\partial_t\psi_j(t) &=& -J\left( \psi_{j-1}(t) + \psi_{j+1}(t)\right) + \frac{1}{2}\kappa_\mathrm{trap}(t)(j-j_0)^2\psi_j(t) + U|\psi_j(t)|^2\psi_j(t), \nonumber \\
	\kappa_\mathrm{trap}(t)&=&(\kappa_f-\kappa_i)t/t_\mathrm{ramp}+ \kappa_i.

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example8.py>`

.. literalinclude:: ../../../examples/scripts/example8.py
	:linenos:
	:language: python
	:lines: 1-
