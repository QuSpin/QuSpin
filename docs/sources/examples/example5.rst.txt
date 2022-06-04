:orphan:

.. _example5-label:


Free Particle Systems: the Fermionic SSH Chain
----------------------------------------------

This example shows how to code up the Su-Schrieffer-Heeger chain:

.. math::
	H = \sum_{j=0}^{L-1} -(J+(-1)^j\delta J)\left(c_jc^\dagger_{j+1} - c^\dagger_{j}c_{j+1}\right) + \Delta(-1)^jn_j.

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example5.py>`

.. literalinclude:: ../../../examples/scripts/example5.py
	:linenos:
	:language: python
	:lines: 1-
