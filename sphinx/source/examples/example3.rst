.. _example3-label:

Quantised Light-Atom Interactions in the Semi-classical Limit: Recovering the Periodically Driven Atom
------------------------------------------------------------------------------------------------------

This example shows how to code up the Hamiltonians:

.. math::
	H&=& \Omega a^\dagger a + \frac{A}{2}\frac{1}{\sqrt{N_\mathrm{ph}}}\left(a^\dagger + a\right)\sigma^x + \Delta\sigma^z, \nonumber\\
	H_\mathrm{sc}(t) &=& A\cos\Omega t\;\sigma^x + \Delta\sigma^z.

Details about the code below can be found in `this tutorial paper <https://scipost.org/SciPostPhys.2.1.003>`_.


Script
------

:download:`download script <../../../examples/scripts/example3.py>`


.. literalinclude:: ../../../examples/scripts/example3.py
	:linenos:
	:language: python
	:lines: 1-
