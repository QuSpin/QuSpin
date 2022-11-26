:orphan:

.. _example6-label:


Many-Body Localization in the Fermi-Hubbard Model
-------------------------------------------------

This example shows how to code up the disordered Fermi-Hubbard chain:

.. math::
	H = -J\sum_{i=0,\sigma}^{L-2} \left(c^\dagger_{i\sigma}c_{i+1,\sigma} - c_{i\sigma}c^\dagger_{i+1,\sigma}\right) +U\sum_{i=0}^{L-1} n_{i\uparrow }n_{i\downarrow } + \sum_{i=0,\sigma}^{L-1} V_i n_{i\sigma}.

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example6.py>`

.. literalinclude:: ../../../examples/scripts/example6.py
	:linenos:
	:language: python
	:lines: 1-
