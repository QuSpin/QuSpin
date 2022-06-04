:orphan:

.. _example7-label:

The Bose-Hubbard Model on Translationally Invariant Ladder
----------------------------------------------------------

This example shows how to code up the Bose-Hubbard model on a ladder geometry:

.. math::
	H = -J\sum_{\langle ij\rangle} \left(b_i^\dagger b_j + \mathrm{h.c.}\right) + \frac{U}{2}\sum_{i}n_i(n_i-1).

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example7.py>`

.. literalinclude:: ../../../examples/scripts/example7.py
	:linenos:
	:language: python
	:lines: 1-
