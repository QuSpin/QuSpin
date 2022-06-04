:orphan:

.. _example13-label:


Fermi-Hubbard model **without** doubly-occupied sites
-----------------------------------------------------

This example shows how to construct the Fermi-Hubbard model of spinful fermions in the subspace without doubly-occupied sites using the `user_basis`. 

.. math::
	H = P\left[ -J\sum_{\langle i,j\rangle, \sigma}\left(c^\dagger_{i\sigma} c_{j\sigma} + \mathrm{h.c.}\right) - \mu\sum_{j,\sigma} n_{j\sigma} + U\sum_j n_{j\uparrow}n_{j\downarrow} \right]P,

where :math:`P` projects out doubly-occupied sites.


The procedure is the same as for the `spinful_fermion_basis_general` class, but it makes use of the optional argument `double_occupancy=False` in the basis constructor (added from v.0.3.3 onwards).

Arbitrary post selection of basis states (which generalizes eliminating double occupancies to more complex examples and models), can be done in QuSpin using the `user_basis` class, see :ref:`user_basis-label`.

Script
------

:download:`download script <../../../examples/scripts/example13.py>`

.. literalinclude:: ../../../examples/scripts/example13.py
	:linenos:
	:language: python
	:lines: 1-