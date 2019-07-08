.. _example13-label:

Fermi-Hubbard model **without** doubly-occupied sites
--------

This example shows how to construct the Fermi-Hubbard model of spinful fermions in the subspace without doubly-occupied sites using the `user_basis`. The procedure is the same as for the `spinful_fermion_basis_general` class, but it makes use of the optional argument `double_occupancy=False` in the basis constructor (added from v.0.3.3 onwards).

Arbitrary post selection of basis states (which generalizes eliminating double occupancies), can be done in QuSpin using the `user_basis` class, see :ref:`user_basis-label`.

Script
------

:download:`download script <../../../examples/scripts/example13.py>`

.. literalinclude:: ../../../examples/scripts/example13.py
	:linenos:
	:language: python
	:lines: 1-

