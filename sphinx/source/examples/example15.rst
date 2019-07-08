.. _example15-label:

Spin-1/2 system with sublattice particle consevation
--------

This example makes use of the `user_basis` class of `basis_general` to define the Hamiltonian

.. math::
	H = \sum_{j=0}^{N/2} t (\tau^+_{j+1}\tau^-_j + \sigma^+_{j+1}\sigma^-_j + \mathrm{h.c.}) + U \sigma^z_j\tau^z_j

where :math:`\sigma` and :math:`\tau` describe hardcore bosons on the two legs of a ladder geometry. Note that particles cannot be exchanged between the
legs of the ladder, which allows to further reduce the Hilbert space dimension.

Please consult this post: :ref:`user_basis-label`, with more detailed explanations for using the `user_basis` class.

Script
------

:download:`download script <../../../examples/scripts/example15.py>`

.. literalinclude:: ../../../examples/scripts/example15.py
	:linenos:
	:language: python
	:lines: 1-

