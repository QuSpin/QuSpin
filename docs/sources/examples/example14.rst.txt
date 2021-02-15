:orphan:

.. _example14-label:


Quantum scars Hamiltonian: spin-1/2 system with Hilbert space constraint
------------------------------------------------------------------------

This example makes use of the `user_basis` class to define the Hamiltonian

.. math::
	H = \sum_j P_{j-1} \sigma^x_j P_{j+1}, \qquad P_j = |\downarrow_j\rangle\langle\downarrow_j|

The projector operators :math:`P_j` are built in the definition of the basis for the constrained Hibert space.

Please consult this post -- :ref:`user_basis-label` -- for more detailed explanations on using the `user_basis` class.

Script
------

:download:`download script <../../../examples/scripts/example14.py>`

.. literalinclude:: ../../../examples/scripts/example14.py
	:linenos:
	:language: python
	:lines: 1-

