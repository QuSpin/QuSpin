.. _example16-label:

Applying symmetries to reduce user-imported bases using QuSpin
--------

This example makes use of the `user_basis` class to take a user-imported basis consisting of integers, and use QuSpin to
compute a desired symmetry sector. 

In the example, we manually construct the basis for a two-legged ladder (and handle it as a user-imported array of basis states), and then use QuSpin to apply translation and parity (reflection) symmetries to reduce the Hilbert space.

Please consult this post: :ref:`user_basis-label`, with more detailed explanations on using the `user_basis` class.

Script
------

:download:`download script <../../../examples/scripts/example16.py>`

.. literalinclude:: ../../../examples/scripts/example16.py
	:linenos:
	:language: python
	:lines: 1-

