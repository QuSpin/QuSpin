:orphan:

.. _example23-label:


Gell-Mann Operators for Spin-1 systems
--------------------------------------

In this example, we show how to use the `user_basis` class to define the Gell-Mann operators to construct spin-1 Hamiltonians in QuSpin -- the SU(3) equivalent of the Pauli operators for spin-1/2. 

The eight generators of SU(3), are defined as

.. math::
	\lambda^1 = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix},\quad \lambda^2 = \begin{pmatrix} 0 & -i & 0 \\ i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \quad \lambda^3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix},

.. math::
	\lambda^4 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix},\quad \lambda^5 = \begin{pmatrix} 0 & 0 & -i \\ 0 & 0 & 0 \\ i & 0 & 0 \end{pmatrix},

.. math::
	\lambda^6 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix},\quad \lambda^7 = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -i \\ 0 & i & 0 \end{pmatrix}, \quad \lambda^8 = \frac{1}{\sqrt{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}


We now define them as operator strings for a custom spin-1 `user_basis` (please consult this post -- :ref:`user_basis-label` -- for more detailed explanations on using the `user_basis` class). In QuSpin, the basis constructor accepts operator strings of unit length; therefore, we use the operator strings `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8` to denote the operator :math:`\lambda^j`. 



Script
------

:download:`download script <../../../examples/scripts/example23.py>`

.. literalinclude:: ../../../examples/scripts/example23.py
	:linenos:
	:language: python
	:lines: 1-