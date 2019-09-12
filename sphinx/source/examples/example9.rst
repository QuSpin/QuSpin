.. _example9-label:

Integrability Breaking and Thermalising Dynamics in the Translationally Invariant 2D Transverse-Field Ising Model
-----------------------------------------------------------------------------------------------------------------

This example shows how to code up the time-periodic 2D transverse-field Ising Hamiltonian:

.. math::
	H(t)=\Bigg\{ \begin{array}{cc}
	H_{zz} +AH_x,& \qquad t\in[0,T/4), \\
	H_{zz} -AH_x,& \qquad t\in[T/4,3T/4),\\
	H_{zz} +AH_x,& \qquad t\in[3T/4,T)
	\end{array}

where

.. math::
	H_{zz} = -\sum_{\langle ij\rangle} S^z_iS^z_{j}, \qquad H_{x} = -\sum_{i}S^x_i.

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download script <../../../examples/scripts/example9.py>`

.. literalinclude:: ../../../examples/scripts/example9.py
	:linenos:
	:language: python
	:lines: 1-
