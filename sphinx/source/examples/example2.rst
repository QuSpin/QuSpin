.. _example2-label:

Heating in Periodically Driven Spin Chains
------------------------------------------

This example shows how to code up the periodically-driven spin Hamiltonian:

.. math::
	H(t) &=& \left\{ \begin{array}{cl}
	J\sum_{j=0}^{L-1} \sigma^z_j\sigma^z_{j+1} + h\sum_{j=0}^{L-1}\sigma^z, &  t\in[-T/4,\phantom{3}T/4] \\
	\kern-8em g\sum_{j=0}^{L-1} \sigma^x_j, &  t\in[\phantom{-} T/4,3T/4]
	\end{array} \right\}  \mathrm{mod}\ T,\nonumber\\
	&=& \sum_{j=0}^{L-1} \frac{1}{2}\left(J \sigma^z_j\sigma^z_{j+1} + h\sigma^z + g\sigma^x_j\right)
	+ \frac{1}{2}\text{sgn}\left[\cos\Omega t\right]\left( J \sigma^z_j\sigma^z_{j+1} + h\sigma^z - g\sigma^x_j \right).

Details about the code below can be found in `this tutorial paper <https://scipost.org/SciPostPhys.2.1.003>`_.


Script
------

:download:`download script <../../../examples/scripts/example2.py>`


.. literalinclude:: ../../../examples/scripts/example2.py
	:linenos:
	:language: python
	:lines: 1-
