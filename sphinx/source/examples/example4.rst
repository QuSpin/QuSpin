.. _example4-label:

The Spectrum of the Transverse Field Ising Model and the Jordan-Wigner Transformation
-------------------------------------------------------------------------------------

This example shows how to code up the transverse-field Ising chain and the Jordan-Wigner-equivalent fermion p-wave superconductor:

.. math::
	H&=&\sum_{j=0}^{L-1}-J\sigma^z_{j+1}\sigma^z_j - h\sigma^x_j, \nonumber\\
	H&=&\sum_{j=0}^{L-1}J\left(-c^\dagger_jc_{j+1} + c_jc^\dagger_{j+1} \right) +J\left( -c^\dagger_jc^\dagger_{j+1} + c_jc_{j+1}\right) + 2h\left(n_j-\frac{1}{2}\right).

Details about the code below can be found in `SciPost Phys. 7, 020 (2019) <https://scipost.org/10.21468/SciPostPhys.7.2.020>`_.


Script
------

:download:`download <../../../examples/scripts/example4.py>`

.. literalinclude:: ../../../examples/scripts/example4.py
	:linenos:
	:language: python
	:lines: 1-
