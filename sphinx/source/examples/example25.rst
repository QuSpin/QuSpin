.. _example25-label:



Majorana Fermions: SYK Model
-----------------------------

In this example, we show how to define the Sachdev-Ye-Kitaev model with Majorana fermions.

The Hamiltonian is given by

.. math::
	H = -\frac{1}{4!}\sum_{i,j,k,l=0}^{L-1} J_{ijkl} c^x_{i}c^x_{j}c^x_{k}c^x_{l},

where :math:`J_{ijkl}` is a random all-to-all interaction strength which is normally distributed with zero mean and unit variance, and :math:`c_j^x` is a Majorana fermion satisfying :math:`c_j^x=(c_j^x)^\dagger`, :math:`(c_j^x)^2=1`, and :math:`\{c_i^x,c_j^x\}=0` for :math:`i\neq j`.   


The script below uses the `spinless_fermion_basis_general` class to define the above Hamiltonian. Note that, the same Hamiltonian can equivalently be built using the alternative Majorana operator :math:`c^y`.


Script
------

:download:`download script <../../../examples/scripts/example25.py>`

.. literalinclude:: ../../../examples/scripts/example25.py
	:linenos:
	:language: python
	:lines: 1-