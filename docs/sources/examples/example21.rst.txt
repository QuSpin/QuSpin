.. _example21-label:

Lanczos module: finite-temperature calculations
-----------------------------------------------

Here we use the transverse-field Ising chain with periodic boundary conditions as an example of how to use QuSpin's Lanczos module to perform finite-temperature calculations. 

We consider the transverse-field Ising model, governed by the Hamiltonian:

.. math::
    H(s) = -s\sum_{i=0}^{L-1} \sigma_i^z\sigma_{i+1}^z - (1-s)\sum_{i=0}^{L-1} \sigma_i^x

with periodic boundary conditions. This system is not ordered at finite temperature and, therefore, we expect to see a paramagnetic phase for any finite temperature. We are interested in computing the finite-temperature expectation value of the squared magnetization:

.. math::
   \langle M^2 \rangle_\beta = \frac{Tr\left(e^{-\beta H/2} M^2 e^{-\beta H/2}\right)}{Tr\left(e^{-\beta H} \right)},\qquad M = \frac{1}{L}\sum_{i=0}^{L-1}\sigma^z_j,

where :math:`\beta` is the inverse temperature. 


The example script below demonstrates the use of both the finite-temperature (`FTLM_static_iteration`) and low-temperature (`LTLM_static_iteration`) Lanczos methods, which gives the user the opportunity to compare the approximate methods at high and low temperatures. Following the discussion under `quspin.tools.lanczos.LTLM_static_iteration` and `quspin.tools.lanczos.FTLM_static_iteration`, we make use of the approximation:

.. math::
	\langle O\rangle_\beta \approx \frac{\overline{\langle O\rangle_r}}{\overline{\langle I\rangle_r}},

where the notation :math:`\overline{\;\cdot\;}` denotes the average over randomly drawn states :math:`|r\rangle`, :math:`\langle O\rangle_r=\langle r| e^{-\beta H/2} O e^{-\beta H/2}|r\rangle`, and :math:`I` is the identity operator.

The first part of the script below define various helper functions and classes that are useful for the calculation. The class `lanczos_wrapper` is a simple class that has all the necessary attributes to calculate the Lanczos basis. The idea here is that we would like to calculate the Lanczos basis for a particular value of the parameter :math:`s` without having to calculate the Hamiltonian at that fixed value every time. This calculation is accomplished by using quspin's `quantum_operator` class combined with the wrapper class: the quantum operator stores both the Ising and the transverse-field Hamiltonians as two separate terms. Then, the wrapper class provides an interface that calls the `dot` method of the quantum operator with the parameters specified, which is required by quspin's Lanczos methods. 

To apply finite-temperature Lanczos methods, we need to average over random states. The loop in the code below is where this calculation for random state vectors occurs. First. a random vector is generated. Then, using that vector, the Lanczos basis is created. With that Lanczos basis, the calculation of the FTLM and LTLM is performed, and we store the results in a list. Finally, the results are averaged, using the bootstrap helper function to calculate the error bars. The results are plotted and compared against exact resultsobtained using exact diagonalization. 

Script
------

:download:`download script <../../../examples/scripts/example21.py>`

.. literalinclude:: ../../../examples/scripts/example21.py
	:linenos:
	:language: python
	:lines: 1-