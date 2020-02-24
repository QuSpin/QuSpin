.. _example21-label:

Lanczos module: finite-temperature calculations
-----------------------------------------------

Here we use the transverse-field Ising chain with periodic boundary conditions as an example of how to use the Lanczos module to perform finite-temperature calculations. The Hamiltonian we consider is given by:

.. math::
    H(s) = -s\sum_{i=0}^{L-1} \sigma_i^z\sigma_{i+1}^z - (1-s)\sum_i \sigma_i^x

This system is not ordered at finite temperature, and therefore we expect to see a paramagnetic phase for any finite temperature. This script calculates both the finite-temperature and low-temperature Lanczos methods at the same time so that you can compare the results at high and low temperatures. 

The first parts of the script are defining various functions and classes that are useful for the calculation. The class `lanczos_wrapper` is a simple class that has all the necessary attributes to calculate the Lanczos basis. The idea here is that we would like to calculate the Lanczos basis for a particular value of the parameter :math:`s` without having to calculate the Hamiltonian at that fixed value. This calculation is accomplished by using the `quantum_operator` class combined with the wrapper class—the quantum operator stores both the Ising and the transverse-field Hamiltonians as two separate parameters. Then, the wrapper class provides an interface that calls the `dot` method of the quantum operator with the parameters specified. 

Next, the loop is where the calculation for random vectors occurs. The vector is generated, then using that vector, the Lanczos basis is created. With that Lanczos basis, the calculation of the FTLM and LTLM and store the results in a list. Finally, the results are averaged, using the bootstrap method to calculate the error bars, then the results are plotted and compared against exact results. 

Script
------

:download:`download script <../../../examples/scripts/example21.py>`

.. literalinclude:: ../../../examples/scripts/example21.py
	:linenos:
	:language: python
	:lines: 1-