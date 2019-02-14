.. _example12-label:

Parallel Computing in QuSpin
--------

:download:`download script <../../../examples/scripts/example12.py>`

The example shows how to speed up QuSpin code 

* using OpenMP;
* using the Intel's MKL library for numpy (which is Anaconda's default NumPy version, starting from Anaconda 2.5 onwards).


To install quspin with OpenMP support using anaconda (see also :ref:`parallelization-label`), run 
::
	$ conda install -c weinbe58 omp quspin

**Note:** there is a common problem with using OpenMP problem on OSX in anaconda packages for Python 3, which may induce an error unrelated to QuSpin. However, this error can be disabled [at one's own risk!] until it is officially fixed, see code line 18 below. 

The example below demonstrates how to use the OpenMP version of quspin for parallel computing.


Script
------

.. literalinclude:: ../../../examples/scripts/example12.py
	:linenos:
	:language: python
	:lines: 1-