.. _example12-label:

Parallel Computing in QuSpin
--------

:download:`download script <../../../examples/scripts/example12.py>`

The example shows how to speed up QuSpin code 

* using OpenMP;
* using the Intel's MKL library for NumPy (which is Anaconda's default NumPy version, starting from Anaconda 2.5 onwards).


To install quspin with OpenMP support using anaconda (see also :ref:`parallelization-label`), run 
::
	$ conda install -c weinbe58 omp quspin

**Note:** there is a common problem with using OpenMP problem on OSX in anaconda packages for Python 3, which may induce an error unrelated to QuSpin. However, this error can be disabled [at one's own risk!] until it is officially fixed, see code line 7 below. 

The example below demonstrates how to use the OpenMP version of quspin for parallel computing. It is set up in such a way that the numner of OpenMP and MKL threads is controlled from the command line (see code lines 8,9). To run the script, run
::
	>>> python example12.py ${OMP_NUM_THREADS} ${MKL_NUM_THREADS}

You can directly compare the speed for different values of the number of threads
::
	>>> python example12.py 1 1 
	>>> python example12.py 4 1 
	>>> python example12.py 1 2 
	>>> python example12.py 4 2 

Script
------

.. literalinclude:: ../../../examples/scripts/example12.py
	:linenos:
	:language: python
	:lines: 1-