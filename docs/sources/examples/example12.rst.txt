:orphan:

.. _example12-label:


Parallel Computing in QuSpin
----------------------------

:download:`download script <../../../examples/scripts/example12.py>`

This example shows how to speed up QuSpin code via multi-threading by using

* OpenMP,
* Intel's MKL library for NumPy/SciPy (which is Anaconda's default NumPy version, starting from Anaconda 2.5 onwards).


To install quspin with OpenMP support using anaconda (see also :ref:`parallelization-label`), run 
::

	$ conda install -c weinbe58 omp quspin

The example below demonstrates how to use the OpenMP version of quspin for parallel computing. It is set up in such a way that the number of OpenMP and MKL threads is controlled from the command line [cf. code lines 8,9]. To run the script, run
::

	$ python example12.py ${OMP_NUM_THREADS} ${MKL_NUM_THREADS}

You can directly compare the speed for different values of the number of threads [make sure your machine's processor has more than one core]
::

	$ python example12.py 1 1 # single-threaded computation
	$ python example12.py 4 1 # multi-threaded OpenMP computation, speedup for basis functions, evolution, and matrix-vector multiplication
	$ python example12.py 1 4 # multi-threaded MKL computation, speedup for diagonalization-like routines
	$ python example12.py 4 4 # simulaneous OpenMP and MKL multi-threading speedup

Notice how, as explained in :ref:`parallelization-label`, `OMP_NUM_THREADS` improves the speed of the basis computation (code line 43), and the time evolution (code line 65), while `MKL_NUM_THREADS` improves the speed of the exact diagonalization step (code line 60).

**Note:** there is a common problem with using OpenMP on OSX with anaconda packages for Python 3, which may induce an error unrelated to QuSpin:
::

	$ OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.

However, this error can be disabled [at one's own risk!] until it is officially fixed, see code line 7 below. 

Script
------

:download:`download script <../../../examples/scripts/example12.py>`

.. literalinclude:: ../../../examples/scripts/example12.py
	:linenos:
	:language: python
	:lines: 1-