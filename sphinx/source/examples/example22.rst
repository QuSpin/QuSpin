.. _example22-label:

Efficient time evolution: expm_multiply_parallel
------------------------------------------------

In this example, we promote the usage of the function `tools.evolution.expm_multiply_parallel()`, designed to compute
matrix exponentials for **static** Hamiltonians (but note that it can also be used for piecel-wise constant dynamics, e.g. for periodically-driven systems, to reach long times for larger system sizes).

This function is an omp-parallelized implementation of [arXiv paper here]. 



Script
------

:download:`download script <../../../examples/scripts/example22.py>`

.. literalinclude:: ../../../examples/scripts/example22.py
	:linenos:
	:language: python
	:lines: 1-