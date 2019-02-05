.. _example11-label:

Parallel Computing using OpenMP. Sampling Expecation Values of Obsevables.
--------------------------------------------------------------------------

:download:`download script <../../../examples/scripts/example11.py>`

The example below demonstrates how to use:
	* the OpenMP version of quspin for parallel computing,
	* the `*_basis_general` methods `Op_bra_ket()` and `representative()` which do not require computing basis.

Physics Problem
----------------

The expectation value of an operator :math:`H` in a state :math:`\psi` can be written as

.. math::
	\langle\psi|H|\psi\rangle = \sum_{s,s'} \psi_s^\ast H_{ss'}\psi_{s'} = \sum_s |\psi_s|^2 E_{s},\qquad E_s = \frac{1}{\psi_s}\sum_{s'}H_{ss'}\psi_{s'},

where :math:`\{|s\rangle\}_s` can be any basis, in particular the Flock (:math:`z`-) basis used in QuSpin. 

The above expression suggests that one can use sampling methods, such as Monte Carlo, to estimate the expectation value of the quantity :math:`E_s` (sometimes referred to as local energy) in the probability distribution :math:`p_s=|\psi_s|^2`. If we have: (i) a function which compues the amplitudes :math:`s\mapsto\psi_s` for every spin configuration :math:`s`, and (ii) the matrix elements :math:`H_{ss'}`, then 

.. math::
	\langle\psi|H|\psi\rangle \approx \frac{1}{N}\sum_{s\in\mathcal{S}} E_s,  

where :math:`\mathcal{S}` contains :math:`N` spin configurations sampled from :math:`p_s=|\psi_s|^2`.

Such ideas allow to look for variational approximations to the wavefunction :math:`\psi_s` in large system sizes, based for instance on restricted boltzmann machines, see `arXiv:1606.02318 <https://arxiv.org/abs/1606.02318>`_.

In this example, we assume that we have a quantum state :math:`\psi_s` in the Fock basis, and we sample the expectation value of an operator `H` using the `*_basis_general` methods `Op_bra_ket()` and `representative()`. These methods do not require to compute the basis, and thus can be combined to reach system sizes beyond exact diagonalization. 


Parallel Computing in QuSpin using OpenMP
-----------------------------------------

We also use the opportunity to show how to speed up QuSpin code using OpenMP. To install quspin with OpenMP support using anaconda, run 
::
	$ conda install -c weinbe58 quspin-omp

**Note:** there is a common problem with using OpenMP problem on OSX in anaconda packages, which may induce an error outside QuSpin. However, this error can be disabled [at one own's risk] until it is officially fixed, see line in the code ?? below. 

.. literalinclude:: ../../../examples/scripts/example11.py
	:linenos:
	:language: python
	:lines: 1-
