.. _example11-label:

Parallel Computing using OpenMP. Sampling Expectation Values of Obsevables.
--------------------------------------------------------------------------

:download:`download script <../../../examples/scripts/example11.py>`

The example below demonstrates how to use:
	* the OpenMP version of quspin for parallel computing [code lines 24, 25 below],
	* the `*_basis_general` methods `Op_bra_ket()` and `representative()` which do not require computing basis [code lines 126, 145, 173 below].

Physics Setup
-------------

The expectation value of an operator :math:`H` in a state :math:`\psi` can be written as

.. math::
	\langle\psi|H|\psi\rangle = \sum_{s,s'} \psi_s^\ast H_{ss'}\psi_{s'} = \sum_s |\psi_s|^2 E_{s},\qquad E_s = \frac{1}{\psi_s}\sum_{s'}H_{ss'}\psi_{s'},

where :math:`\{|s\rangle\}_s` can be any basis, in particular the Flock (:math:`z`-) basis used in QuSpin. 

The above expression suggests that one can use sampling methods, such as Monte Carlo, to estimate the expectation value of :math:`\langle\psi|H|\psi\rangle` using the quantity :math:`E_s` (sometimes referred to as local energy) evaluated in samples drawn from the probability distribution :math:`p_s=|\psi_s|^2`. If we have: (i) a function which compues the amplitudes :math:`s\mapsto\psi_s` for every spin configuration :math:`s`, and (ii) the matrix elements :math:`H_{ss'}`, then 

.. math::
	\langle\psi|H|\psi\rangle \approx \frac{1}{N}\sum_{s\in\mathcal{S}} E_s,  

where :math:`\mathcal{S}` contains :math:`N` spin configurations sampled from :math:`p_s=|\psi_s|^2`.

Since this procedure does not require the the state :math:`|\psi\rangle` to be normalized, ideas along these lines allow to look for variational approximations to the wavefunction :math:`\psi_s` in large system sizes, based for instance on restricted Boltzmann machines [`arXiv:1606.02318 <https://arxiv.org/abs/1606.02318>`_].

In the example below, we assume that we already have a quantum state :math:`\psi_s` in the Fock basis, and we sample the expectation value of an operator `H` using the `*_basis_general` methods `Op_bra_ket()` and `representative()`. These methods do not require to compute the full basis, and thus allow to reach system sizes beyond exact diagonalization. 


Parallel Computing in QuSpin using OpenMP
-----------------------------------------

We also use the opportunity to show how to speed up QuSpin code using OpenMP. To install quspin with OpenMP support using anaconda, run 
::
	$ conda install -c weinbe58 omp quspin

**Note:** there is a common problem with using OpenMP problem on OSX in anaconda packages for ython 3, which may induce an error unrelated to QuSpin. However, this error can be disabled [at one's own risk(!)] until it is officially fixed, see code line 25 below. 

Script
------

.. literalinclude:: ../../../examples/scripts/example11.py
	:linenos:
	:language: python
	:lines: 1-
