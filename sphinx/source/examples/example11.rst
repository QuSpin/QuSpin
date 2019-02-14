.. _example11-label:

Sampling Expectation Values of Obsevables
-------------------------------------------

:download:`download script <../../../examples/scripts/example11.py>`

The example below demonstrates how to use the `*_basis_general` methods `Op_bra_ket()` and `representative()` which do not require computing all basis states.

Physics Setup
-------------

The expectation value of an operator :math:`H` in a state :math:`\psi` can be written as

.. math::
	\langle\psi|H|\psi\rangle = \sum_{s,s'} \psi_s^\ast H_{ss'}\psi_{s'} = \sum_s |\psi_s|^2 E_{s},\qquad E_s = \frac{1}{\psi_s}\sum_{s'}H_{ss'}\psi_{s'},

where :math:`\{|s\rangle\}_s` can be any basis, in particular the Fock (:math:`z`-) basis used in QuSpin. 

The above expression suggests that one can use sampling methods, such as Monte Carlo, to estimate the expectation value of :math:`\langle\psi|H|\psi\rangle` using the quantity :math:`E_s` (sometimes referred to as local energy) evaluated in samples drawn from the probability distribution :math:`p_s=|\psi_s|^2`. If we have: (i) a function :math:`s\mapsto\psi_s` which compues the amplitudes for every spin configuration :math:`s`, and (ii) the matrix elements :math:`H_{ss'}`, then 

.. math::
	\langle\psi|H|\psi\rangle \approx \frac{1}{N}\sum_{s\in\mathcal{S}} E_s,  

where :math:`\mathcal{S}` contains :math:`N` spin configurations sampled from :math:`p_s=|\psi_s|^2`.

Since this procedure does not require the the state :math:`|\psi\rangle` to be normalized, ideas along these lines allow to look for variational approximations to the wavefunction :math:`\psi_s` in large system sizes, for instance with the help of restricted Boltzmann machines [`arXiv:1606.02318 <https://arxiv.org/abs/1606.02318>`_].

In the example below, we assume that we already have a quantum state :math:`\psi_s` in the Fock basis, and we sample the expectation value of an operator `H` using the `*_basis_general` methods `Op_bra_ket()` and `representative()` [cf. code lines 120, 139, 167 below]. These methods do not require to compute the full basis, and thus allow to reach system sizes beyond exact diagonalization. 


Script
------

.. literalinclude:: ../../../examples/scripts/example11.py
	:linenos:
	:language: python
	:lines: 1-
