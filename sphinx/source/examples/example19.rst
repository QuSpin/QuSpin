.. _example19-label:

Autocorrelation functions using symmetries: the Heisenberg Model
----------------------------------------------------------------

This example demonstrates how to use the basis general function method `Op_shift_sector()` to compute autocorrelation functions of operators in the Heisenberg model on a 1d chain.


.. math::
	H = -J\sum_{j=0}^{N-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j.

where :math:`S_j` is the spin-1/2 operator on lattice site :math:`j`; we use periodic boundary conditions. We are interested in the following autocorrelation function:

.. math::
	C(t) = \langle\psi_\mathrm{GS}|\mathcal{O}(t)\mathcal{O}(0)|\psi_\mathrm{GS}\rangle = \langle\psi_\mathrm{GS}|e^{+i t H}\mathcal{O} e^{-i t H}\mathcal{O}|\psi_\mathrm{GS}\rangle

where :math:`\mathcal{O} = \sqrt{2}S^z_{l=0}`.

For the purpose of computing the autocorrelation function, it is advantageous to split the calculation as follows:

.. math::
	C(t) = \langle\psi_\mathrm{GS}(t)|\mathcal{O}|(\mathcal{O}\psi_\mathrm{GS})(t)\rangle,

where :math:`|\psi_\mathrm{GS}(t)\rangle = e^{-i t H}|\psi_\mathrm{GS}\rangle` (this is a trivial phase factor, but we keep it here for generality), and :math:`|(\mathcal{O}\psi_\mathrm{GS})(t)\rangle = e^{-i t H}\mathcal{O}|\psi_\mathrm{GS}\rangle`.


In the example below, we compute :math:`C(t)` (i) in real space, and (ii) in momentum space. 

The real space calculation is straightforward, but it does not make use of the symmetries in the Heisenberg Hamiltonian. 

The momentum-space calculation is interesting, because the operator :math:`\mathcal{O}` carries momentum itself; thus, when it acts on the (time-evolved) ground state, it changes the momentum of the state. In QuSpin, this operation can be done using the general basis class method `Op_shift_sector()`, *provided we know ahead of time exactly by how much the momentum will be changed*. For this purposes, we Fourier decompose the operator as follows:  

.. math::
	\mathcal{O} = \sqrt{2}S^z_{l=0} = \frac{1}{\sqrt{L}}\sum_{q=0}^{N-1} \sqrt{2}S^z_q = \sqrt{2}\frac{1}{L}\sum_{q=0}^{N-1} \sum_{j=0}^{N-1} e^{-i \frac{2\pi q}{L} j} S^z_j 


More generally, the operator Fourier decomposition with respect to any discrete symmetry transformation :math:`Q` of periodicity/cyclicity :math:`m_Q` (:math:`Q^{m_Q}=1`), and eigenvalues :math:`\{\exp(-2\pi i q/m_Q)\}_{q=0}^{m_Q}`, *for one-body operators*, is given by:

.. math::
	S^z_{l=0} = \frac{1}{m_Q}\sum_{q=0}^{m_Q} \sum_{j=0}^{m_Q} e^{-i \frac{2\pi q}{m_Q} j} (Q^j)^\dagger S^z_{l=0} Q^j.

This allows to exploit more symmetries of the Heisenberg model.

Script
------

:download:`download script <../../../examples/scripts/example19.py>`

.. literalinclude:: ../../../examples/scripts/example19.py
	:linenos:
	:language: python
	:lines: 1-

