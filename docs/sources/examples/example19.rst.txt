:orphan:

.. _example19-label:


Autocorrelation functions using symmetries
------------------------------------------

This example demonstrates how to use the general basis function method `Op_shift_sector()` to compute autocorrelation functions of operators in the Heisenberg model on a 1d chain.


.. math::
	H = J\sum_{j=0}^{L-1} S^+_{j+1}S^-_{j} + \mathrm{h.c.} + S^z_{j+1}S^z_j,

where :math:`S_j` is the spin-1/2 operator on lattice site :math:`j`; we use periodic boundary conditions. We are interested in the following autocorrelation function:

.. math::
	C(t) = \langle\psi_\mathrm{GS}|\mathcal{O}^\dagger(t)\mathcal{O}(0)|\psi_\mathrm{GS}\rangle = \langle\psi_\mathrm{GS}|e^{+i t H}\mathcal{O}^\dagger e^{-i t H}\mathcal{O}|\psi_\mathrm{GS}\rangle

where :math:`\mathcal{O} = \sqrt{2}S^z_{j=0}`.

For the purpose of computing the autocorrelation function, it is advantageous to split the calculation as follows:

.. math::
	C(t) = \langle\psi_\mathrm{GS}(t)|\mathcal{O}^\dagger|(\mathcal{O}\psi_\mathrm{GS})(t)\rangle,

where :math:`|\psi_\mathrm{GS}(t)\rangle = e^{-i t H}|\psi_\mathrm{GS}\rangle` (this is a trivial phase factor, but we keep it here for generality), and :math:`|(\mathcal{O}\psi_\mathrm{GS})(t)\rangle = e^{-i t H}\mathcal{O}|\psi_\mathrm{GS}\rangle`.


In the example below, we compute :math:`C(t)` (i) in real space, and (ii) in momentum space. 

(i) The real space calculation is straightforward, but it does not make use of the symmetries in the Heisenberg Hamiltonian. 

(ii) The momentum-space calculation is interesting, because the operator :math:`\mathcal{O}` carries momentum itself; thus, when it acts on the (time-evolved) ground state, it changes the momentum of the state. In QuSpin, this operation can be done using the general basis class method `Op_shift_sector()`, *provided we know ahead of time exactly by how much the momentum of the state will be changed*. To understand how the symmetry calculation works, consider the more general nonequal-space, nonequal-time correlation function:

.. math::
	C_r(t) = \langle\psi_\mathrm{GS}|\mathcal{O}_r^\dagger(t)\mathcal{O}_0(0)|\psi_\mathrm{GS}\rangle = \frac{1}{L}\sum_{j=0}^{L-1}\langle\psi_\mathrm{GS}|\mathcal{O}_{j+r}^\dagger(t)\mathcal{O}_{j}(0)|\psi_\mathrm{GS}\rangle,

where in the second equality we explicitly used translation invariance. Using the Fourier transform :math:`\mathcal{O}_q(t) = 1/\sqrt{L}\sum_{j=0}^{L-1}\mathrm{e}^{-i \frac{2\pi q}{L} j}\mathcal{O}_j(t)`, we arrive at

.. math::
	C_r(t) = \frac{1}{L}\sum_{q=0}^{L-1} \mathrm{e}^{+i\frac{2\pi q}{L} r}\mathcal{O}^\dagger_q(t)\mathcal{O}_q(0).

Substituting the Fourier transform, and the epression for :math:`\mathcal{O}_j=\sqrt{2}S^z_j` from above, setting :math:`r=0`, we arrive at


.. math::
	C_{r=0}(t) = \sum_{q=0}^{L-1}  \left(\frac{1}{L}\sum_{j=0}^{L-1}\mathrm{e}^{-i \frac{2\pi q}{L} j} \sqrt{2}S^z_{j}(t)  \right) \times \left(\frac{1}{L}\sum_{j'=0}^{L-1}\mathrm{e}^{-i \frac{2\pi q}{L} j' } \sqrt{2}S^z_{j'}(0)  \right)

which is the expression we use in the code snippet below (note that since :math:`S^z` is hermitian, it does not matter whether we use :math:`\mathrm{e}^{-i\dots}` or :math:`\mathrm{e}^{+i\dots}` here). 


More generally, the operator Fourier decomposition of an operator :math:`\mathcal{O}_l` with respect to any discrete symmetry transformation :math:`Q` of periodicity/cyclicity :math:`m_Q` (:math:`Q^{m_Q}=1`), and eigenvalues :math:`\{\exp(-2\pi i q/m_Q)\}_{q=0}^{m_Q}`, is given by:

.. math::
	\mathcal{O}_{q} = \frac{1}{\sqrt m_Q}\sum_{j=0}^{m_Q-1} \mathrm{e}^{-i \frac{2\pi q}{m_Q} j} (Q^j)^\dagger \mathcal{O}_{l} Q^j.

For instance, if :math:`Q` is the translation operator then :math:`(Q^j)^\dagger \mathcal{O}_{l} Q^j = \mathcal{O}_{l+j}`; if :math:`Q` is the reflection about the middle of the chain: :math:`(Q)^\dagger \mathcal{O}_{l} Q = \mathcal{O}_{L-1-l}`, etc. The most general symmetry expression for the correlator then reads

.. math::
	C_{r}(t) &= \langle\psi_\mathrm{GS}|\mathcal{O}_{l+r}^\dagger(t)\mathcal{O}_l(0)|\psi_\mathrm{GS}\rangle 
	=  \frac{1}{m_Q}\sum_{j=0}^{m_Q-1}\langle\psi_\mathrm{GS}|(Q^j)^\dagger\mathcal{O}_{r+l}^\dagger(t)Q^j(Q^j)^\dagger\mathcal{O}_{l}(0)Q^j|\psi_\mathrm{GS}\rangle \\
	&= 
	\sum_{q=0}^{m_Q-1} \mathrm{e}^{+i\frac{2\pi q}{L} r} 
	\left(\frac{1}{m_Q}\sum_{j=0}^{m_Q-1}\mathrm{e}^{+i \frac{2\pi q}{L} j} (Q^j)^\dagger \mathcal{O}_{l}^\dagger(t) Q^j  \right) 
	\times 
	\left(\frac{1}{m_Q}\sum_{j'=0}^{m_Q-1}\mathrm{e}^{-i \frac{2\pi q}{m_Q} j' } (Q^{j'})^\dagger \mathcal{O}_{0}(l) Q^{j'}  \right)

This allows to exploit more symmetries of the Heisenberg model, if needed. An example of how this works for `Op_shift_sector()`, for reflection symmetry, is shown `here <https://github.com/weinbe58/QuSpin/blob/dev_0.3.4/tests/Op_shift_sector_test.py#L58>`_.

Script
------

:download:`download script <../../../examples/scripts/example19.py>`

.. literalinclude:: ../../../examples/scripts/example19.py
	:linenos:
	:language: python
	:lines: 1-

