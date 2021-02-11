:orphan:

.. _example26-label:


Direct Calculation of spectral functions using symmetries
---------------------------------------------------------

This example demonstrates how to use the general basis function method `Op_shift_sector()` to compute spectral functions directly without Fourier -transforming auto-correlation functions. The idea here is to use SciPy's iterative solvers for sparse linear systems to calculate the action of an inverted operator on a state. 

The spectral function in ground state :math:`|0\rangle` can be written as:

.. math::
	C(\omega) = -\frac{1}{\pi}\langle 0|A^\dagger\frac{1}{\omega+i\eta + E_0 - H} A|0\rangle,

where :math:`\eta` is the boradening factor for the spectral peaks. For an operator :math:`A` that has quantum numbers that differ from the ground state :math:`|0\rangle` we can use `Op_shift_sector()` to calculate the a new ket: :math:`|A\rangle = A|0\rangle`. With this new ket, we can construct the Hamiltonian :math:`H` in the new sector, and solve the following equation:

.. math::
	(\omega+i\eta + E_0 - H)|x(\omega)\rangle = |A\rangle.

This equation can be solved using the bi-conjugate gradient (cf. `scipy.sparse.linalg.bicg`) method that is implemented in SciPy. In order to solve that equation with SciPy we define a custom class (see code below) that produces the action :math:`(\omega \pm i\eta + E_0 - H)|u\rangle` for an arbitrary state :math:`|u\rangle`. Now we can use :math:`|x(\omega)\rangle` to calculate the spectral function a given value of :math:`\omega` as follows:

.. math::
	C(\omega) = -\frac{1}{\pi}\langle A|x(\omega)\rangle.

Note that this method can also be used to calculate spectral functions for any state by replacing :math:`|0\rangle\rightarrow|\psi\rangle`. This may be useful for calculating spectral functions away from equilibrium by using :math:`|\psi(t)\rangle` that has undergone some kind of time evolution (in the Schrödinger picture). 

In the example below, we look at the Heisenberg chain (:math:`J=1`) with periodic boundary conditions. The periodic boundary condition allows us to use translation symmetry to reduce the total Hilbert space into blocks labeled by the many-body momentum quantum number. We limit the chain lengths to :math:`L=4n` for :math:`n=1,2,3...`, since this is required to have the ground state in the zero-momentum sector (for other system sizes, the ground state of the isotropic Heisenberg chain has finite momentum). 

Below, we calculate two different spectral functions:

.. math::
	G_{zz}(\omega,q) = \langle 0|S^{z}_{-q}\frac{1}{\omega+i\eta + E_0 - H}S^{z}_q|0\rangle

.. math::
	G_{+-}(\omega,q) = \langle 0|S^{-}_{-q}\frac{1}{\omega+i\eta + E_0 - H}S^{+}_q|0\rangle

where we have defined:

.. math::
	S^{z}_q = \frac{1}{\sqrt{L}}\sum_{r=0}^{L-1} \exp\left(-i\frac{2\pi q r}{L}\right) S^z_r

.. math::
	S^{\pm}_q = \frac{1}{\sqrt{2L}}\sum_{r=0}^{L-1} \exp\left(-i\frac{2\pi q r}{L}\right) S^{\pm}_r

Because the model has full SU(2) symmetry, we expect that the two spectral functions should give the same result. We also exclude the spectral function for :math:`q=L/2` (:math:`\pi`-momentum) as the spectral peak is very large due to the quasi long-range antiferromagnetic order in the ground state. The variable `on_the_fly` can be used to switch between using a `hamiltonian` object to a `quantum_LinearOperator` object to calculate the matrix vector product. One can also change the total spin of the local spin operators by changing the variable `S` to compute the spectral function for higher-spin Heisenberg models.


Script
------

:download:`download script <../../../examples/scripts/example26.py>`

.. literalinclude:: ../../../examples/scripts/example26.py
	:linenos:
	:language: python
	:lines: 1-

