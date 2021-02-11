:orphan:

.. _example26-label:


Direct Calculation of spectral functions using symmetries
---------------------------------------------------------

This example demonstrates how to use the general basis function method `Op_shift_sector()` to compute spectral functions directly without fourier transforming auto-correlation functions. The idea here is to use SciPy's iterative solvers for sparse linear systems to calculate the action of an inverted operator on a state. 

The spectral function in ground state can be written as:

.. math::
	C(\omega) = -\frac{1}{\pi}\langle 0|A^\dagger\frac{1}{\omega+i\eta + E_0 - H} A|0\rangle,

where :math:`\eta` is the boradening factor for the spectral peaks. For an operator :math:`A` that has quantum numbers that differ from the ground state :math:`|0\rangle` we can use `Op_shift_sector()` to calculate the a new ket: :math:`|A\rangle = A|0\rangle`. With this new ket, we can construct in :math:`H` in the new sector and solve the following equation:

.. math::
	(\omega+i\eta + E_0 - H)|x(\omega)\rangle = |A\rangle.

This equation can be used using the bi-conjugate gradient (scipy.sparse.linalg.bicg) method that is implemented in SciPy. In order to solve that equation with SciPy we define a new class that produces the action: :math:`(\omega \pm i\eta + E_0 - H)|u\rangle`. Now we can use :math:`|x(\omega)\rangle` to calculate the spectral function a given value of :math:`\omega`:

.. math::
	C(\omega) = -\frac{1}{\pi}\langle A|x(\omega)\rangle

Note that this method can also be used to calculate spectral functions for any state by replacing :math:`|0\rangle\rightarrow|\psi\rangle`. This may be useful for calculating spectral functions out-of-equalibrium by using :math:`|\psi\rangle` that has undergone some kind of time eolvution. 

In this example we look at the Heisenberg chain with :math:`J=1` periodic boundary conditions. Because we have periodic boundary conditions we can use the translation symmetry to reduce the hilbert spaces into blocks labeled by the total momentum. We limit the chain lengths to have :math:`L=4n` for :math:`n=1,2,3...` as this is a requirement to have the ground state be in the sector with total momentum 0. We calculate two different spectral functions:

.. math::
	G_{zz}(\omega,q) = \langle 0|S^{z}_{-q}\frac{1}{\omega+i\eta + E_0 - H}S^{z}_q|0\rangle

.. math::
	G_{+-}(\omega,q) = \langle 0|S^{-}_{-q}\frac{1}{\omega+i\eta + E_0 - H}S^{+}_q|0\rangle

where we have defined:

.. math::
	S^{z}_q = \frac{1}{\sqrt{N}}\sum_{r=0}^{L-1} \exp\left(-i\frac{2\pi q r}{L}\right) S^z_r

.. math::
	S^{\pm}_q = \frac{1}{\sqrt{2N}}\sum_{r=0}^{L-1} \exp\left(-i\frac{2\pi q r}{L}\right) S^{\pm}_r

Because the model has full SU(2) symmetry, we expect that the two spectral function should give the same result. We also exclude the spectral function for :math:`q=L/2` as the spectral peak is very large due to the quasi long-range antiferromagnetic order in the ground state. The parameter `on_the_fly` can be used to switch between using a `hamiltonian` object to a `quantum_LinearOperator` object to calculate the matrix vector product. One can also change the total spin of the local spin operators by changing `S`. 

Script
------

:download:`download script <../../../examples/scripts/example26.py>`

.. literalinclude:: ../../../examples/scripts/example26.py
	:linenos:
	:language: python
	:lines: 1-

