.. _example24-label:



Majorana Fermions: Spinless Fermi-Hubbard Model
-----------------------------------------------

In this example, we show how to use the `user_basis` class to define Majorana fermion operators. We then show how to construct the Hamiltonian for the spinless Fermi-Hubbard model in the Majorana representation. Starting from version 0.3.5 the Majorana operator strings `"x"` and `"y"` are also available in the `*fermion_basis_general` classes. 

Consider first a single fermion described by the creation and annihilation operators :math:`c^\dagger,c`. One can decompose one complex fermion into two real-valued Majorana fermion modes :math:`c^x, c^y`, defined by the operators

.. math::
	c^x = c + c^\dagger,\qquad c^y = -i(c - c^\dagger),

which obey the fermion commutation relations :math:`\{c^x, c^y\}=0`, :math:`\left(c^\alpha\right)^2 = 1`, for :math:`\alpha=x,y`.

The inverse transformation is given by

.. math::
	c = \frac{1}{2}\left( c^x + i c^y \right), \qquad c^\dagger = \frac{1}{2}\left( c^x - i c^y \right).

Here, we choose to denote the Majorana operators by :math:`c^x, c^y`, due to a formal relation with the Pauli matrices: :math:`c^x = \sigma^x`, :math:`c^y = -\sigma^y` (**NB:** the extra negative sign appearing for :math:`c^y` is because of the standard convention for the definition of Majorana modes), :math:`c^\dagger = \sigma^+`, and :math:`c = \sigma^-`. 

One can then generalize the Majorana decomposition to every site :math:`j` of the lattice: :math:`c^x_j, c^y_j`. 

To implement Majorana modes in QuSpin, we use the versatility of the `user_basis` to define the on-site action of the corresponding operators (please consult this post -- :ref:`user_basis-label` -- for more detailed explanations on using the `user_basis` class). To do this, we define new fermionic operator strings `x` and `y`, corresponding to the two Majorana modes :math:`c^x` and :math:c^y`, respectively. The definition of `x` and `y` follows the exact same procedure, as in the spin-1/2 basis (cf. :ref:`user-basis_example0-label`), with the notable difference that one has to accomodate for the `sign`, arising from counting the number of fermions up to the lattice site the operator is applied. 


Having access to Majorana operator strings allows us to implement Hamiltonians in the Majorana representation. To demonstrate this, we use the spinless Fermi-Hubbard model:

.. math::
	H = \sum_{j=0}^{L-1} J\left(c^\dagger_{j} c_{j+1} + \mathrm{h.c.}\right) + U n_{j}n_{j+1},

where :math:`J` is the hopping matrix element, and :math:`U` is the nearest-neighbor interaction strength.   

In the Majorana representation, the same Hamiltonian reads as

.. math::
	H = \sum_{j=0}^{L-1} \frac{iJ}{2}\left( c^x_j c^y_{j+1} - c^y_j c^x_{j+1} \right) + \frac{U}{4}\left(1 + 2ic^x_j c^y_j - c^x_j c^y_j c^x_{j+1} c^y_{j+1} \right).

The script below uses the `user_basis` class to define an enlarged fermion basis which allows to implement both Hamiltonians. We then compare the two matrices. Note that the `user_basis` readily allows one to apply symmetries to the Hamiltonian in the Majorana representation. 


Script
------

:download:`download script <../../../examples/scripts/example24.py>`

.. literalinclude:: ../../../examples/scripts/example24.py
	:linenos:
	:language: python
	:lines: 1-