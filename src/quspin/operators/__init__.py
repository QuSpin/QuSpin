"""
==========================================
Operators module (:mod:`quspin.operators`)
==========================================

Classes and functions for constructing and manipulating quantum operators, and implementing Schroedinger time evolution.

Many-body operators in QuSpin are defined by a string of letters representing the operator types, 
together with a list which holds the indices for the lattice sites that each operator acts on. 

For example, in a spin-1/2 system we can represent any multi-spin operator as:

.. math::
      \\begin{array}{cccc}
         \\text{operator string}  &   \\text{site-coupling list}  &  &   \\text{spin operator}   \\newline 
         \\mu_1,\\dots,\\mu_n     &   [J,i_1,\\dots,i_n]     &  \\Leftrightarrow  &   J\\sigma_{i_1}^{\\mu_1}\\cdots\\sigma_{i_n}^{\\mu_n}      \\newline
      \\end{array}

where :math:`\\mu_i` can be "I", z", "+", "-", "x" or "y".
Here, :math:`\\sigma_{i_n}^{\\mu_n}` is the Pauli spin operator acting on lattice site :math:`i_n`.
This representation provides a way to conveniently define any multi-body spin-1/2 operator, and generalises to bosons, fermions and higher spin in a natural way. 

To construct operators for different particle spieces, check out the `basis` constructor classes for the supported operator strings.

classes
-------

.. currentmodule:: quspin.operators

.. autosummary::
   :toctree: generated/

   hamiltonian
   quantum_operator
   exp_op
   quantum_LinearOperator

functions
---------

.. autosummary::
   :toctree: generated/

   commutator
   anti_commutator
   ishamiltonian
   isquantum_operator
   isexp_op
   isquantum_LinearOperator
   save_zip
   load_zip

"""

from .quantum_operator_core import *
from .quantum_LinearOperator_core import *
from .hamiltonian_core import *
from .exp_op_core import *
