"""
==================================
Basis module (:mod:`quspin.basis`)
==================================

Basis classes for quantum many-body systems.

The following table shows the available operator strings for the different bases (`sps` is the onsite Hilbert space dimension):

.. math::
   \\begin{array}{cccc}
      \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}   &   \\texttt{"x"}   &   \\texttt{"y"}  \\newline  
      \\texttt{spin_basis_*} &   \\hat{1}        &   \\hat S^+(\\hat\\sigma^+)       &   \\hat S^-(\\hat\\sigma^-)      &         -         &   \\hat S^z(\\hat\\sigma^z)       &   \\hat S^x(\\hat\\sigma^x)     &   \\hat S^y(\\hat\\sigma^y)  \\  \\newline
      \\texttt{boson_basis_*}&   \\hat{1}        &   \\hat b^\\dagger      &       \\hat b          & \\hat b^\\dagger \\hat b     &  \\hat b^\\dagger\\hat b - \\frac{\\mathrm{sps}-1}{2}       &   -       &   -  \\newline
      \\texttt{*_fermion_basis_*}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger \\hat c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}       &   \\hat c + \\hat c^\\dagger       &   -i\\left( \\hat c - \\hat c^\\dagger\\right)  \\newline
   \\end{array}

**Notes:** 

* The default operators for spin-1/2 are the Pauli matrices, NOT the spin operators. To change this, see the argument `pauli` of the `spin_basis_*` classes. 
* Higher spins can only be defined using the spin operators, and do NOT support the operator strings "x" and "y". 
* The fermion operator strings "x" and "y" are present only in the *general* basis fermion classes, and correspond to real fermions, i.e. Majorana operators (note the sign difference between "y" and the :math:`\\sigma^y` Pauli matrix, which is convention).
* The variable `sps` in the definition of the bosonic `"z"` operator stands for "states per site", i.e., the local on-site Hilbert space dimension.


.. currentmodule:: quspin.basis


one-dimensional symmetries
--------------------------

.. autosummary::
   :toctree: generated/

   spin_basis_1d
   boson_basis_1d
   spinless_fermion_basis_1d
   spinful_fermion_basis_1d



general lattice symmetries
--------------------------

.. autosummary::
   :toctree: generated/
   
   spin_basis_general
   boson_basis_general
   spinless_fermion_basis_general
   spinful_fermion_basis_general



user basis
----------

.. autosummary::
   :toctree: generated/

   user_basis



combining basis classes
-----------------------

.. autosummary::
   :toctree: generated/

   tensor_basis
   photon_basis


functions
---------

.. autosummary::
   :toctree: generated/

   coherent_state
   photon_Hspace_dim


large integer support for general basis classes
-----------------------------------------------
To construct `*_basis_general` classes with more than 64 lattice sites, QuSpin uses custom unsigned integer types for the integer 
representation of many-body states.

a) custom large integer data types supported in general basis
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   uint256
   uint1024
   uint4096
   uint16384


b) array initialization routines
++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/

   basis_ones
   basis_zeros


c) utilities to use large integers
++++++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: generated/
  
   get_basis_type
   python_int_to_basis_int
   basis_int_to_python_int
   bitwise_not
   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_leftshift
   bitwise_rightshift



"""

from .basis_1d import *
from .basis_general import *
from .base import *
from .lattice import *
from .photon import *
from .tensor import *
from quspin.basis.user import *



