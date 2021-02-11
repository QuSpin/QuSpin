:orphan:

.. _example_00-label:


Basics of QuSpin `basis` objects
-----------------------------------

This tutorial shows how to define and interpret `basis` objects.  

In particular, we discuss how to define and read off physical states from the basis in the presence and absence of symmetries.  

**Notes:** 
	* we advise the users whenever possible to work with the `basis_general` objects, since they have enhanced functionality; However, occasionally it might be more convenient to work with the `basis_1d` objects where creating the basis might be a little bit faster.
	* the `general_basis` objects have a much more pronounced functionality, including some useful methods like `ent_entropy()`, `partial_trace()`, `Op_bra_ket()`, `Op_shift_sector()`, `get_amp()`, `representative()`, `normalization()`, etc., see documentation.


Script
------

:download:`download script <../../../examples/scripts/example00.py>`


.. literalinclude:: ../../../examples/scripts/example00.py
	:linenos:
	:language: python
	:lines: 1-
