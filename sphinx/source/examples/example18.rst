.. _example18-label:

Hexagonal Lattice: Fermi-Hubbard model [courtesy of A. Buyskikh]
----------------------------------------------------------------

This example demonstrates how to use the python package `networkx <https://networkx.github.io/documentation/stable/install.html>`_ to construct a hexagonal (honeycomb) lattice, and define the Fermi-Hubbard Hamiltonian: 


.. math::
	H = -t\sum_{\sigma=\pm}\sum_{i=0}^{N-1}\sum_{j_i=0}^{3} a^\dagger_{i\sigma} b_{j_i\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow}.

where :math:`i` runs over the lattice sites, :math:`j_i` over all nearest neighbours of :math:`i`, and :math:`\sigma` labels the fermion spin index. The creation/annihilation operators on sublattice A and B, denoted :math:`a_i, b_{j_i}`, obey fermion statistics. The tunneling matrix element is :math:`t`, and the interaction strength is denoted by :math:`U`.


Below, we first construct the hexagonal graph using `networkx <https://networkx.github.io/documentation/stable/install.html>`_, and then follow the standard QuSpin procedure to construct the Hamiltonian. The users should feel free to add the symmetries of the graph and send us an improved version of this tutorial, and we will update the script. 

This example can be generalized to other lattice geometrices supported by `networkx <https://networkx.github.io/documentation/stable/install.html>`_. To install `networkx using anaconda <https://anaconda.org/anaconda/networkx>`_, run 
:: 
	$ conda install -c anaconda networkx



Script
------

:download:`download script <../../../examples/scripts/example18.py>`

.. literalinclude:: ../../../examples/scripts/example18.py
	:linenos:
	:language: python
	:lines: 1-

