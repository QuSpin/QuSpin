###########################################################################
#                            example 13                                   #	
# This example exploits networkx for building a connectivity graph        #
# representing the hexagonal lattice geometry for the 2 species           #
# Fermy-Hubbard model.                                                    #
###########################################################################

# %% libraries

import numpy as np
import networkx as nx

from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

# %% parameters

L_x, L_y = 2, 2             # lattice sizes
N_up, N_dn = 2, 2           # number of up/down spin particles
t, U = 1.0, 2.0             # tunneling and interaction

# %% build graph

G = nx.generators.lattice.hexagonal_lattice_graph(L_x, L_y)
L = G.number_of_nodes()

# label nodes from 0 to L-1 for convenience
G = nx.convert_node_labels_to_integers(G)

# show the graph
nx.draw(G, with_labels=True)

# %% build basis & Hamiltonian

basis = spinful_fermion_basis_general(L, Nf=(N_up, N_dn))

interactions = [[U, i, i] for i in range(L)]
tunnelling = []
for i in range(L):
    tunnelling += [[-t, i, j] for j in G.adj[i]]

static = [
    ["n|n", interactions],
    ["+-|", tunnelling],
    ["|+-", tunnelling]]

checks = dict(check_pcon=True, check_symm=True, check_herm=True)

H = hamiltonian(static, [], basis=basis, dtype=np.float64, **checks)

# %% do some interesting stuff...

E = H.eigvalsh()
