###########################################################################
#                            example 18                                   #	
# This example exploits networkx for building a connectivity graph        #
# representing the hexagonal lattice geometry for the 2 species           #
# Fermy-Hubbard model. Using the same syntax one can use many other       #
# predefined geometries from networkx.                                    #
###########################################################################

# %% libraries

import numpy as np
import networkx as nx

from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

import matplotlib.pyplot as plt

# %% parameters

m, n = 2, 2                 # hexagonal lattice sizes
isPBC = False               # if True, use periodic boundary conditions
N_up, N_dn = 1, 1           # number of up/down fermions
t, U = 1.0, 2.0             # tunneling and interaction of fermions

# %% build graph

G = nx.generators.lattice.hexagonal_lattice_graph(m, n, periodic=isPBC)
N = G.number_of_nodes()

# 0-index nodes for quspin
G = nx.convert_node_labels_to_integers(G)

# visualise
pos = nx.spring_layout(G, seed=42, iterations=100)
nx.draw(G, pos=pos, with_labels=True)
plt.show()

# %% build basis & Hamiltonian

basis = spinful_fermion_basis_general(N, Nf=(N_up, N_dn))

interactions = [[U, i, i] for i in range(N)]
tunnelling = [[-t, i, j] for i in range(N) for j in G.adj[i]]

static = [["n|n", interactions], ["+-|", tunnelling], ["|+-", tunnelling]]

H = hamiltonian(static, [], basis=basis, dtype=np.float64)

# %% solve eigenproblem

E, V = H.eigsh()
print(f'First eigenvalues: {E}')
