from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
###########################################################################
#                            example 18                                   #	
# This example exploits the python package 'networkx',                    #
# https://networkx.github.io/documentation/stable/ , for building a       #
# connectivity graph representing the hexagonal lattice geometry, using   #
# the spinful Fermy-Hubbard model on a honeycomb lattice. Using the same  #
# syntax one can define many other geometries predefined in networkx.     #
###########################################################################
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian
import numpy as np
import networkx as nx # networkx package, see https://networkx.github.io/documentation/stable/
import matplotlib.pyplot as plt # plotting library
#
###### create honeycomb lattice
# lattice graph parameters
m = 2  # number of rows of hexagons in the lattice
n = 2  # number of columns of hexagons in the lattice
isPBC = False # if True, use periodic boundary conditions
#
### build graph using networkx
hex_graph = nx.generators.lattice.hexagonal_lattice_graph(m, n, periodic=isPBC)
# label graph nodes by consecutive integers
hex_graph = nx.convert_node_labels_to_integers(hex_graph)
# set number of lattice sites
N = hex_graph.number_of_nodes()
print('constructed hexagonal lattice with {0:d} sites.\n'.format(N))
# visualise graph
pos = nx.spring_layout(hex_graph, seed=42, iterations=100)
nx.draw(hex_graph, pos=pos, with_labels=True)
plt.show()
#
###### model parameters
#
N_up = 2 # number of spin-up fermions
N_down = 2 # number of spin-down fermions
t = 1.0 # tunnelling matrix element
U = 2.0 # on-site fermion interaction strength
#
##### set up Fermi-Hubbard Hubbard Hamiltonian with quspin #####
#
### compute basis
basis = spinful_fermion_basis_general(N, Nf=(N_up, N_down))
print('Hilbert space size: {0:d}.\n'.format(basis.Ns))
#
# define site-coupling lists
tunnelling   = [[-t, i, j] for i in range(N) for j in hex_graph.adj[i]]
interactions = [[ U, i, i] for i in range(N)]
#
# define site-coupling lists [hermitian conjugates "-+|" and "|-+" contained in tunnelling list]
static = [["n|n", interactions], ["+-|", tunnelling], ["|+-", tunnelling]]
dynamic=[]
#
### construct Hamiltonian
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
#
# compute eigensystem
E, V = H.eigsh(k=4,which='SA',maxiter=1E4)
print(f'\nlowest energies: {E}')
