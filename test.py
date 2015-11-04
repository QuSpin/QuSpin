from ED_python import *
from numpy.linalg import eig # diagonalizes any matrix
from numpy.linalg import norm
from numpy import diag # used to make a diagonal matrix
from numpy import zeros
import time
import numpy as np

pi = np.pi


# system size
L = 4
error=1e-15

# model parameters
J0 = 1.0;
U = 0.0;


# setting up coupling lists:

#NNbonds=[[1.0,i,(i+1)%L] for i in xrange(L-1)]
#HoppAtoB=[[1.0,i,(i+1)%L] for i in xrange(0,L-1,2)]
#HoppBtoA=[[1.0,i,(i+1)%L] for i in xrange(1,L-1,2)]
#staggered_potential=[[0,0,-(-1.0)**i] for i in xrange(L)]
#linear_potential = [[0,0,i+1] for i in xrange(L)]

#print staggered_potential
#print linear_potential

list_hopping = [[-J0,i,(i+1)%L] for i in xrange(L-1)]
list_interaction = [[U,i,(i+1)%L] for i in xrange(L-1)]
list_staggered = [[0,0,-(-1.0)**i] for i in xrange(L)]

#static_Grid_0 = [['xy',list_hopping],['z',list_interaction],['h',list_staggered]]
static_Grid_0 = [['xy',list_hopping],['z',list_interaction]]

# initializing the Hamiltonians:


#H = Hamiltonian1D(static_Grid_0,[],L,Nup=L/2,zblock=1)
H = Hamiltonian1D(static_Grid_0,[],L,Nup=L/2,pblock=1)
#H = Hamiltonian1D(static_Grid_0,[],L,Nup=L/2,pblock=1,zblock=1)
#H = Hamiltonian1D(static_Grid_0,[],L,Nup=L/2)

Hp = H.return_H(0);
print real(Hp)

EV = H.DenseEV(0)
E = EV[0];
V = EV[1];

print E


