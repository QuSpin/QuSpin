import numpy as np 
import basis_ops # can call all fns of basis_ops
from scipy.misc import comb

# spin.py shows how examples are used

"""
# to-do list
1) start from spin_basis and re-write functions for hb (should be the same)
2) create the hcb_ops.pyx
3) in basis_ops.pyx add "include ..." in the end
"""

## w/o symmetries: use full basis
# set up variables
L=10
Ns=np.uint32(2**L)

row=np.zeros((Ns,),dtype=np.int32) # basis type same as l31 of basis_ops.pyx
col=np.zeros_like(row)
ME=np.zeros_like(row,dtype=np.float32) # need to change to np.complex64 for cpx symmetries

opstr='+zn-' # a single string
indx=np.array([1,2,3,4],dtype=np.int32) # opstr acts at
J=1.0 # scalar arguments are taken care of automatically by cython

basis=np.arange(Ns,dtype=np.uint32) # unsigned integers 32
blocks={}

# call functions
error=basis_ops.hcb_op(row,col,ME,opstr,indx,J,basis,**blocks)
print(error)

"""
### m-symmetry
Nup=L/2
basis_ops.hcb_n_basis(L, Nup, Ns, basis)
Ns_m=comb(L,Nup,exact=True)
basis=basis[:Ns_m]


# call functions
error=basis_ops.hcb_n_op(row,col,ME,opstr,indx,J,basis,**blocks)
print(error)
"""

### n-symmetry
Nup=L/2
pblock=1
Ns_n_p=basis_ops.hcb_n_p_basis(L,Nup,pblock,np.empty(Ns,dtype=np.int8),basis)
print(Ns_n_p)
basis=basis[:Ns_n_p]

### n-symmetry
Nup=L/2
pblock=1
Ns_n_p=basis_ops.fermion_n_p_basis(L,Nup,pblock,np.empty(Ns,dtype=np.int8),basis)
print(Ns_n_p)
basis=basis[:Ns_n_p]


# call functions
error=basis_ops.hcb_n_op(row,col,ME,opstr,indx,J,basis,**blocks)
print(error)










