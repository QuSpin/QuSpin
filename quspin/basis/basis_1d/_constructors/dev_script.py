import numpy as np 
import basis_ops # can call all fns of basis_ops

# spin.py shows how examples are used

"""
# to-do list
1) start from spin_basis and re-write functions for hb (should be the same)
2) create the hcb_ops.pyx
3) in basis_ops.pyx add "include ..." in the end
"""

## w/o symmetries: use full basis
# set up variables
L=4
Ns=np.uint32(2**L)

row=np.zeros((Ns,),dtype=np.int32) # basis type same as l31 of basis_ops.pyx
col=np.zeros_like(row)
ME=np.zeros_like(row,dtype=np.float32) # need to change to np.complex64 for cpx symmetries

opstr='xz' # a single string
indx=np.array([1,2],dtype=np.int32) # opstr acts at
J=1.0 # scalar arguments are taken care of automatically by cython

basis=np.arange(Ns,dtype=np.uint32) # unsigned integers 32
blocks={}

### m-symmetry
Nup=2
print(basis)
Ns_m=basis_ops.hcb_m_basis(L, Nup, Ns, basis)
print(Ns_m,Ns)
print(basis)



# call functions
error=basis_ops.hcb_op(row,col,ME,opstr,indx,J,basis,**blocks)

print(error)

