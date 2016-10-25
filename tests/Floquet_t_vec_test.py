from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.Floquet import  Floquet_t_vec
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()


"""
This test only makes sure the floquet_t_vec class runs.
"""

Omega=uniform(3.0,10.0) # drive frequency
N_const = randint(100)

####

t = Floquet_t_vec(Omega,N_const,len_T=100)

t.vals
t.i
t.f 
t.T 
t.dt
t.len 
len(t)
t.len_T
t.N
t.strobo.vals
t.strobo.inds

####

t = Floquet_t_vec(Omega,N_const,len_T=100,N_up=randint(100))

t.vals
t.i
t.f
t.tot 
t.T 
t.dt
t.len 
len(t)
t.len_T
t.N
t.strobo.vals
t.strobo.inds

t.up.vals
t.up.i
t.up.f 
t.up.tot
t.up.len 
len(t.up)
t.up.N
t.up.strobo.vals
t.up.strobo.inds

t.const.vals
t.const.i
t.const.f 
t.const.tot
t.const.len 
len(t.const)
t.const.N
t.const.strobo.vals
t.const.strobo.inds

####

t = Floquet_t_vec(Omega,N_const,len_T=100,N_up=randint(100),N_down=randint(100))

t.vals
t.i
t.f
t.tot 
t.T 
t.dt
t.len 
len(t)
t.len_T
t.N
t.strobo.vals
t.strobo.inds

t.up.vals
t.up.i
t.up.f 
t.up.tot
t.up.len 
len(t.up)
t.up.N
t.up.strobo.vals
t.up.strobo.inds

t.const.vals
t.const.i
t.const.f
t.const.tot 
t.const.len 
len(t.const)
t.const.N
t.const.strobo.vals
t.const.strobo.inds

t.down.vals
t.down.i
t.down.f
t.down.tot 
t.down.len 
len(t.down)
t.down.N
t.down.strobo.vals
t.down.strobo.inds
