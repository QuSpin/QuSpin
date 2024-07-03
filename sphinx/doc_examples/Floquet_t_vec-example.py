#
import sys, os


#
from quspin.tools.Floquet import Floquet_t_vec  # Floquet time vector
import numpy as np  # generic math functions

#
Omega = 4.5  # drive frequency
# define time vector with three stages of evolution, labelled by "up", "const" and "down"
t = Floquet_t_vec(Omega, 10, len_T=10, N_up=3, N_down=2)
print(t)
##### attibutes referring to total time vector
# time points values
print(t.vals)
# total number of periods
print(t.N)
# driving period
print(t.T)
# step size
print(t.dt)
# initial, final time and total time duration
print(t.i, t.f, t.tot)
# length of time vector and length within a period
print(t.len, t.len_T)
##### attributes referring to stroboscopic times only
# indices of stroboscopic times
print(t.strobo.inds)
# values of stroboscopic times
print(t.strobo.vals)
##### attributes relating to the "up" stage of the evolution
# time values for the "up"-stage
print(t.up.vals)
# initial, final time and total time of "up" duration (similar for "const" and "down")
print(t.up.i, t.up.f, t.up.tot)
