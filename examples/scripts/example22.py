from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
########################################################################
#                            example 22                                #	
#                      ...                                             #
########################################################################
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np
import matplotlib.pyplot as plt # plotting library