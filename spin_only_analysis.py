import numpy as np

import time
import sys
import os

####################################################################
start_time = time.time()
####################################################################

A = 1.0


#read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
my_dir = str2[n-1]

# define load data directory
load_dir = "%s/data" %(my_dir)

#change path to save directory
os.chdir(load_dir)


for L in xrange(10,16,1):

	for Omega in xrange(1,20,1):

		sectors = [(0,-1)]
		[sectors.append((kblock, 1)) for kblock in xrange( (L+2)/2)]

		# preallocate data
		Data = np.zeros((len(sectors),16), dtype=np.float64)

		for _i, blocks in enumerate(sectors):

			#kblock, pblock = blocks

			# load data
			load_params = (L,) + tuple(np.around([Omega,A],2)) + blocks
			data_name = "data_driven_chain_L=%s_Omega=%s_A=%s_kblock=%s_pblock=%s.txt" %(load_params)

			data = np.loadtxt(data_name)

			Data[_i,:] = data

		save_params = (L,) + tuple(np.around([Omega,A],2))
		save_name = "Data_driven_chain_L=%s_Omega=%s_A=%s.txt" %(save_params)

		# display full strings
		np.set_printoptions(threshold='nan')
		# save adata
		np.savetxt(save_name, Data, delimiter=" ", fmt="%s")




