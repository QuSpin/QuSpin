import numpy as np


def coherent_state(a,n,dtype=np.float64):
	s1 = np.full((n,),-np.abs(a)**2/2.0,dtype=dtype)
	s2 = np.arange(n,dtype=np.float64)
	s3 = np.array(s2)
	s3[0] = 1
	np.log(s3,out=s3)
	s3[1:] = 0.5*np.cumsum(s3[1:])
	state = s1+np.log(a)*s2-s3
	for s in state:
		print s
	return np.exp(state)



Nph = 6 #100

L = 201 #2*Nph**2
print L
v = coherent_state(Nph,L)
	
print np.linalg.norm(v)
