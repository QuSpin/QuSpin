import numpy as np

def map_bits(s,n_map,maps):
 	ss = 0

 	for j in maps[n_map]:
 		ss ^= (s&1)<<j
 		s >>= 1;

 	return ss


def check_state_core_rec(maps,pers,s,norm=1,t=None,depth=0):
	if t is None:
		t = s

	per = pers[depth];
	
	if(depth<len(maps)-1):
		for i in range(per):
			norm = check_state_core(B,t,norm,s,nt,depth+1)
			t = map_state(t,depth,maps)


			if(t > s or np.isnan(norm)):
				return np.nan	
	
	else:
		for i in range(per):
			t = map_state(t,depth,maps)

			if(t > s):
				return np.nan


	return norm


L = 4
t = (np.arange(L)+1)%L
p = np.arange(L)[::-1]

maps = [t]
pers = [L]


for s in range(1<<L):
	n=check_state_core(maps,pers,s)
	print n,s

