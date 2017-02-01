import numpy as _np

# tells whether or not the inputs into the ops needs Ns or 2*Ns elements
op_array_size={"":1,
				"M":1,
				"Z":1,
				"ZA":1,
				"ZB":1,
				"ZA & ZB":1,
				"M & Z":1,
				"M & ZA":1,
				"M & ZB":1,
				"M & ZA & ZB":1,
				"P":1,
				"M & P":1,
				"PZ":1,
				"M & PZ":1,
				"P & Z":1,
				"M & P & Z":1,
				"T":1,
				"M & T":1,
				"T & Z":1,
				"T & ZA":1,
				"T & ZB":1,
				"T & ZA & ZB":1,
				"M & T & Z":1,
				"M & T & ZA":1,
				"M & T & ZB":1,
				"M & T & ZA & ZB":1,
				"T & P":2,
				"M & T & P":2,
				"T & PZ":2,
				"M & T & PZ":2,
				"T & P & Z":2,
				"M & T & P & Z":2
				}


def kblock_Ns_estimate(Ns,L,a=1):
	Ns = int(Ns)
	L = int(L)
	a = int(a)
	return int( (1+1.0/(L//a)**2)*Ns/(L//a)+(L//a) )





