import numpy as _np

# tells whether or not the inputs into the ops needs Ns or 2*Ns elements
op_array_size={"":1,
				"N":1,
				"Z":1,
				"ZA":1,
				"ZB":1,
				"ZA & ZB":1,
				"N & Z":1,
				"N & ZA":1,
				"N & ZB":1,
				"N & ZA & ZB":1,
				"P":1,
				"N & P":1,
				"PZ":1,
				"N & PZ":1,
				"P & Z":1,
				"N & P & Z":1,
				"T":1,
				"N & T":1,
				"T & Z":1,
				"T & ZA":1,
				"T & ZB":1,
				"T & ZA & ZB":1,
				"N & T & Z":1,
				"N & T & ZA":1,
				"N & T & ZB":1,
				"N & T & ZA & ZB":1,
				"T & P":2,
				"N & T & P":2,
				"T & PZ":2,
				"N & T & PZ":2,
				"T & P & Z":2,
				"N & T & P & Z":2
				}


def kblock_Ns_estimate(Ns,L,a=1):
	Ns = int(Ns)
	L = int(L)
	a = int(a)
	return int( (1+1.0/(L//a)**2)*Ns/(L//a)+(L//a) )





