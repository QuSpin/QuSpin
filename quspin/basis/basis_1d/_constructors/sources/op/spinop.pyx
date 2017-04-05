"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def f_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,float complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef float complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[float, ndim=1] ME = _np.ones(Ns,dtype=NP_FLOAT32)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if True:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def d_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,double complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[double, ndim=1] ME = _np.ones(Ns,dtype=NP_FLOAT64)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if True:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def F_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,float complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef float complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[float complex, ndim=1] ME = _np.ones(Ns,dtype=NP_COMPLEX64)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if False:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def D_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,double complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[double complex, ndim=1] ME = _np.ones(Ns,dtype=NP_COMPLEX128)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if False:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def g_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,long double complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef long double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[long double, ndim=1] ME = _np.ones(Ns,dtype=NP_FLOAT128)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if True:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
"""
This script constructs the non-zero matrix elements of an operator acting on a basis state. Consider, for example, the operator S^+ for a single two-level system.

-----------------------
standard QM notation:

basis = |1>,|0>
Sz = [[1,0],[0,-1]]
|down> = |0> = [0,1]  <---
|up> = |1> = [1,0]
S^+ = [[0,1],[0,0]]
-----------------------

-----------------------
QSPIN notation:

basis = |0>,|1>
Sz = [[-1,0],[0,1]]
|down> = |0> = [1,0]  <---
|up> = |1> = [0,1]
S^+ = [[0,0],[1,0]]

S_ij has i running over rows and j running over columns. Given the state |0>, the nonzero matrix element is <1|S^+|0>, which is S_10 in the qspin basis. Thus, we need to return row index to store the matrix element. 
-----------------------
"""

def G_spinop(_np.ndarray[NP_UINT32_t,ndim=1,mode='c'] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1,mode='c'] indx,long double complex J):

	cdef unsigned int i,Ns
	cdef unsigned int r,b

	cdef int j,error
	cdef char a

	cdef long double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef char *c_opstr = <char *>malloc(N_indx * sizeof(char))
	c_opstr = PyString_AsString(opstr)

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"


	cdef _np.ndarray[NP_INT32_t, ndim=1] row = _np.zeros(Ns,dtype=NP_INT32)
	cdef _np.ndarray[long double complex, ndim=1] ME = _np.ones(Ns,dtype=NP_COMPLEX256)

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1

				return row,ME,error

		if False:
			if M_E.imag != 0.0:
				error = -1
				return row,ME,error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return row,ME,error



			
