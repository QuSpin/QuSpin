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

def f_spinop(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,float complex J):

	cdef state_type i,Ns
	cdef state_type r,b

	cdef int j,error
	cdef char a

	cdef float complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

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
				M_E *= (1.0 if a else -1.0)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= (1.0j if a else -1.0j)
			elif c_opstr[j] == p:
				r = r ^ b
				M_E *= (0.0 if a else 2.0)
			elif c_opstr[j] == m:
				r = r ^ b
				M_E *= (2.0 if a else 0.0)
			else:
				error = 1
				return row,ME,error

			if M_E == 0.0:
				r = basis[i]
				break

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

def d_spinop(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,double complex J):

	cdef state_type i,Ns
	cdef state_type r,b

	cdef int j,error
	cdef char a

	cdef double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

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
				M_E *= (1.0 if a else -1.0)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= (1.0j if a else -1.0j)
			elif c_opstr[j] == p:
				r = r ^ b
				M_E *= (0.0 if a else 2.0)
			elif c_opstr[j] == m:
				r = r ^ b
				M_E *= (2.0 if a else 0.0)
			else:
				error = 1
				return row,ME,error

			if M_E == 0.0:
				r = basis[i]
				break

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

def F_spinop(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,float complex J):

	cdef state_type i,Ns
	cdef state_type r,b

	cdef int j,error
	cdef char a

	cdef float complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

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
				M_E *= (1.0 if a else -1.0)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= (1.0j if a else -1.0j)
			elif c_opstr[j] == p:
				r = r ^ b
				M_E *= (0.0 if a else 2.0)
			elif c_opstr[j] == m:
				r = r ^ b
				M_E *= (2.0 if a else 0.0)
			else:
				error = 1
				return row,ME,error

			if M_E == 0.0:
				r = basis[i]
				break

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

def D_spinop(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,double complex J):

	cdef state_type i,Ns
	cdef state_type r,b

	cdef int j,error
	cdef char a

	cdef double complex M_E

	cdef int N_indx = indx.shape[0]
	Ns = basis.shape[0]

	cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

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
				M_E *= (1.0 if a else -1.0)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= (1.0j if a else -1.0j)
			elif c_opstr[j] == p:
				r = r ^ b
				M_E *= (0.0 if a else 2.0)
			elif c_opstr[j] == m:
				r = r ^ b
				M_E *= (2.0 if a else 0.0)
			else:
				error = 1
				return row,ME,error

			if M_E == 0.0:
				r = basis[i]
				break

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






