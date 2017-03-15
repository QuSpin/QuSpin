

cdef basis_type shift(basis_type I,int shift,int period,object[basis_type,ndim=1,mode="c"] pars):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	cdef basis_type one = 1
	cdef basis_type zero = 0
	cdef int l1,l2
	cdef basis_type Imax

	if basis_type is NP_UINT32_t:
		Imax = ((one << period) -1 if period < 32 else ~zero)
	elif basis_type is NP_UINT64_t:
		Imax = ((one << period) -1 if period < 64 else ~zero)
	else:
		Imax = (one << period) -1

	if I==0 or I==Imax:
		return I
	else:
		l1 = shift%period
		l2 = period - l1
		return ((I << l1) & Imax) | ((I & Imax) >> l2)


def py_shift(object[basis_type,ndim=1,mode="c"] x,int d,int length, object[basis_type,ndim=1,mode="c"] pars):
	cdef npy_intp i 
	cdef npy_intp Ns = x.shape[0]
	for i in range(Ns):
		x[i] = shift(x[i],d,length,pars)



cdef NP_INT8_t bit_count(basis_type I, int length):
	cdef NP_INT8_t out = 0
	cdef int i
	for i in range(length):
		out += ((I >> i) & 1) 

	return out



cdef basis_type fliplr(basis_type I, int length, object[basis_type,ndim=1,mode="c"] pars):
# this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
# (generator of) parity symmetry
	cdef basis_type out = 0
	cdef int i
	cdef basis_type j
	j = 1
	for i in range(length):
		out += ((I >> (length-1-i)) & 1 )*j
		j <<= 1
		
	return out



def py_fliplr(object[basis_type,ndim=1,mode="c"] x,int length, object[basis_type,ndim=1,mode="c"] pars):
	cdef npy_intp i 
	cdef npy_intp Ns = x.shape[0]
	for i in range(Ns):
		x[i] = fliplr(x[i],length,pars)





cdef basis_type flip_all(basis_type I, int length,object[basis_type,ndim=1,mode="c"] pars):
#	 flip all bits
	cdef basis_type one = 1
	cdef basis_type zero = 0
	cdef basis_type Imax

	if basis_type is NP_UINT32_t:
		Imax = ((one << length) -1 if length < 32 else ~zero)	
	elif basis_type is NP_UINT64_t:
		Imax = ((one << length) -1 if length < 64 else ~zero)
	else:
		Imax = (one << length) -1


	return I^(Imax & (~zero))


def py_flip_all(object[basis_type,ndim=1,mode="c"] x,int length, object[basis_type,ndim=1,mode="c"] pars):
	cdef npy_intp i 
	cdef npy_intp Ns = x.shape[0]
	for i in range(Ns):
		x[i] = flip_all(x[i],length,pars)




cdef basis_type flip_sublat_A(basis_type I, int length,object[basis_type,ndim=1,mode="c"] pars):
#	 flip all even bits: sublat A
#	 6148914691236517205 = Sum[2^i, (i, 0, 63, 2)]
#    1431655765 = Sum[2^i, (i, 0, 31, 2)]
	cdef basis_type one = 1
	cdef basis_type zero = 0
	cdef basis_type Imax,stag
	cdef int i


	if basis_type is NP_UINT32_t:
		Imax = ((one << length) -1 if length < 32 else ~zero)
		return I^(Imax&1431655765u)
	elif basis_type is NP_UINT64_t:
		Imax = ((one << length) -1 if length < 64 else ~zero)
		return I^(Imax&6148914691236517205llu)
	else:
		Imax = (one << length) - 1 
		stag = sum((one << i) for i in range(0,length,2))

		return I^(Imax&stag)


def py_flip_sublat_A(object[basis_type,ndim=1,mode="c"] x,int length, object[basis_type,ndim=1,mode="c"] pars):
	cdef npy_intp i 
	cdef npy_intp Ns = x.shape[0]
	for i in range(Ns):
		x[i] = flip_sublat_A(x[i],length,pars)




cdef basis_type flip_sublat_B(basis_type I, int length,object[basis_type,ndim=1,mode="c"] pars):
#	 flip all odd bits: sublat B
#	 12297829382473034410 = Sum[2^i, (i, 1, 63, 2)]
#    2863311530 = Sum[2^i, (i, 1, 31, 2)]
	cdef basis_type one = 1
	cdef basis_type zero = 0
	cdef basis_type Imax,stag
	cdef int i

	if basis_type is NP_UINT32_t:
		Imax = ((one << length) -1 if length < 32 else ~zero)
		return I^(Imax&2863311530u)
	elif basis_type is NP_UINT64_t:
		Imax = ((one << length) -1 if length < 64 else ~zero)
		return I^(Imax&12297829382473034410llu)
	else:
		Imax = (one << length) -1 
		stag = sum((one << i) for i in range(1,length,2))

		return I^(Imax&stag)



def py_flip_sublat_B(object[basis_type,ndim=1,mode="c"] x,int length, object[basis_type,ndim=1,mode="c"] pars):
	cdef npy_intp i 
	cdef npy_intp Ns = x.shape[0]
	for i in range(Ns):
		x[i] = flip_sublat_B(x[i],length,pars)






cdef basis_type next_state_pcon_hcp(basis_type s,object[basis_type,ndim=1,mode="c"] pars):
	if s == 0 :
		return s

	cdef basis_type t = (s | (s - 1)) + 1
	return t | ((((t & -t) / (s & -s)) >> 1) - 1)



cdef basis_type next_state_inc_1(basis_type s,object[basis_type,ndim=1,mode="c"] pars):
	return s + 1

