

cdef state_type shift(state_type I,int shift,int period):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	cdef state_type Imax= (1ull << period) -1
	cdef int l1,l2
	if I==0 or I==Imax:
		return I
	else:
		l1 = shift%period
		l2 = period - l1
		return ((I << l1) & Imax) | ((I & Imax) >> l2)



cdef NP_INT8_t bit_count(state_type I, int length):
	cdef NP_INT8_t out = 0
	cdef int i
	for i in range(length):
		out += ((I >> i) & 1) 

	return out




cdef state_type fliplr(state_type I, int length):
# this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
# (generator of) parity symmetry
	cdef state_type out = 0
	cdef int i
	cdef state_type j
	j = 1
	for i in range(length):
		out += ((I >> (length-1-i)) & 1 )*j
		j <<= 1
		
	return out




cdef inline state_type flip_all(state_type I, int length):
	# flip all bits
	return I^( (1llu << length)-1 ) & ~0llu


cdef inline state_type flip_sublat_A(state_type I, int length):
	# flip all even bits: sublat A
	# 6148914691236517205 = Sum[2^i, (i, 0, 63, 2)]
	return I^( (1llu << length)-1) & 6148914691236517205llu
	

cdef inline state_type flip_sublat_B(state_type I, int length):
	# flip all odd bits: sublat B
	# 12297829382473034410 = Sum[2^i, (i, 1, 63, 2)]
	return I^( (1llu << length)-1) & 12297829382473034410llu



cdef state_type next_state_pcon(state_type v):
	if v == 0:
		return v

	cdef state_type t = (v | (v - 1)) + 1
	return t | ((((t & -t) / (v & -v)) >> 1) - 1)

cdef state_type next_state_no_pcon(state_type v):
	return v + 1









