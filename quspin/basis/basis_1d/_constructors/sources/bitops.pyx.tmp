"""
cdef unsigned long long shift(unsigned long long I,int shift,int period):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	cdef unsigned long long Imax= (1ull << period) -1
	if I==0 or I==Imax:
		return I
	else:
		if shift < 0:
			shift=-shift
			return ((I & Imax) >> shift%period) | (I << (period-(shift%period)) & Imax)
		else:
			return (I << shift%period) & Imax | ((I & Imax) >> (period-(shift%period)))
"""

cdef unsigned long long shift(unsigned long long I,int shift,int period):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	cdef unsigned long long Imax= (1ull << period) -1
	cdef int l1,l2
	if I==0 or I==Imax:
		return I
	else:
		l1 = shift%period
		l2 = period - l1
		return ((I << l1) & Imax) | ((I & Imax) >> l2)



cdef NP_INT8_t bit_count(unsigned long long I, int length):
	cdef NP_INT8_t out = 0
	cdef int i
	for i in range(length):
		out += ((I >> i) & 1) 

	return out




cdef unsigned long long fliplr(unsigned long long I, int length):
# this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
# (generator of) parity symmetry
	cdef unsigned long long out = 0
	cdef int i
	cdef unsigned long long j
	j = 1
	for i in range(length):
		out += ((I >> (length-1-i)) & 1 )*j
		j <<= 1
		
	return out




cdef inline unsigned long long flip_all(unsigned long long I, int length):
	# flip all bits
	return I^( (1llu << length)-1 ) & ~0llu


cdef inline unsigned long long flip_sublat_A(unsigned long long I, int length):
	# flip all even bits: sublat A
	# 6148914691236517205 = Sum[2^i, (i, 0, 63, 2)]
	return I^( (1llu << length)-1) & 6148914691236517205llu
	

cdef inline unsigned long long flip_sublat_B(unsigned long long I, int length):
	# flip all odd bits: sublat B
	# 12297829382473034410 = Sum[2^i, (i, 1, 63, 2)]
	return I^( (1llu << length)-1) & 12297829382473034410llu



cdef unsigned long long next_state_pcon(unsigned long long v):
	if v == 0:
		return v

	cdef unsigned long long t = (v | (v - 1)) + 1
	return t | ((((t & -t) / (v & -v)) >> 1) - 1)

cdef unsigned long long next_state_no_pcon(unsigned long long v):
	return v + 1

cdef unsigned long long next_state(unsigned long long v):
	cdef unsigned long long t = (v | (v - 1)) + 1
	return t | ((((t & -t) / (v & -v)) >> 1) - 1)



ctypedef unsigned long long (*bitop)(unsigned long long,int)
ctypedef unsigned long long (*shifter)(unsigned long long,int,int)
ctypedef unsigned long long (*ns_type)(unsigned long long)