



cdef npy_intp findzstate(object[basis_type,ndim=1,mode="c"] A,npy_intp N, basis_type s,bool * found):
	cdef basis_type  A_1
	cdef npy_intp bmin, bmax, b

	bmin = 0
	bmax = N-1
	found[0] = False
	while (bmin <= bmax):
		b = (bmin + bmax)/2
		A_1 = A[b]
		if ( A_1 == s ):
			found[0] = True
			return b
		elif ( s < A_1 ):
			bmin = b + 1
		else:
			bmax = b - 1
	return 0


cdef basis_type RefState_P_template(bitop fliplr,basis_type s,int L, int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t

	R[0] = 0


	t = fliplr(s,L,bitop_pars)
	if t < s:
		R[0] = 1
		return t

	return s





cdef basis_type RefState_PZ_template(bitop fliplr,bitop flip_all,basis_type s,int L, int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t

	R[0] = 0

	t = fliplr(s,L,bitop_pars)
	t = flip_all(t,L,bitop_pars)
	if t < s:
		R[0] = 1
		return t

	return s







cdef basis_type RefState_P_Z_template(bitop fliplr,bitop flip_all,basis_type s,int L, int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t,r

	R[0] = 0
	R[1] = 0

	r = s
	t = fliplr(s,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 1
		R[1] = 0


	t = flip_all(s,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 0
		R[1] = 1



	t = fliplr(s,L,bitop_pars)
	t = flip_all(t,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 1
		R[1] = 1

	return r








cdef basis_type RefState_Z_template(bitop fliplr,basis_type s,int L,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t
	R[0] = 0

	t = flip_all(s,L,bitop_pars)
	if t < s:
		R[0] = 1
		return t

	return s




cdef basis_type RefState_ZA_template(bitop flip_sublat_A,basis_type s,int L,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t
	R[0] = 0


	t = flip_sublat_A(s,L,bitop_pars)
	if t < s:
		R[0] = 1
		return t

	return s


cdef basis_type RefState_ZB_template(bitop flip_sublat_B,basis_type s,int L,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t
	R[0] = 0

	t = flip_sublat_B(s,L,bitop_pars)
	if t < s:
		R[0] = 1
		return t

	return s


cdef basis_type RefState_ZA_ZB_template(bitop flip_sublat_A,bitop flip_sublat_B,bitop flip_all,basis_type s,int L,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef basis_type t, r

	R[0] = 0
	R[1] = 0

	r = s

	t = flip_sublat_A(s,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 1
		R[1] = 0

	t = flip_sublat_B(s,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 0
		R[1] = 1

	t = flip_all(s,L,bitop_pars)
	if t < r:
		r = t
		R[0] = 1
		R[1] = 1

	return r



cdef basis_type RefState_T_template(shifter shift,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l
	cdef basis_type r,t

	r = s
	t = s
	l = 0

	for i in range(1,L/T+1):
		t=shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i;

	R[0] = l

	return r





cdef basis_type RefState_T_Z_template(shifter shift,bitop flip_all,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,g
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	g = 0

	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; g=0;

	t = flip_all(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; g=1;


	R[0] = l
	R[1] = g

	return r



cdef basis_type RefState_T_ZA_template(shifter shift,bitop flip_sublat_A,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,gA
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	gA = 0

	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=0;

	t = flip_sublat_A(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=1;


	R[0] = l
	R[1] = gA

	return r


cdef basis_type RefState_T_ZB_template(shifter shift,bitop flip_sublat_B,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,gB
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	gB = 0

	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gB=0;

	t = flip_sublat_B(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gB=1;


	R[0] = l
	R[1] = gB

	return r


cdef basis_type RefState_T_ZA_ZB_template(shifter shift,bitop flip_sublat_A,bitop flip_sublat_B,bitop flip_all,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,gA,gB
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	gA = 0
	gB = 0
	
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=0;gB=0;

	t = flip_sublat_A(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=1;gB=0;

	t = flip_sublat_B(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=0;gB=1;

	t = flip_all(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; gA=1;gB=1;


	R[0] = l
	R[1] = gA
	R[2] = gB

	return r



cdef basis_type RefState_T_P_template(shifter shift,bitop fliplr,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,q
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	q = 0

	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=0;

	t = fliplr(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=1;


	R[0] = l
	R[1] = q

	return r












cdef basis_type RefState_T_PZ_template(shifter shift,bitop fliplr,bitop flip_all,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i
	cdef basis_type r,t,l,qg

	r = s
	t = s

	l = 0
	qg = 0

	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; qg=0;

	t = fliplr(s,L,bitop_pars)
	t = flip_all(t,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; qg=1;


	R[0] = l
	R[1] = qg

	return r











cdef basis_type RefState_T_P_Z_template(shifter shift,bitop fliplr,bitop flip_all,basis_type s,int L,int T,int * R, object[basis_type,ndim=1,mode="c"] bitop_pars):
	cdef int i,l,q,g
	cdef basis_type r,t

	r = s
	t = s

	l = 0
	q = 0
	g = 0
	
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=0;g=0;

	t = fliplr(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=1;g=0;

	t = flip_all(s,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=0;g=1;

	t = fliplr(s,L,bitop_pars)
	t = flip_all(t,L,bitop_pars)
	for i in range(1,L/T+1):
		t = shift(t,-T,L,bitop_pars)
		if t < r:
			r=t; l=i; q=1;g=1;


	R[0] = l
	R[1] = q
	R[2] = g

	return r

