from numpy import asarray, asarray_chkfinite, array, int32, int64, float32, float64, complex64, complex128


try:
	import _py_lapack_wrap 

	class _LapackError(Exception):
		def __init__(self,message):
			self.message=message
		def __str__(self):
			return self.message

	def eig(a,copy_a=False,check_finite=True,JOBL='N',JOBR='V'):
		if check_finite:
			if copy_a:
				a1=array(a,order='F')
				a1=asarray_chkfinite(a1)
			else:
				a1=asarray_chkfinite(a,order='F')
		else:
			if copy_a:
				a1=array(a,order='F')
			else:
				a1=asarray(a,order='F')

		if a1.dtype == complex64:
			prefix="c"
			v, info = _py_lapack_wrap.py_cgeev(JOBL,JOBR,a1)
		elif a1.dtype == complex128:
			prefix="z"
			v, info = _py_lapack_wrap.py_zgeev(JOBL,JOBR,a1)
		else:
			raise TypeError("eig doesn't support:"+str(a1.dtype))
	
		if info == 2790:
			raise _LapackError("cannot solve for Left and Right eigen vectors simultainously.")
		elif info != 0:
			raise _LapackError(prefix+"geev failed to diagonalize matrix: INFO=%d" % (info,))

		if JOBL=='V' or JOBR=='V':
			return v, a1
		else:
			return v

	def eigvals(a,copy_a=False,check_finite=True,JOBL='N',JOBR='N'):
		return eig(a,copy_a=copy_a,check_finite=check_finite,JOBL=JOBL,JOBR=JOBR)




	def eigh(a,copy_a=False,check_finite=True,JOBZ='V',UPLO='L'):
		if check_finite:
			if copy_a:
				a1=array(a,order='F')
				a1=asarray_chkfinite(a1)
			else:
				a1=asarray_chkfinite(a,order='F')
		else:
			if copy_a:
				a1=array(a,order='F')
			else:
				a1=asarray(a,order='F')

		if a1.dtype == float32:
			prefix='ssy'
			v, info = _py_lapack_wrap.py_ssyev(JOBZ,UPLO,a1)
		elif a1.dtype == float64:
			prefix='dsy'
			v, info = _py_lapack_wrap.py_dsyev(JOBZ,UPLO,a1)
		elif a1.dtype == complex64:
			prefix='che'
			v, info = _py_lapack_wrap.py_cheev(JOBZ,UPLO,a1)
		elif a1.dtype == complex128:
			prefix='zhe'
			v, info = _py_lapack_wrap.py_zheev(JOBZ,UPLO,a1)
		else:
			raise TypeError("eig doesn't support:"+str(a1.dtype))
	
		if info != 0:
			raise _LapackError(prefix+"ev failed to diagonalize matrix: INFO=%d" % (info,))

		if JOBZ=='V':
			return v, a1
		elif JOBZ=='N':
			return v

		def eigvalsh(a,copy_a=False,check_finite=True,JOBZ='N',UPLO='L'):
			return eigh(a,copy_a=copy_a,check_finite=check_finite,JOBZ=JOBZ,UPLO=UPLO)


except ImportError:
	from scipy.linalg import eig,eigh,eigvals,eigvalsh





