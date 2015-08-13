py_lapack_wrap: py_lapack_wrap.f90
	f2py py_lapack_wrap.f90 -m py_lapack_wrap -h py_lapack_wrap.pyf --overwrite-signature --quiet
	f2py -c py_lapack_wrap.pyf py_lapack_wrap.f90 -llapack --opt=-o3 --quiet
