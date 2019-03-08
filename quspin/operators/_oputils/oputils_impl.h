#ifndef __OPUTILS_IMPL_H__
#define __OPUTILS_IMPL_H__

#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"




int get_switch_num(PyArray_Descr * dtype1,PyArray_Descr * dtype2,PyArray_Descr * dtype3){
	const int T1 = dtype1->type_num;
	const int T2 = dtype2->type_num;
	const int T3 = dtype3->type_num;
	if(0){}
	else if(T1==NPY_INT32){
		 if(0){}
		else if(T2==NPY_COMPLEX128 && T3==NPY_COMPLEX128){return 0;}
		else if(T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 1;}
		else if(T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 2;}
		else if(T2==NPY_COMPLEX64 && T3==NPY_COMPLEX128){return 3;}
		else if(T2==NPY_COMPLEX64 && T3==NPY_COMPLEX64){return 4;}
		else if(T2==NPY_FLOAT32 && T3==NPY_COMPLEX128){return 5;}
		else if(T2==NPY_FLOAT32 && T3==NPY_FLOAT64){return 6;}
		else if(T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 7;}
		else if(T2==NPY_FLOAT32 && T3==NPY_FLOAT32){return 8;}
		else {return -1;}
	}
	else if(T1==NPY_INT64){
		 if(0){}
		else if(T2==NPY_COMPLEX128 && T3==NPY_COMPLEX128){return 9;}
		else if(T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 10;}
		else if(T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 11;}
		else if(T2==NPY_COMPLEX64 && T3==NPY_COMPLEX128){return 12;}
		else if(T2==NPY_COMPLEX64 && T3==NPY_COMPLEX64){return 13;}
		else if(T2==NPY_FLOAT32 && T3==NPY_COMPLEX128){return 14;}
		else if(T2==NPY_FLOAT32 && T3==NPY_FLOAT64){return 15;}
		else if(T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 16;}
		else if(T2==NPY_FLOAT32 && T3==NPY_FLOAT32){return 17;}
		else {return -1;}
	}
	else {return -1;}

	return -1;
}

#include "csr.h"

void csr_matvec_gil(const int switch_num,
					const bool overwrite_y,
					const npy_intp n_row,
					const npy_intp n_col,
						  void * Ap,
						  void * Aj,
						  void * Ax,
						  void * a,
					const npy_intp x_stride_byte,
						  void * x,
					const npy_intp y_stride_byte,
						  void * y)
{
	switch(switch_num){
		case 0 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csr_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csr_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	}
}

void csr_matvec_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_byte,
						    void * x,
					  const npy_intp y_stride_byte,
						    void * y)
{
	switch(switch_num){
		case 0 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csr_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csr_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	}
}

void csr_matvecs_gil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
					  const npy_intp n_vecs,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_row_byte,
					  const npy_intp x_stride_col_byte,
						    void * x,
					  const npy_intp y_stride_row_byte,
					  const npy_intp y_stride_col_byte,
						    void * y)
{
	switch(switch_num){
		case 0 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csr_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csr_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	}
}

void csr_matvecs_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
					  const npy_intp n_vecs,
						    void * Ap,
						    void * Aj,
						    void * Ax,
						    void * a,
					  const npy_intp x_stride_row_byte,
					  const npy_intp x_stride_col_byte,
						    void * x,
					  const npy_intp y_stride_row_byte,
					  const npy_intp y_stride_col_byte,
						    void * y)
{
	switch(switch_num){
		case 0 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csr_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csr_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	}
}


#endif