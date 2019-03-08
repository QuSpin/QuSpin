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
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
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
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
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
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
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
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

#include "csc.h"

void csc_matvec_gil(const int switch_num,
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
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csc_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csc_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void csc_matvec_nogil(const int switch_num,
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
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csc_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csc_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void csc_matvecs_gil(const int switch_num,
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
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csc_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csc_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void csc_matvecs_nogil(const int switch_num,
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
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			csc_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			csc_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

#include "dia.h"

void dia_matvec_gil(const int switch_num,
					const bool overwrite_y,
					const npy_intp n_row,
					const npy_intp n_col,
                    const npy_intp n_diags,
                    const npy_intp L,
						  void * offsets,
						  void * diags,
						  void * a,
					const npy_intp x_stride_byte,
						  void * x,
					const npy_intp y_stride_byte,
						  void * y)
{
	switch(switch_num){
		case 0 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			dia_matvec_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			dia_matvec_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void dia_matvec_nogil(const int switch_num,
					  const bool overwrite_y,
					  const npy_intp n_row,
					  const npy_intp n_col,
                      const npy_intp n_diags,
                      const npy_intp L,
						    void * offsets,
						    void * diags,
						    void * a,
					  const npy_intp x_stride_byte,
						    void * x,
					  const npy_intp y_stride_byte,
						    void * y)
{
	switch(switch_num){
		case 0 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 3 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 7 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			dia_matvec_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
		case 9 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 12 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cdouble_wrapper*)x,y_stride_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const double*)x,y_stride_byte,(double*)y);
			break;
		case 16 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const npy_cfloat_wrapper*)x,y_stride_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			dia_matvec_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_byte,(const float*)x,y_stride_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void dia_matvecs_gil(const int switch_num,
					 const bool overwrite_y,
					 const npy_intp n_row,
					 const npy_intp n_col,
					 const npy_intp n_vecs,
                     const npy_intp n_diags,
                     const npy_intp L,
						   void * offsets,
						   void * diags,
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
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			dia_matvecs_noomp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			dia_matvecs_noomp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}

void dia_matvecs_nogil(const int switch_num,
					   const bool overwrite_y,
					   const npy_intp n_row,
					   const npy_intp n_col,
					   const npy_intp n_vecs,
                       const npy_intp n_diags,
                       const npy_intp L,
						     void * offsets,
						     void * diags,
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
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 1 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 2 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 3 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 4 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 5 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 6 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 7 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 8 :
			dia_matvecs_omp(overwrite_y,(const npy_int32)n_row,(const npy_int32)n_col,n_vecs,(const npy_int32)n_diags,(const npy_int32)L,(const npy_int32*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
		case 9 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cdouble_wrapper*)diags,*(const npy_cdouble_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 10 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 11 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const double*)diags,*(const double*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 12 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 13 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const npy_cfloat_wrapper*)diags,*(const npy_cfloat_wrapper*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 14 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cdouble_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cdouble_wrapper*)y);
			break;
		case 15 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const double*)x,y_stride_row_byte,y_stride_col_byte,(double*)y);
			break;
		case 16 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const npy_cfloat_wrapper*)x,y_stride_row_byte,y_stride_col_byte,(npy_cfloat_wrapper*)y);
			break;
		case 17 :
			dia_matvecs_omp(overwrite_y,(const npy_int64)n_row,(const npy_int64)n_col,n_vecs,(const npy_int64)n_diags,(const npy_int64)L,(const npy_int64*)offsets,(const float*)diags,*(const float*)a,x_stride_row_byte,x_stride_col_byte,(const float*)x,y_stride_row_byte,y_stride_col_byte,(float*)y);
			break;
	    default:
	        throw std::runtime_error("internal error: invalid argument typenums");
	}
}
#endif