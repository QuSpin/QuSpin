#ifndef __EXPM_MULTIPLY_PARALLEL_IMPL_H__
#define __EXPM_MULTIPLY_PARALLEL_IMPL_H__

#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "expm_multiply_parallel.h"

inline bool EquivTypes(PyArray_Descr * dtype1,PyArray_Descr * dtype2){
	return PyArray_EquivTypes(dtype1,dtype2);
}



#include <iostream>

int get_switch_expm_multiply(PyArray_Descr * dtype1,PyArray_Descr * dtype2,PyArray_Descr * dtype3,PyArray_Descr * dtype4){
    int I = dtype1->type_num;
    int T1 = dtype2->type_num;
    int T2 = dtype3->type_num;
    int T3 = dtype4->type_num;
    
    if(0){}
	else if(PyArray_EquivTypenums(I,NPY_INT32)){
		if(0){}
		else if(T1==NPY_COMPLEX128 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 0;}
		else if(T1==NPY_FLOAT64 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 1;}
		else if(T1==NPY_FLOAT64 && T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 2;}
		else if(T1==NPY_COMPLEX64 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 3;}
		else if(T1==NPY_COMPLEX64 && T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 4;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 5;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 6;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 7;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT32 && T3==NPY_FLOAT32){return 8;}
		else {return -1;}
	}
	else if(PyArray_EquivTypenums(I,NPY_INT64)){
		if(0){}
		else if(T1==NPY_COMPLEX128 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 9;}
		else if(T1==NPY_FLOAT64 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 10;}
		else if(T1==NPY_FLOAT64 && T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 11;}
		else if(T1==NPY_COMPLEX64 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 12;}
		else if(T1==NPY_COMPLEX64 && T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 13;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT64 && T3==NPY_COMPLEX128){return 14;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT64 && T3==NPY_FLOAT64){return 15;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT32 && T3==NPY_COMPLEX64){return 16;}
		else if(T1==NPY_FLOAT32 && T2==NPY_FLOAT32 && T3==NPY_FLOAT32){return 17;}
		else {return -1;}
	}
	else {return -1;}

    return -1;
}

void expm_multiply_impl(const int switch_num,
                        const npy_intp n,
                              void * Ap,
                              void * Aj,
                              void * Ax,
                        const int s,
                        const int m_star,
                              void * tol,
                              void * mu,
                              void * a,
                              void * F,
                              void * work)

{
    switch(switch_num){
		case 0 :
			expm_multiply<npy_int32,npy_cdouble_wrapper,double,npy_cdouble_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cdouble_wrapper*)Ax,s,m_star,*(const double*)tol,*(const npy_cdouble_wrapper*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 1 :
			expm_multiply<npy_int32,double,double,npy_cdouble_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,s,m_star,*(const double*)tol,*(const double*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 2 :
			expm_multiply<npy_int32,double,double,double>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const double*)Ax,s,m_star,*(const double*)tol,*(const double*)mu,*(const double*)a,(double*)F,(double*)work);
			break;
		case 3 :
			expm_multiply<npy_int32,npy_cfloat_wrapper,double,npy_cdouble_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,s,m_star,*(const double*)tol,*(const npy_cfloat_wrapper*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 4 :
			expm_multiply<npy_int32,npy_cfloat_wrapper,float,npy_cfloat_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const npy_cfloat_wrapper*)Ax,s,m_star,*(const float*)tol,*(const npy_cfloat_wrapper*)mu,*(const npy_cfloat_wrapper*)a,(npy_cfloat_wrapper*)F,(npy_cfloat_wrapper*)work);
			break;
		case 5 :
			expm_multiply<npy_int32,float,double,npy_cdouble_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,s,m_star,*(const double*)tol,*(const float*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 6 :
			expm_multiply<npy_int32,float,double,double>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,s,m_star,*(const double*)tol,*(const float*)mu,*(const double*)a,(double*)F,(double*)work);
			break;
		case 7 :
			expm_multiply<npy_int32,float,float,npy_cfloat_wrapper>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,s,m_star,*(const float*)tol,*(const float*)mu,*(const npy_cfloat_wrapper*)a,(npy_cfloat_wrapper*)F,(npy_cfloat_wrapper*)work);
			break;
		case 8 :
			expm_multiply<npy_int32,float,float,float>((const npy_int32)n,(const npy_int32*)Ap,(const npy_int32*)Aj,(const float*)Ax,s,m_star,*(const float*)tol,*(const float*)mu,*(const float*)a,(float*)F,(float*)work);
			break;
		case 9 :
			expm_multiply<npy_int64,npy_cdouble_wrapper,double,npy_cdouble_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cdouble_wrapper*)Ax,s,m_star,*(const double*)tol,*(const npy_cdouble_wrapper*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 10 :
			expm_multiply<npy_int64,double,double,npy_cdouble_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,s,m_star,*(const double*)tol,*(const double*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 11 :
			expm_multiply<npy_int64,double,double,double>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const double*)Ax,s,m_star,*(const double*)tol,*(const double*)mu,*(const double*)a,(double*)F,(double*)work);
			break;
		case 12 :
			expm_multiply<npy_int64,npy_cfloat_wrapper,double,npy_cdouble_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,s,m_star,*(const double*)tol,*(const npy_cfloat_wrapper*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 13 :
			expm_multiply<npy_int64,npy_cfloat_wrapper,float,npy_cfloat_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const npy_cfloat_wrapper*)Ax,s,m_star,*(const float*)tol,*(const npy_cfloat_wrapper*)mu,*(const npy_cfloat_wrapper*)a,(npy_cfloat_wrapper*)F,(npy_cfloat_wrapper*)work);
			break;
		case 14 :
			expm_multiply<npy_int64,float,double,npy_cdouble_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,s,m_star,*(const double*)tol,*(const float*)mu,*(const npy_cdouble_wrapper*)a,(npy_cdouble_wrapper*)F,(npy_cdouble_wrapper*)work);
			break;
		case 15 :
			expm_multiply<npy_int64,float,double,double>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,s,m_star,*(const double*)tol,*(const float*)mu,*(const double*)a,(double*)F,(double*)work);
			break;
		case 16 :
			expm_multiply<npy_int64,float,float,npy_cfloat_wrapper>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,s,m_star,*(const float*)tol,*(const float*)mu,*(const npy_cfloat_wrapper*)a,(npy_cfloat_wrapper*)F,(npy_cfloat_wrapper*)work);
			break;
		case 17 :
			expm_multiply<npy_int64,float,float,float>((const npy_int64)n,(const npy_int64*)Ap,(const npy_int64*)Aj,(const float*)Ax,s,m_star,*(const float*)tol,*(const float*)mu,*(const float*)a,(float*)F,(float*)work);
			break;
        default:
            throw std::runtime_error("internal error: invalid argument typenums");
    }    
}
#endif