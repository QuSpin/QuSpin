#ifndef _OPENMP_H
#define _OPENMP_H

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

#include <complex>
// #include "complex_ops.h"

namespace basis_general
{

template<class T>
inline void atomic_add(const std::complex<double> m,std::complex<T> *M){
	T * M_v = reinterpret_cast<T*>(M);
	const T m_real = m.real();
	const T m_imag = m.imag();
	
	#pragma omp atomic
	M_v[0] += m_real;

	#pragma omp atomic
	M_v[1] += m_imag;
}

template<class T>
inline void atomic_add(const std::complex<double> m,T *M){
	const T m_real = m.real();
	#pragma omp atomic
	M[0] += m_real;
}







}



/*
namespace basis_general_addition
{



int inline atomic_add(const npy_cdouble_wrapper m,npy_cdouble_wrapper *M){
	double * M_v = reinterpret_cast<double*>(M);
	const double m_real = m.real;
	const double m_imag = m.imag;
	#pragma omp atomic
	M_v[0] += m_real;
	#pragma omp atomic
	M_v[1] += m_imag;
	return 0;
}

int inline atomic_add(const npy_cdouble_wrapper m,npy_cfloat_wrapper *M){
	float * M_v = reinterpret_cast<float*>(M);
	const float m_real = m.real;
	const float m_imag = m.imag;
	#pragma omp atomic
	M_v[0] += m_real;
	#pragma omp atomic
	M_v[1] += m_imag;
	return 0;
}

template<class T>
int inline atomic_add(const npy_cdouble_wrapper m,T *M){
	if(std::abs(m.imag)>1.1e-15){
		return 1;
	}
	else{
		const T m_real = (T)m.real;
		#pragma omp atomic
		M[0] += m_real;
		return 0;
	}
}

}
*/


#endif