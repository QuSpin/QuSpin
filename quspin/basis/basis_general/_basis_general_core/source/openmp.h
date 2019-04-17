#ifndef _OPENMP_H
#define _OPENMP_H

#if defined(_OPENMP)
#include <omp.h>

template<class T>
int inline atomic_add(const std::complex<double> m,std::complex<T> *M){
	T * M_v = reinterpret_cast<T*>(M);
	const T m_real = m.real();
	const T m_imag = m.imag();
	#pragma omp atomic
	M_v[0] += m_real;
	#pragma omp atomic
	M_v[1] += m_imag;
	return 0;
}

template<class T>
int inline atomic_add(const std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		const T m_real = m.real();
		#pragma omp atomic
		M[0] += m_real;
		return 0;
	}
}


#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

#endif