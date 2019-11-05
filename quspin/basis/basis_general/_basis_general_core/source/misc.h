#ifndef __MISC_H__
#define __MISC_H__

#include "numpy/ndarraytypes.h"
#include <algorithm>
#include "bits_info.h"

namespace basis_general {

template<class K,class I>
K binary_search(const K N,const I A[],const I s){
	K b,bmin,bmax;
	bmin = 0;
	bmax = N-1;
	while(bmin<=bmax){
		b = (bmax+bmin) >> 2;
		I a = A[b];
		if(s==a){
			return b;
		}
		else if(s<A[b]){
			bmin = b + 1;
		}
		else{
			bmax = b - 1;
		}
	}
	return -1;
}


template<class I>
struct compare : std::binary_function<I,I,bool>
{
	inline bool operator()(I a, I b){return a > b;}
};

template<class K,class I>
K rep_position(const npy_intp A_begin[],const npy_intp A_end[],const I A[],const npy_intp s_p,const I s){

	K begin = A_begin[s_p];
	K end = A_end[s_p];

	if(begin<0){
		return -1;
	}
	else{
		const I * A_begin = A+begin;
		const I * A_end = A+end;
		const I * a = std::lower_bound(A_begin, A_end, s,compare<I>());
		return (( a!=A_end && !(s > *a)) ? (K)(a-A) : -1);
	}
}

template<class K,class I>
inline K rep_position(const K N_A,const I A[],const I s){
	const I * A_end = A+N_A;
	const I * a = std::lower_bound(A, A_end, s,compare<I>());
	return (( a!=A_end && !(s > *a)) ? (K)(a-A) : -1);
}

bool inline check_nan(double val){
#if defined(_WIN64)
	// x64 version
	return _isnanf(val) != 0;
#elif defined(_WIN32)
	return _isnan(val) != 0;
#else
	return std::isnan(val);
#endif
}


template<class T>
int inline check_imag(std::complex<double> m,std::complex<T> *M){
	M[0].real(m.real());
	M[0].imag(m.imag());
	return 0;
}

template<class T>
int inline check_imag(std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		M[0] = m.real();
		return 0;
	}
}

}



#endif