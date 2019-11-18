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
	inline bool operator()(const I &a,const I &b){return a > b;}
};



template<class K,class I>
K rep_position(const npy_intp A_begin[],const npy_intp A_end[],const I A[],const npy_intp s_p,const I s){

	npy_intp begin = A_begin[s_p];
	npy_intp end = A_end[s_p];
	auto comp = compare<I>();

	if(begin<0){
		return -1;
	}
	else{
		const I * A_begin = A + begin;
		const I * A_end = A + end;
		const I * a = std::lower_bound(A_begin, A_end, s, comp);
		return ((!(a==A_end) && !comp(s,*a)) ? (K)(a-A) : -1);
	}
}

template<class K,class I>
inline K rep_position(const K N_A,const I A[],const I s){
	auto comp = compare<I>();
	const I * A_end = A+N_A;
	const I * a = std::lower_bound(A, A_end, s,comp);
	return ((!(a==A_end) && !comp(s,*a)) ? (K)(a-A) : -1);
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
inline bool equal_zero(std::complex<T> a){
	return (a.real() == 0 && a.imag() == 0);
}

template<class T>
inline bool equal_zero(T a){
	return (a == 0);
}


template<class T>
inline std::complex<double> mul(std::complex<T> a,std::complex<double> z){
	double real = a.real()*z.real() - a.imag()*z.imag();
	double imag = a.real()*z.imag() + a.imag()*z.real();

	return std::complex<double>(real,imag);
}

template<class T>
inline std::complex<double> mul(T a,std::complex<double> z){
	return std::complex<double>(a*z.real(),a*z.imag());
}




template<class T>
int inline check_imag(const std::complex<double> m){
	return 0;
}

template<>
int inline check_imag<double>(const std::complex<double> m){
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}

template<>
int inline check_imag<float>(const std::complex<double> m){
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}

template<class T>
int inline check_imag(const std::complex<double> m,std::complex<T> *M){
	M->real(m.real());
	M->imag(m.imag());
	return 0;
}

template<class T>
int inline check_imag(const std::complex<double> m,T *M){
	(*M) = m.real();
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);
}


template<class T>
int inline check_imag(const std::complex<double> m,const std::complex<T> v,std::complex<T> *M){
	M->real(m.real()*v.real()-m.imag()*v.imag());
	M->imag(m.imag()*v.real()+m.real()*v.imag());
	return 0;
}

template<class T>
int inline check_imag(std::complex<double> m,const T v,T *M){
	(*M) = v*m.real();
	return (std::abs(m.imag())>1.1e-15 ? 1 : 0);

}












}



#endif