#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__ 


#include <cmath>
#include <algorithm>
#include <complex>
#include "complex_ops.h"

namespace math_functions
{

// abs

inline float abs(const float value){
	return std::abs(value);
}

inline double abs(const double value){
	return std::abs(value);
}

inline float abs(const npy_cfloat_wrapper value){
	return std::hypot(value.real,value.imag);
}

inline double abs(const npy_cdouble_wrapper value){
	return std::hypot(value.real,value.imag);
}

// exponentiation

inline npy_cdouble_wrapper exp(const npy_cdouble_wrapper value){
	std::complex<double> res = std::exp(std::complex<double>(value.real,value.imag));
	return npy_cdouble_wrapper(res.real(),res.imag());
}

inline npy_cfloat_wrapper exp(const npy_cfloat_wrapper value){
	std::complex<float> res = std::exp(std::complex<float>(value.real,value.imag));
	return npy_cfloat_wrapper(res.real(),res.imag());
}

inline double exp(const double value){
	return std::exp(value);
}

inline float exp(const float value){
	return std::exp(value);
}

// infinite norm 

template<typename I>
float inf_norm(const float * A,const I begin,const I end){
	float max = 0;
	for(I i=begin;i<end;i++){
		max = std::max(max,abs(A[i]));
	}
	return max;
}

template<typename I>
double inf_norm(const double * A,const I begin,const I end){
	double max = 0;
	for(I i=begin;i<end;i++){
		max = std::max(max,abs(A[i]));
	}
	return max;
}

template<typename I>
float inf_norm(const npy_cfloat_wrapper * A,const I begin,const I end){
	float max = 0;
	for(I i=begin;i<end;i++){
		max = std::max(max,abs(A[i]));
	}
	return max;
}

template<typename I>
double inf_norm(const npy_cdouble_wrapper * A,const I begin,const I end){
	double max = 0;
	for(I i=begin;i<end;i++){
		max = std::max(max,abs(A[i]));
	}
	return max;
}

}

#endif