#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__ 


#include <cmath>
#include <algorithm>
#include <complex>
#include "complex_ops.h"

namespace math_functions
{

// abs

inline float abs(const float &value){
	return std::abs(value);
}

inline double abs(const double &value){
	return std::abs(value);
}

inline float abs(const npy_cfloat_wrapper &value){
	return std::hypot(value.real,value.imag);
}

inline double abs(const npy_cdouble_wrapper &value){
	return std::hypot(value.real,value.imag);
}

inline float compare_abs(const float R,const float B){
	return std::max(R,std::abs(B));
}

inline double compare_abs(const double R,const double B){
	return std::max(R,std::abs(B));
}

// checks of complex number 
inline float compare_abs(const float R,const npy_cfloat_wrapper &B){
	const float x = std::abs(B.real);
	const float y = std::abs(B.imag);

	if(x+y <= R){
		return R;
	}
	if(x > R || y > R){
		return std::hypot(x,y);
	}

	float RR = std::hypot(x,y);
	if(RR < R){
		return R;
	}
	else{
		return RR;
	}
}

inline double compare_abs(const double R,const npy_cdouble_wrapper &B){
	// std::hypot is expensive to calculate, check limiting cases before
	// resorting to using it by checking to see if x,y point falls inside circle 
	// of radius R that defines the current highest magnitude. 
	const double x = std::abs(B.real);
	const double y = std::abs(B.imag);
	
	if(x+y <= R){ // => (x+y)^2 = x^2+y^2+2xy <= R^2 
		return R;
	}
	if(x > R || y > R){ // => x^2+y^2 > R
		return std::hypot(x,y);
	}

	double RR = std::hypot(x,y);
	if(RR <= R){
		return R;
	}
	else{
		return RR;
	}
}




// exponentiation

inline npy_cdouble_wrapper exp(const npy_cdouble_wrapper &value){
	std::complex<double> res = std::exp(std::complex<double>(value.real,value.imag));
	return npy_cdouble_wrapper(res.real(),res.imag());
}

inline npy_cfloat_wrapper exp(const npy_cfloat_wrapper &value){
	std::complex<float> res = std::exp(std::complex<float>(value.real,value.imag));
	return npy_cfloat_wrapper(res.real(),res.imag());
}

inline double exp(const double &value){
	return std::exp(value);
}

inline float exp(const float &value){
	return std::exp(value);
}



inline void print(const npy_cdouble_wrapper value){
	printf("(%f, %f)\n",value.real,value.imag);
}

inline void print(const npy_cfloat_wrapper value){
	printf("(%f, %f)\n",value.real,value.imag);
}

inline void print(const double value){
	printf("%f\n",value);
}

inline void print(const float value){
	printf("%f\n",value);
}

}

#endif