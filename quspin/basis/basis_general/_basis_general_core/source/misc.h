#ifndef __MISC_H__
#define __MISC_H__

#include "numpy/ndarraytypes.h"

template<class K,class I>
K binary_search(const K N,const I A[],const I s){
	K b,bmin,bmax;
	bmin = 0;
	bmax = N-1;
	while(bmin<=bmax){
		b = (bmax+bmin)/2;
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




#endif