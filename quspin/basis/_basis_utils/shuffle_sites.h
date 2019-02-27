#ifndef __SHUFFLE_SITES_
#define __SHUFFLE_SITES_


#include "numpy/ndarraytypes.h"

template<class T>
void shuffle_sites_strid(const npy_int32 nd,
						const npy_intp * A_shape,
						const npy_int32 * T_tup,
						const npy_intp n_row,
						const npy_intp n_col,
						const T * A,
							  T * AT)
{
	npy_intp A_strides[64];
	npy_intp AT_strides[64];
	npy_intp AT_shape[64];

	// calculate transposed shape
	for(npy_int32 i=0;i<nd;i++){
		AT_shape[i] = A_shape[T_tup[i]];
	}	
	// calculate contiguous strides for A
	AT_strides[nd-1] = 1;
	for(npy_int32 i=nd-2;i>=0;i--){
		AT_strides[i] = AT_strides[i+1] * A_shape[i+1];
	}
	// calculate non-contiguous strides for A by transposing contiguous strides
	for(npy_int32 i=0;i<nd;i++){
		A_strides[i] = AT_strides[T_tup[i]];
	}
	// calculate contiguous strides for AT
	AT_strides[nd-1] = 1;
	for(npy_int32 i=nd-2;i>=0;i--){
		AT_strides[i] = AT_strides[i+1] * AT_shape[i+1];
	}

	for(npy_intp l=0;l<n_row;++l){
		for(npy_intp i=0;i<n_col;++i){
			npy_intp j=0;
			for(int k=0;k<nd;++k){
				j += ((i / A_strides[k]) % AT_shape[k]) * AT_strides[k];
			}
			AT[j] = *A++;
		}
		AT += n_col;
	}
}



#endif