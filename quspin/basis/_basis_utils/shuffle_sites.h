#ifndef __SHUFFLE_SITES_
#define __SHUFFLE_SITES_


#include "numpy/ndarraytypes.h"

template<class T>
void shuffle_sites_core_base_2(const npy_int32 nd,const npy_int32 * T_tup,const npy_intp n_row,const npy_intp n_col,const T * A, T * AT){
	npy_intp i_new[64];
	npy_intp i_old[64];

	for(npy_int32 i=0;i<nd;i++){
		i_old[i] = nd - i;
		i_new[i] = nd - T_tup[i];
	}

	for(npy_intp l=0;l<n_row;++l){
		for(npy_intp i=0;i<n_col;++i){
			npy_uintp j=0;
			for(npy_int32 k=0;k<nd;++k){
				j ^= ((i >> i_new[k]) << i_old[k]);
			}

			AT[j] = *A++;
		}
		AT += n_col;
	}
}


template<class I,class J>
I ipow(I base, J exp)
{
    I result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

template<class T>
void shuffle_sites_core(const npy_int32 nd,const npy_intp sps,const npy_int32 * T_tup,const npy_intp n_row,const npy_intp n_col,const T * A, T * AT){
	npy_intp strides_cont[64];
	npy_intp strides_tran[64];


	for(npy_int32 i=0;i<nd;i++){
		strides_cont[i] = ipow(sps,nd - i - 1);
		strides_tran[i] = ipow(sps,nd - T_tup[i] - 1);
	}


	for(npy_intp l=0;l<n_row;++l){
		for(npy_intp i=0;i<n_col;++i){
			npy_intp j=0;
			for(npy_int32 k=0;k<nd;++k){
				j += (((i / strides_tran[k]) % sps) * strides_cont[k]);
			}
			AT[j] = *A++;
			
		}
		AT += n_col;
	}
}

using namespace std;

#include <iostream>
#include <iomanip>

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