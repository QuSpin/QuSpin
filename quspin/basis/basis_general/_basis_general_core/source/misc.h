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


// template<class I>
// void inline map_state_wrapper(void *b,void *basis,npy_intp Ns,int n,npy_int8 *sign){
// 	reinterpret_cast<general_basis_core<I>*>(b)->map_state((I*)basis,Ns,n,sign);
// }

// template<class I,class K>
// void inline row_index_pcon_helper(const void *basis,const void *basis_pcon,K* row,npy_intp Ns,npy_intp Ns_full){
// 	I * basis_ptr = (const I *) basis;
// 	I * basis_pcon_ptr = (const I *) basis_pcon;
// 	for(npy_intp i=0;i<Ns;i++){
// 		row[i] = binary_search(Ns_full,basis_pcon_ptr,basis[i]);
// 	}
// }


#endif