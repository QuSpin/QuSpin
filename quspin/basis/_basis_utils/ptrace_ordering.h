
#include "misc.h"

template<class I,class J>
int inline partition_swaps(I s,const J map[],const int N){
	// counts the number of swaps required to form the proper paritions
	int np = 0;
	int pos_list[64];

	for(int i=N-1;i>=0;--i){
		if(s&1){
			pos_list[np++] = map[i]; 
		}
		s >>= 1;
	}

	return basis_general::countSwaps<npy_uint64>(pos_list,np);

}

