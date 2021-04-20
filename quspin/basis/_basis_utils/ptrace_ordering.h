
#include "misc.h"

template<class I>
int inline partition_swaps(I s,const int map[],const int N){
	// counts the number of swaps required to form the proper paritions
	int np = 0;
	int pos_list[N];

	for(int i=N-1;i>=0;--i){
		if(s&1){
			pos_list[np++] = map[i]; 
		}
		s >>= 1;
	}

	return basis_general::countSwaps(pos_list,np);

}

