#include "spinless_fermion_basis_core.h"


template<class I,class P>
void inline fermion_ptrace_sign_core(I s,const int map[],const int N,P &sign){
	int np = 0;
	int pos_list[basis_general::bit_info<I>::bits];
	int work[basis_general::bit_info<I>::bits];
	bool f_count = 0;

	for(int i=N-1;i>=0;--i){
		if(s&1){
			pos_list[np++] = map[i]; 
		}
		s >>= 1;
	}

	basis_general::getf_count(pos_list,work,0,np-1,f_count);
	if(f_count){sign *= -1;}

}