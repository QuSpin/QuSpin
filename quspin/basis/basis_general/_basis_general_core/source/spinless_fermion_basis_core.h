#ifndef _SPINLESS_FERMION_BASIS_OP_H
#define _SPINLESS_FERMION_BASIS_OP_H

#include <complex>
// #include <stdint.h>
#include "hcb_basis_core.h"
#include "numpy/ndarraytypes.h"

npy_uint32 bit_count(npy_uint32 I, int l,int L){
	I &= (0xFFFFFFFF >> (L-l));
	I = I - ((I >> 1) & 0x55555555);
	I = (I & 0x33333333) + ((I >> 2) & 0x33333333);
	return (((I + (I >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;    
}

npy_uint64 bit_count(npy_uint64 I, int l,int L){
	I &= (0xFFFFFFFFFFFFFFFF >> (L-l));
	I = I - ((I >> 1) & 0x5555555555555555);
	I = (I & 0x3333333333333333) + ((I >> 2) & 0x3333333333333333);
	return (((I + (I >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;

}

template<class I>
I inline spinless_fermion_map_bits(I s,const int map[],const int N,int &sign){
	I ss = 0;
	int pos_list[64];
	int np = 0;
	bool f_count = 0;

	for(int i=N-1;i>=0;i--){
		int j = map[i];
		I n = (s&1);
		if(n){pos_list[np]=( j<0 ? N+j : N-j-1); ++np;}
		ss ^= ( j<0 ? n^1<<(N+j) : n<<(N-j-1) );

		f_count ^= (n && (i&1) && (j<0));

		s >>= 1;
	}

	// sort in decending order counting number of permutations. 
	// starting at 2nd element as first element is already sorted.
	// Loop Invariant - left part of the array is already sorted.
	if(np > 1){
		for (int i = 1; i < np; i++) {
			int moveMe = pos_list[i];
			int j = i;
			while (j > 0 && moveMe > pos_list[j - 1]) {
				//Move element
				pos_list[j] = pos_list[j - 1];
				--j;
				//increase the count as element swap is happend
				f_count ^= 1;
			}
			pos_list[j] = moveMe;
		}
	}

	sign *= (f_count ? -1 : 1);

	return ss;
}


template<class I>
class spinless_fermion_basis_core : public hcb_basis_core<I>
{
	public:
		spinless_fermion_basis_core(const int _N) : \
		hcb_basis_core<I>::hcb_basis_core(_N) {}

		spinless_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
						   const int _pers[], const int _qs[]) : \
		hcb_basis_core<I>::hcb_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~spinless_fermion_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return spinless_fermion_map_bits(s,&general_basis_core<I>::maps[n_map*n],n,sign);
			
		}

		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N;
			const int * map = &general_basis_core<I>::maps[n_map*n];
			#pragma omp for schedule(static,1)
			for(npy_intp i=0;i<M;i++){
				int temp_sign = sign[i];
				s[i] = spinless_fermion_map_bits(s[i],map,n,temp_sign);
				sign[i] = temp_sign;
			}
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			I one = 1;

			for(int j=n_op-1;j>-1;j--){
				int ind = general_basis_core<I>::N-indx[j]-1;
				I f_count = bit_count(r,ind,general_basis_core<I>::N);
				m *= std::complex<double>((f_count&1)?-1:1);
				I b = (one << ind);
				bool a = bool((r >> ind)&one);
				char op = opstr[j];
				switch(op){
					case 'z':
						m *= (a?0.5:-0.5);
						break;
					case 'n':
						m *= (a?1:0);
						break;
					case '+':
						m *= (a?0:1);
						r ^= b;
						break;
					case '-':
						m *= (a?1:0);
						r ^= b;
						break;
					case 'I':
						break;
					default:
						return -1;
				}

				if(std::abs(m)==0){
					r = s;
					break;
				}
			}

			return 0;
		}
};







#endif
