#ifndef _SPINFUL_FERMION_BASIS_CORE_OP_H
#define _SPINFUL_FERMION_BASIS_CORE_OP_H

#include <complex>
#include "general_basis_core.h"
#include "local_pcon_basis_core.h"
#include "spinless_fermion_basis_core.h"
#include "numpy/ndarraytypes.h"


template<class I>
I inline spinful_fermion_map_bits(I s,const int map[],const int N,int &sign){
	I ss = 0;
	int pos_list[2*N];
	int np = 0;
	bool f_count = 0;

	for(int i=0;i<2*N;i++){
		int j = map[i];
		int n = (s&1);
		if(n){pos_list[np]=( j<0 ? -(j+1) : j); ++np;}
		ss ^= ( j<0 ? (n^1)<<(-(j+1)) : n<<j );

		f_count ^= (n & ((i%N)&1));

		s >>= 1;
	}

	//starting at 2nd element as first element is already sorted.
	//Loop Invariant - left part of the array is already sorted.
	if(np > 1){
		for (int i = 1; i < np; i++) {
			int moveMe = pos_list[i];
			int j = i;
			while (j > 0 && moveMe < pos_list[j - 1]) {
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
class spinful_fermion_basis_core : public local_pcon_basis_core<I>
{
	public:
		spinful_fermion_basis_core(const int _N) : \
		local_pcon_basis_core<I>::local_pcon_basis_core(_N) {}

		spinful_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[]) : \
		local_pcon_basis_core<I>::local_pcon_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~spinful_fermion_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return spinful_fermion_map_bits(s,&general_basis_core<I>::maps[n_map*n],n,sign);
			
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
				s[i] = spinful_fermion_map_bits(s[i],map,n,temp_sign);
				sign[i] = temp_sign;
			}
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			I one = 1;

			for(int j=n_op-1;j>-1;j--){
				int ind = 2*general_basis_core<I>::N-indx[j]-1;
				int N_c = indx[j];
				int f_count = 0;
				I rr = r >> ind;
				for(int i=0;i<N_c;i++){f_count^=(rr&1);rr>>=1;}
				m *= (f_count?-1:1);
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
