#ifndef _HCB_BASIS_CORE_H
#define _HCB_BASIS_CORE_H

#include <complex>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

// template<class I>
// I inline hcb_map_bits(I s,const int map[],const I inv,const int N){
// 	I ss = 0;


// 	s ^= inv;
// 	for(int i=N-1;i>=0;--i){
// 		if(s&1){ss ^= ( I(1)<<(N-map[i]-1) );}
// 		s >>= 1;
// 	}
// 	return ss;
// }

template<class I>
I inline hcb_map_bits(I s,const int map[],const int N){
	I ss = 0;

	for(int i=N-1;i>=0;--i){
		int j = map[i];
		ss ^= (j<0 ? ((s&1)^1)<<(N+j) : (s&1)<<(N-j-1) );
		s >>= 1;
	}
	return ss;
}


template<class I>
class hcb_basis_core : public general_basis_core<I>
{
	public:
		hcb_basis_core(const int _N) : \
		general_basis_core<I>::general_basis_core(_N) {}

		hcb_basis_core(const int _N,const int _nt,const int _maps[], \
					const int _pers[], const int _qs[]) : \
		general_basis_core<I>::general_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~hcb_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return hcb_map_bits(s,&general_basis_core<I>::maps[n_map*n],n);
			
		}

		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N;
			const int * map = &general_basis_core<I>::maps[n_map*n];
			#pragma omp for schedule(static,1)
			for(npy_intp i=0;i<M;i++){
				s[i] = hcb_map_bits(s[i],map,n);
			}
		}

		I inline next_state_pcon(I s){
			if(s==0){return s;}
			I t = (s | (s - 1)) + 1;
			return t | ((((t & -t) / (s & -s)) >> 1) - 1);
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			I one = 1;
			for(int j=n_op-1;j>-1;j--){
				int ind = general_basis_core<I>::N-indx[j]-1;
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
					case 'x':
						r ^= b;
						m *= 0.5;
						break;
					case 'y':
						m *= (a?std::complex<double>(0,0.5):std::complex<double>(0,-0.5));
						r ^= b;
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
