#ifndef _HCB_BASIS_CORE_H
#define _HCB_BASIS_CORE_H

#include <complex>
#include <vector>
#include <iostream>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "benes_perm.h"
#include "openmp.h"

namespace basis_general {

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


template<class I,class P=signed char>
class hcb_basis_core : public general_basis_core<I,P>
{

	public:
		std::vector<tr_benes<I>> benes_maps;
		std::vector<I> invs;

		hcb_basis_core(const int _N, const bool _fermionic=false,const bool _pre_check=false) : \
		general_basis_core<I>::general_basis_core(_N,_fermionic,_pre_check) {}

		hcb_basis_core(const int _N,const int _nt,const int _maps[], \
					const int _pers[], const int _qs[], const bool _fermionic=false,const bool _pre_check=false) : \
		general_basis_core<I>::general_basis_core(_N,_nt,_maps,_pers,_qs,_fermionic,_pre_check) {

			benes_maps.resize(_nt);
			invs.resize(_nt);
			ta_index<I> index;
			for(int j=0;j<bit_info<I>::bits;j++){index.data[j] = no_index;}

			for(int i=0;i<_nt;i++){
				const int * map = &general_basis_core<I,P>::maps[i*_N];
				I inv = 0;

				for(int j=0;j<_N;j++){
					int m = map[j];
					int bit_j = _N - j - 1;


					if(m<0){
						int bit_m = _N + m;
						index.data[bit_j] = bit_m;
						inv ^= ((I)1 << bit_j);
					}
					else{
						int bit_m = _N - m -1;
						index.data[bit_j] = bit_m;
					}
				}

				gen_benes<I>(&benes_maps[i],index);
				invs[i] = inv;
			}
		}

		~hcb_basis_core() {}

		npy_intp get_prefix(const I s,const int N_p){
			return integer_cast<npy_intp,I>(s >> (general_basis_core<I,P>::N - N_p));
		}

		I map_state(I s,int n_map,P &sign){
			if(general_basis_core<I,P>::nt<=0){
				return s;
			}
			return benes_bwd(&benes_maps[n_map],s^invs[n_map]);	
		}

		void map_state(I s[],npy_intp M,int n_map,P sign[]){
			if(general_basis_core<I,P>::nt<=0){
				return;
			}
			const tr_benes<I> * benes_map = &benes_maps[n_map];
			const I inv = invs[n_map];
			#pragma omp for schedule(static)
			for(npy_intp i=0;i<M;i++){
				s[i] = benes_bwd(benes_map,s[i]^inv);	
			}
		}

		std::vector<int> count_particles(const I s){
			std::vector<int> v(1);
			v[0] = bit_count(s,general_basis_core<I,P>::N);
			return v;
		}

		// I map_state(I s,int n_map,int &sign){
		// 	if(general_basis_core<I,P>::nt<=0){
		// 		return s;
		// 	}
		// 	const int n = general_basis_core<I,P>::N;
		// 	return hcb_map_bits(s,&general_basis_core<I,P>::maps[n_map*n],n);
			
		// }

		// void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
		// 	if(general_basis_core<I,P>::nt<=0){
		// 		return;
		// 	}
		// 	const int n = general_basis_core<I,P>::N;
		// 	const int * map = &general_basis_core<I,P>::maps[n_map*n];
		// 	#pragma omp for schedule(static,1)
		// 	for(npy_intp i=0;i<M;i++){
		// 		s[i] = hcb_map_bits(s[i],map,n);
		// 	}
		// }

		I inline next_state_pcon(const I s,const I nns){
			if(s==0){return s;}
			I t = (s | (s - 1)) + 1;
			return t | ((((t & (0-t)) / (s & (0-s))) >> 1) - 1);
		}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			const I s = r;
			const I one = 1;
			const int NN = general_basis_core<I,P>::N;
			for(int j=n_op-1;j>-1;j--){

				const int ind = NN-indx[j]-1;
				const I b = (one << ind);
				const bool a = (bool)((r >> ind)&one);
				const char op = opstr[j];
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

				if(m.real()==0 && m.imag()==0){
					r = s;
					break;
				}
			}

			return 0;
		}
};


}




#endif
