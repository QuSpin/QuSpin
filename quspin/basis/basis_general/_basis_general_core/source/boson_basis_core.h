#ifndef _BOSON_BASIS_CORE_H
#define _BOSON_BASIS_CORE_H

#include <complex>
#include <cmath>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

// template<class I>
// I inline boson_map_bits(I s,const int map[],const I inv,const I M[],const int sps,const int N){
// 	I ss = 0;
// 	for(int i=N-1;i>=0;--i){
// 		int j = N-map[i]-1;
// 		ss += ( inv&1 ? (sps-(s%sps)-1)*M[j] : (s%sps)*M[j] );
// 		s /= sps;
// 		inv >>= 1;
// 	}
// 	return ss;
// }

template<class I>
I inline boson_map_bits(I s,const int map[],const I M[],const int sps,const int N){
	I ss = 0;
	for(int i=N-1;i>=0;--i){
		int j = map[i];
		ss += ( j<0 ? (sps-(s%sps)-1)*M[j+N] : (s%sps)*M[N-j-1] );
		s /= sps;
	}
	return ss;
}


template<class I>
class boson_basis_core : public general_basis_core<I>
{
	public:
		I M[64];
		const I sps;

		boson_basis_core(const int _N, const int _sps) : \
		general_basis_core<I>::general_basis_core(_N), sps(_sps) {
			M[0] = 1;
			for(int i=1;i<_N;i++){
				M[i] = M[i-1]*_sps;
			}
		}

		boson_basis_core(const int _N, const int _sps,const int _nt, \
						 const int _maps[], const int _pers[], const int _qs[]) : \
		general_basis_core<I>::general_basis_core(_N,_nt,_maps,_pers,_qs), sps(_sps) {
			M[0] = 1;
			for(int i=1;i<_N;i++){
				M[i] = M[i-1]*_sps;
			}			
		}

		~boson_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return boson_map_bits(s,&general_basis_core<I>::maps[n_map*n],M,sps,n);
			
		}

		void map_state(I s[],npy_intp P,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N;
			const int * map = &general_basis_core<I>::maps[n_map*n];
			#pragma omp for schedule(static,1)
			for(npy_intp i=0;i<P;i++){
				s[i] = boson_map_bits(s[i],map,M,sps,n);
				sign[i] *= 1;
			}
		}

		I inline next_state_pcon(I s){
			if(s == 0){
				return s;
			}
			int n=0;
			for(int i=0;i<general_basis_core<I>::N-1;i++){
				unsigned int b1 = (s/M[i])%sps;
				if(b1>0){
					n += b1;
					unsigned int b2 = (s/M[i+1])%sps;
					if(b2<(sps-1)){
						n -= 1;
						s -= M[i];
						s += M[i+1];
						if(n>0){
							int l = n/(sps-1);
							int n_left = n%(sps-1);
							for(int j=0;j<(i+1);j++){
								s -= ((s/M[j])%sps)*M[j];
								if(j<l){
									s += (sps-1)*M[j];
								}
								else if(j == l){
									s += n_left*M[j];
								}
							}
						}
						break;
					}
				}
			}
			return s;
		}

		int op(I &r,std::complex<double> &me,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			double me_offdiag=1;
			double me_diag=1;
			for(int j=n_op-1;j>-1;j--){
				int ind = general_basis_core<I>::N-indx[j]-1;
				I occ = (r/M[ind])%sps;
				I b = M[ind];
				char op = opstr[j];
				switch(op){
					case 'n':
						me_diag *= occ;
						break;
					case '+':
						me_offdiag *= (occ+1)%sps;
						r += ((occ+1)<sps?b:0);
						break;
					case '-':
						me_offdiag *= occ;
						r -= (occ>0?b:0);
						break;
					case 'I':
						break;
					default:
						return -1;
				}

				if(std::abs(me_diag*me_offdiag)==0){
					r = s;
					break;
				}
			}

			me *= me_diag*std::sqrt(me_offdiag);

			return 0;
		}
};







#endif
