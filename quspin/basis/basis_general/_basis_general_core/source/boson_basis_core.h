#ifndef _BOSON_BASIS_CORE_H
#define _BOSON_BASIS_CORE_H

#include <complex>
#include <cmath>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "openmp.h"

namespace basis_general {


template<class I>
I inline boson_map_bits(I s,const int map[],const I M[],const int sps,const int N){
	I ss = 0;
	for(int i=N-1;i>=0;--i){
		int j = map[i];
		ss += ( j<0 ? (sps-(int)(s%sps)-1)*M[j+N] : (int)(s%sps)*M[N-j-1] );
		s /= sps;
	}
	return ss;
}


template<class I,class P=signed char>
class boson_basis_core : public general_basis_core<I,P>
{
	public:
		std::vector<I> M;
		const int sps;

		boson_basis_core(const int _N, const int _sps) : \
		general_basis_core<I,P>::general_basis_core(_N), sps(_sps) {
			M.resize(_N);
			M[0] = (I)1;
			for(int i=1;i<_N;i++){
				M[i] = (M[i-1] * (I)_sps);
			}
		}

		boson_basis_core(const int _N, const int _sps,const int _nt, \
						 const int _maps[], const int _pers[], const int _qs[]) : \
		general_basis_core<I,P>::general_basis_core(_N,_nt,_maps,_pers,_qs), sps(_sps) {
			M.resize(_N);
			M[0] = (I)1;
			for(int i=1;i<_N;i++){
				M[i] = (M[i-1] * (I)_sps);
			}			
		}

		~boson_basis_core() {}

		npy_intp get_prefix(const I s,const int N_p){
			return integer_cast<npy_intp,I>(s/M[general_basis_core<I,P>::N - N_p]);
		}


		I map_state(I s,int n_map,P &sign){
			if(general_basis_core<I,P>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I,P>::N;
			return boson_map_bits(s,&general_basis_core<I,P>::maps[n_map*n],&M[0],sps,n);
			
		}

		void map_state(I s[],npy_intp MM,int n_map,P sign[]){
			if(general_basis_core<I,P>::nt<=0){
				return;
			}
			const int n = general_basis_core<I,P>::N;
			const int * map = &general_basis_core<I,P>::maps[n_map*n];
			#pragma omp for schedule(static)
			for(npy_intp i=0;i<MM;i++){
				s[i] = boson_map_bits(s[i],map,&M[0],sps,n);
			}
		}

		std::vector<int> count_particles(const I r){
			std::vector<int> v(1);
			int n = 0;
			I s = r;
			for(int i=0;i<general_basis_core<I,P>::N;i++){
				n += (int)(s%sps);
				s /= sps;
			}
			v[0] = n;
			return v;
		}

		I inline next_state_pcon(const I r,const I nns){
			if(r == 0){
				return r;
			}
			I s = r;
			int n=0;
			for(int i=0;i<general_basis_core<I,P>::N-1;i++){
				int b1 = (int)((s/M[i])%sps);
				if(b1>0){
					n += b1;
					int b2 = (int)((s/M[i+1])%sps);
					if(b2<(sps-1)){
						n -= 1;
						s -= M[i];
						s += M[i+1];
						if(n>0){
							int l = n/(sps-1);
							int n_left = n%(sps-1);
							for(int j=0;j<(i+1);j++){
								s -= (int)((s/M[j])%sps) * M[j];
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
			const I s = r;
			double me_offdiag=1;
			double me_diag=1;
			double S = (sps-1.0)/2.0;
			for(int j=n_op-1;j>-1;j--){

				int ind = general_basis_core<I,P>::N-indx[j]-1;
				int occ = (int)((r/M[ind])%sps);
				I b = M[ind];
				char op = opstr[j];
				switch(op){
					case 'z':
						me_diag *= (occ-S);
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

				if(me_diag==0 || me_offdiag==0){
					r = s;
					break;
				}
			}

			me *= me_diag*std::sqrt(me_offdiag);

			return 0;
		}
};




}


#endif
