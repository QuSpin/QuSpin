#ifndef _LOCAL_PCON_BASIS_CORE_OP_H
#define _LOCAL_PCON_BASIS_CORE_OP_H

#include <complex>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

template<class I>
I inline local_pcon_map_bits(I s,const int map[],const int N){
	I ss = 0;
	for(int i=0;i<2*N;i++){
		int j = map[i];
		ss ^= ( j<0 ? ((s&1)^1)<<(-(j+1)) : (s&1)<<j );
		s >>= 1;
	}
	return ss;
}

template<class I>
class local_pcon_basis_core : public general_basis_core<I>
{
	public:
		local_pcon_basis_core(const int _N) : \
		general_basis_core<I>::general_basis_core(_N) {}

		local_pcon_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[]) : \
		general_basis_core<I>::general_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~local_pcon_basis_core() {}

		I inline map_state(I s,int n_map){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N;
			return local_pcon_map_bits(s,&general_basis_core<I>::maps[2*n_map*n],n);
			
		}

		void map_state(I s[],npy_intp M,int n_map){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N;
			const int * map = &general_basis_core<I>::maps[2*n_map*n];
			#pragma omp for schedule(static,1)
			for(npy_intp i=0;i<M;i++){
				s[i] = local_pcon_map_bits(s[i],map,n);
			}
		}

		void print(I s){
			I s_left,s_right;
			split_state(s,s_left,s_right);

			std::cout << "|";
			for(int i=0;i<general_basis_core<I>::N;i++){
				std::cout << (s_left&1) << " ";
				s_left>>=1;
			}
			std::cout << ">";

			std::cout << "|";
			for(int i=0;i<general_basis_core<I>::N;i++){
				std::cout << (s_right&1) << " ";
				s_right>>=1;
			}
			std::cout << ">";
		}

		void split_state(I s,I &s_left,I &s_right){
			s_right = ((I(1) << general_basis_core<I>::N) - 1)&s;
			s_left = (s >> general_basis_core<I>::N);
		}

		I comb_state(I s_left,I s_right){
			return (s_left<<general_basis_core<I>::N)+s_right;
		}		

		bool check_state(I s){
			return check_state_core<I>(this,s,s,general_basis_core<I>::nt,0);
		}

		I ref_state(I s,int g[],int gg[]){
			for(int i=0;i<general_basis_core<I>::nt;i++){
				g[i] = 0;
				gg[i] = 0;
			}
			return ref_state_core<I>(this,s,s,g,gg,general_basis_core<I>::nt,0);
		}

		double get_norm(I s){
			return get_norm_core<I>(this,s,s,general_basis_core<I>::nt,0);
		}

		I next_state_pcon(I s){

			I s_left  = 0;
			I s_right = 0;

			const I one = 1;
			int n_right=0;

			split_state(s,s_left,s_right);
			for(int i=0;i<general_basis_core<I>::N;i++){
				n_right += (s&1);
				s >>= 1;
			}

			I max_right = 0;

			for(int i=0;i<n_right;i++){max_right ^= one << (general_basis_core<I>::N-i-1);}
			if(s_right<max_right){
				I t = (s_right | (s_right - 1)) + 1;
				s_right = t | ((((t & -t) / (s_right & -s_right)) >> 1) - 1);
			}
			else{
				s_right = 0;
				for(int i=0;i<n_right;i++){s_right ^= one<<i;}
				if(s_left>0){
					I t = (s_left | (s_left - 1)) + 1;
					s_left = t | ((((t & -t) / (s_left & -s_left)) >> 1) - 1);
				}
			}

			return comb_state(s_left,s_right);
		}

		int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;

		
};


#endif
