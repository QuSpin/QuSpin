#ifndef _TWO_HCB_BASIS_CORE_H
#define _TWO_HCB_BASIS_CORE_H

#include <complex>
#include "hcb_basis_core.h"
#include "numpy/ndarraytypes.h"

namespace basis_general {

template<class I>
class two_hcb_basis_core : public hcb_basis_core<I>
{
	int N_sys;
	public:
		two_hcb_basis_core(const int _N) : \
		hcb_basis_core<I>::hcb_basis_core(2*_N), N_sys(_N) {}

		two_hcb_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[]) : \
		hcb_basis_core<I>::hcb_basis_core(2*_N,_nt,_maps,_pers,_qs), N_sys(_N)  {}

		~two_hcb_basis_core() {}

		void inline split_state(I s,I &s_left,I &s_right){
			s_right = (bit_info<I>::all_bits >> I(bit_info<I>::bits-N_sys))&s;
			s_left = (s >> N_sys);
		}

		void inline get_right_min_max(I s,I &min,I &max){
			int n = bit_count(s,N_sys);
			if(n){
				min = bit_info<I>::all_bits >> I(bit_info<I>::bits-n);
				max = min << (N_sys - n);
			}
			else{
				min = max = 0;
			}
		}

		I inline comb_state(I s_left,I s_right){
			return (s_left<<N_sys)+s_right;
		}

		I inline next_state_pcon_side(I s){
			if(s==0){return s;}
			I t = (s | (s - 1)) + 1;
			return t | ((((t & -t) / (s & -s)) >> 1) - 1);
		}

		I next_state_pcon(I s){

			I s_left  = 0;
			I s_right = 0;
			I min_right,max_right;

			split_state(s,s_left,s_right);
			get_right_min_max(s_right,min_right,max_right);

			if(s_right<max_right){
				s_right = next_state_pcon_side(s_right);
			}
			else{
				s_right = min_right;
				s_left = next_state_pcon_side(s_left);
			}

			return comb_state(s_left,s_right);
		}
};


}



#endif