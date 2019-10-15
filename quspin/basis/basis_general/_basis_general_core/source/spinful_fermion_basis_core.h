#ifndef _SPINFUL_FERMION_BASIS_CORE_OP_H
#define _SPINFUL_FERMION_BASIS_CORE_OP_H

#include <complex>
#include <iostream>
#include "general_basis_core.h"
#include "spinless_fermion_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "openmp.h"

namespace basis_general {

template<class I,class P=signed char>
class spinful_fermion_basis_core : public spinless_fermion_basis_core<I,P>
{
	const int N_sys;
	const bool not_dble_occ;

	public:
		spinful_fermion_basis_core(const int _N,const bool _dble_occ) : \
		spinless_fermion_basis_core<I,P>::spinless_fermion_basis_core(2*_N), 
		N_sys(_N), not_dble_occ(!_dble_occ) {}

		spinful_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[],const bool _dble_occ) : \
		spinless_fermion_basis_core<I,P>::spinless_fermion_basis_core(2*_N,_nt,_maps,_pers,_qs), 
		N_sys(_N), not_dble_occ(!_dble_occ)  {}

		~spinful_fermion_basis_core() {}


		std::vector<int> count_particles(const I s){
			I s_left,s_right;
			std::vector<int> v(2);
			split_state(s,s_left,s_right);
			v[0] = bit_count(s_left,N_sys);
			v[1] = bit_count(s_right,N_sys);
			return v;
		}


		void inline split_state(const I s,I &s_left,I &s_right){
			s_right = ((~(I)0) >> (bit_info<I>::bits-N_sys))&s;
			s_left = (s >> N_sys);
		}

		void inline get_right_min_max(const I s,I &min,I &max){
			int n = bit_count(s,N_sys);
			if(n){
				min = (~(I)0) >> (bit_info<I>::bits-n);
				max = min << (N_sys - n);
			}
			else{
				min = max = 0;
			}
		}

		I inline comb_state(const I s_left,const I s_right){
			return (s_left<<N_sys)+s_right;
		}

		I inline next_state_pcon_side(const I s){
			if(s==0){return s;}
			I t = (s | (s - 1)) + 1;
			return t | ((((t & (0-t)) / (s & (0-s))) >> 1) - 1);
		}

		I next_state_pcon(const I s,const I nns){

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

		double check_state(I s){
			I s_left  = 0,s_right = 0;
			split_state(s,s_left,s_right);
			if(not_dble_occ && (s_left&s_right)){
				return std::numeric_limits<double>::quiet_NaN();
			}
			else{
				return check_state_core_unrolled<I,P>(this,s,general_basis_core<I,P>::nt);
			}
		}

};





}






#endif