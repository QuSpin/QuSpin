#ifndef _SPINFUL_FERMION_BASIS_CORE_OP_H
#define _SPINFUL_FERMION_BASIS_CORE_OP_H

#include <complex>
#include <iostream>
#include "general_basis_core.h"
#include "spinless_fermion_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "openmp.h"



template<class I>
class spinful_fermion_basis_core : public spinless_fermion_basis_core<I>
{
	int N_sys;
	public:
		spinful_fermion_basis_core(const int _N) : \
		spinless_fermion_basis_core<I>::spinless_fermion_basis_core(2*_N), N_sys(_N) {}

		spinful_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[]) : \
		spinless_fermion_basis_core<I>::spinless_fermion_basis_core(2*_N,_nt,_maps,_pers,_qs), N_sys(_N)  {}

		~spinful_fermion_basis_core() {}

		I map_state(I s,int n_map,int &sign){
			if(general_basis_core<I>::nt<=0){
				return s;
			}
			const int n = general_basis_core<I>::N << 1;
			return spinless_fermion_map_bits(s,&general_basis_core<I>::maps[n_map*n],n,sign);
			
		}

		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
			if(general_basis_core<I>::nt<=0){
				return;
			}
			const int n = general_basis_core<I>::N << 1;
			const int * map = &general_basis_core<I>::maps[n_map*n];
			const npy_intp chunk = M/omp_get_num_threads();
			#pragma omp for schedule(static,chunk)
			for(npy_intp i=0;i<M;i++){
				int temp_sign = sign[i];
				s[i] = spinless_fermion_map_bits(s[i],map,n,temp_sign);
				sign[i] = temp_sign;
			}
		}

		std::vector<int> count_particles(I s){
			I s_left,s_right;
			split_state(s,s_left,s_right);
			int n_left  = bit_count(s_left,general_basis_core<I>::N);
			int n_right = bit_count(s_right,general_basis_core<I>::N);
			std::vector<int> v = {n_left,n_right};
			return v;
		}


		void inline split_state(I s,I &s_left,I &s_right){
			s_right = ((~(I)0) >> I(bit_info<I>::bits-N_sys))&s;
			s_left = (s >> N_sys);
		}

		void inline get_right_min_max(I s,I &min,I &max){
			int n = bit_count(s,N_sys);
			if(n){
				min = (~(I)0) >> I(bit_info<I>::bits-n);
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








#endif