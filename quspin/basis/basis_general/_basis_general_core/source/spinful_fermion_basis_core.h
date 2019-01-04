#ifndef _SPINFUL_FERMION_BASIS_CORE_OP_H
#define _SPINFUL_FERMION_BASIS_CORE_OP_H

#include <complex>
#include <iostream>
#include "general_basis_core.h"
#include "spinless_fermion_basis_core.h"
#include "numpy/ndarraytypes.h"


// template<class I>
// class spinful_fermion_basis_core : public local_pcon_basis_core<I>
// {
// 	public:
// 		spinful_fermion_basis_core(const int _N) : \
// 		local_pcon_basis_core<I>::local_pcon_basis_core(_N) {}

// 		spinful_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
// 								   const int _pers[], const int _qs[]) : \
// 		local_pcon_basis_core<I>::local_pcon_basis_core(_N,_nt,_maps,_pers,_qs) {}

// 		~spinful_fermion_basis_core() {}

// 		I map_state(I s,int n_map,int &sign){
// 			if(general_basis_core<I>::nt<=0){
// 				return s;
// 			}
// 			const int n = general_basis_core<I>::N << 1;
// 			return spinless_fermion_map_bits(s,&general_basis_core<I>::maps[n_map*n],n,sign);
			
// 		}

// 		void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
// 			if(general_basis_core<I>::nt<=0){
// 				return;
// 			}
// 			const int n = general_basis_core<I>::N << 1;
// 			const int * map = &general_basis_core<I>::maps[n_map*n];
// 			#pragma omp for schedule(static,1)
// 			for(npy_intp i=0;i<M;i++){
// 				int temp_sign = sign[i];
// 				s[i] = spinless_fermion_map_bits(s[i],map,n,temp_sign);
// 				sign[i] = temp_sign;
// 			}
// 		}

// 		void split_state(I s,I &s_left,I &s_right){
// 			s_right = (I(bit_info<I>::all_bits) >> (bit_info<I>::bits-general_basis_core<I>::N))&s;
// 			s_left = (s >> general_basis_core<I>::N);
// 		}

// 		void get_right_min_max(I s,I &min,I &max){
// 			int n = bit_count(s,general_basis_core<I>::N);
// 			if(n){
// 				min = I(bit_info<I>::all_bits) >> (bit_info<I>::bits-n);
// 				max = min << (general_basis_core<I>::N - n);
// 			}
// 			else{
// 				min = max = 0;
// 			}
// 		}

// 		I comb_state(I s_left,I s_right){
// 			return (s_left<<general_basis_core<I>::N)+s_right;
// 		}

// 		I next_state_pcon_side(I s){
// 			if(s==0){return s;}
// 			I t = (s | (s - 1)) + 1;
// 			return t | ((((t & -t) / (s & -s)) >> 1) - 1);
// 		}

// 		I next_state_pcon(I s){

// 			I s_left  = 0;
// 			I s_right = 0;
// 			I min_right,max_right;

// 			split_state(s,s_left,s_right);
// 			get_right_min_max(s_right,min_right,max_right);

// 			if(s_right<max_right){
// 				s_right = next_state_pcon_side(s_right);
// 			}
// 			else{
// 				s_right = min_right;
// 				s_left = next_state_pcon_side(s_left);
// 			}

// 			return comb_state(s_left,s_right);
// 		}

// 		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
// 			I s = r;
// 			I one = 1;

// 			for(int j=n_op-1;j>-1;j--){
// 				int ind = (general_basis_core<I>::N<<1)-indx[j]-1;
// 				I f_count = bit_count(r,ind);
// 				double sign = ((f_count&1)?-1:1);
// 				I b = (one << ind);
// 				bool a = bool((r >> ind)&one);
// 				char op = opstr[j];
// 				switch(op){
// 					case 'z':
// 						m *= (a?0.5:-0.5);
// 						break;
// 					case 'n':
// 						m *= (a?1:0);
// 						break;
// 					case '+':
// 						m *= (a?0:sign);
// 						r ^= b;
// 						break;
// 					case '-':
// 						m *= (a?sign:0);
// 						r ^= b;
// 						break;
// 					case 'I':
// 						break;
// 					default:
// 						return -1;
// 				}

// 				if(std::abs(m)==0){
// 					r = s;
// 					break;
// 				}
// 			}

// 			return 0;
// 		}

		
// };



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








#endif