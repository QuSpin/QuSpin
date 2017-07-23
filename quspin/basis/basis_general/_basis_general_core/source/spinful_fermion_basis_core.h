#ifndef _SPINFUL_FERMION_BASIS_CORE_OP_H
#define _SPINFUL_FERMION_BASIS_CORE_OP_H

#include <complex>
#include "general_basis_core.h"
#include "local_pcon_basis_core.h"
#include "spinless_fermion_basis_core.h"
#include "numpy/ndarraytypes.h"



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

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			int split = 0;
			int n_op_left,n_op_right;

			for(int i=0;i<n_op;i++){
				if(opstr[i]=='|'){break;}
				split++;
			}

			n_op_left = split;
			n_op_right = n_op - split;

			I r_left,r_right;

			local_pcon_basis_core<I>::split_state(r,r_left,r_right);

			if(n_op_left>0){
				
				I s_left = r_left;
				I one = 1;

				for(int j=n_op_left-1;j>-1;j--){
					int ind = general_basis_core<I>::N-indx[j]-1;
					int f_count = 0;
					I rr = r_left;
					for(int i=0;i<ind;i++){f_count^=(rr&1);rr>>=1;}
					m *= (f_count?-1:1);
					I b = (one << ind);
					bool a = bool((r_left >> ind)&one);
					char op = opstr[j];
					switch(op){
						case 'n':
							m *= (a?1:0);
							break;
						case '+':
							m *= (a?0:1);
							r_left ^= b;
							break;
						case '-':
							m *= (a?1:0);
							r_left ^= b;
							break;
						case 'I':
							break;
						default:
							return -1;
					}

					if(std::abs(m)==0){
						r_left = s_left;
						break;
					}
				}
			}

			if(n_op_right>0){
				I s_right = r_right;
				I one = 1;

				for(int j=n_op_right-1;j>-1;j--){
					int ind = general_basis_core<I>::N-indx[split+j-1]-1;
					int f_count = 0;
					I rr = r_right;
					for(int i=0;i<ind;i++){f_count^=(rr&1);rr>>=1;}
					m *= (f_count?-1:1);
					I b = (one << ind);
					bool a = bool((r >> ind)&one);
					char op = opstr[split+j];
					switch(op){
						case 'n':
							m *= (a?1:0);
							break;
						case '+':
							m *= (a?0:1);
							r_right ^= b;
							break;
						case '-':
							m *= (a?1:0);
							r_right ^= b;
							break;
						case 'I':
							break;
						default:
							return -1;
					}

					if(std::abs(m)==0){
						r_right = s_right;
						break;
					}
				}
			}

			r = local_pcon_basis_core<I>::comb_state(r_left,r_right);

			return 0;
		}

		
};







#endif
