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
			I s = r;
			I one = 1;

			for(int j=n_op-1;j>-1;j--){
				int ind = 2*general_basis_core<I>::N-indx[j]-1;
				int f_count = 0;
				I rr = r;
				for(int i=0;i<ind;i++){f_count^=(rr&1);rr>>=1;}
				m *= (f_count?-1:1);
				I b = (one << ind);
				bool a = bool((r >> ind)&one);
				char op = opstr[j];
				switch(op){
					case 'n':
						m *= (a?1:0);
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
