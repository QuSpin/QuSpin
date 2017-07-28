#ifndef _SPINLESS_FERMION_BASIS_OP_H
#define _SPINLESS_FERMION_BASIS_OP_H

#include <complex>
#include "hcb_basis_core.h"
#include "numpy/ndarraytypes.h"

template<class I>
I inline spinless_fermion_map_bits(I s,const int map[],const int N){
	I ss = 0;
	for(int i=0;i<N;i++){
		int j = map[i];
		ss ^= ( j<0 ? ((s&1)^1)<<(-(j+1)) : (s&1)<<j );
		s >>= 1;
	}
	return ss;
}


template<class I>
class spinless_fermion_basis_core : public hcb_basis_core<I>
{
	public:
		spinless_fermion_basis_core(const int _N) : \
		hcb_basis_core<I>::hcb_basis_core(_N) {}

		spinless_fermion_basis_core(const int _N,const int _nt,const int _maps[], \
						   const int _pers[], const int _qs[]) : \
		hcb_basis_core<I>::hcb_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~spinless_fermion_basis_core() {}

		int op(I &r,std::complex<double> &m,const int n_op,const char opstr[],const int indx[]){
			I s = r;
			I one = 1;

			for(int j=n_op-1;j>-1;j--){
				int ind = general_basis_core<I>::N-indx[j]-1;
				int f_count = 0;
				I rr = r;
				for(int i=0;i<ind;i++){f_count^=(rr&1);rr>>=1;}
				m *= (f_count?-1:1);
				I b = (one << ind);
				bool a = bool((r >> ind)&one);
				char op = opstr[j];
				switch(op){
					case 'z':
						m *= (a?0.5:-0.5);
						break;
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
