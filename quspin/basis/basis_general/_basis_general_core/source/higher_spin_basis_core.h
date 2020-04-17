#ifndef _HIGHER_SPIN_BASIS_CORE_H
#define _HIGHER_SPIN_BASIS_CORE_H

#include <complex>
#include <cmath>
#include "boson_basis_core.h"

namespace basis_general {

template<class I,class P=signed char>
class higher_spin_basis_core : public boson_basis_core<I,P>
{
	public:
		higher_spin_basis_core(const int _N, const int _sps) : \
		boson_basis_core<I,P>::boson_basis_core(_N,_sps) {}

		higher_spin_basis_core(const int _N, const int _sps,const int _nt,\
						 const int _maps[], const int _pers[], const int _qs[]) : \
		boson_basis_core<I,P>::boson_basis_core(_N,_sps,_nt,_maps,_pers,_qs) {}

		~higher_spin_basis_core() {}

		int op(I &r,std::complex<double> &me,const int n_op,const char opstr[],const int indx[]){
			const I s = r;
			double me_offdiag = 1;
			double me_diag = 1;
			double S = (boson_basis_core<I,P>::sps-1.0)/2.0;

			for(int j=n_op-1;j>-1;j--){
				I b = boson_basis_core<I,P>::M[general_basis_core<I,P>::N-indx[j]-1];
				int occ = (int)((r/b)%boson_basis_core<I,P>::sps);

				char op = opstr[j];
				switch(op){
					case 'z':
						me_diag *= (occ-S);
						break;
					case '+':
						me_offdiag *= ((occ+1)*(boson_basis_core<I,P>::sps-occ-1));
						r += ((occ+1) < boson_basis_core<I,P>::sps ? b : 0);
						break;
					case '-':
						me_offdiag *= (occ*(boson_basis_core<I,P>::sps-occ));
						r -= (occ > 0 ? b : 0);
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

			me *= (me_diag*std::sqrt(me_offdiag));

			return 0;
		}
};
}
#endif
