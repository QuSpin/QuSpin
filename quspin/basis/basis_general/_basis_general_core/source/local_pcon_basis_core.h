#ifndef _LOCAL_PCON_BASIS_CORE_OP_H
#define _LOCAL_PCON_BASIS_CORE_OP_H

#include <complex>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

namespace basis_general {

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

		I next_state_pcon(I) = 0;
		int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;		
};
}

#endif