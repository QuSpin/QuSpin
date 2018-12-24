#ifndef _GENERAL_BASIS_REP_H
#define _GENERAL_BASIS_REP_H

#include <complex>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"



template<class I>
void general_representative(general_basis_core<I> *B,
						  const I s[],
						  		I r[],
						  const npy_intp Ns
						  )
{	int g[128],gg[128];
	int sign;

	for(npy_intp i=0;i<Ns;i++){

		sign = 1;
		r[i] = B->ref_state(s[i],g,gg,sign);
	}

}



#endif
