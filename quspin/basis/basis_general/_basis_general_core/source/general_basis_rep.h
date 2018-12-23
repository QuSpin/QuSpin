#ifndef _GENERAL_BASIS_OP_H
#define _GENERAL_BASIS_OP_H

#include <complex>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"



template<class I>
void general_repesentative(general_basis_core<I> *B,
						  const I s[],
						  const I r[]
						  )
{
	int sign = 1;
	int g[128],gg[128];

	r[0] = B->ref_state(s[0],g,gg,sign);

}



#endif
