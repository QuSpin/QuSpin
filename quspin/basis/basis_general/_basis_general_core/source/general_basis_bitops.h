#ifndef _GENERAL_BASIS_BITOPS_H
#define _GENERAL_BASIS_BITOPS_H

#include <iostream>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"

namespace basis_general {

template<class I>
struct bitwise_and_op : std::binary_function<I,I,I>
{
	I inline operator()(I a, I b){return a & b;}
};

template<class I>
struct bitwise_or_op : std::binary_function<I,I,I>
{
	I inline operator()(I a, I b){return a | b;}
};

template<class I>
struct bitwise_xor_op : std::binary_function<I,I,I>
{
	I inline operator()(I a, I b){return a ^ b;}
};

template<class I>
struct bitwise_not_op : std::unary_function<I,I>
{
	I inline operator()(I a){return ~ a;}
};

template<class I>
struct bitwise_left_shift_op : std::binary_function<I,I,I>
{
	I inline operator()(I a, I b){return a << b;}
};

template<class I>
struct bitwise_right_shift_op : std::binary_function<I,I,I>
{
	I inline operator()(I a, I b){return a >> b;}
};






template<class I, class binary_operator>
void bitwise_op(const I x1[],
						 const I x2[],
							   bool *where,
							   I *out,
						 const npy_intp Ns,
						 binary_operator op
				)
{	
	if(where){
		#pragma omp parallel
		{
			const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1); // bitops should have dynamic workload b/c of where-if statement

			#pragma omp parallel for schedule(dynamic,chunk)
			for(npy_intp i=0;i<Ns;i++){

				if(where[i]){
					out[i]=op(x1[i],x2[i]);  
				}
			}
		}
	}
	else{
		#pragma omp parallel
		{
			const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1); // bitops should have constant workload 

			#pragma omp parallel for schedule(static,chunk)
			for(npy_intp i=0;i<Ns;i++){

				out[i]=op(x1[i],x2[i]);  
			}
		}
	}
	

}
	


}
#endif