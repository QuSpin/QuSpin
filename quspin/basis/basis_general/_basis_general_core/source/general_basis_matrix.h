#ifndef _GENERAL_BASIS_MATRIX
#define _GENERAL_BASIS_MATRIX

#include <iostream>
#include <complex>
#include <algorithm>
#include <map>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"
#include <bits/stdc++.h>

namespace basis_general {



template<class T>
int inline check_imag(std::complex<double> m,std::complex<T> *M){
	M[0].real(m.real());
	M[0].imag(m.imag());
	return 0;
}

template<class T>
int inline check_imag(std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		M[0] = m.real();
		return 0;
	}
}



template<class I, class J, class K, class T>
int general_make_matrix(general_basis_core<I> *B,
						  const int n_ops,
						  const std::vector<std::string> opstrs,
						  const std::vector<std::vector<int>> indxs,
						  const std::vector<std::complex<double>> Js,
						  const bool full_basis,
						  const npy_intp Ns,
						  const I basis[],
						  const J n[],
								std::vector<I> indptr,
								std::vector<I> indices,
								std::vector<T> data
						  )
{
	indptr.resize(Ns+1);
	indices.resize(0);
	data.resize(0);

	std::fill(indptr.begin(),indptr.end(),(I)0);
	npy_intp nnz = 0;
	npy_intp row_size_avg = 0;
	int err = 0;
	std::map<I,T> row_data;

	for(npy_intp i=0;i<Ns && err==0;j++){
		row_data.clear();	

		for(int iop=0;iop<n_ops && err==0;j++){
			std::string opstr = opstrs[iop];
			std::vector<int> indx = indxs[iop];
			const int n_op = indx.size();
			std::reverse(indx.begin(),indx.end());
			std::reverse(opstr.begin(),opstr.end());

			I r = basis[i];
			K j = i;

			std::complex<double> m = Js[iop];
			err = B->op(r,m,n_op,opstr.c_str(),&indx[0]);



			int sign = 1;

			for(int k=0;k<nt;k++){
				g[k]=0;
			}

			
			if(r != basis[j]){
				I rr = B->ref_state(r,g,sign);
				if(full_basis){
					j = Ns - (npy_intp)rr - 1;
				}
				else{
					j = binary_search(Ns,basis,rr);
				}
				
			}

			if(j >= 0){
				T M = 0;

				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,-q));
				}

				m *= sign * std::sqrt(double(n[i])/double(n[j]));

				err = check_imag(m,&M);

				if(row_data.count(j)){ // if column element exists add to matrix element
					row_data[j] += M;
				}
				else{ // else create new column entry
					row_data[j] = M;
				}
			}
		}

		// add matrix elements to csr data.
		const int row_size = row_data.size();
		const npy_intp capacity = indices.capacity();

		// running average of row_size
		if(i>0){
			row_size_avg = row_size_avg + (npy_intp)((row_size_avg - row_size)/(i+1));
		}else{
			row_size_avg = row_size;
		}
		

		nnz += row_size;
		indptr[i+1] = nnz;

		// use running average of row_size to allocate more memory. 
		if(nnz > capacity){
			npy_intp new_capacity = nnz + std::max((Ns*row_size_avg/100),(npy_intp)1);
			indices.reserve(new_capacity);
			data.reserve(new_capacity);
		}

		// add column indices and data to arrays
		std::map<I,T>::iterator it = row_data.begin();

		for(I j=indptr[i];j<indptr[i+1];++j){
			indices.push_back(it->first);
			data[j].push_back(it->second);
			++it;
		}
	}

	return err;
}


}



#endif