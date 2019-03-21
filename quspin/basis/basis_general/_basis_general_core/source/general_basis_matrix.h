#ifndef _GENERAL_BASIS_MATRIX
#define _GENERAL_BASIS_MATRIX

#include <iostream>
#include <complex>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include "general_basis_core.h"
#include "general_basis_op.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"
#include <bits/stdc++.h>

namespace basis_general {


template<class I, class J, class K, class T>
int general_make_matrix(general_basis_core<I> *B,
						  const int n_ops,
						  const std::vector<std::string> &opstrs,
						  const std::vector<std::vector<int>> &indxs,
						  const std::vector<T> &Js,
						  const bool full_basis,
						  const npy_intp Ns,
						  const I basis[],
						  const J n[],
						  		std::vector<std::vector<K>> &indices_vec,
							    std::vector<std::vector<T>> &data_vec
						  )
{
	const int nt = B->get_nt();
	int g[__GENERAL_BASIS_CORE__max_nt];

	indices_vec.resize(Ns);
	data_vec.resize(Ns);
	int err = 0;
	for(npy_intp i=0;i<Ns && err==0;i++){
		indices_vec[i].reserve(n_ops);
		data_vec[i].reserve(n_ops);
	}

	for(npy_intp i=0;i<Ns && err==0;i++){

		for(int iop=0;iop<n_ops && err==0;iop++){
			std::string opstr = opstrs[iop];
			std::vector<int> indx = indxs[iop];
			const int n_op = indx.size();

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

				m *= sign * std::sqrt(double(n[j])/double(n[i]));

				err = check_imag(m,&M);


				typename std::vector<K>::iterator ind_pos = std::lower_bound(indices_vec[j].begin(),indices_vec[j].end(),i);
				typename std::vector<T>::iterator dat_pos = (npy_intp)(ind_pos - indices_vec[j].begin()) + data_vec[j].begin();

				if(!(ind_pos==indices_vec[j].end()) && !(i < *ind_pos)){
					*dat_pos += M;
				}
				else{
					indices_vec[j].insert(ind_pos,i);
					data_vec[j].insert(dat_pos,M);
				}
			}
		}
	}

	return err;
}


}

#endif