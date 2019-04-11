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


template<class I, class J>
struct compare_elements : std::binary_function<std::pair<I,J>,std::pair<I,J>,bool>
{
	bool operator()(std::pair<I,J> a, std::pair<I,J> b){return a.first < b.first;}
};

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

	indices_vec.reserve(Ns);
	data_vec.reserve(Ns);

	indices_vec.resize(Ns);
	data_vec.resize(Ns);


	int err = 0;

	std::vector<std::pair<K,T>> elements;
	elements.reserve(n_ops);

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

				elements.push_back(std::make_pair(j,M));
			}
		}
		std::sort(elements.begin(), elements.end(), compare_elements<K,T>());

		size_t jj = 0;
		size_t col_end = elements.size();
		size_t nnz = 0;

		while(jj<col_end){
			K j = elements[jj].first;
			T x = elements[jj].second;
			while(jj<col_end && elements[jj].first == j){
				x += elements[jj].second;
				++jj;
			}
			elements[nnz].first = j;
			elements[nnz].second = x;
			++nnz;
		}

		indices_vec[i].reserve(nnz);
		data_vec[i].reserve(nnz);
		for(jj=0;jj<nnz;++jj){
			indices_vec[i].push_back(elements[jj].first);
			data_vec[i].push_back(elements[jj].second);
		}
		std::erase(elements.begin(),elements.end());
	}
	return err;
}


}


template <class I, class T>
void data_tocsr(const I n_row,
                const I n_col,
                const I nnz,
                std::vector<std::vector<I>> &indices_vec,
                std::vector<std::vector<T>> &data_vec,
                      I Bp[],
                      I Bj[],
                      T Bx[])
{
	typedef ind_iter std::vector<I>::iterator;
	typedef dat_iter std::vector<T>::iterator;

    //compute number of non-zero entries per column of A
    std::fill(Bp, Bp + n_row, 0);

    for(I col = 0; col < n_col; col++){
    	ind_iter ind_it=indices_vec[col].begin();
    	while(ind_it!=indices_vec.end()){
    		Bp[*ind_it++]++;
    	}
    }

    //cumsum the nnz per column to get Bp[]
    for(I row = 0, cumsum = 0; row < n_row; row++){
        I temp  = Bp[row];
        Bp[row] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz;

    for(I col = 0; col < n_col; col++){
    	
    	ind_iter ind_it = indices_vec[col].begin();
    	dat_iter dat_it = data_vec[col].begin();

    	while(ind_it!=indices_vec[col].end()){
    		I row = *ind_it++;
    		I dest = Bp[row];

    		Bj[dest] = col;
    		Bx[dest] = *dat_it++;

    		Bp[row]++;
    	}
    }

    for(I row = 0, last = 0; row <= n_row; row++){
        I temp  = Bp[row];
        Bp[row] = last;
        last    = temp;
    }

#endif